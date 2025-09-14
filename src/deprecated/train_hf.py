from pathlib import Path
import json

from datasets import Dataset
import torch
from torch import nn
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    # default_data_collator,
)
from accelerate.logging import get_logger

from modeling.gpt import GPTConfig, GPTForCausalLM
from arguments import Args
from optim.lr_scheduler import WSDScheduler
from data import get_data


logger = get_logger(__name__)


def get_config(args: Args) -> GPTConfig:
    '''
    Load from `args.model_config` if it exists, then override the
    attributes with command line arguments.
    '''
    if args.model_config is not None:
        config = json.load(open(args.model_config, 'r'))
        model_config = GPTConfig(**config)
    else:
        model_config = GPTConfig()

    # Override values in the config from the command line arguments
    for k, v in args.as_dict().items():
        if hasattr(model_config, k) and v is not None:
            setattr(model_config, k, v)
    return model_config


def get_model(args: Args, device):
    if args.init_from == "scratch":
        # init a new model from scratch
        print("Initializing a new model from scratch")
        model_config = get_config(args)
        model = GPTForCausalLM(model_config).to(dtype=torch.bfloat16)
    elif args.init_from == "resume":
        raise NotImplementedError
    else:
        raise ValueError(f"Unknown initialization requested: {args.init_from}")

    # crop down the model block size if desired, using model surgery
    if args.max_len < model.config.max_len:
        model.crop_block_size(args.max_len)
        # so that the ckpt will have the right value
        model_config["block_size"] = args.max_len
    model.to(device)
    return model


def get_optimizer(
    model: nn.Module,
    weight_decay: float,
    lr: float,
    betas: tuple[float, float],
):
    # start with all the candidate parameters
    param_dict = {pn: p for pn, p in model.named_parameters()}
    # filter out those that do not require grad
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    print(
        f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
    )
    print(
        f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
    )
    # Create AdamW optimizer and use the fused version if it is available
    optimizer = torch.optim.AdamW(optim_groups, lr=lr, betas=betas)
    return optimizer


def main():
    args = Args().parse_args()
    output_dir = Path(args.output_dir)
    print(f"output_dir: {output_dir}")
    model = get_model(args, args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.tok_path)
    tokenized_data = get_data(
        tokenizer=tokenizer,
        data_name=args.data_name,
        data_path=args.data_path,
        max_len=args.max_len,
        n_workers=8,
    )

    # optimizer
    optimizer = get_optimizer(
        model,
        weight_decay=args.weight_decay,
        lr=args.lr,
        betas=(args.beta1, args.beta2),
    )
    lr_scheduler = WSDScheduler(
        optimizer,
        max_lr=args.lr,
        min_lr=args.min_lr,
        n_train_steps=args.n_train_steps,
        n_warmup_steps=args.n_warmup_steps,
        n_drop_steps=args.n_drop_steps,
    )
    if args.init_from == "resume":
        raise NotImplementedError

    train_args = TrainingArguments(
        output_dir=output_dir,
        do_train=True,
        do_eval=False,
        logging_strategy="steps",
        logging_steps=1,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        report_to="tensorboard",
        save_safetensors=False,
        seed=0,
        save_steps=args.eval_interval,
        max_steps=args.n_train_steps,
    )

    print(
        f"Process rank: {train_args.local_rank},"
        f" device: {train_args.device},"
        f" n_gpu: {train_args.n_gpu}"
    )
    print(
        f"distributed training: {train_args.parallel_mode.value == 'distributed'},"
        f" 16-bits training: {train_args.fp16}"
    )
    # print(f"Training/evaluation parameters {train_args}")

    # training loop
    train_ds: Dataset = tokenized_data["train"].with_format('torch')
    val_ds: Dataset = tokenized_data["validation"].with_format('torch')
    print(type(train_ds))
    print(type(val_ds))
    print("Creating trainer...")
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        optimizers=(optimizer, lr_scheduler),
    )

    print("====== Start training ======")
    trainer.train()


if __name__ == "__main__":
    main()
