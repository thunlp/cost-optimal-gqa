import json
from pathlib import Path

from accelerate.utils import set_seed
import torch
from torch import nn
from transformers import AutoTokenizer
from transformers import PretrainedConfig

from arguments import Args
from modeling import get_model
from trainer.trainer import LMTrainer
from preparation import load_train_config, get_accelerator, get_dataloaders, prepare_optimizers


def get_args() -> Args:
    args = Args().parse_args()

    if args.compile == 1:
        if args.grad_ckpt == 1:
            print("Cannot use grad checkpoint and compile mode together, setting compile to 0.")
            args.compile = 0

    return args


def main():
    torch.set_default_dtype(torch.bfloat16)
    args = get_args()
    load_train_config(args)
    set_seed(args.seed)
    accelerator = get_accelerator(args)

    accelerator.print("================ args ================")
    accelerator.print(args)
    accelerator.print("======================================")

    # Make output dir and dump args.
    output_dir = Path(args.output_dir, args.proj_name, args.run_name)
    if accelerator.is_main_process:
        output_dir.mkdir(exist_ok=True, parents=True)
        args.save(str(output_dir / "args.json"))

    # This is the actual batch size
    tokens_per_iter = (
        accelerator.num_processes
        * args.grad_accum_steps
        * args.batch_size
        * args.max_len
    )
    accelerator.print(f"Tokens per batch: {tokens_per_iter:,}")
    accelerator.print(f"Process: {accelerator.num_processes}")
    accelerator.print(f"Grad accum: {args.grad_accum_steps}")
    accelerator.print(f"Batch size: {args.batch_size}")
    accelerator.print(f"Max len: {args.max_len:,}")

    accelerator.print("Preparing tokenizer...")
    model: nn.Module = get_model(accelerator, args)
    accelerator.print("Preparing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.tok_path)

    accelerator.print("================ model ================")
    accelerator.print(model)
    accelerator.print("=======================================")

    if accelerator.is_main_process:
        # save model config
        model_config: PretrainedConfig = model.config  # type: ignore
        model_config_path = output_dir / "model_config.json"
        with open(model_config_path, "w") as f:
            json.dump(model_config.to_dict(), f, indent=4, sort_keys=True)

    accelerator.print("Preparing optimizers...")
    optimizer, lr_scheduler = prepare_optimizers(model=model, args=args)

    # Compile with PyTorch 2.0, very powerful
    if bool(args.compile):
        assert args.device != "mps", "torch.compile not supported on MPS"
        accelerator.print("compiling the model... (takes a ~minute)")
        # unoptimized_model = model
        model = torch.compile(model)  # requires PyTorch 2.0  # type: ignore

    accelerator.print("Preparing dataloaders...")
    train_loader, val_loader = get_dataloaders(
        args=args,
        tokenizer=tokenizer,
        accelerator=accelerator,
    )

    accelerator.print("Preparing LMTrainer...")
    trainer = LMTrainer(
        args=args,
        output_dir=output_dir,
        accelerator=accelerator,
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
    )

    accelerator.print("===== Start training =====")
    trainer.train()
    accelerator.print('===== Done training =====')


if __name__ == '__main__':
    main()
