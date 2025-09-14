"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

from datetime import datetime
import os
import json
from pathlib import Path
import time
from typing import Optional
from contextlib import nullcontext
from functools import partial

from datasets import Dataset
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.distributed import init_process_group, destroy_process_group
from transformers import AutoTokenizer

from modeling.gpt import GPTConfig, GPT
from arguments import Args
from optim.lr_scheduler import get_wsd_lr
from data import get_data
from torch.utils.tensorboard import SummaryWriter


args = Args().parse_args()

# various inits, derived attributes, I/O setup
device = args.device
ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
print(f"DDP: {ddp}")
if ddp:
    init_process_group(backend=args.backend)
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    is_main_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank  # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert args.grad_accum_steps % ddp_world_size == 0
    args.grad_accum_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    is_main_process = True
    seed_offset = 0
    ddp_world_size = 1

# Make output dir and dump args.
output_dir = Path(args.output_dir)
if is_main_process:
    output_dir.mkdir(exist_ok=True, parents=True)
    print("================ args ================")
    print(args)
    print("======================================")
    args.save(str(output_dir / "args.json"))

tokens_per_iter = args.grad_accum_steps * ddp_world_size * args.batch_size * args.max_len
print(f"Tokens per batch will be: {tokens_per_iter:,}")

torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
device_type = "cuda" if "cuda" in device else "cpu"  # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}[args.dtype]
ctx = (
    nullcontext()
    if device_type == "cpu"
    else torch.autocast(device_type, dtype=ptdtype)
)

# init these up here, can override if init_from='resume' (i.e. from a ckpt)
cur_step = 0
best_val_loss = 1e9


def get_config(args: Args) -> GPTConfig:
    """
    Load from `args.model_config` if it exists, then override the
    attributes with command line arguments.
    """
    if args.model_config is not None:
        config = json.load(open(args.model_config, "r"))
        model_config = GPTConfig(**config)
    else:
        model_config = GPTConfig()

    # Override values in the config from the command line arguments
    for k, v in args.as_dict().items():
        if hasattr(model_config, k) and v is not None:
            setattr(model_config, k, v)
    return model_config


if args.init_from == "scratch":
    # init a new model from scratch
    print("Initializing a new model from scratch")
    model_config = get_config(args)
    model = GPT(model_config)
elif args.init_from == "resume":
    raise NotImplementedError
    print(f"Resuming training from {output_dir}")
    # resume training from a ckpt.
    ckpt_path = output_dir / "ckpt.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = ckpt["model_args"]
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ["n_layer", "n_head", "d_model", "block_size", "bias", "vocab_size"]:
        model_config[k] = checkpoint_model_args[k]
    # create the model
    model_config = GPTConfig(**model_config)
    model = GPT(model_config)
    state_dict = ckpt["model"]
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    cur_step = ckpt["cur_step"]
    best_val_loss = ckpt["best_val_loss"]
elif args.init_from.startswith("gpt2"):
    print(f"Initializing from OpenAI GPT-2 weights: {args.init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=args.dropout)
    model = GPT.from_pretrained(args.init_from, override_args)
    # read off the created config params, so we can store them into ckpt correctly
    for k in ["n_layer", "n_head", "d_model", "block_size", "bias", "vocab_size"]:
        model_config[k] = getattr(model.config, k)
    model_config = GPTConfig(**model_config)
else:
    raise ValueError(f"Unknown initialization requested: {args.init_from}")

# crop down the model block size if desired, using model surgery
if args.max_len < model.config.max_len:
    model.crop_block_size(args.max_len)
    # so that the ckpt will have the right value
    model_config["block_size"] = args.max_len
model.to(device=device)

if is_main_process:
    print("========== model ===========")
    print(model)
    print("============================")

tokenizer = AutoTokenizer.from_pretrained(args.tok_path)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.amp.GradScaler("cuda", enabled=(args.dtype == "float16"))

# optimizer
optimizer = model.configure_optimizers(
    weight_decay=args.weight_decay,
    lr=args.lr,
    betas=(args.beta1, args.beta2),
    device_type=device_type,
)
if args.init_from == "resume":
    assert ckpt is not None and "optimizer" in ckpt
    optimizer.load_state_dict(ckpt["optimizer"])
ckpt = None  # free up memory

if bool(args.compile):
    assert args.device != "mps", "torch.compile not supported on MPS"
    if is_main_process:
        print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model)  # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])


# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss(
    model: nn.Module,
    data_loader: DataLoader,
    n_batches: int,
    device: Optional[str] = None,
) -> torch.Tensor:
    model.eval()
    losses = torch.zeros(n_batches)
    print("==== Evaluation ====")
    i = 0
    for i, batch in enumerate(data_loader):
        if i == n_batches:
            break
        with ctx:
            loss, logits = model(**batch, device=device)
        losses[i] = loss.item()
    print("==== Evaluation Done ====")
    model.train()
    loss = losses.mean()
    return loss


get_lr = partial(
    get_wsd_lr,
    lr=args.lr,
    min_lr=args.min_lr,
    n_drop_iters=args.n_drop_steps,
    n_train_iters=args.n_train_steps,
    n_warmup_iters=args.n_warmup_steps,
)


def get_dataloaders(args: Args, tokenizer, is_main_process: bool):
    # training loop
    train_ds: Dataset = get_data(
        tokenizer=tokenizer,
        data_name=args.data_name,
        data_path=args.data_path,
        max_len=args.max_len,
        is_main_process=is_main_process,
    )["train"]  # type: ignore

    # For distributed training, dunno why tbh.
    print(f"Number of shards: {train_ds.n_shards}")
    # train_ds = split_dataset_by_node(train_ds, rank=int(os.environ["RANK"]), world_size=int(os.environ["WORLD_SIZE"]))

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        num_workers=args.n_workers,
    )
    if args.validation_data_path is not None and args.validation_data_name is not None:
        val_ds: Dataset = get_data(
            tokenizer=tokenizer,
            data_name=args.validation_data_name,
            data_path=args.validation_data_path,
            max_len=args.max_len,
            is_main_process=is_main_process,
        )["train"]  # type: ignore
        # val_ds = split_dataset_by_node(val_ds, rank=int(os.environ["RANK"]), world_size=int(os.environ["WORLD_SIZE"]))
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size * 4,
        )
    else:
        val_loader = None

    return train_loader, val_loader


train_loader, val_loader = get_dataloaders(
    args=args,
    tokenizer=tokenizer,
    is_main_process=is_main_process,
)

local_cur_step = 0  # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model  # unwrap DDP container if needed
running_mfu = -1.0


# TODO: Handle resumption about dataloader.

# logging
if is_main_process:
    if args.report_to == "wandb":
        import wandb

        wandb.init(project=args.proj_name, name=args.run_name, config=args.as_dict())
    elif args.report_to == "tensorboard":
        current_time = datetime.now()
        formatted_time = current_time.strftime("%Y%m%d-%H%M%S")
        writer = SummaryWriter(
            f"tensorboard/{args.proj_name}/{args.run_name}_{formatted_time}"
        )

# Training loop
data_iter = iter(train_loader)
batch = next(data_iter)

if is_main_process:
    print("===== Start training =====")
    print(f"Grad accum: {args.grad_accum_steps}")
    print(f"Batch size: {args.batch_size}")
    print(f"# train iters: {args.n_train_steps}")
    print(f"Eval interval: {args.eval_interval}")
    print(f"Log interval: {args.log_interval}")

# TODO: Handle resumption here
if args.init_from == "resume":
    ckpt_dirs = sorted(
        output_dir.glob("ckpt-*"), key=lambda x: int(x.name.split("-"))[1]
    )
    last_ckpt_dir = ckpt_dirs[-1]
    ckpt = torch.load(last_ckpt_dir / "ckpt.pt")
    raw_model.load_state_dict()


train_loss = None

last_log_time = time.time()

time_data = 0
time_fwd = 0
time_bwd = 0
time_datacp = 0

while True:
    # termination conditions
    if cur_step > args.n_train_steps:
        break

    # determine and set the learning rate for this iteration
    cur_lr = get_lr(cur_step)
    for param_group in optimizer.param_groups:
        param_group["lr"] = cur_lr

    # evaluate the loss on train/val sets and write checkpoints
    if is_main_process and val_loader is not None and cur_step % args.eval_interval == 0:
        val_loss = estimate_loss(
            model=model,
            data_loader=val_loader,
            n_batches=args.eval_iters,
            device=device,
        )
        print(f"Val loss: {val_loss:.4f}")
        if args.report_to == "tensorboard":
            writer.add_scalar("loss/val", val_loss, cur_step)
            # writer.add_hparams(
            #     hparam_dict=args.as_dict(),
            #     metric_dict={
            #         "val_loss": val_loss,
            #         "train_loss": -1.0 if train_loss is None else train_loss,
            #         "cur_step": cur_step,
            #     },
            # )
        if args.report_to == "wandb":
            wandb.log(
                {
                    "iter": cur_step,
                    "loss/val": val_loss,
                }
            )

        should_save = cur_step > 0 and (
            args.always_save_checkpoint or val_loss < best_val_loss
        )
        if should_save:
            best_val_loss = val_loss
            ckpt = {
                "model": raw_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "model_config": model_config.to_dict(),
                # "train_ds": train_loader.state_dict(),
                # "val_ds": val_loader.state_dict(),
                "cur_step": cur_step,
                "best_val_loss": best_val_loss,
                "config": args.as_dict(),
            }
            print(f"Found new best checkpoint, saving to {output_dir}")
            ckpt_dir = output_dir / f"ckpt-{cur_step}"
            ckpt_dir.mkdir(exist_ok=True, parents=True)
            ckpt_path = ckpt_dir / "ckpt.pt"
            torch.save(ckpt, ckpt_path)

    if cur_step == 0 and args.eval_only:
        break  # End training

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(args.grad_accum_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = micro_step == args.grad_accum_steps - 1
        with ctx:
            # Move data to GPU
            start_time = time.time()
            batch = {k: v.to(device) for k, v in batch.items()}
            time_datacp += time.time() - start_time

            # Forward pass
            start_time = time.time()
            loss, logits = model(**batch, device=args.device)
            time_fwd += time.time() - start_time
            # scale the loss to account for gradient accumulation
            loss /= args.grad_accum_steps
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        start_time = time.time()
        batch = next(data_iter)
        time_data += time.time() - start_time

        # Backward pass, with gradient scaling if training in fp16
        start_time = time.time()
        scaler.scale(loss).backward()
        time_bwd = time.time() - start_time
    # clip the gradient
    if args.grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    if is_main_process and cur_step % args.log_interval == 0:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        train_loss = loss.item() * args.grad_accum_steps
        cur_time = time.time()
        iter_time = (cur_time - last_log_time) / args.log_interval
        last_log_time = cur_time
        if local_cur_step >= 5:  # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(
                args.batch_size * args.grad_accum_steps,
                iter_time,
            )
            running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu

        time_data /= args.log_interval
        time_fwd /= args.log_interval
        time_bwd /= args.log_interval

        log_str = (
            f"[{cur_step}/{args.n_train_steps}]"
            f" loss {train_loss:.4f}"
            f", time {iter_time * 1000:.2f}ms"
            f", t_data {time_data * 1000:.2f}ms"
            f", t_datacp {time_datacp * 1000:.2f}ms"
            f", t_fwd {time_fwd * 1000:.2f}ms"
            f", t_bwd {time_bwd * 1000:.2f}ms"
            f", mfu {running_mfu * 100:.2f}%"
            f", lr {cur_lr:.3e}"
        )

        print(log_str)
        if args.report_to == "tensorboard":
            writer.add_scalar("iter", cur_step, cur_step)
            writer.add_scalar("loss/train", train_loss, cur_step)
            # writer.add_scalar('loss/val', val_loss, cur_step)
            writer.add_scalar("lr", cur_lr, cur_step)
            writer.add_scalar("efficiency/mfu", running_mfu * 100, cur_step)
            writer.add_scalar("efficiency/iter_time", iter_time, cur_step)
            writer.add_scalar("efficiency/iter_time", iter_time, cur_step)
            writer.add_scalar("efficiency/data_time", time_data, cur_step)
            writer.add_scalar("efficiency/datacp_time", time_data, cur_step)
            writer.add_scalar("efficiency/fwd_time", time_fwd, cur_step)
        elif args.report_to == "wandb":
            wandb.log(
                {
                    "iter": cur_step,
                    "train/loss": train_loss,
                    "lr": cur_lr,
                    "efficiency/mfu": running_mfu * 100,  # convert to percentage,
                    "efficiency/iter_time": iter_time,
                }
            )

        # Reset timers
        time_data = 0
        time_fwd = 0
        time_bwd = 0
        time_datacp = 0
    cur_step += 1
    local_cur_step += 1


if ddp:
    destroy_process_group()
