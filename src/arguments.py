
from tap import Tap


class Args(Tap):
    '''
    If you want to specify arguments in a JSON file given by --train_config, then
    the possible arguments must have a `None` default value.
    '''
    pretrained_path: str | None = None  # Will load model weights from this path.
    resume_path: str | None = None
    '''
    When specified, will load training states (model weights, optimizer states, etc.) from this path.
    '''
    resume_step: int | None = None

    model_name: str = 'gpt'
    model_config: str | None = 'configs/model/gpt/2-256.json'
    train_config: str | None = None
    exp_group: str | None = None
    comment: str | None = None
    use_cu_seqlens: bool = False

    # Tokenizer
    tok_path: str = "./tokenizer/llama2"

    output_dir: str = "results"
    eval_interval: int | None = None
    log_interval: int = 1
    save_interval: int | None = None
    eval_iters: int | None = None
    eval_only: int = 0  # if not 0, script exits right after the first eval
    always_save_checkpoint: int = 1  # if not 0, always save a checkpoint after each eval
    init_from: str = 'scratch'  # 'scratch' or 'resume' or 'gpt2*'

    # wandb logging
    report_to: str = "tensorboard,swanlab"  # disabled by default
    proj_name: str = "my-project"
    run_name: str = "test"
    '''The code will append str(time.time()) to the run name.'''

    # Data
    n_workers: int = 1  # For DataLoader
    data_name: str | None = None
    data_path: str | None = None
    validation_data_path: str | None = None
    validation_data_name: str | None = None
    n_eval_batches: int | None = None
    grad_accum_steps: int | None = None
    '''per-gpu gradient accumulation steps'''
    batch_size: int | None = None
    '''If grad_accum_steps > 1, this is the per-gpu batch size.'''
    max_len: int | None = None
    is_seq2seq: int | None = None

    # adamw optimizer
    lr_scheduler: str | None = None
    lr: float | None = None  # max learning rate
    weight_decay: float | None = None
    beta1: float | None = None
    beta2: float | None = None
    grad_clip: float | None = None  # clip gradients at this value, or disable if == 0.0
    min_lr: float | None = None  # minimum learning rate, should be ~= lr/10 according to MiniCPM

    # learning rate decay settings
    n_warmup_steps: int | None = None
    n_train_steps: int | None = None
    n_drop_steps: int | None = None
    '''
    Should be 10% of n_train_iters according to MiniCPM, but we use 20% by default from:
    https://arxiv.org/pdf/2405.18392
    '''

    # DDP settings (only used by train_torch.py)
    backend: str = "nccl"  # 'nccl', 'gloo', etc.

    # system
    device: str = "cuda"
    """examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks"""
    machine_rank: int | None = None
    n_machines: int | None = None
    master_ip: str | None = None
    master_port: int | None = None
    use_deepspeed: int = 0
    n_gpus: int | None = None
    use_cp: int | None = None
    use_ulysses_attn: int | None = None

    dtype: str | None = None
    """
    float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler.
    """

    compile: int = 0
    '''
    If 1, the code will use PyTorch 2.0 to compile the model to be faster.
    '''
    seed: int = 0
    grad_ckpt: int = 0  # Whether to use gradient checkpointing
    use_cce_loss: int | None = None  # Whether to use Cut-CE loss (to reduce logits memory cost)
    shift_labels_in_model: int | None = 1  # Whether to shift labels inside the model
    cce_loss_impl: str = 'torch_compile'


if __name__ == "__main__":
    args = Args().parse_args()
    print(args)
