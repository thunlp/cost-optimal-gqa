import json

from safetensors.torch import load_file

from arguments import Args
from .model import GPTForCausalLM, GPTConfig


def get_config(
    path: str | None = None,
    args: Args | None = None,
) -> GPTConfig:
    """
    Load from `args.model_config` if it exists, then override the
    attributes with command line arguments.
    """
    if path is not None:
        model_config: GPTConfig = GPTConfig.from_json_file(path)  # type: ignore
    else:
        model_config = GPTConfig()  # use default values

    if args is not None:
        # Override values in the config from the command line arguments
        for k, v in args.as_dict().items():
            if hasattr(model_config, k) and v is not None:
                setattr(model_config, k, v)
    return model_config


def load_state_dict(model, path):
    '''
    This will load the checkpoint from path, and replace the
    model's parameters with the loaded state dict.
    '''
    print(f"Loading model parameters from: {path}")
    state_dict = load_file(path)
    state_dict = {key.replace('_orig_mod.', ''): val for key, val in state_dict.items()}
    state_dict = {key.replace('.w1.', '.up_proj.'): val for key, val in state_dict.items()}
    state_dict = {key.replace('.w2.', '.down_proj.'): val for key, val in state_dict.items()}
    # state_dict = {key.replace('model.lm_head.', '.lm_head.'): val for key, val in state_dict.items()}
    state_dict = {key.replace('.w_q.', '.q_proj.'): val for key, val in state_dict.items()}
    state_dict = {key.replace('.w_k.', '.k_proj.'): val for key, val in state_dict.items()}
    state_dict = {key.replace('.w_v.', '.v_proj.'): val for key, val in state_dict.items()}
    state_dict = {key.replace('.w_o.', '.o_proj.'): val for key, val in state_dict.items()}
    state_dict = {key.replace('.att.', '.attn.'): val for key, val in state_dict.items()}
    state_dict = {key.replace('.att_norm.', '.attn_norm.'): val for key, val in state_dict.items()}
    state_dict = {key.replace('.ffns.0.', '.mlp.'): val for key, val in state_dict.items()}
    state_dict = {key.replace('.ffn_norms.0.', '.mlp_norm.'): val for key, val in state_dict.items()}

    print(list(state_dict.keys()))
    state_dict['lm_head.weight'] = state_dict['model.input_emb.weight']
    model.load_state_dict(state_dict, strict=True)


def get_model(args: Args, config_path: str | None = None) -> GPTForCausalLM:
    model_config = get_config(
        path=config_path,
        args=args,
    )
    model = GPTForCausalLM(model_config)
    if args.init_from is not None and args.init_from != 'scratch':
        load_state_dict(model, args.init_from)

    if bool(args.grad_ckpt):
        model.gradient_checkpointing_enable()

    # Enable Ulysses attention by monkey patching
    if bool(args.use_ulysses_attn):
        from parallel.ulysses_attn import monkey_patch_attn_fwd_with_ulysses
        print("Monkey patching layers...")
        for layer in model.model.layers:
            print(type(layer))
            monkey_patch_attn_fwd_with_ulysses(layer.attn)

    return model
