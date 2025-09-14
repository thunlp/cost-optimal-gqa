from accelerate import Accelerator
from arguments import Args
from torch import nn
from utils import get_num_params


def get_model(
    accelerator: Accelerator,
    args: Args,
) -> nn.Module:
    if args.model_name == "gpt":
        from modeling import gpt

        model = gpt.get_model(
            args=args,
            config_path=args.model_config,
        )
    elif args.model_name == "llama3":
        from modeling import llama3

        model = llama3.get_model(args=args)
    elif args.model_name in ["gdn", "gated-deltanet", "gated_deltanet"]:
        from modeling import gated_deltanet

        model = gated_deltanet.get_model(
            config_path=args.model_config,
            args=args,
        )
    elif args.model_name in ["multi_step_gd", "rabbit"]:
        from modeling import rabbit

        model = rabbit.get_model(args=args)
    elif args.model_name in ["mamba2"]:
        from modeling import mamba2

        model = mamba2.get_model(
            args=args,
            config_path=args.model_config,
        )
    elif args.model_name in ["ttt", "test-time-training", "ttt-linear", 'ttt-mlp']:
        from modeling import ttt

        model = ttt.get_model(args=args)
    elif args.model_name in ['t5', 'encoder-decoder', 'encoder_decoder', 'enc_dec', 'enc-dec']:
        from modeling import enc_dec

        model = enc_dec.get_model(
            args=args, config_path=args.model_config)
    elif args.model_name in ['rabbit-ed']:
        from modeling import rabbit_ed
        model = rabbit_ed.get_model(
            args=args, config_path=args.model_config,
        )
    else:
        raise ValueError(f"Unknown model name {args.model_name}")

    n_non_embed_param = get_num_params(model, non_embedding=True)
    n_param = get_num_params(model, non_embedding=False)
    accelerator.print("=========================================")
    accelerator.print(f"# parameters: {n_param:,}")
    accelerator.print(f"# parameters (non-embed): {n_non_embed_param:,}")
    accelerator.print("=========================================")

    accelerator.print("======== model config =========")
    accelerator.print(model.config)
    accelerator.print("===============================")

    model.to(accelerator.device)
    return model
