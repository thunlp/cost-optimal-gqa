from torch import nn


def get_num_params(model: nn.Module, non_embedding: bool = False) -> int:
    """
    Get the number of parameters in the model.
    """
    cnt = 0
    for n, p in model.named_parameters():
        if non_embedding:
            if "embed" in n or "emb." in n:
                continue
        cnt += p.numel()
    return cnt
