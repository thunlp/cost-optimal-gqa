import torch.nn.functional as F
import torch
from torch import nn, Tensor
from typing import Callable


def get_act_fn(act_name: str) -> Callable[[Tensor], Tensor]:
    if act_name == 'relu':
        return F.relu
    elif act_name == 'silu':
        return F.silu
    elif act_name == 'gelu':
        return F.gelu
    elif act_name == 'relu2':
        return lambda x: F.relu(x) ** 2
    elif act_name == 'exp':
        return torch.exp
    elif act_name == 'elu':
        return F.elu
    elif act_name == 'softplus':
        return F.softplus
    elif act_name == 'sigmoid':
        return torch.sigmoid
    else:
        raise ValueError(f"Unknown activation function: {act_name}")


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor) -> Tensor:
        output = self._norm(x.float()).type_as(x)
        if self.weight is None:
            return output
        return output * self.weight
