import torch
from torch import nn, Tensor


class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor, labels: Tensor) -> Tensor:
        raise NotImplementedError

    def get_num_params(self, non_embedding: bool = True) -> int:
        raise NotImplementedError

    def get_trainable_parameters(self):
        raise NotImplementedError

    def estimate_mfu(
        self,
        fwdbwd_per_iter: int,
        iter_time: float,
        ideal_flops: float = 312e12,
    ) -> float:
        raise NotImplementedError
