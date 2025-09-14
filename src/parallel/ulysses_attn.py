from typing import Any

import torch
from torch import nn
from torch import Tensor
import torch.distributed as dist
from .comm.all2all import SeqAllToAll4D

from flash_attn import flash_attn_func


class UlyssesAttention(nn.Module):
    """
    A directly replacement to `flash_attn_func` from Flash-Attention that implements
    UlyssesAttention.

    Arguments:
        local_attention (Module): local attention with q,k,v
        sequence_process_group (ProcessGroup): sequence parallel process group
        scatter_idx (int): scatter_idx for all2all comm
        gather_idx (int): gather_idx for all2all comm
        use_sync (bool): whether to synchronize after all-to-all. This flag can save cuda memory but will slow down the speed.
        attn_type (AttnType): attention type enum
    """

    def __init__(
        self,
        sequence_process_group: dist.ProcessGroup | None = None,
        scatter_idx: int = 2,
        gather_idx: int = 1,
        use_sync: bool = False,
        attn_fn=None,
    ) -> None:

        super().__init__()
        self.spg = sequence_process_group
        self.scatter_idx = scatter_idx
        self.gather_idx = gather_idx
        self.use_sync = use_sync

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        gpu_name = torch.cuda.get_device_name(device)
        self.attn_fn = flash_attn_func

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        dropout_p=0.0,
        softmax_scale=None,
        causal=False,
        window_size=(-1, -1),
        # softcap=0.0,
        # alibi_slopes=None,
        # deterministic=False,
        # return_attn_probs=False,
        *args: Any
    ) -> Tensor:
        """forward

        Arguments:
            query (Tensor): query input to the layer
            key (Tensor): key input to the layer
            value (Tensor): value input to the layer
            args: other args

        Returns:
            * output (Tensor): context output
        """
        # scatter 2, gather 1
        q = SeqAllToAll4D.apply(
            self.spg, q, self.scatter_idx, self.gather_idx, self.use_sync
        )
        k = SeqAllToAll4D.apply(
            self.spg, k, self.scatter_idx, self.gather_idx, self.use_sync
        )
        v = SeqAllToAll4D.apply(
            self.spg, v, self.scatter_idx, self.gather_idx, self.use_sync
        )

        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** -0.5

        out = self.attn_fn(
            q=q,
            k=k,
            v=v,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            # softcap=softcap,
            # alibi_slopes=alibi_slopes,
            # deterministic=deterministic,
            # return_attn_probs=return_attn_probs,
        )

        if isinstance(out, tuple):
            out = out[0]

        # (bs, seq_len, head_cnt/N, head_size) -> (bs, seq_len/N, head_cnt, head_size)
        # scatter 1, gather 2
        out: Tensor = SeqAllToAll4D.apply(
            self.spg,
            out,
            self.gather_idx,
            self.scatter_idx,
            self.use_sync,
        )

        # out e.g., [s/p::h]
        return out


def monkey_patch_attn_fwd_with_ulysses(module: nn.Module):
    """
    We expect `module` to have a `attn_fn` method, and this function will replace that
    method with ulysses.
    """
    ulysses_attn = UlyssesAttention(attn_fn=module.attn_fn)
    module.attn_fn = ulysses_attn.forward


def extract_local(
    value: Tensor,
    rank: int,
    world_size: int,
    device,
    dim: int = 1,
) -> Tensor:
    dimension_size = value.shape[dim]
    sub_seq_length = dimension_size // world_size

    sub_seq_start = rank * sub_seq_length
    sub_seq_end = (rank + 1) * sub_seq_length
    local_value = value[:, sub_seq_start:sub_seq_end]

    return local_value.to(device)


def prepare_ulysses_attn_inputs(
    input_ids: Tensor,
    position_ids: Tensor | None,
    labels: Tensor | None,
    rank: int,
    world_size: int,
    device,
) -> dict[str, None | Tensor]:

    local_input_ids = extract_local(
        input_ids,
        rank,
        world_size,
        device,
    )

    if position_ids is not None:
        local_position_ids = extract_local(
            position_ids,
            rank,
            world_size,
            device,
        )
    else:
        local_position_ids = None

    if labels is not None:
        local_labels = extract_local(
            labels,
            rank,
            world_size,
            device,
        )
    else:
        local_labels = None
    return {
        "input_ids": local_input_ids,
        "position_ids": local_position_ids,
        "labels": local_labels,
    }
