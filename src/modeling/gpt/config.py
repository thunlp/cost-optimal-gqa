import json
from typing import Optional
from transformers.configuration_utils import PretrainedConfig


class GPTConfig(PretrainedConfig):
    max_len: int = 512
    n_layer: int = 12
    d_model: int = 768
    initializer_range: float = 0.02

    # Vocab
    vocab_size: int = (
        50304  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    )
    tie_emb: bool = False
    use_cce_loss: bool = False

    # Attention
    n_head: int = 12
    n_kv_head: int = 12
    dim_k: int = 64
    dim_v: int = 64
    att_q_norm: bool = False
    att_k_norm: bool = False
    att_tie_kv: bool = False
    att_window_size: int = -1  # -1 means we don't use sliding window
    attn_bias: bool = False
    attn_output_bias: bool = False

    # Positional Embedding
    pos_emb_type: str = 'rope_simple'  # "rope_simple" is the RoPE used by Qwen.
    rope_theta: int = 500_000

    # FFN
    ffn_bias: bool = False
    ffn_per_layer: int = 1
    ffn_act_fn: str = "silu"
    ffn_is_gated: bool = True
    ffn_d_mid: Optional[int] = None

    # RMSNorm
    norm_eps: float = 1e-6

    # deprecated
    dropout: float = 0.0

    @classmethod
    def from_json_string(cls, json_string: str):
        return cls(**json.loads(json_string))
