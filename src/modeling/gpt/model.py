from typing import Optional, Tuple, List, Union
from functools import partial

from transformers.utils import logging
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
try:
    from cut_cross_entropy import linear_cross_entropy
except Exception:
    linear_cross_entropy = None
from transformers.generation import GenerationMixin
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import (
    CausalLMOutputWithPast,
    BaseModelOutputWithPast,
)
from transformers.cache_utils import Cache, DynamicCache

from ..ffn import MLP
from ..utils import RMSNorm
from .config import GPTConfig
from ..attention import Attention
from ..rope import RoPE, RoPESimple


logger = logging.get_logger(__name__)


class GPTBlock(nn.Module):
    """
    One layer in GPT.

    It does:
    1. x = x + Att(norm(x))
    2. x = x + FFN(norm(x))
    3. return x
    """

    def __init__(self, config: GPTConfig, layer_idx: int):
        super().__init__()
        self.config = config

        self.attn_norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.attn = Attention(
            hidden_size=config.d_model,
            head_dim=config.dim_k,
            n_heads=config.n_head,
            n_kv_heads=config.n_kv_head,
            layer_idx=layer_idx,
            use_q_norm=config.att_q_norm,
            use_k_norm=config.att_k_norm,
            window_size=config.att_window_size,
            bias=config.attn_bias,
            output_bias=config.attn_output_bias,
        )
        self.mlp_norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.mlp = MLP(
            act_name=config.ffn_act_fn,
            hidden_size=config.d_model,
            intermediate_dim=config.ffn_d_mid,
            is_gated=config.ffn_is_gated,
            bias=config.ffn_bias,
        )

        self.layer_idx = layer_idx

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
        **kwargs,
    ) -> Tensor | dict[str, Tensor | tuple[Tensor, Tensor]] | tuple:
        """
        - x: (B, T, d_model)
        - kv_cache: ((B, H, T, Dk), (B, H, T, Dv))
        """
        # ====== Attention ======
        print(self.attn)
        print(type(self.attn.attn_fn))
        residual = hidden_states
        hidden_states = self.attn_norm(hidden_states)
        hidden_states, self_attn_weights, present_key_value = self.attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,  # Position embeddings are always passed.
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + hidden_states

        # ====== MLP ======
        residual = hidden_states
        hidden_states = self.mlp_norm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class GPTPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = GPTConfig
    base_model_prefix = "transformer"
    supports_gradient_checkpointing = True
    _no_split_modules = ["GPTBlock"]

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def _init_weights(self, module):
        """Initialize the weights with Llama 2 scheme."""
        std = self.config.initializer_range
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class GPTModel(GPTPreTrainedModel):
    """
    The backbone model of GPT.

    This uses:
    - learned position embeddings.
    - RMSNorm for all the norms.
    - decoupled input and output embeddings.
    """

    def __init__(self, config: GPTConfig):
        super().__init__(config=config)
        self.config = config
        self.use_grad_ckpt = True

        self.input_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList([GPTBlock(config, i) for i in range(config.n_layer)])
        self.norm = RMSNorm(config.d_model, eps=config.norm_eps)

        if self.config.pos_emb_type == 'rope_naive':
            # RoPE's transformation values are pre-computed and cached in this module,
            # the same value is passed to each layer.
            self.rope = RoPE(
                theta=config.rope_theta,
                head_dim=config.dim_k,
                max_seqlen=config.max_len,
            )

        elif self.config.pos_emb_type == 'rope_simple':
            self.rope = RoPESimple(
                head_dim=config.dim_k,
                theta=config.rope_theta,
            )
            # Note that Qwen does not cache pre-computed position embeddings for each
            # position, so it does not need to know the max length.

        elif self.config.pos_emb_type == 'absolute':
            # Learned absolute position embeddings.
            self.pos_emb = nn.Embedding(config.max_len, config.d_model)

        else:
            raise ValueError(f"Invalid RoPE type: {self.config.rope_type}")

        self.model_parallel = False
        self.gradient_checkpointing = False

        self.post_init()

    def gradient_checkpointing_enable(self):
        """
        Enable gradient checkpointing for the model.
        """
        self.use_grad_ckpt = True

    def _get_position_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        device = input_ids.device
        position_ids = torch.arange(T, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(B, -1)
        return position_ids

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[Tensor] = None,
        past_key_values: Optional[Tensor] = None,
        inputs_embeds: Tensor = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> dict[str, Tensor | List[Tuple[Tensor, Tensor]]] | BaseModelOutputWithPast:
        """
        input_ids: (B, T)
        """
        B, T = input_ids.size()

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        # kept for BC (non `Cache` `past_key_values` inputs)
        return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):
            return_legacy_cache = True
            if past_key_values is None:
                past_key_values = DynamicCache()
            else:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
                logger.warning_once(
                    "We detected that you are passing `past_key_values` as a tuple of tuples. This is deprecated and "
                    "will be removed in v4.47. Please convert your cache or use an appropriate `Cache` class "
                    "(https://huggingface.co/docs/transformers/kv_cache#legacy-cache-format)"
                )

        if inputs_embeds is None:
            # (B, T, d_model)
            inputs_embeds = self.input_emb(input_ids)

        # ========== Position embeddings ===========
        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        hidden_states = inputs_embeds

        if self.config.pos_emb_type.startswith("rope"):
            # Create RoPE embeddings, shared across all layers
            position_embeddings = self.rope(hidden_states, position_ids)  # (B, T, dim_k/2, 2, 2)
        else:
            # Absolute position embeddings, directly added to the hidden states.
            abs_position_embeddings = self.pos_emb(position_ids)  # (T, d_model)
            # note: despite inconsistent shapes, "broadcasting" will work here.
            # (B, T, d_model)
            hidden_states = inputs_embeds + abs_position_embeddings
            position_embeddings = None

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.use_grad_ckpt and self.training:
                layer_forward = partial(checkpoint, layer.__call__, use_reentrant=False)
            else:
                layer_forward = layer.__call__

            layer_outputs = layer_forward(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )  # (B, T, d_model)

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)  # (B, T, d_model)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_decoder_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache:
            next_decoder_cache = next_decoder_cache.to_legacy_cache()

        if not return_dict:
            return tuple(v for v in [hidden_states, next_decoder_cache, all_hidden_states, all_self_attns] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class GPTForCausalLM(GPTPreTrainedModel, GenerationMixin):
    def __init__(self, config: GPTConfig):
        super().__init__(config=config)
        self.config = config
        self.use_grad_ckpt = False
        self.cce_loss_impl = 'torch_compile'

        self.model = GPTModel(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        if self.config.tie_emb:
            print("Tying embeddings...")
            self.lm_head.weight = self.model.input_emb.weight

        self.post_init()

    def get_input_embeddings(self):
        return self.model.input_emb

    def set_input_embeddings(self, value):
        self.model.input_emb = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, value):
        self.lm_head = value

    def get_num_params(
        self,
        non_embedding: bool = True,  # Whether to exclude input embedding parameters.
    ) -> int:
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            # Note that we include the output embedding, because
            # it is a part of the computation graph.
            if not self.config.use_rope:
                n_params -= self.model.pos_emb.weight.numel()
            n_params -= self.model.input_emb.weight.numel()
        return n_params

    def gradient_checkpointing_enable(self):
        """
        Enable gradient checkpointing for the model.
        """
        self.use_grad_ckpt = True
        self.model.gradient_checkpointing_enable()

    def compute_loss(
        self,
        hidden_states: Tensor,
        labels: Tensor,
    ) -> Tensor:
        if linear_cross_entropy is not None:
            # Use Fused CE implementation
            loss = linear_cross_entropy(
                hidden_states,
                self.lm_head.weight,
                labels,
                shift=True,
                impl=self.cce_loss_impl,
            )
        else:
            # PyTorch implementation
            logits = self.lm_head(hidden_states)  # (B, T, vocab_size)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        return loss

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: bool = True,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        **loss_kwargs,
    ) -> tuple | CausalLMOutputWithPast:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            num_logits_to_keep (`int`, *optional*):
                Calculate logits for the last `num_logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, GPTForCausalLM

        >>> model = GPTForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        outputs = self.model(
            input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            return_dict=return_dict,
            cache_position=cache_position,
        )
        hidden_states = outputs['last_hidden_state']  # (B, T, d_model)

        if self.training:
            logits = None
        else:
            # For inference, only need logits for the last token
            logits = self.lm_head(
                hidden_states[:, -num_logits_to_keep:]
            )  # (B, num_logits_to_keep, vocab_size)

        if labels is not None:
            loss = self.compute_loss(hidden_states, labels)
        else:
            loss = None

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @classmethod
    def from_pretrained(cls, model_type: str, override_args=None):
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        override_args = override_args or {}  # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == "dropout" for k in override_args)
        from transformers import GPT2LMHeadModel

        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, H and d_model are determined from model_type
        config_args = {
            "gpt2": dict(n_layer=12, H=12, d_model=768),  # 124M params
            "gpt2-medium": dict(n_layer=24, H=16, d_model=1024),  # 350M params
            "gpt2-large": dict(n_layer=36, H=20, d_model=1280),  # 774M params
            "gpt2-xl": dict(n_layer=48, H=25, d_model=1600),  # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, max_len=1024, bias=True")
        config_args["vocab_size"] = 50257  # always 50257 for GPT model checkpoints
        config_args["max_len"] = 1024  # always 1024 for GPT model checkpoints
        config_args["bias"] = True  # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if "dropout" in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args["dropout"] = override_args["dropout"]
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)  # type: ignore
        model = GPTForCausalLM(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [
            k for k in sd_keys if not k.endswith(".attn.bias")
        ]  # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [
            k for k in sd_keys_hf if not k.endswith(".attn.masked_bias")
        ]  # ignore these, just a buffer
        sd_keys_hf = [
            k for k in sd_keys_hf if not k.endswith(".attn.bias")
        ]  # same, just the mask (buffer)
        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(
            sd_keys
        ), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
