"""HuggingFace remote-code model for HobbyLM-1B.

This is intentionally a custom implementation rather than a subclass of GLM4-MoE:
the trained checkpoint depends on HobbyLM's exact RoPE, fused-QKV conversion,
sigmoid aux-free router bias update, and expert tensor layout.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from packaging.version import Version
from transformers import __version__ as _transformers_version
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel

if not (Version("4.44.0") <= Version(_transformers_version) < Version("4.47.0")):
    raise ImportError(
        "HobbyLM HF port requires 4.44.0 <= transformers < 4.47.0 "
        f"for torch 2.4 compatibility; found transformers=={_transformers_version}"
    )

from .configuration_hobbylm import HobbyLMConfig


def _activation(name: str):
    if name in ("silu", "swish"):
        return F.silu
    raise ValueError(f"HobbyLM only supports SwiGLU/silu, got hidden_act={name!r}")


class HobbyLMRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: Tensor) -> Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


def rotate_half(x: Tensor) -> Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class HobbyLMRotaryEmbedding(nn.Module):
    def __init__(self, config: HobbyLMConfig):
        super().__init__()
        self.head_dim = config.head_dim
        self.base = config.rope_theta
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x: Tensor, position_ids: Tensor) -> tuple[Tensor, Tensor]:
        inv_freq = self.inv_freq.to(device=x.device, dtype=torch.float32)
        freqs = torch.einsum("bs,d->bsd", position_ids.float(), inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(x.dtype), emb.sin().to(x.dtype)


def apply_rotary_pos_emb(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor]:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: Tensor, n_rep: int) -> Tensor:
    if n_rep == 1:
        return hidden_states
    bsz, n_kv_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(bsz, n_kv_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(bsz, n_kv_heads * n_rep, slen, head_dim)


def _cache_update(past_key_values: Any, key_states: Tensor, value_states: Tensor, layer_idx: int):
    if past_key_values is None:
        return key_states, value_states
    if hasattr(past_key_values, "update"):
        return past_key_values.update(key_states, value_states, layer_idx)
    if isinstance(past_key_values, (tuple, list)) and len(past_key_values) > layer_idx:
        past = past_key_values[layer_idx]
        if past is not None:
            pk, pv = past
            key_states = torch.cat([pk, key_states], dim=2)
            value_states = torch.cat([pv, value_states], dim=2)
    return key_states, value_states


def _past_length(past_key_values: Any) -> int:
    if past_key_values is None:
        return 0
    if hasattr(past_key_values, "get_seq_length"):
        return past_key_values.get_seq_length()
    if isinstance(past_key_values, (tuple, list)) and past_key_values and past_key_values[0] is not None:
        return past_key_values[0][0].shape[2]
    return 0


class HobbyLMAttention(nn.Module):
    def __init__(self, config: HobbyLMConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.head_dim = config.head_dim
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)
        if config.use_qk_norm:
            self.q_norm = HobbyLMRMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = HobbyLMRMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def _flash_attention_2(self, q: Tensor, k: Tensor, v: Tensor, causal: bool) -> Tensor:
        try:
            from flash_attn import flash_attn_func
        except ImportError as exc:
            raise ImportError("flash_attention_2 requested but flash-attn is not installed") from exc
        return flash_attn_func(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            dropout_p=0.0 if not self.training else self.config.attention_dropout,
            causal=causal,
        ).transpose(1, 2)

    def forward(
        self,
        hidden_states: Tensor,
        position_embeddings: tuple[Tensor, Tensor],
        attention_mask: Tensor | None = None,
        past_key_values: Any = None,
        use_cache: bool | None = None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        bsz, q_len, _ = hidden_states.shape
        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if self.config.use_qk_norm:
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        key_states, value_states = _cache_update(past_key_values, key_states, value_states, self.layer_idx)

        causal = attention_mask is None and query_states.shape[-2] == key_states.shape[-2]
        impl = getattr(self.config, "_attn_implementation", self.config.attn_implementation)
        if impl == "flash_attention_2":
            attn_output = self._flash_attention_2(query_states, key_states, value_states, causal=causal)
        else:
            key_for_attn = repeat_kv(key_states, self.num_key_value_groups)
            value_for_attn = repeat_kv(value_states, self.num_key_value_groups)
            if impl == "eager":
                attn_weights = torch.matmul(query_states, key_for_attn.transpose(2, 3)) * (self.head_dim**-0.5)
                if causal:
                    q_len = query_states.shape[-2]
                    kv_len = key_for_attn.shape[-2]
                    q_pos = torch.arange(q_len, device=query_states.device)[:, None]
                    k_pos = torch.arange(kv_len, device=query_states.device)[None, :]
                    causal_mask = k_pos > (k_pos.shape[-1] - q_len + q_pos)
                    attn_weights = attn_weights.masked_fill(causal_mask.view(1, 1, q_len, kv_len), torch.finfo(attn_weights.dtype).min)
                if attention_mask is not None:
                    attn_weights = attn_weights + attention_mask
                attn_weights = F.softmax(attn_weights.float(), dim=-1).to(query_states.dtype)
                attn_output = torch.matmul(attn_weights, value_for_attn)
            else:
                attn_output = F.scaled_dot_product_attention(
                    query_states,
                    key_for_attn,
                    value_for_attn,
                    attn_mask=attention_mask,
                    dropout_p=0.0 if not self.training else self.config.attention_dropout,
                    is_causal=causal,
                )

        attn_output = attn_output.transpose(1, 2).reshape(bsz, q_len, self.num_heads * self.head_dim)
        return self.o_proj(attn_output), (key_states, value_states)


class HobbyLMDenseMLP(nn.Module):
    def __init__(self, config: HobbyLMConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = _activation(config.hidden_act)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class HobbyLMExperts(nn.Module):
    """Routed expert stack.

    Stored in HF/GLM orientation: gate_up_proj=(E, 2f, d), down_proj=(E, d, f).
    The gate/up ordering is HobbyLM's original [gate, up].
    """

    def __init__(self, config: HobbyLMConfig, num_experts: int | None = None, intermediate_size: int | None = None):
        super().__init__()
        self.num_experts = config.num_local_experts if num_experts is None else num_experts
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.moe_intermediate_size if intermediate_size is None else intermediate_size
        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, 2 * self.intermediate_size, self.hidden_size))
        self.down_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_size, self.intermediate_size))
        self.act_fn = _activation(config.hidden_act)
        self.last_expert_outputs: dict[int, Tensor] = {}

    def expert_forward(self, hidden_states: Tensor, expert_idx: int) -> Tensor:
        gate, up = F.linear(hidden_states, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
        return F.linear(self.act_fn(gate) * up, self.down_proj[expert_idx])

    def forward(self, hidden_states: Tensor, topk_indices: Tensor, topk_weights: Tensor) -> Tensor:
        final_hidden_states = torch.zeros_like(hidden_states)
        self.last_expert_outputs = {}
        flat_experts = topk_indices.reshape(-1)
        flat_tokens = torch.arange(hidden_states.shape[0], device=hidden_states.device).repeat_interleave(topk_indices.shape[1])
        flat_weights = topk_weights.reshape(-1)
        for expert_idx in range(self.num_experts):
            selected = (flat_experts == expert_idx).nonzero(as_tuple=True)[0]
            if selected.numel() == 0:
                continue
            token_idx = flat_tokens[selected]
            expert_out = self.expert_forward(hidden_states[token_idx], expert_idx).to(hidden_states.dtype)
            self.last_expert_outputs[expert_idx] = expert_out.detach()
            weighted = expert_out * flat_weights[selected].unsqueeze(-1).to(hidden_states.dtype)
            final_hidden_states.index_add_(0, token_idx, weighted)
        return final_hidden_states


class HobbyLMSharedExpert(nn.Module):
    def __init__(self, config: HobbyLMConfig):
        super().__init__()
        self.experts = HobbyLMExperts(
            config,
            num_experts=config.n_shared_experts,
            intermediate_size=config.moe_intermediate_size,
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        out = torch.zeros_like(hidden_states)
        for expert_idx in range(self.experts.num_experts):
            out = out + self.experts.expert_forward(hidden_states, expert_idx).to(hidden_states.dtype)
        return out


class HobbyLMTopKRouter(nn.Module):
    def __init__(self, config: HobbyLMConfig):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_local_experts
        self.weight = nn.Parameter(torch.empty(self.num_experts, config.hidden_size))
        self.register_buffer("expert_bias", torch.zeros(self.num_experts, dtype=torch.float32))
        self.bias_update_rate = config.bias_update_rate
        self.last_topi: Tensor | None = None
        self.last_topv: Tensor | None = None
        self.last_logits: Tensor | None = None
        self.last_aux_loss: Tensor | None = None

    def forward(self, hidden_states: Tensor, routing_mask: Tensor | None = None) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        with torch.autocast(device_type=hidden_states.device.type, enabled=False):
            logits = F.linear(hidden_states.float(), self.weight.float())
            if self.config.gating == "sigmoid":
                scores = torch.sigmoid(logits)
            elif self.config.gating == "softmax":
                scores = torch.softmax(logits, dim=-1)
            else:
                raise ValueError(f"unsupported gating={self.config.gating!r}")

            selection_scores = scores + self.expert_bias.float() if self.config.balancing == "aux_free" else scores
            topk_indices = torch.topk(selection_scores, self.top_k, dim=-1).indices
            topk_weights = torch.gather(scores, -1, topk_indices)
            if self.config.norm_topk_prob:
                topk_weights = topk_weights / (topk_weights.sum(-1, keepdim=True) + 1e-9)
            topk_weights = topk_weights * self.config.routed_scaling_factor

            if routing_mask is None:
                valid = torch.ones(hidden_states.shape[0], dtype=torch.bool, device=hidden_states.device)
            else:
                valid = routing_mask.reshape(-1).to(device=hidden_states.device, dtype=torch.bool)
            valid_tokens = valid.sum().clamp_min(1)
            valid_topk = topk_indices[valid]
            counts = torch.bincount(valid_topk.reshape(-1), minlength=self.num_experts).float()
            f_i = counts / (valid_tokens.float() * self.top_k)
            p_i = scores[valid].mean(dim=0) if bool(valid.any()) else scores.mean(dim=0)
            aux = self.num_experts * (f_i.detach() * p_i).sum()
            z_loss_values = torch.logsumexp(logits, dim=-1) ** 2
            z_loss = z_loss_values[valid].mean() if bool(valid.any()) else z_loss_values.mean()
            aux_total = self.config.aux_loss_coef * aux + self.config.z_loss_coef * z_loss

            if self.config.balancing == "aux_free" and self.training and self.bias_update_rate > 0:
                with torch.no_grad():
                    ideal = valid_tokens.float() * self.top_k / self.num_experts
                    self.expert_bias.add_(self.bias_update_rate * torch.sign(ideal - counts))

        self.last_topi = topk_indices.detach()
        self.last_topv = topk_weights.detach()
        self.last_logits = logits.detach()
        self.last_aux_loss = aux_total.detach()
        return logits, topk_weights, topk_indices, aux_total


class HobbyLMMoE(nn.Module):
    def __init__(self, config: HobbyLMConfig):
        super().__init__()
        self.gate = HobbyLMTopKRouter(config)
        self.experts = HobbyLMExperts(config)
        self.shared_experts = HobbyLMSharedExpert(config) if config.n_shared_experts > 0 else None

    @property
    def last_topi(self) -> Tensor | None:
        return self.gate.last_topi

    @property
    def last_topv(self) -> Tensor | None:
        return self.gate.last_topv

    def forward(self, hidden_states: Tensor, routing_mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        orig_shape = hidden_states.shape
        flat_states = hidden_states.reshape(-1, hidden_states.shape[-1])
        flat_mask = routing_mask.reshape(-1) if routing_mask is not None else None
        _, topk_weights, topk_indices, aux = self.gate(flat_states, flat_mask)
        if flat_mask is not None:
            topk_weights = topk_weights * flat_mask.to(topk_weights.dtype).unsqueeze(-1)
        routed = self.experts(flat_states, topk_indices, topk_weights)
        if self.shared_experts is not None:
            shared = self.shared_experts(flat_states)
            if flat_mask is not None:
                shared = shared * flat_mask.to(shared.dtype).unsqueeze(-1)
            routed = routed + shared
        return routed.reshape(*orig_shape), aux


class HobbyLMDecoderLayer(nn.Module):
    def __init__(self, config: HobbyLMConfig, layer_idx: int):
        super().__init__()
        self.self_attn = HobbyLMAttention(config, layer_idx=layer_idx)
        self.mlp = HobbyLMMoE(config) if layer_idx >= config.first_k_dense_replace else HobbyLMDenseMLP(config)
        self.input_layernorm = HobbyLMRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = HobbyLMRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.is_moe = layer_idx >= config.first_k_dense_replace

    def forward(
        self,
        hidden_states: Tensor,
        position_embeddings: tuple[Tensor, Tensor],
        attention_mask: Tensor | None = None,
        routing_mask: Tensor | None = None,
        past_key_values: Any = None,
        use_cache: bool | None = None,
    ) -> tuple[Tensor, Tensor, tuple[Tensor, Tensor]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_out, present = self.self_attn(
            hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        hidden_states = residual + attn_out

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        if self.is_moe:
            mlp_out, aux = self.mlp(hidden_states, routing_mask=routing_mask)
        else:
            mlp_out = self.mlp(hidden_states)
            if routing_mask is not None:
                mlp_out = mlp_out * routing_mask.to(mlp_out.dtype).unsqueeze(-1)
            aux = hidden_states.new_zeros(())
        hidden_states = residual + mlp_out
        return hidden_states, aux, present


class HobbyLMPreTrainedModel(PreTrainedModel):
    config_class = HobbyLMConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["HobbyLMDecoderLayer"]
    _supports_sdpa = True
    _supports_flash_attn = True
    _supports_attention_backend = True
    _tied_weights_keys = ["lm_head.weight"]

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)
        elif isinstance(module, HobbyLMExperts):
            nn.init.normal_(module.gate_up_proj, mean=0.0, std=std)
            nn.init.normal_(module.down_proj, mean=0.0, std=std)
        elif isinstance(module, HobbyLMTopKRouter):
            nn.init.normal_(module.weight, mean=0.0, std=std * 0.1)
            nn.init.zeros_(module.expert_bias)


class HobbyLMModel(HobbyLMPreTrainedModel):
    def __init__(self, config: HobbyLMConfig):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList([HobbyLMDecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = HobbyLMRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = HobbyLMRotaryEmbedding(config)
        self.gradient_checkpointing = False
        self.post_init()

    def _causal_mask(self, hidden_states: Tensor, attention_mask: Tensor | None, past_key_values: Any) -> Tensor | None:
        if attention_mask is None:
            return None
        bsz, q_len, _ = hidden_states.shape
        past_len = _past_length(past_key_values)
        kv_len = past_len + q_len
        q_pos = torch.arange(q_len, device=hidden_states.device)[:, None] + past_len
        k_pos = torch.arange(kv_len, device=hidden_states.device)[None, :]
        blocked = k_pos > q_pos
        mask = torch.zeros((1, 1, q_len, kv_len), dtype=hidden_states.dtype, device=hidden_states.device)
        mask = mask.masked_fill(blocked.view(1, 1, q_len, kv_len), torch.finfo(hidden_states.dtype).min)
        mask = mask.expand(bsz, 1, q_len, kv_len).clone()
        if attention_mask.dim() == 2:
            pad = (1.0 - attention_mask[:, None, None, :].to(hidden_states.dtype)) * torch.finfo(hidden_states.dtype).min
            mask = mask + pad[:, :, :, -kv_len:]
        fully_masked = torch.isneginf(mask) | (mask == torch.finfo(hidden_states.dtype).min)
        fully_masked = fully_masked.all(dim=-1, keepdim=True)
        if bool(fully_masked.any()):
            mask = mask.masked_fill(fully_masked.expand_as(mask), 0.0)
        return mask

    def _position_ids(
        self,
        inputs_embeds: Tensor,
        attention_mask: Tensor | None,
        past_key_values: Any,
    ) -> Tensor:
        past_seen = _past_length(past_key_values)
        seq_len = inputs_embeds.shape[1]
        if attention_mask is not None and attention_mask.dim() == 2:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 0)
            position_ids = position_ids[:, -seq_len:]
            return position_ids.to(inputs_embeds.device)
        return torch.arange(seq_len, device=inputs_embeds.device).unsqueeze(0) + past_seen

    def forward(
        self,
        input_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        past_key_values: Any = None,
        inputs_embeds: Tensor | None = None,
        use_cache: bool | None = None,
        **_: Any,
    ):
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
            if self.config.scale_embeddings:
                inputs_embeds = inputs_embeds * (self.config.hidden_size**0.5)

        use_cache = self.config.use_cache if use_cache is None else use_cache
        if position_ids is None:
            position_ids = self._position_ids(inputs_embeds, attention_mask, past_key_values)
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)
        causal_mask = self._causal_mask(inputs_embeds, attention_mask, past_key_values)
        routing_mask = None
        if attention_mask is not None and attention_mask.dim() == 2:
            routing_mask = attention_mask[:, -inputs_embeds.shape[1] :].to(device=inputs_embeds.device, dtype=torch.bool)

        hidden_states = inputs_embeds
        aux_sum = hidden_states.new_zeros(())
        next_cache = [] if use_cache and not hasattr(past_key_values, "update") else past_key_values
        for layer in self.layers:
            hidden_states, aux, present = layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=causal_mask,
                routing_mask=routing_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
            )
            aux_sum = aux_sum + aux
            if routing_mask is not None:
                hidden_states = hidden_states * routing_mask.to(hidden_states.dtype).unsqueeze(-1)
            if isinstance(next_cache, list):
                next_cache.append(present)
        hidden_states = self.norm(hidden_states)
        if routing_mask is not None:
            hidden_states = hidden_states * routing_mask.to(hidden_states.dtype).unsqueeze(-1)
        return hidden_states, aux_sum, tuple(next_cache) if isinstance(next_cache, list) else next_cache


@dataclass
class HobbyLMCausalLMOutput:
    loss: Tensor | None = None
    logits: Tensor | None = None
    past_key_values: Any = None
    aux_loss: Tensor | None = None


class HobbyLMForCausalLM(HobbyLMPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: HobbyLMConfig):
        super().__init__(config)
        self.model = HobbyLMModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def tie_weights(self):
        if self.config.tie_word_embeddings:
            self._tie_or_clone_weights(self.lm_head, self.model.embed_tokens)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **kwargs):
        if past_key_values is not None and _past_length(past_key_values) > 0:
            input_ids = input_ids[:, -1:]
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
            "use_cache": kwargs.get("use_cache", True),
        }

    def forward(
        self,
        input_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        past_key_values: Any = None,
        inputs_embeds: Tensor | None = None,
        labels: Tensor | None = None,
        use_cache: bool | None = None,
        return_dict: bool | None = None,
        logits_to_keep: int | Tensor = 0,
        **kwargs: Any,
    ):
        hidden_states, aux_sum, next_cache = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) and logits_to_keep else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :] if logits_to_keep else hidden_states)
        if self.config.logit_softcap > 0:
            logits = self.config.logit_softcap * torch.tanh(logits / self.config.logit_softcap)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous().float()
            shift_labels = labels[..., 1:].contiguous()
            ce = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100)
            z = (torch.logsumexp(logits.float(), dim=-1) ** 2).mean()
            loss = ce + self.config.final_z_loss_coef * z + aux_sum

        if return_dict is False:
            out = (logits, next_cache, aux_sum)
            return ((loss,) + out) if loss is not None else out
        return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=next_cache)

    def count_parameters(self) -> dict[str, int | float]:
        total = sum(p.numel() for p in self.parameters())
        per_expert = self.config.hidden_size * 2 * self.config.moe_intermediate_size + self.config.moe_intermediate_size * self.config.hidden_size
        inactive = self.config.n_moe_layers * (self.config.num_local_experts - self.config.num_experts_per_tok) * per_expert
        active = total - inactive
        return {"total": total, "active": active, "active_pct": 100 * active / total}
