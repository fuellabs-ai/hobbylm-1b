"""HuggingFace configuration for HobbyLM sparse-MoE causal LM."""
from __future__ import annotations

from typing import Any

from packaging.version import Version
from transformers import __version__ as _transformers_version
from transformers.configuration_utils import PretrainedConfig

if not (Version("4.44.0") <= Version(_transformers_version) < Version("4.47.0")):
    raise ImportError(
        "HobbyLM HF port requires 4.44.0 <= transformers < 4.47.0 "
        f"for torch 2.4 compatibility; found transformers=={_transformers_version}"
    )


class HobbyLMConfig(PretrainedConfig):
    model_type = "hobbylm"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 50304,
        hidden_size: int = 1024,
        num_hidden_layers: int = 20,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 8,
        head_dim: int = 128,
        max_position_embeddings: int = 1024,
        rope_theta: float = 10000.0,
        partial_rotary_factor: float = 1.0,
        use_qk_norm: bool = True,
        rms_norm_eps: float = 1e-6,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        hidden_act: str = "silu",
        intermediate_size: int = 2816,
        moe_intermediate_size: int = 224,
        num_local_experts: int = 64,
        num_experts_per_tok: int = 8,
        n_shared_experts: int = 1,
        first_k_dense_replace: int = 1,
        gating: str = "sigmoid",
        norm_topk_prob: bool = False,
        balancing: str = "aux_free",
        aux_loss_coef: float = 1e-3,
        z_loss_coef: float = 1e-3,
        bias_update_rate: float = 1e-3,
        routed_scaling_factor: float = 1.0,
        initializer_range: float = 0.02,
        tie_word_embeddings: bool = True,
        scale_embeddings: bool = False,
        logit_softcap: float = 0.0,
        final_z_loss_coef: float = 1e-4,
        num_mtp_layers: int = 0,
        pad_token_id: int | None = None,
        bos_token_id: int | None = 50256,
        eos_token_id: int | None = 50256,
        use_cache: bool = True,
        attn_implementation: str = "sdpa",
        auto_map: dict[str, str] | None = None,
        **kwargs: Any,
    ):
        if auto_map is None:
            auto_map = {
                "AutoConfig": "configuration_hobbylm.HobbyLMConfig",
                "AutoModelForCausalLM": "modeling_hobbylm.HobbyLMForCausalLM",
            }
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            auto_map=auto_map,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.partial_rotary_factor = partial_rotary_factor
        self.use_qk_norm = use_qk_norm
        self.rms_norm_eps = rms_norm_eps
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_local_experts = num_local_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.n_shared_experts = n_shared_experts
        self.first_k_dense_replace = first_k_dense_replace
        self.gating = gating
        self.norm_topk_prob = norm_topk_prob
        self.balancing = balancing
        self.aux_loss_coef = aux_loss_coef
        self.z_loss_coef = z_loss_coef
        self.bias_update_rate = bias_update_rate
        self.routed_scaling_factor = routed_scaling_factor
        self.initializer_range = initializer_range
        self.scale_embeddings = scale_embeddings
        self.logit_softcap = logit_softcap
        self.final_z_loss_coef = final_z_loss_coef
        self.num_mtp_layers = num_mtp_layers
        self.use_cache = use_cache
        self.attn_implementation = attn_implementation
        self._attn_implementation = attn_implementation

        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        if self.num_experts_per_tok > self.num_local_experts:
            raise ValueError("num_experts_per_tok must be <= num_local_experts")
        if self.partial_rotary_factor != 1.0:
            raise ValueError("HobbyLM uses full RoPE; partial_rotary_factor must be 1.0")

    @property
    def num_experts(self) -> int:
        return self.num_local_experts

    @property
    def n_routed_experts(self) -> int:
        return self.num_local_experts

    @property
    def n_moe_layers(self) -> int:
        return self.num_hidden_layers - self.first_k_dense_replace
