"""HobbyLM-1B (HF release format) -> F32 GGUF, architecture `bailingmoe2` (needs the one-line llama.cpp QKV patch:
bailingmoe2_decoupled_head_dim.patch, because Q width = n_head*head_dim = 2048 != n_embd = 1024).

Source = the published model.safetensors (verified bit-identical to the raw checkpoints) + config.json of the SAME
release directory, so every hyper-parameter comes from the actual checkpoint configuration:
  context length = max_position_embeddings (base 1024, SFT 4096); RMSNorm eps = rms_norm_eps (1e-6);
  rope theta 10000, full rotary (rope_dimension_count = head_dim = 128), NEOX/rotate-half (bailingmoe2 default);
  attention: 16 q heads / 8 kv heads, head_dim 128 (Q width 2048); fused QKV rebuilt as [q; k; v] rows;
  MoE: 64 experts, top-8, sigmoid gating, expert bias used for selection only, no top-k renorm, scale 1.0,
  1 shared expert, 1 leading dense layer; tied embeddings written as output.weight.
Tokenizer: GPT-2 BPE from the staged tokenizer.json, padded to the 50304-row embedding with unused tokens.
Chat template: the verified SFT template from tokenizer_config.json (SFT only).
Usage: python to_gguf_1b.py <release_dir> <out.gguf>
"""
import json
import sys
from pathlib import Path

import numpy as np
from safetensors import safe_open

import gguf

src, out = Path(sys.argv[1]), sys.argv[2]
cfg = json.loads((src / "config.json").read_text())
tcfg = json.loads((src / "tokenizer_config.json").read_text())
tj = json.loads((src / "tokenizer.json").read_text(encoding="utf-8"))

d, L = cfg["hidden_size"], cfg["num_hidden_layers"]
nq, nkv, hd = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
E, K, f, fd = cfg["num_local_experts"], cfg["num_experts_per_tok"], cfg["moe_intermediate_size"], cfg["intermediate_size"]
n_dense, n_sh, V = cfg["first_k_dense_replace"], cfg["n_shared_experts"], cfg["vocab_size"]
assert cfg["gating"] == "sigmoid" and cfg["balancing"] == "aux_free" and cfg["norm_topk_prob"] is False
assert float(cfg.get("routed_scaling_factor", 1.0)) == 1.0 and cfg.get("rope_scaling") is None
assert float(cfg.get("partial_rotary_factor", 1.0)) == 1.0 and cfg["use_qk_norm"] and float(cfg.get("logit_softcap", 0.0)) == 0.0
assert not cfg.get("scale_embeddings") and cfg.get("num_mtp_layers", 0) == 0

w = gguf.GGUFWriter(out, "bailingmoe2")
w.add_name(f"HobbyLM-1B ({src.name})")
w.add_context_length(cfg["max_position_embeddings"])
w.add_embedding_length(d)
w.add_block_count(L)
w.add_feed_forward_length(fd)
w.add_head_count(nq)
w.add_head_count_kv(nkv)
w.add_key_length(hd)
w.add_value_length(hd)
w.add_rope_dimension_count(hd)
w.add_layer_norm_rms_eps(float(cfg["rms_norm_eps"]))
w.add_rope_freq_base(float(cfg["rope_theta"]))
w.add_expert_count(E)
w.add_expert_used_count(K)
w.add_expert_feed_forward_length(f)
w.add_expert_shared_count(n_sh)
w.add_expert_shared_feed_forward_length(f)
w.add_leading_dense_block_count(n_dense)
w.add_expert_gating_func(gguf.ExpertGatingFuncType.SIGMOID)
w.add_expert_weights_norm(False)
w.add_expert_weights_scale(1.0)
w.add_vocab_size(V)
w.add_uint32("bailingmoe2.nextn_predict_layers", 0)
w.add_file_type(gguf.LlamaFileType.ALL_F32)

# ---- tokenizer: GPT-2 BPE, padded to V rows ----
vocab = tj["model"]["vocab"]
inv = {i: t for t, i in vocab.items()}
for at in tj.get("added_tokens", []):
    inv[at["id"]] = at["content"]
n_real = max(inv) + 1
toks, types = [], []
for i in range(V):
    if i in inv:
        toks.append(inv[i])
        types.append(gguf.TokenType.CONTROL if i == 50256 else gguf.TokenType.NORMAL)
    else:
        toks.append(f"[PAD{i}]")
        types.append(gguf.TokenType.UNUSED)
merges = [m if isinstance(m, str) else " ".join(m) for m in tj["model"]["merges"]]
w.add_tokenizer_model("gpt2")
w.add_tokenizer_pre("gpt-2")
w.add_token_list(toks)
w.add_token_types(types)
w.add_token_merges(merges)
w.add_bos_token_id(50256)
w.add_eos_token_id(50256)
w.add_pad_token_id(50256)
w.add_add_bos_token(False)
w.add_add_eos_token(False)
if tcfg.get("chat_template"):
    w.add_chat_template(tcfg["chat_template"])

# ---- tensors ----
st = safe_open(str(src / "model.safetensors"), framework="np")
keys = set(st.keys())
used = set()


def g(k):
    used.add(k)
    return st.get_tensor(k)


def t(name, arr):
    w.add_tensor(name, np.ascontiguousarray(arr.astype(np.float32)))


t("token_embd.weight", g("model.embed_tokens.weight"))
t("output_norm.weight", g("model.norm.weight"))
t("output.weight", g("lm_head.weight") if "lm_head.weight" in keys else st.get_tensor("model.embed_tokens.weight"))
for i in range(L):
    p = f"model.layers.{i}."
    t(f"blk.{i}.attn_norm.weight", g(p + "input_layernorm.weight"))
    q, k, v = g(p + "self_attn.q_proj.weight"), g(p + "self_attn.k_proj.weight"), g(p + "self_attn.v_proj.weight")
    assert q.shape == (nq * hd, d) and k.shape == (nkv * hd, d) and v.shape == (nkv * hd, d)
    t(f"blk.{i}.attn_qkv.weight", np.concatenate([q, k, v], axis=0))
    t(f"blk.{i}.attn_q_norm.weight", g(p + "self_attn.q_norm.weight"))
    t(f"blk.{i}.attn_k_norm.weight", g(p + "self_attn.k_norm.weight"))
    t(f"blk.{i}.attn_output.weight", g(p + "self_attn.o_proj.weight"))
    t(f"blk.{i}.ffn_norm.weight", g(p + "post_attention_layernorm.weight"))
    if i < n_dense:
        t(f"blk.{i}.ffn_gate.weight", g(p + "mlp.gate_proj.weight"))
        t(f"blk.{i}.ffn_up.weight", g(p + "mlp.up_proj.weight"))
        t(f"blk.{i}.ffn_down.weight", g(p + "mlp.down_proj.weight"))
    else:
        t(f"blk.{i}.ffn_gate_inp.weight", g(p + "mlp.gate.weight"))
        t(f"blk.{i}.exp_probs_b.bias", g(p + "mlp.gate.expert_bias"))
        gu = g(p + "mlp.experts.gate_up_proj")            # (E, 2f, d): rows [gate; up]
        assert gu.shape == (E, 2 * f, d)
        t(f"blk.{i}.ffn_gate_exps.weight", gu[:, :f, :])
        t(f"blk.{i}.ffn_up_exps.weight", gu[:, f:, :])
        t(f"blk.{i}.ffn_down_exps.weight", g(p + "mlp.experts.down_proj"))   # (E, d, f)
        sgu = g(p + "mlp.shared_experts.experts.gate_up_proj")[0]            # (2f, d)
        t(f"blk.{i}.ffn_gate_shexp.weight", sgu[:f])
        t(f"blk.{i}.ffn_up_shexp.weight", sgu[f:])
        t(f"blk.{i}.ffn_down_shexp.weight", g(p + "mlp.shared_experts.experts.down_proj")[0])   # (d, f)

unused = sorted(keys - used - {"lm_head.weight"})
assert not unused, f"unmapped tensors: {unused[:10]}"
w.write_header_to_file()
w.write_kv_data_to_file()
w.write_tensors_to_file()
w.close()
print(json.dumps({"out": out, "context_length": cfg["max_position_embeddings"], "rms_eps": cfg["rms_norm_eps"], "tensors_mapped": len(used),
                  "source_tensors": len(keys), "real_tokens": n_real, "padded_vocab": V, "chat_template": bool(tcfg.get("chat_template"))}))
