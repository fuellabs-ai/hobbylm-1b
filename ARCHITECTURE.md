# HobbyLM-1B architecture

Values are read from the released `config.json` files. The code is in [`hobbylm_hf/modeling_hobbylm.py`](hobbylm_hf/modeling_hobbylm.py).

| | Base | Instruct |
|---|---|---|
| Parameters | 1,037,121,536 total; 304,953,344 active per token | same |
| Context (`max_position_embeddings`) | 1,024 | 4,096 |
| Weights dtype | float32 | float32 |

The two released models differ only in context length; all other configuration values are identical.

## Layers

- **Depth:** 20 decoder layers, pre-norm residual stream, RMSNorm (eps 1e-6).
- **Width:** hidden size 1,024; vocabulary 50,304 (GPT-2 BPE padded with unused rows); tied input and output embeddings.
- **Layer 0:** dense SwiGLU MLP, intermediate size 2,816.
- **Layers 1–19:** sparse mixture-of-experts:
  - 64 routed SwiGLU experts (intermediate size 224), **top-8** per token;
  - 1 shared expert, applied to every token;
  - **sigmoid router**; a per-expert bias, trained without an auxiliary loss ("aux-free" balancing), shifts only *which* experts are selected; the selected experts are weighted by their raw sigmoid scores (no top-k renormalisation, scaling factor 1.0).

## Attention

- Grouped-query attention: 16 query heads, 8 key/value heads, head dimension 128.
- The query width (16 × 128 = 2,048) is **larger than the hidden size** (1,024). Runtimes that assume they are equal need a patch (see [`runtimes/`](runtimes/)).
- Per-head RMSNorm on queries and keys (QK-norm) before RoPE.
- Plain RoPE, theta 10,000, full rotary dimension, rotate-half layout; no RoPE scaling at either context length.
- No attention bias, no logit soft-capping.

## Precision

Keep the model in float32. With top-8-of-64 routing, the router-score gap between the 8th and 9th expert is often smaller than bf16 resolution: converting the router to bf16 drops router top-1 agreement to 62% and full-model output agreement from 97.6% to 61% (measured on the release checkpoints). The GGUF conversions keep the router and expert-bias tensors in F32; see [`gguf/README.md`](gguf/README.md).

## Tokenizer and chat template

- GPT-2 byte-level BPE; no BOS/EOS is added automatically; `<|endoftext|>` (50256) ends generation.
- The Instruct model's chat template is in its `tokenizer_config.json` on Hugging Face; [`examples/chat_sft.py`](examples/chat_sft.py) applies it. The base model has no chat template.
