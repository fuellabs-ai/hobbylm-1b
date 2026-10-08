# GGUF converter for HobbyLM-1B

`to_gguf_1b.py` converts a released HobbyLM-1B Hugging Face model folder into an **F32 GGUF** in llama.cpp's `bailingmoe2` architecture layout. It produced the F32 files published in [`gguf/` in harims95/hobbylm-1b-hf](https://huggingface.co/harims95/hobbylm-1b-hf/tree/main/gguf). The Q8_0 and Q4_K_M files were then made with llama.cpp's `llama-quantize`.

**Reproducibility check (2026-10-07):** in a clean environment, it regenerated the published `hobbylm-1b-sft-3450-F32.gguf` **bit for bit** (SHA-256 `649220cef142151e6b38140a4a77123795732bb57421474019ebe4811264ac33`).

## Requirements

- Python 3.11 (tested; 3.11.16).
- `gguf` Python package from llama.cpp's `gguf-py` at commit `63bef2728d5714a5c2fe1524ad39d81b693890f3`. Use that checkout, not a different `gguf` release.
- `safetensors` (tested 0.8.0) and `numpy` (tested 2.4.6; `gguf` requires numpy ≥ 2.2.6).

```bash
git clone https://github.com/ggml-org/llama.cpp
git -C llama.cpp checkout 63bef2728d5714a5c2fe1524ad39d81b693890f3
python -m venv conv && conv/bin/pip install ./llama.cpp/gguf-py safetensors   # Windows: conv\Scripts\pip
```

## Usage

The input folder must hold the released model files **and** the tokenizer files:
- `config.json` and `model.safetensors` from the model repository at its release revision;
- `tokenizer.json` and `tokenizer_config.json`.

```bash
huggingface-cli download harims95/hobbylm-1b-checkpoints-hf --revision bdc371b4aa48bede3f55ba1036e9f27952f48c17 --include "sft-step3450/*" --local-dir ckpt
mv ckpt/sft-step3450 ckpt/sft             # the folder name is written into the file (general.name "HobbyLM-1B (sft)")
python to_gguf_1b.py ckpt/sft hobbylm-1b-sft-3450-F32.gguf
sha256sum hobbylm-1b-sft-3450-F32.gguf      # compare with gguf/SHA256SUMS.txt in harims95/hobbylm-1b-hf at dc11aab4e06fd2000313c8821557a86e8368bf58
```

- **Revisions:** `bdc371b4…` is the checkpoint-repo revision of the converter's *input* (safetensors, config, tokenizer). The published GGUF files are in `harims95/hobbylm-1b-hf/gguf/` at `dc11aab4e06fd2000313c8821557a86e8368bf58`, the hub commit that added them.
- **Same input files:** the `config.json`, `model.safetensors` and tokenizer files in `sft-step3450/` are byte-identical to the files the bit-for-bit reproduction used (the release revision of the former SFT repository plus the release tokenizer).
- **Folder name:** the converter stores the input folder's name in the GGUF metadata (`general.name`). The published files were converted from folders named `sft` and `base`. Any other name changes the file's checksum, though not the model.
- **Base model:** the same steps with `--include "base-anneal-final/*"`, renaming `ckpt/base-anneal-final` to `ckpt/base`, give `hobbylm-1b-base-F32.gguf` (context 1024, no chat template). Alternatively, use the base files at the root of `harims95/hobbylm-1b-hf`.
- **Quantised files:** `llama-quantize <F32.gguf> <out.gguf> Q8_0` (or `Q4_K_M`). The router and expert-bias tensors stay F32.

## Running the GGUF

The `bailingmoe2` code in llama.cpp assumes a query width equal to the hidden size. HobbyLM's is 2048 against a hidden size of 1024. Run the files with the patched runtimes from the [v1.0.0 release](https://github.com/fuellabs-ai/hobbylm-1b/releases/tag/v1.0.0), or apply [`bailingmoe2_decoupled_head_dim.patch`](../runtimes/llama.cpp/patches/bailingmoe2_decoupled_head_dim.patch) to llama.cpp `63bef272` and build (see [`runtimes/`](../runtimes/)). **Stock Ollama v0.35.1 does not load these files (tested).**

## What the converter asserts

The converter reads every hyper-parameter from `config.json`. It refuses configurations it does not handle:
- non-sigmoid gating, top-k renormalisation, routed scaling other than 1.0;
- RoPE scaling or partial rotary;
- logit soft-capping, multi-token-prediction layers.

## Licence and provenance

- **Licence:** Apache License 2.0 (this repository's `LICENSE`).
- **Provenance:** written for the HobbyLM release. It contains no code copied from llama.cpp or other projects; it only *imports* the `gguf` package (llama.cpp's gguf-py, MIT), `numpy` and `safetensors`, which keep their own licences.
