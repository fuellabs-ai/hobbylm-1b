# GGUF converter for HobbyLM-1B

`to_gguf_1b.py` converts a released HobbyLM-1B Hugging Face model folder into an **F32 GGUF** in llama.cpp's `bailingmoe2` architecture layout. It produced the F32 files published in [harims95/hobbylm-1B-gguf](https://huggingface.co/harims95/hobbylm-1B-gguf). The Q8_0 and Q4_K_M files were then made with llama.cpp's `llama-quantize`.

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
huggingface-cli download harims95/hobbylm-1B-instruct --revision 0506eed260c353a259712705fa2eb11662f1f0ac --local-dir ckpt/sft
# the folder name "sft" is written into the file (general.name "HobbyLM-1B (sft)")
python to_gguf_1b.py ckpt/sft hobbylm-1b-sft-3450-F32.gguf
sha256sum hobbylm-1b-sft-3450-F32.gguf      # compare with SHA256SUMS.txt in harims95/hobbylm-1B-gguf
```

- **Revisions:** `0506eed2…` is the `hobbylm-1B-instruct` revision of the converter's *input* (safetensors, config, tokenizer). The published GGUF files are in [`harims95/hobbylm-1B-gguf`](https://huggingface.co/harims95/hobbylm-1B-gguf) at `f8bc034403026b2706f5275702d3801753b87c34`.
- **Same input files:** the `config.json`, `model.safetensors` and tokenizer files of `hobbylm-1B-instruct` are byte-identical to the files the bit-for-bit reproduction used (the release revision of the former SFT repository plus the release tokenizer).
- **Folder name:** the converter stores the input folder's name in the GGUF metadata (`general.name`). The published files were converted from folders named `sft` and `base`. Any other name changes the file's checksum, though not the model.
- **Base model:** download `harims95/hobbylm-1b-checkpoints` at `3aded72dc807683cb9b85c20cfffceaf1f72406c` with `--include "base-anneal-final/*" --local-dir ckpt`, rename `ckpt/base-anneal-final` to `ckpt/base`, and convert it to `hobbylm-1b-base-F32.gguf` (context 1024, no chat template).
- **Quantised files:** `llama-quantize <F32.gguf> <out.gguf> Q8_0` (or `Q4_K_M`). The router and expert-bias tensors stay F32.

## Running the GGUF

The `bailingmoe2` code in llama.cpp assumes a query width equal to the hidden size. HobbyLM's is 2048 against a hidden size of 1024. Run the files with the patched runtimes from the [v1.0.0 release](https://github.com/fuellabs-ai/hobbylm-1b/releases/tag/v1.0.0), or apply [`bailingmoe2_decoupled_head_dim.patch`](../runtimes/llama.cpp/patches/bailingmoe2_decoupled_head_dim.patch) to llama.cpp `63bef272` and build (see [`runtimes/`](../runtimes/)). **Stock Ollama v0.35.1 does not load these files (tested).**

**Why do the files say `bailingmoe2`?** `bailingmoe2` is the name of an architecture already built into llama.cpp, added for inclusionAI's Ling 2.0 models (`BailingMoeV2`). HobbyLM is not based on, derived from or affiliated with those models: it was designed and trained from scratch (architecture by Harish; trained by Fuel Labs). The GGUF files reuse this llama.cpp code path only because its layout matches HobbyLM's: a sigmoid-routed mixture of experts with expert-bias selection, a shared expert and a leading dense layer, plus grouped-query attention with QK-norm. The one difference, HobbyLM's 2,048-wide attention against a 1,024 hidden size, is what `bailingmoe2_decoupled_head_dim.patch` fixes.

## What the converter asserts

The converter reads every hyper-parameter from `config.json`. It refuses configurations it does not handle:
- non-sigmoid gating, top-k renormalisation, routed scaling other than 1.0;
- RoPE scaling or partial rotary;
- logit soft-capping, multi-token-prediction layers.

## Licence and provenance

- **Licence:** Apache License 2.0 (this repository's `LICENSE`).
- **Provenance:** written for the HobbyLM release. It contains no code copied from llama.cpp or other projects; it only *imports* the `gguf` package (llama.cpp's gguf-py, MIT), `numpy` and `safetensors`, which keep their own licences.
