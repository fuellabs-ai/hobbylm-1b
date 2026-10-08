# Patched runtimes: patches, launchers and build steps

HobbyLM-1B's GGUF files need a patched llama.cpp (or the patched Ollama runtime). Stock Ollama v0.35.1 was tested and fails to load them (`attn_qkv.weight` shape mismatch); stock llama.cpp was not run.

**Prebuilt Windows x64 CPU packages** and the **complete corresponding source archive** (upstream source tarballs, the GCC runtime source and these patches) are on the [v1.0.0 release page](https://github.com/fuellabs-ai/hobbylm-1b/releases/tag/v1.0.0). This folder holds the HobbyLM-specific parts as plain files.

These are unofficial builds: not produced, reviewed or endorsed by the llama.cpp/ggml or Ollama projects. Tested only on Windows 11 x86-64, CPU backend, one machine.

**Why do the files say `bailingmoe2`?** `bailingmoe2` is the name of an architecture already built into llama.cpp, added for inclusionAI's Ling 2.0 models (`BailingMoeV2`). HobbyLM is not based on, derived from or affiliated with those models: it was designed and trained from scratch (architecture by Harish; trained by Fuel Labs). The GGUF files reuse this llama.cpp code path only because its layout matches HobbyLM's: a sigmoid-routed mixture of experts with expert-bias selection, a shared expert and a leading dense layer, plus grouped-query attention with QK-norm. The one difference, HobbyLM's 2,048-wide attention against a 1,024 hidden size, is what `bailingmoe2_decoupled_head_dim.patch` fixes.

| Path | Contents |
|---|---|
| [`llama.cpp/`](llama.cpp/) | For llama.cpp `63bef2728d5714a5c2fe1524ad39d81b693890f3`: the QKV-shape patch (`bailingmoe2_decoupled_head_dim.patch`), the Windows exit fix (`004-win-exit-hang-stream-sessions-static.patch`), [`BUILD.md`](llama.cpp/BUILD.md), the build script and toolchain record, and the launchers `start-chat-server.bat` / `run-base-completion.bat`. |
| [`ollama/`](ollama/) | For Ollama v0.35.1 (its llama.cpp `b11232`): patches `001`–`004` applied in order ([`BUILD.md`](ollama/BUILD.md)); `003` and `004` are HobbyLM's, `001`–`002` are Ollama's own. Also the build script, the MSYS2 package list, the launchers and the Modelfile. |
| [`chat-bundle/`](chat-bundle/) | The launcher scripts and `READ ME FIRST.txt` of the double-click Windows chat bundle. |
| [`SOURCE_ARCHIVE.md`](SOURCE_ARCHIVE.md) | What the release's source archive contains, mapped to each shipped binary. |

All text files here are byte-identical to the files inside the v1.0.0 packages and source archive. Build scripts write local work-directory paths as `$WORK`.

**Known outdated reference:** `chat-bundle/READ ME FIRST.txt` (as shipped in 1.0.0) points to `huggingface.co/harims95/hobbylm-1b-broad-sft-3450-hf` for training-data licence questions; that repository is no longer public. The same information is in the `NOTICE` file and the "Training-data licence notes" section of the [base model card](https://huggingface.co/harims95/hobbylm-1B).

## Licence

HobbyLM's own lines in these patches, the launchers, build scripts and package docs are MIT ([`LICENSE-HobbyLM-runtime-MIT.txt`](LICENSE-HobbyLM-runtime-MIT.txt), which states its exact scope). Unchanged upstream lines in the patches remain llama.cpp code under llama.cpp's MIT licence. Ollama's patches 001–002 remain Ollama's (MIT).
