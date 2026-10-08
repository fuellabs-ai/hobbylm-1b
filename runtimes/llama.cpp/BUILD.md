# Building this package

## Sources

| Component | Revision / checksum |
|---|---|
| llama.cpp | commit `63bef2728d5714a5c2fe1524ad39d81b693890f3` |
| `patches/bailingmoe2_decoupled_head_dim.patch` | sha256 `bb8f540024376677352f3f8f32b581ee0bfcc3801230a0676fe6e61db21af809` |
| `patches/004-win-exit-hang-stream-sessions-static.patch` | sha256 `02ccda21861630e2894718178fadcf57afd662088a66e2de74889b26475b5a29` (written against llama.cpp b11232; applies cleanly to 63bef272) |
| Web UI source | llama.cpp `tools/ui` at the same commit; `package-lock.json` sha256 `f019baf51eecfc5e7104df8ca04cd1057be1d93c6e45f1daeec41474961617c9`, `package.json` sha256 `699e0dfd423701a15ee1c25e9bfcf6aaebff6f68200a6a4a07e7df9347204509` |
| Web UI output (embedded) | 70 files. sha256 of the sorted per-file sha256 list (`sha256sum` over `find dist -type f \| sort`): `a26feec3f7b1ad49c388ab6c8ddf9807bead625e47b37b42393fe198e28f4cb2`. The page's `build.json` reads `{"version":"1"}` (the local build number). |

**Why the UI is built from source.** llama.cpp normally downloads a prebuilt UI from the Hugging Face bucket `ggml-org/llama-ui`, keyed by release build number. Commit 63bef272 has no release tag, so no matching prebuilt archive exists, and the default build fell back to the moving `latest` archive. This package instead uses llama.cpp's supported source build (`LLAMA_BUILD_UI=ON`) with downloads disabled (`LLAMA_USE_PREBUILT_UI=OFF`). The UI therefore corresponds exactly to the pinned commit and its lockfile.

## Toolchain (as used)

- **Shell and compiler:** MSYS2 UCRT64 login shell. `mingw-w64-ucrt-x86_64-gcc` 16.2.0-3 (GCC 16.2.0 Rev3), `cmake` 4.4.3-2, `ninja` 1.13.2-1.
- **UI build tools:** Node.js v24.14.0 and npm 11.9.0 (Windows installer), added to PATH. `npm ci` downloads the locked packages from the npm registry at **build time** only.
- **Runtime DLL packages:** `mingw-w64-ucrt-x86_64-gcc-libs` 16.2.0-3 and `mingw-w64-ucrt-x86_64-libwinpthread` 14.0.0.r375.g9c1abbbf5-1.

## Commands

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
git checkout 63bef2728d5714a5c2fe1524ad39d81b693890f3
git apply /path/to/patches/bailingmoe2_decoupled_head_dim.patch
git apply /path/to/patches/004-win-exit-hang-stream-sessions-static.patch
export PATH="$PATH:/c/Program Files/nodejs"
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DGGML_NATIVE=OFF -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON -DBUILD_SHARED_LIBS=ON \
  -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_UI=ON -DLLAMA_USE_PREBUILT_UI=OFF
cmake --build build -j       # about 6.5 min on an 8-core laptop (includes npm ci + UI build)
```

The configure/build log shows `UI: running npm ci`, `UI: running npm run build`, `UI: npm build succeeded` and `UI: embedded 70 assets`, with no bucket download.

## Assembly

`bin/` holds exactly the import closure of `llama-server.exe` and `llama-completion.exe`, plus all 14 `ggml-cpu-*.dll` backends (loaded at run time, not imported). The closure was computed with `objdump -p`.
- 23 files come from `build/bin`.
- 4 come from `C:\msys64\ucrt64\bin`: `libstdc++-6.dll`, `libgcc_s_seh-1.dll`, `libgomp-1.dll`, `libwinpthread-1.dll`. They are byte-identical to the installed MSYS2 packages above.
- Every other import is a Windows system DLL.

`start-chat-server.bat` and `run-base-completion.bat` are plain batch launchers written for this package (CRLF line endings).

**Web UI notices:** the list of npm packages in `licenses/web-ui/` comes from the source maps of an otherwise identical analysis build. That build used the same lockfile, with `vite build --sourcemap`. The list contains every `node_modules` package referenced by the production bundle.

**Reproducibility:** the build is not byte-reproducible, because absolute build paths are embedded. `MANIFEST_SHA256.txt` describes this build.
