# Source code for the HobbyLM Windows runtime packages (release 1.0.0)

This archive is offered **in the same place as the binaries** it corresponds to, the HobbyLM v1.0.0 release assets:
- `HobbyLM-1B-Chat-Windows-x64-1.0.0.zip`;
- `hobbylm-llama.cpp-63bef27-patched-windows-x64-cpu-1.0.0.zip`;
- `hobbylm-ollama-0.35.1-patched-windows-x64-cpu-1.0.0.zip`.

All three are unofficial HobbyLM builds, not produced, reviewed or endorsed by the llama.cpp/ggml or Ollama projects.

## Coverage: shipped component → what this archive contains

| Shipped component (package) | Licence | In this archive | Not in this archive, and why |
|---|---|---|---|
| `libgcc_s_seh-1.dll` `80940372…`, `libstdc++-6.dll` `69200d6d…`, `libgomp-1.dll` `c13023e6…` (all three packages; byte-identical to MSYS2 `mingw-w64-ucrt-x86_64-gcc-libs` 16.2.0-3) | GPL-3.0-or-later WITH GCC-exception-3.1 | `gcc-runtime/mingw-w64-gcc-16.2.0-3.src.tar.zst` (`eb3479a8…`): MSYS2's source package for that exact build, containing the `PKGBUILD` recipe, MSYS2 patches and upstream `gcc-16.2.0.tar.xz`. Plus MSYS2's signature `.sig`. | — |
| llama.cpp runtime: `llama-server.exe`, `llama-completion.exe`, `libllama*.dll`, `ggml*.dll`, `libmtmd.dll` (llama.cpp package; bundle `runtime\`) | MIT (ggml authors), with vendored MIT / BSD / public-domain parts | `llama.cpp-63bef27/llama.cpp-63bef2728d5714a5c2fe1524ad39d81b693890f3.tar.gz` (git archive, unmodified); `patches/` (both HobbyLM patches, which reproduce the built tree byte for byte, verified); `build/build_portable_uisrc.sh` (configure and build commands) and `toolchain_at_build.txt` | — |
| llama.cpp web UI, embedded in `libllama-server-impl.dll` | UI code MIT (llama.cpp); 173 npm package versions under MIT / ISC / BSD-3-Clause / Apache-2.0, and `dompurify` under MPL-2.0 OR Apache-2.0 | UI source (`tools/ui`, including `package-lock.json` sha256 `f019baf5…` with exact versions) inside the llama.cpp archive | the npm packages' own sources (fetched by `npm ci` from the lockfile). They are under permissive licences whose texts are in `licenses/web-ui/`. `dompurify` is used under its **Apache-2.0** option. |
| Ollama runtime `lib/ollama/`: `llama-server.exe`, `llama-quantize.exe`, `.dll` files (Ollama package) | MIT (llama.cpp b11232 + Ollama compat code) | `ollama-0.35.1-hobbylm/ollama-v0.35.1-b0c1ca4f.tar.gz` (Ollama at tag v0.35.1), `llama.cpp-b11232-6f767fe9.tar.gz` (unpatched), `patches/001…004` (applied in order), `build/build_exitfix.sh`, `msys2_installed_packages_at_build.txt` | — |
| `ollama.exe` (Ollama package) | MIT (Ollama) plus the embedded Go modules' licences | Ollama v0.35.1 source tarball (above) | the 52 Go module sources (fetched by `go mod download` from Ollama's `go.sum`). `ollama.exe` is the **unmodified official binary**, signed by Ollama Inc.; its Go-module licence texts are in `licenses/ollama/GO_LICENSE`. |
| `libwinpthread-1.dll` (all packages) | MIT and BSD-3-Clause-Clear | not included | MSYS2 package `mingw-w64-winpthreads` 14.0.0.r375.g9c1abbbf5-1. Notice-type licence; its `COPYING` is in `licenses/winpthreads/`. |
| mingw-w64 CRT startup code statically linked into the GCC-built binaries | ZPL-2.1 (parts public domain or BSD) | not included | MSYS2 package `mingw-w64-crt` 14.0.0.r375.g9c1abbbf5-1. Notice-type licence; texts are in `licenses/mingw-w64-crt/`. |
| Launchers, Modelfile and package docs (all packages) | MIT (HobbyLM; `LICENSE-HobbyLM-runtime-MIT.txt`) | they ship as plain-text source in the packages themselves | — |

## Rebuilding (outline)

1. Install MSYS2 UCRT64 with `mingw-w64-ucrt-x86_64-gcc` 16.2.0-3, `cmake` and `ninja`. The web UI also needs Node 24.14.0 and npm 11.9.0.
2. Unpack the archive, apply the patches with `git apply`, and run the build script. Local work-directory paths in the scripts are written as `$WORK`.

Exact binary reproduction is not guaranteed: no reproducible-build tooling was used.

## HobbyLM's own changes: MIT

The HobbyLM-written lines of the two llama.cpp patches, the launchers and the package documentation are under the MIT licence in `LICENSE-HobbyLM-runtime-MIT.txt`, which states its exact scope. The unchanged upstream lines in the patches remain llama.cpp code under llama.cpp's MIT licence. Ollama's patches 001–002 remain Ollama's.

## Remaining obligations of the distributor (not a compliance certificate)

- Keep this archive available **in the same place** as the three binary packages for as long as they are offered.
- Keep each package's `licenses/` folder and `THIRD_PARTY_NOTICES.md` with the binaries.
- Whether the GCC Runtime Library Exception alone would have made the GCC source unnecessary was not decided. This archive follows the conservative reading.
