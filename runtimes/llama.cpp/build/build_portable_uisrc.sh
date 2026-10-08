#!/usr/bin/bash
# Portable build + web UI built from source at the same commit (no prebuilt download). MSYS2 UCRT64 login shell.
set -u
S=$WORK/src/llama.cpp     # 63bef272 + bailingmoe2 patch + 004
P=$WORK
export PATH="$PATH:/c/Program Files/nodejs:/c/Program Files/Git/cmd"
{ which cmake gcc ninja npm node; gcc --version | head -1; cmake --version | head -1; node --version; npm --version; git -C $S rev-parse HEAD; git -C $S diff --stat; } > $P/build_uisrc_toolchain.txt 2>&1
date > $P/build_uisrc_times.txt
cmake -S $S -B $P/build-uisrc -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DGGML_NATIVE=OFF -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON -DBUILD_SHARED_LIBS=ON \
  -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_UI=ON -DLLAMA_USE_PREBUILT_UI=OFF > $P/build_uisrc_configure.log 2>&1
echo "configure_exit=$?" >> $P/build_uisrc_times.txt; date >> $P/build_uisrc_times.txt
cmake --build $P/build-uisrc -j > $P/build_uisrc_compile.log 2>&1
echo "build_exit=$?" >> $P/build_uisrc_times.txt; date >> $P/build_uisrc_times.txt
