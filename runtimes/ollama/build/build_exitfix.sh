#!/usr/bin/bash
# Isolated build of the patched Ollama v0.35.1 runtime (compat 001-004). Run from an MSYS2 UCRT64 login shell.
set -u
cd $WORK
export PATH="$PATH:/c/Program Files/Git/cmd"     # apply-git-patches.cmake needs git
L=$WORK
{ echo "PATH=$PATH"; which cmake gcc ninja git; gcc --version | head -1; cmake --version | head -1; git --version; } > $L/build_toolchain.txt 2>&1
date > $L/build_times.txt
cmake -S llama/server --preset cpu_windows -G Ninja > $L/build_configure.log 2>&1; echo "configure_exit=$?" >> $L/build_times.txt; date >> $L/build_times.txt
cmake --build build/llama-server-cpu -j > $L/build_compile.log 2>&1; echo "build_exit=$?" >> $L/build_times.txt; date >> $L/build_times.txt
