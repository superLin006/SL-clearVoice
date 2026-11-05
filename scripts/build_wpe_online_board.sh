#!/bin/bash
set -euo pipefail
# Cross-compile the 6ch online WPE + DS for RK3588 (aarch64)
aarch64-linux-gnu-g++ -O3 -ffast-math -fopenmp -std=c++17 -static \
  -o wpe_online_cpp src/spatial/wpe_online_cpp.cpp
echo "built wpe_online_cpp"
# usage: OMP_NUM_THREADS=4 ./wpe_online_cpp in.f32 out.f32 6
