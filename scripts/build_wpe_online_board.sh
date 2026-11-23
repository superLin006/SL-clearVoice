#!/bin/bash
set -euo pipefail

if [ "$#" -gt 1 ]; then
  printf 'Usage: %s [OUTPUT_BINARY]\n' "$0" >&2
  exit 2
fi

# Cross-compile the 6ch online WPE + DS for RK3588 (aarch64)
root="$(cd "$(dirname "$0")/.." && pwd)"
output="${1:-wpe_online_cpp}"
compiler="${AARCH64_CXX:-aarch64-linux-gnu-g++}"
mkdir -p -- "$(dirname -- "$output")"
"$compiler" -O3 -ffast-math -fno-finite-math-only -fopenmp -std=c++17 -static \
  -o "$output" "$root/src/spatial/wpe_online_cpp.cpp"
printf 'built %s\n' "$output"
