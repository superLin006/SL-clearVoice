#!/usr/bin/env bash
# Build the fixed RK3588 mono PCM chain and copy its exact model assets.
set -euo pipefail

if [ "$#" -ne 1 ]; then
    printf 'Usage: %s OUTPUT_DIRECTORY\n' "$0" >&2
    exit 2
fi

: "${RKNN_INCLUDE:?Set the directory containing rknn_api.h}"
: "${RKNN_LIBDIR:?Set the directory containing AArch64 librknnrt.so}"
: "${ORT_INCLUDE:?Set the directory containing onnxruntime_c_api.h}"
: "${ORT_LIBDIR:?Set the directory containing AArch64 libonnxruntime.so}"
: "${FFTW_INCLUDE:?Set the directory containing fftw3.h}"
: "${FFTW_STATIC:?Set the path to AArch64 libfftw3f.a}"

board_dir="$(cd "$(dirname "$0")" && pwd)"
models="$(cd "$board_dir/../../models/rknn/live" && pwd)"
cc="${AARCH64_CC:-aarch64-linux-gnu-gcc}"
bundle="$1"
bin="$bundle/bin"
mkdir -p "$bin" "$bundle/models"
(cd "$models" && sha256sum -c SHA256SUMS)

flags=(-O3 -std=c11 -Wall -Wextra -Werror)
"$cc" "${flags[@]}" -no-pie -I"$ORT_INCLUDE" -I"$FFTW_INCLUDE" \
    "$board_dir/ulunas_pcm.c" "$FFTW_STATIC" -L"$ORT_LIBDIR" -lonnxruntime -lm \
    -o "$bin/ulunas_pcm"
"$cc" "${flags[@]}" -no-pie -I"$RKNN_INCLUDE" -I"$FFTW_INCLUDE" \
    "$board_dir/nvidia_sb_pcm.c" "$FFTW_STATIC" -L"$RKNN_LIBDIR" -lrknnrt -lm \
    -o "$bin/nvidia_sb_pcm"
"$cc" "${flags[@]}" -static "$board_dir/v4_live_pcm.c" -lm -o "$bin/v4_live_pcm"
install -m 755 "$board_dir/run_live_pcm.sh" "$bundle/run.sh"
install -m 644 "$models/ulunas_stream_simple.onnx" "$models/nvidia-t16-spatial-norm.rknn" \
    "$models/SHA256SUMS" "$bundle/models/"
