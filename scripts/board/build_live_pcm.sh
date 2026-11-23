#!/usr/bin/env bash
# Build the RK3588 mono PCM chain for the bundled T16 or a supplied T32 model.
set -euo pipefail

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
    printf 'Usage: %s OUTPUT_DIRECTORY [T32_RKNN]\n' "$0" >&2
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
profile=t16
nvidia_name=nvidia-t16-spatial-norm.rknn
nvidia_model="$models/$nvidia_name"
if [ "$#" -eq 2 ]; then
    profile=t32
    nvidia_name=nvidia-t32-spatial-norm.rknn
    nvidia_model="$2"
    if [ ! -f "$nvidia_model" ] || [ ! -s "$nvidia_model" ]; then
        printf 'Missing or empty T32 model: %s\n' "$nvidia_model" >&2
        exit 2
    fi
fi
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
install -m 644 "$models/ulunas_stream_simple.onnx" "$bundle/models/"
if [ ! "$nvidia_model" -ef "$bundle/models/$nvidia_name" ]; then
    install -m 644 "$nvidia_model" "$bundle/models/$nvidia_name"
fi
(cd "$bundle/models" && sha256sum ulunas_stream_simple.onnx "$nvidia_name") > "$bundle/models/SHA256SUMS"
printf '%s\n' "$profile" > "$bundle/models/profile"
