#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
    printf 'Usage: %s MODEL_DIRECTORY OUTPUT_BUNDLE\n' "$0" >&2
    exit 2
fi
: "${RKNN_INCLUDE:?Set the directory containing rknn_api.h}"
: "${RKNN_LIBDIR:?Set the AArch64 RKNN Runtime library directory}"
: "${ORT_INCLUDE:?Set the directory containing onnxruntime_c_api.h}"
: "${ORT_LIBDIR:?Set the AArch64 ONNX Runtime library directory}"
: "${FFTW_INCLUDE:?Set the directory containing fftw3.h}"
: "${FFTW_STATIC:?Set the AArch64 single-precision FFTW archive}"

source_dir="$(cd "$(dirname "$0")" && pwd)"
models="$(cd "$1" && pwd)"
output="$2"
compiler="${AARCH64_CC:-aarch64-linux-gnu-gcc}"
mkdir -p "$output/bin" "$output/models"
(cd "$models" && sha256sum -c SHA256SUMS)
"$compiler" -O3 -march=armv8.2-a+fp16 -mtune=cortex-a72 \
    -std=c11 -Wall -Wextra -Werror -no-pie \
    -I"$RKNN_INCLUDE" -I"$ORT_INCLUDE" -I"$FFTW_INCLUDE" \
    "$source_dir/dpdfnet_pcm.c" "$FFTW_STATIC" \
    -L"$RKNN_LIBDIR" -L"$ORT_LIBDIR" -lrknnrt -lonnxruntime -lm \
    -o "$output/bin/dpdfnet_pcm"
for name in dpdfnet2-cpu.onnx dpdfnet2-core.rknn initial-state.bin erb-filterbank.bin manifest.json; do
    install -m 644 "$models/$name" "$output/models/$name"
done
if [ -f "$models/dpdfnet2-cpu-int8.onnx" ]; then
    install -m 644 "$models/dpdfnet2-cpu-int8.onnx" "$output/models/dpdfnet2-cpu-int8.onnx"
fi
if [ -n "${DPDFNET_RKNN_RUNTIME:-}" ]; then
    mkdir -p "$output/lib"
    install -m 644 "$DPDFNET_RKNN_RUNTIME" "$output/lib/librknnrt.so"
fi
awk '$2=="dpdfnet2-cpu.onnx" || $2=="dpdfnet2-core.rknn" || $2=="dpdfnet2-cpu-int8.onnx" || $2=="initial-state.bin" || $2=="erb-filterbank.bin"' \
    "$models/SHA256SUMS" > "$output/models/SHA256SUMS"
install -m 644 "$models/LICENSE" "$output/models/LICENSE"
install -m 755 "$source_dir/run.sh" "$output/run.sh"
cat > "$output/README.txt" <<'README'
DPDFNet2, RK3588, 2026-11-27
Input/output: raw mono 16000 Hz signed 16-bit little-endian PCM.
Model: original DPDFNet2 weights, FP16 NPU with CPU FP32 normalization.
Requires the board libraries in /userdata/opt/hdmeeting/lib.
./run.sh --verify-models
./run.sh < input.pcm > output.pcm 2> timing.log
./run.sh --pace < input.pcm > output.pcm 2> timing.log
./run.sh --cpu < input.pcm > output-cpu.pcm 2> timing-cpu.log
./run.sh --cpu-int8 < input.pcm > output-int8.pcm 2> timing-int8.log
The INT8 option quantizes five input projections; other operations remain FP32.
Use --pace for file-based real-time measurement only.
Do not send multichannel interleaved PCM directly.
The stream keeps recurrent state. EOF flushes delayed samples.
Compute time and physical speech latency are different.
See scripts/dpdfnet/README.md in the source repository for measurements
and export/build instructions. Models include the upstream LICENSE.
README
