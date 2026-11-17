#!/bin/sh
# Read 16 kHz mono S16_LE PCM from stdin and emit 16 kHz mono S16_LE PCM.
set -eu
set -o pipefail

usage() {
    printf 'Usage: %s [--pace | --verify-models] < input.pcm > output.pcm\n' "$0" >&2
    exit 2
}

[ "$#" -le 1 ] || usage
root="$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)"
bin="$root/bin"
models="$root/models"

case "${1-}" in
    --verify-models)
        (cd "$models" && sha256sum -c SHA256SUMS)
        exit 0
        ;;
    --pace | "") ;;
    *) usage ;;
esac

# Mask 3 selects cores 0/1. Mask 7 enables the tested three-core mode.
export SB_RKNN_CORE_MASK="${SB_RKNN_CORE_MASK:-3}"
export SB_DRY_GAIN=0.025

if [ "${1-}" = --pace ]; then
    # Give the NPU model time to warm up before measuring steady-state latency.
    (sleep 1; exec "$bin/ulunas_pcm" "$models/ulunas_stream_simple.onnx" --pace) \
        | "$bin/nvidia_sb_pcm" "$models/nvidia-t16-spatial-norm.rknn" 8 32 \
        | "$bin/v4_live_pcm" --timing
else
    "$bin/ulunas_pcm" "$models/ulunas_stream_simple.onnx" \
        | "$bin/nvidia_sb_pcm" "$models/nvidia-t16-spatial-norm.rknn" 8 32 \
        | "$bin/v4_live_pcm"
fi
