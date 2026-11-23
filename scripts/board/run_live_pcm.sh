#!/bin/sh
# Read 16 kHz mono S16_LE PCM from stdin and emit 16 kHz mono S16_LE PCM.
set -eu
set -o pipefail

usage() {
    printf 'Usage: %s [--pace | --denoised | --verify-models] < input.pcm > output.pcm\n' "$0" >&2
    exit 2
}

[ "$#" -le 1 ] || usage
root="$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)"
bin="$root/bin"
models="$root/models"
profile=t16
if [ -f "$models/profile" ]; then
    IFS= read -r profile < "$models/profile"
fi
case "$profile" in
    t16) frames=16; overlap_ms=8; history_ms=32; default_cores=3 ;;
    t32) frames=32; overlap_ms=72; history_ms=48; default_cores=7 ;;
    *)
        printf 'Unknown model profile: %s\n' "$profile" >&2
        exit 2
        ;;
esac
nvidia_model="$models/nvidia-$profile-spatial-norm.rknn"

case "${1-}" in
    --verify-models)
        (cd "$models" && sha256sum -c SHA256SUMS)
        exit 0
        ;;
    --pace | --denoised | "") ;;
    *) usage ;;
esac

export SB_RKNN_CORE_MASK="${SB_RKNN_CORE_MASK:-$default_cores}"
export SB_EXPECT_FRAMES="$frames"
export SB_DRY_GAIN=0.025

if [ "${1-}" = --denoised ]; then
    exec "$bin/nvidia_sb_pcm" "$nvidia_model" "$overlap_ms" "$history_ms"
fi

if [ "${1-}" = --pace ]; then
    # Give the NPU model time to warm up before measuring steady-state latency.
    (sleep 1; exec "$bin/ulunas_pcm" "$models/ulunas_stream_simple.onnx" --pace) \
        | "$bin/nvidia_sb_pcm" "$nvidia_model" "$overlap_ms" "$history_ms" \
        | "$bin/v4_live_pcm" --timing
else
    "$bin/ulunas_pcm" "$models/ulunas_stream_simple.onnx" \
        | "$bin/nvidia_sb_pcm" "$nvidia_model" "$overlap_ms" "$history_ms" \
        | "$bin/v4_live_pcm"
fi
