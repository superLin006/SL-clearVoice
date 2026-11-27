#!/bin/sh
# Mono 16 kHz signed little-endian PCM in and out.
set -eu
root="$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)"
backend=rknn
pacing=--unpaced
model="$root/models/dpdfnet2-core.rknn"
case "${1-}" in
    --verify-models) cd "$root/models"; exec sha256sum -c SHA256SUMS ;;
    --cpu) backend=ort; model="$root/models/dpdfnet2-cpu.onnx"; shift ;;
    --cpu-int8) backend=ort; model="$root/models/dpdfnet2-cpu-int8.onnx"; shift ;;
esac
if [ "${1-}" = --pace ]; then pacing=--pace; shift; fi
if [ "$#" -ne 0 ]; then
    printf 'Usage: %s [--cpu|--cpu-int8] [--pace] | --verify-models\n' "$0" >&2
    exit 2
fi
if [ ! -f "$model" ]; then printf 'Missing model: %s\n' "$model" >&2; exit 1; fi
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}/userdata/opt/hdmeeting/lib"
# A bundled runtime takes precedence for this process only.
if [ -f "$root/lib/librknnrt.so" ]; then
    export LD_LIBRARY_PATH="$root/lib:$LD_LIBRARY_PATH"
fi
set -- "$root/bin/dpdfnet_pcm" "$backend" "$model" "$root/models" \
    "${DPDFNET_NPU_CORES:-1}" "$pacing" "${DPDFNET_CPU_THREADS:-1}"
# CPU numbering differs between board images. Pinning can be selected per deployment.
if [ -n "${DPDFNET_CPU_AFFINITY:-}" ]; then
    exec taskset -c "$DPDFNET_CPU_AFFINITY" "$@"
fi
exec "$@"
