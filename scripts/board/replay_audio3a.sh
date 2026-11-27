#!/bin/sh
# Offline replay only: no ALSA, GPIO, killall, or source recording modification.
set -eu
if [ "$#" -lt 3 ]; then
    echo "Usage: replay_audio3a.sh BUNDLE INPUT.wav OUTPUT.wav [--pace] [--denoise-after-nvidia] [--mic-pregain GAIN] [--dump-stages DIR]" >&2
    exit 2
fi
bundle=$1
input=$2
output=$3
shift 3
export LD_LIBRARY_PATH="$bundle/lib:/userdata/opt/hdmeeting/lib"
export SB_EXPECT_FRAMES=32
# Select this image's high-capacity cores; permit explicit deployment overrides.
if [ -z "${AUDIO3A_CPU_AFFINITY:-}" ]; then
    max=0
    cores=""
    for path in /sys/devices/system/cpu/cpu[0-9]*/cpu_capacity; do
        [ -f "$path" ] || continue
        value=$(cat "$path")
        cpu=${path%/cpu_capacity}
        cpu=${cpu##*/cpu}
        if [ "$value" -gt "$max" ]; then max=$value; cores=$cpu
        elif [ "$value" -eq "$max" ]; then cores="$cores,$cpu"; fi
    done
    AUDIO3A_CPU_AFFINITY=${cores:-0}
fi
# Select latency/quality explicitly; the model filename does not select geometry.
case "${AUDIO3A_NVIDIA_PROFILE:-quality}" in
    quality) NVIDIA_HISTORY_MS=48 ;;
    balanced) NVIDIA_HISTORY_MS=64 ;;
    *) echo "AUDIO3A_NVIDIA_PROFILE must be quality or balanced" >&2; exit 2 ;;
esac

binary="${AUDIO3A_BINARY:-$bundle/bin/p2p_dut_node}"
dpdf_model="${AUDIO3A_DPDF_MODEL:-$bundle/models/dpdfnet2.onnx}"
exec taskset -c "$AUDIO3A_CPU_AFFINITY" "$binary" \
    --mic-channels 10 --ref-channel 8 \
    --wpe --wpe-taps 8 --wpe-delay 3 --no-dereverb \
    --ans dpdfnet --dpdfnet-model "$dpdf_model" \
    --dpdfnet-threads 1 --nvidia \
    --nvidia-rknn "$bundle/models/nvidia-t32-spatial-norm.rknn" \
    --nvidia-bin "$bundle/bin/nvidia_sb_pcm" \
    --nvidia-core-mask 7 --nvidia-dry-gain 0 --nvidia-frames 32 \
    --nvidia-history-ms "$NVIDIA_HISTORY_MS" --nvidia-overlap-ms 48 \
    --offline-wav "$input" "$output" "$@"
