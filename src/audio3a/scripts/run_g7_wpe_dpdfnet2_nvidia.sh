#!/bin/sh
set -e

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PACKAGE_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
BIN="${PACKAGE_DIR}/bin/p2p_dut_node"

if [ ! -x "${BIN}" ]; then
    chmod +x "${BIN}" 2>/dev/null || true
fi

# 1. 自动使能板载喇叭功放 (RK3588 功放 GPIO 134)
if [ -d /sys/class/gpio ]; then
    echo 134 > /sys/class/gpio/export 2>/dev/null || true
    echo out > /sys/class/gpio/gpio134/direction 2>/dev/null || true
    echo 1 > /sys/class/gpio/gpio134/value 2>/dev/null || true
fi
/opt/hdmeeting/script/innerSpeakerCtrl.sh on 2>/dev/null || true

# 2. 配置声卡Card 3为 10CH 模式 (8麦+2软回采)
amixer -c 3 cset name='Software Digital Loopback Mode' 14 >/dev/null 2>&1 || true
amixer -c 3 cset name='Speaker Playback Switch' 1 >/dev/null 2>&1 || true

# 3. 清理残留进程
killall -9 p2p_dut_node nvidia_sb_pcm nvidia_sb_ode_pcm arecord aplay 2>/dev/null || true
sleep 0.2

export LD_LIBRARY_PATH="${PACKAGE_DIR}/lib:${LD_LIBRARY_PATH:-/usr/lib}"

echo "=================================================================="
echo "    [RK3588] 启动 8麦 + 2软回采 旗舰 3A (WPE + DPDFNet2 + NVIDIA T32 ODE)"
echo "    麦克风设备: hw:3,0 (10通道: Ch0~7为8麦, Ch8为喇叭回采)"
echo "    扬声器设备: hw:3,0"
echo "    消混响配置: 80Hz HPF + 8-Mic WPE + DPDFNet2 + 3-Core NPU NVIDIA T32 ODE 级联"
echo "    自动录音保存: /tmp/dut_raw10ch.wav (原声) & /tmp/dut_clean.wav (3A后)"
echo "=================================================================="

# Select latency/quality explicitly; the model filename does not select geometry.
case "${AUDIO3A_NVIDIA_PROFILE:-quality}" in
    quality) NVIDIA_HISTORY_MS=48 ;;
    balanced) NVIDIA_HISTORY_MS=64 ;;
    *) echo "AUDIO3A_NVIDIA_PROFILE must be quality or balanced" >&2; exit 2 ;;
esac

NV_BIN="${PACKAGE_DIR}/bin/nvidia_sb_pcm"
if [ ! -x "$NV_BIN" ]; then NV_BIN="${PACKAGE_DIR}/bin/nvidia_sb_ode_pcm"; fi

CMD_PREFIX=""
if command -v taskset >/dev/null 2>&1 && [ "$(nproc 2>/dev/null || echo 0)" -ge 8 ]; then
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
    affinity="${AUDIO3A_CPU_AFFINITY:-${cores:-0}}"
    CMD_PREFIX="taskset -c $affinity"
fi

# 默认启用 WPE + no-dereverb + dpdfnet + nvidia T32 ODE
# 保持 8 麦平均融合：conf_1to5m 远场实验中单麦 ch2 的 5m ASR 只有 50%，
# 而 8 麦平均为 100%。单麦在近场安静场景占优，但远场必须靠多麦降噪。
exec ${CMD_PREFIX} "${BIN}" \
    --mic-channels 10 \
    --ref-channel 8 \
    --mic-device hw:3,0 \
    --spk-device hw:3,0 \
    --wpe \
    --wpe-taps 8 \
    --wpe-delay 3 \
    --no-dereverb \
    --ans dpdfnet \
    --dpdfnet-model "${PACKAGE_DIR}/models/dpdfnet2.onnx" \
    --nvidia \
    --nvidia-rknn "${PACKAGE_DIR}/models/nvidia-t32-spatial-norm.rknn" \
    --nvidia-bin "$NV_BIN" \
    --nvidia-frames 32 \
    --nvidia-history-ms "$NVIDIA_HISTORY_MS" \
    --nvidia-overlap-ms 48 \
    --nvidia-core-mask 7 \
    --nvidia-dry-gain 0.000 \
    --record-raw /tmp/dut_raw10ch.wav \
    --record-clean /tmp/dut_clean.wav \
    "$@"
