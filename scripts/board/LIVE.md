# RK3588 实时语音增强

固定配置为 **UL-UNAS → NVIDIA SB 单步 → 实时 v4**。UL-UNAS 运行在 CPU，NVIDIA 原权重运行在 NPU。输入和输出均为 16 kHz、单声道、有符号 16 位小端 PCM。

## 构建

分别准备 RKNN Runtime 2.3.2、ONNX Runtime 1.17.1 的 AArch64 动态库和对应头文件，以及 AArch64 FFTW 单精度静态库。使用和目标板 glibc 兼容的交叉编译器。本版用 GCC Linaro 7.2.1 验证。

~~~sh
AARCH64_CC=/path/to/aarch64-linux-gnu-gcc \
  RKNN_INCLUDE=/path/to/rknn/include RKNN_LIBDIR=/path/to/rknn/lib \
  ORT_INCLUDE=/path/to/onnxruntime/include ORT_LIBDIR=/path/to/onnxruntime/lib \
  FFTW_INCLUDE=/path/to/fftw/include FFTW_STATIC=/path/to/fftw/lib/libfftw3f.a \
  bash scripts/board/build_live_pcm.sh out/rk3588-live
~~~

构建目录内有 `bin/`、`models/` 和 `run.sh`。模型版本及许可见[模型说明](../../models/rknn/live/README.md)。

## 板端运行

复制整个构建目录到板卡。令 `LD_LIBRARY_PATH` 包含板卡的 RKNN 和 ONNX Runtime 库目录。

~~~sh
./run.sh --verify-models
./run.sh --pace < input.pcm > output.pcm 2> timing.log
~~~

`--pace` 按 16 ms 节拍回放录音，用于测量。实时输入由拾音前端持续提供时，直接运行 `./run.sh`。多通道 WAV 需由现有波束前端或选定的麦克风通道转换为单声道。比如 8 麦加 2 回采的 16 kHz 录音，可从第 4 路麦克风取出输入。

~~~sh
ffmpeg -v error -i dut_raw10ch.wav -af 'pan=mono|c0=c3' -ar 16000 -f s16le input.pcm
ffmpeg -v error -f s16le -ar 16000 -ac 1 -i output.pcm enhanced.wav
~~~

## 固定参数和实测

- NVIDIA 窗口 120 ms，复用 32 ms 历史，8 ms 交叠，每 80 ms 推进。将同样时间戳的输入以 0.025 倍相加。默认 `SB_RKNN_CORE_MASK=3` 使用 NPU 核 0 和 1。
- 实时 v4 使用原版高低通和 EQ，接流式音量控制、软门限和限峰。按 256 个采样处理并输出。
- 双核配置连续回放 242 秒，PCM 输入到输出 P95 为 191.4 ms，最大 205.3 ms。`SB_RKNN_CORE_MASK=7` 在 002 全段回放为 187.5 ms，双核为 191.2 ms，输出逐样本相同。声卡采播和会议端缓冲另测。
- 时序日志写到标准错误。用 v4 的 `p95_origin_ms` 减去 UL-UNAS 的 `input_clock_ms`，得到整链的 PCM P95 延迟。
