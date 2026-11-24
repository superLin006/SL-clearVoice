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

上面的命令打包仓库随附的 T32 模型，使用 72 ms 交叠和三个 NPU 核。拉取新版后重新构建运行包，板端按包内的 `models/profile` 选择配置。

在相同依赖设置下，可显式构建低延迟的 T16 包，或传入自行编译的 T32 模型。

~~~sh
bash scripts/board/build_live_pcm.sh out/rk3588-live-t16 t16
bash scripts/board/build_live_pcm.sh out/rk3588-live-custom /path/to/nvidia-t32-spatial-norm.rknn
~~~

## 板端运行

复制整个构建目录到板卡。令 `LD_LIBRARY_PATH` 包含板卡的 RKNN 和 ONNX Runtime 库目录。

~~~sh
./run.sh --verify-models
./run.sh --pace < input.pcm > output.pcm 2> timing.log
~~~

`--pace` 按 16 ms 节拍回放录音，用于测量。实时输入由拾音前端持续提供时，直接运行 `./run.sh`。多通道 WAV 需由现有波束前端或选定的麦克风通道转换为单声道。比如 8 麦加 2 回采的 16 kHz 录音，可从第 4 路麦克风取出输入。

已经降噪的单声道 PCM 可使用 `./run.sh --denoised < input.pcm > output.pcm`，直接运行 NVIDIA 阶段。

~~~sh
ffmpeg -v error -i dut_raw10ch.wav -af 'pan=mono|c0=c3' -ar 16000 -f s16le input.pcm
ffmpeg -v error -f s16le -ar 16000 -ac 1 -i output.pcm enhanced.wav
~~~

## 参数与板测

| 配置 | 窗口 | 历史 | 交叠 | 推进 | 默认 NPU 核 | PCM 整链 P95 |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| T16 | 120 ms | 32 ms | 8 ms | 80 ms | 双核（mask=3） | 191.6 ms |
| T32 | 248 ms | 48 ms | 72 ms | 128 ms | 三核（mask=7） | 346.8 ms |

- T16 的数据来自 DUT 的 105.808 秒实时回放。T32 的数据来自 002 的 132.288 秒实时回放，最大 370.6 ms，1,033 个处理块均在预算内。T32 双核短测有 26/94 块超时。
- NVIDIA 重建原单步均值，按 0.025 的系数混入同时间戳的输入，日志记录 `gaussian_sampling=false`。
- 实时 v4 使用原版高低通和 EQ，接流式音量控制、软门限和限峰。按 256 个采样处理并输出。
- 时序日志写到标准错误。用 v4 的 `p95_origin_ms` 减去 UL-UNAS 的 `input_clock_ms`，可计算 PCM 整链 P95 延迟。声卡采播和会议端缓冲单独测量。
