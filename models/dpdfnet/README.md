# DPDFNet2 原权重与 RK3588 模型

对应 2026-11-27 板测版本，16 kHz 单声道、320 点 FFT、160 点帧移。来自同事工程 `Audio3A/models/onnx/dpdfnet/dpdfnet2.onnx`，源文件 SHA256 为 `ce35d6025fc71df0ef10d1540e1b7916837bbfe5f6896deb744508d2cad487a9`。

| 文件 | 用途 |
| --- | --- |
| `dpdfnet2.onnx` | 原始 FP32 参照模型 |
| `dpdfnet2-cpu.onnx` | 等价分组矩阵乘法合并后的 FP32 CPU 模型 |
| `dpdfnet2-cpu-int8.onnx` | 可选 CPU 混合精度，只量化 5 个输入投影 |
| `dpdfnet2-core.rknn` | RK3588 FP16 神经网络，已融合分组投影、GRU 门投影及双向频率 GRU |
| `initial-state.bin` | 45424 个 float32 初始状态 |
| `erb-filterbank.bin` | 161 × 32 float32 ERB 滤波器组 |
| `manifest.json` | 来源、处理配置与模型校验值 |
| `SHA256SUMS` | 可直接执行 `sha256sum -c SHA256SUMS` |
| `LICENSE` | 上游 Apache-2.0 许可副本 |

NPU 版的 ERB 和复谱归一化在 CPU 上以 FP32 执行，避免 FP16 对数与除法在静音段产生非有限数。神经网络权重沿用原模型，导出图做等价运算合并，编译时采用 FP16。FP16 仍会产生数值误差，试听和验证结果见 [运行说明](../../scripts/dpdfnet/README.md)。

模型内部谱缓存延迟为 640 样本，即 40 ms。运行器另有一个 STFT 启动帧，并在文件结束时冲刷状态以保留完整尾部。文件输出与输入样本数相同且已补偿固定时间偏移。实时输出仍需等待这些缓存，因此不能把单帧计算时间作为语音延迟。

重建方式和板端接口见 [scripts/dpdfnet](../../scripts/dpdfnet/README.md)。导出依赖 Python 3.10+、NumPy、ONNX、ONNX Runtime；RKNN 编译已验证 RKNN Toolkit2 2.3.2。模型来源 [Ceva-IP/DPDFNet](https://huggingface.co/Ceva-IP/DPDFNet)，上游项目 [ceva-ip/DPDFNet](https://github.com/ceva-ip/DPDFNet)。

2026-11-27 第二轮增加选择性 CPU INT8，重建命令见运行说明。NPU 模型文件未重新编译，运行包新增独立 Runtime 2.5.0，三段完整录音与 2.3.2 的输出逐样本一致。CPU INT8 的作用范围和量化前 FP32 图改写验证记录写入 manifest.json。

第三轮默认 NPU 核心使用双向频率 GRU 的等价合并。80 帧 FP32 谱与状态误差为 0；002、DUT 155 秒和静音接语音三段板端 PCM 与第二轮逐样本一致。RKNN 约 12.22 MB，保留原始训练参数。核心图与板测记录见运行说明及 manifest.json。
