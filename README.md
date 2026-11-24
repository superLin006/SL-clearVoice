# SL-clearVoice

## RK3588 实时语音增强

**UL-UNAS 降噪 → NVIDIA SB 去混响 → 实时 v4 后处理**。输入和输出均为 16 kHz、单声道、16 位小端 PCM。NVIDIA SB 使用确定性的单步均值重建。

| 配置 | NPU 核数 | 板端 PCM 回放 P95 | 测试录音 |
| --- | ---: | ---: | --- |
| T32（默认） | 3 | 346.8 ms | 002，约 132 秒 |
| T16（低延迟选项） | 2 | 191.6 ms | DUT，约 106 秒 |

板测按实时节拍回放录音，延迟从 PCM 输入算到 PCM 输出。声卡采播和会议端缓冲需另行测量。T32 在 002 的整段回放中，1,033 个处理块均在时间预算内。

默认构建仓库随附的 [T32 模型](models/rknn/live/nvidia-t32-spatial-norm.rknn)，使用 72 ms 窗口交叠和三个 NPU 核。需要低延迟配置时，构建时显式选择 T16。已通过 UL-UNAS 降噪的录音可在板端运行 `./run.sh --denoised`，只执行 NVIDIA SB 阶段。

- [构建、板端运行和延迟测量](scripts/board/LIVE.md)
- [模型重建、校验值和上游许可](models/rknn/live/README.md)

## 辅助工具

- [多麦 WPE 和波束回放](src/spatial/README.md)
- [同响度波形和频谱](scripts/plot_enhancement.py)。先按录音生成同响度 WAV，再用 JSON 描述同一时间轴上的各轨音频。
- [离线 v4 对照](scripts/v4_post.sh)。用于复现原版 v4 的听感对照。

## 本地数据

原始录音保存在 `data/`。近期 DUT 试听和已验证的板端构建位于 `out/`。旧训练、模型变体和评测记录位于本地相邻的 `SL-clearVoice-archive-2026-11-17`，归档清单保留原相对路径。
