# SL-clearVoice

## RK3588 实时基线

**UL-UNAS → NVIDIA 原权重单步 → 实时 v4**。输入为 16 kHz 单声道 PCM。双核 NPU 的整链 PCM 回放 P95 约 191 ms。

- [构建和板端运行](scripts/board/LIVE.md)
- [固定模型、校验值和上游许可](models/rknn/live/README.md)

## 辅助工具

- [多麦 WPE 和波束回放](src/spatial/README.md)
- [同响度波形和频谱](scripts/plot_enhancement.py)。先按录音生成同响度 WAV，再用 JSON 描述同一时间轴上的各轨音频。
- [离线 v4 对照](scripts/v4_post.sh)。用于复现原版 v4 的听感对照。

## 本地数据

原始录音保存在 `data/`。近期 DUT 试听和已验证的板端构建位于 `out/`。旧训练、模型变体和评测记录位于本地相邻的 `SL-clearVoice-archive-2026-11-17`，归档清单保留原相对路径。
