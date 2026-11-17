# SL-clearVoice

实时去噪声、去混响。

## RK3588 实时基线（2026-11-17）

**UL-UNAS → NVIDIA 原权重单步 → 实时 v4**。双核 NPU 全链 PCM 回放 P95 约 191 ms。

- [构建和板端运行](scripts/board/LIVE.md)
- [固定模型、校验值和上游许可](models/rknn/live/README.md)

[项目交接文档](SL-clearVoice-handoff-11-05.md)

## 仓库内容

- `src/`：CRM v2 模型、200k 数据生成、训练、评测、ONNX 导出
- `scripts/v4_post.sh`：与线上一致的 v4 后处理 ffmpeg 链
- `models/crm2_base.onnx`：CRM v2 的 ONNX 核心（复数谱进、增强复数谱出）
- `models/README.md`：ONNX 输入输出与 checkpoint 路径
