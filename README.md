# SL-clearVoice

实时去噪声、去混响。

## 当前状态（2026-11-05）

- 实时主线：`UL-UNAS → CRM v2 → v4`
- 离线高质基线：`UL-UNAS → NVIDIA（50 步）→ v4`（B，不含 MGan）
- 详细交接：[SL-clearVoice-handoff-11-05.md](SL-clearVoice-handoff-11-05.md)

## 仓库内容

- `src/`：CRM v2 模型、200k 数据生成、训练、评测、ONNX 导出
- `scripts/v4_post.sh`：与线上一致的 v4 后处理 ffmpeg 链
- `models/crm2_base.onnx`：CRM v2 的 ONNX 核心（复数谱进、增强复数谱出）
- `models/README.md`：ONNX 输入输出与 checkpoint 路径
