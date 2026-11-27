# SL-clearVoice

RK3588 会议室语音增强整套链路的研究与交付仓库。

**交付物是完整的处理链路，不是单个模型。** 因此优化必须以整链指标判断，
原始数据必须保持未处理状态，任何新算法都要能回到同一批原始数据上重跑对比。

## 当前链路

```
8 麦采集 → 80Hz HPF → 8 麦分段 EQ 校准 → LinearAEC(每麦)
  → 多通道 RLS-WPE → 空间融合 → NeuralRES
  → AGC①(pre-ANS) → 瞬态交叉 → DPDFNet2(降噪, CPU)
  → AGC② → 平滑门 → NVIDIA T32(去混响, NPU)
  → AGC③(post-NV) → V4 后处理
```

| 模块 | 位置 | 说明 |
| --- | --- | --- |
| 链路编排 | `src/audio3a/tools/p2p_aec/p2p_dut_node.c` | 板端主程序，含分级 dump |
| AGC | `src/audio3a/src/adaptive_agc.c` | 三级，含向下扩展 |
| 降噪 | `models/dpdfnet/` | DPDFNet2 |
| 去混响 | `models/rknn/live/` | NVIDIA T32，NPU |
| V4 后处理 | `scripts/v4_post.sh` | 频谱塑形 + 电平 |

板端调度脚本见 `src/audio3a/scripts/`。

## 目录结构

```
data/raw/            原始录音清单与校验和（音频本体不入库）
docs/reference/      评测参考数据（50 步基线、模型横评 JSON）
docs/history/        历史交接文档归档
models/              RKNN/ONNX 模型与校验值
scripts/eval/        分级诊断与 A/B 工具
scripts/board/       板端构建与运行
scripts/npu/         模型导出与 RKNN 转换
scripts/dpdfnet/     DPDFNet 部署
src/audio3a/         Audio3A 链路源码（含 rnnoise、webrtc_vad）
```

## 数据纪律

`data/raw/` 只放**未经任何算法处理的原始录音**，音频本体不进 git（201 MB），
用 `MANIFEST.md` 记录采集条件、`SHA256SUMS` 保证不被替换。评测前先校验：

~~~sh
cd data/raw && sha256sum -c SHA256SUMS
~~~

这条规则是为了保证任何优化都能在同一批原始数据上重跑对比——一旦原始数据里
混入了处理过的版本，后续所有结论都不可信。

当前主测试集是 `meeting_1..6.raw10ch.wav`（会议室实录，8 麦 + 2 参考）。
**其中 `meeting_3` / `meeting_4` 是刻意压到 -69 / -73 dBFS 的低电平样本**，
直接送进链路会因电平过低输出静音，使用前需放大——详见 `data/raw/MANIFEST.md`。

## 评测

分级诊断、模型横评、AGC 与后处理对比的工具在 `scripts/eval/`，里头的 README
记了五个反复踩到的坑（偏置对比、DNSMOS 与听感背离、数字静音段、性能测量噪声、
模型复现必须用现成实现），改链路前值得先读一遍。

## 许可

`src/audio3a/` 内的 rnnoise、webrtc_vad 保留各自上游许可。
NVIDIA 模型与 UL-UNAS 的许可说明见 `models/rknn/live/README.md`。
