# 评测工具

Audio3A 链路的分级诊断与 A/B 工具。多数脚本针对板端 `--dump-stages` 导出的
逐级 WAV（`00_raw_input_avg` … `12_final_output`），也支持直接用真实录音。

## 反复踩到的坑，先读这段

**1. 跨录音集合比均值会产生偏置。** 曾经用"6 段 72ms"对比"4 段 48ms"得出
overlap 更好的结论，实际是缺了两段高分录音。**只能比较两边都有的录音。**

**2. DNSMOS 不总是和听感一致。** 它惩罚增益变动本身，所以对"字间变黑"这类
改动会扣分，人耳却觉得更干净。判据要按问题选：字间噪声看物理的语音/静音
分离度（`tail_audit.py`、`drr_metric.py`），综合质量才看 DNSMOS。

**3. 录音里有数字静音段（-200 dBFS）。** 用最小值做噪声估计会得到 140+ dB 的
SNR，任何基于 SNR 的门控都会失效。先用绝对门限（-90 dBFS）筛掉这些帧。

**4. 单次性能测量噪声很大。** 同一配置连测三次 RTF 可差 0.2 以上，测性能必须重复。

**5. 复现板端行为要用现成实现。** T32 使用仓库内的 `nvidia_runner.py`
（510 点 FFT、per-block 峰值归一化、0.33 幂律压缩）。自己写 512 点 FFT 版本会得出
错误结论。`Pipeline.process()` 返回 `(int16, float, blocks)`，取 `[0]` 后要除 32768。

**6. DPDFNet2 也要复现板端状态。** 共用 `dpdf_runner.py`，采用 Vorbis 窗、
模型元数据中的归一化初始状态和 256 样本输入节拍，保留 10 ms FIFO 预填。
同一段 51,231 样本音频与仓库 C 实现的本机对照，相关系数
0.99999999999997、相对 RMS 误差 2.39e-7。该数值核对不代表板端速度。
输出保留启动延迟、长度与输入相同；比较固定帧或尾音前仍需对齐。
`order_test.py` 现需显式提供单声道 16 kHz 输入和 `--output`；它只比较模型串联，
不包含 WPE、AGC、空间融合或实时调度。

**7. 能量落差和尾音代理不能当成真实 SNR/DRR。** 模型串联脚本保留的
简易代理会分别从各输出估计能量掩码，只用于描述单个输出，不能直接据此排名。
跨阶段比较须先对齐，再共用输入掩码；输出被门控到数字零时应单独报告，
不要把任意数值下限解释成极低的实测噪声。

当前整链使用 T32、48 ms 历史、48 ms 交叠、152 ms 推进和 dry=0。
离线 T32 入口采用同一配置。ONNX 模型可通过命令参数或环境变量指定。

~~~sh
python scripts/eval/t32_correct.py --model /path/to/nvidia-t32-spatial-norm.onnx 10_smooth_gate.wav
NVIDIA_T32_ONNX=/path/to/nvidia-t32-spatial-norm.onnx python scripts/eval/role_split.py 10_smooth_gate.wav
~~~

## 分级诊断

| 脚本 | 用途 |
| --- | --- |
| `aligned_stage_audit.py <stage_dir>... --output report.json` | 先对齐，固定输入能量掩码，排除数字零，输出逐级 JSON |
| `stage_audit.py <stage_dir>...` | 旧版频谱/动态描述；未补偿阶段延迟 |
| `tail_audit.py <stage_dir>...` | 旧版尾音代理；未补偿阶段延迟 |
| `drr_metric.py <f32/wav>...` | 单文件 DRR 代理指标 |
| `role_split.py` | 区分降噪能力与去混响能力，两轴分别评分 |

## 模型对比

| 脚本 | 用途 |
| --- | --- |
| `denoiser_compare.py <wav>...` | DPDFNet2 vs UL-UNAS |
| `dpdfnet_family.py` | DPDFNet baseline/2/4/8 横评 + RTF |
| `fastenhancer_run.py <wav>` | FastEnhancer 替代 NVIDIA 阶段 |
| `t32_correct.py <wav>...` | 板端匹配的 T32 复现 |
| `server_t32_eval.py <wav>...` | 服务器批量跑 T32 变体 |
| `order_test.py` | 降噪→去混响 vs 去混响→降噪 |

## AGC 与后处理

| 脚本 | 用途 |
| --- | --- |
| `agc_offline.c <in.wav> <off.f32> <on.f32>` | 板端 AGC 扩展开关对比 |
| `postnv_agc.c <in> <off.f32> <on.f32>` | 复现 post-NVIDIA AGC（与板端逐样本一致） |
| `v4_chain.py` | V4 后处理的 numpy 复现（ffmpeg 不可用时） |
| `hf_denoise.py <wav>` | 3.6kHz 以上谱减原型 |

## 听感材料

| 脚本 | 用途 |
| --- | --- |
| `make_listen.py` | 从 stage dump 生成 raw/current/pre-AGC 对比 |
| `make_agc_ab.py` | 生成 AGC 修复前后的 A/B |

## 环境

`speechmos`（DNSMOS）、`onnxruntime`、`soundfile`、`scipy`、`numpy`。
本机 `dl` 环境齐全。ffmpeg 用 `/home/xh/miniconda3/envs/subtitle/bin/ffmpeg`。

## 输出位置

评测结果应写入显式指定的仓库外目录，保留输入、模型、参数和 SHA。
原始录音不覆盖；大体积音频、临时模型和阶段导出不加入 Git。

## 这一轮的实测结论（2026-11-27）

改链路前先看这些，能省掉重复试错。

**整链当前状态**（板端 `--dump-stages` 六段平均）：

| 阶段 | 降噪能力 | 去混响 |
| --- | ---: | ---: |
| 原始输入 | 5.0 | 4.1 |
| WPE | 8.0 | 5.4 |
| NeuralRES | 8.2 | 5.4 |
| pre-AGC | **6.3 ↓** | 5.1 |
| DPDFNet2 | 23.5 | 14.6 |
| NVIDIA T32 | **33.4** | **21.2** |
| 最终输出 | 31.8 ↓ | 20.1 ↓ |

两个模型职责不同但互补：DPDFNet2 主要建立降噪基础，T32 两个维度都最强。
串联优于任一单级（`role_split.py` 可复现）。

**试过但不可行的方案，不必重复：**

| 方案 | 结果 |
| --- | --- |
| 50 步扩散采样 | 听感最好、DNSMOS 高 0.146，但需 GPU，RK3588 不能实时 |
| FastEnhancer 替代 T32 | 真实链路上更差（OVRL 2.510 vs 2.728）——它在纯去混响合成数据上 PESQ 更高，但输入分布不同 |
| T32 overlap 72ms | 与 48ms 在 DNSMOS 上等价（-0.016），已回退 |
| 顺序改为 T32→DPDF2 | 已在输出线程实现并完成板测；实录结果见整链报告，不能只据历史 +0.073 判断 |
| DPDFNet 换大模型 | 越大越差（4/8 都低于 2） |
| post-NV AGC 加扩展（仅按 DNSMOS 判断时） | 曾判为负收益；改用物理的语音/静音分离度后结论反转，最终启用 |

**有效并已落地：** pre-AGC 与 post-NV AGC 的向下扩展。post-NV 那级实测把
ODE 之后的静音段从 -52.1 压到 -64.2 dBFS，语音电平不变。

**高频"雪花感"的定位**：3.6 kHz 以上 SNR 断崖下降（语音帧 vs 非语音帧：

1.2-2.4k 22.9 dB、2.4-3.6k 23.1 dB、**3.6-4.8k 12.3 dB、4.8-8k 8.4 dB**）。
V4 后处理自带 6 kHz 低通，实测把 6-8 kHz 压低 7.8 dB，已覆盖这部分；
额外加 4.5 kHz 搁架属于重复处理，反而过度。

## 整链复测入口

构建和回放见 src/audio3a/README.md。compare_chain_runs.py 统一生成同响度音频与频谱；DNSMOS 设置共用 audio_metrics.py（9.01 秒窗口、2 秒推进、评分前 RMS 0.03）。不同窗口或增益口径的历史数值不直接比均值。

T32归一化算子、六段两档全长和原生同输入复核见[最新验收](../../docs/reviews/2026-11-27-t32-operator-validation.md)。该报告评分为1秒步进；上述整链工具默认2秒步进，不能混用缓存或均值。低电平副本可用 prepare_meeting_inputs.py 生成，输入SHA已核对一致。
