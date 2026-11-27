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

**5. 复现板端行为要用现成实现。** T32 必须用 `out/t32-50-v4-20261123/run_nvidia.py`
（510 点 FFT、per-block 峰值归一化、0.33 幂律压缩）。自己写 512 点 FFT 版本会得出
错误结论。`Pipeline.process()` 返回 `(int16, float, blocks)`，取 `[0]` 后要除 32768。

## 分级诊断

| 脚本 | 用途 |
| --- | --- |
| `stage_audit.py <stage_dir>...` | 固定语音掩码的逐级频谱/动态对比 |
| `tail_audit.py <stage_dir>...` | 逐级尾音落差（去混响能力） |
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

音频产物放在**会话 worktree 内**（`.claude/worktrees/<name>/out/`），
放外层仓库的文件面板看不到。

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
| 顺序改为 T32→DPDF2 | +0.073，但 NVIDIA 是异步子进程，实时链路要阻塞等待，不可行 |
| DPDFNet 换大模型 | 越大越差（4/8 都低于 2） |
| post-NV AGC 加扩展（仅按 DNSMOS 判断时） | 曾判为负收益；改用物理的语音/静音分离度后结论反转，最终启用 |

**有效并已落地：** pre-AGC 与 post-NV AGC 的向下扩展。post-NV 那级实测把
ODE 之后的静音段从 -52.1 压到 -64.2 dBFS，语音电平不变。

**高频"雪花感"的定位**：3.6 kHz 以上 SNR 断崖下降（语音帧 vs 非语音帧：

1.2-2.4k 22.9 dB、2.4-3.6k 23.1 dB、**3.6-4.8k 12.3 dB、4.8-8k 8.4 dB**）。
V4 后处理自带 6 kHz 低通，实测把 6-8 kHz 压低 7.8 dB，已覆盖这部分；
额外加 4.5 kHz 搁架属于重复处理，反而过度。
