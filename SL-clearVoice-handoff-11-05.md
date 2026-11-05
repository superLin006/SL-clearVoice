# SL-clearVoice 交接文档（2026-11-05）

## 0. 一句话结论

当前实时主线是 **UL-UNAS 去噪 → CRM v2 去混响 → v4 后处理**。CRM v2 是 UL-UNAS 同族 DPGRNN 主干 + 恒等残差复数掩码，950,915 参数、时间因果，可直接替换 B 链中的 NVIDIA 扩散模型，但质量仍低于离线 B：41s 实录 B 更好；实测 1/3/5m 段 CRM 的 PESQ/STOI/BAK 更好，B 的 DNS SIG/OVRL 更高。

**B 的定义（重要）**：`UL-UNAS → NVIDIA（50 步）→ v4`，不含 MGan。MGan 已从本项目和所有评测中移除。

---

## 1. 最终链路与选型

```
实时/板端：  UL-UNAS（降噪） → CRM v2（去混响） → v4（EQ/leveler/limiter）
离线高质量：  UL-UNAS（降噪） → NVIDIA 50 步（去混响） → v4
```

- 顺序“先降噪、再去混响、最后 v4”与 B 完全同构，已被指标和听感验证。
- CRM v2 的定位是实时替代 NVIDIA 扩散阶段；v4 链路保持线上原样。

---

## 2. 资产与路径

| 资产 | 路径 | 说明 |
|---|---|---|
| CRM v2 基座 checkpoint | `<TRAIN_ROOT>/ulunas_train_code/ckpt_crm_200k/crm_ep7.pt` | 第 7 轮，最终采用；SHA256 `a8159a429190efe53d6d99db99c089e31cfe864c36d44023f519f7f4cde392dd` |
| ONNX 核心 | `models/crm2_base.onnx` | 3.7 MB；SHA256 `61a7d8ce0958df653ba65bc834265d8c4a5b863f2aace20e9e40e15562d5402a` |
| 试听包 | `<SL_DATA_ROOT>/out/ab_final.zip` | base / B / GAN / DPCRN / a70，含 MP3 与 README |
| 训练代码 | `src/train_crm.py`、`src/train_crm_ft.py` | DDP，200k dry 目标与微调 |
| 模型代码 | `src/crm_model.py` | CRMFilterV2 / CRM2Core |
| UL-UNAS 依赖 | `src/ulunas.py` | MIT，版权见 `src/UL-UNAS-LICENSE.txt` |
| v4 后处理 | `scripts/v4_post.sh` | 与线上一致，已验证 1:1 |

服务器 GPU 归属：0/1 曾用于 SGMSE；2/3 是用户自己的卡；4/5/7/8 常有他人任务。交接时本项目训练已全部停止。

---

## 3. 数据

### 3.1 200k 训练集

- 语音：AISHELL-3 train + THCHS-30 train 循环取 200,000 条。
- RIR：OpenSLR RIRS_NOISES simulated_rirs。
- 噪声：MUSAN noise，SNR 0–20 dB，`p_noise=0.85`。
- 段长：32,000 点（2 s），16 kHz。
- 生成：`src/make_200k_data.py <shard> <nshard>` → `<TRAIN_ROOT>/distill_200k/{noisy,dry}`。
- UL-UNAS 输入：`src/gen_ulunas_200k.py <shard> <nshard>` → `<TRAIN_ROOT>/distill_200k/ulunas`。
- 训练软链：`src/build_200k_train.py` → `<TRAIN_ROOT>/distill_200k_train`，`noisy=UL-UNAS 输出`，`teacher=dry`，`dry=dry`；验证集 600 条来自 `<TRAIN_ROOT>/distill_data3s3`。
- 关键归一化：每条 noisy/dry 联合缩放到 RMS 0.04；推理时输入也缩放到 0.04，输出恢复原 RMS。

### 3.2 公平集

- 30 条 AISHELL-3 未见测试说话人，3 s，模拟小/中/大房间 RIR。
- 条件：`rev`（纯混响）、`rev_noise`（混响 + 10 dB MUSAN 噪声）。
- 输入为 UL-UNAS 输出；B 为 NVIDIA-only。

### 3.3 实测集

- 41s 实录（1 m 左右），以及 1/3/5 m 距离片段、rec5/6/7 新录音。
- 1/3/5 m 有 `seg_*_ref.wav` 参考，可算 PESQ/STOI。
- 所有方法在实测集比较时统一接 v4。

---

## 4. 模型

### 4.1 CRM v2

- 输入：单通道 16 kHz 波形；内部 STFT 512/256/512 Hann。
- 特征：复数谱 real/imag + log magnitude，3 通道。
- 主干：`CausalConv` + 2 个 `DPGRNN`（intra 沿频、bidirectional；inter 沿时间、单向），ch=128。
- 输出：2 通道残差复数掩码，`tanh` 限幅；`M = 1 + ΔM`，恒等初始化（输出层零初始化）。
- 参数量：950,915。
- 因果性：时间维因果（卷积只看过去帧，inter GRU 单向），可流式；频域 intra GRU 双向不影响时间因果。
- 代码：`src/crm_model.py` 的 `CRMFilterV2` 与 `CRM2Core`。

### 4.2 ONNX

- `models/crm2_base.onnx` 是 **CRM2Core**：输入 `spec [1,2,T,257]`（STFT real/imag），输出 `enh [1,2,T,257]`；不含 STFT/iSTFT，方便板端复用已有 STFT。
- 动态 T；F 固定 257。
- 目前是整段核心，**尚未导出流式 cache**（CausalConv 历史帧 + DPGRNN inter GRU 状态），上板前必须补。

---

## 5. 训练

实际主线命令（6 卡，200k，12 轮计划；第 7 轮后停止并选用 ep7）：

```bash
cd <TRAIN_ROOT>/ulunas_train_code
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 torchrun --nproc_per_node=6 --master_port=29522 \
  train_crm.py --root <TRAIN_ROOT>/distill_200k_train \
  --out_dir <TRAIN_ROOT>/ulunas_train_code/ckpt_crm_200k \
  --epochs 12 --batch 16 --workers 6 --lr 3e-4 \
  --teacher_w 0 --dry_w 1.0 --arch crm2 --ch 128
```

- 损失：`-SI-SDR + 0.1 × (复数谱 L1 + 幅度 L1)`，目标为 dry。
- 优化器 AdamW，余弦退火，epoch 0–11；实际在 ep7 保存后停止。
- 验证：600 条 dry 目标 SI-SDR；ep7 val dry SI-SDR ≈ 1.15 dB（输入 UL-UNAS 约 −17.7 dB，该指标是相对 dry 的尺度）。
- 微调版 `train_crm_ft.py --init <ckpt>` 支持教师/真实域微调，但均未进入主线（见负结果）。

---

## 6. 评测

```bash
# 41s + 公平集推理（WSL GPU）
python src/run_crm2_eval.py <ckpt> <tag> 128

# 41s PESQ/STOI/DNSMOS（需先过 v4）
bash scripts/v4_post.sh out/full41_models/<tag>.wav out/full41_audition/<tag>_v4exact.wav
python src/metrics_full41.py out/full41_audition/<tag>_v4exact.wav

# 公平集
SL_DATA_ROOT=/path/to/data python src/metrics_fair.py <tag>_rev <tag>_rev_noise

# 实测 1/3/5m
python src/metrics_real.py <tag>
python src/metrics_real_ref.py <tag>
```

- `SL_DATA_ROOT` 默认 `./data`，可在新环境覆盖。
- PESQ/STOI/DNSMOS 在 WSL 的 `dl` 环境；SQUIM 需要 torchaudio 权重。
- 所有比较必须同响度、同 v4 条件；B 必须 NVIDIA-only。

---

## 7. 最终指标

### 7.1 41s 实录（全部接 v4）

| 方法 | PESQ | STOI | DNS OVRL/SIG/BAK |
|---|---|---|---|
| B（UL-UNAS→NVIDIA→v4） | **1.468** | **0.776** | **2.82 / 3.21 / 3.72** |
| UL-UNAS 无去混响 | 1.206 | 0.585 | 2.31 / 2.75 / 3.46 |
| CRM v2 base | 1.397 | 0.763 | 2.47 / 2.80 / 3.69 |
| CRM v2 a70（干湿比 0.7） | 1.320 | 0.748 | 2.66 / 3.01 / 3.76 |
| CRM-GAN（实验，未采用） | 1.411 | 0.764 | 2.45 / 2.78 / 3.66 |
| DPCRN 残差（实验，未采用） | 1.293 | 0.738 | 2.28 / 2.68 / 3.43 |

### 7.2 公平集 30 条（未接 v4，同条件）

| 方法 | rev PESQ / STOI | rev+noise PESQ / STOI | rev DNS OVRL/SIG/BAK | rev+noise DNS OVRL/SIG/BAK |
|---|---|---|---|---|
| UL-UNAS | 2.120 / 0.843 | 1.931 / 0.821 | 2.615/2.988/3.953 | 2.494/2.858/3.917 |
| B | 1.865 / 0.899 | 1.681 / 0.866 | 2.912/3.230/4.035 | 2.792/3.115/4.006 |
| CRM v2 | **2.159 / 0.881** | **1.924 / 0.857** | 2.756/3.086/3.998 | 2.607/2.951/3.936 |
| CRM-GAN | 2.199 / 0.884 | 1.939 / 0.860 | 2.765/3.100/4.001 | 2.629/2.966/3.955 |
| DPCRN 残差 | 2.099 / 0.868 | 1.842 / 0.843 | 2.615/2.951/3.981 | 2.462/2.790/3.935 |

SQUIM（参考型）：UL-UNAS 4.389/4.288，B 4.269/4.291，CRM 4.390/4.286。

### 7.3 实测 1/3/5m（全部接 v4，PESQ/STOI 相对 ref）

| 方法 | PESQ | STOI | DNS OVRL/SIG/BAK |
|---|---|---|---|
| UL-UNAS | 1.201 | 0.665 | 2.241/2.794/3.162 |
| B（NVIDIA-only） | 1.309 | 0.740 | **2.624/3.532**/2.791 |
| CRM v2 | **1.380** | **0.758** | 2.387/2.753/**3.481** |
| CRM a70 | 1.323 | 0.747 | 2.370/2.782/3.386 |
| CRM-GAN | 1.292 | 0.754 | 2.525/2.893/3.596 |
| DPCRN 残差 | 1.270 | 0.738 | 2.235/2.692/3.289 |

### 7.4 ASR（Qwen3-ASR-0.6B，相对 dry 识别结果）

- 10 条 rev_noise：UL-UNAS CER 0.225、B 0.267、CRM v2 0.200。
- 30 条 rev_noise（CRM ep4）：UL-UNAS 0.125、B 0.122、CRM 0.121。
- 样本较小，只能说明“无明显退化/略好”，不能宣称大幅提升。

---

## 8. 关键负结果（避免重复踩坑）

| 实验 | 结果 | 结论 |
|---|---|---|
| stage2：UL-UNAS → 第二个 UL-UNAS | 比 UL-UNAS + v4b 差 | 已放弃 |
| stage3：UL-UNAS 幅度掩码蒸馏 NVIDIA | 初始化/从零都差 | 相位/容量限制 |
| DPCRN（复数比值掩码） | ep5 val dry −1.10，CRM 同期 1.00 | 原版无恒等先验，收敛慢 |
| DPCRN 残差版 | ep4 val dry −0.47，所有指标低于 CRM | CRN+LSTM 不适合本任务 |
| Teacher-FT（NVIDIA 目标 1.0） | 41s PESQ 1.343，DNS 2.29/2.62/3.59 | 学到了生成式相位，丢保真 |
| Mix-FT（教师 0.3 + dry 1.0） | 1.379/0.762，DNS 2.40/2.73/3.65 | 仍低于 base |
| Real-FT（真实段 NVIDIA 目标） | 41s 1.414，DNS 2.41/2.73/3.67；实测段更差 | 未采用 |
| CRM-GAN（dry 目标，1 轮对抗） | 公平集 PESQ +0.04，41s DNS 略降；第 2 轮更差 | 小幅收益，不足以超过 B |
| CRM-GAN（教师真实样本 + 强对抗） | 41s 1.408，DNS 2.42/2.75/3.64 | 无额外收益 |

**为什么 NVIDIA 蒸馏不理想**：NVIDIA 是扩散生成模型，会改相位；确定性的复数掩码学生无法拟合一对多的相位目标，强行拟合会牺牲干净度。真正适合扩散教师的是生成式学生（4 步 Flow/Consistency 或小型扩散模型），但实时性要重新评估。

---

## 9. 部署状态

- ONNX 核心已导出并做过 PyTorch 对齐（corr ≈ 1，maxabs 7.7e-5）。
- WSL CPU 实测：CRM 核心 RTF 0.18（3 s）/0.28（41 s）；UL-UNAS 0.059，约 3 倍算力。
- 按板端 UL-UNAS RTF 0.115 估算，CRM 阶段约 0.35–0.55，整链约 0.5–0.7。**板子交接时连不上（<BOARD_IP>:<PORT> 超时），必须在目标板实测。**
- 尚未做流式导出：需要 CausalConv 历史帧 cache + 每层 DPGRNN inter GRU hidden state；当前 ONNX 只能整段处理。
- 板端如果 RTF 不够，可训 ch96/单 DPGRNN 的 lite 版（算力约减半），或把 CRM 部分算子放 NPU；GRU/LayerNorm 的 RKNN 支持需要单独验证。

---

## 9.5 2026-11-05 补充实测与判断

### 板端 RTF（RK3588，ORT 1.18.1，实测）

| 模型 | 线程 | 3 s RTF | 41 s RTF |
|---|---|---|---|
| CRM2 core | 4 | 0.491 | 0.493 |
| CRM2 core | 8 | 0.581 | - |
| CRM2 core | 2 | 0.707 | - |
| UL-UNAS streaming | 4 | 0.141 | - |

- 4 线程最优；按实测估算整链 `UL-UNAS + CRM2` 约 **0.63 RTF**，仍有约 37% 余量，但未计 AEC/ASR。
- 当前 ONNX 是**整段核心**，未导出流式 cache；真实实时链路还需补 CausalConv/GRU cache。

### 继续训练（从 ep7 继续）

| 模型 | 公平集 rev PESQ/STOI | rev+noise PESQ/STOI | 41s PESQ/STOI | 41s DNS OVRL/SIG/BAK | 实测段 DNS OVRL/SIG/BAK |
|---|---|---|---|---|---|
| base ep7 | 2.159/0.881 | 1.924/0.857 | 1.397/0.763 | 2.47/2.80/3.69 | 2.488/2.845/3.604 |
| +1 epoch | **2.223/0.886** | **1.953/0.861** | 1.402/0.761 | **2.53/2.85/3.70** | **2.586/2.966/3.609** |
| +2 epoch | 2.199/0.885 | 1.942/0.861 | **1.414/0.763** | 2.45/2.78/3.66 | - |

- 再训 1 轮有真实但有限的提升（公平集 PESQ +0.06，41s DNS +0.05，实测 DNS +0.1）；第 2 轮不再单调提升，且仍低于 B。
- 结论：**继续加轮次/同分布数据只能小幅提升，无法补上到 B 的差距**；更高价值的是改数据分布（真实 RIR/真实房间配对）或换生成式学生/多麦。

### SGMSE 教师评估

- 全量 24.6k 标签已完成（6×4100），不是半成品。
- 41s 实录（v4 后）：PESQ 1.469 / STOI 0.729 / DNS 2.46/3.02/3.32。
- 实测 1/3/5m（v4 后）：PESQ 1.452 / STOI 0.731 / DNS 2.413/3.092/3.076。
- 对比 B（41s 1.468/0.776/2.82/3.21/3.72；实测 1.309/0.740/2.624/3.532/2.791）：SGMSE 的 PESQ 有优势，但 STOI/DNS/BAK 并不更好，尤其在真实房间。
- **结论：不建议再拿 SGMSE 做确定性 CRM 的教师**（同样受生成式相位限制）；它的价值只在未来 4 步 Flow/Consistency 生成式学生或离线对照。

### 数据与 Auphonic 判断

- 当前 200k 使用的是 simulated_rirs；服务器另有 **417 条 real RIRs**（real_rirs_isotropic_noises），这是最值得试的数据改动。
- 建议下一步先做 60k 子集：按 50% real RIR + 50% simulated 重新生成 noisy/dry → UL-UNAS 输入 → 训练/微调，评估真实 1/3/5m；这比继续扩同分布 200k 更有针对性。
- Auphonic 目前只有 41s 和 5m 两份参考，且没有可批量调用的 token/API；其输出还包含 EQ/压缩/leveler 的完整后处理，不是纯去混响教师。**不适合作为 200k 蒸馏教师**，只适合作为评价参考或 v4 调参目标。

### 清理记录

- 训练服务器释放约 25 GB：删除 `distill_big`、旧 `distill_data*` 变体、旧 teacher/shards、DPCRN/真实域微调数据、旧 `ckpt_*`。
- D 盘释放约 4 GB：删除旧的 60k/24k 数据、Resemble/ClearVoice/Micro/RKNN 探针、旧试听中间件；保留 200k 主数据、实测录音、最终 A/B、基座/B/ONNX。
- 清理时发现 val 的 UL-UNAS 输入依赖旧 `ulunas_stage1`，已重新生成 600 条 val，训练验证集恢复正常。

## 9.6 2026-11-05 多麦空间链（新突破）

在 1/3/5 m 实测段上，第一次做到各距离都超过单通道 B 的 PESQ/STOI/BAK：

```
6ch WPE（nara_wpe, taps=10, delay=3, iterations=3）
  -> 延迟求和波束（R=42 mm, DOA=90°）
  -> UL-UNAS -> CRM v2 -> v4
```

| 方法（均接 v4） | PESQ | STOI | DNS OVRL/SIG/BAK |
|---|---|---|---|
| B（单通道） | 1.309 | 0.740 | 2.624 / 3.532 / 2.791 |
| 单通道 CRM v2 | 1.380 | 0.758 | 2.387 / 2.753 / 3.481 |
| DS + UL-UNAS + CRM（无 WPE） | 1.409 | 0.757 | 2.459 / 2.824 / 3.582 |
| **WPE + DS + UL-UNAS + CRM** | **1.559** | **0.782** | **2.755** / 3.097 / **3.771** |

- 分距离 PESQ：1 m 1.861（B 1.483）、3 m 1.475（B 1.265）、5 m 1.340（B 1.181）。
- B 的 SIG 仍更高（3.532 vs 3.097），但多麦链的 PESQ/STOI/OVRL/BAK 更好；听感需要用户确认。
- 消融：去掉 WPE 后 PESQ 从 1.559 掉到 1.409，WPE 是主要增益来源；MVDR 波束比 DS 差（1.322/0.717）。
- 关键坑：`nara_wpe` STFT 输出 `(M,T,F)`，WPE 需要 `(F,M,T)`；旧代码轴序错误，导致 WPE 输出与输入不相关。WPE 输出与输入时间对齐，不要额外 shift。
- 代码：`src/spatial/wpe_ds.py`、`src/spatial/run_spatial_chain.py`；试听：`<SL_DATA_ROOT>/out/ab_spatial`。
- **限制**：WPE 目前是离线整段迭代，实时部署需要 block/online WPE；评估只有 1/3/5 m 三段，需再用 8ch 41s 和更多房间验证。

## 9.7 Online WPE 与板端 RTF（2026-11-05）

板端 RK3588 + ORT 1.18（4 线程）实测：

| 处理 | RTF |
|---|---|
| 6ch online WPE（Python/NumPy） | 1.576 |
| 6ch online WPE + DS | 1.583 |
| DS 波束 | 0.007 |
| DS + 单通道 online WPE | 0.06 |
| UL-UNAS streaming | 0.141 |
| CRM2 core | 0.493 |
| **DS + mono online WPE + UL-UNAS + CRM2（合计估计）** | **0.69** |

- 6ch online WPE 单独 1.576，整链约 2.2，不能实时；要实时必须把 6ch WPE 用 C/C++ 重写或换更轻的 WPE。
- 实时折中：`DS 波束 -> 单通道 online WPE -> UL-UNAS -> CRM2`，板端合计约 0.69 RTF；质量：
  - 41s：1.408/0.766/2.56/2.88/3.77（B 1.468/0.776/2.82/3.21/3.72）
  - 1/3/5m：1.374/0.749/2.351/2.685/3.574（B 1.309/0.740/2.624/3.532/2.791）
- 离线 6ch WPE 链仍是质量上限：41s 1.472/0.776/2.72/3.00/3.91；1/3/5m 1.559/0.782/2.755/3.097/3.771。
- 建议双轨：文件后处理用离线 6ch WPE；实时用 DS+mono online WPE+UL+CRM（RTF≈0.69）。若必须实时且要离线质量，下一步用 C/C++ 实现 6ch online WPE。

## 10. 已知限制与下一步

1. **综合没有超过 B**：41s 上 B 更好；实测距离段 CRM 的 PESQ/STOI/BAK 更好，B 的 DNS SIG/OVRL 更高。
2. 单通道、确定性、实时学生已接近上限；继续加训练轮数/参数量只能小幅提升。
3. 下一步优先级：
   - 多麦克风空间前端（WPE/神经波束）——实时上限最高，但需要重新评估此前放弃的 WPE/MVDR；
   - 4 步 Flow/Consistency 学生（NVIDIA/SGMSE 蒸馏）——质量上限最高，但板端实时性存疑，可能只适合离线；
   - 接受 B 离线、CRM 实时双轨。
4. 未做：AEC、立体声输出、真实板端联调、流式 ONNX、SGMSE 全量教师（shard 4/5 未完成）。

---

## 11. 代码地图

```
src/
  crm_model.py          CRM v2 + ONNX core
  ulunas.py             UL-UNAS 依赖（MIT）
  distill_dataset.py    200k 训练集 Dataset
  ulunas_dataset.py     RIRNoiseDataset（数据生成）
  train_crm.py          CRM v2 DDP 训练
  train_crm_ft.py       带 --init 的微调版
  crm_gan.py            GAN 判别器/损失（实验）
  train_crm_gan.py      GAN 训练（实验）
  run_crm2_eval.py      41s/公平集推理
  metrics_fair.py       公平集 PESQ/STOI/SQUIM/DNSMOS
  metrics_full41.py     41s PESQ/STOI/DNSMOS
  metrics_real.py       实测段 DNSMOS
  metrics_real_ref.py   实测段 PESQ/STOI（相对 ref）
  export_crm2_core.py   ONNX 导出
  make_200k_data.py     200k 数据生成
  gen_ulunas_200k.py    200k UL-UNAS 输入生成
  build_200k_train.py   训练软链
scripts/
  v4_post.sh            线上 v4 链
models/
  crm2_base.onnx        CRM2Core ONNX
```

---

## 12. 复现注意事项

- 数据准备脚本里的 `<TRAIN_ROOT>/...` 与 `<RIR_ROOT>/RIRS_NOISES` 是内部服务器/WSL 路径，迁移时先改路径。
- `metrics_*` 支持 `SL_DATA_ROOT` 环境变量，默认 `./data`。
- v4 脚本可用 `FF=/path/to/ffmpeg bash scripts/v4_post.sh in.wav out.wav` 覆盖 ffmpeg。
- 训练/推理都必须保持 RMS 0.04 归一化，否则 CRM 输出会偏轻或过重。
- 比较时统一接 v4、统一响度；B 必须是 NVIDIA-only。
