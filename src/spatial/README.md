# 多麦空间链路（WPE + DS + UL-UNAS + CRM v2）

在 1/3/5 m 实测段上，这条链路第一次在各距离都超过单通道 B 的 PESQ/STOI/BAK：

```
6ch WPE（nara_wpe, taps=10, delay=3, iterations=3）
  -> 延迟求和波束（R=42 mm, DOA=90°）
  -> UL-UNAS
  -> CRM v2
  -> v4
```

## 结果（均接 v4，相对 seg_*_ref.wav）

| 方法 | PESQ | STOI | DNS OVRL/SIG/BAK |
|---|---|---|---|
| B（单通道 UL-UNAS→NVIDIA） | 1.309 | 0.740 | 2.624 / 3.532 / 2.791 |
| 单通道 CRM v2 | 1.380 | 0.758 | 2.387 / 2.753 / 3.481 |
| DS + UL-UNAS + CRM（无 WPE） | 1.409 | 0.757 | 2.459 / 2.824 / 3.582 |
| **WPE + DS + UL-UNAS + CRM** | **1.559** | **0.782** | **2.755** / 3.097 / **3.771** |

分距离 PESQ：1 m 1.861（B 1.483）、3 m 1.475（B 1.265）、5 m 1.340（B 1.181）。B 的 SIG 仍更高（3.53 vs 3.10），但多麦链的 OVRL/BAK 更好。

## 关键点 / 坑

- `nara_wpe` 的 STFT 输出是 `(M,T,F)`，WPE 需要 `(F,M,T)`；旧代码轴序错了，导致 WPE 输出与输入几乎不相关。这里已修正。
- WPE 输出与输入时间对齐，**不要额外 shift**。
- 阵列半径约 42 mm，6 麦，DOA≈90°；用 `fit_geometry` 在真实段上拟合得到。
- WPE 是离线整段迭代；实时部署需要改成 block/online WPE，并重新测 RTF。
- 评估集只有 1/3/5 m 三段，需再用 8ch 41s/更多真实房间验证。

## 用法

```bash
python run_spatial_chain.py \
  --input seg_1m_raw_8ch.wav --output out.wav \
  --theta 90 --radius 0.042 \
  --ulunas-root /path/to/ulunas \
  --ulunas-ckpt /path/to/model_trained_on_dns3.tar \
  --crm-ckpt /path/to/crm_ep7.pt
```
