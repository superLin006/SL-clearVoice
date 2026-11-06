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


## Online WPE / 板端 RTF（2026-11-05）

| 处理（RK3588, ORT 1.18, 4 线程） | RTF |
|---|---|
| 6ch online WPE（纯 Python/NumPy） | 1.576 |
| 6ch online WPE + DS | 1.583 |
| DS 波束 | 0.007 |
| DS + 单通道 online WPE | 0.06 |
| UL-UNAS streaming | 0.141 |
| CRM2 core | 0.493 |
| **DS + mono online WPE + UL-UNAS + CRM2（合计估计）** | **约 0.69** |

- 6ch online WPE 在板端 CPU 上不能实时；6ch + 下游整链约 2.2 RTF。
- 实时推荐：`DS 波束 -> 单通道 online WPE -> UL-UNAS -> CRM2`，板端合计约 0.69 RTF。
- 质量对比（41s / 实测 1/3/5m，均接 v4）：
  - 41s：离线 6ch WPE 链 1.472/0.776/2.72/3.00/3.91；online 1.408/0.766/2.56/2.88/3.77；B 1.468/0.776/2.82/3.21/3.72。
  - 1/3/5m：离线 6ch WPE 链 1.559/0.782/2.755/3.097/3.771；online 1.374/0.749/2.351/2.685/3.574；B 1.309/0.740/2.624/3.532/2.791。
- 结论：离线 6ch WPE 质量最好；实时走 DS+mono online WPE，质量优于单通道 CRM、部分指标仍低于 B。若要把 6ch online WPE 做进实时，需要用 C/C++ 重写并优化矩阵运算。


## C++ 6ch online WPE（2026-11-05）

源码：`wpe_online_cpp.cpp`；构建：`scripts/build_wpe_online_board.sh`。

- RK3588 板端实测（11.6 s 片段）：
  - 单线程：5.78 s，RTF 0.307
  - OpenMP 4 线程：1.19 s，RTF 0.102
  - 加 UL-UNAS 0.141、CRM2 0.493，整链约 **0.73 RTF**
- 41s 输出质量（接 UL-UNAS+CRM+v4）：1.425/0.766/2.55/2.84/3.81；离线 6ch WPE 为 1.472/0.776/2.72/3.00/3.91。
- 单麦/坏通道：程序检测 `channels<6` 或任一通道 RMS 过低时，直接输出第一通道并跳过 WPE/DS；产品侧应把这种情况路由到单通道 `UL-UNAS→CRM→v4`，不要复制单麦成 6 路。
- 注意：online WPE 是因果算法，短片段质量低于离线 WPE；离线文件后处理仍建议用 6ch 离线链。


## 8 麦阵列（2026-11-06 验证）

- RK3588 8 麦硬件当前 7/8 路损坏，0–5 路有效；旧 41s 文件也是 7/8 静音，但最佳拟合是 6 麦 60° 几何。
- 新会议室 3 个文件按 **8 麦 45° 圆周、仅 0–5 路有效** 拟合明显更好：
  - 001/002 通道顺序 `[0,1,5,4,2,3]`，半径 48/55 mm，DOA 294/301°；
  - 000 较短/噪声不同，顺序/DOA 会变，需要每次会话自动标定。
- C++ 现在支持：`./wpe_online_cpp in.f32 out.f32 6 <npos> <order_csv> <theta> <radius>`
  - 6 麦旧设备：`6 6 0,1,2,3,4,5 90 0.042`
  - 8 麦新设备示例：`6 8 0,1,5,4,2,3 294 0.0476`
- 注意：8 麦几何不能套用旧 6 麦 60° 波束；旧 41s 用 8 麦几何反而变差。
