# DPDFNet2 在 RK3588 上的部署

2026-11-27 已在 `172.16.40.208` 完成实际部署。音频接口为 16 kHz、单声道、S16_LE，每次处理 160 样本，即 10 ms。默认使用一个 NPU 核执行神经网络，CPU 完成 FFT、FP32 特征归一化和重叠相加。也提供 FP32 CPU 和选择性 INT8 CPU 后端。

## 已完成的优化

- 将原图 136 个分组小 Gemm 合成 8 组批量 MatMul，减少调度与拆分操作。
- 合并 9 个时序 GRU cell 的输入与状态投影，同时保留候选门的独立偏置及 reset 语义。
- 将对数、开方和带微小常量的归一化保留在 CPU FP32，神经网络使用 FP16。
- 以 FP16 回传 45424 维神经网络状态，直接绑定 RKNN 输入输出内存，按原生布局和行跨度填入数据，减少转换及逐帧分配。状态仍通过 CPU 做完整性检查并回传。
- 使用本地 C 和 FFTW，固定输入输出缓冲；持续运行的统计内存有上限。
- 比较单核与多核、密集块对角投影、分组卷积、动态 INT8 及 NPU INT8。多核及两种线性改写更慢，INT8 的音频误差明显，最终部署采用验证过的 FP16 路径。

原权重保持相同；图改写先经过 FP32 连续状态数值校验。归一化拆分最大绝对误差为 0，最终投影与门融合的 100 帧验证中，谱与状态最大绝对误差分别为 2.10e-7 和 1.19e-6。FP16 编译另有误差，需结合真实录音对照判断。

## 第三轮优化（双向频率 GRU 等价合并）

2026-11-27 第三轮默认模型已改为等价合并后的频率 GRU。每个双向 GRU 的两个方向合为一个隐藏维度翻倍的单向 GRU，输入和循环矩阵按块对角排列；反向支路只在当前帧的频率维度重排。4 个 GRU 节点由双向 64 维改为单向 128 维，原有训练参数保留，时间缓存和 45424 维状态接口相同。

新版导出阶段增加 `_merge_gru_directions.py`。80 帧连续状态 FP32 校验的频谱与状态最大绝对误差均为 0。原始布局的直接改写在 RKNN 上未保持输出；最终导出显式使用 batch-first 输入及标准 GRU 输出转置／重排，板端 3 段完整录音的 PCM SHA256 均与第二轮一致。该布局和板端数值验证是重建流程的一部分。

| 录音 | 输入时长 | 平均计算 | 计算 P95 | 超过 10 ms | 语音输出延迟 P95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 002 | 132.288 s | 6.472 ms | 6.587 ms | 11 / 13229 帧 | 66.669 ms |
| DUT 155 秒 | 155.040 s | 6.542 ms | 6.645 ms | 17 / 15504 帧 | 66.728 ms |
| 静音接语音 | 24.000 s | 6.439 ms | 6.595 ms | 2 / 2400 帧 | 66.677 ms |

以上均按实时节拍回放，单 NPU 核，CPU 限制到本板 A76 核 1–4，Runtime 2.5.0、驱动 0.9.8，维持既有自动调频策略。002 比第二轮 7.040 ms 低约 8.1%，物理 PCM 延迟 P95 减少约 0.53 ms。两段长录音最大计算耗时分别为 18.49／13.40 ms，仍有少量帧超过 10 ms；平均可跟上输入，不能据此承诺硬实时。声卡缓冲、波束、NVIDIA 与 v4 的联运行竞争另测。

交替两轮的同片段短测，旧模型平均 7.040／7.057 ms，新模型 6.521／6.440 ms，全部输出逐样本相同。RKNN 文件从 14766149 降至 12224965 字节，约小 17.2%。当前 RKNN SHA256 为 `fe2c06867a5c4d2bf2fd3f4339a76705f3df544dc716b92679c81195dc8808b6`，重建核心 ONNX SHA256 为 `a4995e480f8a030dedc96e56d7f2d1c4670bab15721d353f46bb5d9a5fa11b79`。

本轮还测试了新版编译、单核加权重压缩、CPU／GPU GRU、移除 I/O 重排、卷积与分组投影混合 INT8、SRAM 和 GPU 优先标志。未满足速度及音频一致性的候选保存在本地实验目录，未进入默认模型。混合量化需同时保护状态出口和状态缓存，单独保护部分计算层不能保证循环反馈精度。

单核固定在 A76 核 1，短测将该核簇的频率下限临时设为硬件允许的 2.208 GHz，旧模型约从 6.9 降到 6.7 ms。该设置影响整个核簇且会提高功耗，实验结束已恢复 1.2 GHz 下限；默认运行包保持系统既有频率策略。本板进程级 uclamp 不受支持。

实验、校验和性能记录位于本地 `out/dpdfnet-rk3588-20261127/npu-round3/`。CPU FP32 与选择性 INT8 模型及运行参数继续可用。

## 第二轮优化（Runtime 2.5.0 与选择性 CPU INT8）

同日继续验证后，运行包默认携带独立 RKNN Runtime 2.5.0。RKNN 模型仍由 Toolkit2 2.3.2 编译，权重和图与首轮相同。新运行库在 002、DUT 和静音接语音三段完整录音中的 PCM 均与 2.3.2 逐样本一致。驱动继续使用 0.9.8。

| 录音 | 条件 | NPU 平均计算 | 计算 P95 | 超过 10 ms | 语音输出延迟 P95 |
| --- | --- | ---: | ---: | ---: | ---: |
| 002，132.288 秒 | 实时节拍 | 7.040 ms | 7.111 ms | 9 / 13229 帧 | 67.196 ms |
| DUT，155.040 秒 | 连续满速 | 6.983 ms | 7.079 ms | 4 / 15504 帧 | 未测 |
| 静音接语音，24 秒 | 连续满速 | 6.989 ms | 7.047 ms | 0 / 2400 帧 | 未测 |

002 相较首轮同条件的 7.787 ms 降低约 9.6%。本次仅加载新运行库，尚未用 Toolkit2 2.5 重新编译模型。官方该版本更新包含 GRU、图融合和多输入输出处理改进，不能据此预先承诺其他模型也有相同收益。来源为 [官方 v2.5.0](https://github.com/airockchip/rknn-toolkit2/releases/tag/v2.5.0)。

CPU 新增 `--cpu-int8`。只量化 5 个大 GRU 输入投影，采用动态激活量化和逐通道 INT8 权重；循环状态投影、原生频率 GRU、归一化、其他矩阵和卷积保留 FP32。这是混合精度模型。旧版大范围动态 INT8 短测为 3.26 ms，但对 FP32 的差分 SNR 只有 6.94 dB，最终选择保留敏感运算精度的新方案。

同一 8 秒片段、实时节拍、A76 `1-4` 和单线程的本轮测试中，前后两次 FP32 平均为 4.531／4.806 ms，选择性 INT8 为 3.994 ms，均无超过 10 ms 的帧。连续满速的完整 002 和 DUT 分别为 2.885 和 2.884 ms；首轮 FP32 同类满速测试为 3.05–3.10 ms。不同节拍和调频条件的数值需分别比较。

追加同包复测时，实时节拍 FP32／INT8 为 4.859／4.749 ms，收益比第一轮短测小。固定到单个 A76 核 1、连续满速交替运行两轮，FP32 为 3.050／3.054 ms，INT8 为 2.921／2.913 ms，平均加速约 4.4%。因此目前可确认 CPU INT8 具有小幅计算收益，不能承诺稳定的 15% 实时延迟降低。调频和调度可能影响实时节拍结果，功耗尚未测量。

| 录音 | 与 FP32 差分 SNR | 波形相关系数 | FP32／选择性 INT8 的 DNSMOS OVRL |
| --- | ---: | ---: | ---: |
| 002 | 31.85 dB | 0.999678 | 1.976／1.966 |
| DUT 155 秒 | 43.17 dB | 0.999976 | 2.752／2.751 |

静音段输出仍为数字零。自动评分接近，但 INT8 并非逐样本相同，需结合同响度试听选择。完整记录、频谱和试听位于本地 `out/dpdfnet-rk3588-20261127/quantization-followup/`。

```sh
# 先执行常规模型导出，再增加可选 CPU INT8 模型。
python scripts/dpdfnet/export_cpu_int8.py \
    models/dpdfnet/dpdfnet2-cpu.onnx models/dpdfnet

# 此参数仅向运行包复制独立库；不修改目标板系统库。
DPDFNET_RKNN_RUNTIME=/path/to/runtime-2.5.0/librknnrt.so \
    bash scripts/dpdfnet/build.sh models/dpdfnet out/dpdfnet-bundle-v2

# 板端实时节拍验证选择性 CPU INT8。
DPDFNET_CPU_AFFINITY=1-4 ./run.sh --cpu-int8 --pace \
    < input.s16le > output-int8.s16le 2> timing-int8.log
```

测试库取自官方 v2.5.0 的 `rknpu2/runtime/Linux/librknn_api/aarch64/librknnrt.so`，SHA256 为 `4b6cc575bedf8952b13db4b1190fa3b3d2ac213f00be774ff6ccd19d651970c0`。包内 `lib/` 优先作用于当前进程，其他应用各自加载其运行库。

追加剖析中，4 个频率 GRU 被 Runtime 2.5 标为 NPU_HYBRID，合计占剖析模式时间约 26.4%；转置、重排和拼接也占有明显比例。该模式有采集开销，不能把其 12.134 ms 总耗时当作正常推理延迟。后续应优先验证 GRU 内部实现、布局与混合 INT8／FP16，当前尚未验证这些改法能进一步提速。

## 首轮板测（Runtime 2.3.2）

原始神经核心直接转换、由 CPU 处理归一化的版本，8 秒 PCM 短测每帧平均约 10.23 ms。最终融合与内存绑定版本相同片段约 7.83 ms，平均计算开销下降约 23.5%。完整录音复测如下，输入 hop 预算为 10 ms。

| 录音 | 时长 | NPU 平均计算 | 计算 P95 | 超过 10 ms | 语音输出延迟 P95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 002 | 132.288 s | 7.787 ms | 7.959 ms | 11 / 13229 帧 | 68.047 ms |
| DUT 新录音 | 155.040 s | 7.758 ms | 7.931 ms | 4 / 15504 帧 | 68.017 ms |
| 20 秒静音后接语音 | 24.000 s | 7.768 ms | 7.991 ms | 3 / 2400 帧 | 68.074 ms |

两段长录音最大计算耗时分别为 18.69 和 22.29 ms，最大语音输出延迟分别为 83.17 和 82.39 ms。能够持续跟上输入，调度抖动仍需业务缓冲处理。

FP32 CPU 后端连续满速处理的平均计算为 3.05–3.10 ms／帧；8 秒实时节拍短测平均 4.229 ms、计算 P95 5.223 ms、语音输出延迟 P95 65.318 ms，未出现超过 10 ms 的帧。现有调频策略及调度会影响这些数值。CPU 后端可给后续 NVIDIA 模型留出全部 NPU 资源。详细测量和音频见本地 `out/dpdfnet-rk3588-20261127/aligned-results/`。

最终内存绑定版本与融合但未绑定内存的版本，在三段完整录音中 PCM 均逐样本一致。DUT 的 NPU／FP32 CPU 波形相关系数为 0.999957、差分 SNR 为 40.67 dB，DNSMOS OVRL 为 2.760／2.752；002 对应相关系数为 0.994799、差分 SNR 为 19.84 dB，DNSMOS OVRL 为 1.996／1.976。分数接近支持部署前后效果基本保留，但听感仍需确认。

另测仅将状态输出，或将状态输入和输出都改为紧凑 4D 布局。原生状态输出占用从 726784 字节降至 90848 字节，短测却分别增加到 8.46 和 8.48 ms／帧，因此最终模型保持已验证的原布局。

实时测试采用 PCM 按 10 ms 节拍输入，包含 FFT、特征计算、神经网络、状态检查、合成和 PCM 写出。单帧耗时排除模型加载、预热和 EOF 冲刷。测试时只运行该模型。本板 `1-4` 是 A76，其他镜像需重新确认 CPU 编号。保持板卡现有频率策略，单 NPU 核，不更改全局调度配置。RKNN Runtime 为 2.3.2，NPU 驱动为 0.9.8。

真实语音延迟约 68 ms，包含模型 40 ms 谱缓存及 STFT 启动与帧聚合。日志 `p95_pcm_ms` 按已补偿的音频样本时间戳计算；`p95_service_ms` 只表示一次计算开销。`--pace` 才能测量 PCM 节拍延迟，未按节拍运行时对应延迟字段为 0。声卡和业务缓冲仍需单独测量。统计分位数最多保留最近 65536 帧，均值、最大值和超时计数覆盖本次运行。

NPU 平均耗时小于 10 ms，长音频中仍观测到少量超过 10 ms 的帧，应通过业务缓冲吸收调度抖动。当前结果覆盖单模型，尚未验证与 NVIDIA T32 同时使用 NPU 时的完整链路。

## 板端运行

已部署位置 `/userdata/sl-clearvoice-dpdfnet2-20261127/bundle`。运行优先加载包内 `lib/librknnrt.so`，ONNX Runtime 和其他动态依赖来自目标板的 `/userdata/opt/hdmeeting/lib`，FFTW 已静态链接。更换运行库时要重新板测。

```sh
cd /userdata/sl-clearvoice-dpdfnet2-20261127/bundle
./run.sh --verify-models

# 默认单核 NPU。输出是裸 PCM，日志写 stderr。
DPDFNET_CPU_AFFINITY=1-4 ./run.sh < input.s16le > output.s16le 2> timing.log

# 文件模拟实时输入，测量实际 PCM 延迟。
DPDFNET_CPU_AFFINITY=1-4 ./run.sh --pace < input.s16le > output.s16le 2> timing.log

# FP32 CPU，默认一个线程。
DPDFNET_CPU_AFFINITY=1-4 ./run.sh --cpu < input.s16le > output-cpu.s16le 2> timing-cpu.log
```

真实麦克风管道直接按采样节拍送入 PCM，不额外添加 `--pace`。例如声卡参数确认后可接入 `arecord -t raw -f S16_LE -r 16000 -c 1`。多麦和回采必须先通过同事的前端取得单声道结果，不能把十通道交织数据直接送入模型。

可选环境变量 `DPDFNET_NPU_CORES`（核掩码，默认 1）、`DPDFNET_CPU_THREADS`（默认 1）、`DPDFNET_CPU_AFFINITY`。线程数和多核掩码已做短测，没有发现更好的部署组合。关闭标准输出的终端打印是必要的，音频数据只出现在 stdout，第三方库的打印已重定向至 stderr。

每个进程启动一次模型并跨帧保留状态。不要每帧启动进程；换流需要重置状态，目前通过重启进程实现。EOF 会自动冲刷模型缓存，输出长度精确匹配输入，包括少于一个 hop 的片段。不完整的 S16_LE 样本及非有限谱／状态会导致明确失败。

## 从源码导出和构建

```sh
# 不加 --rknn 可先在普通 ONNX 环境重现图改写及 FP32 校验。
python scripts/dpdfnet/export_model.py \
    models/dpdfnet/dpdfnet2.onnx out/dpdfnet-export --rknn
```

模型导出仅接受已验证的原始 SHA256，避免误用其他采样率或状态排列的权重。产物含完整 ONNX 图、两个 CPU 模型、RKNN、状态资产、校验文件和数值验证记录。

交叉构建需要 AArch64 编译器、RKNN 和 ONNX Runtime 头文件及板端库、FFTW 单精度头文件与静态库。

```sh
export AARCH64_CC=/path/to/aarch64-linux-gnu-gcc
export RKNN_INCLUDE=/path/to/rknn/include
export RKNN_LIBDIR=/path/to/aarch64/rknn/lib
export ORT_INCLUDE=/path/to/onnxruntime/include
export ORT_LIBDIR=/path/to/aarch64/onnxruntime/lib
export FFTW_INCLUDE=/path/to/fftw/include
export FFTW_STATIC=/path/to/aarch64/libfftw3f.a
bash scripts/dpdfnet/build.sh models/dpdfnet out/dpdfnet-bundle
tar -cf out/dpdfnet-bundle.tar -C out dpdfnet-bundle
```

目标板 BusyBox tar 未支持 gzip，传输时使用未压缩 tar。运行包仅含优化 CPU 模型和 NPU 模型，原始参照 ONNX 留在源码目录。

## 音质验证范围

采用现有 002 录音第 1 通道和同事 155.04 秒 `dut_raw10ch.wav` 第 4 通道，后者是 8 麦加 2 回采输入中的麦克风，未取回采声道。它与此前 105 秒 DUT 不是同一个文件。本次验证只运行 DPDFNet2，不接波束、NVIDIA 或 v4。

另测 20 秒数字静音后接语音、空输入及 1／159／160／161／321 样本短输入。持续静音输出为数字零，状态正常恢复。所有输出样本数与输入相同，尾部冲刷完成。

频谱图统一时间轴、色标与 LUFS；同时提供原电平与同响度音频。DNSMOS 为自动估计，短窗统一 RMS 后对全段取平均，只用于比较部署前后的变化，不能代替会议场景试听。NPU 与 CPU 的差分 SNR 反映部署误差，不表示降噪提升。
