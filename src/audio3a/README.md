# Audio3A 整链构建与回放

当前可独立构建 DPDFNet2 / RNNoise 入口。当前T32含归一化算子优化，权重不变，模型与校验值见 `models/rknn/live/`。需要 Python 3、C 编译器、ONNX Runtime C 头文件和目标平台共享库；编译器与库必须属于同一架构。

```sh
python3 scripts/board/build_audio3a.py /path/to/bundle \
  --cc /path/to/aarch64-linux-gnu-gcc \
  --ort-include /path/to/include --ort-lib /path/to/aarch64/lib
```

构建产生 `bin/p2p_dut_node` 和 `BUILD.json`，记录源码、头文件与二进制 SHA256。模型和库：

```text
bundle/bin/p2p_dut_node
bundle/bin/nvidia_sb_pcm
bundle/models/dpdfnet2.onnx
bundle/models/nvidia-t32-spatial-norm.rknn
bundle/lib/libonnxruntime.so
bundle/lib/librknnrt.so
```

NVIDIA 程序由 `scripts/board/build_live_pcm.sh` 构建；模型分别来自 `models/dpdfnet/`、`models/rknn/live/`。现有 DFNet3 / Sherpa 代码受 `AUDIO3A_OPTIONAL_DENOISERS` 控制，不属于此构建，需要另行提供实现与依赖。明确请求的 DPDF 模型缺失时会失败。

## 两种处理顺序

离线回放不打开声卡：

```sh
sh scripts/board/replay_audio3a.sh /absolute/bundle input.raw10ch.wav output.wav --dump-stages /absolute/results
sh scripts/board/replay_audio3a.sh /absolute/bundle input.raw10ch.wav reverse.wav --denoise-after-nvidia --dump-stages /absolute/reverse
```

默认保留 DPDFNet2 → AGC②/门 → NVIDIA → AGC③。反序保留相同前级、AGC 参数及模型，将 DPDF 移到 NVIDIA 输出线程，在 AGC③之前处理。`13_post_nvidia_ans.wav` 是反序新增节点。前级到 `07_transient_xover.wav` 必须逐样本相同。

接口仍是 PCM16，模型精度未改变。两种顺序皆无高斯采样、dry=0。T32 提供两档参数：默认 `AUDIO3A_NVIDIA_PROFILE=quality` 使用48ms历史/48ms交叠/152ms推进；
`balanced` 使用64ms历史/48ms交叠/136ms推进。两档使用同一原权重T32模型，分别侧重保真和延迟。CPU 亲和性按 sysfs capacity 选择，也可用 `AUDIO3A_CPU_AFFINITY=1-4` 指定。

T16/T32 由 `--nvidia-frames 16|32` 显式选择，默认32，运行器会校验实际模型帧数。
`--nvidia-history-ms`、`--nvidia-overlap-ms` 可覆盖窗口裁剪；数值必须为8ms整数倍，
且历史与两倍交叠之和不超过模型窗口。模型文件名不再参与参数选择。

加 `--bypass-post-nvidia-agc` 可旁路 AGC③，保留前两级；`--post-nvidia-agc` 恢复。
同一命令后出现的选项优先。两档均支持此选项，模型与前级不变。

```sh
AUDIO3A_NVIDIA_PROFILE=balanced sh scripts/board/replay_audio3a.sh /absolute/bundle input.raw10ch.wav output.wav --pace --bypass-post-nvidia-agc
```

脚本未接入 v4。`scripts/v4_post.sh` 是离线参考；`scripts/board/v4_live_pcm.c` 是另一套实时后级，二者不能视为逐样本相同。

## 输入电平和计时

原始文件不改写。本轮T32算子验收用 prepare_meeting_inputs.py 生成已记录SHA的副本；步骤见[验收报告](../../docs/reviews/2026-11-27-t32-operator-validation.md)。此前顺序实验中，meeting_3、4 的派生测试用 `--mic-pregain 43.96` / `69.12`，只放大麦克风；原电平测试另记。不要覆盖 `data/raw`。

加 `--pace` 按 16 ms 节拍回放。`availability_p95_ms` 表示按输出样本索引计算的可用时间，不包含声卡或网络；DPDF/NeuralRES 的信号延迟还需对齐，不能视为声学端到端延迟。帧内 DSP 计时不含 NVIDIA 异步工作，整链吞吐要同时查看墙钟和 NVIDIA 日志。

离线输入按 256 样本帧处理；测试集均整帧。现有 DPDF 流程没有单独冲洗/补偿内部延迟，文件尾部需在评测中明确。比较应对齐并裁成相同有效区间，不能把填补的零参与评分。

## 回归

本地原生构建可验证线程归属、前级一致、帧数和错误处理：

```sh
python tests/check_chain_replay.py /path/to/native/bin/p2p_dut_node --model models/dpdfnet/dpdfnet2-cpu.onnx
```

该检查用恒等管道替代 NVIDIA，不评价 NVIDIA 效果和速度；质量和 NPU 性能另用板端实测。

本轮验收固定原始DPDF模型 dpdfnet2.onnx（ce35d602…487a9），默认回放脚本使用此文件。AUDIO3A_DPDF_MODEL=/absolute/other.onnx 可显式选择其他图，性能与音质需另验。两档全长结果及正式二进制检查见[本轮验收](../../docs/reviews/2026-11-27-t32-operator-validation.md)。
