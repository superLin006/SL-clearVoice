# RK3588 实时链路模型

| 文件 | 用途 | 运行环境 |
| --- | --- | --- |
| ulunas_stream_simple.onnx | UL-UNAS 流式降噪，外部保存三组模型状态 | AArch64 ONNX Runtime 1.17.1，CPU |
| nvidia-t16-spatial-norm.rknn | NVIDIA SB 一步估计器，输入 `1×2×256×16` 复数频谱 | RKNN Runtime 2.3.2，RK3588 NPU |
| [nvidia-t32-spatial-norm.rknn](nvidia-t32-spatial-norm.rknn) | 当前 T32：原权重，归一化算子优化，三核运行 | 已验收 Runtime 2.5.0，RK3588 NPU |

这里保存板端验收过的模型。部署前在本目录运行 `sha256sum -c SHA256SUMS`。

## 来源和许可

- UL-UNAS 由 Xiaobin Rong 和项目贡献者发布，流式实现由 Kailai Shen 提供。此文件为项目已验证的简化流式 ONNX 导出。保留[上游 MIT 许可](../../../src/UL-UNAS-LICENSE.txt)。[UL-UNAS 源仓库](https://github.com/Xiaobin-Rong/ul-unas)
- NVIDIA Corporation 发布 `se_der_sb_16k_small`，上游 `.nemo` 的 SHA256 为 `2490b0a61abc0f24298708cebd941fb1bb71b5bb240d0bda073c244dc965512f`。衍生模型沿用原权重，固定为第一次采样的时间条件，等价改写 GroupNorm 后转换为 RKNN FP16。NVIDIA 将模型用于研发，模型及衍生 RKNN 以 CC BY-NC-SA 4.0 提供。 [NVIDIA 模型卡](https://huggingface.co/nvidia/se_der_sb_16k_small) · [许可条款](https://creativecommons.org/licenses/by-nc-sa/4.0/)

## 重建 NVIDIA 模型

先在 NeMo 环境导出，随后在 RKNN Toolkit2 2.3.2 环境编译。构建脚本从输入文件名推导 `.rknn` 输出路径。部署使用上述已验收校验值。

~~~sh
python scripts/npu/export_nvidia_sb_live.py /path/to/se_der_sb_16k_small.nemo out/nvidia-t16-spatial-norm.onnx
python scripts/npu/build_nvidia_sb_live_rknn.py out/nvidia-t16-spatial-norm.onnx
~~~

当前 T32 的 SHA256 为 `09211b69cfa8a193a9046df334c9ab6105562887c270030bef27507a98444684`。
原检查点、单步条件与输入维度不变。此次只等价改写归一化统计和仿射计算，未训练新权重，
未启用高斯采样。逻辑输入仍是 `1×2×256×32`，运行时会查询并校验实际帧数。
旧 T32 的 SHA 为 `cfd9b689…131b6`；T16 保留历史版本，未在本轮重新编译。

数学检查、实际 FP16 张量误差和重建信息见 [模型元数据](nvidia-t32-model.json)。
流水线通过 `--nvidia-frames`、`--nvidia-history-ms`、`--nvidia-overlap-ms` 指定几何，
不再根据模型文件名猜测 T16/T32。Audio3A 与旧单麦实时入口的窗口设置分别见各自 README。

~~~sh
mkdir -p out
python scripts/npu/export_nvidia_sb_live.py --frames 32 /path/to/se_der_sb_16k_small.nemo out/nvidia-t32-spatial-norm.onnx
python scripts/npu/build_nvidia_sb_live_rknn.py out/nvidia-t32-spatial-norm.onnx
~~~

六段全长板测、两档延迟、原生50步差距和部署入口见[当前验收报告](../../../docs/reviews/2026-11-27-t32-operator-validation.md)。旧文档中的72ms单块推理属于历史测试，本轮整链NPU均值约104ms，不作为可复现指标沿用。
