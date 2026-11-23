# RK3588 实时链路模型

| 文件 | 用途 | 运行环境 |
| --- | --- | --- |
| ulunas_stream_simple.onnx | UL-UNAS 流式降噪，外部保存三组模型状态 | AArch64 ONNX Runtime 1.17.1，CPU |
| nvidia-t16-spatial-norm.rknn | NVIDIA SB 一步估计器，输入 `1×2×256×16` 复数频谱 | RKNN Runtime 2.3.2，RK3588 NPU |
| nvidia-t32-spatial-norm.rknn（按需构建） | 同模型的 32 帧输入，用于三核音质配置 | RKNN Runtime 2.3.2，RK3588 NPU |

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

T32 使用同一检查点和构建工具。板测所用 T32 文件的 SHA256 为 `cfd9b689b665754a06057836eab11f0659226b60da7331aa245b2f06965131b6`。构建运行包时会为传入的模型生成校验值。

~~~sh
mkdir -p out
python scripts/npu/export_nvidia_sb_live.py --frames 32 /path/to/se_der_sb_16k_small.nemo out/nvidia-t32-spatial-norm.onnx
python scripts/npu/build_nvidia_sb_live_rknn.py out/nvidia-t32-spatial-norm.onnx
~~~
