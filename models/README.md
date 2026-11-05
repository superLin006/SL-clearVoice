# models

## crm2_base.onnx

- 来源：`<TRAIN_ROOT>/ulunas_train_code/ckpt_crm_200k/crm_ep7.pt`
- SHA256：`61a7d8ce0958df653ba65bc834265d8c4a5b863f2aace20e9e40e15562d5402a`
- 输入：`spec`，float32，`[1, 2, T, 257]`，STFT real/imag（16 kHz，n_fft=512，hop=256，Hann，center=True）
- 输出：`enh`，float32，`[1, 2, T, 257]`，增强后的复数谱
- 动态轴：T 动态；F 固定 257
- 参数量：950,915
- 说明：不包含 STFT/iSTFT；板端复用 UL-UNAS 已有的 STFT/iSTFT。当前为整段核心，流式 cache 尚未导出。

## checkpoint

基座 checkpoint 未放入 git，位置：

```
<TRAIN_ROOT>/ulunas_train_code/ckpt_crm_200k/crm_ep7.pt
```

SHA256：

```
a8159a429190efe53d6d99db99c089e31cfe864c36d44023f519f7f4cde392dd
```
