#!/usr/bin/env python3
"""Export the release NVIDIA SB one-step estimator with the fixed T16 input."""
import argparse
import copy
import hashlib
import math
from pathlib import Path

import torch
from nemo.collections.audio.models.audio_to_audio import AudioToAudioModel
from torch import nn
from torch.nn import functional as F


FRAMES = 16
BINS = 256
SCALE = 1 / math.sqrt(2)
CHECKPOINT_SHA = "2490b0a61abc0f24298708cebd941fb1bb71b5bb240d0bda073c244dc965512f"


class SpatialGroupNorm(nn.Module):
    """Reduce spatial dimensions before combining the channels of each group."""

    def __init__(self, source):
        super().__init__()
        channels = source.num_channels
        channels_per_group = channels // source.num_groups
        groups = torch.arange(channels) // channels_per_group
        same_group = groups[:, None] == groups[None, :]
        self.eps = source.eps
        self.register_buffer("coefficients", same_group.float().reshape(channels, channels, 1, 1)
                             / channels_per_group)
        self.register_buffer("weight", source.weight.detach().reshape(1, -1, 1, 1))
        self.register_buffer("bias", source.bias.detach().reshape(1, -1, 1, 1))

    def group_mean(self, x):
        return F.conv2d(F.adaptive_avg_pool2d(x, 1), self.coefficients)

    def forward(self, x):
        centered = x - self.group_mean(x)
        variance = self.group_mean(centered.square())
        return centered * torch.rsqrt(variance + self.eps) * self.weight + self.bias


class Block(nn.Module):
    def __init__(self, source, condition):
        super().__init__()
        self.conv1 = source.middle_conv
        self.conv2 = source.output_block[-1]
        self.skip = getattr(source, "residual_projection", nn.Identity())
        self.norm1 = SpatialGroupNorm(source.input_block[0])
        self.norm2 = SpatialGroupNorm(source.output_block[0])
        self.act = source.act
        self.conv1.bias.add_(source.diffusion_step_projection(condition).view(-1))

    def forward(self, x):
        residual = self.conv1(self.act(self.norm1(x)))
        residual = self.conv2(self.act(self.norm2(residual)))
        return (self.skip(x) + residual) * SCALE


class StaticEstimator(nn.Module):
    def __init__(self, base):
        super().__init__()
        net = base.ncsnpp
        # The t=1 diffusion-step embedding depends on log(t) = 0.
        condition = net.time_embedding(torch.zeros(1))
        self.input_groups = nn.ModuleList()
        self.output_groups = nn.ModuleList()
        blocks = net.num_res_blocks
        for level in range(net.num_resolutions):
            start = level * blocks
            reverse = (net.num_resolutions - 1 - level) * blocks
            self.input_groups.append(nn.ModuleList(
                Block(net.input_blocks[start + i], condition) for i in range(blocks)))
            self.output_groups.append(nn.ModuleList(
                Block(net.output_blocks[reverse + i], condition) for i in range(blocks)))
        self.pyramid = nn.ModuleList()
        for conv in net.input_pyramid:
            # At the first step, the prior and the conditioning spectrum are identical.
            folded = nn.Conv2d(2, conv.out_channels, kernel_size=1)
            folded.weight.copy_(conv.weight[:, :2] + conv.weight[:, 2:])
            folded.bias.copy_(conv.bias)
            self.pyramid.append(folded)
        self.projections = net.projection_blocks
        self.output_projection = base.output_projection

    def forward(self, spectrum):
        raw = spectrum
        hidden = None
        history = []
        for level, (project, group) in enumerate(zip(self.pyramid, self.input_groups)):
            feature = project(raw)
            if hidden is not None:
                feature = feature + F.avg_pool2d(hidden, 2, 2)
            hidden = feature * SCALE
            outputs = []
            for block in group:
                hidden = block(hidden)
                outputs.append(hidden)
            history.append(outputs)
            if level + 1 < len(self.input_groups):
                raw = F.avg_pool2d(raw, 2, 2)

        image = None
        for level in reversed(range(len(self.output_groups))):
            if image is not None:
                hidden = F.interpolate(hidden, scale_factor=2, mode="bilinear", align_corners=False)
            for skip, block in zip(reversed(history[level]), self.output_groups[level]):
                hidden = block((hidden + skip) * SCALE)
            projected = F.interpolate(self.projections[level](hidden),
                                      size=spectrum.shape[-2:], mode="nearest")
            image = projected if image is None else image + projected
        return self.output_projection(image)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("onnx", type=Path)
    args = parser.parse_args()
    if hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() != CHECKPOINT_SHA:
        parser.error("The .nemo SHA256 does not match the release checkpoint")
    torch.set_num_threads(4)
    torch.manual_seed(7)
    original = AudioToAudioModel.restore_from(str(args.checkpoint), map_location="cpu").eval()
    original.estimator.ncsnpp.pad_time_to = FRAMES
    estimator = StaticEstimator(copy.deepcopy(original.estimator)).eval()
    probe = torch.randn(1, 2, BINS, FRAMES) * 0.25
    for spectrum in (probe, torch.zeros_like(probe), torch.randn_like(probe) * 0.02):
        z = torch.complex(spectrum[:, 0:1], spectrum[:, 1:2])
        estimate, _ = original.estimator(input=torch.cat([z, z], dim=1),
                                         input_length=torch.tensor([FRAMES]), condition=torch.ones(1))
        reference = torch.cat([estimate.real, estimate.imag], dim=1)
        error = torch.linalg.vector_norm(estimator(spectrum) - reference)
        error /= torch.linalg.vector_norm(reference).clamp_min(1e-8)
        if not torch.isfinite(error) or error.item() > 3e-5:
            raise RuntimeError(f"Static estimator relative error {error.item():g}")
        print(f"relative error {error.item():g}", flush=True)
    torch.onnx.export(estimator, (probe,), str(args.onnx), input_names=["spectrum"],
                      output_names=["estimate"], opset_version=17, do_constant_folding=True,
                      dynamo=False)


if __name__ == "__main__":
    main()
