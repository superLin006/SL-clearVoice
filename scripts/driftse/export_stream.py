"""Export an 8-frame causal DriftSE core with explicit streaming state."""

import argparse
from io import BytesIO
import math
from pathlib import Path
import sys

import onnx
import torch
from torch import nn
from torch.nn import functional as F

from onnx_graph import rewrite_projection, separate_states


class StreamingSpectralCore(nn.Module):
    def __init__(self, net, frames=8, context=16, bins=256):
        super().__init__()
        self.net = net
        self.frames = frames
        self.context = context
        self.bins = bins
        self.conv_history = net.t_ksize - 1
        self.state_shapes = [(1, 4, self.conv_history, bins),
                             (1, net.deconv.out_channels, self.conv_history, bins), (1,)]
        for block in net.blocks:
            padded_bins = bins + 2 * block.olp
            hidden = block.inter_rnn.hidden_size
            c = block.emb_dim
            h = block.n_head
            qk = block.attn_conv_Q.out_channels // h
            shapes = [(padded_bins, c, block.emb_ks - 1), (1, padded_bins, hidden),
                      (1, padded_bins, hidden), (h, context - 1, qk * bins),
                      (h, context - 1, c // h * bins)]
            self.state_shapes.extend(shapes)
        self.state_sizes = [math.prod(s) for s in self.state_shapes]
        self.register_buffer("key_pos", torch.arange(1 - context, frames))
        self.register_buffer("query_pos", torch.arange(frames).reshape(-1, 1))

    def initial_state(self):
        return torch.zeros(1, sum(self.state_sizes))

    def forward(self, spectrum, state):
        parts = [part.reshape(shape) for part, shape in zip(
            state.split(self.state_sizes, dim=1), self.state_shapes
        )]
        conv_tail, deconv_tail, valid = parts[:3]
        conv_input = torch.cat((conv_tail, spectrum), dim=2)
        x = self.net.conv(conv_input)
        age = self.query_pos - self.key_pos
        masked = (age < 0) | (age >= self.context) | (self.key_pos < -valid)
        next_block_states = []
        for index, block in enumerate(self.net.blocks):
            begin = 3 + index * 5
            x, next_parts = self.grid_block(block, x, masked, parts[begin:begin + 5])
            next_block_states.extend(next_parts)

        # Carry the convolution contributions that land in the next chunk.
        deconv = self.net.deconv
        decoded = F.conv_transpose2d(x, deconv.weight, padding=deconv.padding)
        decoded = decoded + F.pad(deconv_tail, (0, 0, 0, self.frames))
        estimate = decoded[:, :, :self.frames] + deconv.bias.reshape(1, -1, 1, 1)
        next_parts = [
            conv_input[:, :, -self.conv_history:],
            decoded[:, :, self.frames:],
            (valid + self.frames).clamp(max=self.context - 1),
            *next_block_states,
        ]
        return estimate, torch.cat([part.reshape(1, -1) for part in next_parts], dim=1)

    def grid_block(self, block, x, masked, state):
        norm_tail, hidden, cell, keys, values = state
        c, t, f = block.emb_dim, self.frames, self.bins
        pad, width = block.olp, block.emb_ks
        padded_bins = f + 2 * pad
        x = F.pad(x.permute(0, 2, 3, 1), (0, 0, pad, pad))
        frequency = block.intra_norm(x).reshape(t, padded_bins, c).transpose(1, 2)
        frequency = F.unfold(frequency[..., None], (width, 1)).transpose(1, 2)
        frequency, _ = block.intra_rnn(frequency)
        frequency = block.intra_linear(frequency.transpose(1, 2))
        x = x + frequency.reshape(1, t, c, padded_bins).transpose(2, 3)

        x = x.transpose(1, 2)
        temporal = block.inter_norm(x).permute(0, 1, 3, 2).reshape(padded_bins, c, t)
        temporal = torch.cat((norm_tail, temporal), dim=2)
        sequence = F.unfold(temporal[..., None], (width, 1)).transpose(1, 2)
        sequence, (next_hidden, next_cell) = block.inter_rnn(sequence, (hidden, cell))
        x = x + block.inter_linear(sequence).reshape(1, padded_bins, t, c)
        x = x[:, pad:pad + f].permute(0, 3, 2, 1).contiguous()

        q = block.attn_norm_Q(block.attn_conv_Q(x))
        k = block.attn_norm_K(block.attn_conv_K(x))
        v = block.attn_norm_V(block.attn_conv_V(x))
        q = q[0].transpose(1, 2).flatten(start_dim=2)
        k = torch.cat((keys, k[0].transpose(1, 2).flatten(start_dim=2)), dim=1)
        v = torch.cat((values, v[0].transpose(1, 2).flatten(start_dim=2)), dim=1)
        attention = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(q.shape[-1])
        attention = torch.softmax(attention.masked_fill(masked, float("-inf")), dim=2)
        attended = torch.matmul(attention, v).reshape(block.n_head, t, c // block.n_head, f)
        attended = attended.transpose(1, 2).reshape(1, c, t, f)
        next_state = [temporal[:, :, -(width - 1):], next_hidden, next_cell,
                      k[:, 1 - self.context:], v[:, 1 - self.context:]]
        return x + block.attn_concat_proj(attended), next_state


def load_backbone(upstream, checkpoint):
    sys.path.insert(0, str(upstream))
    sys.path.insert(0, str(upstream / "backbones"))
    from TFGridNet_Causal import TFGridNet_Causal
    from util.config_loader import load_config

    config = load_config(upstream / "config/DriftSE_reverb/experiments"
                         / "Reverb_EARS_wavcubepro_tfgrid_causal_incond.json")
    backbone = TFGridNet_Causal(
        n_layers=config["n_layers"],
        emb_dim=config["emb_dim"],
        lstm_hidden_units=config["lstm_hidden_units"],
        attn_n_head=config["attn_n_head"],
        attn_qk_output_channel=config.get("attn_qk_output_channel", 4),
    )
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    backbone.load_state_dict(saved["ema"], strict=True)
    return backbone.eval()


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True, help="DriftSE source checkout")
    parser.add_argument("--checkpoint", type=Path, required=True, help="EARS causal WavCube checkpoint")
    parser.add_argument("--output", type=Path, required=True, help="Directory for FP32 and INT8 ONNX cores")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    core = StreamingSpectralCore(load_backbone(args.upstream.resolve(), args.checkpoint)).eval()
    packed = BytesIO()
    spectrum = torch.zeros(1, 4, core.frames, core.bins)
    torch.onnx.export(
        core, (spectrum, core.initial_state()), packed,
        input_names=["spectrum", "state"], output_names=["estimate", "next_state"],
        opset_version=17, do_constant_folding=True,
    )

    model = onnx.load_model_from_string(packed.getvalue())
    separate_states(model)
    rewrite_projection(model)
    onnx.checker.check_model(model)
    fp32 = args.output / "driftse-stream-t8.onnx"
    onnx.save(model, fp32)

    from onnxruntime.quantization import quantize_dynamic, QuantType
    int8 = args.output / "driftse-stream-t8-int8.onnx"
    quantize_dynamic(fp32, int8, op_types_to_quantize=["LSTM"], weight_type=QuantType.QInt8)
    onnx.checker.check_model(str(int8))
    print(fp32)
    print(int8)


if __name__ == "__main__":
    main()
