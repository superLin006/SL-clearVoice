#!/usr/bin/env python3
"""Run board-matched NVIDIA T32 and DPDFNet2 on mono 16 kHz stage audio.

The shared runners reproduce streaming transforms, normalization, initial
states and integer PCM boundaries. Energy/tail values use per-output masks and
are descriptive proxies, not SNR, DRR or clean-reference fidelity scores.
"""
import argparse
from pathlib import Path
import nvidia_runner
import os

import numpy as np
import soundfile as sf
from speechmos import dnsmos

SR = 16000
REPO = Path(__file__).resolve().parents[2]
MODEL = os.environ.get("NVIDIA_T32_ONNX", str(REPO / "models/onnx/nvidia-t32-spatial-norm.onnx"))
DPDF = str(REPO / "models/dpdfnet/dpdfnet2.onnx")


def load_runner():
    return nvidia_runner


def t32_process(x, history_ms=48.0, overlap_ms=48.0, dry_gain=0.0, frames=32, model=None):
    """Run the board's T32 profile over a float signal."""
    m = load_runner()
    est = m.OnnxEstimator(model or MODEL, 8)
    enh = m.Pipeline(frames, m.hann_window(), est, 0.0)   # sigma 0 = no Gaussian term
    pcm = np.clip(np.rint(x * 32768.0), -32768, 32767).astype(np.int16)
    y, _, _ = enh.process(pcm, int(history_ms * 16), int(overlap_ms * 16), dry_gain)
    return np.asarray(y, dtype=np.float64) / 32768.0


def dpdf_process(x):
    """Use the board-matched window, model state and 256-sample FIFO."""
    from dpdf_runner import process
    return process(x)


def tail_gap(x):
    n = len(x) // 160
    e = 10 * np.log10(np.array([np.mean(x[i * 160:(i + 1) * 160] ** 2)
                                for i in range(n)]) + 1e-12)
    sp = e > np.percentile(e, 70)
    d = []
    for i in range(1, len(e) - 25):
        if sp[i - 1] and not sp[i] and not sp[i:i + 25].any():
            d.append(e[i - 1] - np.mean(e[i + 2:i + 18]))
    return float(np.median(d)) if d else float('nan')


def noise_floor(x):
    n = len(x) // 160
    e = 10 * np.log10(np.array([np.mean(x[i * 160:(i + 1) * 160] ** 2)
                                for i in range(n)]) + 1e-12)
    sp = e > np.percentile(e, 70)
    return float(np.median(e[sp]) - np.median(e[~sp]))


def dns(x):
    x = np.asarray(x, dtype=np.float32).copy()
    x -= x.mean()
    rms = float(np.sqrt(np.mean(x * x)))
    if rms < 1e-12:
        return float('nan'), float('nan')
    x = np.clip(x * (.03 / rms), -1, 1)
    r = dnsmos.run(x, sr=SR)
    return r['sig_mos'], r['bak_mos']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--model", default=MODEL, help="Exported NVIDIA T32 ONNX model")
    args = parser.parse_args()
    for p in args.inputs:
        x, sr = sf.read(p)
        if sr != SR or x.ndim != 1:
            parser.error(f"{p}: expected mono 16 kHz stage WAV")
        x = x.astype(np.float64)
        print(f'\n=== {Path(p).name} ===')
        print(f'{"处理":>20s}{"SIG":>8s}{"BAK":>8s}{"能量落差":>9s}{"尾音代理":>9s}')
        s, b = dns(x)
        print(f'{"输入":>20s}{s:>8.3f}{b:>8.3f}{noise_floor(x):>9.1f}{tail_gap(x):>9.1f}')

        d = dpdf_process(x)
        s, b = dns(d)
        print(f'{"DPDFNet2":>20s}{s:>8.3f}{b:>8.3f}{noise_floor(d):>9.1f}{tail_gap(d):>9.1f}')

        n = t32_process(x, model=args.model)
        s, b = dns(n)
        print(f'{"T32 单独":>20s}{s:>8.3f}{b:>8.3f}{noise_floor(n):>9.1f}{tail_gap(n):>9.1f}')

        c = t32_process(dpdf_process(x), model=args.model)
        s, b = dns(c)
        print(f'{"DPDF2 → T32":>20s}{s:>8.3f}{b:>8.3f}{noise_floor(c):>9.1f}{tail_gap(c):>9.1f}')


if __name__ == '__main__':
    main()
