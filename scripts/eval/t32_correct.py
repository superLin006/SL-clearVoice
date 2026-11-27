#!/usr/bin/env python3
"""Correct NVIDIA T32 runner, reusing the project's verified implementation.

Earlier attempts in this session used a 512-point FFT and skipped the per-block
peak normalisation and 0.33 power-law compression, which do not match the board.
This delegates to out/t32-50-v4-20261123/run_nvidia.py so the numbers agree with
the board-tested profile.

Used to answer: does NVIDIA SB replace the denoiser, or complement it?
"""
import importlib.util
import os
import sys

import numpy as np
import soundfile as sf
from speechmos import dnsmos

SR = 16000
RUNNER = '/home/xh/itc_project/superlin/SL-clearVoice/out/t32-50-v4-20261123/run_nvidia.py'
MODEL = '/home/xh/itc_project/superlin/SL-clearVoice/out/model-study-20261123/nvidia/t32-original.onnx'
DPDF = '/home/xh/itc_project/superlin/Audio3A/models/onnx/dpdfnet/dpdfnet2.onnx'


def load_runner():
    spec = importlib.util.spec_from_file_location('run_nvidia', RUNNER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def t32_process(x, history_ms=48.0, overlap_ms=72.0, dry_gain=0.025, frames=32):
    """Run the board's T32 profile over a float signal."""
    m = load_runner()
    est = m.OnnxEstimator(MODEL, 8)
    enh = m.Pipeline(frames, m.hann_window(), est, 0.0)   # sigma 0 = no Gaussian term
    pcm = np.clip(np.rint(x * 32768.0), -32768, 32767).astype(np.int16)
    y = enh.process(pcm, int(history_ms * 16), int(overlap_ms * 16), dry_gain)
    return np.asarray(y, dtype=np.float64) / 32768.0


def dpdf_process(x):
    import onnxruntime as ort
    HOPD, NB = 160, 161
    nfft = 2 * (NB - 1)
    so = ort.SessionOptions()
    so.intra_op_num_threads = 8
    sess = ort.InferenceSession(DPDF, so, providers=['CPUExecutionProvider'])
    sd = int(np.prod(sess.get_inputs()[1].shape))
    win = np.sqrt(np.hanning(nfft + 1)[:nfft])
    ola = np.zeros(nfft)
    hist = np.zeros(nfft - HOPD)
    state = np.zeros(sd, dtype=np.float32)
    out = np.zeros(len(x))
    for t in range(len(x) // HOPD):
        cur = x[t * HOPD:(t + 1) * HOPD]
        buf = np.concatenate([hist, cur]) * win
        hist = cur
        spec = np.fft.rfft(buf)[:NB]
        si = np.stack([spec.real, spec.imag], -1)[None, None].astype(np.float32)
        enh, state = sess.run(None, {'spec': si, 'state_in': state})
        s = enh[0, 0, :, 0] + 1j * enh[0, 0, :, 1]
        full = np.zeros(nfft // 2 + 1, dtype=complex)
        full[:NB] = s
        y = np.fft.irfft(full, nfft) * win
        ola += y
        out[t * HOPD:(t + 1) * HOPD] = ola[:HOPD]
        ola = np.concatenate([ola[HOPD:], np.zeros(HOPD)])
    return out


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
    x = np.asarray(x, dtype=np.float32)
    pk = float(np.max(np.abs(x)))
    if pk > 0.9:
        x = x / pk * 0.9
    r = dnsmos.run(x, sr=SR)
    return r['sig_mos'], r['bak_mos']


def main():
    for p in sys.argv[1:]:
        x, _ = sf.read(p)
        if x.ndim > 1:
            x = x.mean(1)
        x = x.astype(np.float64)
        print(f'\n=== {p.split("/")[-3]}/{p.split("/")[-1]} ===')
        print(f'{"处理":>20s}{"SIG":>8s}{"BAK":>8s}{"降噪":>9s}{"去混响":>9s}')
        s, b = dns(x)
        print(f'{"输入":>20s}{s:>8.3f}{b:>8.3f}{noise_floor(x):>9.1f}{tail_gap(x):>9.1f}')

        d = dpdf_process(x)
        s, b = dns(d)
        print(f'{"DPDFNet2":>20s}{s:>8.3f}{b:>8.3f}{noise_floor(d):>9.1f}{tail_gap(d):>9.1f}')

        n = t32_process(x)
        s, b = dns(n)
        print(f'{"T32 单独":>20s}{s:>8.3f}{b:>8.3f}{noise_floor(n):>9.1f}{tail_gap(n):>9.1f}')

        c = t32_process(dpdf_process(x))
        s, b = dns(c)
        print(f'{"DPDF2 → T32":>20s}{s:>8.3f}{b:>8.3f}{noise_floor(c):>9.1f}{tail_gap(c):>9.1f}')


if __name__ == '__main__':
    main()
