#!/usr/bin/env python3
"""Fair A/B of the two denoisers actually shipped in the pipeline.

DPDFNet2 runs at hop 160 (the board's 10 ms cadence); UL-UNAS runs at 512 FFT /
hop 256 as its streaming export expects. Both get the identical input signal and
the output is scored with DNSMOS, so the comparison is like for like.

Usage: denoiser_compare.py <in.wav> [more.wav...]
"""
import sys
import numpy as np
import onnxruntime as sf_ort
import soundfile as sf
from speechmos import dnsmos

DPDF = '/home/xh/itc_project/superlin/Audio3A/models/onnx/dpdfnet/dpdfnet2.onnx'
ULUNAS = '/home/xh/itc_project/superlin/SL-clearVoice/models/rknn/live/ulunas_stream_simple.onnx'


def read(p):
    x, sr = sf.read(p)
    if x.ndim > 1:
        x = x.mean(1)
    return x.astype(np.float64), sr


def run_dpdfnet(x, n_bins=161):
    """DPDFNet2 streaming, hop 160 as on the board."""
    HOP = 160
    nfft = 2 * (n_bins - 1)
    so = sf_ort.SessionOptions()
    so.intra_op_num_threads = 4
    sess = sf_ort.InferenceSession(DPDF, so, providers=['CPUExecutionProvider'])
    state_dim = int(np.prod(sess.get_inputs()[1].shape))
    win = np.sqrt(np.hanning(nfft + 1)[:nfft])
    ola = np.zeros(nfft)
    hist = np.zeros(nfft - HOP)
    state = np.zeros(state_dim, dtype=np.float32)
    out = np.zeros(len(x))
    for t in range(len(x) // HOP):
        cur = x[t * HOP:(t + 1) * HOP]
        buf = np.concatenate([hist, cur]) * win
        hist = cur
        spec = np.fft.rfft(buf)[:n_bins]
        spec_in = np.stack([spec.real, spec.imag], -1)[None, None].astype(np.float32)
        enh, state = sess.run(None, {'spec': spec_in, 'state_in': state})
        so_spec = enh[0, 0, :, 0] + 1j * enh[0, 0, :, 1]
        full = np.zeros(nfft // 2 + 1, dtype=complex)
        full[:n_bins] = so_spec
        y = np.fft.irfft(full, nfft) * win
        ola += y
        out[t * HOP:(t + 1) * HOP] = ola[:HOP]
        ola = np.concatenate([ola[HOP:], np.zeros(HOP)])
    return out


def run_ulunas(x):
    """UL-UNAS streaming: 512 FFT, hop 256, three external caches."""
    FFT, HOP, BINS = 512, 256, 257
    so = sf_ort.SessionOptions()
    so.intra_op_num_threads = 4
    sess = sf_ort.InferenceSession(ULUNAS, so, providers=['CPUExecutionProvider'])
    conv = np.zeros((1, 5358), dtype=np.float32)
    tfa = np.zeros((1, 402), dtype=np.float32)
    inter = np.zeros((1, 1056), dtype=np.float32)
    win = np.sqrt(np.hanning(FFT + 1)[:FFT])
    ola = np.zeros(FFT)
    hist = np.zeros(FFT - HOP)
    out = np.zeros(len(x))
    for t in range(len(x) // HOP):
        cur = x[t * HOP:(t + 1) * HOP]
        buf = np.concatenate([hist, cur]) * win
        hist = cur
        spec = np.fft.rfft(buf)
        mix = np.stack([spec.real, spec.imag], -1)[None, :, None].astype(np.float32)
        enh, conv, tfa, inter = sess.run(
            None, {'mix': mix, 'conv_cache': conv,
                   'tfa_cache': tfa, 'inter_cache': inter})
        s_out = enh[0, :, 0, 0] + 1j * enh[0, :, 0, 1]
        y = np.fft.irfft(s_out, FFT) * win
        ola += y
        out[t * HOP:(t + 1) * HOP] = ola[:HOP]
        ola = np.concatenate([ola[HOP:], np.zeros(HOP)])
    return out


def score(x, sr):
    r = dnsmos.run(x.astype(np.float32), sr=sr)
    return r['ovrl_mos'], r['sig_mos'], r['bak_mos'], r['p808_mos']


def main():
    for p in sys.argv[1:]:
        x, sr = read(p)
        print(f'\n=== {p.split("/")[-3]}/{p.split("/")[-1]} ({len(x)/sr:.1f}s) ===')
        print(f"{'版本':>10s}{'OVRL':>8s}{'SIG':>8s}{'BAK':>8s}{'P808':>8s}")
        for name, fn in (('输入', lambda s: s), ('DPDFNet2', run_dpdfnet),
                         ('UL-UNAS', run_ulunas)):
            y = fn(x)
            o, s_, b, p8 = score(y, sr)
            print(f'{name:>10s}{o:>8.3f}{s_:>8.3f}{b:>8.3f}{p8:>8.3f}')


if __name__ == '__main__':
    main()
