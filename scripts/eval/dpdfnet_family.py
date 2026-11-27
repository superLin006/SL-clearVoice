#!/usr/bin/env python3
"""Score every DPDFNet variant on the same real input, with timing.

All four exports share the spec/state interface (hop 160, 161 bins) so only the
state dimension differs; that lets one loop drive them all. Wall-clock per frame
is reported because the winner has to run in real time on RK3588 CPU.
"""
import glob
import os
import sys
import time
import numpy as np
import onnxruntime as ort
import soundfile as sf
from speechmos import dnsmos

MODEL_DIR = '/home/xh/itc_project/superlin/Audio3A/models/onnx/dpdfnet'
HOP, NFFT, NBINS = 160, 320, 161


def read(p):
    x, sr = sf.read(p)
    if x.ndim > 1:
        x = x.mean(1)
    return x.astype(np.float64), sr


def run(model, x):
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    sess = ort.InferenceSession(model, so, providers=['CPUExecutionProvider'])
    sdim = int(np.prod(sess.get_inputs()[1].shape))
    win = np.sqrt(np.hanning(NFFT + 1)[:NFFT])
    ola = np.zeros(NFFT)
    hist = np.zeros(NFFT - HOP)
    state = np.zeros(sdim, dtype=np.float32)
    out = np.zeros(len(x))
    n = len(x) // HOP
    t0 = time.perf_counter()
    for t in range(n):
        cur = x[t * HOP:(t + 1) * HOP]
        buf = np.concatenate([hist, cur]) * win
        hist = cur
        spec = np.fft.rfft(buf)[:NBINS]
        spec_in = np.stack([spec.real, spec.imag], -1)[None, None].astype(np.float32)
        enh, state = sess.run(None, {'spec': spec_in, 'state_in': state})
        s_out = enh[0, 0, :, 0] + 1j * enh[0, 0, :, 1]
        full = np.zeros(NFFT // 2 + 1, dtype=complex)
        full[:NBINS] = s_out
        y = np.fft.irfft(full, NFFT) * win
        ola += y
        out[t * HOP:(t + 1) * HOP] = ola[:HOP]
        ola = np.concatenate([ola[HOP:], np.zeros(HOP)])
    elapsed = time.perf_counter() - t0
    audio_sec = n * HOP / 16000.0
    return out, elapsed / audio_sec   # real-time factor


def main():
    models = sorted(glob.glob(os.path.join(MODEL_DIR, '*.onnx')))
    recs = sys.argv[1:] or ['1', '2', '3', '5', '6']
    acc = {os.path.basename(m): [] for m in models}
    rtf = {os.path.basename(m): [] for m in models}

    for rec in recs:
        p = f'/home/xh/itc_project/superlin/Audio3A/recordings/20261124/{rec}/stages_g7_t32/05_neural_res.wav'
        if not os.path.exists(p):
            continue
        x, sr = read(p)
        for m in models:
            y, r = run(m, x)
            s = dnsmos.run(y.astype(np.float32), sr=sr)
            acc[os.path.basename(m)].append(
                [s['ovrl_mos'], s['sig_mos'], s['bak_mos'], s['p808_mos']])
            rtf[os.path.basename(m)].append(r)

    print(f'\n{"模型":>24s}{"OVRL":>8s}{"SIG":>8s}{"BAK":>8s}{"P808":>8s}{"RTF":>8s}')
    for k in sorted(acc):
        a = np.array(acc[k]).mean(0)
        print(f'{k:>24s}' + ''.join(f'{v:>8.3f}' for v in a)
              + f'{np.mean(rtf[k]):>8.2f}')


if __name__ == '__main__':
    main()
