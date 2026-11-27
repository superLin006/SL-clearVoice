#!/usr/bin/env python3
"""Compare denoise-then-dereverb against dereverb-then-denoise.

Uses the board-matched T32 runner (510-point FFT, per-block normalisation, 0.33
power-law compression) so the ordering question is answered with the real model
behaviour rather than a re-implementation.
"""
import importlib.util
import os
import sys

import numpy as np
import soundfile as sf

WORK = '/tmp/nveval'
SR = 16000

_spec = importlib.util.spec_from_file_location('rn', os.path.join(WORK, 'run_nvidia.py'))
m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m)


def t32(x, dry=0.025, overlap=72, history=48, frames=32):
    est = m.OnnxEstimator(os.path.join(WORK, 't32-original.onnx'), 8)
    pipe = m.Pipeline(frames, m.hann_window(), est, 0.0)
    pcm = np.clip(np.rint(x * 32768.0), -32768, 32767).astype(np.int16)
    r = pipe.process(pcm, int(history * 16), int(overlap * 16), dry)
    return np.asarray(r[0], dtype=np.float64) / 32768.0


def dpdf(x):
    import onnxruntime as ort
    HOPD, NB = 160, 161
    nfft = 2 * (NB - 1)
    so = ort.SessionOptions()
    so.intra_op_num_threads = 8
    s = ort.InferenceSession(os.path.join(WORK, 'dpdfnet2.onnx'), so,
                             providers=['CPUExecutionProvider'])
    sd = int(np.prod(s.get_inputs()[1].shape))
    win = np.sqrt(np.hanning(nfft + 1)[:nfft])
    ola = np.zeros(nfft)
    hist = np.zeros(nfft - HOPD)
    st = np.zeros(sd, dtype=np.float32)
    out = np.zeros(len(x))
    for t in range(len(x) // HOPD):
        cur = x[t * HOPD:(t + 1) * HOPD]
        buf = np.concatenate([hist, cur]) * win
        hist = cur
        sp = np.fft.rfft(buf)[:NB]
        si = np.stack([sp.real, sp.imag], -1)[None, None].astype(np.float32)
        e, st = s.run(None, {'spec': si, 'state_in': st})
        z = e[0, 0, :, 0] + 1j * e[0, 0, :, 1]
        full = np.zeros(nfft // 2 + 1, dtype=complex)
        full[:NB] = z
        y = np.fft.irfft(full, nfft) * win
        ola += y
        out[t * HOPD:(t + 1) * HOPD] = ola[:HOPD]
        ola = np.concatenate([ola[HOPD:], np.zeros(HOPD)])
    return out


def fdb(x):
    n = len(x) // 160
    return 10 * np.log10(np.array([np.mean(x[i * 160:(i + 1) * 160] ** 2)
                                   for i in range(n)]) + 1e-12)


def tail_gap(x):
    e = fdb(x)
    sp = e > np.percentile(e, 70)
    d = []
    for i in range(1, len(e) - 25):
        if sp[i - 1] and not sp[i] and not sp[i:i + 25].any():
            d.append(e[i - 1] - np.mean(e[i + 2:i + 18]))
    return float(np.median(d)) if d else float('nan')


def noise_floor(x):
    e = fdb(x)
    sp = e > np.percentile(e, 70)
    return float(np.median(e[sp]) - np.median(e[~sp]))


def main():
    outs = sys.argv[1] if len(sys.argv) > 1 else '/tmp/nveval/order_out'
    os.makedirs(outs, exist_ok=True)
    print(f'{"rec":>4s}{"顺序":>22s}{"降噪":>8s}{"去混响":>9s}')
    for rec in '123456':
        p = (f'/mnt/data/zjw/source/3308/Audio3A/recordings/20261124/'
             f'{rec}/stages_g7_t32/00_raw_input_avg.wav')
        x, _ = sf.read(p)
        x = x.astype(np.float64)
        d = dpdf(x)
        a = t32(d)          # current: denoise -> dereverb
        b = dpdf(t32(x))    # proposed: dereverb -> denoise
        for name, tag, y in (('DPDF2→T32(当前)', 'cur', a),
                             ('T32→DPDF2(提议)', 'new', b)):
            print(f'{rec:>4s}{name:>22s}{noise_floor(y):>8.1f}{tail_gap(y):>9.1f}')
            sf.write(os.path.join(outs, f'{rec}-{tag}.wav'),
                     np.asarray(y, dtype=np.float32), SR)
        print()


if __name__ == '__main__':
    main()
