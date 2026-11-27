#!/usr/bin/env python3
"""Server-side evaluation of the NVIDIA SB stage using the board-matched runner.

Answers the open question from the local runs: does the NVIDIA dereverberator
make the preceding denoiser redundant, or do they genuinely complement?

Times are measured too, since the point is whether a variant can run in real time.
"""
import argparse
import nvidia_runner
from t32_correct import DPDF, MODEL
import os
import time

import numpy as np
import soundfile as sf

SR = 16000
HERE = os.path.dirname(os.path.abspath(__file__))


def load_runner():
    return nvidia_runner


def read(p):
    x, sr = sf.read(p)
    if sr != SR:
        raise ValueError(f"{p}: expected 16 kHz audio, got {sr}")
    if x.ndim > 1:
        x = x.mean(1)
    return x.astype(np.float64)


def frame_db(x):
    n = len(x) // 160
    return 10 * np.log10(np.array([np.mean(x[i * 160:(i + 1) * 160] ** 2)
                                   for i in range(n)]) + 1e-12)


def tail_gap(x):
    """Dereverberation axis: how far the post-speech tail sits below speech."""
    e = frame_db(x)
    sp = e > np.percentile(e, 70)
    d = []
    for i in range(1, len(e) - 25):
        if sp[i - 1] and not sp[i] and not sp[i:i + 25].any():
            d.append(e[i - 1] - np.mean(e[i + 2:i + 18]))
    return float(np.median(d)) if d else float('nan')


def noise_floor(x):
    """Denoising axis: non-speech level relative to speech."""
    e = frame_db(x)
    sp = e > np.percentile(e, 70)
    return float(np.median(e[sp]) - np.median(e[~sp]))


def t32(x, model, frames=32, history_ms=48, overlap_ms=48, dry=0.0):
    m = load_runner()
    est = m.OnnxEstimator(model, 8)
    pipe = m.Pipeline(frames, m.hann_window(), est, 0.0)
    pcm = np.clip(np.rint(x * 32768.0), -32768, 32767).astype(np.int16)
    t0 = time.perf_counter()
    result = pipe.process(pcm, int(history_ms * 16), int(overlap_ms * 16), dry)
    y = np.asarray(result[0], dtype=np.float64)   # int16 PCM, matches the board exactly
    dt = time.perf_counter() - t0
    return np.asarray(y, dtype=np.float64) / 32768.0, dt / (len(x) / SR)


def dpdf(x, model):
    import onnxruntime as ort
    HOPD, NB = 160, 161
    nfft = 2 * (NB - 1)
    so = ort.SessionOptions()
    so.intra_op_num_threads = 8
    sess = ort.InferenceSession(model, so, providers=['CPUExecutionProvider'])
    sd = int(np.prod(sess.get_inputs()[1].shape))
    win = np.sqrt(np.hanning(nfft + 1)[:nfft])
    ola = np.zeros(nfft)
    hist = np.zeros(nfft - HOPD)
    state = np.zeros(sd, dtype=np.float32)
    out = np.zeros(len(x))
    t0 = time.perf_counter()
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
    return out, time.perf_counter() - t0


def scores(x):
    try:
        from speechmos import dnsmos
        xf = np.asarray(x, dtype=np.float32)
        pk = float(np.max(np.abs(xf)))
        if pk > 0.9:
            xf = xf / pk * 0.9
        r = dnsmos.run(xf, sr=SR)
        return r['sig_mos'], r['bak_mos'], r['ovrl_mos'], r['p808_mos']
    except Exception:
        return (float('nan'),) * 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('inputs', nargs='+')
    ap.add_argument('--t32', default=MODEL)
    ap.add_argument('--dpdf', default=DPDF)
    ap.add_argument('--frames', type=int, default=32)
    ap.add_argument('--history-ms', type=int, default=48)
    ap.add_argument('--overlap-ms', type=int, default=48)
    ap.add_argument('--dry', type=float, default=0.0)
    ap.add_argument('--save-dir', default='')
    args = ap.parse_args()

    print(f'{"输入":>10s}{"处理":>18s}{"SIG":>8s}{"BAK":>8s}{"OVRL":>8s}{"P808":>8s}'
          f'{"降噪":>8s}{"去混响":>8s}{"RTF":>7s}')
    for p in args.inputs:
        x = read(p)
        tag = os.path.basename(os.path.dirname(os.path.dirname(p)))
        rows = [('输入', x, 0.0)]
        d, ddt = dpdf(x, args.dpdf)
        rows.append(('DPDFNet2', d, ddt))
        n, tdt = t32(x, args.t32, args.frames, args.history_ms, args.overlap_ms, args.dry)
        rows.append(('T32 单独', n, tdt))
        c, cdt = t32(d, args.t32, args.frames, args.history_ms, args.overlap_ms, args.dry)
        rows.append(('DPDF2→T32', c, cdt))

        for name, y, dt in rows:
            s = scores(y)
            print(f'{tag:>10s}{name:>18s}'
                  + ''.join(f'{v:>8.3f}' for v in s)
                  + f'{noise_floor(y):>8.1f}{tail_gap(y):>8.1f}{dt:>7.3f}')
            if args.save_dir:
                os.makedirs(args.save_dir, exist_ok=True)
                sf.write(os.path.join(args.save_dir,
                                      f'{tag}-{name.replace(" ", "_")}.wav'),
                         np.asarray(y, dtype=np.float32), SR)
        print()


if __name__ == '__main__':
    main()
