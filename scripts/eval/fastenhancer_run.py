#!/usr/bin/env python3
"""Run FastEnhancer in place of the NVIDIA SB stage, on identical input.

FastEnhancer scored PESQ 1.93 vs T32's 1.48 in the model-selection study, though
its SI-SDR was lower (2.21 vs 4.30) -- i.e. it is more faithful but dereverbs
less. This script lets that trade-off be heard on the real chain input.

Interface: 512-point STFT (hop 256, 257 bins), spec_in [1,257,1,2] plus three
or four cache tensors depending on the variant.
"""
import sys
import numpy as np
import onnxruntime as ort
import soundfile as sf
from speechmos import dnsmos

SR = 16000
NFFT = 512
HOP = 256
BINS = 257

MODELS = {
    's': '/home/xh/itc_project/superlin/SL-clearVoice/out/model-study-20261123/fastenhancer/fastenhancer-s-dereverb.onnx',
    'm': '/home/xh/itc_project/superlin/SL-clearVoice/out/model-selection-20261123/fast-medium/fastenhancer-m-dereverb.onnx',
}


def read(p):
    x, _ = sf.read(p)
    if x.ndim > 1:
        x = x.mean(1)
    return x.astype(np.float64)


def run_fastenhancer(x, size='m'):
    so = ort.SessionOptions()
    so.intra_op_num_threads = 8
    sess = ort.InferenceSession(MODELS[size], so, providers=['CPUExecutionProvider'])
    ins = sess.get_inputs()
    in_names = [i.name for i in ins]
    cache_names = [n for n in in_names if n.startswith('cache_in_')]
    caches = [np.zeros(i.shape, dtype=np.float32) for i in ins if i.name.startswith('cache_in_')]

    win = np.sqrt(np.hanning(NFFT + 1)[:NFFT])
    ola = np.zeros(NFFT)
    hist = np.zeros(NFFT - HOP)
    out = np.zeros(len(x))

    for t in range(len(x) // HOP):
        cur = x[t * HOP:(t + 1) * HOP]
        buf = np.concatenate([hist, cur]) * win
        hist = cur
        spec = np.fft.rfft(buf)[:BINS]
        si = np.stack([spec.real, spec.imag], -1)[None, :, None].astype(np.float32)

        feed = {'spec_in': si}
        for n, c in zip(cache_names, caches):
            feed[n] = c
        res = sess.run(None, feed)
        so_spec = res[0]
        new_caches = res[1:]

        z = so_spec[0, :, 0, 0] + 1j * so_spec[0, :, 0, 1]
        full = np.zeros(NFFT // 2 + 1, dtype=complex)
        full[:BINS] = z
        y = np.fft.irfft(full, NFFT) * win
        ola += y
        out[t * HOP:(t + 1) * HOP] = ola[:HOP]
        ola = np.concatenate([ola[HOP:], np.zeros(HOP)])
        caches = list(new_caches)
    return out


def score(x):
    xf = np.asarray(x, dtype=np.float32)
    pk = float(np.max(np.abs(xf)))
    if pk > 0.9:
        xf = xf / pk * 0.9
    r = dnsmos.run(xf, sr=SR)
    return r['sig_mos'], r['bak_mos'], r['ovrl_mos'], r['p808_mos']


def band_ltas(x):
    nfft = 1024
    w = np.hanning(nfft)
    fr = []
    for i in range(0, len(x) - nfft, 512):
        fr.append(np.abs(np.fft.rfft(x[i:i + nfft] * w)) ** 2)
    F = np.array(fr)
    e = F.sum(1)
    sp = e > np.percentile(e, 60)
    P = F[sp].mean(0)
    return np.fft.rfftfreq(nfft, 1 / SR), P / (P.sum() + 1e-20)


def main():
    src = sys.argv[1]
    x = read(src)
    print(f'\n{src}  ({len(x)/SR:.1f}s)')
    print(f'{"版本":>18s}{"SIG":>8s}{"BAK":>8s}{"OVRL":>8s}{"P808":>8s}'
          f'{"1.2-2.4k":>10s}{"2.4-3.6k":>10s}{"3.6-4.8k":>10s}')

    rows = [('输入', x)]
    for size in ('s', 'm'):
        rows.append((f'FastEnhancer-{size}', run_fastenhancer(x, size)))

    for name, y in rows:
        s = score(y)
        f, P = band_ltas(y)
        b = [10 * np.log10(P[(f >= lo) & (f < hi)].sum() + 1e-20)
             for lo, hi in ((1200, 2400), (2400, 3600), (3600, 4800))]
        print(f'{name:>18s}' + ''.join(f'{v:>8.3f}' for v in s)
              + ''.join(f'{v:>10.1f}' for v in b))
        sf.write(f'/tmp/fe_{size}.wav',
                 (y / max(np.max(np.abs(y)), 1e-9) * 0.9).astype(np.float32), SR)


if __name__ == '__main__':
    main()
