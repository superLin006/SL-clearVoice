#!/usr/bin/env python3
"""End-to-end comparison of front-end variants feeding the same neural tail.

The point is not to reimplement the board DSP but to answer one question with
real recordings: does changing how the 8 mics are combined change what the
downstream dereverb/denoise stage can do? The tail is represented by DPDFNet2
(the model the board actually runs) so the ordering it sees matters.

Variants: mean8 (production), ch2 (best single mic), and both after
multichannel RLS-WPE, since WPE is the stage that is supposed to raise DRR.
"""
import os
import subprocess
import sys
import numpy as np
import onnxruntime as ort

SR, HOP = 16000, 160
AUDIO3A = '/home/xh/itc_project/superlin/Audio3A'
MODEL = os.path.join(AUDIO3A, 'models/onnx/dpdfnet/dpdfnet2.onnx')
WPE = '/tmp/wpe_sweep'
FLOOR = 1e-12


def load_variants(rec):
    raw = np.fromfile(f'/tmp/raw{rec}.f32', dtype=np.float32).reshape(-1, 10)
    mics = raw[:, :8].astype(np.float64)
    out = {'mean8': mics.mean(1), 'ch2': mics[:, 2]}
    for name, label in (('ch2wpe', 'c2'), ('mean8wpe', 'mean8')):
        prefix = f'/tmp/cc{rec}_{name}'
        subprocess.run([WPE, f'/tmp/raw{rec}.f32', '10', prefix,
                        '8', '3', '0.99', '2'], check=True, capture_output=True)
        suffix = '_c2.f32' if label == 'c2' else '_mean8.f32'
        out[name] = np.fromfile(prefix + suffix, dtype=np.float32).astype(np.float64)
    return out


def hpf(x, fc=80.0, order=2):
    from scipy.signal import butter, sosfilt
    sos = butter(order, fc / (SR / 2), btype='highpass', output='sos')
    return sosfilt(sos, x)


def dpdfnet(x):
    """Streaming DPDFNet2 at the board's 160-sample hop.

    The model exposes a single flat state vector and a [1,1,161,2] spectrum, so
    the STFT geometry is fixed by the export, not by this script."""
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    sess = ort.InferenceSession(MODEL, so, providers=['CPUExecutionProvider'])
    ins = {i.name: i for i in sess.get_inputs()}
    in_names = [i.name for i in sess.get_inputs()]
    state_dim = int(np.prod(ins["state_in"].shape))

    n_bins = 161
    nfft = 2 * (n_bins - 1)
    win = np.sqrt(np.hanning(nfft + 1)[:nfft])
    ola = np.zeros(nfft)
    hist = np.zeros(nfft - HOP)
    state = np.zeros(state_dim, dtype=np.float32)
    out = np.zeros(len(x))
    n = len(x) // HOP

    for t in range(n):
        cur = x[t * HOP:(t + 1) * HOP]
        buf = np.concatenate([hist, cur]) * win
        hist = cur
        spec = np.fft.rfft(buf)[:n_bins]
        spec_in = np.stack([spec.real, spec.imag], -1)[None, None].astype(np.float32)

        res = sess.run(None, {'spec': spec_in, 'state_in': state})
        enh, state = res[0], res[1]

        spec_out = enh[0, 0, :, 0] + 1j * enh[0, 0, :, 1]
        full = np.zeros(nfft // 2 + 1, dtype=complex)
        full[:n_bins] = spec_out
        y = np.fft.irfft(full, nfft) * win
        ola += y
        out[t * HOP:(t + 1) * HOP] = ola[:HOP]
        ola = np.concatenate([ola[HOP:], np.zeros(HOP)])
    return out


def metrics(x):
    n = len(x) // HOP
    e = 10 * np.log10(np.array([np.mean(x[i * HOP:(i + 1) * HOP] ** 2)
                                for i in range(n)]) + FLOOR)
    spread = float(np.percentile(e, 95) - np.percentile(e, 10))
    sp = e > np.percentile(e, 70)
    drops = []
    for i in range(1, len(e) - 20):
        if sp[i - 1] and not sp[i] and not sp[i:i + 20].any():
            drops.append(e[i - 1] - np.mean(e[i + 5:i + 20]))
    return spread, (float(np.median(drops)) if drops else float('nan'))


def main():
    for rec in (sys.argv[1:] or ['1', '2']):
        variants = load_variants(rec)
        print(f'\n=== 录音 {rec} ===')
        print(f"{'variant':12s}{'前端后 尾音衰减':>18s}{'DPDFNet2后 尾音衰减':>22s}")
        for name, sig in variants.items():
            pre = metrics(hpf(sig))
            post = metrics(dpdfnet(hpf(sig)))
            print(f'{name:12s}{pre[1]:>18.1f}{post[1]:>22.1f}')


if __name__ == '__main__':
    main()
