#!/usr/bin/env python3
"""Quantify what each model actually does, instead of one blended DNSMOS.

DPDFNet2 is the denoiser and NVIDIA SB is the dereverberator in the shipped
chain, so they must be judged on different axes:

  BAK  (background)  -> denoising capability
  tail gap           -> dereverberation capability
  SIG  (speech)      -> how much speech is damaged doing it

Run on a controlled signal with known DRR so the dereverb number means something.
"""
import sys
import numpy as np
import onnxruntime as ort
import soundfile as sf
from speechmos import dnsmos

SR = 16000
T32_ONNX = '/home/xh/itc_project/superlin/SL-clearVoice/out/model-study-20261123/nvidia/t32-original.onnx'
DPDF_ONNX = '/home/xh/itc_project/superlin/Audio3A/models/onnx/dpdfnet/dpdfnet2.onnx'
FRAMES = 32
BINS = 256
HOP = 128          # 32 frames * 128 = 4096-sample window (256 ms) per the board profile
FFT = 512


def read(p):
    x, sr = sf.read(p)
    if x.ndim > 1:
        x = x.mean(1)
    return x.astype(np.float64)


def run_t32(x):
    """NVIDIA SB one-step, 32-frame window, 72 ms overlap equivalent."""
    so = ort.SessionOptions()
    so.intra_op_num_threads = 8
    sess = ort.InferenceSession(T32_ONNX, so, providers=['CPUExecutionProvider'])
    win = np.sqrt(np.hanning(FFT + 1)[:FFT])
    ola = np.zeros(FFT)
    out = np.zeros(len(x))
    stride = (FRAMES - 18) * HOP       # 48 ms history + 72 ms overlap -> 128 ms advance
    for pos in range(0, len(x) - FRAMES * HOP, stride):
        seg = np.zeros(FRAMES * HOP)
        seg[:] = x[pos:pos + FRAMES * HOP]
        spec = np.zeros((BINS, FRAMES), dtype=complex)
        for t in range(FRAMES):
            buf = np.zeros(FFT)
            buf[FFT - HOP:] = seg[t * HOP:(t + 1) * HOP]
            spec[:, t] = np.fft.rfft(buf)[:BINS]
        inp = np.stack([spec.real, spec.imag], 0)[None].astype(np.float32)
        est = sess.run(None, {'spectrum': inp})[0]
        s_out = est[0, 0] + 1j * est[0, 1]
        # take the newest hop of the estimated window
        frame = np.fft.irfft(s_out[:, -1], FFT) * win
        out[pos:pos + HOP] += frame[FFT - HOP:]
    return out


def run_dpdfnet(x):
    HOPD, NB = 160, 161
    nfft = 2 * (NB - 1)
    so = ort.SessionOptions()
    so.intra_op_num_threads = 8
    sess = ort.InferenceSession(DPDF_ONNX, so, providers=['CPUExecutionProvider'])
    sd = int(np.prod(sess.get_inputs()[1].shape))
    win = np.sqrt(np.hanning(nfft + 1)[:nfft])
    ola = np.zeros(nfft); hist = np.zeros(nfft - HOPD)
    state = np.zeros(sd, dtype=np.float32)
    out = np.zeros(len(x))
    for t in range(len(x) // HOPD):
        cur = x[t * HOPD:(t + 1) * HOPD]
        buf = np.concatenate([hist, cur]) * win
        hist = cur
        spec = np.fft.rfft(buf)[:NB]
        si = np.stack([spec.real, spec.imag], -1)[None, None].astype(np.float32)
        enh, state = sess.run(None, {'spec': si, 'state_in': state})
        so_s = enh[0, 0, :, 0] + 1j * enh[0, 0, :, 1]
        full = np.zeros(nfft // 2 + 1, dtype=complex); full[:NB] = so_s
        y = np.fft.irfft(full, nfft) * win
        ola += y
        out[t * HOPD:(t + 1) * HOPD] = ola[:HOPD]
        ola = np.concatenate([ola[HOPD:], np.zeros(HOPD)])
    return out


def tail_gap(x):
    """Post-speech drop below speech level: the dereverb axis."""
    n = len(x) // 160
    e = 10 * np.log10(np.array([np.mean(x[i * 160:(i + 1) * 160] ** 2)
                                for i in range(n)]) + 1e-12)
    sp = e > np.percentile(e, 70)
    drops = []
    for i in range(1, len(e) - 25):
        if sp[i - 1] and not sp[i] and not sp[i:i + 25].any():
            drops.append(e[i - 1] - np.mean(e[i + 2:i + 18]))
    return float(np.median(drops)) if drops else float('nan')


def noise_floor(x):
    """Non-speech level relative to speech: the denoise axis."""
    n = len(x) // 160
    e = 10 * np.log10(np.array([np.mean(x[i * 160:(i + 1) * 160] ** 2)
                                for i in range(n)]) + 1e-12)
    sp = e > np.percentile(e, 70)
    return float(np.median(e[sp]) - np.median(e[~sp]))


def dns(x):
    x = x.astype(np.float32)
    pk = float(np.max(np.abs(x)))
    if pk > 0.9:
        x = x / pk * 0.9
    r = dnsmos.run(x, sr=SR)
    return r['sig_mos'], r['bak_mos']


def main():
    for p in sys.argv[1:]:
        x = read(p)
        print(f'\n=== {"/".join(p.split("/")[-4:-2])} ({len(x)/SR:.1f}s) ===')
        print(f'{"处理":>22s}{"SIG":>8s}{"BAK":>8s}{"降噪能力":>10s}{"去混响":>9s}')
        sig, bak = dns(x)
        print(f'{"输入":>22s}{sig:>8.3f}{bak:>8.3f}{noise_floor(x):>10.1f}{tail_gap(x):>9.1f}')

        d = run_dpdfnet(x)
        sig, bak = dns(d)
        print(f'{"DPDFNet2 (降噪)":>22s}{sig:>8.3f}{bak:>8.3f}{noise_floor(d):>10.1f}{tail_gap(d):>9.1f}')

        n = run_t32(x)
        sig, bak = dns(n)
        print(f'{"NVIDIA T32 (去混响)":>22s}{sig:>8.3f}{bak:>8.3f}{noise_floor(n):>10.1f}{tail_gap(n):>9.1f}')

        dn = run_t32(d)
        sig, bak = dns(dn)
        print(f'{"串联 DPDFNet2→T32":>22s}{sig:>8.3f}{bak:>8.3f}{noise_floor(dn):>10.1f}{tail_gap(dn):>9.1f}')


if __name__ == '__main__':
    main()
