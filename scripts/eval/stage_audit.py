#!/usr/bin/env python3
"""Fixed-mask per-stage audit for Audio3A stage dumps.

A frame mask derived once from RAW input is applied to every stage at the same
frame indices, so AGC level changes cannot shift which frames get measured.
All spectral numbers are relative shapes (per-stage normalized), so only
timbre changes between stages are meaningful.

Usage: stage_audit.py <stage_dump_dir> [more dirs...]
"""
import sys, os, glob
import numpy as np
import soundfile as sf

SR = 16000
NFFT, HOP = 1024, 256
BANDS = [(100, 300), (300, 600), (600, 1200), (1200, 2400),
         (2400, 3600), (3600, 4800), (4800, 6400), (6400, 8000)]
LBL = ['100-300', '300-600', '600-1.2k', '1.2-2.4k', '2.4-3.6k',
       '3.6-4.8k', '4.8-6.4k', '6.4-8k']


def read(path):
    x, _ = sf.read(path)
    if x.ndim > 1:
        x = x.mean(1)
    return x.astype(np.float64)


def stft_power(x):
    w = np.hanning(NFFT)
    n = max(0, (len(x) - NFFT) // HOP)
    if n <= 0:
        return np.zeros((0, NFFT // 2 + 1))
    idx = np.arange(NFFT)[None, :] + HOP * np.arange(n)[:, None]
    return np.abs(np.fft.rfft(x[idx] * w, axis=1)) ** 2


def frame_db(P):
    return 10 * np.log10(P.sum(1) + 1e-12)


def speech_mask(P_raw):
    """Frames within 20 dB of the loud decile. Deliberately strict: this room
    has a very low direct-to-reverberant ratio, so a loose threshold selects
    nearly every frame and hides level differences entirely."""
    edb = frame_db(P_raw)
    return edb > (np.percentile(edb, 95) - 20)


def spectral_flux(P):
    """Mean positive log-spectral change per frame -- proxy for onset sharpness."""
    L = np.log(P + 1e-12)
    return float(np.mean(np.maximum(np.diff(L, axis=0), 0.0)))


def onset_rise_ms(x, mask):
    """Median time for an energy envelope to climb out of the local trough."""
    e = np.convolve(x ** 2, np.ones(HOP) / HOP, mode='same')
    edb = 10 * np.log10(np.maximum(e, 1e-12))
    peaks = np.where(mask)[0]
    if len(peaks) == 0:
        return float('nan')
    rises = []
    for i in peaks[::max(1, len(peaks) // 300)]:
        j = i
        while j > 0 and edb[j] > edb[j - 1]:
            j -= 1
        if j == 0:
            continue
        lo = j
        while lo > 0 and edb[lo] < edb[j] - 20:
            lo -= 1
        if lo == 0:
            continue
        rises.append((j - lo) * 1000.0 / SR)
    return float(np.median(rises)) if rises else float('nan')


def band_db(P, mask, f):
    Ps = P[mask].mean(0)
    Ps = Ps / (Ps[(f >= 100) & (f < 8000)].sum() + 1e-12)
    return [10 * np.log10(Ps[(f >= lo) & (f < hi)].sum() + 1e-15)
            for lo, hi in BANDS]


def contrast_db(x, mask):
    """Speech-frame level minus non-speech-frame level, on a frame grid that
    matches the mask length exactly."""
    n = len(mask)
    if n == 0 or mask.all() or (~mask).all():
        return float('nan')
    e = np.array([np.mean(x[i * HOP:i * HOP + NFFT] ** 2) for i in range(n)])
    edb = 10 * np.log10(e + 1e-12)
    return float(np.percentile(edb[mask], 50) - np.percentile(edb[~mask], 50))


def audit(d):
    stages = sorted(glob.glob(os.path.join(d, '*.wav')))
    if not stages:
        print(f'{d}: no wav found')
        return
    P_raw = stft_power(read(stages[0]))
    f = np.fft.rfftfreq(NFFT, 1 / SR)
    mask = speech_mask(P_raw)
    edb = frame_db(P_raw)

    print(f'\n=== {d} ===')
    print(f'    raw speech mask: {mask.sum()}/{len(mask)} = {100.0*mask.mean():.0f}%'
          f' | raw p95-p10 = {np.percentile(edb,95)-np.percentile(edb,10):.1f} dB'
          f' (低=声学动态范围小=混响/噪声填满字间)')
    print(f"{'stage':22s}{'HF倾斜':>8s}{'flux':>9s}{'起音ms':>8s}{'语音-非语音':>10s}"
          + ''.join(f'{l:>9s}' for l in LBL))
    for s in stages:
        x = read(s)
        P = stft_power(x)
        m = mask[:len(P)]
        if m.sum() < 10:
            print(f'{os.path.basename(s)[:-4]:22s}  (mask empty)')
            continue
        Pm = P[m]
        hf = 10 * np.log10((Pm[:, (f >= 2400) & (f < 8000)].sum() + 1e-12) /
                           (Pm[:, (f >= 300) & (f < 1200)].sum() + 1e-12))
        print(f'{os.path.basename(s)[:-4]:22s}{hf:>8.1f}{spectral_flux(Pm):>9.4f}'
              f'{onset_rise_ms(x, m):>8.1f}{contrast_db(x, m):>10.1f}'
              + ''.join(f'{v:>9.1f}' for v in band_db(P, m, f)))


if __name__ == '__main__':
    for d in sys.argv[1:]:
        audit(d)
