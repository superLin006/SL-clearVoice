#!/usr/bin/env python3
"""Does correct-geometry spatial filtering actually beat plain averaging?

The shipped beamformer assumes 6 mics on a 56/60 mm ring, but the product board
is 8 mics at 40 mm, so it has never been a fair test. This implements the two
standard options directly so the question can be answered before any C work:

  delay-and-sum  -- phase-align each mic to a candidate direction, sum
  MVDR-lite      -- DAS plus a per-bin coherence weighting

Both sweep all directions and keep the best, which is the optimistic case for
spatial processing. Compared against the current mean-of-8 using the same DRR
proxy and DNSMOS.
"""
import sys
import numpy as np
import soundfile as sf
from speechmos import dnsmos

SR = 16000
NFFT, HOP = 512, 256
RADIUS_M = 0.034   # product board: 8-mic ring, 34 mm radius
N_MIC = 8

# Board calibration table from p2p_dut_node.c (kCalib8). Without this the array
# is wildly unbalanced (290x vs 80x) and any beamformer sees garbage.
CALIB8 = np.array([290.0, 84.0, 81.0, 80.0, 89.0, 250.0, 92.0, 88.0])
SPEED_OF_SOUND = 343.0


def read_multichannel(p, n_mic=N_MIC, calibrate=True):
    x, sr = sf.read(p)
    x = x[:, :n_mic].astype(np.float64)
    if calibrate and n_mic == 8:
        # Equalise channel sensitivities, then renormalise so the beamformer and
        # DNSMOS see a sane amplitude range.
        x = x * CALIB8[None, :]
        peak = np.max(np.abs(x))
        if peak > 0:
            x = x / peak * 0.5
    return x, sr


def mic_positions(n_mic=N_MIC, radius=RADIUS_M):
    """Evenly spaced ring; first mic at 0 deg, going counter-clockwise."""
    a = 2 * np.pi * np.arange(n_mic) / n_mic
    return radius * np.cos(a), radius * np.sin(a)


def steering_delays(theta_rad, n_mic=N_MIC, radius=RADIUS_M):
    """Per-mic delay (seconds) for a far-field source at theta."""
    mx, my = mic_positions(n_mic, radius)
    proj = mx * np.cos(theta_rad) + my * np.sin(theta_rad)
    return -proj / SPEED_OF_SOUND


def stft(x):
    n = (len(x) - NFFT) // HOP + 1
    idx = np.arange(NFFT)[None, :] + HOP * np.arange(n)[:, None]
    return np.fft.rfft(x[idx] * np.hanning(NFFT), axis=1)


def istft(X, length):
    frames = np.fft.irfft(X, NFFT, axis=1) * np.hanning(NFFT)
    out = np.zeros(length + NFFT)
    norm = np.zeros(length + NFFT)
    for i in range(X.shape[0]):
        s = i * HOP
        out[s:s + NFFT] += frames[i]
        norm[s:s + NFFT] += np.hanning(NFFT) ** 2
    out = out / (norm + 1e-9)
    return out[:length]


def delay_and_sum(mics, theta_deg):
    """Align phases for a source at theta, then sum. Returns time signal."""
    theta = np.deg2rad(theta_deg)
    d = steering_delays(theta)
    f = np.fft.rfftfreq(NFFT, 1 / SR)
    specs = np.stack([stft(mics[:, m]) for m in range(mics.shape[1])], axis=0)
    # phase rotation to undo the propagation delay per mic
    rot = np.exp(-2j * np.pi * f[None, :] * d[:, None])
    aligned = (specs * rot[:, None, :]).mean(0)
    return istft(aligned, len(mics))


def mvdr_lite(mics, theta_deg):
    """DAS followed by a per-bin coherence weighting across mics."""
    theta = np.deg2rad(theta_deg)
    d = steering_delays(theta)
    f = np.fft.rfftfreq(NFFT, 1 / SR)
    specs = np.stack([stft(mics[:, m]) for m in range(mics.shape[1])], axis=0)
    rot = np.exp(-2j * np.pi * f[None, :] * d[:, None])
    aligned = specs * rot[:, None, :]
    # coherence weight: 1 where mics agree in phase, low where they disagree
    mag = aligned / (np.abs(aligned) + 1e-12)
    coh = np.abs(mag.mean(0))
    w = coh ** 2
    out = (aligned * w[None]).mean(0)
    return istft(out, len(mics))


def drr_proxy(x):
    n = len(x) // 160
    e = 10 * np.log10(np.array([np.mean(x[i * 160:(i + 1) * 160] ** 2)
                                for i in range(n)]) + 1e-12)
    sp = e > np.percentile(e, 70)
    drops = []
    for i in range(1, len(e) - 20):
        if sp[i - 1] and not sp[i] and not sp[i:i + 20].any():
            drops.append(e[i - 1] - np.mean(e[i + 5:i + 20]))
    return float(np.median(drops)) if drops else float('nan')


def score(x):
    x = x.astype(np.float32)
    if not np.all(np.isfinite(x)):
        return (np.nan,) * 4
    peak = float(np.max(np.abs(x)))
    if peak > 0.9:
        x = x / peak * 0.9
    r = dnsmos.run(x, sr=SR)
    return r['ovrl_mos'], r['sig_mos'], r['bak_mos'], r['p808_mos']


def main():
    for p in sys.argv[1:]:
        mics, sr = read_multichannel(p)
        mics = mics[:len(mics) // HOP * HOP]
        print(f'\n=== {p.split("/")[-3]} ({len(mics)/sr:.1f}s, {mics.shape[1]} mic @ {RADIUS_M*1000:.0f}mm) ===')
        print(f'{"方案":>16s}{"DRR":>8s}{"OVRL":>8s}{"SIG":>8s}{"BAK":>8s}{"P808":>8s}')

        base = mics.mean(1)
        print(f'{"mean8(当前)":>16s}{drr_proxy(base):>8.1f}'
              + ''.join(f'{v:>8.3f}' for v in score(base)))

        for name, fn in (('DAS', delay_and_sum), ('MVDR-lite', mvdr_lite)):
            best = None
            for ang in range(0, 360, 15):
                y = fn(mics, ang)
                d = drr_proxy(y)
                if best is None or d > best[0]:
                    best = (d, ang, y)
            d, ang, y = best
            print(f'{f"{name}@{ang}°":>16s}{d:>8.1f}'
                  + ''.join(f'{v:>8.3f}' for v in score(y)))


if __name__ == '__main__':
    main()
