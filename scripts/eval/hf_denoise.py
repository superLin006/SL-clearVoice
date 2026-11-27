#!/usr/bin/env python3
"""Causal HF suppressor targeting the band measured as noisy after T32.

Measured on the real chain (T32 + AGC), speech-frame vs non-speech-frame SNR:

    300-600 Hz   23.3 dB      2.4-3.6 kHz  23.1 dB
    600-1.2k     29.2 dB      3.6-4.8 kHz  12.3 dB   <- breaks down here
    1.2-2.4k     22.9 dB      4.8-8 kHz     8.4 dB

So suppression starts at 3.6 kHz and ramps in, leaving the speech body intact.

Noise tracking must ignore digital-silence frames: the recording contains
-200 dBFS gaps which drag a plain minimum estimator to nonsense (an earlier
attempt computed 140+ dB SNR and therefore never attenuated anything). Frames
are gated by an absolute floor first, and the noise estimate is a low
percentile *of the gated frames only*.

Fully causal, no lookahead: safe to run on the board.
"""
import sys
import numpy as np
import soundfile as sf
from speechmos import dnsmos

SR = 16000
NFFT = 512
HOP = 160
START_HZ = 3600.0
FRAME_GATE_DB = -90.0


def stft_power(x):
    w = np.hanning(NFFT)
    n = (len(x) - NFFT) // HOP + 1
    idx = np.arange(NFFT)[None, :] + HOP * np.arange(n)[:, None]
    return np.abs(np.fft.rfft(x[idx] * w, axis=1)) ** 2, w, n


def hf_suppress(x, oversub=2.0, floor_db=-15.0, boot_frames=60):
    P, w, n = stft_power(x)
    f = np.fft.rfftfreq(NFFT, 1 / SR)
    e_db = 10 * np.log10(P.sum(1) + 1e-20)
    gated = e_db > FRAME_GATE_DB

    if gated.sum() < boot_frames:
        return x.copy()

    # Noise estimate from the quiet fifth of *gated* frames, per bin.
    gated_P = P[gated]
    quiet_cut = np.percentile(gated_P.sum(1), 20)
    quiet = gated_P[gated_P.sum(1) <= quiet_cut]
    noise = np.percentile(quiet, 50, axis=0) + 1e-20

    # Frequency ramp: 0 at/below START_HZ, fully on above 1.8x START_HZ.
    ramp = np.clip((f - START_HZ) / (START_HZ * 0.8), 0.0, 1.0)

    power_floor = 10 ** (floor_db / 10.0)
    gain = np.ones(P.shape)
    for t in range(n):
        if not gated[t]:
            continue
        snr = P[t] / (oversub * noise)
        g = snr / (1.0 + snr)
        g = np.clip(g, power_floor, 1.0)
        gain[t] = 1.0 - ramp * (1.0 - g)

    X = np.fft.rfft(x[np.arange(NFFT)[None, :] + HOP * np.arange(n)[:, None]]
                    * w, axis=1)
    Y = X * gain
    y = np.fft.irfft(Y, NFFT, axis=1) * w
    ola = np.zeros(len(x) + NFFT)
    norm = np.zeros(len(x) + NFFT)
    for t in range(n):
        s = t * HOP
        ola[s:s + NFFT] += y[t]
        norm[s:s + NFFT] += w ** 2
    return (ola[:len(x)] / np.maximum(norm[:len(x)], 1e-9))


def score(x):
    xf = np.asarray(x, dtype=np.float32)
    pk = float(np.max(np.abs(xf)))
    if pk > 0.9:
        xf = xf / pk * 0.9
    r = dnsmos.run(xf, sr=SR)
    return r['sig_mos'], r['bak_mos'], r['ovrl_mos'], r['p808_mos']


def band_snr(x):
    """Speech-vs-nonspeech SNR per band -- the axis the hiss lives on."""
    P, _, _ = stft_power(x)
    f = np.fft.rfftfreq(NFFT, 1 / SR)
    e = 10 * np.log10(P.sum(1) + 1e-20)
    gated = e > FRAME_GATE_DB
    sp = (e > np.percentile(e[gated], 70)) & gated
    qs = gated & ~sp
    out = []
    for lo, hi in ((3600, 4800), (4800, 6400), (6400, 8000)):
        s = P[sp][:, (f >= lo) & (f < hi)].sum(1)
        q = P[qs][:, (f >= lo) & (f < hi)].sum(1)
        out.append(10 * np.log10((np.median(s) + 1e-20) / (np.median(q) + 1e-20)))
    return out


def main():
    src = sys.argv[1]
    x, _ = sf.read(src)
    x = x.astype(np.float64)
    print(f'\n{src}')
    print(f'{"版本":>14s}{"SIG":>8s}{"BAK":>8s}{"OVRL":>8s}{"P808":>8s}'
          f'{"3.6-4.8k":>10s}{"4.8-6.4k":>10s}{"6.4-8k":>9s}')
    s = score(x)
    b = band_snr(x)
    print(f'{"原始":>14s}' + ''.join(f'{v:>8.3f}' for v in s)
          + ''.join(f'{v:>10.1f}' for v in b))

    for oversub, floor_db in ((1.5, -12.0), (2.5, -15.0), (4.0, -20.0)):
        y = hf_suppress(x, oversub=oversub, floor_db=floor_db)
        s = score(y)
        b = band_snr(y)
        tag = f'x{oversub}/{floor_db:.0f}dB'
        print(f'{tag:>14s}' + ''.join(f'{v:>8.3f}' for v in s)
              + ''.join(f'{v:>10.1f}' for v in b))
        out = f'/tmp/hfsup_{oversub}.wav'
        sf.write(out, (y / max(np.max(np.abs(y)), 1e-9) * 0.9).astype(np.float32), SR)


if __name__ == '__main__':
    main()
