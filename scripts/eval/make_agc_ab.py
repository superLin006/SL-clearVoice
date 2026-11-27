#!/usr/bin/env python3
"""Produce A/B listening files for the AGC expansion fix.

Uses the board's own pre-AGC stage dump (05_neural_res.wav) so the only thing
that changes between the two files is the AGC. Both are loudness-normalised to
the same speech level, otherwise the louder one always "sounds better".

Output per recording:
  NN_agc_before.wav  current board behaviour
  NN_agc_after.wav   with downward expansion enabled
  NN_AB.wav          before, 0.6 s gap, after  (continuous for easy switching)
"""
import os
import subprocess
import sys
import numpy as np
import soundfile as sf

SR = 16000
AGC_BIN = '/tmp/agc_off'


def read_f32(p):
    return np.fromfile(p, dtype=np.float32).astype(np.float64)


def speech_level(x):
    n = len(x) // 160
    e = np.array([np.mean(x[i * 160:(i + 1) * 160] ** 2) for i in range(n)])
    sp = e > np.percentile(e, 70)
    return np.sqrt(np.mean(e[sp]) + 1e-12)


def match(x, target):
    return x * (target / (speech_level(x) + 1e-12))


def excerpt(x, seconds=18.0, frac=0.3):
    n = int(seconds * SR)
    if len(x) <= n:
        return x
    s = min(int(len(x) * frac), len(x) - n)
    return x[s:s + n]


def fade(x, ms=12):
    k = int(SR * ms / 1000)
    if len(x) < 2 * k:
        return x
    w = np.ones(len(x))
    w[:k] = np.linspace(0, 1, k)
    w[-k:] = np.linspace(1, 0, k)
    return x * w


def main():
    root = '/home/xh/itc_project/superlin/Audio3A/recordings/20261124'
    out = '/home/xh/itc_project/superlin/SL-clearVoice/out/listen_agc'
    os.makedirs(out, exist_ok=True)

    for rec in ['1', '2', '3', '4', '5', '6']:
        src = os.path.join(root, rec, 'stages_g7_t32', '05_neural_res.wav')
        if not os.path.exists(src):
            print(f'skip {rec}')
            continue
        off = f'/tmp/mk{rec}_off.f32'
        on = f'/tmp/mk{rec}_on.f32'
        subprocess.run([AGC_BIN, src, off, on], check=True, capture_output=True)

        before = excerpt(read_f32(off))
        after = excerpt(read_f32(on))

        # normalise both to the same speech level so loudness is not the variable
        target = 0.06
        before = match(before, target)
        after = match(after, target)

        sf.write(os.path.join(out, f'{rec}_agc_before.wav'), fade(before), SR, subtype='PCM_16')
        sf.write(os.path.join(out, f'{rec}_agc_after.wav'), fade(after), SR, subtype='PCM_16')

        gap = np.zeros(int(SR * 0.6))
        sf.write(os.path.join(out, f'{rec}_AB.wav'),
                 np.concatenate([fade(before), gap, fade(after)]), SR, subtype='PCM_16')
        print(f'  rec{rec}: wrote before/after/AB')


if __name__ == '__main__':
    main()
