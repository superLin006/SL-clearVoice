#!/usr/bin/env python3
"""Blind DRR-related metric for a mono 16 kHz signal.

Without a clean reference the usable proxy is the modulation depth of the energy
envelope: reverberation fills every inter-syllable dip, so a stronger direct path
shows up as a larger spread between loud and quiet frames AND a faster decay
right after speech offsets. Both are reported together because a level shift
alone can inflate one of them.

usage: drr_metric.py <a.f32|a.wav> [more...]
"""
import sys
import os
import numpy as np
import soundfile as sf

SR, HOP = 16000, 160
FLOOR = 1e-12


def load(path):
    if path.endswith('.f32'):
        return np.fromfile(path, dtype=np.float32).astype(np.float64)
    x, _ = sf.read(path)
    if x.ndim > 1:
        x = x.mean(1)
    return x.astype(np.float64)


def env(x):
    n = len(x) // HOP
    e = np.array([np.mean(x[i * HOP:(i + 1) * HOP] ** 2) for i in range(n)])
    return 10 * np.log10(e + FLOOR)


def metrics(x):
    e = env(x)
    spread = np.percentile(e, 95) - np.percentile(e, 10)
    # decay: drop from the loud-frame median to the frame ~150-300 ms later,
    # sampled only where speech clearly ended
    sp = e > np.percentile(e, 70)
    drops = []
    for i in range(1, len(e) - 20):
        if sp[i - 1] and not sp[i] and not sp[i:i + 20].any():
            drops.append(e[i - 1] - np.mean(e[i + 5:i + 20]))
    decay = float(np.median(drops)) if drops else float('nan')
    return spread, decay, len(drops)


if __name__ == '__main__':
    print(f"{'signal':34s}{'动态范围':>10s}{'尾音衰减':>10s}{'事件数':>7s}")
    for p in sys.argv[1:]:
        s, d, n = metrics(load(p))
        print(f'{os.path.basename(p):34s}{s:>10.1f}{d:>10.1f}{n:>7d}')
