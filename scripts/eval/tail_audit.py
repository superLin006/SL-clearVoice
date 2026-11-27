#!/usr/bin/env python3
"""Measure how far the post-speech tail sits below the speech level, per stage.

The audible complaint "ghost noise between words" is exactly this number: if the
tail right after a word is only ~12 dB below speech it reads as residue; at ~35 dB
it reads as clean. The mask is taken from RAW and reused on every stage so the
comparison is like-for-like.

Usage: tail_audit.py <stage_dump_dir> [more dirs...]
"""
import sys, os, glob
import numpy as np
import soundfile as sf

SR, HOP = 16000, 160
TAIL_FRAMES = 30          # 0-300 ms after a speech offset
LOOKBACK = 30             # speech must have been active within the last 300 ms


def read(path):
    x, _ = sf.read(path)
    if x.ndim > 1:
        x = x.mean(1)
    return x.astype(np.float64)


def env_db(x):
    n = len(x) // HOP
    e = np.array([np.mean(x[i * HOP:(i + 1) * HOP] ** 2) for i in range(n)])
    return 10 * np.log10(e + 1e-12)


def speech_mask_from_raw(edb):
    """Top 30% of frames by energy. This room has a ~7 dB total dynamic range,
    so anything looser marks every frame as speech and the tail window empties."""
    return edb > np.percentile(edb, 70)


def offsets(sp, n):
    out = []
    for i in range(1, n - TAIL_FRAMES):
        if sp[i - 1] and not sp[i]:
            # require the following window to stay non-speech
            if not sp[i:i + TAIL_FRAMES].any():
                out.append(i)
    return out


def audit(d):
    stages = sorted(glob.glob(os.path.join(d, '*.wav')))
    if not stages:
        print(f'{d}: no wav')
        return
    er = env_db(read(stages[0]))
    n = len(er)
    sp = speech_mask_from_raw(er)
    off = offsets(sp, n)
    if not off:
        print(f'{d}: no clean offsets found')
        return

    print(f'\n=== {d} ===  speech frames {100.0*sp.mean():.0f}%, '
          f'post-speech windows: {len(off)}')
    print(f"{'stage':22s}{'语音':>8s}{'尾音0-300ms':>12s}{'落差':>8s}"
          f"{'尾音p90':>9s}{'尾音落在语音内比例':>12s}")
    for s in stages:
        ef = env_db(read(s))
        m = min(len(ef), n)
        spm = sp[:m]
        speech_lvl = np.median(ef[:m][spm])
        tails = [np.mean(ef[i:i + TAIL_FRAMES]) for i in off if i + TAIL_FRAMES < m]
        if not tails:
            print(f'{os.path.basename(s)[:-4]:22s}  (no tail window)')
            continue
        tails = np.array(tails)
        # fraction of tail windows that are louder than the speech median
        leak = float(np.mean(tails > speech_lvl - 15.0))
        print(f'{os.path.basename(s)[:-4]:22s}{speech_lvl:>8.1f}{np.median(tails):>12.1f}'
              f'{speech_lvl-np.median(tails):>8.1f}{np.percentile(tails,90):>9.1f}'
              f'{100.0*leak:>11.0f}%')


if __name__ == '__main__':
    for d in sys.argv[1:]:
        audit(d)
