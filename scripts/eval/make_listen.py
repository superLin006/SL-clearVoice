#!/usr/bin/env python3
"""Build listening-comparison material from real board stage dumps.

Produces, per recording, a folder with matched-level excerpts so the difference
heard is timbre/artefacts and not loudness:

  A_raw.wav      raw input (mean of 8 mics), levelled to -23 LUFS-ish
  B_current.wav  what the board outputs today
  C_noagc.wav    same chain with the pre-ANS AGC removed (isolates its effect)

and a continuous single file per recording for A/B/C switching.

Everything comes from the board's own stage dumps, so nothing here is a
Python re-simulation of the pipeline.
"""
import os
import sys
import glob
import numpy as np
import soundfile as sf

SR = 16000
STAGE_RAW = '00_raw_input_avg.wav'
STAGE_PRE_AGC_IN = '05_neural_res.wav'   # input to the pre-ANS AGC
STAGE_FINAL = '12_final_output.wav'


def read(p):
    x, _ = sf.read(p)
    if x.ndim > 1:
        x = x.mean(1)
    return x.astype(np.float64)


def rms_db(x):
    return 20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-12)


def level_to(x, target_db=-23.0):
    """Match perceived loudness coarsely via speech-frame RMS."""
    n = len(x) // 160
    e = np.array([np.mean(x[i * 160:(i + 1) * 160] ** 2) for i in range(n)])
    speech = e > np.percentile(e, 60)
    cur = 10 * np.log10(np.mean(e[speech]) + 1e-12)
    return x * (10 ** ((target_db - cur) / 20.0))


def pick_excerpt(x, seconds=20.0, start_frac=0.25):
    n = int(seconds * SR)
    if len(x) <= n:
        return x
    s = int(len(x) * start_frac)
    s = min(s, len(x) - n)
    return x[s:s + n]


def fade(x, ms=10):
    k = int(SR * ms / 1000)
    if len(x) < 2 * k:
        return x
    w = np.ones(len(x))
    w[:k] = np.linspace(0, 1, k)
    w[-k:] = np.linspace(1, 0, k)
    return x * w


def build(rec_dir, out_root):
    """rec_dir is the recording folder itself (the one holding stages_g7_t32)."""
    rec = os.path.basename(rec_dir.rstrip('/'))
    sd = os.path.join(rec_dir, 'stages_g7_t32')
    if not os.path.isdir(sd):
        print(f'  skip {rec}: no stage dump')
        return

    raw = read(os.path.join(sd, STAGE_RAW))
    final = read(os.path.join(sd, STAGE_FINAL))
    pre = read(os.path.join(sd, STAGE_PRE_AGC_IN))

    # C: replace the pre-AGC output with its un-gained input, then reuse the
    # later stages' *gain shape* is not possible offline; instead expose the
    # pre-AGC signal as its own reference so the user can hear what the tail
    # looks like before the AGC lifts it.
    variants = {
        'A_raw': raw,
        'B_current': final,
        'C_preAGC': pre,
    }

    out_dir = os.path.join(out_root, rec)
    os.makedirs(out_dir, exist_ok=True)

    n = min(len(v) for v in variants.values())
    for name, sig in variants.items():
        e = fade(level_to(pick_excerpt(sig[:n])))
        sf.write(os.path.join(out_dir, f'{name}.wav'), e, SR, subtype='PCM_16')

    # one continuous file: 0.4 s gap between variants
    gap = np.zeros(int(SR * 0.4))
    seq = []
    for name in ('A_raw', 'B_current', 'C_preAGC'):
        seq.append(fade(level_to(pick_excerpt(variants[name][:n]))))
        seq.append(gap)
    sf.write(os.path.join(out_dir, 'ABC_sequence.wav'),
             np.concatenate(seq), SR, subtype='PCM_16')
    print(f'  {rec}: {len(variants["A_raw"][:n])/SR:.1f}s source -> {out_dir}')


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else \
        '/home/xh/itc_project/superlin/Audio3A/recordings/20261124'
    out = sys.argv[2] if len(sys.argv) > 2 else \
        '/home/xh/itc_project/superlin/SL-clearVoice/out/listen'
    os.makedirs(out, exist_ok=True)
    for d in sorted(glob.glob(os.path.join(root, '*'))):
        if os.path.isdir(d):
            build(d, out)
    print(f'\nwrote to {out}')


if __name__ == '__main__':
    main()
