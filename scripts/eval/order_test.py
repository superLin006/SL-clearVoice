#!/usr/bin/env python3
"""Compare denoise-then-dereverb against dereverb-then-denoise.

Uses the board-matched T32 runner (510-point FFT, per-block normalisation, 0.33
power-law compression) so the ordering question is answered with the real model
behaviour rather than a re-implementation.

Energy/tail values are descriptive proxies with per-output energy masks.
They are not SNR, DRR or clean-reference fidelity scores. Align audio and use
a shared input-derived mask before interpreting changes as processing gains.
"""
import argparse
from pathlib import Path
from t32_correct import MODEL, t32_process

import numpy as np
import soundfile as sf

SR = 16000


def t32(x, dry=0.0, overlap=48, history=48, frames=32, model=None):
    return t32_process(x, history_ms=history, overlap_ms=overlap,
                       dry_gain=dry, frames=frames, model=model)


def dpdf(x):
    """Use the board-matched window, model state and 256-sample FIFO."""
    from dpdf_runner import process
    return process(x)


def fdb(x):
    n = len(x) // 160
    return 10 * np.log10(np.array([np.mean(x[i * 160:(i + 1) * 160] ** 2)
                                   for i in range(n)]) + 1e-12)


def tail_gap(x):
    e = fdb(x)
    sp = e > np.percentile(e, 70)
    d = []
    for i in range(1, len(e) - 25):
        if sp[i - 1] and not sp[i] and not sp[i:i + 25].any():
            d.append(e[i - 1] - np.mean(e[i + 2:i + 18]))
    return float(np.median(d)) if d else float('nan')


def noise_floor(x):
    e = fdb(x)
    sp = e > np.percentile(e, 70)
    return float(np.median(e[sp]) - np.median(e[~sp]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="Mono 16 kHz input, e.g. 07_transient_xover.wav")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", default=MODEL)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    print(f'{"rec":>24s}{"顺序":>22s}{"能量落差":>8s}{"尾音代理":>9s}')
    for index, path in enumerate(args.inputs, 1):
        x, sr = sf.read(path)
        if sr != SR or x.ndim != 1:
            parser.error(f"{path}: expected mono 16 kHz stage WAV")
        x = x.astype(np.float64)
        a = t32(dpdf(x), model=args.model)
        b = dpdf(t32(x, model=args.model))
        for name, tag, y in (("DPDF2→T32", "cur", a), ("T32→DPDF2", "new", b)):
            print(f"{path.name:>24s}{name:>22s}{noise_floor(y):>8.1f}{tail_gap(y):>9.1f}")
            sf.write(args.output / f"{index:02d}-{path.stem}-{tag}.wav",
                     np.asarray(y, dtype=np.float32), SR)


if __name__ == "__main__":
    main()
