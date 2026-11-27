#!/usr/bin/env python3
"""Compare board-matched model stages using DNSMOS and acoustic proxies.

Energy/tail values use per-output energy masks. They are not SNR, DRR or
clean-reference fidelity scores; align stages and share an input-derived mask
for a controlled cross-stage comparison.
"""
import sys
from t32_correct import t32_process, dns as score_dns
import numpy as np
import soundfile as sf
from speechmos import dnsmos

SR = 16000


def read(p):
    x, sr = sf.read(p)
    if sr != SR or x.ndim != 1:
        raise ValueError(f"{p}: expected mono 16 kHz stage WAV")
    return x.astype(np.float64)


def run_t32(x):
    """Use the current board-matched T32 transform and streaming profile."""
    return t32_process(x)


def run_dpdfnet(x):
    """Use the board-matched window, model state and 256-sample FIFO."""
    from dpdf_runner import process
    return process(x)


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
    return score_dns(x)


def main():
    for p in sys.argv[1:]:
        x = read(p)
        print(f'\n=== {"/".join(p.split("/")[-4:-2])} ({len(x)/SR:.1f}s) ===')
        print(f'{"处理":>22s}{"SIG":>8s}{"BAK":>8s}{"能量落差":>10s}{"尾音代理":>9s}')
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
