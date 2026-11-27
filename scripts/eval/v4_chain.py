#!/usr/bin/env python3
"""Numpy/scipy reimplementation of scripts/v4_post.sh.

ffmpeg is not installed here, so the filter chain is reproduced directly:
  highpass 150 (2) -> EQ 300 +0.5 / 1200 +6 / 2200 +2 / 3200 +4 / 5000 -1
  -> lowpass 6000 (2) -> dynaudnorm -> loudnorm -> agate

dynaudnorm/loudnorm/agate are level-domain stages and are approximated by a
sliding-RMS normaliser plus a downward gate. The spectral stages (HP/EQ/LP) are
exact, which is what matters when comparing HF shelf variants *through* V4.
"""
import numpy as np
from scipy.signal import butter, sosfilt, lfilter


def _peak(x, f0, q, gain_db, sr=16000):
    """Standard RBJ peaking EQ."""
    if abs(gain_db) < 1e-6:
        return x
    a = 10 ** (gain_db / 40.0)
    w0 = 2 * np.pi * f0 / sr
    alpha = np.sin(w0) / (2 * q)
    b0 = 1 + alpha * a
    b1 = -2 * np.cos(w0)
    b2 = 1 - alpha * a
    a0 = 1 + alpha / a
    a1 = -2 * np.cos(w0)
    a2 = 1 - alpha / a
    return lfilter([b0 / a0, b1 / a0, b2 / a0], [1.0, a1 / a0, a2 / a0], x)


def v4_spectral(x, sr=16000):
    sos = butter(2, 150 / (sr / 2), btype='highpass', output='sos')
    y = sosfilt(sos, x)
    y = _peak(y, 300, 1.2, 0.5, sr)
    y = _peak(y, 1200, 1.0, 6.0, sr)
    y = _peak(y, 2200, 1.0, 2.0, sr)
    y = _peak(y, 3200, 1.0, 4.0, sr)
    y = _peak(y, 5000, 1.0, -1.0, sr)
    sos = butter(2, 6000 / (sr / 2), btype='lowpass', output='sos')
    return sosfilt(sos, y)


def v4_level(x, sr=16000, target_rms_db=-20.0, gate_db=-34.0):
    win = max(1, int(0.300 * sr))
    rms = np.sqrt(np.convolve(x ** 2, np.ones(win) / win, mode='same') + 1e-12)
    target = 10 ** (target_rms_db / 20.0)
    gain = np.clip(target / (rms + 1e-12), 10 ** (-12 / 20), 10 ** (20 / 20))
    gk = max(1, int(0.05 * sr))
    gain = np.convolve(gain, np.ones(gk) / gk, mode='same')
    y = x * gain
    level_db = 20 * np.log10(np.abs(y) + 1e-12)
    mask = level_db < gate_db
    if mask.any():
        atten = np.ones(len(y))
        atten[mask] = 0.02
        ak = max(1, int(0.25 * sr))
        atten = np.convolve(atten, np.ones(ak) / ak, mode='same')
        y = y * atten
    return y


def v4(x, sr=16000):
    return v4_level(v4_spectral(x, sr), sr)
