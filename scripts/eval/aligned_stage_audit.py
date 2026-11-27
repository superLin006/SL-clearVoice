"""Aligned fixed-frame diagnostics for unpaired real Audio3A recordings.

Energy groups and tails are acoustic proxies, not labelled speech/noise or DRR.
No PESQ/STOI/clean-reference fidelity claim is made.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy import signal

SR, HOP, WINDOW = 16000, 160, 320
BANDS = [(100, 300), (300, 1200), (1200, 2400), (2400, 3600),
         (3600, 4800), (4800, 6400), (6400, 8000)]


def db(power):
    return 10 * np.log10(np.maximum(power, 1e-20))


def frames(x):
    if len(x) < WINDOW:
        return np.empty((0, WINDOW))
    return np.lib.stride_tricks.sliding_window_view(x, WINDOW)[::HOP]


def read(path):
    x, sr = sf.read(path, dtype="float64", always_2d=True)
    if sr != SR or x.shape[1] != 1:
        raise ValueError((str(path), sr, x.shape))
    return x[:, 0]


def alignment(reference, track):
    """Median of local band-limited waveform delays; never a quality score."""
    n = min(len(reference), len(track))
    filt = signal.butter(3, [180, 3600], btype="bandpass", fs=SR, output="sos")
    candidates = []
    for start in np.linspace(0, max(0, n - 3 * SR), 9, dtype=int):
        u = signal.sosfilt(filt, reference[start:start + 3 * SR])
        v = signal.sosfilt(filt, track[start:start + 3 * SR])
        if len(u) < SR or np.mean(u * u) < 1e-10:
            continue
        cc = signal.correlate(v, u, mode="full", method="fft")
        lags = signal.correlation_lags(len(v), len(u))
        keep = abs(lags) <= 3200
        idx = np.argmax(abs(cc[keep]))
        lag = int(lags[keep][idx])
        if lag >= 0:
            a, b = u[:len(u)-lag or None], v[lag:]
        else:
            a, b = u[-lag:], v[:len(v)+lag]
        corr = float(np.dot(a, b) / np.sqrt(np.dot(a, a) * np.dot(b, b) + 1e-30))
        candidates.append((lag, corr))
    if not candidates:
        return {"samples": 0, "ms": 0, "reliable": False, "windows": []}
    # Filter weak or ambiguous waveform matches; record every candidate.
    usable = [lag for lag, cc in candidates if abs(cc) >= 0.15]
    lag = int(np.median(usable)) if usable else 0
    dispersion = float(np.median(abs(np.array(usable) - lag))) if usable else None
    reliable = len(usable) >= 3 and dispersion <= 160
    return {"samples": lag, "ms": lag / 16, "reliable": reliable,
            "median_abs_deviation_samples": dispersion, "windows": candidates}


def aligned(track, lag, count):
    idx = np.arange(count) + lag
    available = (idx >= 0) & (idx < len(track))
    out = np.zeros(count)
    out[available] = track[idx[available]]
    # Exclude padded boundary frames from all group metrics.
    valid_frames = frames(available).all(axis=1)
    return out, valid_frames


def mask_from_raw(raw):
    energy = db(np.mean(frames(raw) ** 2, axis=1))
    valid = energy > -90
    if valid.sum() < 30:
        raise ValueError("Insufficient nonzero raw frames")
    lo, high = np.percentile(energy[valid], [30, 70])
    return energy, valid, valid & (energy >= high), valid & (energy <= lo)


def offsets(high, valid):
    """High-energy offsets followed by 300ms without high-energy frames."""
    found = []
    for k in range(6, len(high) - 30):
        if high[k-6:k].all() and not high[k:k+30].any() and valid[k:k+30].all():
            found.append(k)
    return found


def summarize(raw, path):
    x = read(path)
    match = alignment(raw, x)
    y, boundary = aligned(x, match["samples"], len(raw))
    raw_energy, valid, high, low = mask_from_raw(raw)
    valid &= boundary
    high &= valid
    low &= valid
    f = frames(y)
    energy = db(np.mean(f * f, axis=1))
    def med(mask):
        if not mask.any() or np.median(np.mean(f[mask]**2, axis=1)) == 0:
            return None
        return float(np.median(energy[mask]))
    spectrum = abs(np.fft.rfft(f * np.hanning(WINDOW), axis=1)) ** 2
    hz = np.fft.rfftfreq(WINDOW, 1 / SR)
    def band_level(mask, lo, hi):
        if not mask.any():
            return None
        power = spectrum[mask].mean(axis=0)
        total = power[(hz >= 100) & (hz < 8000)].sum()
        if total <= 0:
            return None
        return float(db(power[(hz >= lo) & (hz < hi)].sum() / max(total, 1e-30)))
    bands = {f"{lo}-{hi}": {"high_relative_db": band_level(high, lo, hi),
                               "low_relative_db": band_level(low, lo, hi)}
             for lo, hi in BANDS}
    shifts = energy - raw_energy
    relative_gain = shifts[high] - np.median(shifts[high])
    falls = offsets(high, valid)
    tails = []
    for at in falls:
        before = np.mean(f[max(0,at-10):at] ** 2)
        after = np.mean(f[at:at+30] ** 2)
        tails.append(float(db(before) - db(after)))
    return {
        "file": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "samples": len(x), "raw_samples": len(raw), "alignment": match,
        "high_frames": int(high.sum()), "low_frames": int(low.sum()),
        "raw_digital_silence_frames_excluded": int((~mask_from_raw(raw)[1]).sum()),
        "high_median_dbfs": med(high), "low_median_dbfs": med(low),
        "high_zero_frame_fraction": float(np.mean(np.mean(f[high]**2, axis=1)==0)),
        "low_zero_frame_fraction": float(np.mean(np.mean(f[low]**2, axis=1)==0)) if low.any() else None,
        "high_low_gap_db": med(high) - med(low) if med(high) is not None and med(low) is not None else None,
        "high_gain_p10_p50_p90_db": list(map(float, np.percentile(shifts[high],[10,50,90]))),
        "high_relative_gain_below_minus20db_fraction": float(np.mean(relative_gain < -20)),
        "peak_dbfs": float(db(np.max(x*x))),
        "clip_fraction": float(np.mean(abs(x) >= 32767/32768)),
        "exact_zero_fraction": float(np.mean(x == 0)),
        "tail_windows": len(tails),
        "median_pre_post_tail_gap_db": float(np.median(tails)) if tails else None,
        "bands": bands,
        "interpretation": "Energy groups are not speech labels. Zero median power is reported as null, not an arbitrary very low noise level. Alignment reliability must be checked."
    }


def audit(directory):
    stages = [p for p in sorted(directory.glob("[0-1][0-9]_*.wav")) if sf.info(p).frames > 0]
    if not stages:
        raise ValueError(f"No stage WAV files in {directory}")
    raw = read(directory / "00_raw_input_avg.wav")
    return {path.stem: summarize(raw, path) for path in stages}


def selftest():
    # A known delay/gain must preserve the energy contrast and exclude leading
    # digital silence. A muted active interval must trigger the dropout proxy.
    rng = np.random.default_rng(16)
    n = 12 * SR
    t = np.arange(n) / SR
    envelope = .03 * (0.2 + np.sin(t * 2.4)**4)
    x = signal.sosfilt(signal.butter(3, [200, 3000], fs=SR, btype="bandpass", output="sos"),
                       rng.normal(size=n)) * envelope
    x[:SR] = 0
    lag = 240
    y = np.r_[np.zeros(lag), x[:-lag]] * 2
    result = alignment(x, y)
    assert result["samples"] == lag and result["reliable"], result
    z, available = aligned(y, lag, n)
    er, valid, high, low = mask_from_raw(x)
    assert (~valid[:98]).all()
    assert np.max(abs(z[:-lag] - 2*x[:-lag])) < 1e-12
    ey = db(np.mean(frames(z)**2, axis=1))
    good = high & available
    assert abs(np.median(ey[good]-er[good])-20*np.log10(2)) < 1e-9
    z[3*SR:7*SR] = 0
    ez = db(np.mean(frames(z)**2, axis=1))
    gain = ez[good]-er[good]
    assert np.mean(gain-np.median(gain) < -20) > .1
    print("Known-delay/gain, digital-silence and active-dropout checks passed")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directories", nargs="*", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--selftest", action="store_true")
    args = p.parse_args()
    if args.selftest:
        selftest()
    report = {str(d): audit(d) for d in args.directories}
    if report:
        if not args.output:
            p.error("--output is required for stage analysis")
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
        print(args.output)
