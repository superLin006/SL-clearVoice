import random

import numpy as np
import soundfile as sf
import torch
from scipy.signal import fftconvolve, resample_poly
from torch.utils.data import Dataset


def read_mono(path, sr=16000):
    x, fs = sf.read(path, dtype='float32', always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if fs != sr:
        x = resample_poly(x, sr, fs).astype(np.float32)
    return x.astype(np.float32)


def crop_or_pad(x, n, rng):
    if len(x) >= n:
        st = rng.randint(0, len(x) - n)
        return x[st:st + n]
    reps = int(np.ceil(n / max(1, len(x))))
    return np.tile(x, reps)[:n]


class RIRNoiseDataset(Dataset):
    def __init__(self, speech_files, rir_files, noise_files=(), seg=64000, sr=16000,
                 p_rir=0.85, p_noise=0.8, seed=0, snr_range=(0.0, 20.0), target_rms=0.04):
        self.speech_files = list(speech_files)
        self.rir_files = list(rir_files)
        self.noise_files = list(noise_files)
        self.seg = seg
        self.sr = sr
        self.p_rir = p_rir
        self.p_noise = p_noise
        self.seed = seed
        self.snr_range = snr_range
        self.target_rms = target_rms

    def __len__(self):
        return len(self.speech_files)

    def __getitem__(self, idx):
        rng = random.Random((self.seed + idx) * 1000003)
        dry = crop_or_pad(read_mono(self.speech_files[idx], self.sr), self.seg, rng)
        rev = dry
        if self.rir_files and rng.random() < self.p_rir:
            rir = read_mono(rng.choice(self.rir_files), self.sr)
            max_len = int(0.8 * self.sr)
            if len(rir) > max_len:
                rir = rir[:max_len]
            peak = int(np.argmax(np.abs(rir)))
            rir = rir[max(0, peak - 8):]
            if len(rir) > 8:
                p0 = int(np.argmax(np.abs(rir)))
                # normalize direct-path gain to 0 dB so reverb does not change the level
                rir = rir / (abs(rir[p0]) + 1e-8)
                rev = fftconvolve(dry, rir)[:self.seg].astype(np.float32)
        noisy = rev.copy()
        if self.noise_files and rng.random() < self.p_noise:
            n = crop_or_pad(read_mono(rng.choice(self.noise_files), self.sr), self.seg, rng)
            snr = rng.uniform(*self.snr_range)
            ps = float(np.mean(rev ** 2)) + 1e-8
            pn = float(np.mean(n ** 2)) + 1e-8
            noisy = rev + np.sqrt(ps / (pn * (10 ** (snr / 10.0)))) * n
        rms = float(np.sqrt(np.mean(noisy ** 2))) + 1e-8
        g = self.target_rms / rms
        peak = max(float(np.abs(noisy).max()), float(np.abs(dry).max()), 1e-8)
        g = min(g, 0.95 / peak)
        noisy = noisy * g
        dry = dry * g
        return (torch.from_numpy(noisy).float().unsqueeze(0),
                torch.from_numpy(dry).float().unsqueeze(0))
