import sys
import time

import numpy as np

TAPS, DELAY, ALPHA = 10, 3, 0.9999
N_FFT, HOP = 512, 256


def stft(x):
    x = np.pad(x, ((N_FFT // 2, N_FFT // 2), (0, 0)))
    win = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N_FFT) / N_FFT)
    n = 1 + (len(x) - N_FFT) // HOP
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(n)[:, None]
    return np.fft.rfft(x[idx] * win[None, :, None], axis=1), win, len(x)


def istft(spec, win, length):
    n, f, m = spec.shape
    out = np.zeros((n * HOP + N_FFT, m), np.float64)
    norm = np.zeros(n * HOP + N_FFT, np.float64)
    for t in range(n):
        z = np.fft.irfft(spec[t], n=N_FFT, axis=0) * win[:, None]
        s = t * HOP
        out[s:s + N_FFT] += z
        norm[s:s + N_FFT] += win ** 2
    out /= np.maximum(norm, 1e-8)[:, None]
    return out[N_FFT // 2:N_FFT // 2 + length]


def beamform(spec, win, length, theta=86.0, radius=0.042):
    n, f, m = spec.shape
    freqs = np.fft.rfftfreq(N_FFT, 1 / 16000)
    phi = np.deg2rad(np.arange(0, 360, 360 // m))
    tau = -(radius / 343.0) * np.cos(np.deg2rad(theta) - phi)
    a = np.exp(-1j * 2 * np.pi * freqs[:, None] * tau[None, :])
    return np.einsum('tfm,fm->tf', spec, np.conj(a)) / m


def stable_inv(p):
    eps = 1e-10 * np.max(p)
    return np.ones_like(p) if eps == 0 else 1 / np.maximum(p, eps)


class OnlineWPE:
    def __init__(self, f, d):
        self.alpha = ALPHA
        self.taps, self.delay = TAPS, DELAY
        self.inv_cov = np.stack([np.eye(d * TAPS) for _ in range(f)])
        self.filter_taps = np.zeros((f, d * TAPS, d))
        self.power = np.ones(f, dtype=np.float32)
        self.buffer = np.zeros((TAPS + DELAY + 1, f, d), dtype=np.complex128)

    def step(self, frame):
        f, d = frame.shape
        window = self.buffer[:-DELAY - 1].transpose(1, 2, 0).reshape(f, TAPS * d)
        obs = frame[None, ...]
        pred = obs[-1] - np.einsum('fid,fi->fd', np.conjugate(self.filter_taps), window)
        self.buffer = np.roll(self.buffer, -1, axis=0)
        self.buffer[-1] = frame
        self.power = np.mean(np.abs(self.buffer.transpose(1, 2, 0)) ** 2, axis=(1, 2))
        nominator = np.einsum('fij,fj->fi', self.inv_cov, window)
        denominator = self.alpha * self.power.astype(window.dtype)
        denominator += np.einsum('fi,fi->f', np.conjugate(window), nominator)
        self.kalman_gain = nominator * stable_inv(denominator)[:, None]
        self.inv_cov = self.inv_cov - np.einsum('fj,fjm,fi->fim', np.conjugate(window), self.inv_cov, self.kalman_gain, optimize='optimal')
        self.inv_cov /= self.alpha
        self.filter_taps = self.filter_taps + np.einsum('fi,fm->fim', self.kalman_gain, np.conjugate(pred))
        return pred


def online(x):
    spec, win, length = stft(x)
    n, f, d = spec.shape
    w = OnlineWPE(f, d)
    out = np.stack([w.step(spec[t]) for t in range(n)], axis=0)  # [T,F,D]
    return istft(out, win, length)


x = np.load(sys.argv[1]).astype(np.float32)
print('input', x.shape, 'duration', len(x) / 16000, flush=True)
t0 = time.perf_counter()
y6 = online(x)
t1 = time.perf_counter()
spec, win, ln = stft(y6)
ds6 = istft(beamform(spec, win, ln)[..., None], win, ln)[:, 0]
t2 = time.perf_counter()
spec, win, ln = stft(x)
ds = istft(beamform(spec, win, ln)[..., None], win, ln)[:, 0]
t3 = time.perf_counter()
ym = online(ds[:, None])
t4 = time.perf_counter()
dur = len(x) / 16000
print('6ch online WPE', round(t1 - t0, 2), 'RTF', round((t1 - t0) / dur, 3), flush=True)
print('6ch WPE+DS', round(t2 - t1, 2), 'total RTF', round((t2 - t0) / dur, 3), flush=True)
print('DS raw', round(t3 - t2, 2), 'RTF', round((t3 - t2) / dur, 3), flush=True)
print('DS+mono online WPE', round(t4 - t3, 2), 'total RTF', round((t4 - t3) / dur, 3), flush=True)
