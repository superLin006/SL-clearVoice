import numpy as np
import soundfile as sf
from nara_wpe.utils import istft, stft
from nara_wpe.wpe import wpe_v8

SR = 16000


def read_8ch(path, mics=6):
    x, _ = sf.read(path, dtype='float32', always_2d=True)
    return x[:, :mics]


def wpe_6ch(x, taps=10, delay=3, iterations=3, hop=256):
    """x: [T, M] -> WPE-dereverberated [T, M] (already time aligned)."""
    y = stft(x.T, size=512, shift=hop)        # [M, T, F]
    z = wpe_v8(y.transpose(2, 0, 1), taps=taps, delay=delay, iterations=iterations)
    out = istft(z.transpose(1, 2, 0), size=512, shift=hop).T  # [T, M]
    out = out[:len(x)]
    return out.astype(np.float32)


def beamform(y, theta_deg=90.0, radius=0.042, n_mics=6):
    """Delay-and-sum beamformer for a circular array; returns [T]."""
    n, m = y.shape
    win, hop, nfft = 512, 256, 512
    w = np.hanning(win).astype(np.float32)
    pad = win
    yp = np.pad(y, ((pad, pad), (0, 0)))
    nf = 1 + (len(yp) - win) // hop
    idx = np.arange(win)[None, :] + hop * np.arange(nf)[:, None]
    spec = np.fft.rfft(yp[idx] * w[None, :, None], n=nfft, axis=1)  # [T,F,M]
    freqs = np.fft.rfftfreq(nfft, 1 / SR)
    phi = np.deg2rad(np.arange(0, 360, 360 // n_mics))
    tau = -(radius / 343.0) * np.cos(np.deg2rad(theta_deg) - phi)
    steer = np.exp(-1j * 2 * np.pi * freqs[:, None] * tau[None, :])
    fused = np.einsum('tfm,fm->tf', spec, np.conj(steer)) / m

    out = np.zeros(len(yp), np.float64)
    norm = np.zeros(len(yp), np.float64)
    for t in range(nf):
        z = np.fft.irfft(fused[t], n=nfft) * w
        s = t * hop
        out[s:s + win] += z
        norm[s:s + win] += w ** 2
    out = out[pad:pad + n]
    out /= np.maximum(norm[pad:pad + n], 1e-8)
    return out.astype(np.float32)
