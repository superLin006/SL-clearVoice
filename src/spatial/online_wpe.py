import numpy as np
from nara_wpe.utils import istft, stft
from nara_wpe.wpe import OnlineWPE

SR = 16000


def wpe_online(x, taps=10, delay=3, alpha=0.9999, warmup=0):
    """x: [T, M] -> online (causal) WPE output [T, M]."""
    y = stft(x.T, size=512, shift=256)  # [M,T,F]
    m, t, f = y.shape
    wpe = OnlineWPE(taps=taps, delay=delay, alpha=alpha, channel=m, frequency_bins=f)
    outs = []
    for i in range(t):
        pred = wpe.step_frame(y[:, i, :].T)  # [F,M]
        outs.append(pred.T)                  # [M,F]
    z = np.stack(outs, axis=1)               # [M,T,F]
    out = istft(z, size=512, shift=256).T    # [T,M]
    out = out[:len(x)]
    if warmup and warmup < len(out):
        out[:warmup] = x[:warmup]
    return out.astype(np.float32)
