import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from ulunas import DPGRNN
except Exception:
    DPGRNN = None


class CausalConv(nn.Module):
    def __init__(self, cin, cout, kt=3, kf=3, dilation=1):
        super().__init__()
        self.pad_t = (kt - 1) * dilation
        self.pad_f = kf // 2
        self.conv = nn.Conv2d(cin, cout, (kt, kf), padding=(0, self.pad_f), dilation=(dilation, 1))

    def forward(self, x):
        x = F.pad(x, (0, 0, self.pad_t, 0))
        return self.conv(x)


class CRMFilter(nn.Module):
    """Complex residual mask applied to an already-enhanced complex STFT."""

    def __init__(self, ch=64, n_fft=512, hop=256, win=512, dilations=(1, 2, 4, 8)):
        super().__init__()
        self.n_fft = n_fft
        self.hop = hop
        self.win = win
        self.inp = CausalConv(3, ch, 3, 3, 1)
        self.blocks = nn.ModuleList([
            CausalConv(ch, ch, 3, 3, d) for d in dilations
        ])
        self.act = nn.PReLU()
        self.out = nn.Conv2d(ch, 2, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def _stft(self, x):
        return torch.stft(x, self.n_fft, self.hop, self.win,
                          torch.hann_window(self.win, device=x.device, dtype=x.dtype),
                          return_complex=True)

    def _istft(self, z, n):
        return torch.istft(z, self.n_fft, self.hop, self.win,
                           torch.hann_window(self.win, device=z.device, dtype=torch.float32),
                           length=n)

    def forward(self, x):
        n = x.shape[-1]
        X = self._stft(x)  # (B,F,T)
        Xc = torch.view_as_real(X).permute(0, 3, 2, 1)  # (B,2,T,F)
        mag = torch.log10(torch.norm(Xc, dim=1, keepdim=True).clamp(1e-8))
        h = torch.cat([Xc, mag], dim=1)
        h = self.act(self.inp(h))
        for b in self.blocks:
            h = h + self.act(b(h))
        d = 2.0 * torch.tanh(self.out(h))  # (B,2,T,F)
        mr = 1.0 + d[:, 0]
        mi = d[:, 1]
        xr, xi = Xc[:, 0], Xc[:, 1]
        yr = xr * mr - xi * mi
        yi = xr * mi + xi * mr
        Y = torch.complex(yr, yi).permute(0, 2, 1)  # (B,F,T)
        return self._istft(Y, n)


class CRMFilterV2(nn.Module):
    """Complex residual mask with dual-path GRU temporal context."""

    def __init__(self, ch=128, n_fft=512, hop=256, win=512, blocks=2):
        super().__init__()
        assert DPGRNN is not None
        self.n_fft = n_fft
        self.hop = hop
        self.win = win
        self.inp = CausalConv(3, ch, 3, 3, 1)
        self.conv1 = CausalConv(ch, ch, 3, 3, 1)
        self.blocks = blocks
        self.dp1 = DPGRNN(ch, 257, ch)
        self.dp2 = DPGRNN(ch, 257, ch) if blocks >= 2 else None
        self.conv2 = CausalConv(ch, ch, 3, 3, 2)
        self.conv3 = CausalConv(ch, ch, 3, 3, 4)
        self.act = nn.PReLU()
        self.out = nn.Conv2d(ch, 2, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def _stft(self, x):
        return torch.stft(x, self.n_fft, self.hop, self.win,
                          torch.hann_window(self.win, device=x.device, dtype=x.dtype),
                          return_complex=True)

    def _istft(self, z, n):
        return torch.istft(z, self.n_fft, self.hop, self.win,
                           torch.hann_window(self.win, device=z.device, dtype=torch.float32),
                           length=n)

    def forward(self, x):
        n = x.shape[-1]
        X = self._stft(x)
        Xc = torch.view_as_real(X).permute(0, 3, 2, 1)
        mag = torch.log10(torch.norm(Xc, dim=1, keepdim=True).clamp(1e-8))
        h = torch.cat([Xc, mag], dim=1)
        h = self.act(self.inp(h))
        h = h + self.act(self.conv1(h))
        h = self.dp1(h)
        if self.dp2 is not None:
            h = self.dp2(h)
        h = h + self.act(self.conv2(h))
        h = h + self.act(self.conv3(h))
        d = 2.0 * torch.tanh(self.out(h))
        mr = 1.0 + d[:, 0]
        mi = d[:, 1]
        xr, xi = Xc[:, 0], Xc[:, 1]
        yr = xr * mr - xi * mi
        yi = xr * mi + xi * mr
        Y = torch.complex(yr, yi).permute(0, 2, 1)
        return self._istft(Y, n)

class CRM2Core(nn.Module):
    """ONNX-friendly core: complex spectrum in, enhanced complex spectrum out."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, xc):
        mag = torch.log10(torch.norm(xc, dim=1, keepdim=True).clamp(1e-8))
        h = torch.cat([xc, mag], dim=1)
        h = self.net.act(self.net.inp(h))
        h = h + self.net.act(self.net.conv1(h))
        h = self.net.dp1(h)
        if self.net.dp2 is not None:
            h = self.net.dp2(h)
        h = h + self.net.act(self.net.conv2(h))
        h = h + self.net.act(self.net.conv3(h))
        d = 2.0 * torch.tanh(self.net.out(h))
        mr = 1.0 + d[:, 0]
        mi = d[:, 1]
        xr, xi = xc[:, 0], xc[:, 1]
        return torch.stack([xr * mr - xi * mi, xr * mi + xi * mr], dim=1)
