import torch
import torch.nn as nn
import torch.nn.functional as F


class STFTDisc(nn.Module):
    def __init__(self, n_fft, hop, win, ch=32):
        super().__init__()
        self.n_fft = n_fft
        self.hop = hop
        self.win = win
        self.register_buffer('window', torch.hann_window(win), persistent=False)
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.utils.parametrizations.spectral_norm(nn.Conv2d(2, ch, (3, 9), stride=(1, 2), padding=(1, 4))),
                nn.LeakyReLU(0.2, inplace=True)),
            nn.Sequential(
                nn.utils.parametrizations.spectral_norm(nn.Conv2d(ch, ch * 2, (3, 9), stride=(2, 2), padding=(1, 4))),
                nn.LeakyReLU(0.2, inplace=True)),
            nn.Sequential(
                nn.utils.parametrizations.spectral_norm(nn.Conv2d(ch * 2, ch * 4, (3, 9), stride=(2, 2), padding=(1, 4))),
                nn.LeakyReLU(0.2, inplace=True)),
        ])
        self.out = nn.utils.parametrizations.spectral_norm(nn.Conv2d(ch * 4, 1, (3, 3), padding=(1, 1)))

    def forward(self, x):
        spec = torch.stft(x, self.n_fft, hop_length=self.hop, win_length=self.win,
                          window=self.window.to(x.device), return_complex=True, center=True)
        h = torch.view_as_real(spec).permute(0, 3, 1, 2)
        feats = []
        for layer in self.convs:
            h = layer(h)
            feats.append(h)
        return self.out(h), feats


class MSSTFTDisc(nn.Module):
    def __init__(self, ch=32):
        super().__init__()
        self.discs = nn.ModuleList([
            STFTDisc(512, 128, 512, ch),
            STFTDisc(1024, 256, 1024, ch),
            STFTDisc(2048, 512, 2048, ch),
        ])

    def forward(self, x):
        outs, feats = [], []
        for d in self.discs:
            o, f = d(x)
            outs.append(o)
            feats.extend(f)
        return outs, feats


def d_loss_ls(real_outs, fake_outs):
    loss = 0.0
    for r, f in zip(real_outs, fake_outs):
        loss = loss + 0.5 * (F.mse_loss(r, torch.ones_like(r)) + F.mse_loss(f, torch.zeros_like(f)))
    return loss / len(real_outs)


def g_loss_ls(fake_outs):
    loss = 0.0
    for f in fake_outs:
        loss = loss + 0.5 * F.mse_loss(f, torch.ones_like(f))
    return loss / len(fake_outs)


def feat_loss(real_feats, fake_feats):
    loss = 0.0
    for r, f in zip(real_feats, fake_feats):
        loss = loss + F.l1_loss(f, r.detach())
    return loss / len(real_feats)
