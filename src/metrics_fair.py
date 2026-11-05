import glob
import os
import sys

import numpy as np
import soundfile as sf
import torch
import torchaudio
from pesq import pesq
from pystoi import stoi
from speechmos import dnsmos

SR = 16000
DATA_ROOT = os.environ.get('SL_DATA_ROOT', './data')
ROOT = f'{DATA_ROOT}/out/fair_eval'
DRY = f'{ROOT}/dry'

USE_SQUIM = os.environ.get('SQUIM', '1') == '1'
LIMIT = int(os.environ.get('LIMIT', '0'))
if USE_SQUIM:
    bundle = torchaudio.pipelines.SQUIM_SUBJECTIVE
    squim = bundle.get_model().cuda().eval()


def squim_score(y, ref):
    t = torch.from_numpy(y).float().cuda()[None]
    r = torch.from_numpy(ref).float().cuda()[None]
    with torch.no_grad():
        return float(squim(t, r)[0])


def rd(p):
    x, _ = sf.read(p, dtype='float32')
    return x[:, 0] if x.ndim > 1 else x


def score(folder):
    rows = []
    files = sorted(glob.glob(f'{ROOT}/{folder}/*.wav'))
    if LIMIT:
        files = files[:LIMIT]
    for f in files:
        ref = rd(os.path.join(DRY, os.path.basename(f)))
        y = rd(f)
        L = min(len(ref), len(y))
        p = pesq(SR, ref[:L], y[:L], 'wb')
        s = stoi(ref[:L], y[:L], SR)
        d = dnsmos.run(y, sr=SR)
        q = squim_score(y[:L], ref[:L]) if USE_SQUIM else float('nan')
        rows.append((p, s, q, float(d['ovrl_mos']), float(d['sig_mos']), float(d['bak_mos'])))
    a = np.array(rows)
    return a.mean(0), len(rows)


if __name__ == '__main__':
    print('folder n PESQ STOI SQUIM OVRL SIG BAK')
    for folder in sys.argv[1:]:
        m, n = score(folder)
        print(folder, n, *[round(float(v), 3) for v in m], flush=True)
