import os
import sys

import numpy as np
import soundfile as sf
from pesq import pesq
from pystoi import stoi

DATA_ROOT = os.environ.get('SL_DATA_ROOT', '<SL_DATA_ROOT>')
SEGS = ['seg1m', 'seg3m', 'seg5m']
REFS = [f'{DATA_ROOT}/out/distance_segments/seg_{s[3:]}_ref.wav' for s in SEGS]


def rd(p):
    x, _ = sf.read(p, dtype='float32')
    return x[:, 0] if x.ndim > 1 else x


print('group PESQ STOI')
for tag in sys.argv[1:]:
    if tag == 'ulunas':
        paths = [f'{DATA_ROOT}/out/ulunas/{s}_ulunas.wav' for s in SEGS]
    elif tag in ('B_nvidia_v4', 'B'):
        paths = [f'{DATA_ROOT}/out/distance_segments/seg_{s[3:]}_nvidia_v4.wav' for s in SEGS]
    else:
        paths = [f'{DATA_ROOT}/out/real_eval/{tag}/{s}.wav' for s in SEGS]
    vals = []
    for p, rf in zip(paths, REFS):
        if not os.path.exists(p):
            continue
        r = rd(rf)
        y = rd(p)
        L = min(len(r), len(y))
        r, y = r[:L], y[:L]
        y = y * (np.sqrt((r ** 2).mean()) / (np.sqrt((y ** 2).mean()) + 1e-12))
        vals.append([pesq(16000, r, y, 'wb'), stoi(r, y, 16000)])
    a = np.array(vals)
    print(tag, *[round(float(v), 3) for v in a.mean(0)], flush=True)
