import os
import sys

import numpy as np
import soundfile as sf
from speechmos import dnsmos

SR = 16000
DATA_ROOT = os.environ.get('SL_DATA_ROOT', '<SL_DATA_ROOT>')
SEGS = ['seg1m', 'seg3m', 'seg5m']


def rd(p):
    x, _ = sf.read(p, dtype='float32')
    return x[:, 0] if x.ndim > 1 else x


def show(name, paths):
    vals = []
    for p in paths:
        if not os.path.exists(p):
            print('missing', p)
            continue
        d = dnsmos.run(rd(p), sr=SR)
        vals.append([float(d['ovrl_mos']), float(d['sig_mos']), float(d['bak_mos'])])
    if not vals:
        return
    a = np.array(vals)
    print(name, len(vals), *[round(float(v), 3) for v in a.mean(0)])


groups = {}
for tag in sys.argv[1:]:
    if tag == 'ulunas':
        paths = [f'{DATA_ROOT}/out/ulunas/{s}_ulunas.wav' for s in SEGS]
    elif tag in ('B_nvidia_v4', 'B'):
        paths = [f'{DATA_ROOT}/out/distance_segments/seg_{s[3:]}_nvidia_v4.wav' for s in SEGS]
    else:
        paths = [f'{DATA_ROOT}/out/real_eval/{tag}/{s}.wav' for s in SEGS]
    show(tag, paths)
