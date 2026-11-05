import os
import sys

import numpy as np
import soundfile as sf
from pesq import pesq
from pystoi import stoi
from speechmos import dnsmos

SR = 16000
DATA_ROOT = os.environ.get('SL_DATA_ROOT', './data')
REF = f'{DATA_ROOT}/out/ref_speech_41s.wav'


def rd(p):
    x, _ = sf.read(p, dtype='float32')
    return x[:, 0] if x.ndim > 1 else x


ref = rd(REF)
print('file PESQ STOI OVRL SIG BAK')
for p in sys.argv[1:]:
    y = rd(p)
    L = min(len(ref), len(y))
    pv = pesq(SR, ref[:L], y[:L], 'wb')
    sv = stoi(ref[:L], y[:L], SR)
    d = dnsmos.run(y, sr=SR)
    print(os.path.basename(p), round(pv, 3), round(sv, 3),
          round(float(d['ovrl_mos']), 2), round(float(d['sig_mos']), 2), round(float(d['bak_mos']), 2), flush=True)
