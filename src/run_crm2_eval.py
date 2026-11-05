import glob
import os
import sys

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from crm_model import CRMFilterV2

DATA_ROOT = os.environ.get('SL_DATA_ROOT', './data')
ROOT = f'{DATA_ROOT}/out/fair_eval'
SR = 16000
CKPT = sys.argv[1]
TAG = sys.argv[2] if len(sys.argv) > 2 else 'crm2'
CH = int(sys.argv[3]) if len(sys.argv) > 3 else 128

m = CRMFilterV2(ch=CH).cuda().eval()
ck = torch.load(CKPT, map_location='cpu')
sd = ck['model'] if isinstance(ck, dict) and 'model' in ck else ck
m.load_state_dict({k.replace('module.', ''): v for k, v in sd.items()})


def enhance(x):
    rms_in = np.sqrt((x ** 2).mean() + 1e-12)
    z = x * (0.04 / rms_in)
    with torch.no_grad():
        y = m(torch.from_numpy(z)[None].float().cuda()).squeeze().cpu().numpy()[:len(x)]
    y = y * (rms_in / 0.04)
    return (y * (np.sqrt((x ** 2).mean()) / (np.sqrt((y ** 2).mean()) + 1e-12))).astype(np.float32)


x = sf.read(f'{DATA_ROOT}/out/variants/full_ulunas_raw.wav', dtype='float32')[0]
sf.write(f'{DATA_ROOT}/out/full41_models/{TAG}.wav', enhance(x), SR)

for split in ('rev', 'rev_noise'):
    out = f'{ROOT}/{TAG}_{split}'
    os.makedirs(out, exist_ok=True)
    files = sorted(glob.glob(f'{ROOT}/ulunas_{split}/*.wav'))
    for f in files:
        x = sf.read(f, dtype='float32')[0]
        sf.write(os.path.join(out, os.path.basename(f)), enhance(x), SR)
    print(split, len(files), 'done', flush=True)
print('done')
