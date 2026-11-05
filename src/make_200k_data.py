import glob
import os
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ulunas_dataset import RIRNoiseDataset

shard = int(sys.argv[1])
nshard = int(sys.argv[2])
DATA_ROOT = os.environ.get('CRM_DATA_ROOT', '/data')
RIR_ROOT = os.environ.get('CRM_RIR_ROOT', '/data/RIRS_NOISES')
out = os.environ.get('CRM_200K_OUT', '/data/distill_200k')
os.makedirs(os.path.join(out, 'noisy'), exist_ok=True)
os.makedirs(os.path.join(out, 'dry'), exist_ok=True)

thchs = sorted(glob.glob(f'{DATA_ROOT}/data_thchs30/train/*.wav'))
ais = sorted(glob.glob(f'{DATA_ROOT}/train/wav/*/*.wav'))
rirs = sorted(glob.glob(f'{RIR_ROOT}/simulated_rirs/**/*.wav', recursive=True))
noises = sorted(glob.glob(f'{DATA_ROOT}/musan/noise/**/*.wav', recursive=True))
base = thchs + ais
n = 200000
files = [base[i % len(base)] for i in range(n)]
ds = RIRNoiseDataset(files, rirs, noises, seg=32000, p_rir=1.0, p_noise=0.85,
                     snr_range=(0.0, 20.0), seed=2028)
start = shard * n // nshard
end = (shard + 1) * n // nshard
print('shard', shard, 'range', start, end, 'speech', len(base), 'rirs', len(rirs), flush=True)
for i in range(start, end):
    noisy, dry = ds[i]
    name = f'train_{i:06d}'
    sf.write(os.path.join(out, 'noisy', name + '.wav'), noisy.squeeze(0).numpy(), 16000, subtype='PCM_16')
    sf.write(os.path.join(out, 'dry', name + '.wav'), dry.squeeze(0).numpy(), 16000, subtype='PCM_16')
    if (i + 1) % 1000 == 0:
        print(i + 1, flush=True)
print('done', end - start, flush=True)