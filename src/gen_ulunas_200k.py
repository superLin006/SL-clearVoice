import glob
import os
import sys

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ulunas import ULUNAS

shard = int(sys.argv[1])
nshard = int(sys.argv[2])
root = os.environ.get('CRM_200K_OUT', '/data/distill_200k')
out = os.path.join(root, 'ulunas')
os.makedirs(out, exist_ok=True)
files = sorted(glob.glob(os.path.join(root, 'noisy', '*.wav')))
files = files[shard * len(files) // nshard:(shard + 1) * len(files) // nshard]

model = ULUNAS().cuda().eval()
ck = torch.load(os.environ.get('ULUNAS_CKPT', '/data/model_trained_on_dns3.tar'), map_location='cuda')
model.load_state_dict(ck['model'] if isinstance(ck, dict) and 'model' in ck else ck)
bs = 32
for st in range(0, len(files), bs):
    batch = files[st:st + bs]
    xs = []
    for f in batch:
        x, _ = sf.read(f, dtype='float32')
        xs.append(x[:, 0] if x.ndim > 1 else x)
    inp = np.stack(xs).astype(np.float32)
    with torch.no_grad():
        y = model(torch.from_numpy(inp).cuda()).cpu().numpy()
    for i, f in enumerate(batch):
        yi = y[i][:len(inp[i])]
        yi = yi * (np.sqrt((inp[i] ** 2).mean()) / (np.sqrt((yi ** 2).mean()) + 1e-12))
        sf.write(os.path.join(out, os.path.basename(f)), yi.astype(np.float32), 16000, subtype='PCM_16')
    if (st // bs + 1) % 50 == 0:
        print(st + len(batch), flush=True)
print('done', len(files), flush=True)