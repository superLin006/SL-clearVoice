import glob
import os

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset


class DistillDataset(Dataset):
    def __init__(self, root, split='train', seg=32000):
        self.root = root
        self.seg = seg
        self.teacher = os.path.join(root, 'teacher')
        names = [os.path.basename(p) for p in sorted(glob.glob(os.path.join(root, 'noisy', '*.wav')))]
        self.names = [n for n in names if n.startswith(split) and os.path.exists(os.path.join(self.teacher, n))]

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        out = []
        for sub in ('noisy', 'teacher', 'dry'):
            x, _ = sf.read(os.path.join(self.root, sub, name), dtype='float32')
            if x.ndim > 1:
                x = x[:, 0]
            if len(x) < self.seg:
                x = np.pad(x, (0, self.seg - len(x)))
            else:
                x = x[:self.seg]
            out.append(torch.from_numpy(x).float().unsqueeze(0))
        return out[0], out[1], out[2]
