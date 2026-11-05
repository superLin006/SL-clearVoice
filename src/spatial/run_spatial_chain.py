import argparse
import os
import sys

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wpe_ds import beamform, read_8ch, wpe_6ch


def rms_match(y, x):
    return y * (np.sqrt((x ** 2).mean() + 1e-12) / (np.sqrt((y ** 2).mean()) + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True, help='8-channel wav')
    ap.add_argument('--output', required=True)
    ap.add_argument('--theta', type=float, default=90.0)
    ap.add_argument('--radius', type=float, default=0.042)
    ap.add_argument('--ulunas-root', required=True)
    ap.add_argument('--ulunas-ckpt', required=True)
    ap.add_argument('--crm-ckpt', required=True)
    args = ap.parse_args()

    sys.path.insert(0, args.ulunas_root)
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from ulunas import ULUNAS
    from crm_model import CRMFilterV2

    x = read_8ch(args.input)
    y = wpe_6ch(x)
    z = beamform(y, theta_deg=args.theta, radius=args.radius)

    ul = ULUNAS().cuda().eval()
    ck = torch.load(args.ulunas_ckpt, map_location='cpu')
    ul.load_state_dict(ck['model'] if isinstance(ck, dict) and 'model' in ck else ck)
    with torch.inference_mode():
        u = ul(torch.from_numpy(z)[None].cuda()).squeeze().cpu().numpy()[:len(z)]
    u = rms_match(u, z)

    crm = CRMFilterV2(ch=128).cuda().eval()
    ck = torch.load(args.crm_ckpt, map_location='cpu')
    crm.load_state_dict({k.replace('module.', ''): v for k, v in ck['model'].items()})
    q = u * (0.04 / (np.sqrt((u ** 2).mean()) + 1e-12))
    with torch.inference_mode():
        q = crm(torch.from_numpy(q)[None].cuda()).squeeze().cpu().numpy()[:len(u)]
    q = rms_match(q, z)
    sf.write(args.output, q.astype(np.float32), 16000)
    print('wrote', args.output, len(q))


if __name__ == '__main__':
    main()
