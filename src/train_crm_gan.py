import argparse
import glob
import os
import random
import sys
import time

import numpy as np
import soundfile as sf
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from crm_model import CRMFilterV2
from crm_gan import MSSTFTDisc, d_loss_ls, g_loss_ls, feat_loss
from distill_dataset import DistillDataset


def si_sdr(est, ref):
    est = est - est.mean(dim=-1, keepdim=True)
    ref = ref - ref.mean(dim=-1, keepdim=True)
    alpha = (est * ref).sum(dim=-1, keepdim=True) / (ref.pow(2).sum(dim=-1, keepdim=True) + 1e-8)
    target = alpha * ref
    return 10 * torch.log10((target.pow(2).sum(dim=-1) + 1e-8) / ((est - target).pow(2).sum(dim=-1) + 1e-8))


def spec_l1(est, ref, model):
    E = model._stft(est)
    R = model._stft(ref)
    return (E - R).abs().mean() + (E.abs() - R.abs()).abs().mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=os.environ.get('CRM_ROOT', '/data/distill_200k_train'))
    ap.add_argument('--out_dir', default=os.environ.get('CRM_OUT', './ckpt_crm_gan'))
    ap.add_argument('--epochs', type=int, default=2)
    ap.add_argument('--batch', type=int, default=12)
    ap.add_argument('--lr_g', type=float, default=2e-5)
    ap.add_argument('--lr_d', type=float, default=1e-4)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--max_steps', type=int, default=0)
    ap.add_argument('--init', default='')
    ap.add_argument('--adv_w', type=float, default=0.05)
    ap.add_argument('--fm_w', type=float, default=1.0)
    ap.add_argument('--spec_w', type=float, default=0.1)
    ap.add_argument('--teacher_real_dir', default='')
    ap.add_argument('--teacher_real_p', type=float, default=0.0)
    args = ap.parse_args()

    local_rank = int(os.environ['LOCAL_RANK'])
    rank = int(os.environ['RANK'])
    world = int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(local_rank)
    dist.init_process_group('nccl', rank=rank, world_size=world)

    train_ds = DistillDataset(args.root, 'train')
    val_ds = DistillDataset(args.root, 'val')
    if rank == 0:
        print('train', len(train_ds), 'val', len(val_ds), flush=True)
    sampler = DistributedSampler(train_ds, num_replicas=world, rank=rank, shuffle=True, drop_last=True)
    dl = DataLoader(train_ds, batch_size=args.batch, sampler=sampler, num_workers=args.workers,
                    pin_memory=True, drop_last=True, persistent_workers=args.workers > 0)

    G = CRMFilterV2(ch=128, blocks=2).cuda(local_rank)
    if args.init:
        ck = torch.load(args.init, map_location='cpu')
        sd = ck['model'] if isinstance(ck, dict) and 'model' in ck else ck
        G.load_state_dict({k.replace('module.', ''): v for k, v in sd.items()})
        if rank == 0:
            print('init from', args.init, flush=True)
    D = MSSTFTDisc(ch=32).cuda(local_rank)
    G = DDP(G, device_ids=[local_rank], find_unused_parameters=False)
    D = DDP(D, device_ids=[local_rank], find_unused_parameters=False)

    opt_g = torch.optim.AdamW(G.parameters(), lr=args.lr_g, betas=(0.5, 0.9), weight_decay=0.0)
    opt_d = torch.optim.AdamW(D.parameters(), lr=args.lr_d, betas=(0.5, 0.9), weight_decay=0.0)
    os.makedirs(args.out_dir, exist_ok=True)
    teacher_files = sorted(glob.glob(os.path.join(args.teacher_real_dir, '*.wav'))) if args.teacher_real_dir else []
    if rank == 0:
        print('teacher_real', len(teacher_files), flush=True)

    def make_real(dry_batch, L):
        if teacher_files and random.random() < args.teacher_real_p:
            t, _ = sf.read(random.choice(teacher_files), dtype='float32')
            if t.ndim > 1:
                t = t[:, 0]
            if len(t) < L:
                t = np.pad(t, (0, L - len(t)))
            t = t[:L]
            t = t * (0.04 / (np.sqrt((t ** 2).mean()) + 1e-8))
            return torch.from_numpy(t).float().cuda(local_rank)[None]
        return dry_batch

    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        G.train()
        D.train()
        t0 = time.time()
        run_g = run_d = run_fm = run_s = 0.0
        n = 0
        stop = False
        for noisy, teacher, dry in dl:
            noisy = noisy.squeeze(1).cuda(local_rank, non_blocking=True)
            dry = dry.squeeze(1).cuda(local_rank, non_blocking=True)
            est = G(noisy)
            L = min(est.shape[-1], dry.shape[-1])
            est, dry_ = est[..., :L], dry[..., :L]
            real = make_real(dry_, L)

            fake = est.detach()
            real_outs, _ = D(real)
            fake_outs, _ = D(fake)
            loss_d = d_loss_ls(real_outs, fake_outs)
            opt_d.zero_grad(set_to_none=True)
            loss_d.backward()
            torch.nn.utils.clip_grad_norm_(D.parameters(), 5.0)
            opt_d.step()

            real_outs, real_feats = D(real)
            fake_outs, fake_feats = D(est)
            loss_fid = -si_sdr(est, dry_).mean() + args.spec_w * spec_l1(est, dry_, G.module)
            loss_fm = feat_loss(real_feats, fake_feats)
            loss_g = loss_fid + args.adv_w * g_loss_ls(fake_outs) + args.fm_w * loss_fm
            opt_g.zero_grad(set_to_none=True)
            loss_g.backward()
            torch.nn.utils.clip_grad_norm_(G.parameters(), 5.0)
            opt_g.step()

            run_g += float(loss_g.detach())
            run_d += float(loss_d.detach())
            run_fm += float(loss_fm.detach())
            run_s += float(loss_fid.detach())
            n += 1
            if rank == 0 and n % 20 == 0:
                print(f'ep {epoch} step {n}/{len(dl)} g {run_g/n:.4f} d {run_d/n:.4f} '
                      f'fid {run_s/n:.4f} fm {run_fm/n:.4f} {time.time()-t0:.0f}s', flush=True)
            if args.max_steps and n >= args.max_steps:
                stop = True
                break

        if rank == 0 and len(val_ds) > 0:
            G.eval()
            vals = []
            with torch.no_grad():
                for i in range(min(64, len(val_ds))):
                    dn, dt, dd = val_ds[i]
                    de = G.module(dn.cuda())[0]
                    L = min(de.shape[-1], dd.shape[-1])
                    vals.append(float(si_sdr(de[..., :L], dd.squeeze(0)[:L].cuda())))
            print(f'ep {epoch} val_dry_si_sdr {np.mean(vals):.2f}', flush=True)
            torch.save({'model': G.module.state_dict(), 'epoch': epoch},
                       os.path.join(args.out_dir, f'crm_gan_ep{epoch}.pt'))
            print(f'ep {epoch} saved {time.time()-t0:.0f}s', flush=True)
        if stop:
            break
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
