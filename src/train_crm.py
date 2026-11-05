import argparse
import os
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from crm_model import CRMFilter, CRMFilterV2
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
    ap.add_argument('--root', default='<TRAIN_ROOT>/distill_data3s3')
    ap.add_argument('--out_dir', default='<TRAIN_ROOT>/ulunas_train_code/ckpt_crm')
    ap.add_argument('--epochs', type=int, default=12)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--max_steps', type=int, default=0)
    ap.add_argument('--teacher_w', type=float, default=1.0)
    ap.add_argument('--dry_w', type=float, default=0.2)
    ap.add_argument('--ch', type=int, default=64)
    ap.add_argument('--blocks', type=int, default=2)
    ap.add_argument('--dilations', default='1,2,4,8')
    ap.add_argument('--arch', default='crm')
    args = ap.parse_args()

    local_rank = int(os.environ['LOCAL_RANK'])
    rank = int(os.environ['RANK'])
    world = int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(local_rank)
    dist.init_process_group('nccl', rank=rank, world_size=world)
    torch.backends.cudnn.benchmark = True

    train_ds = DistillDataset(args.root, 'train')
    val_ds = DistillDataset(args.root, 'val')
    if rank == 0:
        print('train', len(train_ds), 'val', len(val_ds), flush=True)
    sampler = DistributedSampler(train_ds, num_replicas=world, rank=rank, shuffle=True, drop_last=True)
    dl = DataLoader(train_ds, batch_size=args.batch, sampler=sampler, num_workers=args.workers,
                    pin_memory=True, drop_last=True, persistent_workers=args.workers > 0)

    if args.arch == 'crm2':
        model = CRMFilterV2(ch=args.ch, blocks=args.blocks).cuda(local_rank)
    else:
        model = CRMFilter(ch=args.ch, dilations=tuple(int(x) for x in args.dilations.split(','))).cuda(local_rank)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    total = max(1, args.epochs * len(dl))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total, eta_min=args.lr * 0.1)
    os.makedirs(args.out_dir, exist_ok=True)

    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        model.train()
        t0 = time.time()
        run = 0.0
        n = 0
        stop = False
        for noisy, teacher, dry in dl:
            noisy = noisy.squeeze(1).cuda(local_rank, non_blocking=True)
            teacher = teacher.squeeze(1).cuda(local_rank, non_blocking=True)
            dry = dry.squeeze(1).cuda(local_rank, non_blocking=True)
            est = model(noisy)
            L = min(est.shape[-1], teacher.shape[-1], dry.shape[-1])
            est, teacher, dry = est[..., :L], teacher[..., :L], dry[..., :L]
            loss = args.teacher_w * (-si_sdr(est, teacher).mean() + 0.2 * spec_l1(est, teacher, model.module))
            loss = loss + args.dry_w * (-si_sdr(est, dry).mean() + 0.1 * spec_l1(est, dry, model.module))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            run += float(loss.detach())
            n += 1
            if rank == 0 and n % 20 == 0:
                print(f'ep {epoch} step {n}/{len(dl)} loss {run/n:.4f} {time.time()-t0:.0f}s', flush=True)
            if args.max_steps and n >= args.max_steps:
                stop = True
                break
        if rank == 0 and len(val_ds) > 0:
            model.eval()
            vals = []
            with torch.no_grad():
                for i in range(min(64, len(val_ds))):
                    dn, dt, dd = val_ds[i]
                    de = model.module(dn.cuda())[0]
                    L = min(de.shape[-1], dt.shape[-1])
                    vals.append((float(si_sdr(de[..., :L], dt.squeeze(0)[:L].cuda())),
                                 float(si_sdr(de[..., :L], dd.squeeze(0)[:L].cuda()))))
            a = np.array(vals)
            print(f'ep {epoch} val_si_sdr teacher {a[:,0].mean():.2f} dry {a[:,1].mean():.2f}', flush=True)
            torch.save({'model': model.module.state_dict(), 'epoch': epoch},
                       os.path.join(args.out_dir, f'crm_ep{epoch}.pt'))
            print(f'ep {epoch} saved {time.time()-t0:.0f}s', flush=True)
        if stop:
            break
    dist.destroy_process_group()


if __name__ == '__main__':
    main()