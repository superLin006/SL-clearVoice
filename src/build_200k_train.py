import glob
import os

SRC = os.environ.get('CRM_200K_OUT', '/data/distill_200k')
D3 = os.environ.get('CRM_3S3', '/data/distill_data3s3')
OUT = os.environ.get('CRM_ROOT', '/data/distill_200k_train')
for sub in ('noisy', 'teacher', 'dry'):
    os.makedirs(f'{OUT}/{sub}', exist_ok=True)

names = sorted(os.path.basename(p) for p in glob.glob(f'{SRC}/noisy/train_*.wav'))
for n in names:
    os.symlink(f'{SRC}/ulunas/{n}', f'{OUT}/noisy/{n}')
    os.symlink(f'{SRC}/dry/{n}', f'{OUT}/teacher/{n}')
    os.symlink(f'{SRC}/dry/{n}', f'{OUT}/dry/{n}')
val = sorted(os.path.basename(p) for p in glob.glob(f'{D3}/noisy/val_*.wav'))
for n in val:
    for sub in ('noisy', 'teacher', 'dry'):
        os.symlink(f'{D3}/{sub}/{n}', f'{OUT}/{sub}/{n}')
print('train', len(names), 'val', len(val))