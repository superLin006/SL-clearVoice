#!/bin/bash
set -euo pipefail
# 200k CRM v2 训练启动脚本（按需改 ROOT/OUT/GPUS）
ROOT=${ROOT:-<TRAIN_ROOT>/distill_200k_train}
OUT=${OUT:-<TRAIN_ROOT>/ulunas_train_code/ckpt_crm_200k}
GPUS=${GPUS:-2,3,4,5,6,7}
NPROC=${NPROC:-6}
EPOCHS=${EPOCHS:-12}
BATCH=${BATCH:-16}
LR=${LR:-3e-4}
PORT=${PORT:-29522}
cd "$(dirname "$0")/../src"
CUDA_VISIBLE_DEVICES="$GPUS" torchrun --nproc_per_node="$NPROC" --master_port="$PORT" \
  train_crm.py --root "$ROOT" --out_dir "$OUT" \
  --epochs "$EPOCHS" --batch "$BATCH" --workers 6 --lr "$LR" \
  --teacher_w 0 --dry_w 1.0 --arch crm2 --ch 128
