#!/usr/bin/env bash
set -e

# 配置项（可按需修改）
MASTER_ADDR=${MASTER_ADDR:-$(hostname -I | awk '{print $1}')}
MASTER_PORT=${MASTER_PORT:-29511}
GPUS=${GPUS:-4}
FEATURES_ROOT=${FEATURES_ROOT:-/hpc2ssd/JH_DATA/spooler/xfeng733/FakeAV_feats}
BATCH_SIZE=${BATCH_SIZE:-64}       # per GPU
EPOCHS=${EPOCHS:-50}
SEQ_LEN=${SEQ_LEN:-256}
HIDDEN=${HIDDEN:-768}
NUM_LAYERS=${NUM_LAYERS:-4}
NUM_HEADS=${NUM_HEADS:-8}
LR=${LR:-3e-4}
WDECAY=${WDECAY:-1e-4}
WARMUP=${WARMUP:-3}
DROPOUT=${DROPOUT:-0.1}
CONTRAST_W=${CONTRAST_W:-0.5}
NUM_WORKERS=${NUM_WORKERS:-8}
SAVE_PATH=${SAVE_PATH:-best_model_4gpu_50ep.pt}

export MASTER_ADDR MASTER_PORT
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
export NCCL_ASYNC_ERROR_HANDLING=1
export PYTHONNOUSERSITE=1
export ALBUMENTATIONS_DISABLE_VERSION_CHECK=1

echo "MASTER_ADDR=${MASTER_ADDR}"
echo "MASTER_PORT=${MASTER_PORT}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "FEATURES_ROOT=${FEATURES_ROOT}"

torchrun --nproc_per_node=${GPUS} --master_addr ${MASTER_ADDR} --master_port ${MASTER_PORT} \
  train_crossmodal_ddp.py \
  --features_root ${FEATURES_ROOT} \
  --batch_size ${BATCH_SIZE} \
  --epochs ${EPOCHS} \
  --seq_len ${SEQ_LEN} \
  --hidden ${HIDDEN} \
  --num_layers ${NUM_LAYERS} \
  --num_heads ${NUM_HEADS} \
  --lr ${LR} \
  --weight_decay ${WDECAY} \
  --warmup_epochs ${WARMUP} \
  --dropout ${DROPOUT} \
  --use_contrastive \
  --contrastive_weight ${CONTRAST_W} \
  --label_smoothing 0.1 \
  --num_workers ${NUM_WORKERS} \
  --pin_memory \
  --save_path ${SAVE_PATH}

