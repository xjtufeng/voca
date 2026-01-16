#!/bin/bash
# 4-GPU training with DFD-FCG CLIP integration
# Usage: bash scripts/train_4gpu_clip.sh

set -e

# Activate conda environment
source ~/.bashrc
conda activate voca

# Paths
FEATURES_ROOT="/hpc2ssd/JH_DATA/spooler/xfeng733/FakeAV_feats"
VIDEO_ROOT="/hpc2ssd/JH_DATA/spooler/xfeng733/FakeAV_stage"
OUTPUT_DIR="checkpoints/three_branch_4gpu_clip"
LOG_DIR="logs"

# Create directories
mkdir -p ${OUTPUT_DIR}
mkdir -p ${LOG_DIR}

# NCCL settings (optional)
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0

echo "=========================================="
echo "4-GPU Training with CLIP"
echo "=========================================="
echo "Features: ${FEATURES_ROOT}"
echo "Videos:   ${VIDEO_ROOT}"
echo "Output:   ${OUTPUT_DIR}"
echo "=========================================="

# Training
torchrun \
  --nproc_per_node=4 \
  --master_port=29500 \
  train_three_branch.py \
  --features_root ${FEATURES_ROOT} \
  --video_root ${VIDEO_ROOT} \
  --load_video_frames \
  --use_dfdfcg \
  --dfdfcg_freeze \
  --batch_size 12 \
  --epochs 100 \
  --lr 1e-3 \
  --weight_decay 0.01 \
  --warmup_epochs 5 \
  --max_frames 150 \
  --d_model 512 \
  --nhead 8 \
  --cm_layers 4 \
  --ao_layers 3 \
  --vo_layers 3 \
  --dropout 0.1 \
  --fusion_method weighted \
  --fusion_loss_weight 1.0 \
  --cm_loss_weight 0.3 \
  --ao_loss_weight 0.2 \
  --vo_loss_weight 0.2 \
  --output_dir ${OUTPUT_DIR} \
  --save_every 5 \
  --num_workers 6 \
  2>&1 | tee ${LOG_DIR}/train_4gpu_clip_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "=========================================="
echo "Training completed!"
echo "=========================================="


