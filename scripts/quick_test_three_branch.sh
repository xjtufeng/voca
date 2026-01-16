#!/bin/bash
#
# Quick test script for three-branch model (5 epochs, small data)
# Use this to verify everything works before full training
#

source ~/.bashrc
conda activate voca

FEATURES_ROOT="/hpc2ssd/JH_DATA/spooler/xfeng733/FakeAV_feats"
OUTPUT_DIR="checkpoints/quick_test"
LOG_DIR="logs"

mkdir -p ${OUTPUT_DIR}
mkdir -p ${LOG_DIR}

echo "=========================================="
echo "Quick Test (5 epochs)"
echo "=========================================="

python train_three_branch.py \
  --features_root ${FEATURES_ROOT} \
  --splits train dev \
  --batch_size 8 \
  --epochs 5 \
  --lr 1e-3 \
  --max_frames 100 \
  --d_model 256 \
  --nhead 4 \
  --cm_layers 2 \
  --ao_layers 2 \
  --vo_layers 2 \
  --fusion_method weighted \
  --output_dir ${OUTPUT_DIR} \
  --save_every 2 \
  --num_workers 2 \
  2>&1 | tee ${LOG_DIR}/quick_test_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "Quick test completed!"


