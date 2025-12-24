#!/bin/bash
#
# Training script for three-branch baseline (video-level classification)
# Single GPU, no DFD-FCG
#

# 激活环境
source ~/.bashrc
conda activate voca

# 设置路径
FEATURES_ROOT="/hpc2ssd/JH_DATA/spooler/xfeng733/FakeAV_feats"
OUTPUT_DIR="checkpoints/three_branch_baseline"
LOG_DIR="logs"

# 创建输出目录
mkdir -p ${OUTPUT_DIR}
mkdir -p ${LOG_DIR}

# 训练参数
BATCH_SIZE=16
EPOCHS=100
LR=1e-3
MAX_FRAMES=150

# 模型参数
D_MODEL=512
NHEAD=8
CM_LAYERS=4
AO_LAYERS=3
VO_LAYERS=3
FUSION_METHOD="weighted"  # or 'concat' or 'attention'

# Loss weights
FUSION_WEIGHT=1.0
CM_WEIGHT=0.3
AO_WEIGHT=0.2
VO_WEIGHT=0.2

echo "=========================================="
echo "Three-Branch Baseline Training"
echo "=========================================="
echo "Features: ${FEATURES_ROOT}"
echo "Output:   ${OUTPUT_DIR}"
echo "Batch:    ${BATCH_SIZE}"
echo "Epochs:   ${EPOCHS}"
echo "Fusion:   ${FUSION_METHOD}"
echo "=========================================="

# 训练
python train_three_branch.py \
  --features_root ${FEATURES_ROOT} \
  --splits train dev \
  --batch_size ${BATCH_SIZE} \
  --epochs ${EPOCHS} \
  --lr ${LR} \
  --weight_decay 0.01 \
  --warmup_epochs 5 \
  --max_frames ${MAX_FRAMES} \
  --d_model ${D_MODEL} \
  --nhead ${NHEAD} \
  --cm_layers ${CM_LAYERS} \
  --ao_layers ${AO_LAYERS} \
  --vo_layers ${VO_LAYERS} \
  --dropout 0.1 \
  --fusion_method ${FUSION_METHOD} \
  --fusion_loss_weight ${FUSION_WEIGHT} \
  --cm_loss_weight ${CM_WEIGHT} \
  --ao_loss_weight ${AO_WEIGHT} \
  --vo_loss_weight ${VO_WEIGHT} \
  --output_dir ${OUTPUT_DIR} \
  --save_every 5 \
  --num_workers 4 \
  2>&1 | tee ${LOG_DIR}/train_baseline_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "=========================================="
echo "Training completed!"
echo "Checkpoints saved to: ${OUTPUT_DIR}"
echo "=========================================="

