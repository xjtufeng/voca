#!/bin/bash
# Training script for three-branch joint model
# Usage: bash scripts/train_three_branch.sh

set -e

# ============= Configuration =============
FEATURES_ROOT="/hpc2ssd/JH_DATA/spooler/xfeng733/lavdf_feats"
OUTPUT_DIR="./checkpoints/three_branch"
SPLITS="train dev"

# Model hyperparameters
D_MODEL=512
NHEAD=8
CM_LAYERS=4
AO_LAYERS=3
VO_LAYERS=3
DROPOUT=0.1
FUSION_METHOD="attention"  # options: concat, weighted, attention

# Loss weights
FUSION_LOSS_WEIGHT=1.0
CM_LOSS_WEIGHT=0.3
AO_LOSS_WEIGHT=0.2
VO_LOSS_WEIGHT=0.2

# Training hyperparameters
BATCH_SIZE=32
EPOCHS=100
LR=1e-4
WEIGHT_DECAY=1e-5
WARMUP_EPOCHS=5
MAX_FRAMES=256

# Hardware
NUM_WORKERS=4
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)

echo "=========================================="
echo "Three-Branch Joint Training"
echo "=========================================="
echo "Features root: $FEATURES_ROOT"
echo "Output dir: $OUTPUT_DIR"
echo "GPUs: $NUM_GPUS"
echo "Fusion method: $FUSION_METHOD"
echo "Batch size: $BATCH_SIZE"
echo "Epochs: $EPOCHS"
echo "=========================================="

# ============= Training =============

if [ $NUM_GPUS -eq 1 ]; then
    # Single GPU training
    echo "Starting single-GPU training..."
    
    python train_three_branch.py \
        --features_root $FEATURES_ROOT \
        --splits $SPLITS \
        --max_frames $MAX_FRAMES \
        --d_model $D_MODEL \
        --nhead $NHEAD \
        --cm_layers $CM_LAYERS \
        --ao_layers $AO_LAYERS \
        --vo_layers $VO_LAYERS \
        --dropout $DROPOUT \
        --fusion_method $FUSION_METHOD \
        --fusion_loss_weight $FUSION_LOSS_WEIGHT \
        --cm_loss_weight $CM_LOSS_WEIGHT \
        --ao_loss_weight $AO_LOSS_WEIGHT \
        --vo_loss_weight $VO_LOSS_WEIGHT \
        --batch_size $BATCH_SIZE \
        --epochs $EPOCHS \
        --lr $LR \
        --weight_decay $WEIGHT_DECAY \
        --warmup_epochs $WARMUP_EPOCHS \
        --num_workers $NUM_WORKERS \
        --output_dir $OUTPUT_DIR \
        --save_every 5

else
    # Multi-GPU DDP training
    echo "Starting multi-GPU DDP training with $NUM_GPUS GPUs..."
    
    torchrun \
        --nproc_per_node=$NUM_GPUS \
        --master_port=29502 \
        train_three_branch.py \
        --features_root $FEATURES_ROOT \
        --splits $SPLITS \
        --max_frames $MAX_FRAMES \
        --d_model $D_MODEL \
        --nhead $NHEAD \
        --cm_layers $CM_LAYERS \
        --ao_layers $AO_LAYERS \
        --vo_layers $VO_LAYERS \
        --dropout $DROPOUT \
        --fusion_method $FUSION_METHOD \
        --fusion_loss_weight $FUSION_LOSS_WEIGHT \
        --cm_loss_weight $CM_LOSS_WEIGHT \
        --ao_loss_weight $AO_LOSS_WEIGHT \
        --vo_loss_weight $VO_LOSS_WEIGHT \
        --batch_size $BATCH_SIZE \
        --epochs $EPOCHS \
        --lr $LR \
        --weight_decay $WEIGHT_DECAY \
        --warmup_epochs $WARMUP_EPOCHS \
        --num_workers $NUM_WORKERS \
        --output_dir $OUTPUT_DIR \
        --save_every 5
fi

echo ""
echo "=========================================="
echo "Training complete!"
echo "Checkpoints saved to: $OUTPUT_DIR"
echo "=========================================="

