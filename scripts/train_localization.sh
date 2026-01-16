#!/bin/bash
# Training script for frame-level deepfake localization
# Usage: bash scripts/train_localization.sh

set -e

# ============= Configuration =============
FEATURES_ROOT="/hpc2ssd/JH_DATA/spooler/xfeng733/lavdf_feats"
OUTPUT_DIR="./checkpoints/localization"
SPLITS="train dev"

# Model hyperparameters
D_MODEL=512
NHEAD=8
NUM_LAYERS=4
DROPOUT=0.1

# Training hyperparameters
BATCH_SIZE=8
EPOCHS=100
LR=1e-4
WEIGHT_DECAY=1e-5
WARMUP_EPOCHS=5
MAX_FRAMES=512

# Loss weights
VIDEO_LOSS_WEIGHT=0.3
SMOOTH_LOSS_WEIGHT=0.1

# Hardware
NUM_WORKERS=4
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)

echo "=========================================="
echo "LAV-DF Localization Training"
echo "=========================================="
echo "Features root: $FEATURES_ROOT"
echo "Output dir: $OUTPUT_DIR"
echo "GPUs: $NUM_GPUS"
echo "Batch size: $BATCH_SIZE"
echo "Epochs: $EPOCHS"
echo "=========================================="

# ============= Training =============

if [ $NUM_GPUS -eq 1 ]; then
    # Single GPU training
    echo "Starting single-GPU training..."
    
    python train_lavdf_localization.py \
        --features_root $FEATURES_ROOT \
        --splits $SPLITS \
        --max_frames $MAX_FRAMES \
        --d_model $D_MODEL \
        --nhead $NHEAD \
        --num_layers $NUM_LAYERS \
        --dropout $DROPOUT \
        --pos_weight -1 \
        --video_loss_weight $VIDEO_LOSS_WEIGHT \
        --smooth_loss_weight $SMOOTH_LOSS_WEIGHT \
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
        --master_port=29501 \
        train_lavdf_localization.py \
        --features_root $FEATURES_ROOT \
        --splits $SPLITS \
        --max_frames $MAX_FRAMES \
        --d_model $D_MODEL \
        --nhead $NHEAD \
        --num_layers $NUM_LAYERS \
        --dropout $DROPOUT \
        --pos_weight -1 \
        --video_loss_weight $VIDEO_LOSS_WEIGHT \
        --smooth_loss_weight $SMOOTH_LOSS_WEIGHT \
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


