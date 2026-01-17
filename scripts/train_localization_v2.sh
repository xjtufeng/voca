#!/bin/bash
# Enhanced LAV-DF Localization Training Script V2
# Uses learned inconsistency, reliability gating, ranking loss, and hard negatives

FEATURES_ROOT="/hpc2ssd/JH_DATA/spooler/xfeng733/lavdf_feats"
OUTPUT_DIR="./checkpoints/localization_v2"
EPOCHS=100
BATCH_SIZE=8
LR=1e-4

# Enhanced model settings
USE_INCONSISTENCY=true
USE_GATING=true
ALPHA_INIT=0.5
TEMPERATURE=0.1

# Loss weights (carefully tuned)
VIDEO_LOSS_WEIGHT=0.3
SMOOTH_LOSS_WEIGHT=0.1
RANKING_LOSS_WEIGHT=0.5    # Main contribution from inconsistency
FAKE_HINGE_WEIGHT=0.05     # Auxiliary constraint
RANKING_MARGIN=0.3

# Hard negatives config
NEG_SHIFT_MIN=3
NEG_SHIFT_MAX=10
NEG_SWAP_PROB=0.5

# Check if running on single GPU or multi-GPU
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    echo "No CUDA_VISIBLE_DEVICES set, using all available GPUs"
    NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
else
    NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
fi

echo "========================================"
echo "Enhanced Localization Training V2"
echo "========================================"
echo "Features: $FEATURES_ROOT"
echo "Output: $OUTPUT_DIR"
echo "GPUs: $NUM_GPUS"
echo "Batch size: $BATCH_SIZE"
echo "Ranking loss weight: $RANKING_LOSS_WEIGHT"
echo "Fake hinge weight: $FAKE_HINGE_WEIGHT"
echo "========================================"

# Build command
CMD="python train_lavdf_localization_v2.py \
    --features_root $FEATURES_ROOT \
    --splits train,dev \
    --batch_size $BATCH_SIZE \
    --num_workers 4 \
    --max_frames 512 \
    --v_dim 512 \
    --a_dim 1024 \
    --d_model 512 \
    --nhead 8 \
    --num_layers 4 \
    --dropout 0.1 \
    --video_loss_weight $VIDEO_LOSS_WEIGHT \
    --smooth_loss_weight $SMOOTH_LOSS_WEIGHT \
    --ranking_loss_weight $RANKING_LOSS_WEIGHT \
    --fake_hinge_weight $FAKE_HINGE_WEIGHT \
    --ranking_margin $RANKING_MARGIN \
    --neg_shift_min $NEG_SHIFT_MIN \
    --neg_shift_max $NEG_SHIFT_MAX \
    --neg_swap_prob $NEG_SWAP_PROB \
    --alpha_init $ALPHA_INIT \
    --temperature $TEMPERATURE \
    --epochs $EPOCHS \
    --lr $LR \
    --weight_decay 1e-5 \
    --warmup_epochs 5 \
    --max_grad_norm 1.0 \
    --output_dir $OUTPUT_DIR \
    --save_every 5"

# Add enhanced features flags
if [ "$USE_INCONSISTENCY" = true ]; then
    CMD="$CMD --use_inconsistency_module"
fi

if [ "$USE_GATING" = true ]; then
    CMD="$CMD --use_reliability_gating"
fi

# Multi-GPU training with DDP
if [ $NUM_GPUS -gt 1 ]; then
    echo "Using DDP with $NUM_GPUS GPUs"
    torchrun --nproc_per_node=$NUM_GPUS $CMD
else
    echo "Using single GPU"
    $CMD
fi

echo ""
echo "Training complete! Check results in $OUTPUT_DIR"

