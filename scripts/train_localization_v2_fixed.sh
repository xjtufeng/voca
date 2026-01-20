#!/bin/bash
# Fixed LAV-DF Localization Training Script
# Key fixes:
# - alpha: 2.0 -> 0.3 (reduce inconsistency branch dominance)
# - temperature: 0.05 -> 0.5 (prevent gradient vanishing)
# - ranking_loss_weight: 1.0 -> 0.1 (reduce ranking focus)
# - neg_shift: 5-20 -> 3-10 (more reasonable hard negatives)

FEATURES_ROOT="/hpc2ssd/JH_DATA/spooler/xfeng733/LAV-DF_feats"
OUTPUT_DIR="./checkpoints/localization_v2_fixed"
EPOCHS=10
BATCH_SIZE=32
LR=1e-4

# Check GPU count
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    NUM_GPUS=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
else
    NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
fi

echo "========================================"
echo "Fixed Localization Training V2"
echo "========================================"
echo "Features: $FEATURES_ROOT"
echo "Output: $OUTPUT_DIR"
echo "GPUs: $NUM_GPUS"
echo "Batch size: $BATCH_SIZE"
echo "Key fixes:"
echo "  - alpha: 2.0 -> 0.3"
echo "  - temperature: 0.05 -> 0.5"
echo "  - ranking_loss_weight: 1.0 -> 0.1"
echo "========================================"

# Training command with fixed parameters
torchrun --nproc_per_node=$NUM_GPUS train_lavdf_localization_v2.py \
  --features_root $FEATURES_ROOT \
  --splits train,dev \
  --batch_size $BATCH_SIZE \
  --epochs $EPOCHS \
  --lr $LR \
  --ranking_loss_weight 0.1 \
  --inconsistency_sup_weight 0.1 \
  --gate_warmup_epochs 0 \
  --temperature 0.5 \
  --alpha_fixed 0.3 \
  --alpha_min 0.1 \
  --neg_shift_min 3 \
  --neg_shift_max 10 \
  --neg_swap_prob 0.5 \
  --a_dim 512 \
  --use_inconsistency_module \
  --no_reliability_gating \
  --use_boundary_head \
  --use_boundary_aware_smooth \
  --output_dir $OUTPUT_DIR

echo ""
echo "Training complete! Check results in $OUTPUT_DIR"

