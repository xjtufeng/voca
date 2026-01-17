#!/bin/bash
# Quick test for enhanced localization training V2
# Uses small subset for rapid iteration

FEATURES_ROOT="/hpc2ssd/JH_DATA/spooler/xfeng733/lavdf_feats"
OUTPUT_DIR="./test_checkpoints/localization_v2_quick"

echo "========================================"
echo "Quick Test: Enhanced Localization V2"
echo "========================================"
echo "Features: $FEATURES_ROOT"
echo "Output: $OUTPUT_DIR"
echo "This will train for 2 epochs with small batch"
echo "========================================"

python train_lavdf_localization_v2.py \
    --features_root $FEATURES_ROOT \
    --splits train,dev \
    --batch_size 4 \
    --num_workers 2 \
    --max_frames 256 \
    --event_centric_prob 0.5 \
    --v_dim 512 \
    --a_dim 1024 \
    --d_model 256 \
    --nhead 4 \
    --num_layers 2 \
    --dropout 0.1 \
    --use_inconsistency_module \
    --use_reliability_gating \
    --use_boundary_head \
    --use_boundary_aware_smooth \
    --alpha_init 0.5 \
    --temperature 0.1 \
    --video_loss_weight 0.3 \
    --boundary_loss_weight 0.5 \
    --smooth_loss_weight 0.05 \
    --ranking_loss_weight 0.5 \
    --fake_hinge_weight 0.05 \
    --ranking_margin 0.3 \
    --boundary_tolerance 5 \
    --neg_shift_min 3 \
    --neg_shift_max 10 \
    --neg_swap_prob 0.5 \
    --epochs 2 \
    --lr 1e-4 \
    --weight_decay 1e-5 \
    --warmup_epochs 1 \
    --max_grad_norm 1.0 \
    --output_dir $OUTPUT_DIR \
    --save_every 1

echo ""
echo "Quick test complete! Check results in $OUTPUT_DIR"

