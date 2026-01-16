#!/bin/bash
# Complete diagnosis workflow for test set issues

set -e  # Exit on error

VOCA_DIR=~/jhspoolers/voca
FEATURES_ROOT=/hpc2ssd/JH_DATA/spooler/xfeng733/LAV-DF_feats
CKPT_DIR=$VOCA_DIR/checkpoints/localization_ssd_2gpu
LOG_DIR=$VOCA_DIR/logs

cd $VOCA_DIR

echo "======================================================================="
echo "COMPREHENSIVE TEST SET DIAGNOSIS"
echo "======================================================================="
echo ""

# Step 1: Find best validation epoch
echo "[STEP 1] Finding best validation epoch from training logs..."
echo "-----------------------------------------------------------------------"
echo "Top 10 validation Frame AUC values:"
grep -E "Val.*Frame AUC" $LOG_DIR/localization_ssd_2gpu_*.log | \
    grep -oP "Frame AUC: \K[0-9.]+|Epoch \K[0-9]+" | \
    paste - - | sort -k2 -rn | head -10 | \
    awk '{print "  Epoch " $1 ": AUC = " $2}'

echo ""
echo "Finding which epoch has best.pth..."
python3 << 'PYEOF'
import torch
try:
    ckpt = torch.load('/hpc2ssd/JH_DATA/spooler/xfeng733/voca/checkpoints/localization_ssd_2gpu/best.pth', 
                      map_location='cpu', weights_only=False)
    print(f"  best.pth = Epoch {ckpt.get('epoch', '?')}")
    if 'metrics' in ckpt:
        frame_auc = ckpt['metrics'].get('frame_auc', 0)
        print(f"  Val Frame AUC: {frame_auc:.4f}")
except Exception as e:
    print(f"  Error loading best.pth: {e}")
PYEOF

# Step 2: Check available checkpoints
echo ""
echo "[STEP 2] Available checkpoints..."
echo "-----------------------------------------------------------------------"
ls -lh $CKPT_DIR/checkpoint_epoch*.pth | tail -5 | awk '{print "  " $9 " (" $5 ")"}'

# Determine which checkpoint to test
if [ -f "$CKPT_DIR/checkpoint_epoch99.pth" ]; then
    TEST_CKPT="$CKPT_DIR/checkpoint_epoch99.pth"
    TEST_EPOCH=99
    echo ""
    echo "✅ Will test Epoch 99 (last epoch)"
elif [ -f "$CKPT_DIR/best.pth" ]; then
    TEST_CKPT="$CKPT_DIR/best.pth"
    TEST_EPOCH="best"
    echo ""
    echo "✅ Will test best.pth"
else
    echo ""
    echo "❌ No suitable checkpoint found!"
    exit 1
fi

# Step 3: Check frame probability distribution
echo ""
echo "[STEP 3] Analyzing frame probability distribution..."
echo "-----------------------------------------------------------------------"
echo "This shows if model can actually discriminate real/fake frames"
echo ""

python check_frame_probs_distribution.py \
    --checkpoint $TEST_CKPT \
    --features_root $FEATURES_ROOT \
    --split test \
    --batch_size 24 \
    --num_workers 8 \
    --max_frames 512 \
    --d_model 512 --nhead 8 --num_layers 4 --dropout 0.1 \
    --output_dir results/prob_distribution_test

# Step 4: Run test with FIXED threshold (no data leakage)
echo ""
echo "[STEP 4] Running test evaluation with FIXED threshold..."
echo "-----------------------------------------------------------------------"
echo "Using threshold=0.2 (chosen from validation, not swept on test)"
echo ""

python evaluate_lavdf_localization.py \
    --checkpoint $TEST_CKPT \
    --features_root $FEATURES_ROOT \
    --split test \
    --batch_size 24 \
    --num_workers 8 \
    --max_frames 512 \
    --d_model 512 --nhead 8 --num_layers 4 --dropout 0.1 \
    --fixed_threshold 0.2 \
    --output_dir results/localization_test_epoch${TEST_EPOCH}_fixed

# Step 5: Also run dev (val) evaluation for comparison
echo ""
echo "[STEP 5] Running dev evaluation (for comparison)..."
echo "-----------------------------------------------------------------------"

if [ -d "$FEATURES_ROOT/dev_temp" ]; then
    python evaluate_lavdf_localization.py \
        --checkpoint $TEST_CKPT \
        --features_root $FEATURES_ROOT \
        --split dev_temp \
        --batch_size 24 \
        --num_workers 8 \
        --max_frames 512 \
        --d_model 512 --nhead 8 --num_layers 4 --dropout 0.1 \
        --output_dir results/localization_dev_epoch${TEST_EPOCH}
else
    echo "⚠️  dev_temp split not found, skipping dev evaluation"
fi

# Step 6: Summary
echo ""
echo "======================================================================="
echo "DIAGNOSIS COMPLETE"
echo "======================================================================="
echo ""
echo "Results saved to:"
echo "  - Probability distribution: results/prob_distribution_test/"
echo "  - Test evaluation: results/localization_test_epoch${TEST_EPOCH}_fixed/"
echo ""
echo "Key files to check:"
echo "  1. results/prob_distribution_test/test_prob_distribution_epoch*.png"
echo "  2. results/prob_distribution_test/test_prob_stats_epoch*.txt"
echo "  3. results/localization_test_epoch${TEST_EPOCH}_fixed/test_results_epoch*.txt"
echo ""
echo "What to look for:"
echo "  1. Probability separation: Fake mean - Real mean should be > 0.5"
echo "  2. Frame AUC on test: Should be close to val AUC (within 0.05)"
echo "  3. If Frame AUC is still < 0.7, check dataset splitting or labels"
echo ""

