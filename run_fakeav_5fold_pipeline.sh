#!/bin/bash
# Complete 5-Fold Cross-Validation Pipeline for FakeAVCeleb (MRDF Protocol)

set -e  # Exit on error

cd ~/jhspoolers/voca

FEATURES_ROOT="/hpc2ssd/JH_DATA/spooler/xfeng733/FakeAV_feats"
SPLITS_DIR="data/fakeav_5fold_splits"
OUTPUT_DIR="checkpoints/fakeav_5fold"

echo "======================================================================="
echo "FAKEAV 5-FOLD CROSS-VALIDATION PIPELINE (MRDF PROTOCOL)"
echo "======================================================================="
echo ""
echo "Protocol:"
echo "  - 4-class classification: FAFV/FARV/RAFV/RARV"
echo "  - Identity-independent splits"
echo "  - 1:1:1:1 balanced sampling"
echo "  - 5-fold cross-validation"
echo "  - Report mean ACC/AUC"
echo ""
echo "======================================================================="
echo ""

# Step 1: Generate 5-fold splits (if not exists)
if [ ! -d "$SPLITS_DIR" ]; then
    echo "[Step 1] Generating 5-fold splits..."
    python scripts/create_fakeav_5fold_splits.py \
        --features_root "$FEATURES_ROOT" \
        --output_dir "$SPLITS_DIR" \
        --seed 42
    echo ""
else
    echo "[Step 1] ✅ Splits already exist: $SPLITS_DIR"
    echo ""
fi

# Step 2: Train all 5 folds
echo "[Step 2] Training 5-fold models..."
echo "-----------------------------------------------------------------------"
echo ""

python train_fakeav_5fold.py \
    --features_root "$FEATURES_ROOT" \
    --splits_dir "$SPLITS_DIR" \
    --max_frames 256 \
    --audio_dim 512 \
    --visual_dim 512 \
    --d_model 512 \
    --cm_layers 4 \
    --ao_layers 4 \
    --vo_layers 4 \
    --nhead 8 \
    --dropout 0.1 \
    --fusion_method weighted \
    --batch_size 32 \
    --epochs 30 \
    --lr 1e-4 \
    --weight_decay 1e-5 \
    --num_workers 8 \
    --output_dir "$OUTPUT_DIR" \
    --folds 0 1 2 3 4 \
    2>&1 | tee logs/fakeav_5fold_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "======================================================================="
echo "PIPELINE COMPLETE"
echo "======================================================================="
echo ""
echo "Results:"
echo "  - Checkpoints: $OUTPUT_DIR/fold_*/best.pth"
echo "  - Summary: $OUTPUT_DIR/5fold_results.json"
echo ""
echo "View results:"
echo "  cat $OUTPUT_DIR/5fold_results.json | python -m json.tool"
echo ""

