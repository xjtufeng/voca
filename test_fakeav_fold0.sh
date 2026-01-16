#!/bin/bash
# Test Fold 0 training (quick test before running all 5 folds)

cd ~/jhspoolers/voca

FEATURES_ROOT="/hpc2ssd/JH_DATA/spooler/xfeng733/FakeAV_feats"
SPLITS_DIR="data/fakeav_5fold_splits"
OUTPUT_DIR="checkpoints/fakeav_5fold_test"

echo "======================================================================="
echo "TESTING FOLD 0 (QUICK TEST)"
echo "======================================================================="
echo ""

# Step 1: Generate splits if needed
if [ ! -d "$SPLITS_DIR" ]; then
    echo "[Step 1] Generating 5-fold splits..."
    python scripts/create_fakeav_5fold_splits.py \
        --features_root "$FEATURES_ROOT" \
        --output_dir "$SPLITS_DIR" \
        --seed 42
    echo ""
else
    echo "[Step 1] ✅ Splits exist"
    echo ""
fi

# Step 2: Test data loading
echo "[Step 2] Testing data loading..."
python dataset_fakeav_fourclass.py || echo "⚠️  Data loading test failed (expected on Windows, run on server)"
echo ""

# Step 3: Train Fold 0 (5 epochs for quick test)
echo "[Step 3] Training Fold 0 (5 epochs)..."
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
    --epochs 5 \
    --lr 1e-4 \
    --weight_decay 1e-5 \
    --num_workers 8 \
    --output_dir "$OUTPUT_DIR" \
    --folds 0 \
    2>&1 | tee logs/fakeav_fold0_test_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "======================================================================="
echo "FOLD 0 TEST COMPLETE"
echo "======================================================================="
echo ""
echo "If successful, run the full 5-fold pipeline:"
echo "  bash run_fakeav_5fold_pipeline.sh"
echo ""

