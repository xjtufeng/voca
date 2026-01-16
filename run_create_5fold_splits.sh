#!/bin/bash
# Generate 5-fold splits for FakeAVCeleb

cd ~/jhspoolers/voca

echo "======================================================================="
echo "GENERATING FAKEAV 5-FOLD SPLITS"
echo "======================================================================="
echo ""

python scripts/create_fakeav_5fold_splits.py \
  --features_root /hpc2ssd/JH_DATA/spooler/xfeng733/FakeAV_feats \
  --output_dir data/fakeav_5fold_splits \
  --seed 42

echo ""
echo "✅ Splits generated!"
echo ""
echo "Check the results:"
echo "  ls -lh data/fakeav_5fold_splits/"
echo "  cat data/fakeav_5fold_splits/metadata.json | python -m json.tool"

