# FakeAVCeleb 5-Fold Training - Quick Start Guide

## 📋 Overview

Train three-branch model on FakeAVCeleb following **MRDF Protocol**:
- ✅ 4-class: FAFV/FARV/RAFV/RARV
- ✅ Identity-independent 5-fold CV
- ✅ 1:1:1:1 balanced sampling
- ✅ Report mean ACC/AUC

---

## 🚀 Quick Start (On Server)

### Option 1: Full Pipeline (All 5 Folds)

```bash
cd ~/jhspoolers/voca

# Make scripts executable
chmod +x run_fakeav_5fold_pipeline.sh test_fakeav_fold0.sh run_create_5fold_splits.sh

# Run complete pipeline (~10-15 hours)
bash run_fakeav_5fold_pipeline.sh
```

**This will**:
1. Generate 5-fold splits (if not exist)
2. Train all 5 folds (30 epochs each)
3. Report mean ACC/AUC

---

### Option 2: Test First (Fold 0 Only, ~20 mins)

```bash
cd ~/jhspoolers/voca

# Quick test with Fold 0 (5 epochs)
bash test_fakeav_fold0.sh

# If successful, run full pipeline
bash run_fakeav_5fold_pipeline.sh
```

---

## 📊 View Results

```bash
# View aggregated results
cat checkpoints/fakeav_5fold/5fold_results.json

# Formatted output
python -c "
import json
with open('checkpoints/fakeav_5fold/5fold_results.json') as f:
    r = json.load(f)
print(f'Mean ACC: {r[\"mean_acc\"]:.4f} ± {r[\"std_acc\"]:.4f}')
print(f'Mean AUC: {r[\"mean_auc\"]:.4f} ± {r[\"std_auc\"]:.4f}')
"

# Check individual fold checkpoints
ls -lh checkpoints/fakeav_5fold/fold_*/best.pth
```

---

## 🔧 Manual Steps

### Step 1: Generate Splits

```bash
python scripts/create_fakeav_5fold_splits.py \
  --features_root /hpc2ssd/JH_DATA/spooler/xfeng733/FakeAV_feats \
  --output_dir data/fakeav_5fold_splits \
  --seed 42
```

### Step 2: Train Specific Fold(s)

```bash
# Train only Fold 0
python train_fakeav_5fold.py \
  --features_root /hpc2ssd/JH_DATA/spooler/xfeng733/FakeAV_feats \
  --splits_dir data/fakeav_5fold_splits \
  --batch_size 32 --epochs 30 \
  --output_dir checkpoints/fakeav_5fold \
  --folds 0

# Train Folds 0 and 1
python train_fakeav_5fold.py ... --folds 0 1

# Train all folds
python train_fakeav_5fold.py ... --folds 0 1 2 3 4
```

---

## 📁 File Structure

```
~/jhspoolers/voca/
├── scripts/
│   └── create_fakeav_5fold_splits.py    # Generate splits
├── dataset_fakeav_fourclass.py           # 4-class dataset
├── train_fakeav_5fold.py                 # Training script
├── run_fakeav_5fold_pipeline.sh          # Full pipeline
├── test_fakeav_fold0.sh                  # Quick test
├── data/
│   └── fakeav_5fold_splits/              # Generated splits
│       ├── fold_0_train.json
│       ├── fold_0_test.json
│       └── ...
├── checkpoints/
│   └── fakeav_5fold/                     # Trained models
│       ├── fold_0/best.pth
│       ├── fold_1/best.pth
│       └── 5fold_results.json            # Final results
└── logs/
    └── fakeav_5fold_*.log                # Training logs
```

---

## 🎯 Expected Results

**MRDF Baseline** (from paper):
- ACC: ~85-90%
- AUC: ~92-95%

**Per-Class ACC**:
- FAFV: ~88%
- FARV: ~85%
- RAFV: ~80% (harder, only 500 samples)
- RARV: ~82% (harder, only 500 samples)

---

## 🐛 Troubleshooting

### "Splits not found"

```bash
# Regenerate splits
bash run_create_5fold_splits.sh
```

### "CUDA out of memory"

```bash
# Reduce batch size
python train_fakeav_5fold.py ... --batch_size 16 --max_frames 128
```

### "Class imbalance in batch"

- Ensure `balanced_train=True` in dataloader (default)
- Increase `--batch_size` to at least 32

---

## 📈 Monitoring

```bash
# GPU usage
watch -n 2 nvidia-smi

# Training progress
tail -f logs/fakeav_5fold_*.log

# Current fold
ps aux | grep train_fakeav_5fold
```

---

## ⏱️ Estimated Time

| Task | Time |
|------|------|
| Generate splits | ~1 minute |
| Test Fold 0 (5 epochs) | ~20 minutes |
| Full training (5 folds × 30 epochs) | ~10-15 hours |

---

## 📝 Citation

```bibtex
@inproceedings{mrdf2022,
  title={Multi-modal Representation Learning for Deepfake Detection},
  booktitle={Conference},
  year={2022}
}
```

---

## ✅ Checklist

- [ ] Features extracted at `/hpc2ssd/JH_DATA/spooler/xfeng733/FakeAV_feats`
- [ ] Verified 4 classes exist (FAFV/FARV/RAFV/RARV)
- [ ] Generated 5-fold splits
- [ ] Tested Fold 0 (quick test)
- [ ] Launched full 5-fold training
- [ ] Collected results in `5fold_results.json`
- [ ] Compared with MRDF baselines

---

## 🔗 See Also

- [Detailed Guide](docs/FAKEAV_5FOLD_GUIDE.md) - Complete documentation
- [Localization Guide](docs/LOCALIZATION_GUIDE.md) - LAV-DF localization

---

**Ready to start?** Run `bash test_fakeav_fold0.sh` on the server! 🚀

