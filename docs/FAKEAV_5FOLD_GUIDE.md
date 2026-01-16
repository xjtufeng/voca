# FakeAVCeleb 5-Fold Cross-Validation Guide (MRDF Protocol)

## Overview

This guide describes the complete pipeline for training and evaluating the three-branch model on FakeAVCeleb following the **MRDF paper protocol**:

- **4-Class Classification**: FAFV, FARV, RAFV, RARV (1:1:1:1 balanced)
- **Identity-Independent**: 5-fold cross-validation based on speaker identities
- **Metrics**: Report mean ACC and AUC across 5 folds

---

## Data Structure

### FakeAVCeleb Features

```
/hpc2ssd/JH_DATA/spooler/xfeng733/FakeAV_feats/
├── real/
│   └── RealVideo-RealAudio_*_id*_*/ (500 videos - RARV)
│       ├── audio_embeddings.npz
│       └── visual_embeddings.npz
└── fake/
    ├── FakeVideo-FakeAudio_*/ (10,835 videos - FAFV)
    ├── FakeVideo-RealAudio_*/ (9,709 videos - FARV)
    └── RealVideo-FakeAudio_*/ (500 videos - RAFV)
```

### Four Classes

| Label | Name | Description | Count |
|-------|------|-------------|-------|
| 0 | FAFV | Fake Audio + Fake Video | 10,835 (50.3%) |
| 1 | FARV | Fake Audio + Real Video | 9,709 (45.1%) |
| 2 | RAFV | Real Audio + Fake Video | 500 (2.3%) |
| 3 | RARV | Real Audio + Real Video | 500 (2.3%) |

⚠️ **Note**: Classes are highly imbalanced. The training uses **1:1:1:1 balanced sampling** via `WeightedRandomSampler`.

---

## Pipeline Steps

### Step 1: Generate 5-Fold Splits

Generate identity-independent splits:

```bash
cd ~/jhspoolers/voca

python scripts/create_fakeav_5fold_splits.py \
  --features_root /hpc2ssd/JH_DATA/spooler/xfeng733/FakeAV_feats \
  --output_dir data/fakeav_5fold_splits \
  --seed 42
```

**Output**:
```
data/fakeav_5fold_splits/
├── fold_0_train.json  # Fold 0: train on folds 1,2,3,4
├── fold_0_test.json   # Fold 0: test on fold 0
├── fold_1_train.json
├── fold_1_test.json
├── ...
├── fold_4_test.json
└── metadata.json      # Overall statistics
```

**Each JSON file** contains:
```json
[
  {
    "video_id": "RealVideo-RealAudio_..._id05383_00015",
    "identity": "id05383",
    "label": 3,
    "label_name": "RARV",
    "path": "real/RealVideo-RealAudio_..._id05383_00015"
  },
  ...
]
```

---

### Step 2: Train All 5 Folds

Train all folds sequentially:

```bash
python train_fakeav_5fold.py \
  --features_root /hpc2ssd/JH_DATA/spooler/xfeng733/FakeAV_feats \
  --splits_dir data/fakeav_5fold_splits \
  --max_frames 256 \
  --d_model 512 \
  --cm_layers 4 --ao_layers 4 --vo_layers 4 \
  --nhead 8 --dropout 0.1 \
  --fusion_method weighted \
  --batch_size 32 \
  --epochs 30 \
  --lr 1e-4 \
  --num_workers 8 \
  --output_dir checkpoints/fakeav_5fold \
  --folds 0 1 2 3 4
```

**Or use the provided script**:

```bash
bash run_fakeav_5fold_pipeline.sh
```

---

### Step 3: View Results

After training completes, view the aggregated results:

```bash
cat checkpoints/fakeav_5fold/5fold_results.json | python -m json.tool
```

**Example Output**:
```json
{
  "fold_results": [
    {"fold": 0, "accuracy": 0.8234, "auc": 0.9012, ...},
    {"fold": 1, "accuracy": 0.8156, "auc": 0.8987, ...},
    ...
  ],
  "mean_acc": 0.8195,
  "std_acc": 0.0045,
  "mean_auc": 0.8998,
  "std_auc": 0.0032,
  "per_class_acc_mean": [0.85, 0.82, 0.78, 0.81],
  "per_class_acc_std": [0.02, 0.03, 0.05, 0.04]
}
```

---

## Quick Test (Fold 0 Only)

Before running all 5 folds, test with Fold 0 (5 epochs):

```bash
bash test_fakeav_fold0.sh
```

This will:
1. Generate splits
2. Test data loading
3. Train Fold 0 for 5 epochs (quick sanity check)

---

## Key Implementation Details

### 1. Identity-Independent Splitting

- Extracts identity from video names: `id\d+`
- Splits identities (not videos) into 5 folds
- Ensures same identity only appears in one fold

### 2. Balanced Sampling (1:1:1:1)

```python
# In dataset_fakeav_fourclass.py
sampler = dataset.get_balanced_sampler()
train_loader = DataLoader(dataset, sampler=sampler, ...)
```

- Uses `WeightedRandomSampler`
- Weight = 1 / class_count
- Each batch has (approximately) equal samples from each class

### 3. Four-Class Classification

```python
# Model output: [B, 4] logits
loss = nn.CrossEntropyLoss()(logits, labels)

# Metrics
probs = torch.softmax(logits, dim=1)  # [B, 4]
preds = torch.argmax(probs, dim=1)    # [B]

acc = accuracy_score(labels, preds)
auc = roc_auc_score(labels, probs, multi_class='ovr', average='macro')
```

### 4. Model Architecture

Modified `ThreeBranchJointModel` with `num_classes=4`:
- Cross-Modal branch
- Audio-Only branch
- Visual-Only branch
- Learnable fusion → 4-class classifier

---

## Monitoring Training

```bash
# Watch GPU usage
watch -n 2 nvidia-smi

# Monitor training log
tail -f logs/fakeav_5fold_*.log

# Check intermediate results
ls -lh checkpoints/fakeav_5fold/fold_*/
```

---

## Expected Training Time

- **Per Fold**: ~2-3 hours (30 epochs, batch_size=32, 2 GPUs)
- **Total (5 folds)**: ~10-15 hours

---

## Troubleshooting

### Issue: "num_samples=0"

**Cause**: Split file not found or empty

**Solution**:
```bash
# Check if splits exist
ls data/fakeav_5fold_splits/

# Regenerate if needed
python scripts/create_fakeav_5fold_splits.py --features_root ... --output_dir ...
```

### Issue: "RuntimeError: CUDA out of memory"

**Solution**: Reduce `--batch_size` or `--max_frames`:
```bash
--batch_size 16 --max_frames 128
```

### Issue: Imbalanced class in batch despite balanced sampling

**Cause**: Batch size too small

**Solution**: Increase `--batch_size` to at least 16-32

---

## Citation

If you use this implementation, please cite the MRDF paper for the protocol:

```bibtex
@inproceedings{mrdf2022,
  title={Multi-modal Representation Learning for Deepfake Detection},
  author={...},
  booktitle={...},
  year={2022}
}
```

---

## Files Created

- `scripts/create_fakeav_5fold_splits.py` - Generate identity-independent 5-fold splits
- `dataset_fakeav_fourclass.py` - Four-class dataset with balanced sampler
- `train_fakeav_5fold.py` - 5-fold training script
- `run_fakeav_5fold_pipeline.sh` - Complete pipeline script
- `test_fakeav_fold0.sh` - Quick test script

---

## Next Steps

1. ✅ Run quick test: `bash test_fakeav_fold0.sh`
2. ✅ If successful, run full pipeline: `bash run_fakeav_5fold_pipeline.sh`
3. ✅ Analyze results: `cat checkpoints/fakeav_5fold/5fold_results.json`
4. 📝 Compare with MRDF paper baselines
5. 📊 Generate visualizations and ablation studies

