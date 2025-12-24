# Three-Branch Training Guide

Complete guide for training the three-branch joint deepfake detection model on HPC.

---

## 🎯 Quick Start

### Step 1: Login to HPC and Prepare Environment

```bash
# Login
ssh xfeng733@hpc2login.hpc.hkust-gz.edu.cn

# Navigate to project
cd ~/jhspoolers/voca

# Activate environment
conda activate voca

# Pull latest code
git pull
```

### Step 2: Quick Test (5 epochs, ~30 minutes)

```bash
# Test to verify everything works
bash scripts/quick_test_three_branch.sh

# Monitor progress
tail -f logs/quick_test_*.log

# Check output
ls -lh checkpoints/quick_test/
```

**Expected output:**
```
Epoch [1/5]
  Train - Total: 0.4532, Fused: 0.4012, CM: 0.4201, AO: 0.5123, VO: 0.4876
  Val   - Fused AUC: 0.7890, CM AUC: 0.7456, AO AUC: 0.6823, VO AUC: 0.7234
...
Epoch [5/5]
  Train - Total: 0.2134, Fused: 0.1912, CM: 0.2001, AO: 0.2523, VO: 0.2176
  Val   - Fused AUC: 0.8890, CM AUC: 0.8656, AO AUC: 0.7923, VO AUC: 0.8434
```

### Step 3: Full Training (100 epochs, ~24-48 hours)

```bash
# Run in background with tmux
tmux new -s train

# Inside tmux
bash scripts/train_three_branch_baseline.sh

# Detach: Ctrl+B then D
# Reattach: tmux attach -t train
```

---

## 📊 Training Details

### Dataset

**FakeAVCeleb Features:**
```
/hpc2ssd/JH_DATA/spooler/xfeng733/FakeAV_feats/
├── real/
│   ├── RealVideo-RealAudio_African_men_id00076_00109/
│   │   ├── visual_embeddings.npz  # [T, 512] InsightFace
│   │   └── audio_embeddings.npz   # [T, 1024] HuBERT
│   └── ...
└── fake/
    ├── FakeVideo-RealAudio_*.../
    └── ...
```

**Data Split:**
- Train: 70% (~15,000 videos)
- Dev/Val: 15% (~3,200 videos)
- Test: 15% (~3,200 videos)

### Model Architecture

```
Three-Branch Joint Model
├─ Cross-Modal Branch (d_model=512, 4 layers)
│  └─ Input: Visual (512-d) + Audio (1024-d)
├─ Audio-Only Branch (d_model=512, 3 layers)
│  └─ Input: Audio (1024-d)
└─ Visual-Only Branch (d_model=512, 3 layers)
   └─ Input: Visual (512-d)

Fusion: Weighted (learnable weights)
Output: Video-level fake probability
```

**Total Parameters:** ~36M

### Training Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Batch Size** | 16 | Per GPU |
| **Epochs** | 100 | ~24-48 hours |
| **Learning Rate** | 1e-3 | With cosine decay |
| **Warmup** | 5 epochs | Linear warmup |
| **Weight Decay** | 0.01 | L2 regularization |
| **Max Frames** | 150 | Truncate/pad |
| **Fusion Method** | weighted | Learnable branch weights |

**Loss Weights:**
- Fused loss: 1.0
- Cross-Modal: 0.3
- Audio-Only: 0.2
- Visual-Only: 0.2

---

## 📈 Expected Performance

### Baseline (100 epochs)

| Metric | Cross-Modal | Audio-Only | Visual-Only | **Fused** |
|--------|-------------|------------|-------------|-----------|
| **AUC** | 88-92% | 75-80% | 82-86% | **90-94%** |
| **F1** | 85-89% | 72-77% | 79-83% | **87-91%** |
| **Precision** | 86-90% | 73-78% | 80-84% | **88-92%** |

### Training Curves

**Loss:**
```
Epoch  1: 0.45
Epoch 10: 0.28
Epoch 20: 0.22
Epoch 50: 0.15
Epoch 100: 0.10
```

**Fused AUC:**
```
Epoch  1: 0.78
Epoch 10: 0.85
Epoch 20: 0.89
Epoch 50: 0.92
Epoch 100: 0.93
```

---

## 🔍 Monitoring Training

### Real-time Monitoring

```bash
# Watch log
tail -f logs/train_baseline_*.log

# Watch GPU
watch -n 2 nvidia-smi

# Check checkpoints
ls -lht checkpoints/three_branch_baseline/
```

### TensorBoard (Optional)

```bash
# On HPC
tensorboard --logdir logs --port 6006 --bind_all

# On local machine (SSH tunnel)
ssh -L 6006:localhost:6006 xfeng733@hpc2login.hpc.hkust-gz.edu.cn

# Open browser: http://localhost:6006
```

### Check Specific Metrics

```bash
# Fused AUC over time
grep "Fused AUC" logs/train_baseline_*.log

# Best model info
grep "New best" logs/train_baseline_*.log

# Training time per epoch
grep "Epoch \[" logs/train_baseline_*.log | head -20
```

---

## 🎛️ Hyperparameter Tuning

### Adjust Fusion Method

```bash
# Try different fusion strategies
bash scripts/train_three_branch_baseline.sh
# Edit FUSION_METHOD in script: 'concat' | 'weighted' | 'attention'
```

### Adjust Loss Weights

```python
# If one branch is weak, increase its weight
# Example: Audio-Only is weak
--ao_loss_weight 0.5  # Increase from 0.2 to 0.5
```

### Adjust Model Size

```bash
# Larger model (if memory allows)
--d_model 768
--cm_layers 6
--ao_layers 4
--vo_layers 4

# Smaller model (if memory is tight)
--d_model 256
--cm_layers 2
--ao_layers 2
--vo_layers 2
```

---

## 🐛 Troubleshooting

### Issue 1: CUDA Out of Memory

**Error:** `RuntimeError: CUDA out of memory`

**Solution:**
```bash
# Reduce batch size
--batch_size 8  # or even 4

# Reduce sequence length
--max_frames 100

# Reduce model size
--d_model 256
```

### Issue 2: NaN Loss

**Error:** Loss becomes NaN

**Solution:**
```bash
# Reduce learning rate
--lr 5e-4

# Increase gradient clipping
--max_grad_norm 0.5

# Check data for anomalies
python verify_features.py --features_root /path/to/features
```

### Issue 3: Training Too Slow

**Problem:** Each epoch takes > 1 hour

**Solution:**
```bash
# Increase num_workers
--num_workers 8

# Reduce max_frames
--max_frames 100

# Use smaller model
--d_model 256
```

### Issue 4: Low Validation AUC

**Problem:** Training AUC is high but validation AUC is low (overfitting)

**Solution:**
```bash
# Increase dropout
--dropout 0.2

# Increase weight decay
--weight_decay 0.05

# Add data augmentation (requires code modification)
```

---

## 📁 Output Files

After training, you'll have:

```
checkpoints/three_branch_baseline/
├── args.json               # Training arguments
├── checkpoint_epoch5.pth   # Periodic checkpoints
├── checkpoint_epoch10.pth
├── ...
├── best.pth               # Best model (highest val AUC)
└── final.pth              # Final model (last epoch)

logs/
└── train_baseline_20241224_123456.log  # Training log
```

**Checkpoint contents:**
```python
checkpoint = torch.load('best.pth')
# Contains:
# - epoch: int
# - model_state_dict: OrderedDict
# - optimizer_state_dict: OrderedDict
# - metrics: dict (fused_auc, cm_auc, ao_auc, vo_auc, etc.)
# - args: dict (all training arguments)
```

---

## 🎯 Next Steps After Training

### 1. Evaluate on Test Set

```bash
python evaluate_three_branch.py \
  --checkpoint checkpoints/three_branch_baseline/best.pth \
  --features_root /hpc2ssd/JH_DATA/spooler/xfeng733/FakeAV_feats \
  --split test \
  --output_dir results/baseline_test
```

### 2. Visualize Branch Contributions

```bash
python visualize_three_branch.py \
  --checkpoint checkpoints/three_branch_baseline/best.pth \
  --features_root /hpc2ssd/JH_DATA/spooler/xfeng733/FakeAV_feats \
  --output_dir visualizations/branch_analysis \
  --num_samples 50
```

### 3. Analyze Failure Cases

```bash
python analyze_failures.py \
  --checkpoint checkpoints/three_branch_baseline/best.pth \
  --features_root /hpc2ssd/JH_DATA/spooler/xfeng733/FakeAV_feats \
  --output_dir analysis/failures
```

### 4. Extend to Frame-Level Localization

```bash
# Transfer learned weights to localization model
python train_lavdf_localization.py \
  --features_root /hpc2ssd/JH_DATA/spooler/xfeng733/lavdf_feats \
  --load_checkpoint checkpoints/three_branch_baseline/best.pth \
  --freeze_branches \
  --epochs 50
```

---

## 📞 Support

If you encounter issues:
1. Check this guide's Troubleshooting section
2. Check `docs/THREE_BRANCH_GUIDE.md` for architecture details
3. Review training logs for error messages

---

## 🎉 Success Checklist

- [ ] Environment activated (`conda activate voca`)
- [ ] Latest code pulled (`git pull`)
- [ ] Quick test passed (5 epochs)
- [ ] Full training launched (100 epochs)
- [ ] Training monitored (logs checked)
- [ ] Best model saved (fused AUC > 90%)
- [ ] Results evaluated on test set
- [ ] Branch contributions visualized

---

Happy Training! 🚀

