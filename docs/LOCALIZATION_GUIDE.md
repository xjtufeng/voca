# Frame-Level Deepfake Localization Guide

This guide covers the complete pipeline for training and evaluating frame-level deepfake localization models on the LAV-DF dataset.

## Overview

The localization pipeline consists of:
1. **Feature Extraction**: Extract visual/audio embeddings + frame-level labels
2. **Model Training**: Train cross-modal Transformer for frame-level classification
3. **Evaluation**: Compute frame/video-level metrics and IoU
4. **Visualization**: Generate time-series curves showing detection results
5. **Inference**: Apply trained model to new videos

---

## 1. Feature Extraction

First, extract features from the LAV-DF dataset with frame-level labels:

```bash
# Extract features for all splits
python prepare_lavdf_features.py \
  --dataset_root /hpc2hdd/home/xfeng733/LAV-DF/LAV-DF \
  --metadata /hpc2hdd/home/xfeng733/LAV-DF/LAV-DF/metadata.min.json \
  --output_root /hpc2ssd/JH_DATA/spooler/xfeng733/lavdf_feats \
  --splits train dev test \
  --use_gpu \
  --skip_existing
```

**Output structure:**
```
lavdf_feats/
  train/
    {video_id}/
      visual_embeddings.npz    # {embeddings: [T, 512], frame_labels: [T]}
      audio_embeddings.npz     # {embeddings: [T_a, 1024]}
      similarity_stats.npz     # (optional) audio-visual similarity
  dev/
  test/
```

---

## 2. Model Training

### Quick Start

```bash
# Single GPU training
python train_lavdf_localization.py \
  --features_root /hpc2ssd/JH_DATA/spooler/xfeng733/lavdf_feats \
  --splits train dev \
  --batch_size 8 \
  --epochs 100 \
  --lr 1e-4 \
  --output_dir ./checkpoints/localization

# Multi-GPU DDP training (4 GPUs)
torchrun --nproc_per_node=4 train_lavdf_localization.py \
  --features_root /hpc2ssd/JH_DATA/spooler/xfeng733/lavdf_feats \
  --splits train dev \
  --batch_size 8 \
  --epochs 100 \
  --lr 1e-4 \
  --output_dir ./checkpoints/localization
```

### Using Shell Script

```bash
# Edit configuration in scripts/train_localization.sh
bash scripts/train_localization.sh
```

### Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--max_frames` | 512 | Max frames per video (truncate/split if longer) |
| `--d_model` | 512 | Transformer hidden dimension |
| `--num_layers` | 4 | Number of Transformer layers |
| `--pos_weight` | -1 | BCE positive weight (auto if -1) |
| `--use_focal` | False | Use Focal Loss instead of BCE |
| `--video_loss_weight` | 0.3 | Weight for video-level auxiliary loss |
| `--smooth_loss_weight` | 0.1 | Weight for temporal smoothness regularization |
| `--no_cross_attn` | False | Disable cross-modal attention |
| `--no_video_head` | False | Disable video-level classification head |

### Training Losses

**Total Loss:**
```
L_total = L_frame + α * L_video + β * L_smooth
```

- **Frame Loss**: BCE with `pos_weight` or Focal Loss
- **Video Loss**: Auxiliary video-level classification
- **Smoothness Loss**: Temporal consistency regularization

---

## 3. Evaluation

Evaluate trained model on test set:

```bash
python evaluate_localization.py \
  --features_root /hpc2ssd/JH_DATA/spooler/xfeng733/lavdf_feats \
  --split test \
  --checkpoint ./checkpoints/localization/best.pth \
  --threshold 0.5 \
  --output ./results/eval_results.json
```

**Metrics computed:**

- **Frame-Level**: AUC, AP, F1, Precision, Recall
- **Video-Level**: AUC, AP, F1, Precision, Recall
- **Localization**: IoU (mean, median, std)

**Example output:**
```
EVALUATION RESULTS
==================================================
Frame-Level Metrics:
  AUC:       0.8752
  AP:        0.8234
  F1:        0.7891
  Precision: 0.8123
  Recall:    0.7654

Video-Level Metrics:
  AUC:       0.9234
  AP:        0.9012
  F1:        0.8567
  Precision: 0.8734
  Recall:    0.8412

Localization IoU:
  Mean:   0.6543
  Median: 0.6789
  Std:    0.1234
```

---

## 4. Visualization

Generate time-series curves for localization results:

```bash
python visualize_localization.py \
  --input ./results/eval_results.json \
  --output_dir ./visualizations \
  --fps 25.0 \
  --threshold 0.5 \
  --max_videos 20
```

**Output visualizations:**

1. **Summary Statistics** (`summary_statistics.png`):
   - IoU distribution
   - F1 score distribution
   - IoU vs Fake Ratio scatter
   - F1 vs Fake Ratio scatter

2. **Per-Video Time-Series** (if ground truth available):
   - Frame-level fake probability curve
   - Ground truth vs prediction comparison
   - Audio-visual similarity curve (if available)
   - Annotated fake segments

---

## 5. Inference

Apply trained model to new videos:

```bash
python infer_localization.py \
  --checkpoint ./checkpoints/localization/best.pth \
  --features_dir /hpc2ssd/JH_DATA/spooler/xfeng733/lavdf_feats/test \
  --output_dir ./inference_results \
  --threshold 0.5 \
  --visualize \
  --fps 25.0
```

**Output:**
- `inference_results.json`: Predictions for all videos
- `{video_id}.png`: Visualization for each video (if `--visualize`)

**JSON format:**
```json
{
  "video_id": "xxx",
  "num_frames": 500,
  "video_prob": 0.87,
  "video_pred": 1,
  "num_fake_frames_pred": 125,
  "fake_ratio_pred": 0.25,
  "segments": [
    [100, 150, 0.89],
    [300, 350, 0.82]
  ]
}
```

---

## 6. Complete Workflow

### Step 1: Feature Extraction (Multi-GPU Parallel)

```bash
# GPU 0: train split
CUDA_VISIBLE_DEVICES=0 nohup python prepare_lavdf_features.py \
  --dataset_root /hpc2hdd/home/xfeng733/LAV-DF/LAV-DF \
  --metadata /hpc2hdd/home/xfeng733/LAV-DF/LAV-DF/metadata.min.json \
  --output_root /hpc2ssd/JH_DATA/spooler/xfeng733/lavdf_feats \
  --splits train --use_gpu --skip_existing > train_feats.log 2>&1 &

# GPU 1: dev split
CUDA_VISIBLE_DEVICES=1 nohup python prepare_lavdf_features.py \
  --dataset_root /hpc2hdd/home/xfeng733/LAV-DF/LAV-DF \
  --metadata /hpc2hdd/home/xfeng733/LAV-DF/LAV-DF/metadata.min.json \
  --output_root /hpc2ssd/JH_DATA/spooler/xfeng733/lavdf_feats \
  --splits dev --use_gpu --skip_existing > dev_feats.log 2>&1 &

# GPU 2: test split
CUDA_VISIBLE_DEVICES=2 nohup python prepare_lavdf_features.py \
  --dataset_root /hpc2hdd/home/xfeng733/LAV-DF/LAV-DF \
  --metadata /hpc2hdd/home/xfeng733/LAV-DF/LAV-DF/metadata.min.json \
  --output_root /hpc2ssd/JH_DATA/spooler/xfeng733/lavdf_feats \
  --splits test --use_gpu --skip_existing > test_feats.log 2>&1 &
```

### Step 2: Train Model (Multi-GPU DDP)

```bash
nohup bash scripts/train_localization.sh > training.log 2>&1 &
tail -f training.log
```

### Step 3: Evaluate Best Model

```bash
python evaluate_localization.py \
  --features_root /hpc2ssd/JH_DATA/spooler/xfeng733/lavdf_feats \
  --split test \
  --checkpoint ./checkpoints/localization/best.pth \
  --output ./results/test_results.json
```

### Step 4: Generate Visualizations

```bash
python visualize_localization.py \
  --input ./results/test_results.json \
  --output_dir ./visualizations/test
```

### Step 5: Inference on New Data

```bash
python infer_localization.py \
  --checkpoint ./checkpoints/localization/best.pth \
  --features_dir /path/to/new/features \
  --output_dir ./inference_results \
  --visualize
```

---

## 7. Model Architecture

```
Input: Visual [B, T, 512] + Audio [B, T, 1024]
  ↓
Feature Projection → [B, T, d_model]
  ↓
Cross-Modal Attention (Bidirectional)
  ↓
Temporal Transformer Encoder (N layers)
  ↓
├─ Frame Classifier → [B, T, 1] (frame logits)
└─ Video Classifier → [B, 1] (video logit)
```

**Key Components:**
- **Cross-Modal Attention**: Audio-to-Video & Video-to-Audio
- **Temporal Transformer**: Models temporal dependencies
- **Dual Heads**: Frame-level + Video-level classification

---

## 8. Expected Performance

Based on similar works and LAV-DF characteristics:

| Metric | Expected Range |
|--------|----------------|
| Frame AUC | 0.85 - 0.92 |
| Frame F1 | 0.70 - 0.85 |
| Video AUC | 0.90 - 0.95 |
| Video F1 | 0.85 - 0.92 |
| Mean IoU | 0.60 - 0.75 |

**Factors affecting performance:**
- Forgery type (audio-visual sync vs single-modal)
- Segment duration (short segments harder to localize)
- Class imbalance (fake frames often < 20%)
- Boundary ambiguity (transition frames)

---

## 9. Troubleshooting

### Issue: Out of Memory (OOM)

**Solutions:**
- Reduce `--batch_size` (try 4 or 2)
- Reduce `--max_frames` (try 256)
- Use gradient checkpointing (requires code modification)

### Issue: Poor Frame-Level Recall

**Solutions:**
- Increase `--pos_weight` manually (try 10-20)
- Enable Focal Loss: `--use_focal`
- Reduce threshold during evaluation (try 0.3-0.4)

### Issue: Training Loss Not Decreasing

**Solutions:**
- Check data loading (run test in `dataset_localization.py`)
- Reduce learning rate: `--lr 5e-5`
- Increase warmup: `--warmup_epochs 10`
- Disable temporal smoothness: `--smooth_loss_weight 0`

### Issue: High False Positive Rate

**Solutions:**
- Increase threshold (try 0.6-0.7)
- Add temporal smoothing in post-processing
- Filter short segments: increase `--min_segment_length`

---

## 10. Advanced Options

### Custom Loss Configuration

```bash
# Focal Loss with custom parameters
python train_lavdf_localization.py \
  --use_focal \
  --focal_alpha 0.25 \
  --focal_gamma 2.0 \
  --pos_weight -1

# No auxiliary losses
python train_lavdf_localization.py \
  --video_loss_weight 0 \
  --smooth_loss_weight 0
```

### Architecture Variants

```bash
# Larger model
python train_lavdf_localization.py \
  --d_model 768 \
  --num_layers 6 \
  --nhead 12

# No cross-attention (simple fusion)
python train_lavdf_localization.py \
  --no_cross_attn

# Frame-level only (no video head)
python train_lavdf_localization.py \
  --no_video_head \
  --video_loss_weight 0
```

### Resume Training

```bash
python train_lavdf_localization.py \
  --resume ./checkpoints/localization/checkpoint_epoch50.pth \
  --epochs 150
```

---

## 11. File Structure

```
VOCA-Lens/
├── dataset_localization.py       # Dataset and DataLoader
├── model_localization.py         # Model architecture and losses
├── train_lavdf_localization.py   # Training script
├── evaluate_localization.py      # Evaluation script
├── visualize_localization.py     # Visualization tools
├── infer_localization.py         # Inference script
├── scripts/
│   └── train_localization.sh     # Training shell script
└── docs/
    └── LOCALIZATION_GUIDE.md     # This guide
```

---

## References

- LAV-DF Dataset: [https://github.com/ControlNet/LAV-DF](https://github.com/ControlNet/LAV-DF)
- Cross-Modal Attention: Vaswani et al., "Attention is All You Need"
- Focal Loss: Lin et al., "Focal Loss for Dense Object Detection"

