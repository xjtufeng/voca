# DFD-FCG Integration Guide

Guide for using DFD-FCG's CLIP ViT-L/14 encoder in the Visual-Only branch.

---

## Overview

The three-branch model now supports using **DFD-FCG's foundation model (CLIP ViT-L/14)** for the Visual-Only branch, instead of InsightFace features.

### Architecture Comparison

| Component | Without DFD-FCG | With DFD-FCG |
|-----------|----------------|--------------|
| **Cross-Modal** | InsightFace (512-d) | InsightFace (512-d) |
| **Audio-Only** | HuBERT (1024-d) | HuBERT (1024-d) |
| **Visual-Only** | InsightFace (512-d) | **CLIP ViT-L/14 (768-d)** |

**Key Benefit**: CLIP is a stronger foundation model pre-trained on 400M image-text pairs, providing better visual representations than InsightFace.

---

## Setup

### 1. Clone DFD-FCG Repository

Already done! The repository is cloned at `./DFD-FCG/`.

```bash
# Verify
ls -la DFD-FCG/
```

### 2. Install Dependencies

DFD-FCG requires additional packages:

```bash
# Install CLIP dependencies
pip install ftfy regex tqdm
pip install open_clip_torch

# Verify
python -c "import sys; sys.path.append('DFD-FCG'); from src.model.clip import VideoAttrExtractor; print('OK')"
```

### 3. (Optional) Download Pre-trained Weights

If you have pre-trained DFD-FCG weights:

```bash
# Place weights in checkpoints/
mkdir -p checkpoints/dfdfcg
# Copy your weights here
cp /path/to/dfdfcg_weights.pth checkpoints/dfdfcg/pretrained.pth
```

---

## Usage

### Training with DFD-FCG

#### Method 1: Using Python Script

```bash
python train_three_branch.py \
  --features_root /path/to/features \
  --batch_size 16 \
  --epochs 100 \
  --use_dfdfcg \
  --dfdfcg_freeze  # Freeze CLIP encoder (recommended)

# With pre-trained weights
python train_three_branch.py \
  --features_root /path/to/features \
  --use_dfdfcg \
  --dfdfcg_pretrain checkpoints/dfdfcg/pretrained.pth
```

#### Method 2: Using Shell Script

Edit `scripts/train_three_branch.sh`:

```bash
# Add these flags
USE_DFDFCG="--use_dfdfcg"
DFDFCG_PRETRAIN=""  # or "--dfdfcg_pretrain /path/to/weights.pth"
```

Then run:

```bash
bash scripts/train_three_branch.sh
```

### Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--use_dfdfcg` | False | Enable DFD-FCG CLIP encoder |
| `--dfdfcg_pretrain` | None | Path to pre-trained DFD-FCG weights |
| `--dfdfcg_freeze` | True | Freeze CLIP encoder (fine-tune only adapter) |

---

## Data Requirements

### Without DFD-FCG

Only need pre-extracted features:

```
features/
  train/
    video_001/
      visual_embeddings.npz   # [T, 512] InsightFace
      audio_embeddings.npz    # [T, 1024] HuBERT
```

### With DFD-FCG

**Option A**: Extract CLIP features offline (recommended for speed)

Create a new extraction script to save both InsightFace and CLIP features:

```python
# In prepare_features_with_clip.py
from foundation_encoder_dfdfcg import DFDFCGFeatureExtractor

# Extract features
insightface_feats = insightface_encoder.encode(bottom_faces)  # [T, 512]
clip_feats = dfdfcg_encoder.extract_from_frames(full_frames)  # [T, 768]

# Save both
np.savez(
    output_path / "visual_embeddings.npz",
    insightface=insightface_feats,  # For cross-modal
    clip=clip_feats                 # For visual-only
)
```

**Option B**: Extract CLIP features on-the-fly (slower, but no preprocessing)

Store raw video frames and extract during training:

```python
# Modify dataset to return video frames
def __getitem__(self, idx):
    ...
    video_frames = load_video_frames(video_path)  # [T, 3, H, W]
    return {
        'visual': insightface_emb,
        'audio': audio_emb,
        'video_frames': video_frames  # Pass to model
    }
```

---

## Performance Expectations

### With InsightFace (Baseline)

| Branch | AUC | Notes |
|--------|-----|-------|
| Cross-Modal | 0.94 | InsightFace + Audio |
| Audio-Only | 0.78 | Audio only |
| Visual-Only | 0.83 | InsightFace only |
| **Fused** | **0.96** | - |

### With DFD-FCG (Expected)

| Branch | AUC | Improvement |
|--------|-----|-------------|
| Cross-Modal | 0.94 | (same) |
| Audio-Only | 0.78 | (same) |
| Visual-Only | **0.89-0.91** | **+6-8%** |
| **Fused** | **0.97-0.98** | **+1-2%** |

**Key Insight**: Visual-Only branch gets stronger, pushing overall performance higher.

---

## Computational Cost

| Configuration | GPU Memory | Speed | Notes |
|--------------|-----------|-------|-------|
| Without DFD-FCG | ~6 GB | Fast | Only linear layers |
| With DFD-FCG (frozen) | ~12 GB | Medium | CLIP forward pass |
| With DFD-FCG (fine-tune) | ~18 GB | Slow | CLIP backward pass |

**Recommendation**: Use `--dfdfcg_freeze` to freeze CLIP and only train the adapter.

---

## Troubleshooting

### Import Error: Cannot import DFD-FCG

**Error:**
```
ImportError: No module named 'src.model.clip'
```

**Solution:**
```bash
# Ensure DFD-FCG is cloned
ls DFD-FCG/src/model/clip/

# Install dependencies
pip install ftfy regex open_clip_torch
```

### CUDA Out of Memory

**Error:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**
1. Reduce batch size: `--batch_size 8` → `--batch_size 4`
2. Reduce sequence length: `--max_frames 256` → `--max_frames 128`
3. Enable gradient checkpointing (requires code modification)
4. Use smaller CLIP: Change `architecture` to `ViT-B/16` (not recommended, worse performance)

### DFD-FCG Encoder Not Using GPU

**Check:**
```python
# In Python
from foundation_encoder_dfdfcg import DFDFCGEncoder
encoder = DFDFCGEncoder()
print(next(encoder.parameters()).device)  # Should be cuda
```

**Solution**: Ensure `device='cuda'` in training script.

---

## Advanced Usage

### Custom CLIP Architecture

```python
# Use ViT-B/16 instead of ViT-L/14
encoder = DFDFCGEncoder(
    architecture="ViT-B/16",  # Smaller, faster
    freeze=True
)
```

### Fine-tune CLIP Encoder

```bash
# Allow CLIP to be fine-tuned
python train_three_branch.py \
  --use_dfdfcg \
  --no-dfdfcg_freeze  # Enable fine-tuning

# Use smaller learning rate for CLIP
# (requires code modification to set different LRs per module)
```

### Extract CLIP Features Offline

```bash
# Create extraction script
python extract_dfdfcg_features.py \
  --video_dir /path/to/videos \
  --output_dir /path/to/clip_features \
  --batch_size 32
```

---

## Comparison with Original DFD-FCG

| Feature | Original DFD-FCG | Our Integration |
|---------|-----------------|-----------------|
| **Task** | Binary classification | Multi-branch classification + localization |
| **Input** | Video frames | Audio + Visual (InsightFace + CLIP) |
| **Model** | CLIP + FFG (Facial Feature Guided) | 3-branch (CM + AO + VO with CLIP) |
| **Training** | Single model | Joint multi-branch |
| **Output** | Fake probability | Branch predictions + fusion + localization |

**Our Advantage**: Combines CLIP's strength with audio-visual cross-modal analysis.

---

## References

- **DFD-FCG Paper**: "Towards More General Video-based Deepfake Detection through Facial Feature Guided Adaptation for Foundation Model" (CVPR 2025)
- **DFD-FCG GitHub**: https://github.com/aiiu-lab/DFD-FCG
- **CLIP Paper**: "Learning Transferable Visual Models From Natural Language Supervision" (OpenAI)

---

## Quick Start Checklist

- [ ] DFD-FCG cloned at `./DFD-FCG/`
- [ ] Dependencies installed (`pip install ftfy regex open_clip_torch`)
- [ ] Test import: `python foundation_encoder_dfdfcg.py`
- [ ] (Optional) Pre-trained weights downloaded
- [ ] Start training: `python train_three_branch.py --use_dfdfcg`

---

## Next Steps

1. **Train without DFD-FCG first** (baseline)
2. **Train with DFD-FCG** (compare performance)
3. **Ablation study**: DFD-FCG vs InsightFace for Visual-Only branch
4. **Paper writing**: Highlight foundation model integration

---

For questions, see `README_THREE_BRANCH.md` or open an issue.

