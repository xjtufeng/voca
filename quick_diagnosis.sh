#!/bin/bash
# Quick diagnosis for test set collapse

echo "================================================"
echo "QUICK DIAGNOSIS FOR TEST SET PERFORMANCE COLLAPSE"
echo "================================================"
echo ""

# Step 1: Find best checkpoint from training logs
echo "[1] Finding best validation epoch from training logs..."
echo ""
grep -E "Epoch [0-9]+\]" ~/jhspoolers/voca/logs/localization_ssd_2gpu_*.log -A 2 | \
grep -E "Val.*Frame AUC" | tail -10

echo ""
echo "[2] Checking checkpoint files..."
ls -lh ~/jhspoolers/voca/checkpoints/localization_ssd_2gpu/checkpoint_epoch*.pth | tail -5

echo ""
echo "[3] Checking best.pth metadata..."
python3 << 'PYEOF'
import torch
import sys

try:
    ckpt = torch.load('/hpc2ssd/JH_DATA/spooler/xfeng733/voca/checkpoints/localization_ssd_2gpu/best.pth', 
                      map_location='cpu', weights_only=False)
    print(f"best.pth corresponds to Epoch {ckpt.get('epoch', '?')}")
    if 'metrics' in ckpt:
        print("Validation metrics when saved:")
        for k, v in ckpt['metrics'].items():
            if isinstance(v, (int, float)):
                print(f"  {k}: {v:.4f}")
except Exception as e:
    print(f"Error loading best.pth: {e}")
    sys.exit(1)
PYEOF

echo ""
echo "[4] Checking if Epoch 99 exists..."
if [ -f ~/jhspoolers/voca/checkpoints/localization_ssd_2gpu/checkpoint_epoch99.pth ]; then
    echo "✅ Epoch 99 checkpoint exists"
    echo ""
    echo "Checking Epoch 99 metadata..."
    python3 << 'PYEOF'
import torch
try:
    ckpt = torch.load('/hpc2ssd/JH_DATA/spooler/xfeng733/voca/checkpoints/localization_ssd_2gpu/checkpoint_epoch99.pth', 
                      map_location='cpu', weights_only=False)
    print(f"Epoch: {ckpt.get('epoch', '?')}")
    if 'metrics' in ckpt:
        print("Validation metrics:")
        for k, v in ckpt['metrics'].items():
            if isinstance(v, (int, float)):
                print(f"  {k}: {v:.4f}")
except Exception as e:
    print(f"Error: {e}")
PYEOF
else
    echo "❌ Epoch 99 checkpoint NOT found"
    echo "Available epochs:"
    ls ~/jhspoolers/voca/checkpoints/localization_ssd_2gpu/checkpoint_epoch*.pth 2>/dev/null | tail -5
fi

echo ""
echo "[5] Sample label check on test set..."
python3 << 'PYEOF'
import sys
sys.path.insert(0, '/hpc2ssd/JH_DATA/spooler/xfeng733/voca')

from dataset_localization import get_dataloaders

print("Loading test set...")
dataloaders = get_dataloaders(
    features_root='/hpc2ssd/JH_DATA/spooler/xfeng733/LAV-DF_feats',
    splits=['test'],
    batch_size=1,
    num_workers=0,
    max_frames=512,
    stride=1,
    distributed=False
)

test_dataset = dataloaders['test'].dataset
print(f"Test set: {len(test_dataset)} videos")
print(f"Total frames: {test_dataset.total_frames}, Fake: {test_dataset.fake_frames} ({test_dataset.fake_ratio*100:.2f}%)")
print("")

print("Sampling 10 test videos...")
print(f"{'Video ID':<15} {'Frames':>7} {'Fake':>6} {'%Fake':>7} {'VideoLbl':>8}")
print("-" * 60)

import numpy as np
np.random.seed(42)
indices = np.random.choice(len(test_dataset), size=min(10, len(test_dataset)), replace=False)

for idx in indices:
    sample = test_dataset[int(idx)]
    video_id = sample['video_id']
    frame_labels = sample['frame_labels'].numpy()
    video_label = sample['video_label'].item()
    
    T = len(frame_labels)
    fake_count = int(frame_labels.sum())
    pct_fake = (fake_count / T * 100) if T > 0 else 0
    
    print(f"{video_id:<15} {T:>7} {fake_count:>6} {pct_fake:>6.1f}% {video_label:>8}")

PYEOF

echo ""
echo "================================================"
echo "DIAGNOSIS COMPLETE"
echo "================================================"
echo ""
echo "Next steps:"
echo "1. If Epoch 99 has much better val metrics than Epoch 44, test it"
echo "2. If labels look wrong (all 0/all 1), check dataset preprocessing"
echo "3. If feature dims differ, check feature extraction consistency"

