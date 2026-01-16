#!/bin/bash
# Diagnostic script for test set issues

cd ~/jhspoolers/voca

echo "=== Step 1: Check available checkpoints ==="
ls -lh checkpoints/localization_ssd_2gpu/*.pth | tail -10

echo ""
echo "=== Step 2: Check best.pth info ==="
python -c "
import torch
ckpt = torch.load('checkpoints/localization_ssd_2gpu/best.pth', map_location='cpu', weights_only=False)
print(f'best.pth = Epoch {ckpt.get(\"epoch\", \"?\")}')
if 'metrics' in ckpt:
    for k, v in ckpt['metrics'].items():
        if isinstance(v, (int, float)):
            print(f'  {k}: {v:.4f}')
"

echo ""
echo "=== Step 3: Run diagnostic on Epoch 44 ==="
python diagnose_test_issue.py \
  --checkpoint checkpoints/localization_ssd_2gpu/checkpoint_epoch44.pth \
  --features_root /hpc2ssd/JH_DATA/spooler/xfeng733/LAV-DF_feats \
  --num_samples 20

echo ""
echo "=== Step 4: Check evaluate script mask usage ==="
grep -n "mask" evaluate_lavdf_localization.py | head -20

