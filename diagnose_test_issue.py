#!/usr/bin/env python3
"""
Diagnose test set evaluation issues
Checks for common problems: label polarity, masking, data consistency
"""
import torch
import numpy as np
from pathlib import Path
from dataset_localization import get_dataloaders
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--features_root', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--num_samples', type=int, default=20,
                        help='Number of test samples to inspect')
    args = parser.parse_args()
    
    print("=" * 70)
    print("DIAGNOSTIC REPORT FOR TEST SET ISSUES")
    print("=" * 70)
    
    # 1. Check checkpoint
    print("\n[1] CHECKPOINT INFO")
    print("-" * 70)
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    print(f"Checkpoint file: {args.checkpoint}")
    print(f"Epoch: {ckpt.get('epoch', '?')}")
    if 'metrics' in ckpt:
        print(f"Val metrics from training:")
        for k, v in ckpt['metrics'].items():
            if isinstance(v, (int, float)):
                print(f"  {k}: {v:.4f}")
    
    # 2. Load test data
    print("\n[2] TEST DATA LOADING")
    print("-" * 70)
    dataloaders = get_dataloaders(
        features_root=args.features_root,
        splits=['test'],
        batch_size=1,  # Load one at a time for inspection
        num_workers=0,
        max_frames=512,
        stride=1,
        distributed=False
    )
    
    test_loader = dataloaders['test']
    test_dataset = test_loader.dataset
    
    print(f"Features root: {args.features_root}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Total frames: {test_dataset.total_frames}")
    print(f"Fake frames: {test_dataset.fake_frames} ({test_dataset.fake_ratio*100:.2f}%)")
    
    # 3. Inspect sample features
    print("\n[3] FEATURE DIMENSION CHECK")
    print("-" * 70)
    sample0 = test_dataset[0]
    print(f"Visual shape: {sample0['visual'].shape}")
    print(f"Audio shape: {sample0['audio'].shape}")
    print(f"Frame labels shape: {sample0['frame_labels'].shape}")
    print(f"Video label: {sample0['video_label'].item()}")
    
    # 4. Label statistics
    print("\n[4] LABEL DISTRIBUTION (First 20 samples)")
    print("-" * 70)
    print(f"{'Video ID':<15} {'T':>6} {'Valid':>6} {'Fake':>6} {'%Fake':>8} {'Video':>6}")
    print("-" * 70)
    
    all_real_count = 0
    all_fake_count = 0
    video_label_dist = {0: 0, 1: 0}
    
    for i in range(min(args.num_samples, len(test_dataset))):
        sample = test_dataset[i]
        video_id = sample['video_id']
        frame_labels = sample['frame_labels'].numpy()
        video_label = sample['video_label'].item()
        
        T = len(frame_labels)
        num_fake = int(frame_labels.sum())
        pct_fake = (num_fake / T * 100) if T > 0 else 0
        
        all_real_count += (T - num_fake)
        all_fake_count += num_fake
        video_label_dist[video_label] += 1
        
        print(f"{video_id:<15} {T:>6} {T:>6} {num_fake:>6} {pct_fake:>7.1f}% {video_label:>6}")
    
    print("-" * 70)
    print(f"Total frames: {all_real_count + all_fake_count}")
    print(f"  Real: {all_real_count} ({all_real_count/(all_real_count+all_fake_count)*100:.1f}%)")
    print(f"  Fake: {all_fake_count} ({all_fake_count/(all_real_count+all_fake_count)*100:.1f}%)")
    print(f"Video labels: Real={video_label_dist[0]}, Fake={video_label_dist[1]}")
    
    # 5. Check for common issues
    print("\n[5] POTENTIAL ISSUES")
    print("-" * 70)
    
    issues_found = []
    
    # Issue: All labels are 0 or 1
    if all_fake_count == 0:
        issues_found.append("❌ No fake frames found! Labels might be wrong or all videos are real")
    elif all_real_count == 0:
        issues_found.append("❌ No real frames found! Labels might be flipped or all videos are fake")
    
    # Issue: Extreme imbalance
    if all_fake_count > 0 and all_real_count > 0:
        ratio = max(all_fake_count, all_real_count) / min(all_fake_count, all_real_count)
        if ratio > 50:
            issues_found.append(f"⚠️  Extreme class imbalance: ratio={ratio:.1f}:1")
    
    # Issue: All video labels are same
    if video_label_dist[0] == 0 or video_label_dist[1] == 0:
        issues_found.append("⚠️  All video labels are the same! Check video-level annotation")
    
    if not issues_found:
        issues_found.append("✅ No obvious label distribution issues detected")
    
    for issue in issues_found:
        print(f"  {issue}")
    
    # 6. Check padding/masking
    print("\n[6] PADDING/MASKING CHECK")
    print("-" * 70)
    batch = next(iter(test_loader))
    mask = batch['mask'][0]  # [T]
    frame_labels = batch['frame_labels'][0]  # [T]
    
    num_valid = (~mask).sum().item()
    num_padded = mask.sum().item()
    
    print(f"Batch size=1 sample:")
    print(f"  Total length: {len(mask)}")
    print(f"  Valid frames: {num_valid}")
    print(f"  Padded frames: {num_padded}")
    print(f"  Frame labels in padded region: {frame_labels[mask].sum().item()}")
    
    if frame_labels[mask].sum().item() > 0:
        issues_found.append("❌ Frame labels are non-zero in padded region!")
    
    # 7. Feature statistics
    print("\n[7] FEATURE STATISTICS")
    print("-" * 70)
    visual = batch['visual'][0]  # [T, D]
    audio = batch['audio'][0]    # [T, D]
    
    print(f"Visual features:")
    print(f"  Mean: {visual.mean().item():.4f}, Std: {visual.std().item():.4f}")
    print(f"  Min: {visual.min().item():.4f}, Max: {visual.max().item():.4f}")
    
    print(f"Audio features:")
    print(f"  Mean: {audio.mean().item():.4f}, Std: {audio.std().item():.4f}")
    print(f"  Min: {audio.min().item():.4f}, Max: {audio.max().item():.4f}")
    
    # Check if features are normalized
    if abs(visual.mean().item()) > 5 or abs(audio.mean().item()) > 5:
        print("  ⚠️  Features might not be normalized (mean >> 0)")
    
    print("\n" + "=" * 70)
    print("DIAGNOSIS COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()

