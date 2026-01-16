#!/usr/bin/env python3
"""
Create identity-independent 5-fold cross-validation splits for FakeAVCeleb
Following MRDF paper protocol:
- 4-class balanced setting (FAFV/FARV/RAFV/RARV)
- Identity-independent splits
- 5-fold cross-validation
"""
import re
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
import numpy as np


def extract_identity(video_name: str) -> str:
    """
    Extract identity from video directory name
    
    Examples:
        RealVideo-RealAudio_Asian (East)_men_id05383_00015 -> id05383
        FakeVideo-FakeAudio_..._id00618_00195_id00231_wavtolip -> id00618
    """
    # Find first occurrence of id\d+
    match = re.search(r'(id\d+)', video_name)
    return match.group(1) if match else None


def infer_label_from_name(video_name: str) -> int:
    """
    Infer 4-class label from directory name
    
    Returns:
        0: FAFV (Fake Audio + Fake Video)
        1: FARV (Fake Audio + Real Video)
        2: RAFV (Real Audio + Fake Video)
        3: RARV (Real Audio + Real Video)
        -1: Unknown
    """
    if 'FakeVideo-FakeAudio' in video_name:
        return 0  # FAFV
    elif 'FakeVideo-RealAudio' in video_name:
        return 1  # FARV
    elif 'RealVideo-FakeAudio' in video_name:
        return 2  # RAFV
    elif 'RealVideo-RealAudio' in video_name:
        return 3  # RARV
    else:
        return -1  # Unknown


def create_5fold_splits(
    features_root: str,
    output_dir: str,
    seed: int = 42
):
    """
    Create identity-independent 5-fold splits
    
    Args:
        features_root: Path to FakeAV_feats (with real/ and fake/ subdirs)
        output_dir: Output directory for split files
        seed: Random seed for reproducibility
    """
    features_root = Path(features_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("CREATING IDENTITY-INDEPENDENT 5-FOLD SPLITS")
    print("=" * 70)
    print(f"Features root: {features_root}")
    print(f"Output dir: {output_dir}")
    print(f"Random seed: {seed}")
    print()
    
    # Step 1: Collect all videos with their identities and labels
    videos_by_identity = defaultdict(list)  # identity -> [(video_info)]
    videos_by_class = defaultdict(list)  # label -> [video_info]
    
    print("[Step 1] Scanning video features...")
    for label_dir in ['real', 'fake']:
        label_path = features_root / label_dir
        if not label_path.exists():
            print(f"[WARN] Directory not found: {label_path}")
            continue
        
        for video_dir in label_path.iterdir():
            if not video_dir.is_dir():
                continue
            
            # Check if features exist
            audio_npz = video_dir / "audio_embeddings.npz"
            visual_npz = video_dir / "visual_embeddings.npz"
            if not audio_npz.exists() or not visual_npz.exists():
                continue
            
            video_name = video_dir.name
            identity = extract_identity(video_name)
            if identity is None:
                print(f"[WARN] Cannot extract identity from: {video_name}")
                continue
            
            label = infer_label_from_name(video_name)
            if label == -1:
                print(f"[WARN] Cannot infer label from: {video_name}")
                continue
            
            video_info = {
                'video_id': video_name,
                'identity': identity,
                'label': label,
                'label_name': ['FAFV', 'FARV', 'RAFV', 'RARV'][label],
                'path': f"{label_dir}/{video_name}"
            }
            
            videos_by_identity[identity].append(video_info)
            videos_by_class[label].append(video_info)
    
    num_identities = len(videos_by_identity)
    num_videos = sum(len(v) for v in videos_by_identity.values())
    
    print(f"\n✅ Scanned {num_videos} videos from {num_identities} identities")
    
    # Step 2: Report class distribution
    print(f"\n[Step 2] Overall class distribution:")
    class_names = ['FAFV', 'FARV', 'RAFV', 'RARV']
    class_counts = [len(videos_by_class[i]) for i in range(4)]
    
    for i, (name, count) in enumerate(zip(class_names, class_counts)):
        pct = count / num_videos * 100
        print(f"  {i} - {name}: {count:5d} ({pct:5.1f}%)")
    
    print(f"\n⚠️  Note: Classes are IMBALANCED!")
    print(f"   Balanced sampling (1:1:1:1) will be needed during training.")
    
    # Step 3: Split identities into 5 folds (stratified by class if possible)
    print(f"\n[Step 3] Creating 5 folds (identity-independent)...")
    
    np.random.seed(seed)
    identities = list(videos_by_identity.keys())
    np.random.shuffle(identities)
    
    # Simple split: divide identities into 5 equal parts
    fold_size = len(identities) // 5
    folds = []
    for i in range(5):
        if i < 4:
            fold_ids = identities[i * fold_size:(i + 1) * fold_size]
        else:
            # Last fold gets remaining identities
            fold_ids = identities[i * fold_size:]
        folds.append(fold_ids)
    
    print(f"   Identities per fold: {[len(f) for f in folds]}")
    
    # Step 4: Generate train/test splits for each fold
    print(f"\n[Step 4] Generating train/test splits for each fold...")
    
    fold_stats = []
    
    for fold_idx in range(5):
        test_identities = set(folds[fold_idx])
        train_identities = set()
        for i in range(5):
            if i != fold_idx:
                train_identities.update(folds[i])
        
        # Collect train/test videos
        train_videos = []
        test_videos = []
        
        for identity, videos in videos_by_identity.items():
            if identity in test_identities:
                test_videos.extend(videos)
            elif identity in train_identities:
                train_videos.extend(videos)
        
        # Count class distribution
        train_class_counts = [0, 0, 0, 0]
        test_class_counts = [0, 0, 0, 0]
        
        for video in train_videos:
            train_class_counts[video['label']] += 1
        for video in test_videos:
            test_class_counts[video['label']] += 1
        
        print(f"\n  [Fold {fold_idx}]")
        print(f"    Train: {len(train_videos):5d} videos, {len(train_identities):3d} identities")
        print(f"           FAFV={train_class_counts[0]:4d}, FARV={train_class_counts[1]:4d}, "
              f"RAFV={train_class_counts[2]:4d}, RARV={train_class_counts[3]:4d}")
        print(f"    Test:  {len(test_videos):5d} videos, {len(test_identities):3d} identities")
        print(f"           FAFV={test_class_counts[0]:4d}, FARV={test_class_counts[1]:4d}, "
              f"RAFV={test_class_counts[2]:4d}, RARV={test_class_counts[3]:4d}")
        
        # Save to JSON files
        train_file = output_dir / f'fold_{fold_idx}_train.json'
        test_file = output_dir / f'fold_{fold_idx}_test.json'
        
        with open(train_file, 'w') as f:
            json.dump(train_videos, f, indent=2)
        
        with open(test_file, 'w') as f:
            json.dump(test_videos, f, indent=2)
        
        print(f"    Saved: {train_file.name}, {test_file.name}")
        
        fold_stats.append({
            'fold': fold_idx,
            'train': {
                'num_videos': len(train_videos),
                'num_identities': len(train_identities),
                'class_counts': train_class_counts
            },
            'test': {
                'num_videos': len(test_videos),
                'num_identities': len(test_identities),
                'class_counts': test_class_counts
            }
        })
    
    # Save metadata
    metadata = {
        'num_folds': 5,
        'seed': seed,
        'total_identities': num_identities,
        'total_videos': num_videos,
        'overall_class_counts': class_counts,
        'class_names': class_names,
        'fold_stats': fold_stats
    }
    
    metadata_file = output_dir / 'metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✅ Metadata saved: {metadata_file}")
    print("=" * 70)
    print("5-FOLD SPLITS CREATED SUCCESSFULLY")
    print("=" * 70)
    print(f"\nOutput files in {output_dir}:")
    print("  - fold_0_train.json, fold_0_test.json")
    print("  - fold_1_train.json, fold_1_test.json")
    print("  - ...")
    print("  - fold_4_train.json, fold_4_test.json")
    print("  - metadata.json")
    print()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Create identity-independent 5-fold CV splits for FakeAVCeleb'
    )
    parser.add_argument('--features_root', type=str, required=True,
                        help='Path to FakeAV_feats directory')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for split files')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    create_5fold_splits(
        features_root=args.features_root,
        output_dir=args.output_dir,
        seed=args.seed
    )

