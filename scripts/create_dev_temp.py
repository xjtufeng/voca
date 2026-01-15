#!/usr/bin/env python3
"""
Create dev_temp from train split using stable hash-based sampling (10%)
Creates symlinks from train to dev_temp without affecting existing dev
"""
import os
import sys
from pathlib import Path
import hashlib
import argparse


def get_hash_bucket(video_id: str, num_buckets: int = 10) -> int:
    """Stable hash bucket for splitting"""
    return int(hashlib.md5(video_id.encode("utf-8")).hexdigest(), 16) % num_buckets


def main():
    parser = argparse.ArgumentParser(
        description="Create dev_temp (10% of train) using hash-based sampling"
    )
    parser.add_argument(
        "--features_root", 
        type=str, 
        required=True,
        help="Path to features root directory (e.g., /path/to/LAV-DF_feats)"
    )
    parser.add_argument(
        "--dev_bucket", 
        type=int, 
        default=9,
        help="Which hash bucket to use for dev (0-9, default: 9 = ~10%%)"
    )
    parser.add_argument(
        "--force", 
        action="store_true",
        help="Remove existing dev_temp before creating new one"
    )
    args = parser.parse_args()
    
    features_root = Path(args.features_root)
    train_dir = features_root / "train"
    dev_temp_dir = features_root / "dev_temp"
    
    # Validate train directory exists
    if not train_dir.exists():
        print(f"[ERROR] Train directory not found: {train_dir}")
        sys.exit(1)
    
    # Handle existing dev_temp
    if dev_temp_dir.exists():
        if args.force:
            print(f"[INFO] Removing existing dev_temp: {dev_temp_dir}")
            import shutil
            shutil.rmtree(dev_temp_dir)
        else:
            print(f"[ERROR] dev_temp already exists: {dev_temp_dir}")
            print(f"[ERROR] Use --force to remove and recreate")
            sys.exit(1)
    
    # Create dev_temp directory
    dev_temp_dir.mkdir(exist_ok=True)
    print(f"[INFO] Creating dev_temp at: {dev_temp_dir}")
    
    # Get all train samples (directories only)
    all_samples = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
    print(f"[INFO] Total train samples: {len(all_samples)}")
    
    if len(all_samples) == 0:
        print(f"[ERROR] No samples found in train directory")
        sys.exit(1)
    
    # Select dev samples using hash bucket
    dev_samples = [s for s in all_samples if get_hash_bucket(s) == args.dev_bucket]
    dev_ratio = len(dev_samples) / len(all_samples) * 100
    print(f"[INFO] Selected bucket {args.dev_bucket}: {len(dev_samples)} samples ({dev_ratio:.1f}%)")
    
    # Create symlinks
    created = 0
    failed = 0
    
    for sample in dev_samples:
        src = train_dir / sample
        dst = dev_temp_dir / sample
        
        try:
            # Use relative symlink for better portability
            os.symlink(f"../train/{sample}", dst)
            created += 1
            if created % 1000 == 0:
                print(f"[INFO] Created {created}/{len(dev_samples)} symlinks...")
        except Exception as e:
            print(f"[WARN] Failed to create symlink for {sample}: {e}")
            failed += 1
    
    print(f"\n[SUCCESS] dev_temp created!")
    print(f"  Created: {created} symlinks")
    print(f"  Failed: {failed}")
    print(f"  Total in dev_temp: {len(list(dev_temp_dir.iterdir()))}")
    print(f"  Path: {dev_temp_dir}")
    
    # Verify some symlinks
    sample_links = list(dev_temp_dir.iterdir())[:3]
    if sample_links:
        print(f"\n[INFO] Sample symlinks:")
        for link in sample_links:
            target = link.resolve() if link.is_symlink() else "NOT A SYMLINK"
            print(f"  {link.name} -> {target}")


if __name__ == "__main__":
    main()

