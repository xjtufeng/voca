#!/usr/bin/env python3
"""
Split 10% of train samples to dev using stable hash-based bucketing
Creates symlinks from train to dev_temp, then renames dev_temp to dev
"""
import os
import sys
from pathlib import Path
import hashlib
import argparse


def split_bucket(video_id: str) -> int:
    """Stable 0..9 bucket for splitting"""
    return int(hashlib.md5(video_id.encode("utf-8")).hexdigest(), 16) % 10


def main():
    parser = argparse.ArgumentParser(description="Split dev from train using hash bucketing")
    parser.add_argument("--features_root", type=str, required=True,
                        help="Path to LAV-DF_feats root")
    parser.add_argument("--dev_bucket", type=int, default=9,
                        help="Which bucket to use for dev (0-9, default: 9)")
    parser.add_argument("--backup_old_dev", action="store_true",
                        help="Backup old dev to dev_old before replacing")
    args = parser.parse_args()
    
    features_root = Path(args.features_root)
    train_dir = features_root / "train"
    dev_temp_dir = features_root / "dev_temp"
    dev_dir = features_root / "dev"
    
    if not train_dir.exists():
        print(f"[ERROR] Train directory not found: {train_dir}")
        sys.exit(1)
    
    # Create dev_temp directory
    dev_temp_dir.mkdir(exist_ok=True)
    print(f"[INFO] Creating dev_temp at: {dev_temp_dir}")
    
    # Get all train samples
    all_samples = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
    print(f"[INFO] Total train samples: {len(all_samples)}")
    
    # Select dev samples using hash bucket
    dev_samples = [s for s in all_samples if split_bucket(s) == args.dev_bucket]
    print(f"[INFO] Dev samples (bucket {args.dev_bucket}): {len(dev_samples)} ({len(dev_samples)/len(all_samples)*100:.1f}%)")
    
    # Create symlinks
    created = 0
    skipped = 0
    for sample in dev_samples:
        src = train_dir / sample
        dst = dev_temp_dir / sample
        
        if dst.exists():
            skipped += 1
            continue
        
        try:
            os.symlink(src, dst)
            created += 1
        except Exception as e:
            print(f"[WARN] Failed to create symlink for {sample}: {e}")
    
    print(f"[INFO] Created {created} new symlinks, skipped {skipped} existing")
    print(f"[INFO] Total in dev_temp: {len(list(dev_temp_dir.iterdir()))}")
    
    # Backup old dev if requested
    if args.backup_old_dev and dev_dir.exists():
        dev_old_dir = features_root / "dev_old"
        if dev_old_dir.exists():
            print(f"[INFO] dev_old already exists, skipping backup")
        else:
            print(f"[INFO] Backing up old dev to dev_old")
            dev_dir.rename(dev_old_dir)
    
    # Rename dev_temp to dev
    if dev_dir.exists():
        print(f"[WARN] dev directory already exists. Remove it manually if you want to replace:")
        print(f"  rm -rf {dev_dir}")
        print(f"  mv {dev_temp_dir} {dev_dir}")
    else:
        print(f"[INFO] Renaming dev_temp to dev")
        dev_temp_dir.rename(dev_dir)
        print(f"[INFO] Dev created with {len(list(dev_dir.iterdir()))} samples")
    
    print("\n[SUCCESS] Dev split complete!")
    print(f"  Train samples: {len([s for s in all_samples if split_bucket(s) != args.dev_bucket])}")
    print(f"  Dev samples: {len(dev_samples)}")


if __name__ == "__main__":
    main()

