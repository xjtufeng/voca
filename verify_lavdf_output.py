"""
验证 LAV-DF 特征提取输出结构
检查 .npz 文件是否包含正确的字段和数据
"""

import argparse
from pathlib import Path
import numpy as np


def verify_visual_npz(npz_path: Path):
    """验证 visual_embeddings.npz 的结构"""
    print(f"\n[CHECK] {npz_path}")
    
    if not npz_path.exists():
        print("  ❌ File not found")
        return False
    
    try:
        data = np.load(npz_path, allow_pickle=True)
        keys = list(data.keys())
        print(f"  Keys: {keys}")
        
        # 检查必需字段
        required = ['embeddings', 'paths', 'frame_labels']
        missing = [k for k in required if k not in keys]
        if missing:
            print(f"  ❌ Missing keys: {missing}")
            return False
        
        embeddings = data['embeddings']
        paths = data['paths']
        frame_labels = data['frame_labels']
        
        print(f"  embeddings shape: {embeddings.shape}")
        print(f"  paths length: {len(paths)}")
        print(f"  frame_labels shape: {frame_labels.shape}")
        
        # 检查维度匹配
        if embeddings.shape[0] != len(paths) or embeddings.shape[0] != len(frame_labels):
            print(f"  ❌ Shape mismatch!")
            return False
        
        # 检查标签范围
        unique_labels = np.unique(frame_labels)
        print(f"  frame_labels unique values: {unique_labels}")
        if not np.all(np.isin(unique_labels, [0, 1])):
            print(f"  ❌ Invalid label values (should be 0 or 1)")
            return False
        
        fake_count = np.sum(frame_labels == 1)
        real_count = np.sum(frame_labels == 0)
        print(f"  Real frames: {real_count}, Fake frames: {fake_count}")
        
        print("  ✅ Valid structure")
        return True
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


def verify_audio_npz(npz_path: Path):
    """验证 audio_embeddings.npz 的结构"""
    print(f"\n[CHECK] {npz_path}")
    
    if not npz_path.exists():
        print("  ❌ File not found")
        return False
    
    try:
        data = np.load(npz_path)
        keys = list(data.keys())
        print(f"  Keys: {keys}")
        
        if 'embeddings' not in keys:
            print(f"  ❌ Missing 'embeddings' key")
            return False
        
        embeddings = data['embeddings']
        print(f"  embeddings shape: {embeddings.shape}")
        
        print("  ✅ Valid structure")
        return True
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Verify LAV-DF feature extraction outputs")
    parser.add_argument("--output_root", type=str, required=True, help="Feature output root")
    parser.add_argument("--splits", type=str, default="test", help="Splits to check")
    parser.add_argument("--max_samples", type=int, default=5, help="Max samples to check per split")
    args = parser.parse_args()
    
    output_root = Path(args.output_root)
    if not output_root.exists():
        print(f"[ERROR] Output root not found: {output_root}")
        return
    
    splits = args.splits.split(",")
    
    for split in splits:
        split_dir = output_root / split
        if not split_dir.exists():
            print(f"\n[WARN] Split directory not found: {split_dir}")
            continue
        
        print(f"\n{'='*60}")
        print(f"[INFO] Checking split: {split}")
        print(f"{'='*60}")
        
        video_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()])
        print(f"[INFO] Found {len(video_dirs)} video directories")
        
        if not video_dirs:
            print(f"[WARN] No video directories found in {split_dir}")
            continue
        
        # 检查前 N 个样本
        samples_to_check = video_dirs[:args.max_samples]
        
        valid_count = 0
        for video_dir in samples_to_check:
            print(f"\n[VIDEO] {video_dir.name}")
            
            visual_npz = video_dir / "visual_embeddings.npz"
            audio_npz = video_dir / "audio_embeddings.npz"
            
            visual_ok = verify_visual_npz(visual_npz)
            audio_ok = verify_audio_npz(audio_npz)
            
            if visual_ok and audio_ok:
                valid_count += 1
        
        print(f"\n[SUMMARY] {split}: {valid_count}/{len(samples_to_check)} samples valid")
    
    print(f"\n{'='*60}")
    print("[INFO] Verification complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()


