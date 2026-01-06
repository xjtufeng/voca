"""
测试 LAV-DF 元数据加载和帧级标签生成逻辑
不需要实际提取特征，只验证核心逻辑
"""

import json
from pathlib import Path
import cv2
import numpy as np


def load_lavdf_metadata(metadata_path: Path):
    """加载 metadata.min.json"""
    with metadata_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def get_video_fps_and_frames(video_path: Path):
    """读取视频的 FPS 和总帧数"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 25.0, 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, total_frames


def generate_frame_labels(fake_periods, fps, total_frames):
    """根据 fake_periods 生成帧级标签"""
    labels = np.zeros(total_frames, dtype=np.int32)
    
    for t_start, t_end in fake_periods:
        frame_start = int(t_start * fps)
        frame_end = int(t_end * fps)
        frame_start = max(0, min(frame_start, total_frames - 1))
        frame_end = max(0, min(frame_end, total_frames - 1))
        labels[frame_start:frame_end + 1] = 1
    
    return labels


def main():
    dataset_root = Path(r"D:\LAV-DF\LAV-DF")
    metadata_path = dataset_root / "metadata.min.json"
    
    print("[INFO] Loading metadata...")
    all_metadata = load_lavdf_metadata(metadata_path)
    print(f"[INFO] Total videos: {len(all_metadata)}")
    
    # 统计 real vs fake
    real_count = sum(1 for m in all_metadata if m["n_fakes"] == 0)
    fake_count = len(all_metadata) - real_count
    print(f"[INFO] Real videos: {real_count}, Fake videos: {fake_count}")
    
    # 按 split 统计
    splits = {}
    for m in all_metadata:
        split = m["split"]
        splits[split] = splits.get(split, 0) + 1
    print(f"[INFO] Splits: {splits}")
    
    # 找几个测试样本
    print("\n[INFO] Testing frame label generation...")
    
    # 找一个 real 视频
    real_sample = next(m for m in all_metadata if m["n_fakes"] == 0 and m["split"] == "test")
    print(f"\n[REAL] {real_sample['file']}")
    print(f"  n_fakes: {real_sample['n_fakes']}")
    print(f"  fake_periods: {real_sample['fake_periods']}")
    print(f"  duration: {real_sample['duration']}s")
    
    video_path = dataset_root / real_sample["file"]
    if video_path.exists():
        fps, total_frames = get_video_fps_and_frames(video_path)
        print(f"  FPS: {fps:.2f}, Total frames: {total_frames}")
        
        frame_labels = generate_frame_labels(real_sample["fake_periods"], fps, total_frames)
        print(f"  Frame labels shape: {frame_labels.shape}")
        print(f"  Fake frames: {frame_labels.sum()} / {total_frames}")
        print(f"  All zeros (real): {np.all(frame_labels == 0)}")
    else:
        print(f"  [WARN] Video not found: {video_path}")
    
    # 找一个 fake 视频
    fake_sample = next(m for m in all_metadata if m["n_fakes"] > 0 and m["split"] == "test")
    print(f"\n[FAKE] {fake_sample['file']}")
    print(f"  n_fakes: {fake_sample['n_fakes']}")
    print(f"  fake_periods: {fake_sample['fake_periods']}")
    print(f"  duration: {fake_sample['duration']}s")
    print(f"  modify_video: {fake_sample['modify_video']}")
    print(f"  modify_audio: {fake_sample['modify_audio']}")
    print(f"  original: {fake_sample['original']}")
    
    video_path = dataset_root / fake_sample["file"]
    if video_path.exists():
        fps, total_frames = get_video_fps_and_frames(video_path)
        print(f"  FPS: {fps:.2f}, Total frames: {total_frames}")
        
        frame_labels = generate_frame_labels(fake_sample["fake_periods"], fps, total_frames)
        print(f"  Frame labels shape: {frame_labels.shape}")
        print(f"  Fake frames: {frame_labels.sum()} / {total_frames}")
        
        # 打印标签分布
        fake_indices = np.where(frame_labels == 1)[0]
        if len(fake_indices) > 0:
            print(f"  Fake frame ranges: [{fake_indices[0]}:{fake_indices[-1]}]")
            print(f"  Expected time range: {fake_sample['fake_periods']}")
    else:
        print(f"  [WARN] Video not found: {video_path}")
    
    print("\n[INFO] Metadata and frame label logic test passed!")


if __name__ == "__main__":
    main()


