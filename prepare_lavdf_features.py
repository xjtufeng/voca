"""
LAV-DF Dataset Preprocessing Pipeline:
从 metadata.min.json 读取视频列表，提取 visual/audio embeddings + 帧级标签（frame_labels）

输出结构（在 output_root/<split>/<video_id>/）：
    bottom_faces/               extracted bottom-face frames
    visual_embeddings.npz       {embeddings: (T_v, 512), paths: [...], frame_labels: (T_v,)}
    audio_embeddings.npz        {embeddings: (T_a, 512)}
    similarity_stats.npz        stats + similarities

说明：
- frame_labels[i] = 0 (real) / 1 (fake)，与 visual_embeddings 帧对齐
- 根据 metadata 的 fake_periods 生成
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

from audio_extractor import extract_audio_from_video
from compare_audio_visual import (
    align_embeddings,
    compute_similarity,
    extract_statistics,
    load_embeddings,
    plot_similarity_curve,
)
from face_encoder_insightface import InsightFaceBottomEncoder
from face_extractor import BottomFaceExtractor
from speech_encoder_anitalker import SpeechMotionEncoder


def load_lavdf_metadata(metadata_path: Path) -> List[Dict]:
    """加载 metadata.min.json，返回所有视频元数据列表"""
    with metadata_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def get_video_fps_and_frames(video_path: Path) -> Tuple[float, int]:
    """读取视频的 FPS 和总帧数（用于生成帧级标签）"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 25.0, 0  # fallback
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, total_frames


def generate_frame_labels(
    fake_periods: List[List[float]],
    fps: float,
    total_frames: int
) -> np.ndarray:
    """
    根据 fake_periods（秒）生成帧级标签 (0=real, 1=fake)
    
    Args:
        fake_periods: [[t_start, t_end], ...] 单位秒
        fps: 视频帧率
        total_frames: 视频总帧数
    
    Returns:
        frame_labels: (total_frames,) 的 int array
    """
    labels = np.zeros(total_frames, dtype=np.int32)
    
    for t_start, t_end in fake_periods:
        frame_start = int(t_start * fps)
        frame_end = int(t_end * fps)
        # 确保不越界
        frame_start = max(0, min(frame_start, total_frames - 1))
        frame_end = max(0, min(frame_end, total_frames - 1))
        labels[frame_start:frame_end + 1] = 1
    
    return labels


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def process_lavdf_video(
    meta: Dict,
    dataset_root: Path,
    output_root: Path,
    face_extractor: BottomFaceExtractor,
    face_encoder: InsightFaceBottomEncoder,
    speech_encoder: SpeechMotionEncoder,
    skip_existing: bool = False,
) -> Dict:
    """
    处理单个 LAV-DF 视频：
    1. 提取底脸 + visual embeddings
    2. 提取音频 + audio embeddings
    3. 生成帧级标签（frame_labels）并保存
    4. 计算相似度统计
    """
    file_rel = meta["file"]  # e.g. "train/138009.mp4"
    split = meta["split"]
    video_id = Path(file_rel).stem
    video_path = dataset_root / file_rel
    
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    
    # 输出目录：output_root/<split>/<video_id>
    out_dir = output_root / split / video_id
    ensure_dir(out_dir)
    
    visual_npz = out_dir / "visual_embeddings.npz"
    audio_npz = out_dir / "audio_embeddings.npz"
    
    if skip_existing and visual_npz.exists() and audio_npz.exists():
        return {
            "video": video_id,
            "split": split,
            "label": "fake" if meta["n_fakes"] > 0 else "real",
            "frames_visual": -1,
            "frames_audio": -1,
        }
    
    # 1) 提取底脸
    bottom_dir = out_dir / "bottom_faces"
    ensure_dir(bottom_dir)
    face_extractor.process_video_frames(
        str(video_path),
        str(out_dir),
        sample_rate=1,
        video_list_df=None,
    )
    
    # 2) Visual embeddings
    visual_embeddings, paths = face_encoder.encode_directory(
        str(bottom_dir),
        pattern="frame_*.png",
        is_bottom_face=True,
    )
    
    # 3) 生成帧级标签（与 visual_embeddings 对齐）
    fps, total_frames = get_video_fps_and_frames(video_path)
    fake_periods = meta.get("fake_periods", [])
    frame_labels_full = generate_frame_labels(fake_periods, fps, total_frames)
    
    # 对齐：如果提取的 visual_embeddings 帧数少于 total_frames（可能有跳帧/检测失败）
    # 我们按照提取出的帧路径（frame_xxxx.png）来索引对应标签
    extracted_frame_labels = []
    for p in paths:
        # frame_0001.png -> frame_id=1
        fname = Path(p).stem  # "frame_0001"
        frame_id = int(fname.split("_")[-1])
        if frame_id < len(frame_labels_full):
            extracted_frame_labels.append(frame_labels_full[frame_id])
        else:
            extracted_frame_labels.append(0)  # fallback
    
    extracted_frame_labels = np.array(extracted_frame_labels, dtype=np.int32)
    
    # 保存 visual embeddings + frame_labels
    np.savez(
        visual_npz,
        embeddings=visual_embeddings,
        paths=paths,
        frame_labels=extracted_frame_labels
    )
    
    # 4) 音频提取 + embeddings
    audio_wav = out_dir / "audio.wav"
    extract_audio_from_video(str(video_path), str(audio_wav), sr=16000)
    motion_latent = speech_encoder.process_audio_file(str(audio_wav))
    np.savez(audio_npz, embeddings=motion_latent.cpu().detach().numpy())
    
    # 5) 相似度 + 统计
    stats_npz = out_dir / "similarity_stats.npz"
    plot_png = out_dir / "similarity_curve.png"
    
    audio_emb, visual_emb = load_embeddings(str(audio_npz), str(visual_npz))
    audio_aligned, visual_aligned, _ = align_embeddings(audio_emb, visual_emb, strategy="min_length")
    similarities = compute_similarity(audio_aligned, visual_aligned, metric="cosine")
    stats = extract_statistics(similarities)
    np.savez(stats_npz, similarities=similarities, **stats)
    plot_similarity_curve(
        similarities,
        output_path=str(plot_png),
        title=f"{video_id} - Audio-Visual Similarity (frames={len(similarities)})"
    )
    
    summary = {
        "video": video_id,
        "split": split,
        "label": "fake" if meta["n_fakes"] > 0 else "real",
        "n_fakes": meta["n_fakes"],
        "frames_visual": visual_embeddings.shape[0],
        "frames_audio": motion_latent.shape[0],
        "fake_frames": int(extracted_frame_labels.sum()),
    }
    summary.update(stats)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Prepare LAV-DF features with frame-level labels")
    parser.add_argument("--dataset_root", type=str, required=True, help="LAV-DF root (contains train/, dev/, test/)")
    parser.add_argument("--metadata", type=str, required=True, help="Path to metadata.min.json")
    parser.add_argument("--output_root", type=str, required=True, help="Output root for features")
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU if available")
    parser.add_argument("--splits", type=str, default="train,dev,test", help="Splits to process (comma-separated)")
    parser.add_argument("--start_idx", type=int, default=0, help="Start index (0-based) for sharding within filtered metadata")
    parser.add_argument("--num_videos", type=int, default=None, help="Number of videos to process from start_idx (sharding)")
    parser.add_argument("--max_videos", type=int, default=None, help="Process at most N videos per split")
    parser.add_argument("--skip_existing", action="store_true", help="Skip videos whose outputs already exist")
    args = parser.parse_args()
    
    dataset_root = Path(args.dataset_root)
    metadata_path = Path(args.metadata)
    output_root = Path(args.output_root)
    ensure_dir(output_root)
    
    device = "cuda" if args.use_gpu and os.environ.get("CUDA_VISIBLE_DEVICES", "") != "" else "cpu"
    print(f"[INFO] Using device={device}")
    
    # 加载元数据
    print(f"[INFO] Loading metadata from {metadata_path}")
    all_metadata = load_lavdf_metadata(metadata_path)
    print(f"[INFO] Total videos in metadata: {len(all_metadata)}")
    
    # 筛选 splits
    splits_to_process = args.splits.split(",")
    filtered_metadata = [m for m in all_metadata if m["split"] in splits_to_process]
    print(f"[INFO] Videos to process (splits={splits_to_process}): {len(filtered_metadata)}")

    # 分片：start_idx + num_videos
    if args.start_idx or args.num_videos is not None:
        end_idx = len(filtered_metadata) if args.num_videos is None else args.start_idx + args.num_videos
        filtered_metadata = filtered_metadata[args.start_idx:end_idx]
        print(f"[INFO] Shard applied: start_idx={args.start_idx}, num_videos={args.num_videos}, remaining={len(filtered_metadata)}")
    
    if args.max_videos:
        filtered_metadata = filtered_metadata[:args.max_videos]
        print(f"[INFO] Limited to {args.max_videos} videos")
    
    # 初始化模块
    face_extractor = BottomFaceExtractor()
    ctx_id = 0 if device == "cuda" else -1
    face_encoder = InsightFaceBottomEncoder(model_name="buffalo_l", ctx_id=ctx_id, det_size=(256, 256))
    speech_encoder = SpeechMotionEncoder(device=device)
    
    # 处理视频
    rows = []
    for idx, meta in enumerate(filtered_metadata, 1):
        video_id = Path(meta["file"]).stem
        split = meta["split"]
        print(f"\n[INFO] ({idx}/{len(filtered_metadata)}) Processing {meta['file']} [split={split}]")
        try:
            summary = process_lavdf_video(
                meta, dataset_root, output_root,
                face_extractor, face_encoder, speech_encoder,
                skip_existing=args.skip_existing
            )
            rows.append(summary)
        except Exception as e:
            print(f"[WARN] Failed {meta['file']}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 保存摘要
    import csv
    csv_path = output_root / "lavdf_summary.csv"
    if rows:
        fieldnames = list(rows[0].keys())
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
    
    print(f"\n[INFO] Done. Summary saved to {csv_path}")


if __name__ == "__main__":
    main()


