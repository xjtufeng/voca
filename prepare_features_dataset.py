"""
Dataset preprocessing pipeline:
For each video, extract bottom-face embeddings, audio embeddings, and compute
audio-visual similarity statistics. Saves per-video artifacts and a summary CSV.

Expected directory structure:
    dataset_root/
        real/  *.mp4
        fake/  *.mp4

Outputs (under output_root/<split>/<video_id>/):
    bottom_faces/               extracted bottom-face frames
    visual_embeddings.npz       {embeddings: (T_v, 512), paths: [...]}
    audio_embeddings.npz        {embeddings: (T_a, 512)}
    similarity_stats.npz        stats + similarities
    similarity_curve.png        visualization

A summary CSV (output_root/summary.csv) is also generated with one row per video.
"""

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

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


def _slugify_path(path: Path) -> str:
    """Convert a relative path to a safe id by replacing separators with underscores."""
    return "_".join(path.with_suffix("").parts)


def find_videos(dataset_root: Path, video_list: Path | None = None) -> List[Tuple[Path, str, str]]:
    """
    - 若提供 video_list（每行一个 mp4 路径），按列表读取；标签根据路径包含 real/fake 判断。
    - 否则递归搜索 dataset_root/{real,fake}/**/*.mp4。
    返回 (视频绝对路径, 标签, 唯一ID)。ID 由相对路径去后缀并用下划线连接，避免重名。
    """
    videos: List[Tuple[Path, str, str]] = []
    if video_list:
        with video_list.open("r", encoding="utf-8") as f:
            for line in f:
                p = Path(line.strip())
                if not p.suffix.lower() == ".mp4":
                    continue
                label = "real" if "real" in p.parts or "Real" in p.parts else "fake"
                vid = _slugify_path(Path(p.stem))
                videos.append((p, label, vid))
        return videos

    for label in ["real", "fake"]:
        label_dir = dataset_root / label
        if not label_dir.exists():
            continue
        for mp4 in sorted(label_dir.rglob("*.mp4")):
            rel = mp4.relative_to(label_dir)  # 不含 real/fake
            vid = _slugify_path(rel)
            videos.append((mp4, label, vid))
    return videos


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def process_video(
    video_path: Path,
    label: str,
    vid: str,
    output_root: Path,
    face_extractor: BottomFaceExtractor,
    face_encoder: InsightFaceBottomEncoder,
    speech_encoder: SpeechMotionEncoder,
    skip_existing: bool = False,
) -> Dict:
    """
    Process a single video: bottom-face extraction, visual/audio embeddings,
    similarity stats. Returns a dict of summary stats.
    """
    out_dir = output_root / label / vid
    ensure_dir(out_dir)

    visual_npz = out_dir / "visual_embeddings.npz"
    audio_npz = out_dir / "audio_embeddings.npz"

    if skip_existing and visual_npz.exists() and audio_npz.exists():
        # 已处理，直接汇总 stats（简单返回占位）
        return {
            "video": vid,
            "label": label,
            "frames_visual": -1,
            "frames_audio": -1,
        }

    # 1) Extract bottom faces
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
    np.savez(visual_npz, embeddings=visual_embeddings, paths=paths)

    # 3) Audio extraction + embeddings
    audio_wav = out_dir / "audio.wav"
    extract_audio_from_video(str(video_path), str(audio_wav), sr=16000)
    audio_npz = out_dir / "audio_embeddings.npz"
    motion_latent = speech_encoder.process_audio_file(str(audio_wav))
    np.savez(audio_npz, embeddings=motion_latent.cpu().detach().numpy())

    # 4) Similarity + stats
    stats_npz = out_dir / "similarity_stats.npz"
    plot_png = out_dir / "similarity_curve.png"

    audio_emb, visual_emb = load_embeddings(str(audio_npz), str(visual_npz))
    audio_aligned, visual_aligned, _ = align_embeddings(audio_emb, visual_emb, strategy="min_length")
    similarities = compute_similarity(audio_aligned, visual_aligned, metric="cosine")
    stats = extract_statistics(similarities)
    np.savez(stats_npz, similarities=similarities, **stats)
    plot_similarity_curve(similarities, output_path=str(plot_png),
                          title=f"{vid} - Audio-Visual Similarity (frames={len(similarities)})")

    summary = {
        "video": vid,
        "label": label,
        "frames_visual": visual_embeddings.shape[0],
        "frames_audio": motion_latent.shape[0],
    }
    summary.update(stats)
    return summary


def save_summary_csv(rows: List[Dict], csv_path: Path):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def main():
    parser = argparse.ArgumentParser(description="Prepare AV features dataset")
    parser.add_argument("--dataset_root", type=str, required=True, help="Root dir with real/ and fake/ mp4")
    parser.add_argument("--output_root", type=str, required=True, help="Output root for features")
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU if available")
    parser.add_argument("--video_list", type=str, default=None, help="Optional: file with mp4 paths (one per line)")
    parser.add_argument("--max_videos", type=int, default=None, help="Process at most N videos")
    parser.add_argument("--start_idx", type=int, default=0, help="Start index when slicing videos")
    parser.add_argument("--skip_existing", action="store_true", help="Skip videos whose outputs already exist")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    output_root = Path(args.output_root)
    ensure_dir(output_root)

    device = "cuda" if args.use_gpu and os.environ.get("CUDA_VISIBLE_DEVICES", "") != "" else "cpu"
    print(f"[INFO] Using device={device}")

    video_list_path = Path(args.video_list) if args.video_list else None
    videos = find_videos(dataset_root, video_list=video_list_path)
    if not videos:
        print(f"[ERROR] No videos found under {dataset_root}/(real|fake)/*.mp4")
        sys.exit(1)
    end_idx = None if args.max_videos is None else args.start_idx + args.max_videos
    videos = videos[args.start_idx:end_idx]

    print(f"[INFO] Found {len(videos)} videos")

    # Initialize modules
    face_extractor = BottomFaceExtractor()
    ctx_id = 0 if device == "cuda" else -1
    face_encoder = InsightFaceBottomEncoder(model_name="buffalo_l", ctx_id=ctx_id, det_size=(256, 256))
    speech_encoder = SpeechMotionEncoder(device=device)

    rows = []
    for idx, (vpath, label, vid) in enumerate(videos, 1):
        print(f"\n[INFO] ({idx}/{len(videos)}) Processing {vpath.name} [{label}]")
        try:
            summary = process_video(
                vpath, label, vid, output_root,
                face_extractor, face_encoder, speech_encoder,
                skip_existing=args.skip_existing,
            )
            rows.append(summary)
        except Exception as e:
            print(f"[WARN] Failed {vpath}: {e}")
            continue

    csv_path = output_root / "summary.csv"
    save_summary_csv(rows, csv_path)
    print(f"\n[INFO] Done. Summary saved to {csv_path}")


if __name__ == "__main__":
    main()

