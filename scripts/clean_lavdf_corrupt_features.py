#!/usr/bin/env python3
"""
Clean (detect + quarantine/delete) corrupted LAV-DF feature files.

Why:
  - In DDP training, a single corrupted .npz (CRC error / incomplete write) can
    stall one rank and cause NCCL ALLREDUCE timeouts across all ranks.

What it does:
  - Recursively scans features_root/{split}/*/(visual_embeddings.npz|audio_embeddings.npz)
  - Forces reading arrays to trigger CRC checks (not just listing keys)
  - Optionally validates required keys and basic shape consistency
  - Writes a report and can quarantine or delete bad samples

Typical usage (HPC):
  python scripts/clean_lavdf_corrupt_features.py \
    --features_root /hpc2hdd/home/xfeng733/LAV-DF_feats \
    --splits train dev test \
    --action quarantine \
    --quarantine_root /hpc2hdd/home/xfeng733/LAV-DF_feats_quarantine
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


@dataclass
class BadSample:
    split: str
    video_id: str
    visual_file: str
    audio_file: str
    reason: str


def _safe_npz_read(npz_path: Path, required_keys: Iterable[str]) -> Tuple[bool, Optional[str]]:
    """
    Return (ok, reason). Forces reading arrays to trigger CRC checks.
    """
    try:
        with np.load(npz_path, allow_pickle=False) as data:
            for k in required_keys:
                if k not in data.files:
                    return False, f"missing_key:{k}"
                # Force read to trigger CRC/integrity checks
                _ = data[k]
        return True, None
    except Exception as e:
        return False, f"npz_load_error:{type(e).__name__}:{e}"


def _scan_split_dir(
    split_dir: Path,
    split_name: str,
    min_frames: int,
    check_shapes: bool,
) -> Tuple[List[BadSample], int]:
    """
    Returns (bad_samples, total_video_dirs_seen).
    """
    bad: List[BadSample] = []
    total = 0

    if not split_dir.exists():
        return bad, total

    # Deterministic ordering
    for video_dir in sorted(split_dir.iterdir(), key=lambda p: p.name):
        if not video_dir.is_dir():
            continue
        total += 1

        visual_file = video_dir / "visual_embeddings.npz"
        audio_file = video_dir / "audio_embeddings.npz"

        if not visual_file.exists() or not audio_file.exists():
            # Not necessarily an error (incomplete extraction); keep it in report as "bad"
            reason = []
            if not visual_file.exists():
                reason.append("missing_visual")
            if not audio_file.exists():
                reason.append("missing_audio")
            bad.append(
                BadSample(
                    split=split_name,
                    video_id=video_dir.name,
                    visual_file=str(visual_file),
                    audio_file=str(audio_file),
                    reason=";".join(reason),
                )
            )
            continue

        ok_v, reason_v = _safe_npz_read(visual_file, required_keys=("embeddings", "frame_labels"))
        if not ok_v:
            bad.append(
                BadSample(
                    split=split_name,
                    video_id=video_dir.name,
                    visual_file=str(visual_file),
                    audio_file=str(audio_file),
                    reason=f"visual:{reason_v}",
                )
            )
            continue

        ok_a, reason_a = _safe_npz_read(audio_file, required_keys=("embeddings",))
        if not ok_a:
            bad.append(
                BadSample(
                    split=split_name,
                    video_id=video_dir.name,
                    visual_file=str(visual_file),
                    audio_file=str(audio_file),
                    reason=f"audio:{reason_a}",
                )
            )
            continue

        # Optional deeper checks (loads arrays; can cost time but catches weird partial outputs)
        if check_shapes:
            try:
                with np.load(visual_file, allow_pickle=False) as vd, np.load(audio_file, allow_pickle=False) as ad:
                    v = vd["embeddings"]
                    fl = vd["frame_labels"]
                    a = ad["embeddings"]

                if v.ndim != 2:
                    raise ValueError(f"visual_embeddings.ndim={v.ndim} (expected 2)")
                if fl.ndim != 1:
                    raise ValueError(f"frame_labels.ndim={fl.ndim} (expected 1)")
                if len(v) != len(fl):
                    raise ValueError(f"len(visual)={len(v)} != len(frame_labels)={len(fl)}")
                if len(v) < min_frames:
                    raise ValueError(f"too_few_frames:{len(v)}<{min_frames}")
                if a.ndim != 2:
                    raise ValueError(f"audio_embeddings.ndim={a.ndim} (expected 2)")
                if len(a) < 1:
                    raise ValueError("audio_embeddings_empty")
            except Exception as e:
                bad.append(
                    BadSample(
                        split=split_name,
                        video_id=video_dir.name,
                        visual_file=str(visual_file),
                        audio_file=str(audio_file),
                        reason=f"shape_check:{type(e).__name__}:{e}",
                    )
                )

    return bad, total


def _quarantine_sample(sample: BadSample, features_root: Path, quarantine_root: Path) -> None:
    """
    Move the whole video directory to quarantine_root/{split}/{video_id}.
    """
    src_dir = Path(sample.visual_file).parent
    # Ensure src is within features_root
    try:
        rel = src_dir.relative_to(features_root)
    except Exception:
        rel = Path(sample.split) / sample.video_id
    dst_dir = quarantine_root / rel
    dst_dir.parent.mkdir(parents=True, exist_ok=True)

    # If destination exists, avoid overwriting silently
    if dst_dir.exists():
        # Add a suffix
        i = 1
        while (dst_dir.parent / f"{dst_dir.name}__dup{i}").exists():
            i += 1
        dst_dir = dst_dir.parent / f"{dst_dir.name}__dup{i}"

    shutil.move(str(src_dir), str(dst_dir))


def _delete_sample(sample: BadSample) -> None:
    src_dir = Path(sample.visual_file).parent
    if src_dir.exists():
        shutil.rmtree(src_dir, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Clean corrupted LAV-DF feature .npz files")
    parser.add_argument("--features_root", type=str, required=True, help="Path to features root (contains train/dev/test)")
    parser.add_argument("--splits", nargs="+", default=["train", "dev", "test"], help="Splits to scan")
    parser.add_argument("--min_frames", type=int, default=10, help="Min frames check (shape_check only)")
    parser.add_argument(
        "--check_shapes",
        action="store_true",
        help="Also validate array dims/lengths after CRC check (slower but safer)",
    )
    parser.add_argument(
        "--action",
        choices=["report", "quarantine", "delete"],
        default="report",
        help="What to do with bad samples",
    )
    parser.add_argument(
        "--quarantine_root",
        type=str,
        default="",
        help="Where to move bad samples when --action quarantine",
    )
    parser.add_argument(
        "--report_path",
        type=str,
        default="",
        help="Optional path to save JSON report (defaults to ./lavdf_bad_samples_<ts>.json)",
    )
    args = parser.parse_args()

    features_root = Path(args.features_root)
    if not features_root.exists():
        print(f"[ERROR] features_root does not exist: {features_root}")
        return 2

    quarantine_root = Path(args.quarantine_root) if args.quarantine_root else None
    if args.action == "quarantine" and quarantine_root is None:
        print("[ERROR] --action quarantine requires --quarantine_root")
        return 2

    all_bad: List[BadSample] = []
    totals: Dict[str, int] = {}

    print(f"[INFO] Scanning features_root: {features_root}")
    print(f"[INFO] splits={args.splits} action={args.action} check_shapes={args.check_shapes}")

    for split in args.splits:
        split_dir = features_root / split
        bad, total = _scan_split_dir(
            split_dir=split_dir,
            split_name=split,
            min_frames=int(args.min_frames),
            check_shapes=bool(args.check_shapes),
        )
        totals[split] = total
        all_bad.extend(bad)
        print(f"[INFO] split={split}: total_dirs={total}, bad={len(bad)}")

    # Write report
    import time

    ts = time.strftime("%Y%m%d_%H%M%S")
    report_path = Path(args.report_path) if args.report_path else Path(f"lavdf_bad_samples_{ts}.json")
    report = {
        "features_root": str(features_root),
        "splits": args.splits,
        "totals": totals,
        "bad_count": len(all_bad),
        "bad_samples": [asdict(x) for x in all_bad],
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[INFO] Report saved: {report_path} (bad={len(all_bad)})")

    # Take action
    if args.action == "report":
        return 0

    if len(all_bad) == 0:
        print("[INFO] No bad samples found; nothing to do.")
        return 0

    if args.action == "quarantine":
        assert quarantine_root is not None
        print(f"[INFO] Quarantining bad samples to: {quarantine_root}")
        moved = 0
        for s in all_bad:
            try:
                _quarantine_sample(s, features_root=features_root, quarantine_root=quarantine_root)
                moved += 1
            except Exception as e:
                print(f"[WARN] Failed to quarantine {s.split}/{s.video_id}: {e}")
        print(f"[INFO] Quarantined: {moved}/{len(all_bad)}")
        return 0

    if args.action == "delete":
        print("[INFO] Deleting bad samples (removing their directories)...")
        deleted = 0
        for s in all_bad:
            try:
                _delete_sample(s)
                deleted += 1
            except Exception as e:
                print(f"[WARN] Failed to delete {s.split}/{s.video_id}: {e}")
        print(f"[INFO] Deleted: {deleted}/{len(all_bad)}")
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


