"""
Create a real/fake-compatible symlink dataset from FakeAVCeleb_v1.2 layout.

Source layout (after unzip):
  <src_root>/
    RealVideo-RealAudio/        # real
    RealVideo-FakeAudio/        # fake
    FakeVideo-RealAudio/        # fake
    FakeVideo-FakeAudio/        # fake

Target layout (stage_root):
  stage_root/
    real/<preserved_subdirs>/file.mp4
    fake/<subdir>/<preserved_subdirs>/file.mp4

This keeps original subdirectory structure to avoid name collisions.
"""

import argparse
import os
import shutil
from pathlib import Path


def symlink_preserve_tree(src_root: Path, stage_root: Path, label: str, subdir: str | None = None):
    """
    Symlink all mp4 under src_root/(subdir)/... into stage_root/label/(subdir)/...
    preserving the relative tree to avoid filename collisions.
    """
    base = src_root if subdir is None else src_root / subdir
    if not base.exists():
        print(f"[WARN] skip missing: {base}")
        return

    for mp4 in base.rglob("*.mp4"):
        rel = mp4.relative_to(base)
        dest_dir = stage_root / label
        if subdir:
            dest_dir = dest_dir / subdir
        dest = dest_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            dest.symlink_to(mp4.resolve())
        except FileExistsError:
            # If exists, keep the first one to avoid collisions
            pass


def main():
    ap = argparse.ArgumentParser(description="Create FakeAVCeleb stage (real/fake) via symlinks")
    ap.add_argument("--src_root", required=True, help="Path to FakeAVCeleb_v1.2 directory (contains RealVideo-RealAudio etc.)")
    ap.add_argument("--stage_root", required=True, help="Output directory for staged real/fake structure")
    ap.add_argument("--force", action="store_true", help="Remove existing stage_root before creating")
    args = ap.parse_args()

    src = Path(args.src_root).expanduser().resolve()
    stage = Path(args.stage_root).expanduser().resolve()

    if stage.exists() and args.force:
        shutil.rmtree(stage)
    stage.mkdir(parents=True, exist_ok=True)

    # Real
    symlink_preserve_tree(src, stage, label="real", subdir="RealVideo-RealAudio")
    # Fake
    for sub in ["RealVideo-FakeAudio", "FakeVideo-RealAudio", "FakeVideo-FakeAudio"]:
        symlink_preserve_tree(src, stage, label="fake", subdir=sub)

    print(f"[INFO] Done. Stage root: {stage}")


if __name__ == "__main__":
    main()


