#!/usr/bin/env python3
"""
Download LAV-DF dataset from Hugging Face
Requires: huggingface_hub and valid HF token with dataset access
"""
import os
from pathlib import Path
from huggingface_hub import snapshot_download, login

def download_lavdf(output_dir="LAV-DF", token=None):
    """
    Download LAV-DF dataset
    
    Args:
        output_dir: Local directory to save dataset
        token: Hugging Face token (optional if already logged in)
    """
    print("="*60)
    print("LAV-DF Dataset Downloader")
    print("="*60)
    
    # Login if token provided
    if token:
        print("[INFO] Logging in with provided token...")
        login(token=token)
    else:
        print("[INFO] Using cached credentials (run 'huggingface-cli login' first)")
    
    output_path = Path(output_dir)
    print(f"[INFO] Download destination: {output_path.absolute()}")
    print(f"[INFO] Starting download (this may take a while)...")
    
    try:
        snapshot_download(
            repo_id="ControlNet/LAV-DF",
            repo_type="dataset",
            local_dir=str(output_path),
            resume_download=True,
            local_dir_use_symlinks=False  # Direct files, not symlinks
        )
        print("\n" + "="*60)
        print("[SUCCESS] Download complete!")
        print(f"[INFO] Dataset saved to: {output_path.absolute()}")
        print("="*60)
        
        # Show directory structure
        print("\n[INFO] Dataset structure:")
        if output_path.exists():
            for item in sorted(output_path.iterdir())[:10]:  # Show first 10 items
                size = ""
                if item.is_file():
                    size = f" ({item.stat().st_size / 1024**2:.1f} MB)"
                print(f"  - {item.name}{size}")
            if len(list(output_path.iterdir())) > 10:
                print(f"  ... and {len(list(output_path.iterdir())) - 10} more items")
        
    except Exception as e:
        print(f"\n[ERROR] Download failed: {e}")
        print("\nPossible solutions:")
        print("1. Make sure you have accepted the dataset terms at:")
        print("   https://huggingface.co/datasets/ControlNet/LAV-DF")
        print("2. Get your access token from:")
        print("   https://huggingface.co/settings/tokens")
        print("3. Login with: huggingface-cli login")
        print("   or provide token: python download_lavdf.py --token YOUR_TOKEN")
        raise


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Download LAV-DF dataset")
    parser.add_argument("--output_dir", type=str, default="LAV-DF",
                        help="Output directory for dataset")
    parser.add_argument("--token", type=str, default=None,
                        help="Hugging Face access token (optional)")
    args = parser.parse_args()
    
    download_lavdf(args.output_dir, args.token)


if __name__ == "__main__":
    main()



