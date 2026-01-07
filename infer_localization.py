#!/usr/bin/env python3
"""
Inference script for frame-level deepfake localization
Load model and predict on individual videos
"""
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import numpy as np
from tqdm import tqdm

from model_localization import FrameLocalizationModel
from visualize_localization import visualize_localization, extract_fake_segments


def load_model(checkpoint_path: str, device: torch.device) -> FrameLocalizationModel:
    """Load model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model_args = checkpoint.get('args', {})
    state = checkpoint.get('model_state_dict', {})

    # Infer feature dims from checkpoint (supports DDP prefix "module.")
    v_key = 'v_proj.weight' if 'v_proj.weight' in state else 'module.v_proj.weight'
    a_key = 'a_proj.weight' if 'a_proj.weight' in state else 'module.a_proj.weight'
    if v_key not in state or a_key not in state:
        raise KeyError("Cannot infer v_dim/a_dim from checkpoint (missing v_proj.weight/a_proj.weight).")
    inferred_v_dim = int(state[v_key].shape[1])
    inferred_a_dim = int(state[a_key].shape[1])
    print(f"[INFO] Inferred dims from checkpoint: v_dim={inferred_v_dim}, a_dim={inferred_a_dim}")
    
    model = FrameLocalizationModel(
        v_dim=inferred_v_dim,
        a_dim=inferred_a_dim,
        d_model=model_args.get('d_model', 512),
        nhead=model_args.get('nhead', 8),
        num_layers=model_args.get('num_layers', 4),
        dropout=model_args.get('dropout', 0.1),
        use_cross_attn=not model_args.get('no_cross_attn', False),
        use_video_head=not model_args.get('no_video_head', False)
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model


def interpolate_audio(audio_emb: np.ndarray, target_len: int) -> np.ndarray:
    """Interpolate audio embeddings to match visual length"""
    T_a, D = audio_emb.shape
    
    indices = np.linspace(0, T_a - 1, target_len)
    interpolated = np.zeros((target_len, D), dtype=audio_emb.dtype)
    
    for i, idx in enumerate(indices):
        idx_low = int(np.floor(idx))
        idx_high = min(int(np.ceil(idx)), T_a - 1)
        weight = idx - idx_low
        
        interpolated[i] = (1 - weight) * audio_emb[idx_low] + weight * audio_emb[idx_high]
    
    return interpolated


@torch.no_grad()
def infer_video(
    model: FrameLocalizationModel,
    visual_emb: np.ndarray,
    audio_emb: np.ndarray,
    device: torch.device,
    max_frames: int = 512
) -> Tuple[np.ndarray, float]:
    """
    Run inference on a single video
    
    Args:
        model: Trained model
        visual_emb: [T, 512] visual embeddings
        audio_emb: [T_a, 1024] audio embeddings
        device: Device
        max_frames: Maximum frames (split if longer)
    
    Returns:
        frame_probs: [T] frame-level fake probabilities
        video_prob: Video-level fake probability
    """
    T_v = len(visual_emb)
    T_a = len(audio_emb)
    
    # Align audio to visual
    if T_a != T_v:
        audio_emb = interpolate_audio(audio_emb, T_v)
    
    # Split into chunks if too long
    if T_v <= max_frames:
        # Single forward pass
        visual_tensor = torch.from_numpy(visual_emb).unsqueeze(0).float().to(device)
        audio_tensor = torch.from_numpy(audio_emb).unsqueeze(0).float().to(device)
        
        frame_logits, video_logit = model(visual_tensor, audio_tensor)
        
        frame_probs = torch.sigmoid(frame_logits.squeeze()).cpu().numpy()
        video_prob = torch.sigmoid(video_logit.squeeze()).item() if video_logit is not None else frame_probs.max()
    
    else:
        # Sliding window with overlap
        stride = max_frames // 2
        frame_probs = np.zeros(T_v)
        counts = np.zeros(T_v)
        
        for start in range(0, T_v, stride):
            end = min(start + max_frames, T_v)
            
            visual_chunk = visual_emb[start:end]
            audio_chunk = audio_emb[start:end]
            
            visual_tensor = torch.from_numpy(visual_chunk).unsqueeze(0).float().to(device)
            audio_tensor = torch.from_numpy(audio_chunk).unsqueeze(0).float().to(device)
            
            frame_logits, _ = model(visual_tensor, audio_tensor)
            chunk_probs = torch.sigmoid(frame_logits.squeeze()).cpu().numpy()
            
            frame_probs[start:end] += chunk_probs
            counts[start:end] += 1
        
        # Average overlapping predictions
        frame_probs /= np.maximum(counts, 1)
        video_prob = frame_probs.max()
    
    return frame_probs, video_prob


def infer_directory(
    model: FrameLocalizationModel,
    features_dir: Path,
    device: torch.device,
    output_dir: Path,
    threshold: float = 0.5,
    visualize: bool = True,
    fps: float = 25.0
):
    """
    Run inference on all videos in a directory
    
    Args:
        model: Trained model
        features_dir: Directory with video features
        device: Device
        output_dir: Output directory for results
        threshold: Detection threshold
        visualize: Generate visualizations
        fps: Video frame rate
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all video directories
    video_dirs = [d for d in features_dir.iterdir() if d.is_dir()]
    
    print(f"[INFO] Found {len(video_dirs)} videos")
    
    results = []
    
    for video_dir in tqdm(video_dirs, desc="Inferring"):
        video_id = video_dir.name
        
        visual_file = video_dir / "visual_embeddings.npz"
        audio_file = video_dir / "audio_embeddings.npz"
        
        if not (visual_file.exists() and audio_file.exists()):
            print(f"[WARN] Missing files for {video_id}, skipping")
            continue
        
        # Load features
        try:
            visual_data = np.load(visual_file)
            audio_data = np.load(audio_file)
            
            visual_emb = visual_data['embeddings']
            audio_emb = audio_data['embeddings']
            
            # Check for ground truth labels
            frame_labels = visual_data.get('frame_labels')
            
        except Exception as e:
            print(f"[ERROR] Failed to load {video_id}: {e}")
            continue
        
        # Run inference
        frame_probs, video_prob = infer_video(model, visual_emb, audio_emb, device)
        
        # Extract segments
        segments = extract_fake_segments(frame_probs, threshold, min_duration=5)
        
        # Video-level prediction
        video_pred = 1 if video_prob > threshold else 0
        
        result = {
            'video_id': video_id,
            'num_frames': len(frame_probs),
            'video_prob': float(video_prob),
            'video_pred': int(video_pred),
            'num_fake_frames_pred': int((frame_probs > threshold).sum()),
            'fake_ratio_pred': float((frame_probs > threshold).mean()),
            'segments': [(int(s), int(e), float(c)) for s, e, c in segments]
        }
        
        # Add ground truth if available
        if frame_labels is not None:
            result['num_fake_frames_gt'] = int(frame_labels.sum())
            result['fake_ratio_gt'] = float(frame_labels.mean())
        
        results.append(result)
        
        # Visualize
        if visualize and frame_labels is not None:
            vis_path = output_dir / f"{video_id}.png"
            visualize_localization(
                frame_probs=frame_probs,
                frame_labels=frame_labels,
                fps=fps,
                threshold=threshold,
                save_path=str(vis_path),
                title=f"Localization: {video_id}"
            )
    
    # Save results
    import json
    results_file = output_dir / 'inference_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n[INFO] Inference complete!")
    print(f"[INFO] Results saved to {results_file}")
    
    # Print summary
    video_preds = [r['video_pred'] for r in results]
    num_fake_pred = sum(video_preds)
    print(f"\n[SUMMARY]")
    print(f"  Total videos: {len(results)}")
    print(f"  Predicted fake: {num_fake_pred} ({num_fake_pred/len(results)*100:.1f}%)")
    print(f"  Average fake ratio: {np.mean([r['fake_ratio_pred'] for r in results]):.3f}")


def main():
    parser = argparse.ArgumentParser(description='Inference for localization')
    
    # Model
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    
    # Input
    parser.add_argument('--features_dir', type=str, required=True,
                        help='Directory containing video features')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='./inference_results',
                        help='Output directory')
    
    # Options
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Detection threshold')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualizations')
    parser.add_argument('--fps', type=float, default=25.0,
                        help='Video frame rate')
    parser.add_argument('--max_frames', type=int, default=512,
                        help='Max frames per forward pass')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using device: {device}")
    
    # Load model
    print(f"[INFO] Loading model from {args.checkpoint}")
    model = load_model(args.checkpoint, device)
    print(f"[INFO] Model loaded successfully")
    
    # Run inference
    features_dir = Path(args.features_dir)
    output_dir = Path(args.output_dir)
    
    infer_directory(
        model=model,
        features_dir=features_dir,
        device=device,
        output_dir=output_dir,
        threshold=args.threshold,
        visualize=args.visualize,
        fps=args.fps
    )


if __name__ == '__main__':
    main()

