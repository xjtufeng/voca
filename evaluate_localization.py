#!/usr/bin/env python3
"""
Evaluation script for frame-level deepfake localization
Computes comprehensive metrics on test set
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    precision_recall_curve, average_precision_score
)

from dataset_localization import LAVDFLocalizationDataset, collate_variable_length
from torch.utils.data import DataLoader
from model_localization import FrameLocalizationModel


def compute_iou(pred_segments: List[tuple], gt_segments: List[tuple], total_frames: int) -> float:
    """
    Compute IoU between predicted and ground truth segments
    
    Args:
        pred_segments: List of (start, end) tuples
        gt_segments: List of (start, end) tuples
        total_frames: Total number of frames
    
    Returns:
        IoU score
    """
    if len(gt_segments) == 0:
        return 1.0 if len(pred_segments) == 0 else 0.0
    
    # Create binary masks
    pred_mask = np.zeros(total_frames, dtype=bool)
    gt_mask = np.zeros(total_frames, dtype=bool)
    
    for start, end in pred_segments:
        pred_mask[start:end+1] = True
    
    for start, end in gt_segments:
        gt_mask[start:end+1] = True
    
    # Compute IoU
    intersection = (pred_mask & gt_mask).sum()
    union = (pred_mask | gt_mask).sum()
    
    if union == 0:
        return 1.0
    
    return intersection / union


def extract_segments(frame_labels: np.ndarray, min_length: int = 1) -> List[tuple]:
    """
    Extract continuous segments from binary frame labels
    
    Args:
        frame_labels: Binary array [T]
        min_length: Minimum segment length
    
    Returns:
        List of (start, end) tuples
    """
    segments = []
    start = None
    
    for i, label in enumerate(frame_labels):
        if label == 1 and start is None:
            start = i
        elif label == 0 and start is not None:
            if i - start >= min_length:
                segments.append((start, i - 1))
            start = None
    
    if start is not None and len(frame_labels) - start >= min_length:
        segments.append((start, len(frame_labels) - 1))
    
    return segments


@torch.no_grad()
def evaluate_dataset(
    model: FrameLocalizationModel,
    loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
    min_segment_length: int = 5
) -> Dict:
    """
    Comprehensive evaluation on dataset
    
    Returns:
        Dictionary with metrics and per-video results
    """
    model.eval()
    
    # Collect all predictions
    all_frame_probs = []
    all_frame_labels = []
    all_video_probs = []
    all_video_labels = []
    
    per_video_results = []
    all_ious = []
    
    for batch in tqdm(loader, desc="Evaluating"):
        visual = batch['visual'].to(device)
        audio = batch['audio'].to(device)
        frame_labels = batch['frame_labels'].to(device)
        video_labels = batch['video_labels'].to(device)
        mask = batch['mask'].to(device)
        video_ids = batch['video_ids']
        
        # Forward pass
        frame_logits, video_logit = model(visual, audio, mask)
        frame_probs = torch.sigmoid(frame_logits.squeeze(-1))
        
        # Process each video in batch
        for i in range(len(video_ids)):
            valid_mask = ~mask[i]
            valid_len = valid_mask.sum().item()
            
            # Get valid predictions and labels
            vid_frame_probs = frame_probs[i, :valid_len].cpu().numpy()
            vid_frame_labels = frame_labels[i, :valid_len].cpu().numpy()
            vid_video_label = video_labels[i].item()
            
            # Collect for global metrics
            all_frame_probs.append(vid_frame_probs)
            all_frame_labels.append(vid_frame_labels)
            all_video_labels.append(vid_video_label)
            
            # Video-level prediction (max pooling)
            vid_video_prob = vid_frame_probs.max()
            all_video_probs.append(vid_video_prob)
            
            # Binary predictions
            vid_frame_preds = (vid_frame_probs > threshold).astype(int)
            
            # Extract segments
            pred_segments = extract_segments(vid_frame_preds, min_segment_length)
            gt_segments = extract_segments(vid_frame_labels, min_segment_length)
            
            # Compute IoU
            iou = compute_iou(pred_segments, gt_segments, valid_len)
            all_ious.append(iou)
            
            # Per-video metrics
            vid_f1 = f1_score(vid_frame_labels, vid_frame_preds, zero_division=0)
            vid_precision = precision_score(vid_frame_labels, vid_frame_preds, zero_division=0)
            vid_recall = recall_score(vid_frame_labels, vid_frame_preds, zero_division=0)
            
            per_video_results.append({
                'video_id': video_ids[i],
                'num_frames': valid_len,
                'video_label': vid_video_label,
                'video_prob': float(vid_video_prob),
                'num_fake_frames': int(vid_frame_labels.sum()),
                'num_pred_fake_frames': int(vid_frame_preds.sum()),
                'frame_f1': float(vid_f1),
                'frame_precision': float(vid_precision),
                'frame_recall': float(vid_recall),
                'iou': float(iou),
                'num_pred_segments': len(pred_segments),
                'num_gt_segments': len(gt_segments)
            })
    
    # Concatenate all predictions
    all_frame_probs = np.concatenate(all_frame_probs)
    all_frame_labels = np.concatenate(all_frame_labels)
    all_video_probs = np.array(all_video_probs)
    all_video_labels = np.array(all_video_labels)
    
    # Global frame-level metrics
    frame_preds_binary = (all_frame_probs > threshold).astype(int)
    
    frame_auc = roc_auc_score(all_frame_labels, all_frame_probs)
    frame_ap = average_precision_score(all_frame_labels, all_frame_probs)
    frame_f1 = f1_score(all_frame_labels, frame_preds_binary)
    frame_precision = precision_score(all_frame_labels, frame_preds_binary, zero_division=0)
    frame_recall = recall_score(all_frame_labels, frame_preds_binary, zero_division=0)
    
    # Video-level metrics
    video_preds_binary = (all_video_probs > threshold).astype(int)
    video_auc = roc_auc_score(all_video_labels, all_video_probs)
    video_ap = average_precision_score(all_video_labels, all_video_probs)
    video_f1 = f1_score(all_video_labels, video_preds_binary)
    video_precision = precision_score(all_video_labels, video_preds_binary, zero_division=0)
    video_recall = recall_score(all_video_labels, video_preds_binary, zero_division=0)
    
    # IoU statistics
    mean_iou = np.mean(all_ious)
    median_iou = np.median(all_ious)
    
    # Summary metrics
    metrics = {
        'frame': {
            'auc': float(frame_auc),
            'ap': float(frame_ap),
            'f1': float(frame_f1),
            'precision': float(frame_precision),
            'recall': float(frame_recall)
        },
        'video': {
            'auc': float(video_auc),
            'ap': float(video_ap),
            'f1': float(video_f1),
            'precision': float(video_precision),
            'recall': float(video_recall)
        },
        'iou': {
            'mean': float(mean_iou),
            'median': float(median_iou),
            'std': float(np.std(all_ious))
        },
        'dataset': {
            'num_videos': len(per_video_results),
            'num_frames': len(all_frame_labels),
            'num_fake_frames': int(all_frame_labels.sum()),
            'fake_ratio': float(all_frame_labels.mean())
        }
    }
    
    return {
        'metrics': metrics,
        'per_video': per_video_results
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate localization model')
    
    # Data
    parser.add_argument('--features_root', type=str, required=True,
                        help='Path to extracted features')
    parser.add_argument('--split', type=str, default='test',
                        help='Split to evaluate')
    parser.add_argument('--max_frames', type=int, default=512,
                        help='Max frames per video')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers')
    
    # Model
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    
    # Evaluation
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Classification threshold')
    parser.add_argument('--min_segment_length', type=int, default=5,
                        help='Minimum segment length for IoU')
    
    # Output
    parser.add_argument('--output', type=str, default='./results/eval_results.json',
                        help='Output JSON file')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using device: {device}")
    
    # Load checkpoint
    print(f"[INFO] Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
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
    
    # Create model
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
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"[INFO] Loaded model from epoch {checkpoint['epoch']}")
    
    # Load dataset
    print(f"[INFO] Loading {args.split} split from {args.features_root}")
    
    dataset = LAVDFLocalizationDataset(
        features_root=args.features_root,
        split=args.split,
        max_frames=args.max_frames
    )
    
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_variable_length,
        pin_memory=True
    )
    
    print(f"[INFO] Loaded {len(dataset)} videos")
    
    # Evaluate
    print(f"\n[INFO] Evaluating...")
    results = evaluate_dataset(
        model, loader, device,
        threshold=args.threshold,
        min_segment_length=args.min_segment_length
    )
    
    # Print metrics
    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS")
    print(f"{'='*60}")
    
    metrics = results['metrics']
    
    print(f"\nFrame-Level Metrics:")
    print(f"  AUC:       {metrics['frame']['auc']:.4f}")
    print(f"  AP:        {metrics['frame']['ap']:.4f}")
    print(f"  F1:        {metrics['frame']['f1']:.4f}")
    print(f"  Precision: {metrics['frame']['precision']:.4f}")
    print(f"  Recall:    {metrics['frame']['recall']:.4f}")
    
    print(f"\nVideo-Level Metrics:")
    print(f"  AUC:       {metrics['video']['auc']:.4f}")
    print(f"  AP:        {metrics['video']['ap']:.4f}")
    print(f"  F1:        {metrics['video']['f1']:.4f}")
    print(f"  Precision: {metrics['video']['precision']:.4f}")
    print(f"  Recall:    {metrics['video']['recall']:.4f}")
    
    print(f"\nLocalization IoU:")
    print(f"  Mean:   {metrics['iou']['mean']:.4f}")
    print(f"  Median: {metrics['iou']['median']:.4f}")
    print(f"  Std:    {metrics['iou']['std']:.4f}")
    
    print(f"\nDataset Statistics:")
    print(f"  Videos:      {metrics['dataset']['num_videos']}")
    print(f"  Frames:      {metrics['dataset']['num_frames']}")
    print(f"  Fake Frames: {metrics['dataset']['num_fake_frames']} ({metrics['dataset']['fake_ratio']*100:.2f}%)")
    
    print(f"\n{'='*60}")
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n[INFO] Results saved to {output_path}")


if __name__ == '__main__':
    main()

