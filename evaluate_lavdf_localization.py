#!/usr/bin/env python3
"""
Evaluation script for LAV-DF frame-level localization model
Evaluates a trained model on test set with comprehensive metrics
"""
import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset_localization import get_dataloaders
from model_localization import FrameLocalizationModel
from train_lavdf_localization import evaluate


def main():
    parser = argparse.ArgumentParser(description='Evaluate LAV-DF localization model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--features_root', type=str, required=True,
                        help='Path to extracted features')
    parser.add_argument('--split', type=str, default='test',
                        help='Dataset split to evaluate (default: test)')
    parser.add_argument('--batch_size', type=int, default=24,
                        help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of data loading workers')
    parser.add_argument('--max_frames', type=int, default=512,
                        help='Maximum frames per video')
    parser.add_argument('--stride', type=int, default=1,
                        help='Frame sampling stride')
    
    # Model architecture (should match training config)
    parser.add_argument('--d_model', type=int, default=512,
                        help='Model dimension')
    parser.add_argument('--nhead', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=4,
                        help='Number of transformer layers')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    parser.add_argument('--no_cross_attn', action='store_true',
                        help='Disable cross-modal attention')
    parser.add_argument('--no_video_head', action='store_true',
                        help='Disable video-level prediction head')
    
    # Loss weights (for evaluation loss computation)
    parser.add_argument('--pos_weight', type=float, default=-1.0,
                        help='Positive class weight for BCE loss')
    parser.add_argument('--use_focal', action='store_true',
                        help='Use focal loss instead of BCE')
    parser.add_argument('--focal_alpha', type=float, default=0.25,
                        help='Focal loss alpha')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='Focal loss gamma')
    parser.add_argument('--video_loss_weight', type=float, default=0.3,
                        help='Weight for video-level loss')
    parser.add_argument('--smooth_loss_weight', type=float, default=0.1,
                        help='Weight for temporal smoothness loss')
    parser.add_argument('--similarity_loss_weight', type=float, default=0.2,
                        help='Weight for audio-visual similarity loss')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='results/localization_test',
                        help='Directory to save evaluation results')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[INFO] Using device: {device}')
    
    # Load checkpoint
    print(f'[INFO] Loading checkpoint: {args.checkpoint}')
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    epoch = ckpt.get('epoch', -1)
    print(f'✅ Loaded model from epoch {epoch}')
    
    if "metrics" in ckpt and ckpt["metrics"]:
        print(f'   Dev/Val metrics from training:')
        for k, v in ckpt["metrics"].items():
            if isinstance(v, (int, float)):
                print(f'     {k}: {v:.4f}')
    print()
    
    # Load data
    print(f'[INFO] Loading {args.split} data from {args.features_root}')
    dataloaders = get_dataloaders(
        features_root=args.features_root,
        splits=[args.split],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_frames=args.max_frames,
        stride=args.stride,
        distributed=False
    )
    
    if args.split not in dataloaders:
        print(f'[ERROR] Split {args.split} not found in dataloaders')
        return
    
    test_loader = dataloaders[args.split]
    print(f'[INFO] {args.split.capitalize()} set: {len(test_loader.dataset)} samples')
    print()
    
    # Infer feature dimensions from data
    sample0 = test_loader.dataset[0]
    inferred_v_dim = int(sample0['visual'].shape[-1])
    inferred_a_dim = int(sample0['audio'].shape[-1])
    print(f'[INFO] Inferred feature dims: v_dim={inferred_v_dim}, a_dim={inferred_a_dim}')
    print()
    
    # Load model
    print('[INFO] Building model...')
    model = FrameLocalizationModel(
        v_dim=inferred_v_dim,
        a_dim=inferred_a_dim,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dropout=args.dropout,
        use_cross_attn=not args.no_cross_attn,
        use_video_head=not args.no_video_head
    )
    
    # Load weights
    model.load_state_dict(ckpt['model'])
    model = model.to(device)
    print(f'✅ Model loaded on {device}')
    print()
    
    # Evaluate
    print(f'[INFO] Evaluating on {args.split} set...')
    print()
    
    test_metrics = evaluate(model, test_loader, device, args, rank=0)
    
    # Print results
    print()
    print('=' * 70)
    print(f'{args.split.upper()} SET RESULTS (Epoch {epoch})')
    print('=' * 70)
    print(f"Loss:        {test_metrics.get('loss', 0):.4f}")
    print()
    
    print("Frame-level Metrics:")
    print(f"  AUC:       {test_metrics.get('frame_auc', 0):.4f}")
    print(f"  AP:        {test_metrics.get('frame_ap', 0):.4f}")
    print(f"  F1:        {test_metrics.get('frame_f1', 0):.4f}")
    print(f"  Precision: {test_metrics.get('frame_precision', 0):.4f}")
    print(f"  Recall:    {test_metrics.get('frame_recall', 0):.4f}")
    
    if 'video_auc' in test_metrics:
        print()
        print("Video-level Metrics:")
        print(f"  AUC:       {test_metrics['video_auc']:.4f}")
        print(f"  AP:        {test_metrics.get('video_ap', 0):.4f}")
        print(f"  F1:        {test_metrics.get('video_f1', 0):.4f}")
    
    if 'mAP' in test_metrics:
        print()
        print("Temporal Localization (Segment-level):")
        print(f"  mAP:       {test_metrics['mAP']:.4f}")
        print(f"  AP@0.5:    {test_metrics['AP@0.5']:.4f}")
        print(f"  AP@0.75:   {test_metrics['AP@0.75']:.4f}")
        print(f"  AP@0.95:   {test_metrics['AP@0.95']:.4f}")
        print(f"  Best Thr:  {test_metrics.get('best_threshold', 0.3):.2f}")
    
    print('=' * 70)
    
    # Save results
    result_file = output_dir / f'{args.split}_results_epoch{epoch}.json'
    with open(result_file, 'w') as f:
        # Convert all values to JSON-serializable types
        json_metrics = {}
        for k, v in test_metrics.items():
            if isinstance(v, (int, float)):
                json_metrics[k] = float(v)
            else:
                json_metrics[k] = str(v)
        
        json.dump({
            'checkpoint': str(args.checkpoint),
            'epoch': epoch,
            'split': args.split,
            'metrics': json_metrics
        }, f, indent=2)
    
    print(f"\n✅ Results saved to {result_file}")
    
    # Also save a summary text file
    summary_file = output_dir / f'{args.split}_results_epoch{epoch}.txt'
    with open(summary_file, 'w') as f:
        f.write(f"{'=' * 70}\n")
        f.write(f"{args.split.upper()} SET RESULTS (Epoch {epoch})\n")
        f.write(f"{'=' * 70}\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Split: {args.split}\n")
        f.write(f"Samples: {len(test_loader.dataset)}\n")
        f.write(f"\nLoss: {test_metrics.get('loss', 0):.4f}\n")
        f.write(f"\nFrame-level Metrics:\n")
        f.write(f"  AUC:       {test_metrics.get('frame_auc', 0):.4f}\n")
        f.write(f"  AP:        {test_metrics.get('frame_ap', 0):.4f}\n")
        f.write(f"  F1:        {test_metrics.get('frame_f1', 0):.4f}\n")
        f.write(f"  Precision: {test_metrics.get('frame_precision', 0):.4f}\n")
        f.write(f"  Recall:    {test_metrics.get('frame_recall', 0):.4f}\n")
        
        if 'video_auc' in test_metrics:
            f.write(f"\nVideo-level Metrics:\n")
            f.write(f"  AUC:       {test_metrics['video_auc']:.4f}\n")
            f.write(f"  AP:        {test_metrics.get('video_ap', 0):.4f}\n")
            f.write(f"  F1:        {test_metrics.get('video_f1', 0):.4f}\n")
        
        if 'mAP' in test_metrics:
            f.write(f"\nTemporal Localization (Segment-level):\n")
            f.write(f"  mAP:       {test_metrics['mAP']:.4f}\n")
            f.write(f"  AP@0.5:    {test_metrics['AP@0.5']:.4f}\n")
            f.write(f"  AP@0.75:   {test_metrics['AP@0.75']:.4f}\n")
            f.write(f"  AP@0.95:   {test_metrics['AP@0.95']:.4f}\n")
            f.write(f"  Best Thr:  {test_metrics.get('best_threshold', 0.3):.2f}\n")
        
        f.write(f"{'=' * 70}\n")
    
    print(f"✅ Summary saved to {summary_file}")


if __name__ == '__main__':
    main()

