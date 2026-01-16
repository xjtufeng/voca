#!/usr/bin/env python3
"""
Check frame-level probability distribution to diagnose test set issues.

This tool helps identify if the model's frame_probs are meaningful:
- Real frames should have low probs (near 0)
- Fake frames should have high probs (near 1)
- If distributions overlap heavily, model is not discriminating properly
"""
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # No display needed
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

from dataset_localization import get_dataloaders
from model_localization import FrameLocalizationModel


def main():
    parser = argparse.ArgumentParser(description='Check frame probability distribution')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--features_root', type=str, required=True,
                        help='Path to extracted features')
    parser.add_argument('--split', type=str, default='test',
                        help='Dataset split to check (default: test)')
    parser.add_argument('--batch_size', type=int, default=24,
                        help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of data loading workers')
    parser.add_argument('--max_frames', type=int, default=512,
                        help='Maximum frames per video')
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--no_cross_attn', action='store_true')
    parser.add_argument('--no_video_head', action='store_true')
    parser.add_argument('--output_dir', type=str, default='results/prob_distribution',
                        help='Directory to save plots and statistics')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[INFO] Using device: {device}')
    
    # Load checkpoint
    print(f'[INFO] Loading checkpoint: {args.checkpoint}')
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    epoch = ckpt.get('epoch', -1)
    print(f'✅ Loaded model from epoch {epoch}')
    
    # Load data
    print(f'[INFO] Loading {args.split} data from {args.features_root}')
    dataloaders = get_dataloaders(
        features_root=args.features_root,
        splits=[args.split],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_frames=args.max_frames,
        stride=1,
        distributed=False
    )
    
    loader = dataloaders[args.split]
    print(f'[INFO] {args.split.capitalize()} set: {len(loader.dataset)} samples\n')
    
    # Infer feature dimensions
    sample0 = loader.dataset[0]
    v_dim = int(sample0['visual'].shape[-1])
    a_dim = int(sample0['audio'].shape[-1])
    print(f'[INFO] Feature dims: v_dim={v_dim}, a_dim={a_dim}\n')
    
    # Build and load model
    print('[INFO] Building model...')
    model = FrameLocalizationModel(
        v_dim=v_dim,
        a_dim=a_dim,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dropout=args.dropout,
        use_cross_attn=not args.no_cross_attn,
        use_video_head=not args.no_video_head
    )
    
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)
    model.eval()
    print(f'✅ Model loaded\n')
    
    # Collect frame probs
    print('[INFO] Collecting frame probabilities...')
    real_probs = []  # Probabilities for real frames (label=0)
    fake_probs = []  # Probabilities for fake frames (label=1)
    
    with torch.no_grad():
        for batch in tqdm(loader, desc='Processing'):
            visual = batch['visual'].to(device)
            audio = batch['audio'].to(device)
            frame_labels = batch['frame_labels'].to(device)
            mask = batch['mask'].to(device)
            
            # Forward pass
            frame_logits, _, _ = model(visual, audio, mask)
            frame_probs = torch.sigmoid(frame_logits)
            
            # Extract valid frames
            for i in range(len(mask)):
                valid_mask = ~mask[i]
                valid_probs = frame_probs[i][valid_mask].cpu().numpy()
                valid_labels = frame_labels[i][valid_mask].cpu().numpy()
                
                # Separate by label
                real_mask = (valid_labels == 0)
                fake_mask = (valid_labels == 1)
                
                real_probs.extend(valid_probs[real_mask].tolist())
                fake_probs.extend(valid_probs[fake_mask].tolist())
    
    real_probs = np.array(real_probs)
    fake_probs = np.array(fake_probs)
    
    print(f'\n[INFO] Collected:')
    print(f'  Real frames: {len(real_probs):,}')
    print(f'  Fake frames: {len(fake_probs):,}')
    
    # Compute statistics
    print('\n' + '=' * 70)
    print('FRAME PROBABILITY DISTRIBUTION STATISTICS')
    print('=' * 70)
    
    print('\nReal frames (should have LOW probabilities, near 0):')
    print(f'  Mean:   {real_probs.mean():.4f}')
    print(f'  Std:    {real_probs.std():.4f}')
    print(f'  Median: {np.median(real_probs):.4f}')
    print(f'  Min:    {real_probs.min():.4f}')
    print(f'  Max:    {real_probs.max():.4f}')
    print(f'  Q1:     {np.percentile(real_probs, 25):.4f}')
    print(f'  Q3:     {np.percentile(real_probs, 75):.4f}')
    
    print('\nFake frames (should have HIGH probabilities, near 1):')
    print(f'  Mean:   {fake_probs.mean():.4f}')
    print(f'  Std:    {fake_probs.std():.4f}')
    print(f'  Median: {np.median(fake_probs):.4f}')
    print(f'  Min:    {fake_probs.min():.4f}')
    print(f'  Max:    {fake_probs.max():.4f}')
    print(f'  Q1:     {np.percentile(fake_probs, 25):.4f}')
    print(f'  Q3:     {np.percentile(fake_probs, 75):.4f}')
    
    # Separation metric
    separation = fake_probs.mean() - real_probs.mean()
    print(f'\nSeparation (Fake mean - Real mean): {separation:.4f}')
    print(f'  ✅ Good: > 0.5')
    print(f'  ⚠️  Fair: 0.2 - 0.5')
    print(f'  ❌ Poor: < 0.2')
    
    if separation > 0.5:
        print(f'  → Status: ✅ GOOD - Model is discriminating well')
    elif separation > 0.2:
        print(f'  → Status: ⚠️  FAIR - Model has some discrimination but weak')
    else:
        print(f'  → Status: ❌ POOR - Model is NOT discriminating (explains low AUC)')
    
    print('=' * 70)
    
    # Plot distributions
    print('\n[INFO] Generating distribution plot...')
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Histogram
    ax = axes[0]
    bins = np.linspace(0, 1, 51)
    ax.hist(real_probs, bins=bins, alpha=0.6, label=f'Real (n={len(real_probs):,})', 
            color='blue', density=True)
    ax.hist(fake_probs, bins=bins, alpha=0.6, label=f'Fake (n={len(fake_probs):,})', 
            color='red', density=True)
    ax.set_xlabel('Frame Probability', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'Frame Probability Distribution ({args.split} set, Epoch {epoch})', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.axvline(0.5, color='black', linestyle='--', linewidth=1, alpha=0.5, label='Threshold 0.5')
    
    # Box plot
    ax = axes[1]
    ax.boxplot([real_probs, fake_probs], labels=['Real', 'Fake'], 
                vert=False, widths=0.5, patch_artist=True,
                boxprops=dict(facecolor='lightblue', alpha=0.6))
    ax.set_xlabel('Frame Probability', fontsize=12)
    ax.set_title('Box Plot Comparison', fontsize=13, fontweight='bold')
    ax.grid(alpha=0.3, axis='x')
    ax.axvline(0.5, color='red', linestyle='--', linewidth=1, alpha=0.5)
    
    plt.tight_layout()
    
    plot_file = output_dir / f'{args.split}_prob_distribution_epoch{epoch}.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f'✅ Plot saved to {plot_file}')
    
    # Save statistics to file
    stats_file = output_dir / f'{args.split}_prob_stats_epoch{epoch}.txt'
    with open(stats_file, 'w') as f:
        f.write('=' * 70 + '\n')
        f.write(f'FRAME PROBABILITY DISTRIBUTION ({args.split} set, Epoch {epoch})\n')
        f.write('=' * 70 + '\n')
        f.write(f'Checkpoint: {args.checkpoint}\n')
        f.write(f'Total real frames: {len(real_probs):,}\n')
        f.write(f'Total fake frames: {len(fake_probs):,}\n\n')
        
        f.write('Real frames:\n')
        f.write(f'  Mean:   {real_probs.mean():.4f}\n')
        f.write(f'  Std:    {real_probs.std():.4f}\n')
        f.write(f'  Median: {np.median(real_probs):.4f}\n')
        f.write(f'  Range:  [{real_probs.min():.4f}, {real_probs.max():.4f}]\n\n')
        
        f.write('Fake frames:\n')
        f.write(f'  Mean:   {fake_probs.mean():.4f}\n')
        f.write(f'  Std:    {fake_probs.std():.4f}\n')
        f.write(f'  Median: {np.median(fake_probs):.4f}\n')
        f.write(f'  Range:  [{fake_probs.min():.4f}, {fake_probs.max():.4f}]\n\n')
        
        f.write(f'Separation: {separation:.4f}\n')
        f.write('=' * 70 + '\n')
    
    print(f'✅ Statistics saved to {stats_file}')
    print('\n✅ Analysis complete!')


if __name__ == '__main__':
    main()

