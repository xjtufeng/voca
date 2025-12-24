#!/usr/bin/env python3
"""
Training script for three-branch joint deepfake detection model
Supports single-GPU and multi-GPU distributed training
"""
import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

from dataset_localization import get_dataloaders
from model_three_branch import (
    ThreeBranchJointModel,
    compute_three_branch_loss
)


def setup_ddp():
    """Initialize DDP from environment variables"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        
        return rank, world_size, local_rank
    else:
        return 0, 1, 0


def cleanup_ddp():
    """Cleanup DDP"""
    if dist.is_initialized():
        dist.destroy_process_group()


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    args: argparse.Namespace,
    rank: int = 0
) -> Dict[str, float]:
    """Train for one epoch"""
    model.train()
    
    total_losses = {
        'total': 0, 'fused': 0,
        'cross_modal': 0, 'audio_only': 0, 'visual_only': 0
    }
    
    if rank == 0:
        pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]")
    else:
        pbar = loader
    
    for batch_idx, batch in enumerate(pbar):
        visual = batch['visual'].to(device)
        audio = batch['audio'].to(device)
        video_labels = batch['video_labels'].to(device)
        mask = batch['mask'].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass with AMP
        with autocast('cuda'):
            outputs = model(
                audio=audio,
                visual=visual,
                mask=mask,
                return_branch_outputs=True
            )
            
            # Compute multi-task loss
            losses = compute_three_branch_loss(
                outputs=outputs,
                labels=video_labels,
                branch_weights=(
                    args.cm_loss_weight,
                    args.ao_loss_weight,
                    args.vo_loss_weight
                ),
                fusion_weight=args.fusion_loss_weight
            )
            
            loss = losses['total']
        
        # Backward pass with AMP
        scaler.scale(loss).backward()
        
        # Gradient clipping
        if args.max_grad_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        
        scaler.step(optimizer)
        scaler.update()
        
        # Accumulate losses
        for key in total_losses:
            total_losses[key] += losses[key].item()
        
        # Update progress bar
        if rank == 0:
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'fused': f'{losses["fused"].item():.4f}',
                'cm': f'{losses["cross_modal"].item():.4f}'
            })
    
    num_batches = len(loader)
    return {k: v / num_batches for k, v in total_losses.items()}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    rank: int = 0
) -> Dict[str, float]:
    """Evaluate on validation/test set"""
    model.eval()
    
    all_preds = {'fused': [], 'cm': [], 'ao': [], 'vo': []}
    all_labels = []
    all_fusion_weights = []
    
    if rank == 0:
        pbar = tqdm(loader, desc="Evaluating")
    else:
        pbar = loader
    
    for batch in pbar:
        visual = batch['visual'].to(device)
        audio = batch['audio'].to(device)
        video_labels = batch['video_labels'].to(device)
        mask = batch['mask'].to(device)
        
        # Forward pass
        outputs = model(
            audio=audio,
            visual=visual,
            mask=mask,
            return_branch_outputs=True
        )
        
        # Collect predictions
        fused_probs = torch.sigmoid(outputs['fused_logit']).squeeze(-1).cpu().numpy()
        all_preds['fused'].extend(fused_probs)
        
        if 'branch_logits' in outputs:
            cm_probs = torch.sigmoid(outputs['branch_logits']['cross_modal']).squeeze(-1).cpu().numpy()
            ao_probs = torch.sigmoid(outputs['branch_logits']['audio_only']).squeeze(-1).cpu().numpy()
            vo_probs = torch.sigmoid(outputs['branch_logits']['visual_only']).squeeze(-1).cpu().numpy()
            
            all_preds['cm'].extend(cm_probs)
            all_preds['ao'].extend(ao_probs)
            all_preds['vo'].extend(vo_probs)
        
        if 'fusion_weights' in outputs:
            weights = outputs['fusion_weights'].cpu().numpy()
            all_fusion_weights.append(weights)
        
        all_labels.extend(video_labels.cpu().numpy())
    
    # Convert to numpy
    for key in all_preds:
        all_preds[key] = np.array(all_preds[key])
    all_labels = np.array(all_labels)
    
    # Compute metrics for each branch
    metrics = {}
    
    for branch_name, preds in all_preds.items():
        if len(preds) == 0:
            continue
        
        preds_binary = (preds > 0.5).astype(int)
        
        try:
            auc = roc_auc_score(all_labels, preds)
            f1 = f1_score(all_labels, preds_binary)
            precision = precision_score(all_labels, preds_binary, zero_division=0)
            recall = recall_score(all_labels, preds_binary, zero_division=0)
            
            metrics[f'{branch_name}_auc'] = auc
            metrics[f'{branch_name}_f1'] = f1
            metrics[f'{branch_name}_precision'] = precision
            metrics[f'{branch_name}_recall'] = recall
        except:
            pass
    
    # Fusion weights statistics
    if len(all_fusion_weights) > 0:
        all_fusion_weights = np.concatenate(all_fusion_weights, axis=0)
        metrics['fusion_weight_cm'] = all_fusion_weights[:, 0].mean()
        metrics['fusion_weight_ao'] = all_fusion_weights[:, 1].mean()
        metrics['fusion_weight_vo'] = all_fusion_weights[:, 2].mean()
    
    return metrics


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    args: argparse.Namespace,
    filename: str
):
    """Save checkpoint"""
    model_state = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'args': vars(args)
    }
    
    torch.save(checkpoint, filename)


def main():
    parser = argparse.ArgumentParser(description='Train three-branch joint model')
    
    # Data
    parser.add_argument('--features_root', type=str, required=True,
                        help='Path to extracted features')
    parser.add_argument('--splits', nargs='+', default=['train', 'dev'],
                        help='Dataset splits to use')
    parser.add_argument('--max_frames', type=int, default=256,
                        help='Maximum frames per video')
    
    # Model
    parser.add_argument('--d_model', type=int, default=512,
                        help='Hidden dimension')
    parser.add_argument('--nhead', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--cm_layers', type=int, default=4,
                        help='Layers in cross-modal branch')
    parser.add_argument('--ao_layers', type=int, default=3,
                        help='Layers in audio-only branch')
    parser.add_argument('--vo_layers', type=int, default=3,
                        help='Layers in visual-only branch')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    parser.add_argument('--fusion_method', type=str, default='attention',
                        choices=['concat', 'weighted', 'attention'],
                        help='Branch fusion method')
    
    # Loss weights
    parser.add_argument('--fusion_loss_weight', type=float, default=1.0,
                        help='Weight for fused prediction loss')
    parser.add_argument('--cm_loss_weight', type=float, default=0.3,
                        help='Weight for cross-modal branch loss')
    parser.add_argument('--ao_loss_weight', type=float, default=0.2,
                        help='Weight for audio-only branch loss')
    parser.add_argument('--vo_loss_weight', type=float, default=0.2,
                        help='Weight for visual-only branch loss')
    
    # Training
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size per GPU')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='Warmup epochs')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='Max gradient norm for clipping')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of dataloader workers')
    
    # Checkpointing
    parser.add_argument('--output_dir', type=str, default='./checkpoints/three_branch',
                        help='Output directory for checkpoints')
    parser.add_argument('--save_every', type=int, default=5,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    # Setup DDP
    rank, world_size, local_rank = setup_ddp()
    device = torch.device(f'cuda:{local_rank}')
    
    if rank == 0:
        print(f"[INFO] Training three-branch model with {world_size} GPUs")
        print(f"[INFO] Fusion method: {args.fusion_method}")
        print(f"[INFO] Loss weights: fusion={args.fusion_loss_weight}, cm={args.cm_loss_weight}, ao={args.ao_loss_weight}, vo={args.vo_loss_weight}")
        
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_dir / 'args.json', 'w') as f:
            json.dump(vars(args), f, indent=2)
    
    # Load data
    if rank == 0:
        print(f"\n[INFO] Loading data from {args.features_root}")
    
    dataloaders = get_dataloaders(
        features_root=args.features_root,
        splits=args.splits,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_frames=args.max_frames
    )
    
    train_loader = dataloaders.get('train')
    val_loader = dataloaders.get('dev') or dataloaders.get('test')
    
    # Create model
    if rank == 0:
        print(f"\n[INFO] Creating three-branch model...")
    
    model = ThreeBranchJointModel(
        v_dim=512,
        a_dim=1024,
        d_model=args.d_model,
        nhead=args.nhead,
        cm_layers=args.cm_layers,
        ao_layers=args.ao_layers,
        vo_layers=args.vo_layers,
        dropout=args.dropout,
        fusion_method=args.fusion_method
    ).to(device)
    
    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"[INFO] Model parameters: {total_params:.2f}M")
    
    # Wrap with DDP
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
    
    # Optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Cosine annealing with warmup
    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        else:
            progress = (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)
            return 0.5 * (1 + np.cos(np.pi * progress))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # AMP scaler
    scaler = GradScaler('cuda')
    
    # Resume from checkpoint
    start_epoch = 0
    best_metric = 0.0
    
    if args.resume:
        if rank == 0:
            print(f"\n[INFO] Resuming from {args.resume}")
        
        checkpoint = torch.load(args.resume, map_location=device)
        model_state = checkpoint['model_state_dict']
        
        if isinstance(model, DDP):
            model.module.load_state_dict(model_state)
        else:
            model.load_state_dict(model_state)
        
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_metric = checkpoint.get('metrics', {}).get('fused_auc', 0.0)
    
    # Training loop
    if rank == 0:
        print(f"\n[INFO] Starting training for {args.epochs} epochs...")
    
    for epoch in range(start_epoch, args.epochs):
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, scaler, device, epoch, args, rank
        )
        
        # Evaluate
        if val_loader is not None:
            val_metrics = evaluate(model, val_loader, device, args, rank)
        else:
            val_metrics = {}
        
        # Step scheduler
        scheduler.step()
        
        # Log
        if rank == 0:
            print(f"\n[Epoch {epoch}]")
            print(f"  Train - Total: {train_metrics['total']:.4f}, "
                  f"Fused: {train_metrics['fused']:.4f}, "
                  f"CM: {train_metrics['cross_modal']:.4f}, "
                  f"AO: {train_metrics['audio_only']:.4f}, "
                  f"VO: {train_metrics['visual_only']:.4f}")
            
            if val_metrics:
                print(f"  Val   - Fused AUC: {val_metrics.get('fused_auc', 0):.4f}, "
                      f"CM AUC: {val_metrics.get('cm_auc', 0):.4f}, "
                      f"AO AUC: {val_metrics.get('ao_auc', 0):.4f}, "
                      f"VO AUC: {val_metrics.get('vo_auc', 0):.4f}")
                
                if 'fusion_weight_cm' in val_metrics:
                    print(f"  Fusion Weights - CM: {val_metrics['fusion_weight_cm']:.3f}, "
                          f"AO: {val_metrics['fusion_weight_ao']:.3f}, "
                          f"VO: {val_metrics['fusion_weight_vo']:.3f}")
            
            # Save checkpoint
            if (epoch + 1) % args.save_every == 0:
                ckpt_path = Path(args.output_dir) / f'checkpoint_epoch{epoch}.pth'
                save_checkpoint(model, optimizer, epoch, val_metrics, args, ckpt_path)
                print(f"  Saved checkpoint: {ckpt_path}")
            
            # Save best model
            current_metric = val_metrics.get('fused_auc', 0.0)
            if current_metric > best_metric:
                best_metric = current_metric
                ckpt_path = Path(args.output_dir) / 'best.pth'
                save_checkpoint(model, optimizer, epoch, val_metrics, args, ckpt_path)
                print(f"  New best model! Fused AUC: {best_metric:.4f}")
    
    # Final save
    if rank == 0:
        final_path = Path(args.output_dir) / 'final.pth'
        save_checkpoint(model, optimizer, args.epochs - 1, val_metrics, args, final_path)
        print(f"\n[INFO] Training complete! Final model saved to {final_path}")
        print(f"[INFO] Best Fused AUC: {best_metric:.4f}")
    
    cleanup_ddp()


if __name__ == '__main__':
    main()

