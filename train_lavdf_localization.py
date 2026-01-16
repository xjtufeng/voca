#!/usr/bin/env python3
"""
Training script for frame-level deepfake localization on LAV-DF dataset
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
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, average_precision_score

from dataset_localization import get_dataloaders
from evaluation.temporal_localization import evaluate_temporal_localization
from model_localization import (
    FrameLocalizationModel,
    compute_frame_loss,
    compute_video_loss,
    compute_temporal_smoothness_loss,
    compute_combined_loss
)


def _autocast_ctx(device: torch.device):
    """
    Version-safe autocast context.
    - Newer PyTorch: torch.amp.autocast(device_type='cuda', enabled=...)
    - Older PyTorch: torch.cuda.amp.autocast(enabled=...)
    """
    enabled = (device.type == "cuda")
    # Prefer torch.amp.autocast if available (PyTorch 2.x)
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type="cuda", enabled=enabled)
    # Fallback
    from torch.cuda.amp import autocast  # type: ignore
    return autocast(enabled=enabled)


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
    
    total_loss = 0
    total_frame_loss = 0
    total_video_loss = 0
    total_smooth_loss = 0
    total_similarity_loss = 0
    
    if rank == 0:
        pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]")
    else:
        pbar = loader
    
    for batch_idx, batch in enumerate(pbar):
        visual = batch['visual'].to(device)
        audio = batch['audio'].to(device)
        frame_labels = batch['frame_labels'].to(device)
        video_labels = batch['video_labels'].to(device)
        mask = batch['mask'].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass with AMP (version-safe)
        with _autocast_ctx(device):
            frame_logits, video_logit, frame_similarity = model(visual, audio, mask)
            
            # Compute losses using combined loss function
            losses = compute_combined_loss(
                frame_logits=frame_logits,
                frame_labels=frame_labels,
                mask=mask,
                frame_similarity=frame_similarity,
                video_logit=video_logit,
                video_label=video_labels,
                frame_loss_weight=1.0,
                video_loss_weight=args.video_loss_weight,
                smooth_loss_weight=args.smooth_loss_weight,
                similarity_loss_weight=args.similarity_loss_weight if hasattr(args, 'similarity_loss_weight') else 0.2,
                pos_weight=args.pos_weight if args.pos_weight > 0 else None,
                use_focal=args.use_focal,
                focal_alpha=args.focal_alpha,
                focal_gamma=args.focal_gamma
            )
            
            loss = losses['total']
            frame_loss = losses['frame']
            video_loss = losses['video']
            smooth_loss = losses['smooth']
            similarity_loss = losses['similarity']
        
        # Backward pass with AMP
        scaler.scale(loss).backward()
        
        # Gradient clipping
        if args.max_grad_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        
        scaler.step(optimizer)
        scaler.update()
        
        # Accumulate losses
        total_loss += loss.item()
        total_frame_loss += frame_loss.item()
        total_video_loss += video_loss.item()
        total_smooth_loss += smooth_loss.item()
        total_similarity_loss = total_similarity_loss + similarity_loss.item() if 'total_similarity_loss' in locals() else similarity_loss.item()
        
        # Update progress bar
        if rank == 0:
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'frame': f'{frame_loss.item():.4f}',
                'sim': f'{similarity_loss.item():.4f}'
            })
    
    num_batches = len(loader)
    return {
        'loss': total_loss / num_batches,
        'frame_loss': total_frame_loss / num_batches,
        'video_loss': total_video_loss / num_batches,
        'smooth_loss': total_smooth_loss / num_batches,
        'similarity_loss': total_similarity_loss / num_batches if 'total_similarity_loss' in locals() else 0.0
    }


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

    # Handle empty loader/dataset gracefully
    try:
        if loader is None or len(loader) == 0:  # type: ignore[arg-type]
            return {}
    except TypeError:
        # Some DataLoader implementations may not support len(); proceed.
        pass
    
    total_loss = 0
    all_frame_probs = []  # List of arrays (one per video)
    all_frame_labels = []  # List of arrays (one per video)
    all_video_ids = []  # List of video IDs
    all_video_probs = []
    all_video_labels = []
    
    if rank == 0:
        pbar = tqdm(loader, desc="Evaluating")
    else:
        pbar = loader
    
    for batch in pbar:
        visual = batch['visual'].to(device)
        audio = batch['audio'].to(device)
        frame_labels = batch['frame_labels'].to(device)
        video_labels = batch['video_labels'].to(device)
        mask = batch['mask'].to(device)
        video_ids = batch['video_ids']
        
        # Forward pass
        frame_logits, video_logit, frame_similarity = model(visual, audio, mask)
        
        # Compute loss
        frame_loss = compute_frame_loss(
            frame_logits, frame_labels, mask,
            pos_weight=args.pos_weight if args.pos_weight > 0 else None,
            use_focal=args.use_focal
        )
        total_loss += frame_loss.item()
        
        # Collect predictions (remove padding)
        frame_probs = torch.sigmoid(frame_logits.squeeze(-1))  # [B, T]
        
        for i in range(len(mask)):
            valid_mask = ~mask[i]
            valid_probs = frame_probs[i][valid_mask].cpu().numpy()
            valid_labels = frame_labels[i][valid_mask].cpu().numpy()
            
            # Store per-video (for temporal localization evaluation)
            all_frame_probs.append(valid_probs)
            all_frame_labels.append(valid_labels)
            all_video_ids.append(video_ids[i])
        
        # Video-level predictions
        if video_logit is not None:
            video_probs = torch.sigmoid(video_logit.squeeze(-1)).cpu().numpy()
            all_video_probs.extend(video_probs)
            all_video_labels.extend(video_labels.cpu().numpy())
    
    # No samples collected (e.g., empty split)
    if len(all_frame_probs) == 0:
        return {}

    # Compute frame-level metrics (flatten for traditional metrics)
    all_frame_probs_flat = np.concatenate(all_frame_probs)
    all_frame_labels_flat = np.concatenate(all_frame_labels)
    frame_preds_binary = (all_frame_probs_flat > 0.5).astype(int)
    
    frame_auc = roc_auc_score(all_frame_labels_flat, all_frame_probs_flat)
    frame_ap = average_precision_score(all_frame_labels_flat, all_frame_probs_flat)
    frame_f1 = f1_score(all_frame_labels_flat, frame_preds_binary)
    frame_precision = precision_score(all_frame_labels_flat, frame_preds_binary, zero_division=0)
    frame_recall = recall_score(all_frame_labels_flat, frame_preds_binary, zero_division=0)
    
    metrics = {
        'loss': total_loss / len(loader),
        'frame_auc': frame_auc,
        'frame_ap': frame_ap,
        'frame_f1': frame_f1,
        'frame_precision': frame_precision,
        'frame_recall': frame_recall
    }
    
    # Temporal localization metrics (segment-level AP@IoU)
    if rank == 0:  # Only compute on rank 0 to save time
        try:
            temporal_metrics = evaluate_temporal_localization(
                all_frame_probs,
                all_frame_labels,
                video_ids=all_video_ids,
                thresholds=[0.1, 0.2, 0.3, 0.4, 0.5],
                iou_thresholds=[0.5, 0.75, 0.95],
                nms_iou=0.7,
                min_length=3,
                merge_gap=2,
                use_best_threshold=True
            )
            
            # Add to metrics
            metrics['AP@0.5'] = temporal_metrics['AP@0.5']
            metrics['AP@0.75'] = temporal_metrics['AP@0.75']
            metrics['AP@0.95'] = temporal_metrics['AP@0.95']
            metrics['mAP'] = temporal_metrics['mAP']
            metrics['best_threshold'] = temporal_metrics['best_threshold']
        except Exception as e:
            print(f"[WARN] Temporal localization evaluation failed: {e}")
            metrics['AP@0.5'] = 0.0
            metrics['AP@0.75'] = 0.0
            metrics['AP@0.95'] = 0.0
            metrics['mAP'] = 0.0
    
    # Video-level metrics (if available)
    if len(all_video_probs) > 0:
        all_video_probs = np.array(all_video_probs)
        all_video_labels = np.array(all_video_labels)
        video_preds_binary = (all_video_probs > 0.5).astype(int)
        
        video_auc = roc_auc_score(all_video_labels, all_video_probs)
        video_ap = average_precision_score(all_video_labels, all_video_probs)
        video_f1 = f1_score(all_video_labels, video_preds_binary)
        
        metrics['video_auc'] = video_auc
        metrics['video_ap'] = video_ap
        metrics['video_f1'] = video_f1
    
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
    # Unwrap DDP if needed
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
    parser = argparse.ArgumentParser(description='Train frame-level localization model')
    
    # Data
    parser.add_argument('--features_root', type=str, required=True,
                        help='Path to extracted LAV-DF features')
    parser.add_argument('--splits', nargs='+', default=['train', 'dev'],
                        help='Dataset splits to use')
    parser.add_argument('--max_frames', type=int, default=512,
                        help='Maximum frames per video')
    parser.add_argument('--stride', type=int, default=1,
                        help='Frame sampling stride')
    
    # Model
    parser.add_argument('--d_model', type=int, default=512,
                        help='Transformer hidden dimension')
    parser.add_argument('--nhead', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=4,
                        help='Number of Transformer layers')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    parser.add_argument('--no_cross_attn', action='store_true',
                        help='Disable cross-modal attention')
    parser.add_argument('--no_video_head', action='store_true',
                        help='Disable video-level classification head')
    
    # Loss
    parser.add_argument('--pos_weight', type=float, default=-1,
                        help='Positive class weight for BCE (auto if -1)')
    parser.add_argument('--use_focal', action='store_true',
                        help='Use Focal Loss instead of BCE')
    parser.add_argument('--focal_alpha', type=float, default=0.25,
                        help='Focal loss alpha')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='Focal loss gamma')
    parser.add_argument('--video_loss_weight', type=float, default=0.3,
                        help='Weight for video-level auxiliary loss')
    parser.add_argument('--smooth_loss_weight', type=float, default=0.1,
                        help='Weight for temporal smoothness regularization')
    parser.add_argument('--similarity_loss_weight', type=float, default=0.2,
                        help='Weight for similarity supervision loss')
    
    # Training
    parser.add_argument('--batch_size', type=int, default=8,
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
    parser.add_argument('--output_dir', type=str, default='./checkpoints/localization',
                        help='Output directory for checkpoints')
    parser.add_argument('--save_every', type=int, default=1,
                        help='Save checkpoint every N epochs (default: 1, save all epochs)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    # Setup DDP
    rank, world_size, local_rank = setup_ddp()
    device = torch.device(f'cuda:{local_rank}')
    
    if rank == 0:
        print(f"[INFO] Training with {world_size} GPUs")
        print(f"[INFO] Arguments: {args}")
        
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save args
        with open(output_dir / 'args.json', 'w') as f:
            json.dump(vars(args), f, indent=2)
    
    # Load data
    if rank == 0:
        print(f"\n[INFO] Loading data from {args.features_root}")
    
    # Support dev_temp (train-split dev) if it exists
    actual_splits = []
    for split in args.splits:
        if split == 'dev':
            # Check if dev_temp exists (train-split dev), use it instead
            dev_temp_path = Path(args.features_root) / 'dev_temp'
            if dev_temp_path.exists():
                actual_splits.append('dev_temp')
                if rank == 0:
                    print(f"[INFO] Using dev_temp (train-split dev) instead of dev")
            else:
                actual_splits.append('dev')
        else:
            actual_splits.append(split)
    
    dataloaders = get_dataloaders(
        features_root=args.features_root,
        splits=actual_splits,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_frames=args.max_frames,
        stride=args.stride,
        distributed=(world_size > 1),
        rank=rank,
        world_size=world_size,
    )
    
    # Remap dev_temp back to dev for consistency
    if 'dev_temp' in dataloaders:
        dataloaders['dev'] = dataloaders.pop('dev_temp')
        if 'dev_temp_sampler' in dataloaders:
            dataloaders['dev_sampler'] = dataloaders.pop('dev_temp_sampler')
    
    train_loader = dataloaders.get('train')
    val_loader = dataloaders.get('dev') or dataloaders.get('test')

    # If dev split exists but is empty, fall back to test; otherwise skip evaluation.
    if val_loader is not None:
        try:
            val_len = len(val_loader.dataset)  # type: ignore[attr-defined]
        except Exception:
            val_len = None

        if val_len == 0:
            test_loader = dataloaders.get('test')
            test_len = None
            if test_loader is not None:
                try:
                    test_len = len(test_loader.dataset)  # type: ignore[attr-defined]
                except Exception:
                    test_len = None

            if rank == 0:
                print("[WARN] Validation split is empty. Falling back to test split for evaluation.")

            if test_loader is not None and test_len != 0:
                val_loader = test_loader
            else:
                val_loader = None

    # DDP safety: only run evaluation on rank0 to avoid per-rank DataLoader failures
    # causing progress divergence (and subsequent NCCL/allreduce timeouts).
    if world_size > 1 and rank != 0:
        val_loader = None
    
    # Infer feature dimensions from dataset to avoid hardcoding.
    # LAV-DF audio embeddings can be 512-d (motion latent) or 1024-d depending on extraction.
    if train_loader is None:
        raise ValueError("Train split DataLoader is None. Check --splits and features_root.")
    sample0 = train_loader.dataset[0]
    inferred_v_dim = int(sample0['visual'].shape[-1])
    inferred_a_dim = int(sample0['audio'].shape[-1])
    if rank == 0:
        print(f"[INFO] Inferred dims: v_dim={inferred_v_dim}, a_dim={inferred_a_dim}")

    # Auto pos_weight
    if args.pos_weight < 0:
        args.pos_weight = dataloaders.get('pos_weight', 1.0)
        if rank == 0:
            print(f"[INFO] Auto pos_weight: {args.pos_weight:.2f}")
    
    # Create model
    if rank == 0:
        print(f"\n[INFO] Creating model...")
    
    model = FrameLocalizationModel(
        v_dim=inferred_v_dim,
        a_dim=inferred_a_dim,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dropout=args.dropout,
        use_cross_attn=not args.no_cross_attn,
        use_video_head=not args.no_video_head
    ).to(device)
    
    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"[INFO] Model parameters: {total_params:.2f}M")
    
    # Wrap with DDP
    if world_size > 1:
        # find_unused_parameters adds overhead and can worsen performance/stability.
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
    scaler = GradScaler(enabled=(device.type == "cuda"))
    
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
        best_metric = checkpoint.get('metrics', {}).get('frame_auc', 0.0)
    
    # Training loop
    if rank == 0:
        print(f"\n[INFO] Starting training...")
        print(f"[INFO] Training for {args.epochs} epochs")
    
    for epoch in range(start_epoch, args.epochs):
        # Ensure each rank uses a different shard each epoch when using DistributedSampler
        train_sampler = dataloaders.get('train_sampler')
        if train_sampler is not None and hasattr(train_sampler, 'set_epoch'):
            train_sampler.set_epoch(epoch)

        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, scaler, device, epoch, args, rank
        )

        # Sync all ranks before evaluation so no rank starts the next epoch early.
        if world_size > 1 and dist.is_initialized():
            dist.barrier()

        # Evaluate (rank0 only in DDP)
        if val_loader is not None:
            val_metrics = evaluate(model, val_loader, device, args, rank)
        else:
            val_metrics = {}

        # Sync all ranks after evaluation as well.
        if world_size > 1 and dist.is_initialized():
            dist.barrier()
        
        # Step scheduler
        scheduler.step()
        
        # Log
        if rank == 0:
            print(f"\n[Epoch {epoch}]")
            print(f"  Train - Loss: {train_metrics['loss']:.4f}, "
                  f"Frame: {train_metrics['frame_loss']:.4f}, "
                  f"Video: {train_metrics['video_loss']:.4f}")
            
            if val_metrics:
                print(f"  Val   - Loss: {val_metrics['loss']:.4f}, "
                      f"Frame AUC: {val_metrics['frame_auc']:.4f}, "
                      f"AP: {val_metrics['frame_ap']:.4f}, "
                      f"F1: {val_metrics['frame_f1']:.4f}")
                
                if 'video_auc' in val_metrics:
                    print(f"          Video AUC: {val_metrics['video_auc']:.4f}, "
                          f"AP: {val_metrics['video_ap']:.4f}, "
                          f"F1: {val_metrics['video_f1']:.4f}")
                
                if 'mAP' in val_metrics:
                    print(f"          Temporal: mAP={val_metrics['mAP']:.4f}, "
                          f"AP@0.5={val_metrics['AP@0.5']:.4f}, "
                          f"AP@0.75={val_metrics['AP@0.75']:.4f}, "
                          f"AP@0.95={val_metrics['AP@0.95']:.4f} "
                          f"(threshold={val_metrics.get('best_threshold', 0.3):.2f})")
            
            # Save checkpoint
            if (epoch + 1) % args.save_every == 0:
                ckpt_path = output_dir / f'checkpoint_epoch{epoch}.pth'
                save_checkpoint(model, optimizer, epoch, val_metrics, args, ckpt_path)
                print(f"  Saved checkpoint: {ckpt_path}")
            
            # Save best model
            current_metric = val_metrics.get('frame_auc', 0.0)
            if current_metric > best_metric:
                best_metric = current_metric
                ckpt_path = output_dir / 'best.pth'
                save_checkpoint(model, optimizer, epoch, val_metrics, args, ckpt_path)
                print(f"  New best model! Frame AUC: {best_metric:.4f}")
    
    # Final save
    if rank == 0:
        final_path = output_dir / 'final.pth'
        save_checkpoint(model, optimizer, args.epochs - 1, val_metrics, args, final_path)
        print(f"\n[INFO] Training complete! Final model saved to {final_path}")
        print(f"[INFO] Best Frame AUC: {best_metric:.4f}")
    
    cleanup_ddp()


if __name__ == '__main__':
    main()

