#!/usr/bin/env python3
"""
Enhanced training script for frame-level deepfake localization on LAV-DF dataset
Uses improved model with:
- Learned inconsistency scoring
- Soft reliability gating
- Ranking loss with hard negatives
- Fake hinge loss
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
from model_localization import (
    FrameLocalizationModel,
    compute_combined_loss,
    generate_hard_negatives
)
from evaluation_boundary import two_stage_localization, evaluate_segment_level, get_segments_from_binary


def _autocast_ctx(device: torch.device):
    """Version-safe autocast context"""
    enabled = (device.type == "cuda")
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type="cuda", enabled=enabled)
    from torch.cuda.amp import autocast
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
    """Train for one epoch with enhanced loss"""
    model.train()
    
    total_loss = 0
    total_frame_loss = 0
    total_video_loss = 0
    total_boundary_loss = 0
    total_smooth_loss = 0
    total_ranking_loss = 0
    total_fake_hinge_loss = 0
    
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
        
        # Forward pass with AMP
        with _autocast_ctx(device):
            # Forward with correct audio (positive pair)
            outputs_pos = model(visual, audio, mask)
            frame_logits = outputs_pos['frame_logits']
            video_logit = outputs_pos['video_logit']
            start_logits = outputs_pos['start_logits']
            end_logits = outputs_pos['end_logits']
            inconsistency_pos = outputs_pos['inconsistency_score']
            inconsistency_gated = outputs_pos['inconsistency_gated']
            reliability_gate = outputs_pos['reliability_gate']
            
            # Generate hard negatives and forward (for ranking loss)
            audio_neg = generate_hard_negatives(
                audio, mask,
                shift_range=(args.neg_shift_min, args.neg_shift_max),
                swap_prob=args.neg_swap_prob
            )
            outputs_neg = model(visual, audio_neg, mask)
            inconsistency_neg = outputs_neg['inconsistency_score']
            
            # Compute combined loss
            losses = compute_combined_loss(
                frame_logits=frame_logits,
                frame_labels=frame_labels,
                mask=mask,
                video_logit=video_logit,
                video_label=video_labels,
                start_logits=start_logits,
                end_logits=end_logits,
                inconsistency_pos=inconsistency_pos,
                inconsistency_neg=inconsistency_neg,
                inconsistency_gated=inconsistency_gated,
                reliability_gate=reliability_gate,
                frame_loss_weight=1.0,
                video_loss_weight=args.video_loss_weight,
                boundary_loss_weight=args.boundary_loss_weight,
                smooth_loss_weight=args.smooth_loss_weight,
                ranking_loss_weight=args.ranking_loss_weight,
                fake_hinge_weight=args.fake_hinge_weight,
                boundary_tolerance=args.boundary_tolerance,
                ranking_margin=args.ranking_margin,
                use_boundary_aware_smooth=args.use_boundary_aware_smooth,
                pos_weight=args.pos_weight if args.pos_weight > 0 else None,
                use_focal=args.use_focal,
                focal_alpha=args.focal_alpha,
                focal_gamma=args.focal_gamma
            )
            
            loss = losses['total']
            frame_loss = losses['frame']
            video_loss = losses['video']
            boundary_loss = losses['boundary']
            smooth_loss = losses['smooth']
            ranking_loss = losses['ranking']
            fake_hinge_loss = losses['fake_hinge']
        
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
        total_boundary_loss += boundary_loss.item()
        total_smooth_loss += smooth_loss.item()
        total_ranking_loss += ranking_loss.item()
        total_fake_hinge_loss += fake_hinge_loss.item()
        
        # Update progress bar
        if rank == 0:
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'frame': f'{frame_loss.item():.4f}',
                'bound': f'{boundary_loss.item():.4f}',
                'alpha': f'{model.module.alpha.item() if hasattr(model, "module") else model.alpha.item():.3f}'
            })
    
    num_batches = len(loader)
    return {
        'loss': total_loss / num_batches,
        'frame_loss': total_frame_loss / num_batches,
        'video_loss': total_video_loss / num_batches,
        'boundary_loss': total_boundary_loss / num_batches,
        'smooth_loss': total_smooth_loss / num_batches,
        'ranking_loss': total_ranking_loss / num_batches,
        'fake_hinge_loss': total_fake_hinge_loss / num_batches
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    rank: int = 0,
    do_segment_eval: bool = True
) -> Dict[str, float]:
    """Evaluate on validation set
    
    Args:
        do_segment_eval: If False, skip expensive segment-level evaluation
    """
    model.eval()
    
    try:
        if loader is None or len(loader) == 0:
            return {}
    except TypeError:
        pass
    
    all_frame_probs = []
    all_frame_labels = []
    all_video_probs = []
    all_video_labels = []
    
    # Additional metrics for enhanced model
    all_inconsistency_scores = []
    all_gate_values = []
    
    # Segment-level evaluation (only if requested)
    segment_metrics_list = []
    
    if rank == 0:
        pbar = tqdm(loader, desc="Evaluating")
    else:
        pbar = loader
    
    max_eval_videos = int(getattr(args, "eval_max_videos", 0) or 0)
    eval_stride = max(1, int(getattr(args, "eval_stride", 1) or 1))
    videos_seen = 0
    stop_eval = False

    for batch in pbar:
        visual = batch['visual'].to(device)
        audio = batch['audio'].to(device)
        frame_labels = batch['frame_labels'].to(device)
        video_labels = batch['video_labels'].to(device)
        mask = batch['mask'].to(device)
        
        with _autocast_ctx(device):
            outputs = model(visual, audio, mask)
            frame_logits = outputs['frame_logits']
            video_logit = outputs['video_logit']
            start_logits = outputs.get('start_logits')
            end_logits = outputs.get('end_logits')
            inconsistency_score = outputs['inconsistency_score']
            reliability_gate = outputs['reliability_gate']
        
        # Frame-level predictions
        frame_probs = torch.sigmoid(frame_logits.squeeze(-1))  # [B, T]
        
        # Boundary predictions
        start_probs = torch.sigmoid(start_logits.squeeze(-1)) if start_logits is not None else None  # [B, T]
        end_probs = torch.sigmoid(end_logits.squeeze(-1)) if end_logits is not None else None  # [B, T]
        
        # Video-level predictions
        if video_logit is not None:
            video_probs = torch.sigmoid(video_logit.squeeze(-1))  # [B]
        else:
            video_probs = frame_probs.mean(dim=1)  # [B]
        
        # Filter out padded frames and run two-stage inference
        for i in range(frame_probs.size(0)):
            valid_mask = ~mask[i]
            if valid_mask.sum() > 0:
                if max_eval_videos > 0 and videos_seen >= max_eval_videos:
                    stop_eval = True
                    break
                # Frame-level
                frame_probs_i = frame_probs[i][valid_mask].cpu().numpy()
                frame_labels_i = frame_labels[i][valid_mask].cpu().numpy()
                if eval_stride > 1:
                    frame_probs_i = frame_probs_i[::eval_stride]
                    frame_labels_i = frame_labels_i[::eval_stride]
                all_frame_probs.append(frame_probs_i)
                all_frame_labels.append(frame_labels_i)
                inc_i = inconsistency_score[i][valid_mask].squeeze(-1).cpu().numpy()
                gate_i = reliability_gate[i][valid_mask].squeeze(-1).cpu().numpy()
                if eval_stride > 1:
                    inc_i = inc_i[::eval_stride]
                    gate_i = gate_i[::eval_stride]
                inc_i = np.nan_to_num(inc_i, nan=0.0, posinf=50.0, neginf=-50.0)
                gate_i = np.nan_to_num(gate_i, nan=0.0, posinf=1.0, neginf=0.0)
                inc_i = np.clip(inc_i, -50.0, 50.0)
                gate_i = np.clip(gate_i, 0.0, 1.0)
                all_inconsistency_scores.append(inc_i)
                all_gate_values.append(gate_i)
                
                # Segment-level: Two-stage inference (only if requested)
                if do_segment_eval and start_probs is not None and end_probs is not None:
                    start_probs_i = start_probs[i][valid_mask].cpu().numpy()
                    end_probs_i = end_probs[i][valid_mask].cpu().numpy()
                    
                    # Run two-stage localization
                    try:
                        pred_segments = two_stage_localization(
                            frame_probs_i, start_probs_i, end_probs_i,
                            thresholds=[0.3, 0.4, 0.5],
                            refine_delta=10,
                            min_len=5
                        )
                        # Get GT segments
                        gt_segments = get_segments_from_binary((frame_labels_i == 1).astype(int))
                        if pred_segments or gt_segments:
                            metrics_i = evaluate_segment_level(
                                pred_segments, gt_segments,
                                iou_thresholds=[0.3, 0.5, 0.7, 0.9]
                            )
                            segment_metrics_list.append(metrics_i)
                    except Exception as e:
                        # Skip if evaluation fails (avoid numpy overflow)
                        pass
                videos_seen += 1
        
        all_video_probs.extend(video_probs.cpu().numpy())
        all_video_labels.extend(video_labels.cpu().numpy())
        if stop_eval:
            break
    
    # Compute frame-level metrics
    frame_probs_flat = np.concatenate(all_frame_probs)
    frame_labels_flat = np.concatenate(all_frame_labels)
    
    frame_auc = roc_auc_score(frame_labels_flat, frame_probs_flat)
    frame_ap = average_precision_score(frame_labels_flat, frame_probs_flat)
    
    frame_preds = (frame_probs_flat > 0.5).astype(int)
    frame_f1 = f1_score(frame_labels_flat, frame_preds, zero_division=0)
    frame_precision = precision_score(frame_labels_flat, frame_preds, zero_division=0)
    frame_recall = recall_score(frame_labels_flat, frame_preds, zero_division=0)
    
    # Compute video-level metrics
    video_probs_arr = np.array(all_video_probs)
    video_labels_arr = np.array(all_video_labels)
    
    video_auc = roc_auc_score(video_labels_arr, video_probs_arr)
    video_ap = average_precision_score(video_labels_arr, video_probs_arr)
    
    # Enhanced metrics
    inconsistency_flat = np.concatenate(all_inconsistency_scores)
    gate_flat = np.concatenate(all_gate_values)
    
    # Mean inconsistency for real vs fake frames
    real_inc_mean = inconsistency_flat[frame_labels_flat == 0].mean()
    fake_inc_mean = inconsistency_flat[frame_labels_flat == 1].mean()
    inc_separation = fake_inc_mean - real_inc_mean  # Should be positive (fake > real)
    
    # Gate statistics
    gate_mean = gate_flat.mean()
    gate_std = gate_flat.std()
    
    # Segment-level metrics (per-video mean to avoid cross-video matching)
    segment_metrics = {}
    if len(segment_metrics_list) > 0:
        keys = segment_metrics_list[0].keys()
        segment_metrics = {
            k: float(np.mean([m.get(k, 0.0) for m in segment_metrics_list]))
            for k in keys
        }
    
    return {
        'frame_auc': frame_auc,
        'frame_ap': frame_ap,
        'frame_f1': frame_f1,
        'frame_precision': frame_precision,
        'frame_recall': frame_recall,
        'video_auc': video_auc,
        'video_ap': video_ap,
        'real_inc_mean': real_inc_mean,
        'fake_inc_mean': fake_inc_mean,
        'inc_separation': inc_separation,
        'gate_mean': gate_mean,
        'gate_std': gate_std,
        **segment_metrics
    }


def main():
    parser = argparse.ArgumentParser(description="Enhanced LAV-DF Localization Training")
    
    # Data
    parser.add_argument('--features_root', type=str, required=True)
    parser.add_argument('--splits', type=str, default='train,dev')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--max_frames', type=int, default=512)
    parser.add_argument('--event_centric_prob', type=float, default=0.5)
    
    # Enhanced Model
    parser.add_argument('--v_dim', type=int, default=512)
    parser.add_argument('--a_dim', type=int, default=1024)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--use_cross_attn', action='store_true', default=True)
    parser.add_argument('--use_video_head', action='store_true', default=True)
    parser.add_argument('--use_inconsistency_module', action='store_true', default=True)
    parser.add_argument('--use_reliability_gating', action='store_true', default=True)
    parser.add_argument('--use_boundary_head', action='store_true', default=True)
    parser.add_argument('--alpha_init', type=float, default=0.3)
    parser.add_argument('--temperature', type=float, default=0.5)
    
    # Loss weights
    parser.add_argument('--video_loss_weight', type=float, default=0.3)
    parser.add_argument('--boundary_loss_weight', type=float, default=0.5)
    parser.add_argument('--smooth_loss_weight', type=float, default=0.05)
    parser.add_argument('--ranking_loss_weight', type=float, default=0.1)
    parser.add_argument('--fake_hinge_weight', type=float, default=0.05)
    parser.add_argument('--ranking_margin', type=float, default=0.3)
    parser.add_argument('--boundary_tolerance', type=int, default=5)
    parser.add_argument('--use_boundary_aware_smooth', action='store_true', default=True)
    
    # Hard negatives
    parser.add_argument('--neg_shift_min', type=int, default=3)
    parser.add_argument('--neg_shift_max', type=int, default=10)
    parser.add_argument('--neg_swap_prob', type=float, default=0.5)
    
    # Frame loss
    parser.add_argument('--pos_weight', type=float, default=-1)
    parser.add_argument('--use_focal', action='store_true')
    parser.add_argument('--focal_alpha', type=float, default=0.25)
    parser.add_argument('--focal_gamma', type=float, default=2.0)
    
    # Training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)

    # Evaluation controls
    parser.add_argument('--eval_max_videos', type=int, default=0,
                        help='Limit number of videos for validation (0 = no limit)')
    parser.add_argument('--eval_stride', type=int, default=1,
                        help='Downsample frames during validation metrics (1 = no downsample)')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='./checkpoints/localization_v2')
    parser.add_argument('--save_every', type=int, default=5)
    parser.add_argument('--resume', type=str, default=None)
    
    args = parser.parse_args()
    
    # Setup DDP
    rank, world_size, local_rank = setup_ddp()
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    
    if rank == 0:
        print("=" * 60)
        print("Enhanced LAV-DF Localization Training V2")
        print("=" * 60)
        print(f"Features root: {args.features_root}")
        print(f"Batch size: {args.batch_size}")
        print(f"Epochs: {args.epochs}")
        print(f"Ranking loss weight: {args.ranking_loss_weight}")
        print(f"Fake hinge weight: {args.fake_hinge_weight}")
        print(f"Device: {device}")
        print(f"World size: {world_size}")
        print("=" * 60)
    
    # Create output directory
    if rank == 0:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(args.output_dir) / 'args.json', 'w') as f:
            json.dump(vars(args), f, indent=2)
    
    # Create dataloaders
    loaders = get_dataloaders(
        features_root=args.features_root,
        splits=args.splits.split(','),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_frames=args.max_frames,
        event_centric_prob=args.event_centric_prob,
        distributed=(world_size > 1)
    )
    
    # Extract train and dev loaders
    train_loader = loaders.get('train')
    val_loader = loaders.get('dev')
    
    # Create enhanced model
    model = FrameLocalizationModel(
        v_dim=args.v_dim,
        a_dim=args.a_dim,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dropout=args.dropout,
        use_cross_attn=args.use_cross_attn,
        use_video_head=args.use_video_head,
        use_inconsistency_module=args.use_inconsistency_module,
        use_reliability_gating=args.use_reliability_gating,
        use_boundary_head=args.use_boundary_head,
        alpha_init=args.alpha_init,
        temperature=args.temperature
    ).to(device)
    
    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"Model parameters: {total_params:.2f}M")
        print(f"Inconsistency module: {args.use_inconsistency_module}")
        print(f"Reliability gating: {args.use_reliability_gating}")
        print(f"Boundary head: {args.use_boundary_head}")
        print(f"Event-centric sampling prob: {args.event_centric_prob}")
    
    # Wrap with DDP
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
    
    # Auto pos_weight
    if args.pos_weight < 0 and rank == 0:
        print("Computing automatic pos_weight from train set...")
        total_frames = 0
        total_fake = 0
        for batch in train_loader:
            labels = batch['frame_labels']
            mask = batch['mask']
            valid_labels = labels[~mask]
            total_frames += valid_labels.numel()
            total_fake += valid_labels.sum().item()
        
        if total_fake > 0:
            args.pos_weight = (total_frames - total_fake) / total_fake
            print(f"Auto pos_weight: {args.pos_weight:.2f}")
        else:
            args.pos_weight = 1.0
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler()
    
    # Learning rate scheduler with warmup
    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        else:
            return 0.5 * (1 + np.cos(np.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Resume if specified
    start_epoch = 0
    best_frame_auc = 0
    
    if args.resume and os.path.exists(args.resume):
        if rank == 0:
            print(f"Resuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_frame_auc = checkpoint.get('best_frame_auc', 0)
    
    # Training loop
    for epoch in range(start_epoch, args.epochs):
        if rank == 0:
            print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, scaler, device, epoch + 1, args, rank)
        
        if rank == 0:
            print(f"Train - Loss: {train_metrics['loss']:.4f}, "
                  f"Frame: {train_metrics['frame_loss']:.4f}, "
                  f"Boundary: {train_metrics['boundary_loss']:.4f}, "
                  f"Ranking: {train_metrics['ranking_loss']:.4f}")
        
        # Evaluate
        if val_loader is not None:
            if rank == 0:
                val_metrics = evaluate(model, val_loader, device, args, rank, do_segment_eval=True)
            else:
                val_metrics = {}
            
            if rank == 0 and val_metrics:
                print(f"Val - Frame AUC: {val_metrics['frame_auc']:.4f}, "
                      f"AP: {val_metrics['frame_ap']:.4f}, "
                      f"F1: {val_metrics['frame_f1']:.4f}")
                
                # Segment-level metrics (if boundary head is used)
                if 'AP@0.5' in val_metrics:
                    print(f"      Segment AP@0.5: {val_metrics.get('AP@0.5', 0.0):.4f}, "
                          f"AP@0.75: {val_metrics.get('AP@0.75', 0.0):.4f}, "
                          f"mAP: {val_metrics.get('mAP', 0.0):.4f}")
                
                print(f"      Inc separation: {val_metrics['inc_separation']:.4f}, "
                      f"Gate: {val_metrics['gate_mean']:.3f}±{val_metrics['gate_std']:.3f}")
                
                # Save best model
                if val_metrics['frame_auc'] > best_frame_auc:
                    best_frame_auc = val_metrics['frame_auc']
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.module.state_dict() if world_size > 1 else model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'metrics': val_metrics,
                        'best_frame_auc': best_frame_auc,
                        'args': vars(args)
                    }, Path(args.output_dir) / 'best.pth')
                    print(f"✓ New best Frame AUC: {best_frame_auc:.4f}")
        
        # Save checkpoint
        if rank == 0 and (epoch + 1) % args.save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.module.state_dict() if world_size > 1 else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_frame_auc': best_frame_auc,
                'args': vars(args)
            }, Path(args.output_dir) / f'checkpoint_epoch{epoch + 1}.pth')
        
        scheduler.step()
    
    # Save final model
    if rank == 0:
        torch.save({
            'epoch': args.epochs - 1,
            'model_state_dict': model.module.state_dict() if world_size > 1 else model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_frame_auc': best_frame_auc,
            'args': vars(args)
        }, Path(args.output_dir) / 'final.pth')
        print(f"\n✓ Training complete! Best Frame AUC: {best_frame_auc:.4f}")
    
    cleanup_ddp()


if __name__ == '__main__':
    main()

