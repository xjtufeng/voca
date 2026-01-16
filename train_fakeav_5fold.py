#!/usr/bin/env python3
"""
5-Fold Cross-Validation Training for FakeAVCeleb (MRDF Protocol)
- 4-class classification (FAFV/FARV/RAFV/RARV)
- Identity-independent splits
- 1:1:1:1 balanced sampling
- Report mean ACC/AUC across 5 folds
"""
import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

from dataset_fakeav_fourclass import get_fakeav_dataloaders
from model_three_branch import ThreeBranchJointModel


def train_epoch(
    model: nn.Module,
    loader,
    optimizer,
    scaler,
    device,
    epoch: int
) -> Dict[str, float]:
    """Train for one epoch"""
    model.train()
    
    total_loss = 0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]")
    
    for batch in pbar:
        audio = batch['audio'].to(device)
        visual = batch['visual'].to(device)
        labels = batch['label'].to(device)  # [B], values in {0,1,2,3}
        mask = batch['mask'].to(device)
        
        optimizer.zero_grad()
        
        with autocast(enabled=(device.type == "cuda")):
            # Forward: model outputs fused logits [B, num_classes]
            outputs = model(
                audio=audio,
                visual=visual,
                video_frames=None,
                mask=mask,
                return_branch_outputs=False
            )
            
            fused_logits = outputs['fused']  # [B, 4]
            
            # Cross-entropy loss for 4-class classification
            loss = nn.CrossEntropyLoss()(fused_logits, labels)
        
        # Backward
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Metrics
        total_loss += loss.item()
        preds = torch.argmax(fused_logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    # Compute metrics
    acc = accuracy_score(all_labels, all_preds)
    
    return {
        'loss': total_loss / len(loader),
        'accuracy': acc
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    device
) -> Dict[str, float]:
    """Evaluate model"""
    model.eval()
    
    all_probs = []  # [N, 4] probabilities
    all_labels = []  # [N] ground truth
    
    for batch in tqdm(loader, desc="Evaluating"):
        audio = batch['audio'].to(device)
        visual = batch['visual'].to(device)
        labels = batch['label'].to(device)
        mask = batch['mask'].to(device)
        
        outputs = model(
            audio=audio,
            visual=visual,
            video_frames=None,
            mask=mask,
            return_branch_outputs=False
        )
        
        fused_logits = outputs['fused']
        probs = torch.softmax(fused_logits, dim=1)  # [B, 4]
        
        all_probs.append(probs.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    all_probs = np.concatenate(all_probs, axis=0)  # [N, 4]
    all_labels = np.array(all_labels)  # [N]
    
    # Predictions
    preds = np.argmax(all_probs, axis=1)
    
    # Metrics
    acc = accuracy_score(all_labels, preds)
    
    # Multi-class AUC (one-vs-rest)
    try:
        auc_ovr = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
    except:
        auc_ovr = 0.0
    
    # Per-class metrics
    per_class_acc = []
    for cls in range(4):
        mask = all_labels == cls
        if mask.sum() > 0:
            cls_acc = accuracy_score(all_labels[mask], preds[mask])
            per_class_acc.append(cls_acc)
        else:
            per_class_acc.append(0.0)
    
    return {
        'accuracy': acc,
        'auc': auc_ovr,
        'per_class_acc': per_class_acc
    }


def train_one_fold(fold_id: int, args) -> Dict[str, float]:
    """Train and evaluate one fold"""
    print("=" * 70)
    print(f"TRAINING FOLD {fold_id}")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create dataloaders
    dataloaders = get_fakeav_dataloaders(
        features_root=args.features_root,
        fold_id=fold_id,
        splits_dir=args.splits_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_frames=args.max_frames,
        balanced_train=True  # 1:1:1:1 balanced sampling
    )
    
    train_loader = dataloaders['train']
    test_loader = dataloaders['test']
    
    # Create model (4-class output)
    model = ThreeBranchJointModel(
        audio_dim=args.audio_dim,
        visual_dim=args.visual_dim,
        d_model=args.d_model,
        cm_layers=args.cm_layers,
        ao_layers=args.ao_layers,
        vo_layers=args.vo_layers,
        nhead=args.nhead,
        dropout=args.dropout,
        num_classes=4,  # IMPORTANT: 4-class
        fusion_method=args.fusion_method,
        use_clip=False
    ).to(device)
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs
    )
    
    scaler = GradScaler(enabled=(device.type == "cuda"))
    
    # Training loop
    best_acc = 0.0
    best_epoch = 0
    best_metrics = {}
    
    for epoch in range(args.epochs):
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, scaler, device, epoch
        )
        
        # Evaluate
        test_metrics = evaluate(model, test_loader, device)
        
        scheduler.step()
        
        # Print
        print(f"\n[Epoch {epoch}]")
        print(f"  Train - Loss: {train_metrics['loss']:.4f}, ACC: {train_metrics['accuracy']:.4f}")
        print(f"  Test  - ACC: {test_metrics['accuracy']:.4f}, AUC: {test_metrics['auc']:.4f}")
        print(f"          Per-class ACC: FAFV={test_metrics['per_class_acc'][0]:.3f}, "
              f"FARV={test_metrics['per_class_acc'][1]:.3f}, "
              f"RAFV={test_metrics['per_class_acc'][2]:.3f}, "
              f"RARV={test_metrics['per_class_acc'][3]:.3f}")
        
        # Save best
        if test_metrics['accuracy'] > best_acc:
            best_acc = test_metrics['accuracy']
            best_epoch = epoch
            best_metrics = test_metrics
            
            # Save checkpoint
            ckpt_dir = Path(args.output_dir) / f'fold_{fold_id}'
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': test_metrics
            }, ckpt_dir / 'best.pth')
    
    print(f"\n✅ Fold {fold_id} Best: Epoch {best_epoch}, ACC={best_acc:.4f}, AUC={best_metrics['auc']:.4f}")
    
    return best_metrics


def main():
    parser = argparse.ArgumentParser(description='5-Fold CV Training for FakeAVCeleb')
    
    # Data
    parser.add_argument('--features_root', type=str, required=True,
                        help='Path to FakeAV_feats')
    parser.add_argument('--splits_dir', type=str, required=True,
                        help='Directory with 5-fold split JSON files')
    parser.add_argument('--max_frames', type=int, default=256)
    
    # Model
    parser.add_argument('--audio_dim', type=int, default=512)
    parser.add_argument('--visual_dim', type=int, default=512)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--cm_layers', type=int, default=4)
    parser.add_argument('--ao_layers', type=int, default=4)
    parser.add_argument('--vo_layers', type=int, default=4)
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--fusion_method', type=str, default='weighted',
                        choices=['concat', 'average', 'weighted'])
    
    # Training
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--num_workers', type=int, default=4)
    
    # Output
    parser.add_argument('--output_dir', type=str, default='checkpoints/fakeav_5fold')
    
    # Folds to run
    parser.add_argument('--folds', type=int, nargs='+', default=[0, 1, 2, 3, 4],
                        help='Which folds to train (default: all 5)')
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Train each fold
    fold_results = []
    
    for fold_id in args.folds:
        fold_metrics = train_one_fold(fold_id, args)
        fold_results.append({
            'fold': fold_id,
            'accuracy': fold_metrics['accuracy'],
            'auc': fold_metrics['auc'],
            'per_class_acc': fold_metrics['per_class_acc']
        })
    
    # Aggregate results
    print("\n" + "=" * 70)
    print("5-FOLD CROSS-VALIDATION RESULTS")
    print("=" * 70)
    
    accs = [r['accuracy'] for r in fold_results]
    aucs = [r['auc'] for r in fold_results]
    
    print(f"\nPer-Fold Results:")
    for r in fold_results:
        print(f"  Fold {r['fold']}: ACC={r['accuracy']:.4f}, AUC={r['auc']:.4f}")
    
    print(f"\nMean ± Std:")
    print(f"  ACC: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"  AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    
    # Per-class aggregation
    per_class_accs = np.array([r['per_class_acc'] for r in fold_results])  # [num_folds, 4]
    mean_per_class = per_class_accs.mean(axis=0)
    std_per_class = per_class_accs.std(axis=0)
    
    print(f"\nPer-Class ACC (mean ± std):")
    for i, name in enumerate(['FAFV', 'FARV', 'RAFV', 'RARV']):
        print(f"  {name}: {mean_per_class[i]:.4f} ± {std_per_class[i]:.4f}")
    
    # Save results
    results_file = Path(args.output_dir) / '5fold_results.json'
    with open(results_file, 'w') as f:
        json.dump({
            'fold_results': fold_results,
            'mean_acc': float(np.mean(accs)),
            'std_acc': float(np.std(accs)),
            'mean_auc': float(np.mean(aucs)),
            'std_auc': float(np.std(aucs)),
            'per_class_acc_mean': mean_per_class.tolist(),
            'per_class_acc_std': std_per_class.tolist()
        }, f, indent=2)
    
    print(f"\n✅ Results saved to {results_file}")
    print("=" * 70)


if __name__ == '__main__':
    main()

