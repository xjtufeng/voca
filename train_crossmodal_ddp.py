"""
Multi-GPU Distributed Training for Cross-Modal Deepfake Detection
Transformer + Cross-Attention + Contrastive Learning

Usage:
    torchrun --nproc_per_node=4 train_crossmodal_ddp.py \
        --features_root /path/to/features \
        --batch_size 64 --epochs 50 --seq_len 256 --hidden 768
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Tuple
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch import amp

try:
    from sklearn.metrics import accuracy_score, roc_auc_score
    _HAS_SKLEARN = True
except:
    _HAS_SKLEARN = False


# ==================== Data Loading ====================
def list_samples(features_root: Path) -> List[Tuple[Path, int]]:
    samples = []
    for label_name, label in [("real", 1), ("fake", 0)]:
        label_dir = features_root / label_name
        if not label_dir.exists():
            continue
        for vid_dir in sorted(label_dir.iterdir()):
            if not vid_dir.is_dir():
                continue
            v_npz = vid_dir / "visual_embeddings.npz"
            a_npz = vid_dir / "audio_embeddings.npz"
            if v_npz.exists() and a_npz.exists():
                samples.append((vid_dir, label))
    return samples


def compute_pos_weight(samples: List[Tuple[Path, int]]) -> float:
    """计算正样本权重 pos_weight = #fake / #real"""
    n_real = sum(1 for _, lab in samples if lab == 1)
    n_fake = sum(1 for _, lab in samples if lab == 0)
    if n_real == 0:
        return 1.0
    return max(1.0, n_fake / n_real)


def split_dataset_stratified(samples: List[Tuple[Path, int]], train_ratio=0.6, val_ratio=0.2):
    """分层划分，确保每个split都有正负样本"""
    from random import shuffle
    real = [s for s in samples if s[1] == 1]
    fake = [s for s in samples if s[1] == 0]
    
    shuffle(real)
    shuffle(fake)
    
    def split_one(cls_list):
        n = len(cls_list)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        return (cls_list[:n_train],
                cls_list[n_train:n_train+n_val],
                cls_list[n_train+n_val:])
    
    real_train, real_val, real_test = split_one(real)
    fake_train, fake_val, fake_test = split_one(fake)
    
    train = real_train + fake_train
    val = real_val + fake_val
    test = real_test + fake_test
    
    shuffle(train)
    shuffle(val)
    shuffle(test)
    
    return train, val, test


class AVFeatDataset(Dataset):
    def __init__(self, items: List[Tuple[Path, int]], seq_len: int, train: bool):
        self.items = items
        self.seq_len = seq_len
        self.train = train
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        vid_dir, label = self.items[idx]
        
        visual = np.load(vid_dir / "visual_embeddings.npz")["embeddings"]
        audio = np.load(vid_dir / "audio_embeddings.npz")["embeddings"]
        
        # 对齐并裁剪
        t = min(len(visual), len(audio))
        visual = visual[:t]
        audio = audio[:t]
        
        # 时序增强（训练时随机偏移）
        if self.train and np.random.rand() < 0.2:
            shift = np.random.randint(-3, 4)
            if shift > 0:
                audio = np.roll(audio, shift, axis=0)
            elif shift < 0:
                visual = np.roll(visual, -shift, axis=0)
        
        # 截取或填充到seq_len
        if t >= self.seq_len:
            if self.train:
                start = np.random.randint(0, t - self.seq_len + 1)
            else:
                start = max(0, (t - self.seq_len) // 2)
            visual = visual[start:start+self.seq_len]
            audio = audio[start:start+self.seq_len]
        else:
            v_pad = np.tile(visual[-1:], (self.seq_len - t, 1))
            a_pad = np.tile(audio[-1:], (self.seq_len - t, 1))
            visual = np.concatenate([visual, v_pad], axis=0)
            audio = np.concatenate([audio, a_pad], axis=0)
        
        visual = torch.from_numpy(visual).float()
        audio = torch.from_numpy(audio).float()
        label_t = torch.tensor(label, dtype=torch.float32)
        
        return visual, audio, label_t


# ==================== Model ====================
class CrossModalTransformer(nn.Module):
    """
    Transformer + Cross-Attention + Contrastive Learning
    """
    def __init__(
        self, 
        dv: int = 512, 
        da: int = 512, 
        hidden: int = 768, 
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # 投影到统一维度
        self.v_proj = nn.Sequential(
            nn.Linear(dv, hidden),
            nn.LayerNorm(hidden),
            nn.Dropout(dropout)
        )
        self.a_proj = nn.Sequential(
            nn.Linear(da, hidden),
            nn.LayerNorm(hidden),
            nn.Dropout(dropout)
        )

        # 双向 Cross-Attention
        # 音频->视觉
        self.a2v_attn = nn.MultiheadAttention(
            embed_dim=hidden, 
            num_heads=num_heads, 
            dropout=dropout, 
            batch_first=True
        )
        # 视觉->音频
        self.v2a_attn = nn.MultiheadAttention(
            embed_dim=hidden,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.a2v_norm = nn.LayerNorm(hidden)
        self.v2a_norm = nn.LayerNorm(hidden)

        # Temporal Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=num_heads,
            dim_feedforward=hidden * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1)
        )
        
        # 对比学习投影头
        self.contrast_proj = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 256)
        )
    
    def forward(self, v, a, return_contrast=False):
        """
        Args:
            v: (B, T, dv)
            a: (B, T, da)
            return_contrast: bool
        Returns:
            logits: (B,)
            (optional) v_contrast, a_contrast: (B, 256)
        """
        # 投影
        v_feat = self.v_proj(v)  # (B, T, H)
        a_feat = self.a_proj(a)  # (B, T, H)
        
        # 双向 Cross-Attention
        # 音频查询视觉
        a2v, _ = self.a2v_attn(a_feat, v_feat, v_feat)
        a2v = self.a2v_norm(a2v + a_feat)
        # 视觉查询音频
        v2a, _ = self.v2a_attn(v_feat, a_feat, a_feat)
        v2a = self.v2a_norm(v2a + v_feat)

        # 融合：简单平均
        fused = 0.5 * (a2v + v2a)  # (B, T, H)
        
        # Temporal modeling
        temporal_feat = self.temporal_encoder(fused)  # (B, T, H)
        
        # 池化：mean + max
        pooled_mean = temporal_feat.mean(dim=1)
        pooled_max = temporal_feat.max(dim=1)[0]
        pooled = pooled_mean + pooled_max  # (B, H)
        
        # 分类
        logits = self.classifier(pooled).squeeze(-1)  # (B,)
        
        if return_contrast:
            # 对比学习投影
            v_contrast = self.contrast_proj(v_feat.mean(dim=1))
            a_contrast = self.contrast_proj(a_feat.mean(dim=1))
            return logits, v_contrast, a_contrast
        
        return logits


def contrastive_loss(v_proj, a_proj, labels, temperature=0.07):
    """
    InfoNCE对比学习损失
    """
    # L2归一化
    v_proj = F.normalize(v_proj, dim=1)
    a_proj = F.normalize(a_proj, dim=1)
    
    # 相似度矩阵
    sim_matrix = torch.matmul(v_proj, a_proj.T) / temperature  # (B, B)
    
    # 正对：同标签
    labels = labels.unsqueeze(1)
    pos_mask = (labels == labels.T).float()
    
    # 对角线权重更高（同一样本的跨模态）
    diag_mask = torch.eye(len(labels), device=labels.device)
    pos_mask = pos_mask * (1 - diag_mask) + diag_mask * 2.0
    
    # InfoNCE
    exp_sim = torch.exp(sim_matrix)
    pos_sim = (exp_sim * pos_mask).sum(dim=1)
    all_sim = exp_sim.sum(dim=1)
    
    loss = -torch.log(pos_sim / (all_sim + 1e-8)).mean()
    return loss


# ==================== Training ====================
def train_epoch(model, loader, optimizer, scaler, device, use_contrastive, contrastive_weight, label_smoothing, pos_weight=None):
    model.train()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_con_loss = 0.0
    
    for visual, audio, labels in loader:
        visual = visual.to(device)
        audio = audio.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        with amp.autocast(device_type="cuda"):
            if use_contrastive:
                logits, v_c, a_c = model(visual, audio, return_contrast=True)
                
                # 分类损失
                if label_smoothing > 0:
                    # Label smoothing
                    labels_smooth = labels * (1 - label_smoothing) + 0.5 * label_smoothing
                    loss_cls = F.binary_cross_entropy_with_logits(
                        logits, labels_smooth, pos_weight=pos_weight
                    )
                else:
                    loss_cls = F.binary_cross_entropy_with_logits(
                        logits, labels, pos_weight=pos_weight
                    )
                
                # 对比损失
                loss_con = contrastive_loss(v_c, a_c, labels, temperature=0.07)
                
                loss = loss_cls + contrastive_weight * loss_con
                
                total_cls_loss += loss_cls.item()
                total_con_loss += loss_con.item()
            else:
                logits = model(visual, audio, return_contrast=False)
                loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)
        
        scaler.scale(loss).backward()
        
        # 梯度裁剪
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
    
    n_batches = len(loader)
    avg_loss = total_loss / max(1, n_batches)
    avg_cls = total_cls_loss / max(1, n_batches) if use_contrastive else avg_loss
    avg_con = total_con_loss / max(1, n_batches) if use_contrastive else 0.0
    
    return avg_loss, avg_cls, avg_con


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_logits = []
    all_labels = []
    
    for visual, audio, labels in loader:
        visual = visual.to(device)
        audio = audio.to(device)
        logits = model(visual, audio, return_contrast=False)
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())
    
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).float()
    
    acc = (preds == labels).float().mean().item()
    auc = None
    if _HAS_SKLEARN and len(labels.unique()) > 1:
        try:
            auc = roc_auc_score(labels.numpy(), probs.numpy())
        except:
            pass
    
    return acc, auc


def setup_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_ddp():
    dist.destroy_process_group()


def main_worker(rank, world_size, args):
    setup_ddp(rank, world_size)
    
    if rank == 0:
        print("="*60)
        print("Cross-Modal Transformer DDP Training")
        print("="*60)
        print(f"World size: {world_size}")
        print(f"Batch size per GPU: {args.batch_size}")
        print(f"Total batch size: {args.batch_size * world_size}")
        print(f"Epochs: {args.epochs}")
        print(f"Seq len: {args.seq_len}")
        print(f"Hidden dim: {args.hidden}")
        print(f"Num layers: {args.num_layers}")
        print(f"Contrastive: {args.use_contrastive}")
        print("="*60)
    
    # 加载数据
    features_root = Path(args.features_root)
    samples = list_samples(features_root)
    # 计算 pos_weight（正样本权重）
    pos_w_value = compute_pos_weight(samples)
    pos_weight_tensor = torch.tensor(pos_w_value, device=rank)
    
    if rank == 0:
        print(f"[INFO] Found {len(samples)} samples")
    
    if not samples:
        if rank == 0:
            print(f"[ERROR] No samples found")
        cleanup_ddp()
        return
    
    train_set, val_set, test_set = split_dataset_stratified(samples, 0.6, 0.2)
    
    if rank == 0:
        print(f"[INFO] Split: train={len(train_set)}, val={len(val_set)}, test={len(test_set)}")
        print(f"[INFO] pos_weight (fake/real) = {pos_w_value:.2f}")
    
    # Dataset & DataLoader
    train_dataset = AVFeatDataset(train_set, args.seq_len, train=True)
    val_dataset = AVFeatDataset(val_set, args.seq_len, train=False)
    test_dataset = AVFeatDataset(test_set, args.seq_len, train=False)
    
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=train_sampler,
        num_workers=args.num_workers, pin_memory=args.pin_memory, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, sampler=val_sampler,
        num_workers=args.num_workers, pin_memory=args.pin_memory
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, sampler=test_sampler,
        num_workers=args.num_workers, pin_memory=args.pin_memory
    )
    
    # 模型
    v_dim = 512  # 假设已知
    a_dim = 512
    
    model = CrossModalTransformer(
        dv=v_dim, da=a_dim, hidden=args.hidden,
        num_layers=args.num_layers, num_heads=args.num_heads, dropout=args.dropout
    ).to(rank)
    
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    
    # 优化器 & 调度器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Warmup + Cosine
    from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=args.warmup_epochs)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs - args.warmup_epochs, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[args.warmup_epochs])
    
    scaler = amp.GradScaler("cuda")
    
    best_val_acc = 0.0
    best_epoch = 0
    
    for epoch in range(1, args.epochs + 1):
        train_sampler.set_epoch(epoch)
        
        avg_loss, avg_cls, avg_con = train_epoch(
            model, train_loader, optimizer, scaler, rank,
            args.use_contrastive, args.contrastive_weight, args.label_smoothing,
            pos_weight=pos_weight_tensor
        )
        
        val_acc, val_auc = evaluate(model, val_loader, rank)
        
        scheduler.step()
        
        if rank == 0:
            if args.use_contrastive:
                print(f"[Epoch {epoch}/{args.epochs}] loss={avg_loss:.4f} cls={avg_cls:.4f} con={avg_con:.4f} "
                      f"val_acc={val_acc*100:.2f}% val_auc={val_auc if val_auc else 'NA'}")
            else:
                print(f"[Epoch {epoch}/{args.epochs}] loss={avg_loss:.4f} "
                      f"val_acc={val_acc*100:.2f}% val_auc={val_auc if val_auc else 'NA'}")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                if args.save_path:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_acc': val_acc,
                        'val_auc': val_auc,
                        'config': vars(args)
                    }, args.save_path)
    
    if rank == 0:
        print(f"\n[INFO] Best val_acc={best_val_acc*100:.2f}% at epoch {best_epoch}")
        
        # 测试集评估
        if args.save_path and os.path.exists(args.save_path):
            ckpt = torch.load(args.save_path, map_location=f'cuda:{rank}')
            model.module.load_state_dict(ckpt['model_state_dict'])
        
        test_acc, test_auc = evaluate(model, test_loader, rank)
        print(f"[TEST] acc={test_acc*100:.2f}% auc={test_auc if test_auc else 'NA'}")
    
    cleanup_ddp()


def main():
    parser = argparse.ArgumentParser(description="DDP Cross-Modal Transformer Training")
    parser.add_argument("--features_root", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size per GPU")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=768)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_epochs", type=int, default=3)
    parser.add_argument("--use_contrastive", action="store_true", help="Use contrastive loss")
    parser.add_argument("--contrastive_weight", type=float, default=0.5)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--save_path", type=str, default="best_crossmodal_ddp.pt")
    
    args = parser.parse_args()
    
    world_size = torch.cuda.device_count()
    if world_size < 1:
        print("[ERROR] No GPU available")
        return
    
    torch.multiprocessing.spawn(
        main_worker,
        args=(world_size, args),
        nprocs=world_size,
        join=True
    )


if __name__ == "__main__":
    main()

