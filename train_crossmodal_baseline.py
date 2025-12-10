from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    from sklearn.metrics import accuracy_score, roc_auc_score
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False


def list_samples(features_root: Path) -> List[Tuple[Path, int]]:
    
    samples: List[Tuple[Path, int]] = []
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


def _load_npz_embeddings(npz_path: Path) -> np.ndarray:
    data = np.load(npz_path)
    if "embeddings" not in data:
        raise KeyError(f"'embeddings' not found in {npz_path}")
    return data["embeddings"]


def _align_and_crop(
    visual: np.ndarray,
    audio: np.ndarray,
    seq_len: int,
    train: bool,
) -> Tuple[np.ndarray, np.ndarray]:
   
    t = min(len(visual), len(audio))
    visual = visual[:t]
    audio = audio[:t]

    if t >= seq_len:
        if train:
            start = np.random.randint(0, t - seq_len + 1)
        else:
            start = max(0, (t - seq_len) // 2)
        end = start + seq_len
        visual = visual[start:end]
        audio = audio[start:end]
    else:
        # 填充到 seq_len
        v_pad = np.tile(visual[-1:], (seq_len - t, 1))
        a_pad = np.tile(audio[-1:], (seq_len - t, 1))
        visual = np.concatenate([visual, v_pad], axis=0)
        audio = np.concatenate([audio, a_pad], axis=0)

    return visual, audio


@dataclass
class AVItem:
    visual: torch.Tensor
    audio: torch.Tensor
    label: torch.Tensor


class AVFeatDataset(Dataset):
    def __init__(self, items: List[Tuple[Path, int]], seq_len: int, train: bool):
        self.items = items
        self.seq_len = seq_len
        self.train = train

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        vid_dir, label = self.items[idx]
        visual = _load_npz_embeddings(vid_dir / "visual_embeddings.npz")
        audio = _load_npz_embeddings(vid_dir / "audio_embeddings.npz")
        visual, audio = _align_and_crop(visual, audio, self.seq_len, self.train)
        # 转 torch
        visual = torch.from_numpy(visual).float()
        audio = torch.from_numpy(audio).float()
        label_t = torch.tensor(label, dtype=torch.float32)
        return AVItem(visual=visual, audio=audio, label=label_t)


class CrossModalBaseline(nn.Module):
    """
    轻量跨模态基线：对齐序列 -> 线性投影 -> 逐帧 MLP 打分 -> 时间平均。
    """

    def __init__(self, dv: int, da: int, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.v_proj = nn.Sequential(
            nn.Linear(dv, hidden),
            nn.LayerNorm(hidden),
        )
        self.a_proj = nn.Sequential(
            nn.Linear(da, hidden),
            nn.LayerNorm(hidden),
        )
        self.mlp = nn.Sequential(
            nn.Linear(hidden * 3, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, v: torch.Tensor, a: torch.Tensor):
        # v, a: (B, T, D)
        v = self.v_proj(v)
        a = self.a_proj(a)
        diff = torch.abs(v - a)
        x = torch.cat([v, a, diff], dim=-1)  # (B, T, 3H)
        logits_t = self.mlp(x).squeeze(-1)   # (B, T)
        logits = logits_t.mean(dim=1)        # 段级
        sim = F.cosine_similarity(v, a, dim=-1, eps=1e-8).mean(dim=1)
        return logits, sim



    def __init__(self, dv: int, da: int, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.v_proj = nn.Sequential(
            nn.Linear(dv, hidden),
            nn.LayerNorm(hidden),
        )
        self.a_proj = nn.Sequential(
            nn.Linear(da, hidden),
            nn.LayerNorm(hidden),
        )
        self.mlp = nn.Sequential(
            nn.Linear(hidden * 3, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, v: torch.Tensor, a: torch.Tensor):
        """
      
        """
        v = self.v_proj(v)
        a = self.a_proj(a)
      
        diff = torch.abs(v - a)
        x = torch.cat([v, a, diff], dim=-1)  # (B,T,3H)
        logits_t = self.mlp(x).squeeze(-1)   # (B,T)
        logits = logits_t.mean(dim=1)        # 段平均
        sim = F.cosine_similarity(v, a, dim=-1, eps=1e-8).mean(dim=1)
        return logits, sim


def split_dataset(samples: List[Tuple[Path, int]], train_ratio=0.6, val_ratio=0.2):
    n = len(samples)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train = samples[:n_train]
    val = samples[n_train:n_train + n_val]
    test = samples[n_train + n_val:]
    return train, val, test


def evaluate(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            v = batch.visual.to(device)
            a = batch.audio.to(device)
            labels = batch.label.to(device)
            logits, _ = model(v, a)
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).float()
    acc = (preds == labels).float().mean().item()
    auc = None
    if _HAS_SKLEARN:
        try:
            auc = roc_auc_score(labels.numpy(), probs.numpy())
        except Exception:
            auc = None
    return acc, auc


def train(args):
    device = torch.device(args.device)
    features_root = Path(args.features_root)
    samples = list_samples(features_root)
    if not samples:
        raise FileNotFoundError(f"No samples found under {features_root}/(real|fake)")
    print(f"[INFO] Found {len(samples)} samples")

  
    train_set, val_set, test_set = split_dataset(samples, train_ratio=0.6, val_ratio=0.2)
    print(f"[INFO] Split: train={len(train_set)}, val={len(val_set)}, test={len(test_set)}")

    v_dim = _load_npz_embeddings(train_set[0][0] / "visual_embeddings.npz").shape[1]
    a_dim = _load_npz_embeddings(train_set[0][0] / "audio_embeddings.npz").shape[1]
    print(f"[INFO] Feature dims: visual={v_dim}, audio={a_dim}")

    def collate_fn(batch: List[AVItem]):
        visuals = torch.stack([b.visual for b in batch], dim=0)
        audios = torch.stack([b.audio for b in batch], dim=0)
        labels = torch.stack([b.label for b in batch], dim=0)
        return AVItem(visual=visuals, audio=audios, label=labels)

    train_loader = DataLoader(
        AVFeatDataset(train_set, seq_len=args.seq_len, train=True),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        AVFeatDataset(val_set, seq_len=args.seq_len, train=False),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        AVFeatDataset(test_set, seq_len=args.seq_len, train=False),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    model = CrossModalBaseline(dv=v_dim, da=a_dim, hidden=args.hidden, dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    best_val_acc = 0.0
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            v = batch.visual.to(device)
            a = batch.audio.to(device)
            labels = batch.label.to(device)
            logits, _ = model(v, a)
            loss = criterion(logits, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        avg_loss = total_loss / max(1, len(train_loader))
        val_acc, val_auc = evaluate(model, val_loader, device)
        print(f"[Epoch {epoch}] loss={avg_loss:.4f} val_acc={val_acc*100:.2f}% val_auc={val_auc if val_auc else 'NA'}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict()

    if best_state is not None:
        model.load_state_dict(best_state)

    test_acc, test_auc = evaluate(model, test_loader, device)
    print(f"[TEST] acc={test_acc*100:.2f}% auc={test_auc if test_auc else 'NA'}")

    if args.save_path:
        torch.save({"state_dict": model.state_dict(), "config": vars(args)},
                   args.save_path)
        print(f"[INFO] Model saved to {args.save_path}")


def parse_args():
    p = argparse.ArgumentParser(description="CPU-friendly cross-modal baseline training (FakeAV)")
    p.add_argument("--features_root", type=str, required=True, help="Root dir produced by prepare_features_dataset.py")
    p.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    p.add_argument("--seq_len", type=int, default=128, help="Aligned sequence length per sample")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--save_path", type=str, default="crossmodal_baseline.pt")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)



