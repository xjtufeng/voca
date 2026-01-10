#!/usr/bin/env python3
"""
Standalone evaluation script for Three-Branch Joint Model.

This script loads a checkpoint produced by train_three_branch.py and evaluates it on
one or more dataset splits (e.g., dev/test). It reports:
- AUC, Accuracy, F1, Precision, Recall, Specificity, Balanced Accuracy, PR-AUC
- Confusion matrix counts (TP/TN/FP/FN)
- Fusion weight mean/std (if available)

It also optionally saves per-sample predictions to a CSV for further analysis.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    balanced_accuracy_score,
)

from dataset_three_branch import get_three_branch_dataloaders
from model_three_branch import ThreeBranchJointModel


@torch.no_grad()
def evaluate_split(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    model.eval()

    preds: Dict[str, list] = {"fused": [], "cm": [], "ao": [], "vo": []}
    labels: list = []
    video_ids: list = []
    fusion_weights: list = []

    for batch in loader:
        audio = batch["audio"].to(device)
        visual = batch["visual"].to(device)
        y = batch["label"].to(device)
        mask = batch["mask"].to(device)

        # Optional: frames for CLIP branch
        video_frames = batch.get("video_frames")
        if video_frames is not None:
            video_frames = video_frames.to(device)

        outputs = model(
            audio=audio,
            visual=visual,
            video_frames=video_frames,
            mask=mask,
            return_branch_outputs=True,
        )

        fused_p = torch.sigmoid(outputs["fused_logit"]).squeeze(-1).cpu().numpy()
        preds["fused"].append(fused_p)

        if "branch_logits" in outputs:
            preds["cm"].append(torch.sigmoid(outputs["branch_logits"]["cross_modal"]).squeeze(-1).cpu().numpy())
            preds["ao"].append(torch.sigmoid(outputs["branch_logits"]["audio_only"]).squeeze(-1).cpu().numpy())
            preds["vo"].append(torch.sigmoid(outputs["branch_logits"]["visual_only"]).squeeze(-1).cpu().numpy())

        if "fusion_weights" in outputs and outputs["fusion_weights"] is not None:
            w = outputs["fusion_weights"].detach().cpu().numpy()
            w = np.asarray(w)
            if w.ndim == 1:
                w = w.reshape(1, -1)
            if w.ndim > 2:
                w = w.reshape(-1, w.shape[-1])
            if w.shape[-1] >= 3:
                fusion_weights.append(w[:, :3])

        labels.append(y.cpu().numpy())
        # dataset_three_branch uses 'video_id' key (list[str])
        batch_vids = batch.get("video_id", None)
        if batch_vids is not None:
            video_ids.extend(list(batch_vids))

    # Stack arrays
    y_true = np.concatenate(labels, axis=0).astype(int)
    out: Dict[str, Any] = {"n": int(y_true.shape[0])}

    for k in preds:
        if len(preds[k]) == 0:
            continue
        p = np.concatenate(preds[k], axis=0).astype(float)
        y_hat = (p > threshold).astype(int)

        # Metrics
        out[f"{k}_auc"] = float(roc_auc_score(y_true, p)) if len(np.unique(y_true)) > 1 else None
        out[f"{k}_pr_auc"] = float(average_precision_score(y_true, p)) if len(np.unique(y_true)) > 1 else None
        out[f"{k}_accuracy"] = float((y_hat == y_true).mean())
        out[f"{k}_balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_hat)) if len(np.unique(y_true)) > 1 else None
        out[f"{k}_f1"] = float(f1_score(y_true, y_hat, zero_division=0))
        out[f"{k}_precision"] = float(precision_score(y_true, y_hat, zero_division=0))
        out[f"{k}_recall"] = float(recall_score(y_true, y_hat, zero_division=0))

        # Confusion matrix + specificity
        tn, fp, fn, tp = confusion_matrix(y_true, y_hat, labels=[0, 1]).ravel()
        out[f"{k}_tn"] = int(tn)
        out[f"{k}_fp"] = int(fp)
        out[f"{k}_fn"] = int(fn)
        out[f"{k}_tp"] = int(tp)
        out[f"{k}_specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else None

    if len(fusion_weights) > 0:
        w_all = np.concatenate(fusion_weights, axis=0)  # [N,3]
        out["fusion_weight_cm_mean"] = float(w_all[:, 0].mean())
        out["fusion_weight_ao_mean"] = float(w_all[:, 1].mean())
        out["fusion_weight_vo_mean"] = float(w_all[:, 2].mean())
        out["fusion_weight_cm_std"] = float(w_all[:, 0].std())
        out["fusion_weight_ao_std"] = float(w_all[:, 1].std())
        out["fusion_weight_vo_std"] = float(w_all[:, 2].std())

    # Optional per-sample outputs (fused only)
    out["_preds_fused"] = np.concatenate(preds["fused"], axis=0).astype(float).tolist() if len(preds["fused"]) else []
    out["_labels"] = y_true.astype(int).tolist()
    out["_video_ids"] = video_ids
    out["_threshold"] = float(threshold)
    return out


def save_predictions_csv(
    out_dir: Path,
    split: str,
    split_result: Dict[str, Any],
):
    vids = split_result.get("_video_ids", [])
    preds = split_result.get("_preds_fused", [])
    labels = split_result.get("_labels", [])

    if not vids or len(vids) != len(preds) or len(labels) != len(preds):
        return

    csv_path = out_dir / f"predictions_{split}.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("video_id,label,fused_prob\n")
        for vid, y, p in zip(vids, labels, preds):
            f.write(f"{vid},{int(y)},{float(p):.6f}\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Three-Branch Joint Model checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint (.pth)")
    parser.add_argument("--features_root", type=str, required=True, help="Path to extracted features root")
    parser.add_argument("--video_root", type=str, default=None, help="Path to original videos (optional)")
    parser.add_argument("--splits", nargs="+", default=["dev", "test"], help="Splits to evaluate")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for evaluation")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--max_frames", type=int, default=150, help="Max frames per video")
    parser.add_argument("--load_video_frames", action="store_true", help="Load video frames (for CLIP branch)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold")
    parser.add_argument("--output_dir", type=str, default="results/three_branch_eval", help="Output directory")
    parser.add_argument("--save_csv", action="store_true", help="Save per-sample fused predictions to CSV")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    ckpt_args = ckpt.get("args", {})

    # Build dataloaders for requested splits
    dataloaders = get_three_branch_dataloaders(
        features_root=args.features_root,
        video_root=args.video_root,
        splits=args.splits,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_frames=args.max_frames,
        load_video_frames=args.load_video_frames,
    )

    # Infer dims from dataset
    first_split = args.splits[0]
    ds0 = dataloaders[first_split].dataset  # type: ignore[attr-defined]
    sample0 = ds0[0]
    inferred_a_dim = int(sample0["audio"].shape[-1])
    inferred_v_dim = int(sample0["visual"].shape[-1])

    # Create model
    model = ThreeBranchJointModel(
        v_dim=inferred_v_dim,
        a_dim=inferred_a_dim,
        d_model=int(ckpt_args.get("d_model", 512)),
        nhead=int(ckpt_args.get("nhead", 8)),
        cm_layers=int(ckpt_args.get("cm_layers", 4)),
        ao_layers=int(ckpt_args.get("ao_layers", 3)),
        vo_layers=int(ckpt_args.get("vo_layers", 3)),
        dropout=float(ckpt_args.get("dropout", 0.1)),
        fusion_method=str(ckpt_args.get("fusion_method", "weighted")),
        use_dfdfcg=bool(ckpt_args.get("use_dfdfcg", False)),
        dfdfcg_freeze=bool(ckpt_args.get("dfdfcg_freeze", True)),
        dfdfcg_pretrain=ckpt_args.get("dfdfcg_pretrain", None),
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])

    report: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "checkpoint": str(args.checkpoint),
        "epoch": int(ckpt.get("epoch", -1)),
        "device": str(device),
        "features_root": args.features_root,
        "video_root": args.video_root,
        "splits": {},
        "inferred_dims": {"a_dim": inferred_a_dim, "v_dim": inferred_v_dim},
        "threshold": float(args.threshold),
    }

    for split in args.splits:
        loader = dataloaders.get(split)
        if loader is None:
            continue
        split_result = evaluate_split(model, loader, device, threshold=args.threshold)
        report["splits"][split] = {k: v for k, v in split_result.items() if not k.startswith("_")}
        if args.save_csv:
            save_predictions_csv(out_dir, split, split_result)

    out_path = out_dir / "report.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"[OK] Saved report to {out_path}")
    for split, m in report["splits"].items():
        fused_auc = m.get("fused_auc", None)
        fused_acc = m.get("fused_accuracy", None)
        fused_f1 = m.get("fused_f1", None)
        print(f"[{split}] fused_auc={fused_auc} fused_acc={fused_acc} fused_f1={fused_f1}")


if __name__ == "__main__":
    main()


