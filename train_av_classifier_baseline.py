"""
Baseline trainer: use precomputed audio-visual similarity statistics to train
a lightweight classifier (Logistic Regression or small MLP) for real/fake.

Input: a CSV produced by prepare_features_dataset.py (summary.csv)
Columns expected at minimum:
    video, label, mean, std, min, max, median, p10, p25, p50, p75, p90,
    ratio_low_sim_frames, num_frames, ...

Usage:
    python train_av_classifier_baseline.py --csv /path/to/summary.csv \
        --model mlp --epochs 20 --batch_size 64
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import csv


FEATURE_KEYS = [
    "mean",
    "std",
    "min",
    "max",
    "median",
    "p10",
    "p25",
    "p50",
    "p75",
    "p90",
    "range",
    "ratio_low_sim_frames",
]


def load_csv(csv_path: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    xs = []
    ys = []
    vids = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                feat = [float(row[k]) for k in FEATURE_KEYS]
            except KeyError as e:
                raise ValueError(f"Missing feature {e} in CSV. Available columns: {list(row.keys())}")
            xs.append(feat)
            ys.append(0 if row["label"].lower() == "real" else 1)
            vids.append(row.get("video", ""))
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.int64), vids


def train_baseline(
    X: np.ndarray,
    y: np.ndarray,
    model_type: str = "mlp",
    test_size: float = 0.2,
    random_state: int = 42,
    max_iter: int = 200,
):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    if model_type == "mlp":
        clf = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            max_iter=max_iter,
            random_state=random_state,
            verbose=False,
        )
    elif model_type == "logreg":
        clf = LogisticRegression(
            max_iter=max_iter,
            class_weight="balanced",
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    return clf, scaler, {"acc": acc, "auc": auc}


def save_model(clf, scaler, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    import joblib

    joblib.dump(clf, out_dir / "classifier.joblib")
    joblib.dump(scaler, out_dir / "scaler.joblib")


def main():
    parser = argparse.ArgumentParser(description="Train baseline classifier on AV stats")
    parser.add_argument("--csv", type=str, required=True, help="Path to summary.csv")
    parser.add_argument("--model", type=str, default="mlp", choices=["mlp", "logreg"])
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--max_iter", type=int, default=200)
    parser.add_argument("--out_dir", type=str, default="av_baseline_ckpt")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    X, y, vids = load_csv(csv_path)
    clf, scaler, metrics = train_baseline(
        X, y,
        model_type=args.model,
        test_size=args.test_size,
        max_iter=args.max_iter,
    )

    print(f"[RESULT] ACC={metrics['acc']:.4f}, AUC={metrics['auc']:.4f}")
    save_model(clf, scaler, Path(args.out_dir))
    with open(Path(args.out_dir) / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"[INFO] Saved model and scaler to {args.out_dir}")


if __name__ == "__main__":
    main()

