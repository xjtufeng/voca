#!/usr/bin/env python3
"""
Stable GPU occupier script - maintains target GPU utilization with heavy matmul loops.
"""
import argparse
import sys
import time

import torch
import torch.nn as nn


class MatmulOccupier(nn.Module):
    def __init__(self, dim=2048):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim, bias=False)
        self.linear2 = nn.Linear(dim, dim, bias=False)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.act(self.linear1(x))
        x = self.linear2(x)
        return x


def build_workload(device, batch, dim):
    data = torch.randn(batch, dim, device=device, requires_grad=True)
    model = MatmulOccupier(dim=dim).to(device)
    return model, data


def compute_once(model, data, repeats):
    out = data
    for _ in range(repeats):
        out = model(out)
    loss = (out * out).mean()
    loss.backward()
    model.zero_grad(set_to_none=True)


def configure_load(target_percent):
    if target_percent < 20:
        return {"batch": 32, "repeats": 2, "dim": 2048, "sleep": 0.1}
    if target_percent < 40:
        return {"batch": 64, "repeats": 4, "dim": 4096, "sleep": 0.05}
    if target_percent < 60:
        return {"batch": 96, "repeats": 6, "dim": 6144, "sleep": 0.01}
    if target_percent < 80:
        return {"batch": 128, "repeats": 8, "dim": 8192, "sleep": 0.0}
    return {"batch": 160, "repeats": 10, "dim": 8192, "sleep": 0.0}


def occupy_gpu(gpu_id=0, target_percent=35, dim=None):
    device = torch.device(f"cuda:{gpu_id}")
    if not torch.cuda.is_available():
        print("[ERROR] CUDA not available")
        sys.exit(1)

    if target_percent < 10 or target_percent > 95:
        print(f"[ERROR] target_percent out of range: {target_percent}")
        sys.exit(1)

    cfg = configure_load(target_percent)
    if dim is None:
        dim = cfg["dim"]
    batch, repeats, sleep = cfg["batch"], cfg["repeats"], cfg["sleep"]

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    model, data = build_workload(device, batch, dim)

    print(f"[INFO] Device: {device}")
    print(f"[INFO] Target utilization: {target_percent}%")
    print(f"[INFO] Config: batch={batch}, repeats={repeats}, dim={dim}, sleep={sleep}s")
    print("[INFO] Press Ctrl+C to stop")
    sys.stdout.flush()

    step = 0
    start = time.time()
    try:
        while True:
            compute_once(model, data, repeats)
            step += 1
            if step % 50 == 0:
                elapsed = time.time() - start
                print(f"[RUNNING] step={step}, elapsed={elapsed/60:.1f} min")
                sys.stdout.flush()
            if sleep > 0:
                time.sleep(sleep)
    except KeyboardInterrupt:
        elapsed = time.time() - start
        print(f"\n[INFO] Stopped. Elapsed={elapsed/60:.1f} min, steps={step}")


def main():
    parser = argparse.ArgumentParser(description="Stable GPU occupier")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID")
    parser.add_argument("--percent", type=int, default=35, help="Target GPU utilization (10-95)")
    parser.add_argument("--dim", type=int, default=2048, help="Feature dimension for matmul workload")
    args = parser.parse_args()

    occupy_gpu(gpu_id=args.gpu, target_percent=args.percent, dim=args.dim)


if __name__ == "__main__":
    main()


