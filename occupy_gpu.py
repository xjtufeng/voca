#!/usr/bin/env python3
"""
Stable GPU occupier script - maintains target GPU utilization
"""
import torch
import torch.nn as nn
import time
import argparse
import sys


class LightweightOccupier(nn.Module):
    """Lightweight model for GPU occupation"""
    def __init__(self, size=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(size, size * 2),
            nn.ReLU(),
            nn.Linear(size * 2, size * 2),
            nn.ReLU(),
            nn.Linear(size * 2, size),
        )
    
    def forward(self, x):
        return self.net(x)


def occupy_gpu_stable(gpu_id=0, target_percent=30, verbose=True):
    """
    Occupy GPU at target percentage
    
    Args:
        gpu_id: GPU device ID
        target_percent: Target utilization percentage (10-90)
        verbose: Print status messages
    """
    device = torch.device(f'cuda:{gpu_id}')
    
    if verbose:
        print(f"[INFO] GPU occupation started")
        print(f"[INFO] Device: {device}")
        print(f"[INFO] Target utilization: {target_percent}%")
        print(f"[INFO] Press Ctrl+C to stop")
        sys.stdout.flush()
    
    model = LightweightOccupier(size=1024).to(device)
    
    if target_percent < 20:
        batch_size, iterations = 16, 1
        sleep_time = 0.2
    elif target_percent < 40:
        batch_size, iterations = 32, 2
        sleep_time = 0.1
    elif target_percent < 60:
        batch_size, iterations = 64, 4
        sleep_time = 0.05
    else:
        batch_size, iterations = 128, 8
        sleep_time = 0.01
    
    data = torch.randn(batch_size, 1024, device=device)
    
    if verbose:
        params_m = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"[INFO] Config: batch={batch_size}, iters={iterations}, sleep={sleep_time}s")
        print(f"[INFO] Model params: {params_m:.2f}M")
        sys.stdout.flush()
    
    step = 0
    start_time = time.time()
    
    try:
        while True:
            for _ in range(iterations):
                output = model(data)
                loss = output.mean()
                loss.backward()
                model.zero_grad()
            
            step += 1
            
            if verbose and step % 100 == 0:
                elapsed = time.time() - start_time
                print(f"[RUNNING] Step {step} | Elapsed: {elapsed/60:.1f} min")
                sys.stdout.flush()
            
            time.sleep(sleep_time)
    
    except KeyboardInterrupt:
        if verbose:
            elapsed = time.time() - start_time
            print(f"\n[INFO] Stopped")
            print(f"[INFO] Total runtime: {elapsed/60:.1f} min ({step} steps)")
            sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser(description='Stable GPU occupation')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    parser.add_argument('--percent', type=int, default=30, 
                        help='Target utilization percentage (10-90, default 30)')
    parser.add_argument('--quiet', action='store_true', 
                        help='Quiet mode for background running')
    
    args = parser.parse_args()
    
    if args.percent < 10 or args.percent > 90:
        print(f"[ERROR] Percent must be between 10-90, got: {args.percent}")
        sys.exit(1)
    
    occupy_gpu_stable(
        gpu_id=args.gpu,
        target_percent=args.percent,
        verbose=not args.quiet
    )


if __name__ == '__main__':
    main()


