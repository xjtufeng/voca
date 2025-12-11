#!/usr/bin/env python3
"""
Simple GPU occupier script
Loads features and does continuous computation to maintain GPU usage
"""
import torch
import torch.nn as nn
import time
import glob
import numpy as np
from pathlib import Path
import argparse


class DummyModel(nn.Module):
    """Lightweight model for GPU computation"""
    def __init__(self, input_dim=512, hidden=1024):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, input_dim)
        )
    
    def forward(self, x):
        return self.layers(x)


def load_random_features(features_root, batch_size=32, seq_len=256):
    """Load random features from dataset"""
    print(f"[INFO] Loading features from {features_root}")
    
    # Find some feature files
    visual_files = list(Path(features_root).rglob("visual_embeddings.npz"))
    if not visual_files:
        print("[WARN] No features found, using random data")
        return torch.randn(batch_size, seq_len, 512)
    
    # Load a few files
    features = []
    for vf in visual_files[:batch_size]:
        try:
            data = np.load(vf)
            emb = data['embeddings']  # [T, 512]
            if len(emb) >= seq_len:
                features.append(emb[:seq_len])
            else:
                # Pad if needed
                padded = np.zeros((seq_len, 512), dtype=emb.dtype)
                padded[:len(emb)] = emb
                features.append(padded)
        except:
            continue
        
        if len(features) >= batch_size:
            break
    
    if not features:
        print("[WARN] Failed to load features, using random data")
        return torch.randn(batch_size, seq_len, 512)
    
    # Stack to batch
    while len(features) < batch_size:
        features.append(features[0])  # Repeat if not enough
    
    batch = np.stack(features[:batch_size], axis=0)  # [B, T, 512]
    print(f"[INFO] Loaded batch: {batch.shape}")
    return torch.from_numpy(batch).float()


def occupy_gpu(features_root, gpu_id=0, target_usage=30, duration=None):
    """
    Occupy GPU with continuous computation
    
    Args:
        features_root: Path to features directory
        gpu_id: GPU device ID
        target_usage: Target GPU usage percentage (approximate)
        duration: How long to run (seconds), None = infinite
    """
    device = torch.device(f'cuda:{gpu_id}')
    print(f"[INFO] Using device: {device}")
    print(f"[INFO] Target usage: ~{target_usage}%")
    
    # Create model
    model = DummyModel(input_dim=512, hidden=1024).to(device)
    print(f"[INFO] Model created with {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")
    
    # Load features
    batch_size = 32
    seq_len = 256
    data = load_random_features(features_root, batch_size, seq_len).to(device)
    print(f"[INFO] Data loaded: {data.shape} ({data.element_size() * data.nelement() / 1024**2:.2f} MB)")
    
    # Adjust computation intensity based on target usage
    # Higher target = more iterations per loop
    iterations_per_loop = max(1, target_usage // 10)
    sleep_time = 0.1 if target_usage < 50 else 0.01
    
    print(f"[INFO] Starting GPU occupation loop...")
    print(f"[INFO] Press Ctrl+C to stop")
    
    start_time = time.time()
    step = 0
    
    try:
        while True:
            # Do some computation
            for _ in range(iterations_per_loop):
                # Forward pass through each frame
                output = model(data.reshape(-1, 512))  # [B*T, 512]
                output = output.reshape(batch_size, seq_len, 512)
                
                # Some matrix operations
                attn = torch.matmul(output, output.transpose(-2, -1))  # [B, T, T]
                attn = torch.softmax(attn, dim=-1)
                output = torch.matmul(attn, output)  # [B, T, 512]
                
                # Backward to keep gradients
                loss = output.mean()
                loss.backward()
                model.zero_grad()
            
            step += 1
            
            # Periodic status
            if step % 100 == 0:
                elapsed = time.time() - start_time
                print(f"[STEP {step}] Running for {elapsed:.1f}s")
            
            # Check duration
            if duration and (time.time() - start_time) >= duration:
                print(f"[INFO] Reached target duration: {duration}s")
                break
            
            # Small sleep to control usage
            time.sleep(sleep_time)
    
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
    
    elapsed = time.time() - start_time
    print(f"[INFO] Stopped after {elapsed:.1f}s ({step} steps)")
    print(f"[INFO] GPU occupation complete")


def main():
    parser = argparse.ArgumentParser(description='Occupy GPU with computation')
    parser.add_argument('--features_root', type=str, 
                        default='/hpc2ssd/JH_DATA/spooler/xfeng733/FakeAV_feats',
                        help='Path to features directory')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    parser.add_argument('--usage', type=int, default=30, 
                        help='Target GPU usage percentage (approximate)')
    parser.add_argument('--duration', type=int, default=None,
                        help='Duration in seconds (None = infinite)')
    
    args = parser.parse_args()
    
    occupy_gpu(
        features_root=args.features_root,
        gpu_id=args.gpu,
        target_usage=args.usage,
        duration=args.duration
    )


if __name__ == '__main__':
    main()

