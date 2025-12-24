#!/usr/bin/env python3
"""
Test script to verify localization pipeline components
Run this to ensure everything is properly set up
"""
import sys
from pathlib import Path

def test_imports():
    """Test that all required modules can be imported"""
    print("[TEST] Checking imports...")
    
    try:
        import torch
        import numpy as np
        from sklearn.metrics import roc_auc_score
        import matplotlib.pyplot as plt
        print("  ✓ Core dependencies OK")
    except ImportError as e:
        print(f"  ✗ Missing dependency: {e}")
        return False
    
    try:
        from dataset_localization import LAVDFLocalizationDataset, collate_variable_length
        from model_localization import FrameLocalizationModel, compute_frame_loss
        print("  ✓ Pipeline modules OK")
    except ImportError as e:
        print(f"  ✗ Failed to import pipeline modules: {e}")
        return False
    
    return True


def test_model():
    """Test model creation and forward pass"""
    print("\n[TEST] Testing model...")
    
    try:
        import torch
        from model_localization import (
            FrameLocalizationModel,
            compute_frame_loss,
            compute_video_loss,
            compute_temporal_smoothness_loss
        )
        
        # Create model
        model = FrameLocalizationModel(
            v_dim=512,
            a_dim=1024,
            d_model=512,
            nhead=8,
            num_layers=2,
            use_cross_attn=True,
            use_video_head=True
        )
        
        total_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"  ✓ Model created: {total_params:.2f}M parameters")
        
        # Test forward pass
        B, T = 2, 64
        visual = torch.randn(B, T, 512)
        audio = torch.randn(B, T, 1024)
        mask = torch.zeros(B, T, dtype=torch.bool)
        mask[:, 50:] = True
        
        frame_logits, video_logit = model(visual, audio, mask)
        
        assert frame_logits.shape == (B, T, 1), f"Wrong frame_logits shape: {frame_logits.shape}"
        assert video_logit.shape == (B, 1), f"Wrong video_logit shape: {video_logit.shape}"
        print(f"  ✓ Forward pass OK: frame {frame_logits.shape}, video {video_logit.shape}")
        
        # Test losses
        frame_labels = torch.randint(0, 2, (B, T))
        video_labels = torch.randint(0, 2, (B,))
        
        frame_loss = compute_frame_loss(frame_logits, frame_labels, mask, pos_weight=5.0)
        video_loss = compute_video_loss(video_logit, video_labels)
        
        frame_probs = torch.sigmoid(frame_logits.squeeze(-1))
        smooth_loss = compute_temporal_smoothness_loss(frame_probs, mask)
        
        assert frame_loss.item() >= 0, "Frame loss should be non-negative"
        assert video_loss.item() >= 0, "Video loss should be non-negative"
        print(f"  ✓ Loss computation OK: frame={frame_loss.item():.4f}, video={video_loss.item():.4f}, smooth={smooth_loss.item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataset(features_root=None):
    """Test dataset loading (requires actual data)"""
    print("\n[TEST] Testing dataset...")
    
    if features_root is None:
        print("  ⊘ Skipping dataset test (no features_root provided)")
        print("    Use: python test_localization_pipeline.py --features_root /path/to/features")
        return True
    
    features_root = Path(features_root)
    if not features_root.exists():
        print(f"  ⊘ Skipping dataset test (path not found: {features_root})")
        return True
    
    try:
        from dataset_localization import LAVDFLocalizationDataset, collate_variable_length
        from torch.utils.data import DataLoader
        
        # Try to load dataset
        dataset = LAVDFLocalizationDataset(
            features_root=str(features_root),
            split='dev',
            max_frames=256
        )
        
        print(f"  ✓ Dataset loaded: {len(dataset)} videos")
        print(f"    Fake ratio: {dataset.fake_ratio*100:.2f}%")
        print(f"    Pos weight: {dataset.pos_weight:.2f}")
        
        if len(dataset) == 0:
            print("  ⊘ Dataset is empty, cannot test further")
            return True
        
        # Test loading a sample
        sample = dataset[0]
        print(f"  ✓ Sample loaded: visual {sample['visual'].shape}, audio {sample['audio'].shape}")
        print(f"    Fake frames: {sample['frame_labels'].sum().item()} / {len(sample['frame_labels'])}")
        
        # Test DataLoader
        loader = DataLoader(dataset, batch_size=2, collate_fn=collate_variable_length)
        batch = next(iter(loader))
        
        print(f"  ✓ DataLoader OK: batch size {len(batch['video_ids'])}")
        print(f"    Batch shapes: visual {batch['visual'].shape}, audio {batch['audio'].shape}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Dataset test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_visualization():
    """Test visualization functions"""
    print("\n[TEST] Testing visualization...")
    
    try:
        from visualize_localization import visualize_localization, extract_fake_segments
        import numpy as np
        
        # Create dummy data
        T = 250
        frame_probs = np.random.rand(T) * 0.3
        frame_probs[50:100] = np.random.rand(50) * 0.5 + 0.5  # Fake segment
        frame_probs[150:180] = np.random.rand(30) * 0.4 + 0.6  # Another segment
        
        frame_labels = np.zeros(T)
        frame_labels[50:100] = 1
        frame_labels[150:180] = 1
        
        # Extract segments
        segments = extract_fake_segments(frame_probs, threshold=0.5, min_duration=5)
        print(f"  ✓ Segment extraction OK: found {len(segments)} segments")
        
        # Test visualization (don't save)
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            temp_path = f.name
        
        try:
            visualize_localization(
                frame_probs=frame_probs,
                frame_labels=frame_labels,
                fps=25.0,
                threshold=0.5,
                save_path=temp_path,
                title="Test Visualization"
            )
            print(f"  ✓ Visualization OK")
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        
        return True
        
    except Exception as e:
        print(f"  ✗ Visualization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Test localization pipeline')
    parser.add_argument('--features_root', type=str, default=None,
                        help='Path to features for dataset test')
    args = parser.parse_args()
    
    print("="*60)
    print("Localization Pipeline Test Suite")
    print("="*60)
    
    tests = [
        ("Imports", test_imports),
        ("Model", test_model),
        ("Dataset", lambda: test_dataset(args.features_root)),
        ("Visualization", test_visualization)
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            success = test_fn()
            results.append((name, success))
        except Exception as e:
            print(f"\n[ERROR] Test {name} crashed: {e}")
            results.append((name, False))
    
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    for name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status:8s} {name}")
    
    all_passed = all(success for _, success in results)
    
    print("="*60)
    if all_passed:
        print("✓ All tests passed!")
        return 0
    else:
        print("✗ Some tests failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())

