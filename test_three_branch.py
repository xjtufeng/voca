#!/usr/bin/env python3
"""
Test script for three-branch model
Verify model creation, forward pass, and loss computation
"""
import sys
import torch
import numpy as np

def test_model_creation():
    """Test that model can be created"""
    print("[TEST 1] Testing model creation...")
    
    try:
        from model_three_branch import ThreeBranchJointModel
        
        model = ThreeBranchJointModel(
            v_dim=512,
            a_dim=1024,
            d_model=512,
            nhead=8,
            cm_layers=4,
            ao_layers=3,
            vo_layers=3,
            fusion_method='attention'
        )
        
        total_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"  [OK] Model created successfully: {total_params:.2f}M parameters")
        
        return True
    except Exception as e:
        print(f"  [FAIL] Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_forward_pass():
    """Test forward pass with dummy data"""
    print("\n[TEST 2] Testing forward pass...")
    
    try:
        from model_three_branch import ThreeBranchJointModel
        
        model = ThreeBranchJointModel(
            v_dim=512,
            a_dim=1024,
            d_model=512,
            nhead=8,
            fusion_method='attention'
        )
        
        # Create dummy inputs
        B, T = 4, 128
        audio = torch.randn(B, T, 1024)
        visual = torch.randn(B, T, 512)
        mask = torch.zeros(B, T, dtype=torch.bool)
        mask[:, 100:] = True
        
        # Forward pass
        outputs = model(audio, visual, mask, return_branch_outputs=True)
        
        # Check outputs
        assert 'fused_logit' in outputs, "Missing fused_logit"
        assert outputs['fused_logit'].shape == (B, 1), f"Wrong fused shape: {outputs['fused_logit'].shape}"
        
        assert 'branch_logits' in outputs, "Missing branch_logits"
        assert 'cross_modal' in outputs['branch_logits'], "Missing cross_modal logit"
        assert 'audio_only' in outputs['branch_logits'], "Missing audio_only logit"
        assert 'visual_only' in outputs['branch_logits'], "Missing visual_only logit"
        
        if 'fusion_weights' in outputs:
            print(f"  [OK] Fusion weights shape: {outputs['fusion_weights'].shape}")
            print(f"  [OK] Sample fusion weights: {outputs['fusion_weights'][0]}")
        
        print(f"  [OK] Forward pass successful")
        print(f"     Fused logit: {outputs['fused_logit'].shape}")
        print(f"     Branch logits: {[v.shape for v in outputs['branch_logits'].values()]}")
        
        return True
    except Exception as e:
        print(f"  [FAIL] Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_loss_computation():
    """Test loss computation"""
    print("\n[TEST 3] Testing loss computation...")
    
    try:
        from model_three_branch import ThreeBranchJointModel, compute_three_branch_loss
        
        model = ThreeBranchJointModel(
            v_dim=512,
            a_dim=1024,
            d_model=512,
            nhead=8,
            fusion_method='attention'
        )
        
        # Create dummy inputs
        B, T = 4, 128
        audio = torch.randn(B, T, 1024)
        visual = torch.randn(B, T, 512)
        mask = torch.zeros(B, T, dtype=torch.bool)
        labels = torch.randint(0, 2, (B,))
        
        # Forward pass
        outputs = model(audio, visual, mask, return_branch_outputs=True)
        
        # Compute loss
        losses = compute_three_branch_loss(
            outputs=outputs,
            labels=labels,
            branch_weights=(0.3, 0.2, 0.2),
            fusion_weight=1.0
        )
        
        # Check losses
        assert 'total' in losses, "Missing total loss"
        assert 'fused' in losses, "Missing fused loss"
        assert 'cross_modal' in losses, "Missing cross_modal loss"
        assert 'audio_only' in losses, "Missing audio_only loss"
        assert 'visual_only' in losses, "Missing visual_only loss"
        
        print(f"  [OK] Loss computation successful")
        for name, loss in losses.items():
            print(f"     {name:12s} loss: {loss.item():.4f}")
        
        # Test backward
        losses['total'].backward()
        print(f"  [OK] Backward pass successful")
        
        return True
    except Exception as e:
        print(f"  [FAIL] Loss computation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_different_fusion_methods():
    """Test all fusion methods"""
    print("\n[TEST 4] Testing different fusion methods...")
    
    try:
        from model_three_branch import ThreeBranchJointModel
        
        B, T = 2, 64
        audio = torch.randn(B, T, 1024)
        visual = torch.randn(B, T, 512)
        
        for fusion_method in ['concat', 'weighted', 'attention']:
            print(f"  Testing {fusion_method}...")
            
            model = ThreeBranchJointModel(
                v_dim=512,
                a_dim=1024,
                d_model=512,
                nhead=8,
                fusion_method=fusion_method
            )
            
            outputs = model(audio, visual, return_branch_outputs=True)
            
            assert outputs['fused_logit'].shape == (B, 1)
            print(f"    [OK] {fusion_method} fusion works")
        
        print(f"  [OK] All fusion methods work")
        return True
    except Exception as e:
        print(f"  [FAIL] Fusion method test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_with_mask():
    """Test with padding mask"""
    print("\n[TEST 5] Testing with padding mask...")
    
    try:
        from model_three_branch import ThreeBranchJointModel
        
        model = ThreeBranchJointModel(
            v_dim=512,
            a_dim=1024,
            d_model=512,
            nhead=8
        )
        
        # Different sequence lengths
        B = 4
        lengths = [100, 120, 80, 150]
        max_len = max(lengths)
        
        audio = torch.randn(B, max_len, 1024)
        visual = torch.randn(B, max_len, 512)
        mask = torch.zeros(B, max_len, dtype=torch.bool)
        
        for i, length in enumerate(lengths):
            mask[i, length:] = True
        
        outputs = model(audio, visual, mask, return_branch_outputs=True)
        
        assert outputs['fused_logit'].shape == (B, 1)
        print(f"  [OK] Padding mask handled correctly")
        print(f"     Sequence lengths: {lengths}")
        
        return True
    except Exception as e:
        print(f"  [FAIL] Mask test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gradient_flow():
    """Test that gradients flow to all branches"""
    print("\n[TEST 6] Testing gradient flow...")
    
    try:
        from model_three_branch import ThreeBranchJointModel, compute_three_branch_loss
        
        model = ThreeBranchJointModel(
            v_dim=512,
            a_dim=1024,
            d_model=512,
            nhead=8
        )
        
        B, T = 2, 64
        audio = torch.randn(B, T, 1024)
        visual = torch.randn(B, T, 512)
        labels = torch.randint(0, 2, (B,))
        
        # Forward + backward
        outputs = model(audio, visual, return_branch_outputs=True)
        losses = compute_three_branch_loss(outputs, labels)
        losses['total'].backward()
        
        # Check gradients
        cm_has_grad = any(p.grad is not None for p in model.cross_modal_branch.parameters())
        ao_has_grad = any(p.grad is not None for p in model.audio_only_branch.parameters())
        vo_has_grad = any(p.grad is not None for p in model.visual_only_branch.parameters())
        
        assert cm_has_grad, "Cross-modal branch has no gradients"
        assert ao_has_grad, "Audio-only branch has no gradients"
        assert vo_has_grad, "Visual-only branch has no gradients"
        
        print(f"  [OK] Gradients flow to all branches")
        print(f"     Cross-Modal: {cm_has_grad}")
        print(f"     Audio-Only: {ao_has_grad}")
        print(f"     Visual-Only: {vo_has_grad}")
        
        return True
    except Exception as e:
        print(f"  [FAIL] Gradient flow test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("="*60)
    print("Three-Branch Model Test Suite")
    print("="*60)
    
    tests = [
        ("Model Creation", test_model_creation),
        ("Forward Pass", test_forward_pass),
        ("Loss Computation", test_loss_computation),
        ("Fusion Methods", test_different_fusion_methods),
        ("Padding Mask", test_with_mask),
        ("Gradient Flow", test_gradient_flow)
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
        status = "[PASS]" if success else "[FAIL]"
        print(f"{status:8s} {name}")
    
    all_passed = all(success for _, success in results)
    
    print("="*60)
    if all_passed:
        print("[OK] All tests passed!")
        return 0
    else:
        print("[FAIL] Some tests failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())

