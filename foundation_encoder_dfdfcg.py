#!/usr/bin/env python3
"""
DFD-FCG Foundation Model Encoder Wrapper
Uses CLIP ViT-L/14 from DFD-FCG for visual feature extraction
"""
import sys
import torch
import torch.nn as nn
from pathlib import Path
import numpy as np
from typing import Tuple, Optional

# Add DFD-FCG to path
DFDFCG_PATH = Path(__file__).parent / "DFD-FCG"
sys.path.insert(0, str(DFDFCG_PATH))

try:
    from src.model.clip import VideoAttrExtractor
    DFDFCG_AVAILABLE = True
except ImportError:
    DFDFCG_AVAILABLE = False
    print("[WARN] DFD-FCG not available. Please run: git clone https://github.com/aiiu-lab/DFD-FCG.git")


class DFDFCGEncoder(nn.Module):
    """
    Wrapper for DFD-FCG's CLIP ViT-L/14 encoder
    Extracts visual features from video frames
    """
    def __init__(
        self,
        architecture: str = "ViT-L/14",
        pretrain_path: Optional[str] = None,
        freeze: bool = True,
        text_embed: bool = False
    ):
        """
        Args:
            architecture: CLIP architecture (default: ViT-L/14)
            pretrain_path: Path to pre-trained weights (optional)
            freeze: Freeze encoder weights
            text_embed: Use text embedding projection
        """
        super().__init__()
        
        if not DFDFCG_AVAILABLE:
            raise ImportError("DFD-FCG is not available. Please clone the repository first.")
        
        self.architecture = architecture
        self.freeze = freeze
        
        # Load CLIP visual encoder
        self.encoder = VideoAttrExtractor(
            architecture=architecture,
            text_embed=text_embed,
            store_attrs=[],
            attn_record=False,
            pretrain=pretrain_path
        )
        
        # Feature dimension
        self.feat_dim = self.encoder.feat_dim  # 768 for ViT-L/14
        
        if freeze:
            self.encoder.eval()
            for param in self.encoder.parameters():
                param.requires_grad = False
    
    @property
    def output_dim(self):
        return self.feat_dim
    
    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Extract CLIP features from video frames
        
        Args:
            frames: [B, T, C, H, W] video frames (RGB, normalized)
        
        Returns:
            features: [B, T, feat_dim] CLIP features
        """
        if len(frames.shape) != 5:
            raise ValueError(f"Expected 5D input [B, T, C, H, W], got {frames.shape}")
        
        B, T, C, H, W = frames.shape
        
        # Forward through encoder (DFD-FCG expects [B, T, C, H, W])
        with torch.set_grad_enabled(not self.freeze):
            output = self.encoder(frames)
            features = output['embeds']  # Shape depends on model output
        
        # Check and adjust output shape
        if len(features.shape) == 4:  # [B, T, ..., feat_dim]
            features = features.squeeze(2)  # [B, T, feat_dim]
        elif len(features.shape) == 2:  # [B, feat_dim] (pooled over time)
            features = features.unsqueeze(1).expand(B, T, -1)
        elif len(features.shape) != 3:
            raise ValueError(f"Unexpected feature shape: {features.shape}")
        
        return features
    
    def train(self, mode=True):
        """Override train to keep encoder frozen if specified"""
        if self.freeze:
            super().train(False)
            self.encoder.eval()
        else:
            super().train(mode)
        return self


class DFDFCGFeatureExtractor:
    """
    Utility class for extracting DFD-FCG features from videos
    """
    def __init__(
        self,
        device: str = 'cuda',
        architecture: str = "ViT-L/14",
        pretrain_path: Optional[str] = None
    ):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        self.encoder = DFDFCGEncoder(
            architecture=architecture,
            pretrain_path=pretrain_path,
            freeze=True
        ).to(self.device)
        
        self.encoder.eval()
        
        # CLIP preprocessing parameters
        self.input_size = 224
        self.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(self.device)
        self.std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(self.device)
    
    def preprocess_frames(self, frames: np.ndarray) -> torch.Tensor:
        """
        Preprocess frames for CLIP
        
        Args:
            frames: [T, H, W, 3] numpy array (0-255, RGB)
        
        Returns:
            preprocessed: [T, 3, 224, 224] torch tensor
        """
        import torchvision.transforms.functional as TF
        from PIL import Image
        
        T = len(frames)
        processed = []
        
        for frame in frames:
            # Convert to PIL
            if frame.dtype == np.uint8:
                img = Image.fromarray(frame)
            else:
                img = Image.fromarray((frame * 255).astype(np.uint8))
            
            # Resize and center crop
            img = TF.resize(img, self.input_size, interpolation=TF.InterpolationMode.BICUBIC)
            img = TF.center_crop(img, self.input_size)
            
            # To tensor
            img_tensor = TF.to_tensor(img)  # [3, 224, 224], [0, 1]
            
            processed.append(img_tensor)
        
        # Stack and normalize
        processed = torch.stack(processed).to(self.device)  # [T, 3, 224, 224]
        processed = (processed - self.mean) / self.std
        
        return processed
    
    @torch.no_grad()
    def extract_from_frames(self, frames: np.ndarray, batch_size: int = 32) -> np.ndarray:
        """
        Extract CLIP features from frames
        
        Args:
            frames: [T, H, W, 3] numpy array
            batch_size: Batch size for processing
        
        Returns:
            features: [T, feat_dim] numpy array
        """
        # Preprocess
        frames_tensor = self.preprocess_frames(frames)  # [T, 3, 224, 224]
        
        T = len(frames_tensor)
        all_features = []
        
        # Process in batches (add dummy time dimension)
        for i in range(0, T, batch_size):
            batch = frames_tensor[i:i + batch_size]  # [batch, 3, 224, 224]
            # Add time dimension: [batch, 3, 224, 224] -> [1, batch, 3, 224, 224]
            batch_5d = batch.unsqueeze(0)  # [1, batch, 3, 224, 224]
            features = self.encoder(batch_5d)  # [1, batch, feat_dim]
            features = features.squeeze(0)  # [batch, feat_dim]
            all_features.append(features.cpu())
        
        # Concatenate
        all_features = torch.cat(all_features, dim=0).numpy()  # [T, feat_dim]
        
        return all_features
    
    @torch.no_grad()
    def extract_from_video(self, video_path: str, max_frames: int = None) -> np.ndarray:
        """
        Extract CLIP features from video file
        
        Args:
            video_path: Path to video file
            max_frames: Maximum number of frames to extract
        
        Returns:
            features: [T, 768] numpy array
        """
        import cv2
        
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            
            if max_frames and len(frames) >= max_frames:
                break
        
        cap.release()
        
        if len(frames) == 0:
            raise ValueError(f"No frames extracted from {video_path}")
        
        frames = np.array(frames)  # [T, H, W, 3]
        
        # Extract features
        features = self.extract_from_frames(frames)
        
        return features


def test_encoder():
    """Test DFD-FCG encoder"""
    print("[TEST] Testing DFD-FCG encoder...")
    
    if not DFDFCG_AVAILABLE:
        print("[SKIP] DFD-FCG not available")
        return
    
    # Create encoder
    encoder = DFDFCGEncoder(architecture="ViT-L/14", freeze=True)
    print(f"[OK] Encoder created, output dim: {encoder.output_dim}")
    
    # Test forward pass
    B, T, C, H, W = 2, 10, 3, 224, 224
    frames = torch.randn(B, T, C, H, W)
    
    features = encoder(frames)
    print(f"[OK] Forward pass: {frames.shape} -> {features.shape}")
    
    assert features.shape == (B, T, encoder.output_dim)
    print(f"[OK] Output shape correct: {features.shape}")
    
    # Test feature extractor
    print("\n[TEST] Testing feature extractor...")
    extractor = DFDFCGFeatureExtractor(device='cpu')
    
    # Dummy frames
    frames_np = np.random.randint(0, 255, (50, 224, 224, 3), dtype=np.uint8)
    features_np = extractor.extract_from_frames(frames_np, batch_size=16)
    
    print(f"[OK] Feature extraction: {frames_np.shape} -> {features_np.shape}")
    
    print("\n[OK] All tests passed!")


if __name__ == '__main__':
    test_encoder()

