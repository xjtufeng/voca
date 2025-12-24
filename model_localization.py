#!/usr/bin/env python3
"""
Frame-Level Deepfake Localization Model
Cross-modal Transformer with temporal encoding for frame-wise classification
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class FrameLocalizationModel(nn.Module):
    """
    Frame-level deepfake localization model
    
    Architecture:
    1. Project visual/audio features to common dimension
    2. Cross-modal attention (bidirectional)
    3. Temporal Transformer encoder
    4. Frame-level classifier (per-frame logits)
    5. Video-level classifier (auxiliary task)
    """
    def __init__(
        self,
        v_dim: int = 512,
        a_dim: int = 1024,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        use_cross_attn: bool = True,
        use_video_head: bool = True
    ):
        """
        Args:
            v_dim: Visual feature dimension
            a_dim: Audio feature dimension
            d_model: Transformer hidden dimension
            nhead: Number of attention heads
            num_layers: Number of Transformer layers
            dim_feedforward: FFN dimension
            dropout: Dropout rate
            use_cross_attn: Use cross-modal attention
            use_video_head: Add video-level classification head
        """
        super().__init__()
        
        self.d_model = d_model
        self.use_cross_attn = use_cross_attn
        self.use_video_head = use_video_head
        
        # Feature projection
        self.v_proj = nn.Linear(v_dim, d_model)
        self.a_proj = nn.Linear(a_dim, d_model)
        
        # Cross-modal attention (bidirectional)
        if use_cross_attn:
            self.cross_attn_v2a = nn.MultiheadAttention(
                d_model, nhead, dropout=dropout, batch_first=True
            )
            self.cross_attn_a2v = nn.MultiheadAttention(
                d_model, nhead, dropout=dropout, batch_first=True
            )
            self.cross_norm_v = nn.LayerNorm(d_model)
            self.cross_norm_a = nn.LayerNorm(d_model)
        
        # Temporal Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Frame-level classifier
        self.frame_classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
        
        # Video-level classifier (auxiliary task)
        if use_video_head:
            self.video_classifier = nn.Sequential(
                nn.Linear(d_model, 256),
                nn.GELU(),
                nn.Dropout(dropout * 2),
                nn.Linear(256, 1)
            )
    
    def forward(
        self, 
        visual: torch.Tensor, 
        audio: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            visual: [B, T, v_dim] visual features
            audio: [B, T, a_dim] audio features
            mask: [B, T] bool mask (True = padding)
        
        Returns:
            frame_logits: [B, T, 1] per-frame fake logits
            video_logit: [B, 1] video-level fake logit (if use_video_head)
        """
        B, T, _ = visual.shape
        
        # Project to common dimension
        v = self.v_proj(visual)  # [B, T, d_model]
        a = self.a_proj(audio)   # [B, T, d_model]
        
        # Cross-modal attention
        if self.use_cross_attn:
            # Video -> Audio attention
            v_enhanced, _ = self.cross_attn_v2a(
                v, a, a, key_padding_mask=mask
            )
            v_enhanced = self.cross_norm_v(v + v_enhanced)
            
            # Audio -> Video attention
            a_enhanced, _ = self.cross_attn_a2v(
                a, v, v, key_padding_mask=mask
            )
            a_enhanced = self.cross_norm_a(a + a_enhanced)
            
            # Fuse cross-attended features
            fused = (v_enhanced + a_enhanced) / 2
        else:
            # Simple average fusion
            fused = (v + a) / 2
        
        # Temporal encoding
        encoded = self.temporal_encoder(fused, src_key_padding_mask=mask)  # [B, T, d_model]
        
        # Frame-level classification
        frame_logits = self.frame_classifier(encoded)  # [B, T, 1]
        
        # Video-level classification (pooling + classifier)
        video_logit = None
        if self.use_video_head:
            if mask is not None:
                # Masked average pooling
                masked_encoded = encoded.masked_fill(mask.unsqueeze(-1), 0)
                valid_counts = (~mask).sum(dim=1, keepdim=True).clamp(min=1)
                pooled = masked_encoded.sum(dim=1) / valid_counts  # [B, d_model]
            else:
                pooled = encoded.mean(dim=1)  # [B, d_model]
            
            video_logit = self.video_classifier(pooled)  # [B, 1]
        
        return frame_logits, video_logit


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [N] raw logits
            targets: [N] binary targets (0 or 1)
        """
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction='none')
        pt = torch.exp(-bce_loss)  # p_t
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def compute_frame_loss(
    frame_logits: torch.Tensor,
    frame_labels: torch.Tensor,
    mask: torch.Tensor,
    pos_weight: Optional[float] = None,
    use_focal: bool = False,
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0
) -> torch.Tensor:
    """
    Compute frame-level classification loss
    
    Args:
        frame_logits: [B, T, 1] frame logits
        frame_labels: [B, T] frame labels (0=real, 1=fake)
        mask: [B, T] padding mask (True=padding)
        pos_weight: Positive class weight for BCE
        use_focal: Use Focal Loss instead of BCE
        focal_alpha: Focal loss alpha
        focal_gamma: Focal loss gamma
    
    Returns:
        Scalar loss
    """
    B, T = frame_labels.shape
    
    # Flatten
    logits_flat = frame_logits.squeeze(-1).reshape(-1)  # [B*T]
    labels_flat = frame_labels.reshape(-1).float()      # [B*T]
    mask_flat = mask.reshape(-1)                        # [B*T]
    
    # Filter out padding
    valid_logits = logits_flat[~mask_flat]
    valid_labels = labels_flat[~mask_flat]
    
    if len(valid_logits) == 0:
        return torch.tensor(0.0, device=frame_logits.device)
    
    # Compute loss
    if use_focal:
        criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        loss = criterion(valid_logits, valid_labels)
    else:
        if pos_weight is not None:
            weight = torch.tensor([pos_weight], device=frame_logits.device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=weight)
        else:
            criterion = nn.BCEWithLogitsLoss()
        loss = criterion(valid_logits, valid_labels)
    
    return loss


def compute_video_loss(
    video_logit: torch.Tensor,
    video_labels: torch.Tensor
) -> torch.Tensor:
    """
    Compute video-level classification loss
    
    Args:
        video_logit: [B, 1] video logits
        video_labels: [B] video labels (0=real, 1=fake)
    
    Returns:
        Scalar loss
    """
    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(video_logit.squeeze(-1), video_labels.float())
    return loss


def compute_temporal_smoothness_loss(
    frame_probs: torch.Tensor,
    mask: torch.Tensor
) -> torch.Tensor:
    """
    Temporal smoothness regularization
    Encourages adjacent frames to have similar predictions
    
    Args:
        frame_probs: [B, T] frame probabilities (after sigmoid)
        mask: [B, T] padding mask
    
    Returns:
        Scalar loss
    """
    B, T = frame_probs.shape
    
    if T <= 1:
        return torch.tensor(0.0, device=frame_probs.device)
    
    # Compute absolute difference between adjacent frames
    diff = torch.abs(frame_probs[:, 1:] - frame_probs[:, :-1])  # [B, T-1]
    
    # Mask out padding (exclude if either frame is padding)
    valid_mask = ~mask[:, 1:] & ~mask[:, :-1]  # [B, T-1]
    
    if valid_mask.sum() == 0:
        return torch.tensor(0.0, device=frame_probs.device)
    
    # Average over valid positions
    loss = (diff * valid_mask).sum() / valid_mask.sum()
    
    return loss


if __name__ == '__main__':
    # Test model
    print("[TEST] Creating model...")
    model = FrameLocalizationModel(
        v_dim=512,
        a_dim=1024,
        d_model=512,
        nhead=8,
        num_layers=4,
        use_cross_attn=True,
        use_video_head=True
    )
    
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # Test forward pass
    B, T = 4, 256
    visual = torch.randn(B, T, 512)
    audio = torch.randn(B, T, 1024)
    mask = torch.zeros(B, T, dtype=torch.bool)
    mask[:, 200:] = True  # Mask last 56 frames
    
    print(f"\n[TEST] Forward pass...")
    frame_logits, video_logit = model(visual, audio, mask)
    print(f"Frame logits: {frame_logits.shape}")
    print(f"Video logit: {video_logit.shape}")
    
    # Test loss computation
    frame_labels = torch.randint(0, 2, (B, T))
    video_labels = torch.randint(0, 2, (B,))
    
    frame_loss = compute_frame_loss(frame_logits, frame_labels, mask, pos_weight=5.0)
    video_loss = compute_video_loss(video_logit, video_labels)
    
    frame_probs = torch.sigmoid(frame_logits.squeeze(-1))
    smooth_loss = compute_temporal_smoothness_loss(frame_probs, mask)
    
    print(f"\n[TEST] Losses:")
    print(f"Frame loss: {frame_loss.item():.4f}")
    print(f"Video loss: {video_loss.item():.4f}")
    print(f"Smoothness loss: {smooth_loss.item():.4f}")
    
    print(f"\n[TEST] Model test passed!")

