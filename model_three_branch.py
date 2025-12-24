#!/usr/bin/env python3
"""
Three-Branch Joint Training Model for Deepfake Detection
Combines: Cross-Modal + Audio-Only + Visual-Only (DFD-FCG) branches
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class AudioOnlyBranch(nn.Module):
    """
    Pure audio-based detection branch
    Processes only audio features for deepfake detection
    """
    def __init__(
        self,
        a_dim: int = 1024,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Audio encoder
        self.audio_encoder = nn.Sequential(
            nn.Linear(a_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Temporal modeling
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout * 2),
            nn.Linear(d_model // 2, 1)
        )
    
    def extract_features(self, audio: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Extract audio features without classification"""
        audio_feat = self.audio_encoder(audio)  # [B, T, d_model]
        encoded = self.temporal_encoder(audio_feat, src_key_padding_mask=mask)
        
        # Pooling
        if mask is not None:
            masked_encoded = encoded.masked_fill(mask.unsqueeze(-1), 0)
            valid_counts = (~mask).sum(dim=1, keepdim=True).clamp(min=1)
            pooled = masked_encoded.sum(dim=1) / valid_counts
        else:
            pooled = encoded.mean(dim=1)
        
        return pooled  # [B, d_model]
    
    def forward(self, audio: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with classification"""
        features = self.extract_features(audio, mask)
        logits = self.classifier(features)  # [B, 1]
        return logits


class VisualOnlyBranch(nn.Module):
    """
    Pure visual-based detection branch
    Can use DFD-FCG foundation model or InsightFace features
    """
    def __init__(
        self,
        v_dim: int = 512,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 3,
        dropout: float = 0.1,
        use_dfdfcg: bool = False,
        dfdfcg_pretrain: str = None,
        dfdfcg_freeze: bool = True
    ):
        super().__init__()
        
        self.use_dfdfcg = use_dfdfcg
        
        if use_dfdfcg:
            # Use DFD-FCG CLIP encoder
            try:
                from foundation_encoder_dfdfcg import DFDFCGEncoder
                self.dfdfcg_encoder = DFDFCGEncoder(
                    architecture="ViT-L/14",
                    pretrain_path=dfdfcg_pretrain,
                    freeze=dfdfcg_freeze,
                    text_embed=False
                )
                pretrained_dim = self.dfdfcg_encoder.output_dim  # 768 for ViT-L/14
                
                # Adapter from CLIP to model dimension
                self.visual_encoder = nn.Sequential(
                    nn.Linear(pretrained_dim, d_model),
                    nn.LayerNorm(d_model),
                    nn.GELU(),
                    nn.Dropout(dropout)
                )
                print(f"[INFO] Visual-Only branch using DFD-FCG CLIP (dim={pretrained_dim})")
            except ImportError as e:
                print(f"[WARN] Failed to load DFD-FCG: {e}")
                print(f"[WARN] Falling back to InsightFace features")
                self.use_dfdfcg = False
                self.visual_encoder = nn.Sequential(
                    nn.Linear(v_dim, d_model),
                    nn.LayerNorm(d_model),
                    nn.GELU(),
                    nn.Dropout(dropout)
                )
        else:
            # Use InsightFace features
            self.visual_encoder = nn.Sequential(
                nn.Linear(v_dim, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Dropout(dropout)
            )
        
        # Temporal modeling
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout * 2),
            nn.Linear(d_model // 2, 1)
        )
    
    def extract_features(self, visual: torch.Tensor, mask: Optional[torch.Tensor] = None, video_frames: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Extract visual features without classification
        
        Args:
            visual: [B, T, v_dim] pre-extracted features (InsightFace)
            mask: [B, T] padding mask
            video_frames: [B, T, 3, H, W] raw video frames (only for DFD-FCG)
        """
        if self.use_dfdfcg:
            # Use DFD-FCG to extract features from raw frames
            if video_frames is None:
                raise ValueError("video_frames required when use_dfdfcg=True")
            
            # Extract CLIP features
            clip_features = self.dfdfcg_encoder(video_frames)  # [B, T, 768]
            visual_feat = self.visual_encoder(clip_features)
        else:
            # Use pre-extracted InsightFace features
            visual_feat = self.visual_encoder(visual)
        
        # Temporal encoding
        encoded = self.temporal_encoder(visual_feat, src_key_padding_mask=mask)
        
        # Pooling
        if mask is not None:
            masked_encoded = encoded.masked_fill(mask.unsqueeze(-1), 0)
            valid_counts = (~mask).sum(dim=1, keepdim=True).clamp(min=1)
            pooled = masked_encoded.sum(dim=1) / valid_counts
        else:
            pooled = encoded.mean(dim=1)
        
        return pooled  # [B, d_model]
    
    def forward(self, visual: torch.Tensor, mask: Optional[torch.Tensor] = None, video_frames: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with classification
        
        Args:
            visual: [B, T, v_dim] pre-extracted features
            mask: [B, T] padding mask
            video_frames: [B, T, 3, H, W] raw frames (for DFD-FCG)
        """
        features = self.extract_features(visual, mask, video_frames)
        logits = self.classifier(features)  # [B, 1]
        return logits


class CrossModalBranch(nn.Module):
    """
    Cross-modal branch with bidirectional attention
    Based on existing CrossModalTransformer architecture
    """
    def __init__(
        self,
        v_dim: int = 512,
        a_dim: int = 1024,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Feature projection
        self.v_proj = nn.Linear(v_dim, d_model)
        self.a_proj = nn.Linear(a_dim, d_model)
        
        # Bidirectional cross-attention
        self.cross_attn_v2a = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.cross_attn_a2v = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.cross_norm_v = nn.LayerNorm(d_model)
        self.cross_norm_a = nn.LayerNorm(d_model)
        
        # Temporal Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout * 2),
            nn.Linear(d_model // 2, 1)
        )
    
    def extract_features(
        self, 
        audio: torch.Tensor, 
        visual: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Extract cross-modal features without classification"""
        # Project to common space
        v = self.v_proj(visual)
        a = self.a_proj(audio)
        
        # Bidirectional cross-attention
        v_enhanced, _ = self.cross_attn_v2a(v, a, a, key_padding_mask=mask)
        v_enhanced = self.cross_norm_v(v + v_enhanced)
        
        a_enhanced, _ = self.cross_attn_a2v(a, v, v, key_padding_mask=mask)
        a_enhanced = self.cross_norm_a(a + a_enhanced)
        
        # Fuse
        fused = (v_enhanced + a_enhanced) / 2
        
        # Temporal encoding
        encoded = self.temporal_encoder(fused, src_key_padding_mask=mask)
        
        # Pooling
        if mask is not None:
            masked_encoded = encoded.masked_fill(mask.unsqueeze(-1), 0)
            valid_counts = (~mask).sum(dim=1, keepdim=True).clamp(min=1)
            pooled = masked_encoded.sum(dim=1) / valid_counts
        else:
            pooled = encoded.mean(dim=1)
        
        return pooled  # [B, d_model]
    
    def forward(
        self, 
        audio: torch.Tensor, 
        visual: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass with classification"""
        features = self.extract_features(audio, visual, mask)
        logits = self.classifier(features)  # [B, 1]
        return logits


class ThreeBranchJointModel(nn.Module):
    """
    Three-branch joint training model
    
    Branches:
    1. Cross-Modal: Audio + Visual fusion
    2. Audio-Only: Pure audio analysis
    3. Visual-Only: Pure visual analysis (can use pre-trained DFD-FCG)
    
    Training: All branches trained jointly with multi-task loss
    Inference: Feature-level fusion for final prediction
    """
    def __init__(
        self,
        v_dim: int = 512,
        a_dim: int = 1024,
        d_model: int = 512,
        nhead: int = 8,
        cm_layers: int = 4,
        ao_layers: int = 3,
        vo_layers: int = 3,
        dropout: float = 0.1,
        fusion_method: str = 'concat',  # 'concat', 'weighted', 'attention'
        use_dfdfcg: bool = False,
        dfdfcg_pretrain: str = None,
        dfdfcg_freeze: bool = True
    ):
        """
        Args:
            v_dim: Visual feature dimension (InsightFace)
            a_dim: Audio feature dimension
            d_model: Hidden dimension
            nhead: Number of attention heads
            cm_layers: Number of layers in cross-modal branch
            ao_layers: Number of layers in audio-only branch
            vo_layers: Number of layers in visual-only branch
            dropout: Dropout rate
            fusion_method: How to fuse branch features ('concat', 'weighted', 'attention')
            use_dfdfcg: Use DFD-FCG CLIP encoder for visual-only branch
            dfdfcg_pretrain: Path to DFD-FCG pre-trained weights
            dfdfcg_freeze: Freeze DFD-FCG encoder weights
        """
        super().__init__()
        
        self.fusion_method = fusion_method
        self.use_dfdfcg = use_dfdfcg
        
        # Three branches
        self.cross_modal_branch = CrossModalBranch(
            v_dim=v_dim, a_dim=a_dim, d_model=d_model,
            nhead=nhead, num_layers=cm_layers, dropout=dropout
        )
        
        self.audio_only_branch = AudioOnlyBranch(
            a_dim=a_dim, d_model=d_model,
            nhead=nhead, num_layers=ao_layers, dropout=dropout
        )
        
        self.visual_only_branch = VisualOnlyBranch(
            v_dim=v_dim, d_model=d_model,
            nhead=nhead, num_layers=vo_layers, dropout=dropout,
            use_dfdfcg=use_dfdfcg,
            dfdfcg_pretrain=dfdfcg_pretrain,
            dfdfcg_freeze=dfdfcg_freeze
        )
        
        # Fusion module
        if fusion_method == 'concat':
            # Simple concatenation + MLP
            self.fusion_layer = nn.Sequential(
                nn.Linear(d_model * 3, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, 1)
            )
        
        elif fusion_method == 'weighted':
            # Learnable weighted sum
            self.branch_weights = nn.Parameter(torch.ones(3) / 3)
            self.fusion_layer = nn.Linear(d_model, 1)
        
        elif fusion_method == 'attention':
            # Attention-based fusion
            self.attn_query = nn.Parameter(torch.randn(1, 1, d_model))
            self.attn_layer = nn.MultiheadAttention(d_model, nhead, batch_first=True)
            self.fusion_layer = nn.Linear(d_model, 1)
    
    def forward(
        self,
        audio: torch.Tensor,
        visual: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_branch_outputs: bool = False,
        video_frames: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through all branches
        
        Args:
            audio: [B, T, a_dim] audio features
            visual: [B, T, v_dim] visual features (InsightFace)
            mask: [B, T] padding mask
            return_branch_outputs: Whether to return individual branch predictions
            video_frames: [B, T, 3, H, W] raw video frames (for DFD-FCG visual-only branch)
        
        Returns:
            Dictionary containing:
                - fused_logit: Final fused prediction
                - branch_logits: Individual branch predictions (if return_branch_outputs=True)
                - branch_features: Individual branch features (if return_branch_outputs=True)
                - fusion_weights: Branch fusion weights (if applicable)
        """
        B = audio.size(0)
        
        # Extract features from all branches
        cm_feat = self.cross_modal_branch.extract_features(audio, visual, mask)  # [B, d_model]
        ao_feat = self.audio_only_branch.extract_features(audio, mask)           # [B, d_model]
        vo_feat = self.visual_only_branch.extract_features(visual, mask, video_frames)  # [B, d_model]
        
        # Individual branch predictions (for multi-task loss)
        cm_logit = self.cross_modal_branch.classifier(cm_feat)  # [B, 1]
        ao_logit = self.audio_only_branch.classifier(ao_feat)   # [B, 1]
        vo_logit = self.visual_only_branch.classifier(vo_feat)  # [B, 1]
        
        # Fuse features
        if self.fusion_method == 'concat':
            # Concatenate features
            concat_feat = torch.cat([cm_feat, ao_feat, vo_feat], dim=-1)  # [B, d_model*3]
            fused_logit = self.fusion_layer(concat_feat)  # [B, 1]
            fusion_weights = None
        
        elif self.fusion_method == 'weighted':
            # Weighted sum of features
            weights = F.softmax(self.branch_weights, dim=0)
            fused_feat = (
                weights[0] * cm_feat +
                weights[1] * ao_feat +
                weights[2] * vo_feat
            )
            fused_logit = self.fusion_layer(fused_feat)  # [B, 1]
            fusion_weights = weights
        
        elif self.fusion_method == 'attention':
            # Attention-based fusion
            branch_feats = torch.stack([cm_feat, ao_feat, vo_feat], dim=1)  # [B, 3, d_model]
            query = self.attn_query.expand(B, -1, -1)  # [B, 1, d_model]
            
            fused_feat, attn_weights = self.attn_layer(
                query, branch_feats, branch_feats
            )  # [B, 1, d_model], [B, 1, 3]
            
            fused_logit = self.fusion_layer(fused_feat.squeeze(1))  # [B, 1]
            fusion_weights = attn_weights.squeeze(1)  # [B, 3]
        
        # Prepare output
        output = {'fused_logit': fused_logit}
        
        if return_branch_outputs:
            output['branch_logits'] = {
                'cross_modal': cm_logit,
                'audio_only': ao_logit,
                'visual_only': vo_logit
            }
            output['branch_features'] = {
                'cross_modal': cm_feat,
                'audio_only': ao_feat,
                'visual_only': vo_feat
            }
        
        if fusion_weights is not None:
            output['fusion_weights'] = fusion_weights
        
        return output


def compute_three_branch_loss(
    outputs: Dict[str, torch.Tensor],
    labels: torch.Tensor,
    branch_weights: Tuple[float, float, float] = (0.3, 0.2, 0.2),
    fusion_weight: float = 1.0
) -> Dict[str, torch.Tensor]:
    """
    Compute multi-task loss for three-branch model
    
    Args:
        outputs: Model outputs containing fused_logit and branch_logits
        labels: Ground truth labels [B]
        branch_weights: Weights for (cross_modal, audio_only, visual_only) branch losses
        fusion_weight: Weight for fused prediction loss
    
    Returns:
        Dictionary of losses
    """
    criterion = nn.BCEWithLogitsLoss()
    
    # Fused loss (main task)
    loss_fused = criterion(outputs['fused_logit'].squeeze(-1), labels.float())
    
    # Branch losses (auxiliary tasks)
    if 'branch_logits' in outputs:
        loss_cm = criterion(outputs['branch_logits']['cross_modal'].squeeze(-1), labels.float())
        loss_ao = criterion(outputs['branch_logits']['audio_only'].squeeze(-1), labels.float())
        loss_vo = criterion(outputs['branch_logits']['visual_only'].squeeze(-1), labels.float())
    else:
        loss_cm = loss_ao = loss_vo = torch.tensor(0.0, device=labels.device)
    
    # Total loss
    loss_total = (
        fusion_weight * loss_fused +
        branch_weights[0] * loss_cm +
        branch_weights[1] * loss_ao +
        branch_weights[2] * loss_vo
    )
    
    return {
        'total': loss_total,
        'fused': loss_fused,
        'cross_modal': loss_cm,
        'audio_only': loss_ao,
        'visual_only': loss_vo
    }


if __name__ == '__main__':
    # Test model
    print("[TEST] Creating three-branch model...")
    
    model = ThreeBranchJointModel(
        v_dim=512,
        a_dim=1024,
        d_model=512,
        nhead=8,
        fusion_method='attention'
    )
    
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Total parameters: {total_params:.2f}M")
    
    # Test forward pass
    B, T = 4, 128
    audio = torch.randn(B, T, 1024)
    visual = torch.randn(B, T, 512)
    mask = torch.zeros(B, T, dtype=torch.bool)
    mask[:, 100:] = True
    labels = torch.randint(0, 2, (B,))
    
    print(f"\n[TEST] Forward pass...")
    outputs = model(audio, visual, mask, return_branch_outputs=True)
    
    print(f"Fused logit: {outputs['fused_logit'].shape}")
    print(f"Branch logits: {[v.shape for v in outputs['branch_logits'].values()]}")
    
    if 'fusion_weights' in outputs:
        print(f"Fusion weights: {outputs['fusion_weights'].shape}")
        print(f"Sample weights: {outputs['fusion_weights'][0]}")
    
    # Test loss
    print(f"\n[TEST] Computing loss...")
    losses = compute_three_branch_loss(outputs, labels)
    
    for name, loss in losses.items():
        print(f"{name} loss: {loss.item():.4f}")
    
    print(f"\n[TEST] All tests passed!")

