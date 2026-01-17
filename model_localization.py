#!/usr/bin/env python3
"""
Frame-Level Deepfake Localization Model
Cross-modal Transformer with temporal encoding for frame-wise classification

Enhanced with:
- Learned inconsistency scoring (replaces simple cosine)
- Soft reliability gating (reduces false positives on uncertain frames)
- Logit-level fusion with learnable weight
- Support for ranking loss with hard negatives
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict


class LearnedInconsistencyModule(nn.Module):
    """
    Learned inconsistency scoring network.
    Uses multiple cues (cosine, diff, product) to detect audio-visual mismatch.
    Much more robust than raw cosine similarity.
    """
    def __init__(self, d_model: int = 512, dropout: float = 0.1):
        super().__init__()
        
        # Input: v (d_model) + a (d_model) + diff (d_model) + prod (d_model) + sim (1)
        input_dim = d_model * 4 + 1
        
        self.layer_norm = nn.LayerNorm(input_dim)
        
        # Lightweight MLP
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
    
    def forward(self, v: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """
        Args:
            v: [B, T, d_model] visual features
            a: [B, T, d_model] audio features
        
        Returns:
            inconsistency: [B, T, 1] higher = more inconsistent/fake
        """
        # 1. Cosine similarity (explicit cue)
        v_norm = F.normalize(v, dim=-1, eps=1e-8)
        a_norm = F.normalize(a, dim=-1, eps=1e-8)
        sim = (v_norm * a_norm).sum(dim=-1, keepdim=True)  # [B, T, 1]
        
        # 2. Absolute difference (more stable than cosine)
        diff = torch.abs(v - a)  # [B, T, d_model]
        
        # 3. Element-wise product (captures interaction patterns)
        prod = v * a  # [B, T, d_model]
        
        # 4. Concatenate all cues
        z = torch.cat([v, a, diff, prod, sim], dim=-1)  # [B, T, 4*d_model+1]
        
        # 5. Layer norm + MLP
        z = self.layer_norm(z)
        inconsistency = self.mlp(z)  # [B, T, 1]
        
        return inconsistency


class SoftReliabilityGating(nn.Module):
    """
    Soft reliability gating using a learnable network.
    Prevents false positives on uncertain frames (silence, B-roll, cuts).
    """
    def __init__(self, v_dim: int = 512, a_dim: int = 1024, hidden_dim: int = 128):
        super().__init__()
        
        # Learnable gating network
        self.gate_net = nn.Sequential(
            nn.Linear(v_dim + a_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Output in [0, 1]
        )
    
    def forward(self, visual: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        """
        Args:
            visual: [B, T, v_dim] raw visual features
            audio: [B, T, a_dim] raw audio features
        
        Returns:
            gate: [B, T, 1] reliability score in [0, 1]
        """
        concat = torch.cat([visual, audio], dim=-1)  # [B, T, v_dim+a_dim]
        gate = self.gate_net(concat)  # [B, T, 1]
        return gate


class FrameLocalizationModel(nn.Module):
    """
    Enhanced frame-level deepfake localization model.
    
    Architecture:
    1. Project visual/audio features to common dimension
    2. Cross-modal attention (bidirectional)
    3. Temporal Transformer encoder
    4. Learned inconsistency scoring (replaces simple cosine)
    5. Soft reliability gating
    6. Logit-level fusion with learnable weight
    7. Frame-level + video-level classifiers
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
        use_video_head: bool = True,
        use_inconsistency_module: bool = True,
        use_reliability_gating: bool = True,
        use_boundary_head: bool = True,
        alpha_init: float = 0.5,
        temperature: float = 0.1
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
            use_inconsistency_module: Use learned inconsistency scoring
            use_reliability_gating: Use soft reliability gating
            use_boundary_head: Add start/end boundary detection heads
            alpha_init: Initial weight for inconsistency branch fusion
            temperature: Temperature for inconsistency score scaling
        """
        super().__init__()
        
        self.d_model = d_model
        self.use_cross_attn = use_cross_attn
        self.use_video_head = use_video_head
        self.use_inconsistency_module = use_inconsistency_module
        self.use_reliability_gating = use_reliability_gating
        self.use_boundary_head = use_boundary_head
        self.temperature = temperature
        
        # Feature projection
        self.v_proj = nn.Linear(v_dim, d_model)
        self.a_proj = nn.Linear(a_dim, d_model)
        
        # Learned inconsistency module
        if use_inconsistency_module:
            self.inconsistency_module = LearnedInconsistencyModule(d_model, dropout)
            # Learnable fusion weight (logit-level fusion)
            self.alpha = nn.Parameter(torch.tensor(alpha_init))
        
        # Soft reliability gating
        if use_reliability_gating:
            self.reliability_gating = SoftReliabilityGating(v_dim, a_dim)
        
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
        
        # Frame-level classifier (main branch, no concatenation)
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
        
        # Boundary detection heads (extremely lightweight)
        if use_boundary_head:
            self.start_classifier = nn.Linear(d_model, 1)
            self.end_classifier = nn.Linear(d_model, 1)
    
    def forward(
        self, 
        visual: torch.Tensor, 
        audio: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            visual: [B, T, v_dim] visual features
            audio: [B, T, a_dim] audio features
            mask: [B, T] bool mask (True = padding)
        
        Returns:
            Dict containing:
                - frame_logits: [B, T, 1] final per-frame logits (main + inconsistency)
                - video_logit: [B, 1] video-level logit (optional)
                - frame_logits_main: [B, T, 1] main branch logits only
                - inconsistency_score: [B, T, 1] raw inconsistency score
                - inconsistency_gated: [B, T, 1] gated inconsistency score
                - reliability_gate: [B, T, 1] reliability gate values
                - start_logits: [B, T, 1] start boundary logits (optional)
                - end_logits: [B, T, 1] end boundary logits (optional)
        """
        B, T, _ = visual.shape
        
        # Project to common dimension
        v = self.v_proj(visual)  # [B, T, d_model]
        a = self.a_proj(audio)   # [B, T, d_model]
        
        # Compute reliability gate (before cross-attention)
        reliability_gate = None
        if self.use_reliability_gating:
            reliability_gate = self.reliability_gating(visual, audio)  # [B, T, 1]
        
        # Compute learned inconsistency score
        inconsistency_score = None
        inconsistency_gated = None
        if self.use_inconsistency_module:
            inconsistency_score = self.inconsistency_module(v, a)  # [B, T, 1]
            
            # Apply reliability gating
            if reliability_gate is not None:
                inconsistency_gated = reliability_gate * inconsistency_score
            else:
                inconsistency_gated = inconsistency_score
        
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
        
        # Main branch: frame-level classification
        frame_logits_main = self.frame_classifier(encoded)  # [B, T, 1]
        
        # Logit-level fusion: frame_logit = L_t + alpha * I_t / temperature
        if self.use_inconsistency_module and inconsistency_gated is not None:
            frame_logits = frame_logits_main + self.alpha * (inconsistency_gated / self.temperature)
        else:
            frame_logits = frame_logits_main
        
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
        
        # Boundary detection (start/end)
        start_logits = None
        end_logits = None
        if self.use_boundary_head:
            start_logits = self.start_classifier(encoded)  # [B, T, 1]
            end_logits = self.end_classifier(encoded)      # [B, T, 1]
        
        return {
            'frame_logits': frame_logits,
            'video_logit': video_logit,
            'frame_logits_main': frame_logits_main,
            'inconsistency_score': inconsistency_score,
            'inconsistency_gated': inconsistency_gated,
            'reliability_gate': reliability_gate,
            'start_logits': start_logits,
            'end_logits': end_logits
        }


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


def generate_boundary_labels(
    frame_labels: torch.Tensor,
    mask: torch.Tensor,
    tolerance: int = 5
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate start/end boundary labels with tolerance window.
    Handles noise in frame-level annotations by allowing ±tolerance margin.
    
    Args:
        frame_labels: [B, T] binary frame labels (0=real, 1=fake)
        mask: [B, T] padding mask (True=padding)
        tolerance: Boundary tolerance in frames (±delta)
    
    Returns:
        start_labels: [B, T] binary (1=start boundary)
        end_labels: [B, T] binary (1=end boundary)
    """
    B, T = frame_labels.shape
    device = frame_labels.device
    
    start_labels = torch.zeros_like(frame_labels)
    end_labels = torch.zeros_like(frame_labels)
    
    for b in range(B):
        labels = frame_labels[b]
        valid_mask = ~mask[b]
        
        if valid_mask.sum() == 0:
            continue
        
        # Find transitions (only in valid region)
        labels_padded = torch.cat([torch.zeros(1, device=device), labels, torch.zeros(1, device=device)])
        diff = labels_padded[1:] - labels_padded[:-1]  # [T+1]
        diff = diff[:-1]  # Remove last element -> [T]
        
        # Start points: 0→1 transition (diff == 1)
        start_points = torch.where((diff == 1) & valid_mask)[0]
        for start in start_points:
            left = max(0, start - tolerance)
            right = min(T, start + tolerance + 1)
            start_labels[b, left:right] = 1
        
        # End points: 1→0 transition (diff == -1)
        end_points = torch.where((diff == -1) & valid_mask)[0]
        for end in end_points:
            # End is the last fake frame (before transition)
            end = end - 1 if end > 0 else 0
            left = max(0, end - tolerance)
            right = min(T, end + tolerance + 1)
            end_labels[b, left:right] = 1
        
        # Handle edge cases: segment at video start
        if valid_mask[0] and labels[0] == 1:
            start_labels[b, :min(T, tolerance + 1)] = 1
        
        # Handle edge cases: segment at video end
        valid_end = valid_mask.sum().item()
        if valid_end > 0 and labels[valid_end - 1] == 1:
            end_labels[b, max(0, valid_end - tolerance - 1):valid_end] = 1
    
    return start_labels, end_labels


def compute_boundary_loss(
    start_logits: torch.Tensor,
    end_logits: torch.Tensor,
    start_labels: torch.Tensor,
    end_labels: torch.Tensor,
    mask: torch.Tensor,
    focal_gamma: float = 2.0,
    focal_alpha: float = 0.25
) -> torch.Tensor:
    """
    Focal loss for extremely sparse boundary labels.
    Boundary labels are <0.5% of all frames, so Focal Loss is critical.
    
    Args:
        start_logits: [B, T, 1] start boundary logits
        end_logits: [B, T, 1] end boundary logits
        start_labels: [B, T] start boundary labels
        end_labels: [B, T] end boundary labels
        mask: [B, T] padding mask
        focal_gamma: Focal loss gamma parameter
        focal_alpha: Focal loss alpha parameter
    
    Returns:
        Scalar boundary loss (start + end)
    """
    # Flatten and filter valid frames
    valid = ~mask
    
    start_logits_flat = start_logits.squeeze(-1)[valid]
    end_logits_flat = end_logits.squeeze(-1)[valid]
    start_labels_flat = start_labels[valid].float()
    end_labels_flat = end_labels[valid].float()
    
    if len(start_logits_flat) == 0:
        return torch.tensor(0.0, device=start_logits.device)
    
    # Focal loss for start
    bce_start = F.binary_cross_entropy_with_logits(
        start_logits_flat, start_labels_flat, reduction='none'
    )
    p_start = torch.sigmoid(start_logits_flat)
    pt_start = start_labels_flat * p_start + (1 - start_labels_flat) * (1 - p_start)
    focal_weight_start = (1 - pt_start) ** focal_gamma
    loss_start = (focal_alpha * focal_weight_start * bce_start).mean()
    
    # Focal loss for end
    bce_end = F.binary_cross_entropy_with_logits(
        end_logits_flat, end_labels_flat, reduction='none'
    )
    p_end = torch.sigmoid(end_logits_flat)
    pt_end = end_labels_flat * p_end + (1 - end_labels_flat) * (1 - p_end)
    focal_weight_end = (1 - pt_end) ** focal_gamma
    loss_end = (focal_alpha * focal_weight_end * bce_end).mean()
    
    return loss_start + loss_end


def compute_boundary_aware_smooth_loss(
    frame_probs: torch.Tensor,
    frame_labels: torch.Tensor,
    mask: torch.Tensor
) -> torch.Tensor:
    """
    Boundary-aware smoothness loss.
    Only smooths within same-label segments, preserving boundaries.
    Critical improvement over global smoothing which blurs boundaries.
    
    Args:
        frame_probs: [B, T] frame probabilities (after sigmoid)
        frame_labels: [B, T] GT frame labels
        mask: [B, T] padding mask
    
    Returns:
        Scalar loss
    """
    B, T = frame_probs.shape
    
    if T <= 1:
        return torch.tensor(0.0, device=frame_probs.device)
    
    # Identify where adjacent frames have same label
    same_label = (frame_labels[:, 1:] == frame_labels[:, :-1])  # [B, T-1]
    
    # Both frames must be valid (not padding)
    both_valid = ~mask[:, 1:] & ~mask[:, :-1]  # [B, T-1]
    
    # Smooth only within same-label segments
    smooth_mask = same_label & both_valid
    
    if smooth_mask.sum() == 0:
        return torch.tensor(0.0, device=frame_probs.device)
    
    # Compute squared difference
    diff = (frame_probs[:, 1:] - frame_probs[:, :-1]) ** 2  # [B, T-1]
    
    # Apply mask and average
    loss = (diff * smooth_mask.float()).sum() / smooth_mask.sum()
    
    return loss


def compute_ranking_loss(
    inconsistency_pos: torch.Tensor,
    inconsistency_neg: torch.Tensor,
    reliability_gate: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
    margin: float = 0.3
) -> torch.Tensor:
    """
    Margin ranking loss for inconsistency learning.
    Encourages: I_neg (misaligned) > I_pos (aligned) + margin
    
    Args:
        inconsistency_pos: [B, T, 1] inconsistency for correct alignment
        inconsistency_neg: [B, T, 1] inconsistency for wrong alignment (hard negatives)
        reliability_gate: [B, T, 1] reliability gate (optional, only compute loss on reliable frames)
        mask: [B, T] padding mask
        margin: Margin for ranking loss
    
    Returns:
        Scalar loss
    """
    # Squeeze to [B, T]
    I_pos = inconsistency_pos.squeeze(-1)
    I_neg = inconsistency_neg.squeeze(-1)
    
    # Margin ranking: max(0, margin - (I_neg - I_pos))
    loss_per_frame = F.relu(margin - (I_neg - I_pos))  # [B, T]
    
    # Apply reliability gating if provided (only compute loss on reliable frames)
    if reliability_gate is not None:
        gate = reliability_gate.squeeze(-1)  # [B, T]
        loss_per_frame = loss_per_frame * gate
    
    # Apply padding mask
    if mask is not None:
        loss_per_frame = loss_per_frame.masked_fill(mask, 0)
        valid_count = (~mask).sum()
        if reliability_gate is not None:
            # Count only unmasked AND reliable frames
            valid_count = ((~mask) & (gate > 0.5)).sum()
    else:
        if reliability_gate is not None:
            valid_count = (gate > 0.5).sum()
        else:
            valid_count = loss_per_frame.numel()
    
    if valid_count == 0:
        return torch.tensor(0.0, device=inconsistency_pos.device)
    
    loss = loss_per_frame.sum() / valid_count
    return loss


def compute_fake_hinge_loss(
    inconsistency_gated: torch.Tensor,
    frame_labels: torch.Tensor,
    mask: torch.Tensor,
    margin_sync: float = 0.5
) -> torch.Tensor:
    """
    Optional: Hinge loss to prevent fake frames from having too high consistency.
    L = mean(relu((1 - I_eff) - margin_sync) for y==1)
    
    Args:
        inconsistency_gated: [B, T, 1] gated inconsistency score
        frame_labels: [B, T] frame labels (0=real, 1=fake)
        mask: [B, T] padding mask
        margin_sync: Margin threshold
    
    Returns:
        Scalar loss
    """
    I_eff = inconsistency_gated.squeeze(-1)  # [B, T]
    
    # Only compute on fake frames (label==1)
    fake_mask = (frame_labels == 1)
    if mask is not None:
        fake_mask = fake_mask & (~mask)
    
    if fake_mask.sum() == 0:
        return torch.tensor(0.0, device=inconsistency_gated.device)
    
    # Hinge: max(0, (1 - I_eff) - margin_sync) on fake frames
    # Penalize if fake frames have low inconsistency (too consistent)
    consistency_score = 1.0 - I_eff  # High = consistent
    loss_per_frame = F.relu(consistency_score - margin_sync)
    
    # Apply mask
    loss_per_frame = loss_per_frame * fake_mask.float()
    
    loss = loss_per_frame.sum() / fake_mask.sum()
    return loss


def compute_combined_loss(
    frame_logits: torch.Tensor,
    frame_labels: torch.Tensor,
    mask: torch.Tensor,
    video_logit: Optional[torch.Tensor] = None,
    video_label: Optional[torch.Tensor] = None,
    start_logits: Optional[torch.Tensor] = None,
    end_logits: Optional[torch.Tensor] = None,
    inconsistency_pos: Optional[torch.Tensor] = None,
    inconsistency_neg: Optional[torch.Tensor] = None,
    inconsistency_gated: Optional[torch.Tensor] = None,
    reliability_gate: Optional[torch.Tensor] = None,
    frame_loss_weight: float = 1.0,
    video_loss_weight: float = 0.3,
    boundary_loss_weight: float = 0.5,
    smooth_loss_weight: float = 0.05,
    ranking_loss_weight: float = 0.5,
    fake_hinge_weight: float = 0.05,
    boundary_tolerance: int = 5,
    ranking_margin: float = 0.3,
    use_boundary_aware_smooth: bool = True,
    pos_weight: Optional[float] = None,
    use_focal: bool = False,
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0
) -> Dict[str, torch.Tensor]:
    """
    Compute combined loss with frame, video, boundary, smooth, ranking, and fake hinge losses.
    
    Args:
        frame_logits: [B, T, 1] final frame logits
        frame_labels: [B, T] frame labels
        mask: [B, T] padding mask
        video_logit: [B, 1] video logit (optional)
        video_label: [B] video label (optional)
        start_logits: [B, T, 1] start boundary logits (optional)
        end_logits: [B, T, 1] end boundary logits (optional)
        inconsistency_pos: [B, T, 1] inconsistency for correct alignment (optional)
        inconsistency_neg: [B, T, 1] inconsistency for misalignment (optional)
        inconsistency_gated: [B, T, 1] gated inconsistency (optional)
        reliability_gate: [B, T, 1] reliability gate (optional)
        frame_loss_weight: Weight for frame BCE/Focal loss
        video_loss_weight: Weight for video loss
        boundary_loss_weight: Weight for boundary loss (start + end)
        smooth_loss_weight: Weight for smoothness loss
        ranking_loss_weight: Weight for ranking loss
        fake_hinge_weight: Weight for fake hinge loss (optional)
        boundary_tolerance: Tolerance window for boundary labels (±frames)
        ranking_margin: Margin for ranking loss
        use_boundary_aware_smooth: Use boundary-aware smooth (recommended)
        pos_weight: Positive class weight for BCE
        use_focal: Use Focal Loss instead of BCE
        focal_alpha: Focal loss alpha
        focal_gamma: Focal loss gamma
    
    Returns:
        Dict with 'total', 'frame', 'video', 'boundary', 'smooth', 'ranking', 'fake_hinge' losses
    """
    # Frame loss
    frame_loss = compute_frame_loss(
        frame_logits, frame_labels, mask,
        pos_weight=pos_weight,
        use_focal=use_focal,
        focal_alpha=focal_alpha,
        focal_gamma=focal_gamma
    )
    
    # Video loss
    video_loss = torch.tensor(0.0, device=frame_logits.device)
    if video_logit is not None and video_label is not None:
        video_loss = compute_video_loss(video_logit, video_label)
    
    # Boundary loss (NEW: critical for precise localization)
    boundary_loss = torch.tensor(0.0, device=frame_logits.device)
    if boundary_loss_weight > 0 and start_logits is not None and end_logits is not None:
        # Generate boundary labels with tolerance
        start_labels, end_labels = generate_boundary_labels(
            frame_labels, mask, tolerance=boundary_tolerance
        )
        boundary_loss = compute_boundary_loss(
            start_logits, end_logits, start_labels, end_labels, mask,
            focal_gamma=focal_gamma, focal_alpha=focal_alpha
        )
    
    # Smoothness loss
    smooth_loss = torch.tensor(0.0, device=frame_logits.device)
    if smooth_loss_weight > 0:
        frame_probs = torch.sigmoid(frame_logits.squeeze(-1))
        if use_boundary_aware_smooth:
            # Boundary-aware: only smooth within same-label segments
            smooth_loss = compute_boundary_aware_smooth_loss(
                frame_probs, frame_labels, mask
            )
        else:
            # Original: global smoothness (not recommended with boundary head)
            smooth_loss = compute_temporal_smoothness_loss(frame_probs, mask)
    
    # Ranking loss (from inconsistency module)
    ranking_loss = torch.tensor(0.0, device=frame_logits.device)
    if ranking_loss_weight > 0 and inconsistency_pos is not None and inconsistency_neg is not None:
        ranking_loss = compute_ranking_loss(
            inconsistency_pos, inconsistency_neg,
            reliability_gate, mask, ranking_margin
        )
    
    # Fake hinge loss (optional, light constraint on fake frames)
    fake_hinge_loss = torch.tensor(0.0, device=frame_logits.device)
    if fake_hinge_weight > 0 and inconsistency_gated is not None:
        fake_hinge_loss = compute_fake_hinge_loss(
            inconsistency_gated, frame_labels, mask
        )
    
    # Total loss
    total_loss = (
        frame_loss_weight * frame_loss +
        video_loss_weight * video_loss +
        boundary_loss_weight * boundary_loss +
        smooth_loss_weight * smooth_loss +
        ranking_loss_weight * ranking_loss +
        fake_hinge_weight * fake_hinge_loss
    )
    
    return {
        'total': total_loss,
        'frame': frame_loss,
        'video': video_loss,
        'boundary': boundary_loss,
        'smooth': smooth_loss,
        'ranking': ranking_loss,
        'fake_hinge': fake_hinge_loss
    }


def generate_hard_negatives(
    audio: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    shift_range: Tuple[int, int] = (3, 10),
    swap_prob: float = 0.5
) -> torch.Tensor:
    """
    Generate hard negative samples for ranking loss.
    
    Args:
        audio: [B, T, a_dim] audio features
        mask: [B, T] padding mask
        shift_range: (min_shift, max_shift) for temporal misalignment
        swap_prob: Probability of doing swap vs shift
    
    Returns:
        audio_neg: [B, T, a_dim] misaligned audio features
    """
    B, T, a_dim = audio.shape
    device = audio.device
    
    if torch.rand(1).item() < swap_prob:
        # Cross-video swap (shuffle batch)
        indices = torch.randperm(B, device=device)
        audio_neg = audio[indices]
    else:
        # Temporal shift
        shifts = torch.randint(shift_range[0], shift_range[1] + 1, (B,), device=device)
        audio_neg = torch.zeros_like(audio)
        
        for i in range(B):
            shift = shifts[i].item()
            # Circular shift
            audio_neg[i] = torch.roll(audio[i], shifts=shift, dims=0)
            
            # Adjust mask if provided
            if mask is not None:
                # Don't shift padding
                valid_length = (~mask[i]).sum().item()
                if shift < valid_length:
                    # Only shift valid part
                    audio_neg[i, :valid_length] = torch.roll(audio[i, :valid_length], shifts=shift, dims=0)
    
    return audio_neg


if __name__ == '__main__':
    # Test model with boundary heads
    print("[TEST] Creating enhanced model with boundary heads...")
    model = FrameLocalizationModel(
        v_dim=512,
        a_dim=1024,
        d_model=512,
        nhead=8,
        num_layers=4,
        use_cross_attn=True,
        use_video_head=True,
        use_inconsistency_module=True,
        use_reliability_gating=True,
        use_boundary_head=True
    )
    
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # Test forward pass
    B, T = 4, 256
    visual = torch.randn(B, T, 512)
    audio = torch.randn(B, T, 1024)
    mask = torch.zeros(B, T, dtype=torch.bool)
    mask[:, 200:] = True  # Mask last 56 frames
    
    print(f"\n[TEST] Forward pass...")
    outputs = model(visual, audio, mask)
    print(f"Frame logits: {outputs['frame_logits'].shape}")
    print(f"Video logit: {outputs['video_logit'].shape}")
    print(f"Start logits: {outputs['start_logits'].shape}")
    print(f"End logits: {outputs['end_logits'].shape}")
    print(f"Inconsistency score: {outputs['inconsistency_score'].shape}")
    print(f"Reliability gate: {outputs['reliability_gate'].shape}")
    print(f"Inconsistency range: [{outputs['inconsistency_score'].min():.3f}, {outputs['inconsistency_score'].max():.3f}]")
    print(f"Gate range: [{outputs['reliability_gate'].min():.3f}, {outputs['reliability_gate'].max():.3f}]")
    print(f"Alpha (fusion weight): {model.alpha.item():.3f}")
    
    # Test boundary label generation
    print(f"\n[TEST] Generating boundary labels...")
    frame_labels = torch.randint(0, 2, (B, T))
    frame_labels[:, 50:100] = 1  # Create a fake segment
    frame_labels[:, 150:180] = 1  # Another fake segment
    start_labels, end_labels = generate_boundary_labels(frame_labels, mask, tolerance=5)
    print(f"Start labels shape: {start_labels.shape}, positives: {start_labels.sum().item()}")
    print(f"End labels shape: {end_labels.shape}, positives: {end_labels.sum().item()}")
    
    # Test hard negative generation
    print(f"\n[TEST] Generating hard negatives...")
    audio_neg = generate_hard_negatives(audio, mask)
    print(f"Audio negative: {audio_neg.shape}")
    
    # Forward with negatives
    outputs_neg = model(visual, audio_neg, mask)
    inconsistency_pos = outputs['inconsistency_score']
    inconsistency_neg = outputs_neg['inconsistency_score']
    print(f"Mean inconsistency (pos): {inconsistency_pos.mean():.3f}")
    print(f"Mean inconsistency (neg): {inconsistency_neg.mean():.3f}")
    
    # Test loss computation with boundary loss
    video_labels = torch.randint(0, 2, (B,))
    
    losses = compute_combined_loss(
        frame_logits=outputs['frame_logits'],
        frame_labels=frame_labels,
        mask=mask,
        video_logit=outputs['video_logit'],
        video_label=video_labels,
        start_logits=outputs['start_logits'],
        end_logits=outputs['end_logits'],
        inconsistency_pos=inconsistency_pos,
        inconsistency_neg=inconsistency_neg,
        inconsistency_gated=outputs['inconsistency_gated'],
        reliability_gate=outputs['reliability_gate'],
        frame_loss_weight=1.0,
        video_loss_weight=0.3,
        boundary_loss_weight=0.5,
        smooth_loss_weight=0.05,
        ranking_loss_weight=0.5,
        fake_hinge_weight=0.05,
        boundary_tolerance=5,
        use_boundary_aware_smooth=True,
        pos_weight=5.0
    )
    
    print(f"\n[TEST] Losses:")
    print(f"Total loss: {losses['total'].item():.4f}")
    print(f"Frame loss: {losses['frame'].item():.4f}")
    print(f"Video loss: {losses['video'].item():.4f}")
    print(f"Boundary loss: {losses['boundary'].item():.4f}")
    print(f"Smooth loss: {losses['smooth'].item():.4f}")
    print(f"Ranking loss: {losses['ranking'].item():.4f}")
    print(f"Fake hinge loss: {losses['fake_hinge'].item():.4f}")
    
    print(f"\n[TEST] Enhanced model with boundary heads test passed!")

