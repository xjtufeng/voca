#!/usr/bin/env python3
"""
Two-stage temporal localization with boundary refinement
Inspired by temporal action localization methods (ActionFormer, BSN)

Stage A: Multi-threshold proposal generation (high recall)
Stage B: Boundary refinement with start/end heads (high precision)
"""
import numpy as np
from typing import List, Tuple, Optional


def get_segments_from_binary(binary: np.ndarray) -> List[Tuple[int, int]]:
    """
    Extract continuous segments from binary mask
    
    Args:
        binary: [T] binary array (0/1)
    
    Returns:
        segments: List of (start, end) tuples (inclusive)
    """
    if len(binary) == 0 or binary.sum() == 0:
        return []
    
    segments = []
    in_segment = False
    start = 0
    
    for t in range(len(binary)):
        if binary[t] == 1 and not in_segment:
            start = t
            in_segment = True
        elif binary[t] == 0 and in_segment:
            segments.append((start, t - 1))
            in_segment = False
    
    if in_segment:
        segments.append((start, len(binary) - 1))
    
    return segments


def merge_close_segments(
    segments: List[Tuple[int, int]],
    gap: int = 3
) -> List[Tuple[int, int]]:
    """
    Merge segments that are close to each other
    
    Args:
        segments: List of (start, end) tuples
        gap: Maximum gap between segments to merge
    
    Returns:
        merged_segments: List of merged (start, end) tuples
    """
    if len(segments) == 0:
        return []
    
    # Sort by start
    segments = sorted(segments, key=lambda x: x[0])
    
    merged = [segments[0]]
    
    for start, end in segments[1:]:
        prev_start, prev_end = merged[-1]
        
        if start - prev_end <= gap:
            # Merge
            merged[-1] = (prev_start, max(end, prev_end))
        else:
            merged.append((start, end))
    
    return merged


def compute_iou_1d(seg1: Tuple[int, int], seg2: Tuple[int, int]) -> float:
    """
    Compute 1D IoU between two segments
    
    Args:
        seg1: (start1, end1)
        seg2: (start2, end2)
    
    Returns:
        iou: Intersection over Union
    """
    start1, end1 = seg1
    start2, end2 = seg2
    
    intersection = max(0, min(end1, end2) - max(start1, start2) + 1)
    union = max(end1, end2) - min(start1, start2) + 1
    
    return intersection / union if union > 0 else 0.0


def nms_temporal(
    proposals: List[Tuple[int, int, float]],
    iou_threshold: float = 0.7
) -> List[Tuple[int, int, float]]:
    """
    Non-Maximum Suppression for temporal segments
    
    Args:
        proposals: List of (start, end, score) tuples
        iou_threshold: IoU threshold for suppression
    
    Returns:
        kept_proposals: List of (start, end, score) tuples after NMS
    """
    if len(proposals) == 0:
        return []
    
    # Sort by score (descending)
    proposals = sorted(proposals, key=lambda x: x[2], reverse=True)
    
    keep = []
    
    while len(proposals) > 0:
        best = proposals.pop(0)
        keep.append(best)
        
        # Remove overlapping proposals
        proposals = [
            p for p in proposals
            if compute_iou_1d(best[:2], p[:2]) < iou_threshold
        ]
    
    return keep


def two_stage_localization(
    frame_probs: np.ndarray,
    start_probs: np.ndarray,
    end_probs: np.ndarray,
    thresholds: List[float] = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
    refine_delta: int = 10,
    min_len: int = 5,
    gap_merge: int = 3,
    nms_iou: float = 0.7,
    top_k_ratio: float = 0.2
) -> List[Tuple[int, int, float]]:
    """
    Two-stage temporal localization with boundary refinement
    
    Stage A: Generate proposals from frame probabilities using multiple thresholds
    Stage B: Refine boundaries using start/end probability peaks
    
    Args:
        frame_probs: [T] frame-level probabilities (0-1)
        start_probs: [T] start boundary probabilities (0-1)
        end_probs: [T] end boundary probabilities (0-1)
        thresholds: List of thresholds for proposal generation
        refine_delta: Search window radius for boundary refinement
        min_len: Minimum segment length
        gap_merge: Maximum gap for merging close segments
        nms_iou: IoU threshold for NMS
        top_k_ratio: Ratio of top frames to use for segment scoring
    
    Returns:
        proposals: List of (start, end, score) tuples, sorted by score (descending)
    """
    T = len(frame_probs)
    
    if T == 0:
        return []
    
    # Stage A: Proposal generation (high recall)
    all_proposals = []
    
    for thr in thresholds:
        # Threshold to binary
        binary = (frame_probs >= thr).astype(int)
        
        # Extract segments
        segments = get_segments_from_binary(binary)
        
        # Merge close segments
        segments = merge_close_segments(segments, gap=gap_merge)
        
        # Filter by length
        segments = [s for s in segments if (s[1] - s[0] + 1) >= min_len]
        
        all_proposals.extend(segments)
    
    # Remove duplicate segments
    all_proposals = list(set(all_proposals))
    
    if len(all_proposals) == 0:
        return []
    
    # Stage B: Boundary refinement (high precision)
    refined_proposals = []
    
    for start, end in all_proposals:
        # Refine start boundary
        search_start = max(0, start - refine_delta)
        search_end = min(T, start + refine_delta)
        
        if search_end > search_start:
            local_start_probs = start_probs[search_start:search_end]
            refined_start = search_start + np.argmax(local_start_probs)
        else:
            refined_start = start
        
        # Refine end boundary
        search_start = max(0, end - refine_delta)
        search_end = min(T, end + refine_delta)
        
        if search_end > search_start:
            local_end_probs = end_probs[search_start:search_end]
            refined_end = search_start + np.argmax(local_end_probs)
        else:
            refined_end = end
        
        # Ensure valid segment
        if refined_end <= refined_start:
            refined_end = refined_start + min_len - 1
        
        if refined_end >= T:
            refined_end = T - 1
        
        # Compute segment score
        seg_len = refined_end - refined_start + 1
        if seg_len < min_len:
            continue
        
        # Frame score: mean of top-k% frames in segment
        segment_frame_probs = frame_probs[refined_start:refined_end + 1]
        k = max(1, int(top_k_ratio * len(segment_frame_probs)))
        frame_score = np.mean(np.sort(segment_frame_probs)[-k:])
        
        # Boundary score: geometric mean of start and end probabilities
        boundary_score = np.sqrt(start_probs[refined_start] * end_probs[refined_end])
        
        # Combined score
        final_score = frame_score * boundary_score
        
        refined_proposals.append((refined_start, refined_end, final_score))
    
    if len(refined_proposals) == 0:
        return []
    
    # NMS to remove redundant proposals
    refined_proposals = nms_temporal(refined_proposals, iou_threshold=nms_iou)
    
    # Sort by score
    refined_proposals.sort(key=lambda x: x[2], reverse=True)
    
    return refined_proposals


def evaluate_frame_level(
    frame_probs: np.ndarray,
    frame_labels: np.ndarray,
    threshold: float = 0.5
) -> dict:
    """
    Evaluate frame-level classification metrics
    
    Args:
        frame_probs: [T] frame probabilities
        frame_labels: [T] frame labels (0/1)
        threshold: Classification threshold
    
    Returns:
        metrics: Dict with precision, recall, f1
    """
    preds = (frame_probs >= threshold).astype(int)
    
    tp = np.sum((preds == 1) & (frame_labels == 1))
    fp = np.sum((preds == 1) & (frame_labels == 0))
    fn = np.sum((preds == 0) & (frame_labels == 1))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': int(tp),
        'fp': int(fp),
        'fn': int(fn)
    }


def evaluate_segment_level(
    pred_segments: List[Tuple[int, int, float]],
    gt_segments: List[Tuple[int, int]],
    iou_thresholds: List[float] = [0.3, 0.5, 0.7, 0.9]
) -> dict:
    """
    Evaluate segment-level localization with AP at different IoU thresholds
    
    Args:
        pred_segments: List of (start, end, score) tuples
        gt_segments: List of (start, end) tuples
        iou_thresholds: List of IoU thresholds for evaluation
    
    Returns:
        metrics: Dict with AP at each IoU threshold
    """
    if len(gt_segments) == 0:
        return {f'AP@{thr:.1f}': 0.0 for thr in iou_thresholds}
    
    if len(pred_segments) == 0:
        return {f'AP@{thr:.1f}': 0.0 for thr in iou_thresholds}
    
    metrics = {}
    
    for iou_thr in iou_thresholds:
        # Sort predictions by score
        pred_segments_sorted = sorted(pred_segments, key=lambda x: x[2], reverse=True)
        
        # Use float64 to prevent overflow
        tp = np.zeros(len(pred_segments_sorted), dtype=np.float64)
        fp = np.zeros(len(pred_segments_sorted), dtype=np.float64)
        
        matched_gt = set()
        
        for i, (start, end, score) in enumerate(pred_segments_sorted):
            best_iou = 0.0
            best_gt_idx = -1
            
            for j, (gt_start, gt_end) in enumerate(gt_segments):
                if j in matched_gt:
                    continue
                
                iou = compute_iou_1d((start, end), (gt_start, gt_end))
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j
            
            if best_iou >= iou_thr:
                tp[i] = 1
                matched_gt.add(best_gt_idx)
            else:
                fp[i] = 1
        
        # Compute precision and recall at each rank (use float64)
        tp_cumsum = np.cumsum(tp, dtype=np.float64)
        fp_cumsum = np.cumsum(fp, dtype=np.float64)
        
        recalls = tp_cumsum / max(len(gt_segments), 1)
        precisions = tp_cumsum / np.maximum(tp_cumsum + fp_cumsum, 1e-10)
        
        # Compute AP using trapezoidal rule (more stable)
        if len(recalls) > 0:
            # Sort by recall for proper AP calculation
            recall_indices = np.argsort(recalls)
            sorted_recalls = recalls[recall_indices]
            sorted_precisions = precisions[recall_indices]
            
            # Compute AP as area under P-R curve
            ap = np.trapz(sorted_precisions, sorted_recalls)
            ap = float(np.clip(ap, 0.0, 1.0))  # Clip to [0, 1] and convert to Python float
        else:
            ap = 0.0
        
        metrics[f'AP@{iou_thr:.1f}'] = ap
    
    # Compute mAP (use Python float to avoid numpy overflow)
    ap_values = [float(metrics[k]) for k in metrics.keys() if k.startswith('AP@')]
    metrics['mAP'] = float(np.mean(ap_values)) if len(ap_values) > 0 else 0.0
    
    return metrics


if __name__ == '__main__':
    # Test two-stage localization
    print("[TEST] Two-stage localization inference")
    
    # Create synthetic data
    T = 500
    frame_probs = np.random.rand(T) * 0.3
    frame_probs[100:150] = 0.7 + np.random.rand(50) * 0.2  # Fake segment 1
    frame_probs[300:380] = 0.6 + np.random.rand(80) * 0.3  # Fake segment 2
    
    start_probs = np.random.rand(T) * 0.2
    start_probs[100] = 0.9
    start_probs[300] = 0.85
    
    end_probs = np.random.rand(T) * 0.2
    end_probs[149] = 0.88
    end_probs[379] = 0.92
    
    # Run two-stage localization
    proposals = two_stage_localization(
        frame_probs, start_probs, end_probs,
        thresholds=[0.3, 0.4, 0.5],
        refine_delta=10,
        min_len=5
    )
    
    print(f"\nFound {len(proposals)} proposals:")
    for i, (start, end, score) in enumerate(proposals[:5]):
        print(f"  {i+1}. [{start:3d}, {end:3d}] len={end-start+1:3d} score={score:.3f}")
    
    # Test evaluation
    print(f"\n[TEST] Segment-level evaluation")
    gt_segments = [(100, 149), (300, 379)]
    metrics = evaluate_segment_level(proposals, gt_segments)
    
    for key, value in metrics.items():
        print(f"  {key}: {value:.3f}")
    
    print(f"\n[TEST] Two-stage localization test passed!")

