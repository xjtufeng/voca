#!/usr/bin/env python3
"""
Temporal Localization Evaluation for Frame-level Deepfake Detection

Converts frame-level predictions to segment-level proposals and computes AP@IoU metrics.
"""
import numpy as np
from typing import List, Tuple, Dict
from sklearn.metrics import average_precision_score


def extract_segments_from_binary(binary_mask: np.ndarray) -> List[Tuple[int, int]]:
    """
    Extract continuous segments of 1s from a binary mask.
    
    Args:
        binary_mask: Binary array of shape [T], values {0, 1}
    
    Returns:
        List of (start, end) tuples (both inclusive)
    
    Example:
        [0,0,1,1,1,0,1,0] -> [(2, 4), (6, 6)]
    """
    if len(binary_mask) == 0:
        return []
    
    segments = []
    in_segment = False
    start = 0
    
    for i, val in enumerate(binary_mask):
        if val == 1 and not in_segment:
            # Start new segment
            start = i
            in_segment = True
        elif val == 0 and in_segment:
            # End current segment
            segments.append((start, i - 1))
            in_segment = False
    
    # Handle segment extending to end
    if in_segment:
        segments.append((start, len(binary_mask) - 1))
    
    return segments


def merge_close_segments(
    segments: List[Tuple[int, int]], 
    max_gap: int = 2
) -> List[Tuple[int, int]]:
    """
    Merge segments that are separated by ≤ max_gap frames.
    
    Args:
        segments: List of (start, end) tuples
        max_gap: Maximum gap to merge (default: 2 frames)
    
    Returns:
        Merged segments
    """
    if len(segments) <= 1:
        return segments
    
    # Sort by start time
    segments = sorted(segments, key=lambda x: x[0])
    
    merged = [segments[0]]
    
    for start, end in segments[1:]:
        prev_start, prev_end = merged[-1]
        
        # Check if gap is small enough to merge
        gap = start - prev_end - 1
        if gap <= max_gap:
            # Merge with previous
            merged[-1] = (prev_start, end)
        else:
            # Add as new segment
            merged.append((start, end))
    
    return merged


def extract_segments_from_probs(
    frame_probs: np.ndarray,
    threshold: float = 0.3,
    min_length: int = 3,
    video_length: int = None,
    merge_gap: int = 2,
    use_max_score: bool = True
) -> List[Tuple[int, int, float]]:
    """
    Extract segment proposals from frame-level probabilities.
    
    Args:
        frame_probs: Frame probabilities, shape [T], values [0, 1]
        threshold: Binarization threshold (default: 0.3)
        min_length: Minimum segment length in frames (default: 3)
        video_length: Total video length, used for adaptive min_length (default: None)
        merge_gap: Merge segments separated by ≤ this many frames (default: 2)
        use_max_score: Use max prob as segment score (else use mean)
    
    Returns:
        List of (start, end, score) tuples
        - start, end: frame indices (both inclusive)
        - score: segment confidence
    """
    if len(frame_probs) == 0:
        return []
    
    # Adaptive min_length
    if video_length is not None:
        adaptive_min = max(3, int(0.005 * video_length))
        min_length = max(min_length, adaptive_min)
    
    # Binarize
    binary_mask = (frame_probs >= threshold).astype(int)
    
    # Extract segments
    segments = extract_segments_from_binary(binary_mask)
    
    # Merge close segments
    if merge_gap > 0:
        segments = merge_close_segments(segments, max_gap=merge_gap)
    
    # Add scores and filter by length
    scored_segments = []
    for start, end in segments:
        length = end - start + 1
        if length < min_length:
            continue
        
        # Compute segment score
        segment_probs = frame_probs[start:end+1]
        if use_max_score:
            score = float(np.max(segment_probs))
        else:
            score = float(np.mean(segment_probs))
        
        scored_segments.append((start, end, score))
    
    return scored_segments


def temporal_iou(seg1: Tuple[int, int], seg2: Tuple[int, int]) -> float:
    """
    Compute temporal IoU between two segments.
    
    Args:
        seg1: (start, end) tuple
        seg2: (start, end) tuple
    
    Returns:
        IoU value [0, 1]
    """
    start1, end1 = seg1
    start2, end2 = seg2
    
    # Compute intersection
    inter_start = max(start1, start2)
    inter_end = min(end1, end2)
    inter = max(0, inter_end - inter_start + 1)
    
    # Compute union
    union = (end1 - start1 + 1) + (end2 - start2 + 1) - inter
    
    if union == 0:
        return 0.0
    
    return inter / union


def nms_temporal(
    segments: List[Tuple[int, int, float]], 
    iou_threshold: float = 0.7
) -> List[Tuple[int, int, float]]:
    """
    Non-Maximum Suppression for temporal segments.
    
    Args:
        segments: List of (start, end, score) tuples
        iou_threshold: IoU threshold for suppression (default: 0.7)
    
    Returns:
        Filtered segments after NMS
    """
    if len(segments) == 0:
        return []
    
    # Sort by score (descending)
    segments = sorted(segments, key=lambda x: x[2], reverse=True)
    
    keep = []
    suppressed = set()
    
    for i, seg in enumerate(segments):
        if i in suppressed:
            continue
        
        keep.append(seg)
        
        # Suppress overlapping segments
        for j in range(i + 1, len(segments)):
            if j in suppressed:
                continue
            
            iou = temporal_iou(seg[:2], segments[j][:2])
            if iou >= iou_threshold:
                suppressed.add(j)
    
    return keep


def compute_ap_at_iou(
    predictions: List[Tuple[str, int, int, float]],
    ground_truths: List[Tuple[str, int, int]],
    iou_threshold: float = 0.5
) -> float:
    """
    Compute Average Precision at given IoU threshold.
    
    Args:
        predictions: List of (video_id, start, end, score)
        ground_truths: List of (video_id, start, end)
        iou_threshold: IoU threshold for matching (default: 0.5)
    
    Returns:
        Average Precision value [0, 1]
    """
    if len(ground_truths) == 0:
        return 0.0
    
    if len(predictions) == 0:
        return 0.0
    
    # Sort predictions by score (descending)
    predictions = sorted(predictions, key=lambda x: x[3], reverse=True)
    
    # Group ground truths by video_id
    gt_by_video = {}
    for video_id, start, end in ground_truths:
        if video_id not in gt_by_video:
            gt_by_video[video_id] = []
        gt_by_video[video_id].append((start, end))
    
    # Track matched ground truths
    matched_gts = {video_id: [False] * len(gts) 
                   for video_id, gts in gt_by_video.items()}
    
    # Compute TP and FP for each prediction
    tp = np.zeros(len(predictions))
    fp = np.zeros(len(predictions))
    
    for i, (video_id, pred_start, pred_end, score) in enumerate(predictions):
        if video_id not in gt_by_video:
            # No GT for this video
            fp[i] = 1
            continue
        
        # Find best matching GT
        best_iou = 0.0
        best_gt_idx = -1
        
        for gt_idx, (gt_start, gt_end) in enumerate(gt_by_video[video_id]):
            if matched_gts[video_id][gt_idx]:
                # GT already matched
                continue
            
            iou = temporal_iou((pred_start, pred_end), (gt_start, gt_end))
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        
        # Check if match is good enough
        if best_iou >= iou_threshold and best_gt_idx >= 0:
            tp[i] = 1
            matched_gts[video_id][best_gt_idx] = True
        else:
            fp[i] = 1
    
    # Compute cumulative TP and FP
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    
    # Compute precision and recall
    recalls = tp_cumsum / len(ground_truths)
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-10)
    
    # Compute AP using 11-point interpolation (standard for temporal localization)
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        if np.sum(recalls >= t) == 0:
            p = 0
        else:
            p = np.max(precisions[recalls >= t])
        ap += p / 11
    
    return ap


def evaluate_temporal_localization(
    frame_probs_list: List[np.ndarray],
    frame_labels_list: List[np.ndarray],
    video_ids: List[str] = None,
    thresholds: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5],
    iou_thresholds: List[float] = [0.5, 0.75, 0.95],
    nms_iou: float = 0.7,
    min_length: int = 3,
    merge_gap: int = 2,
    use_best_threshold: bool = True
) -> Dict[str, float]:
    """
    Evaluate temporal localization performance.
    
    Args:
        frame_probs_list: List of frame probabilities for each video
        frame_labels_list: List of frame labels for each video
        video_ids: List of video IDs (default: use indices)
        thresholds: List of thresholds to try (default: [0.1, 0.2, 0.3, 0.4, 0.5])
        iou_thresholds: List of IoU thresholds for AP (default: [0.5, 0.75, 0.95])
        nms_iou: IoU threshold for NMS (default: 0.7)
        min_length: Minimum segment length (default: 3)
        merge_gap: Gap for merging segments (default: 2)
        use_best_threshold: Sweep thresholds and use best (default: True)
    
    Returns:
        Dictionary of metrics: {
            'AP@0.5': float,
            'AP@0.75': float,
            'AP@0.95': float,
            'mAP': float,  # mean of AP@0.5, AP@0.75, AP@0.95
            'best_threshold': float
        }
    """
    if video_ids is None:
        video_ids = [str(i) for i in range(len(frame_probs_list))]
    
    assert len(frame_probs_list) == len(frame_labels_list) == len(video_ids)
    
    # Extract ground truth segments
    all_gt_segments = []
    for video_id, labels in zip(video_ids, frame_labels_list):
        segments = extract_segments_from_binary(labels)
        for start, end in segments:
            all_gt_segments.append((video_id, start, end))
    
    if len(all_gt_segments) == 0:
        print("[WARN] No ground truth segments found!")
        return {f'AP@{t}': 0.0 for t in iou_thresholds}
    
    # Try different thresholds if needed
    best_threshold = thresholds[0]
    best_map = 0.0
    best_metrics = {}
    
    for threshold in thresholds:
        # Extract prediction segments
        all_pred_segments = []
        for video_id, probs in zip(video_ids, frame_probs_list):
            segments = extract_segments_from_probs(
                probs,
                threshold=threshold,
                min_length=min_length,
                video_length=len(probs),
                merge_gap=merge_gap,
                use_max_score=True
            )
            
            # Apply NMS within video
            segments = nms_temporal(segments, iou_threshold=nms_iou)
            
            for start, end, score in segments:
                all_pred_segments.append((video_id, start, end, score))
        
        # Compute AP for each IoU threshold
        metrics = {}
        for iou_thr in iou_thresholds:
            ap = compute_ap_at_iou(all_pred_segments, all_gt_segments, iou_thr)
            metrics[f'AP@{iou_thr}'] = ap
        
        # Compute mAP
        metrics['mAP'] = np.mean([metrics[f'AP@{t}'] for t in iou_thresholds])
        
        # Track best threshold
        if metrics['mAP'] > best_map:
            best_map = metrics['mAP']
            best_threshold = threshold
            best_metrics = metrics
    
    if use_best_threshold:
        best_metrics['best_threshold'] = best_threshold
        return best_metrics
    else:
        # Just use first threshold
        return best_metrics


def evaluate_with_multiple_videos(
    all_frame_probs: List[np.ndarray],
    all_frame_labels: List[np.ndarray],
    video_ids: List[str] = None
) -> Dict[str, float]:
    """
    Convenient wrapper for evaluating multiple videos.
    
    Args:
        all_frame_probs: List of [T] arrays, one per video
        all_frame_labels: List of [T] arrays, one per video
        video_ids: List of video IDs (optional)
    
    Returns:
        Dictionary of metrics
    """
    return evaluate_temporal_localization(
        all_frame_probs,
        all_frame_labels,
        video_ids=video_ids,
        thresholds=[0.1, 0.2, 0.3, 0.4, 0.5],
        iou_thresholds=[0.5, 0.75, 0.95],
        nms_iou=0.7,
        min_length=3,
        merge_gap=2,
        use_best_threshold=True
    )

