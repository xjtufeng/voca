#!/usr/bin/env python3
"""
Visualization tools for frame-level deepfake localization
Generate time-series curves showing detection results
"""
import argparse
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec


def extract_fake_segments(
    frame_probs: np.ndarray,
    threshold: float = 0.5,
    min_duration: int = 5
) -> List[Tuple[int, int, float]]:
    """
    Extract continuous fake segments from frame probabilities
    
    Args:
        frame_probs: [T] frame probabilities
        threshold: Detection threshold
        min_duration: Minimum segment length (frames)
    
    Returns:
        List of (start_frame, end_frame, avg_confidence) tuples
    """
    binary = (frame_probs > threshold).astype(int)
    segments = []
    start = None
    
    for i in range(len(binary)):
        if binary[i] == 1 and start is None:
            start = i
        elif binary[i] == 0 and start is not None:
            if i - start >= min_duration:
                segments.append((start, i - 1, frame_probs[start:i].mean()))
            start = None
    
    if start is not None and len(binary) - start >= min_duration:
        segments.append((start, len(binary) - 1, frame_probs[start:].mean()))
    
    return segments


def visualize_localization(
    frame_probs: np.ndarray,
    frame_labels: np.ndarray,
    audio_visual_sim: Optional[np.ndarray] = None,
    fps: float = 25.0,
    threshold: float = 0.5,
    save_path: Optional[str] = None,
    title: str = "Deepfake Localization Results",
    figsize: Tuple[int, int] = (14, 10)
):
    """
    Visualize frame-level localization results with multiple time-series curves
    
    Args:
        frame_probs: [T] predicted fake probabilities
        frame_labels: [T] ground truth labels (0=real, 1=fake)
        audio_visual_sim: [T] optional audio-visual similarity scores
        fps: Video frame rate
        threshold: Detection threshold
        save_path: Path to save figure
        title: Figure title
        figsize: Figure size
    """
    T = len(frame_probs)
    time_axis = np.arange(T) / fps
    
    # Create figure with subplots
    num_plots = 3 if audio_visual_sim is not None else 2
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(num_plots, 1, figure=fig, hspace=0.3)
    
    # ===== Plot 1: Frame-Level Fake Probability =====
    ax1 = fig.add_subplot(gs[0])
    
    # Plot probability curve
    ax1.plot(time_axis, frame_probs, linewidth=2, label='Fake Probability', 
             color='#e74c3c', alpha=0.9)
    
    # Threshold line
    ax1.axhline(threshold, color='orange', linestyle='--', linewidth=1.5, 
                label=f'Threshold ({threshold})', alpha=0.7)
    
    # Fill detected fake regions
    ax1.fill_between(time_axis, 0, 1, where=(frame_probs > threshold),
                     alpha=0.25, color='red', label='Detected Fake')
    
    # Extract and annotate segments
    segments = extract_fake_segments(frame_probs, threshold, min_duration=10)
    for start, end, conf in segments:
        t_start = start / fps
        t_end = end / fps
        duration = t_end - t_start
        mid_time = (t_start + t_end) / 2
        
        # Highlight segment
        ax1.axvspan(t_start, t_end, alpha=0.15, color='darkred', zorder=0)
        
        # Annotate
        if duration > 0.5:  # Only annotate if segment is long enough
            ax1.annotate(f'{duration:.1f}s\n{conf:.2f}',
                        xy=(mid_time, 0.92), ha='center', va='top',
                        fontsize=8, color='darkred', weight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', 
                                 facecolor='yellow', alpha=0.6))
    
    ax1.set_ylabel('Fake Probability', fontsize=11, weight='bold')
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(alpha=0.3, linestyle=':', linewidth=0.5)
    ax1.set_title('Frame-Level Deepfake Detection', fontsize=13, weight='bold', pad=10)
    
    # ===== Plot 2: Ground Truth vs Prediction =====
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    
    # Ground truth (filled area)
    ax2.fill_between(time_axis, 0, frame_labels, alpha=0.5, color='darkred',
                     step='post', label='Ground Truth Fake', edgecolor='black', linewidth=0.5)
    
    # Prediction (binary)
    pred_binary = (frame_probs > threshold).astype(int)
    ax2.plot(time_axis, pred_binary * 0.95, linewidth=2.5, color='#3498db',
             alpha=0.8, drawstyle='steps-post', label='Prediction')
    
    # Mark false positives and false negatives
    fp_mask = (pred_binary == 1) & (frame_labels == 0)
    fn_mask = (pred_binary == 0) & (frame_labels == 1)
    
    if fp_mask.any():
        ax2.scatter(time_axis[fp_mask], np.ones(fp_mask.sum()) * 0.5,
                   color='orange', marker='x', s=30, alpha=0.6, 
                   label='False Positive', zorder=5)
    
    if fn_mask.any():
        ax2.scatter(time_axis[fn_mask], np.ones(fn_mask.sum()) * 0.5,
                   color='purple', marker='+', s=30, alpha=0.6,
                   label='False Negative', zorder=5)
    
    ax2.set_ylabel('Label (0=Real, 1=Fake)', fontsize=11, weight='bold')
    ax2.set_ylim(-0.1, 1.2)
    ax2.set_yticks([0, 1])
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(alpha=0.3, linestyle=':', linewidth=0.5)
    ax2.set_title('Ground Truth vs Prediction', fontsize=13, weight='bold', pad=10)
    
    # ===== Plot 3: Audio-Visual Similarity (if available) =====
    if audio_visual_sim is not None:
        ax3 = fig.add_subplot(gs[2], sharex=ax1)
        
        # Plot similarity curve
        ax3.plot(time_axis, audio_visual_sim, linewidth=2, color='#2ecc71',
                label='Audio-Visual Similarity', alpha=0.9)
        
        # Anomaly threshold (low similarity = potential fake)
        sim_threshold = np.percentile(audio_visual_sim, 25)
        ax3.axhline(sim_threshold, color='darkorange', linestyle='--', 
                   linewidth=1.5, label=f'Anomaly Threshold ({sim_threshold:.2f})', 
                   alpha=0.7)
        
        # Highlight low similarity regions
        ax3.fill_between(time_axis, 0, 1, where=(audio_visual_sim < sim_threshold),
                        alpha=0.2, color='yellow', label='Low Similarity')
        
        # Overlay ground truth fake regions (semi-transparent red bars)
        for i in range(len(frame_labels)):
            if frame_labels[i] == 1:
                ax3.axvspan(time_axis[i], time_axis[i] + 1/fps, 
                           alpha=0.1, color='red', zorder=0)
        
        ax3.set_xlabel('Time (seconds)', fontsize=11, weight='bold')
        ax3.set_ylabel('Similarity Score', fontsize=11, weight='bold')
        ax3.set_ylim(0, 1.05)
        ax3.legend(loc='upper right', fontsize=9)
        ax3.grid(alpha=0.3, linestyle=':', linewidth=0.5)
        ax3.set_title('Audio-Visual Synchronization', fontsize=13, weight='bold', pad=10)
    else:
        ax2.set_xlabel('Time (seconds)', fontsize=11, weight='bold')
    
    # Overall title
    fig.suptitle(title, fontsize=15, weight='bold', y=0.995)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[INFO] Saved visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_batch(
    results_list: List[dict],
    save_dir: str,
    fps: float = 25.0,
    threshold: float = 0.5,
    max_videos: int = 10
):
    """
    Generate visualizations for a batch of videos
    
    Args:
        results_list: List of dicts with keys: 'video_id', 'frame_probs', 
                      'frame_labels', 'audio_visual_sim' (optional)
        save_dir: Directory to save visualizations
        fps: Video frame rate
        threshold: Detection threshold
        max_videos: Maximum number of videos to visualize
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    for i, result in enumerate(results_list[:max_videos]):
        video_id = result['video_id']
        frame_probs = result['frame_probs']
        frame_labels = result['frame_labels']
        audio_visual_sim = result.get('audio_visual_sim')
        
        save_path = save_dir / f"{video_id}.png"
        
        visualize_localization(
            frame_probs=frame_probs,
            frame_labels=frame_labels,
            audio_visual_sim=audio_visual_sim,
            fps=fps,
            threshold=threshold,
            save_path=str(save_path),
            title=f"Localization Results: {video_id}"
        )


def plot_summary_statistics(
    per_video_results: List[dict],
    save_path: str
):
    """
    Plot summary statistics across all videos
    
    Args:
        per_video_results: List of per-video result dicts
        save_path: Path to save figure
    """
    # Extract metrics
    ious = [r['iou'] for r in per_video_results]
    f1_scores = [r['frame_f1'] for r in per_video_results]
    fake_ratios = [r['num_fake_frames'] / max(r['num_frames'], 1) 
                   for r in per_video_results]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # IoU distribution
    axes[0, 0].hist(ious, bins=30, color='#3498db', alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(np.mean(ious), color='red', linestyle='--', 
                       linewidth=2, label=f'Mean: {np.mean(ious):.3f}')
    axes[0, 0].set_xlabel('IoU Score', fontsize=11)
    axes[0, 0].set_ylabel('Count', fontsize=11)
    axes[0, 0].set_title('IoU Distribution', fontsize=12, weight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # F1 distribution
    axes[0, 1].hist(f1_scores, bins=30, color='#2ecc71', alpha=0.7, edgecolor='black')
    axes[0, 1].axvline(np.mean(f1_scores), color='red', linestyle='--',
                       linewidth=2, label=f'Mean: {np.mean(f1_scores):.3f}')
    axes[0, 1].set_xlabel('Frame F1 Score', fontsize=11)
    axes[0, 1].set_ylabel('Count', fontsize=11)
    axes[0, 1].set_title('F1 Score Distribution', fontsize=12, weight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # IoU vs Fake Ratio
    axes[1, 0].scatter(fake_ratios, ious, alpha=0.5, color='#e74c3c', s=20)
    axes[1, 0].set_xlabel('Fake Frame Ratio', fontsize=11)
    axes[1, 0].set_ylabel('IoU Score', fontsize=11)
    axes[1, 0].set_title('IoU vs Fake Ratio', fontsize=12, weight='bold')
    axes[1, 0].grid(alpha=0.3)
    
    # F1 vs Fake Ratio
    axes[1, 1].scatter(fake_ratios, f1_scores, alpha=0.5, color='#9b59b6', s=20)
    axes[1, 1].set_xlabel('Fake Frame Ratio', fontsize=11)
    axes[1, 1].set_ylabel('F1 Score', fontsize=11)
    axes[1, 1].set_title('F1 vs Fake Ratio', fontsize=12, weight='bold')
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[INFO] Saved summary statistics to {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize localization results')
    
    # Input
    parser.add_argument('--input', type=str, required=True,
                        help='Path to evaluation results JSON')
    parser.add_argument('--output_dir', type=str, default='./visualizations',
                        help='Output directory for visualizations')
    
    # Visualization options
    parser.add_argument('--fps', type=float, default=25.0,
                        help='Video frame rate')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Detection threshold')
    parser.add_argument('--max_videos', type=int, default=20,
                        help='Maximum videos to visualize')
    
    args = parser.parse_args()
    
    # Load results
    import json
    print(f"[INFO] Loading results from {args.input}")
    with open(args.input, 'r') as f:
        results = json.load(f)
    
    per_video_results = results['per_video']
    print(f"[INFO] Loaded {len(per_video_results)} video results")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot summary statistics
    print(f"[INFO] Generating summary statistics...")
    plot_summary_statistics(
        per_video_results,
        save_path=str(output_dir / 'summary_statistics.png')
    )
    
    print(f"\n[INFO] Visualization complete!")
    print(f"[INFO] Results saved to {output_dir}")


if __name__ == '__main__':
    main()

