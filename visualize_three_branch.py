#!/usr/bin/env python3
"""
Visualization tools for three-branch model results
Compare predictions from Cross-Modal, Audio-Only, and Visual-Only branches
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec


def visualize_three_branch_comparison(
    video_id: str,
    predictions: Dict[str, np.ndarray],
    ground_truth: np.ndarray,
    fusion_weights: np.ndarray = None,
    fps: float = 25.0,
    threshold: float = 0.5,
    save_path: str = None
):
    """
    Visualize comparison of three-branch predictions
    
    Args:
        video_id: Video identifier
        predictions: Dict with keys 'fused', 'cross_modal', 'audio_only', 'visual_only'
        ground_truth: Ground truth labels [T] (for frame-level) or single value
        fusion_weights: Branch fusion weights [3] or [T, 3]
        fps: Video frame rate
        threshold: Detection threshold
        save_path: Path to save figure
    """
    # Handle video-level labels
    if isinstance(ground_truth, (int, float)):
        T = len(predictions['fused'])
        ground_truth = np.ones(T) * ground_truth
    
    T = len(ground_truth)
    time_axis = np.arange(T) / fps
    
    # Create figure
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(4, 1, figure=fig, hspace=0.3)
    
    # ===== Plot 1: Three-Branch Predictions Comparison =====
    ax1 = fig.add_subplot(gs[0])
    
    # Plot each branch
    ax1.plot(time_axis, predictions['cross_modal'], 
             label='Cross-Modal (A+V)', linewidth=2.5, 
             color='#e74c3c', alpha=0.9, zorder=3)
    
    ax1.plot(time_axis, predictions['audio_only'], 
             label='Audio-Only', linewidth=2, 
             color='#3498db', alpha=0.8, zorder=2)
    
    ax1.plot(time_axis, predictions['visual_only'], 
             label='Visual-Only', linewidth=2, 
             color='#2ecc71', alpha=0.8, zorder=2)
    
    if 'fused' in predictions:
        ax1.plot(time_axis, predictions['fused'], 
                 label='Fused (Final)', linewidth=3, 
                 color='#9b59b6', alpha=0.95, linestyle='--', zorder=4)
    
    # Threshold line
    ax1.axhline(threshold, color='orange', linestyle=':', 
                linewidth=1.5, alpha=0.6, label=f'Threshold ({threshold})')
    
    ax1.set_ylabel('Fake Probability', fontsize=12, weight='bold')
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax1.grid(alpha=0.3, linestyle=':', linewidth=0.5)
    ax1.set_title(f'Three-Branch Predictions: {video_id}', 
                  fontsize=14, weight='bold', pad=10)
    
    # ===== Plot 2: Ground Truth =====
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    
    ax2.fill_between(time_axis, 0, ground_truth, 
                     alpha=0.5, color='darkred', step='post', 
                     edgecolor='black', linewidth=0.5, label='Ground Truth')
    
    # Mark predictions
    if 'fused' in predictions:
        pred_binary = (predictions['fused'] > threshold).astype(int)
        ax2.plot(time_axis, pred_binary * 0.95, 
                linewidth=2.5, color='#9b59b6', alpha=0.8, 
                drawstyle='steps-post', label='Fused Prediction')
    
    ax2.set_ylabel('Label (0=Real, 1=Fake)', fontsize=12, weight='bold')
    ax2.set_ylim(-0.1, 1.2)
    ax2.set_yticks([0, 1])
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(alpha=0.3, linestyle=':', linewidth=0.5)
    ax2.set_title('Ground Truth vs Fused Prediction', 
                  fontsize=13, weight='bold', pad=10)
    
    # ===== Plot 3: Branch Agreement Analysis =====
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    
    # Compute pairwise differences
    diff_cm_ao = np.abs(predictions['cross_modal'] - predictions['audio_only'])
    diff_cm_vo = np.abs(predictions['cross_modal'] - predictions['visual_only'])
    diff_ao_vo = np.abs(predictions['audio_only'] - predictions['visual_only'])
    
    ax3.fill_between(time_axis, 0, diff_cm_ao, 
                     alpha=0.5, color='#e67e22', label='|CM - AO|')
    ax3.fill_between(time_axis, 0, diff_cm_vo, 
                     alpha=0.4, color='#1abc9c', label='|CM - VO|')
    ax3.fill_between(time_axis, 0, diff_ao_vo, 
                     alpha=0.3, color='#34495e', label='|AO - VO|')
    
    # Average disagreement
    avg_disagreement = (diff_cm_ao + diff_cm_vo + diff_ao_vo) / 3
    ax3.plot(time_axis, avg_disagreement, 
             color='red', linewidth=2, linestyle='--', 
             label='Avg Disagreement', alpha=0.8)
    
    ax3.set_ylabel('Prediction Difference', fontsize=12, weight='bold')
    ax3.set_ylim(-0.05, 1.05)
    ax3.legend(loc='upper right', fontsize=9, ncol=2)
    ax3.grid(alpha=0.3, linestyle=':', linewidth=0.5)
    ax3.set_title('Branch Agreement Analysis', 
                  fontsize=13, weight='bold', pad=10)
    
    # ===== Plot 4: Fusion Weights =====
    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    
    if fusion_weights is not None:
        if len(fusion_weights.shape) == 1:
            # Static weights [3]
            labels = ['Cross-Modal', 'Audio-Only', 'Visual-Only']
            colors = ['#e74c3c', '#3498db', '#2ecc71']
            bars = ax4.bar(labels, fusion_weights, color=colors, alpha=0.7, 
                          edgecolor='black', linewidth=1.5)
            
            # Add percentage labels
            for bar, weight in zip(bars, fusion_weights):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height,
                        f'{weight*100:.1f}%',
                        ha='center', va='bottom', fontsize=11, weight='bold')
            
            ax4.set_ylabel('Branch Weight', fontsize=12, weight='bold')
            ax4.set_ylim(0, max(fusion_weights) * 1.2)
            ax4.set_title('Static Branch Fusion Weights', 
                         fontsize=13, weight='bold', pad=10)
            ax4.grid(axis='y', alpha=0.3, linestyle=':', linewidth=0.5)
        
        else:
            # Dynamic weights [T, 3]
            ax4.fill_between(time_axis, 0, fusion_weights[:, 0], 
                            alpha=0.6, color='#e74c3c', label='CM Weight')
            ax4.fill_between(time_axis, 0, fusion_weights[:, 1], 
                            alpha=0.5, color='#3498db', label='AO Weight')
            ax4.fill_between(time_axis, 0, fusion_weights[:, 2], 
                            alpha=0.4, color='#2ecc71', label='VO Weight')
            
            ax4.set_ylabel('Dynamic Weight', fontsize=12, weight='bold')
            ax4.set_ylim(0, 1.05)
            ax4.legend(loc='upper right', fontsize=10)
            ax4.set_title('Dynamic Branch Fusion Weights Over Time', 
                         fontsize=13, weight='bold', pad=10)
            ax4.grid(alpha=0.3, linestyle=':', linewidth=0.5)
    else:
        ax4.text(0.5, 0.5, 'Fusion weights not available', 
                ha='center', va='center', fontsize=14, 
                color='gray', transform=ax4.transAxes)
        ax4.set_title('Branch Fusion Weights', 
                     fontsize=13, weight='bold', pad=10)
    
    ax4.set_xlabel('Time (seconds)', fontsize=12, weight='bold')
    
    # Overall title
    fig.suptitle('Three-Branch Model Analysis', 
                fontsize=16, weight='bold', y=0.995)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[INFO] Saved visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_branch_comparison_statistics(
    results: List[Dict],
    save_path: str = None
):
    """
    Plot statistical comparison of three branches across dataset
    
    Args:
        results: List of per-video results with predictions from each branch
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Extract metrics
    fused_correct = []
    cm_correct = []
    ao_correct = []
    vo_correct = []
    
    agreement_3 = []  # All 3 branches agree
    agreement_2 = []  # 2 out of 3 agree
    agreement_1 = []  # No consensus
    
    for result in results:
        label = result['label']
        threshold = result.get('threshold', 0.5)
        
        fused_pred = 1 if result['fused_prob'] > threshold else 0
        cm_pred = 1 if result['cm_prob'] > threshold else 0
        ao_pred = 1 if result['ao_prob'] > threshold else 0
        vo_pred = 1 if result['vo_prob'] > threshold else 0
        
        fused_correct.append(fused_pred == label)
        cm_correct.append(cm_pred == label)
        ao_correct.append(ao_pred == label)
        vo_correct.append(vo_pred == label)
        
        # Count agreements
        preds = [cm_pred, ao_pred, vo_pred]
        num_fake = sum(preds)
        
        if num_fake == 3 or num_fake == 0:
            agreement_3.append(fused_pred == label)
        elif num_fake == 2 or num_fake == 1:
            agreement_2.append(fused_pred == label)
    
    # Plot 1: Overall Accuracy
    accuracies = {
        'Fused': np.mean(fused_correct),
        'Cross-Modal': np.mean(cm_correct),
        'Audio-Only': np.mean(ao_correct),
        'Visual-Only': np.mean(vo_correct)
    }
    
    colors = ['#9b59b6', '#e74c3c', '#3498db', '#2ecc71']
    bars = axes[0, 0].bar(accuracies.keys(), accuracies.values(), 
                          color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    for bar in bars:
        height = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., height,
                       f'{height*100:.1f}%',
                       ha='center', va='bottom', fontsize=10, weight='bold')
    
    axes[0, 0].set_ylabel('Accuracy', fontsize=11, weight='bold')
    axes[0, 0].set_title('Overall Accuracy by Branch', fontsize=12, weight='bold')
    axes[0, 0].set_ylim(0, 1.1)
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # Plot 2: Agreement Analysis
    agreement_data = {
        '3/3 Agree': np.mean(agreement_3) if agreement_3 else 0,
        '2/3 Agree': np.mean(agreement_2) if agreement_2 else 0
    }
    
    bars = axes[0, 1].bar(agreement_data.keys(), agreement_data.values(),
                         color=['#27ae60', '#f39c12'], alpha=0.7, 
                         edgecolor='black', linewidth=1.5)
    
    for bar in bars:
        height = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                       f'{height*100:.1f}%',
                       ha='center', va='bottom', fontsize=10, weight='bold')
    
    axes[0, 1].set_ylabel('Accuracy', fontsize=11, weight='bold')
    axes[0, 1].set_title('Accuracy by Agreement Level', fontsize=12, weight='bold')
    axes[0, 1].set_ylim(0, 1.1)
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # Plot 3: Agreement Distribution
    agreement_counts = {
        '3/3 Agree': len(agreement_3),
        '2/3 Agree': len(agreement_2)
    }
    
    axes[0, 2].pie(agreement_counts.values(), labels=agreement_counts.keys(),
                   autopct='%1.1f%%', colors=['#27ae60', '#f39c12'],
                   startangle=90, textprops={'fontsize': 10, 'weight': 'bold'})
    axes[0, 2].set_title('Agreement Distribution', fontsize=12, weight='bold')
    
    # Plot 4-6: Per-branch confusion patterns (simplified)
    for idx, (name, correct_list, color) in enumerate([
        ('Cross-Modal', cm_correct, '#e74c3c'),
        ('Audio-Only', ao_correct, '#3498db'),
        ('Visual-Only', vo_correct, '#2ecc71')
    ]):
        ax = axes[1, idx]
        
        # Count correct/incorrect
        num_correct = sum(correct_list)
        num_incorrect = len(correct_list) - num_correct
        
        bars = ax.bar(['Correct', 'Incorrect'], 
                     [num_correct, num_incorrect],
                     color=[color, '#e74c3c'], alpha=0.7,
                     edgecolor='black', linewidth=1.5)
        
        for bar, val in zip(bars, [num_correct, num_incorrect]):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val}\n({val/len(correct_list)*100:.1f}%)',
                   ha='center', va='bottom', fontsize=9, weight='bold')
        
        ax.set_ylabel('Count', fontsize=11, weight='bold')
        ax.set_title(f'{name} Branch', fontsize=12, weight='bold')
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[INFO] Saved statistics plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize three-branch results')
    
    parser.add_argument('--results', type=str, required=True,
                        help='Path to results JSON file')
    parser.add_argument('--output_dir', type=str, default='./visualizations/three_branch',
                        help='Output directory for visualizations')
    parser.add_argument('--fps', type=float, default=25.0,
                        help='Video frame rate')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Detection threshold')
    parser.add_argument('--max_videos', type=int, default=20,
                        help='Maximum videos to visualize')
    
    args = parser.parse_args()
    
    # Load results
    print(f"[INFO] Loading results from {args.results}")
    with open(args.results, 'r') as f:
        results = json.load(f)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot overall statistics
    print(f"[INFO] Generating statistics plots...")
    plot_branch_comparison_statistics(
        results,
        save_path=str(output_dir / 'branch_statistics.png')
    )
    
    print(f"\n[INFO] Visualizations saved to {output_dir}")


if __name__ == '__main__':
    main()

