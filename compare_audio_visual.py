"""
Audio-Visual Feature Alignment and Similarity Computation
Compares speech motion latent with bottom-face visual embeddings for deepfake detection
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity


def load_embeddings(audio_npz, visual_npz):
    """
    Load audio and visual embeddings from .npz files
    
    Args:
        audio_npz: Path to audio embeddings (T_audio, 512)
        visual_npz: Path to visual embeddings (T_visual, 512)
    
    Returns:
        audio_emb: (T_audio, 512)
        visual_emb: (T_visual, 512)
    """
    print(f"Loading audio embeddings from: {audio_npz}")
    audio_data = np.load(audio_npz)
    audio_emb = audio_data['embeddings']
    print(f"  Audio embeddings shape: {audio_emb.shape}")
    print(f"  Mean: {audio_emb.mean():.6f}, Std: {audio_emb.std():.6f}")
    
    print(f"\nLoading visual embeddings from: {visual_npz}")
    visual_data = np.load(visual_npz)
    visual_emb = visual_data['embeddings']
    print(f"  Visual embeddings shape: {visual_emb.shape}")
    print(f"  Mean: {visual_emb.mean():.6f}, Std: {visual_emb.std():.6f}")
    
    return audio_emb, visual_emb


def align_embeddings(audio_emb, visual_emb, strategy='min_length'):
    """
    Align audio and visual embeddings to same temporal length
    
    Args:
        audio_emb: (T_audio, D)
        visual_emb: (T_visual, D)
        strategy: 'min_length' | 'interpolate' | 'pad'
    
    Returns:
        audio_aligned: (T, D)
        visual_aligned: (T, D)
        alignment_info: dict
    """
    T_audio, D_audio = audio_emb.shape
    T_visual, D_visual = visual_emb.shape
    
    print(f"\n=== Temporal Alignment ===")
    print(f"Audio frames: {T_audio}, Visual frames: {T_visual}")
    
    if T_audio == T_visual:
        print("  Frames already aligned! No action needed.")
        return audio_emb, visual_emb, {'method': 'none', 'original_lengths': (T_audio, T_visual)}
    
    if strategy == 'min_length':
        # Truncate to shorter sequence
        T = min(T_audio, T_visual)
        audio_aligned = audio_emb[:T]
        visual_aligned = visual_emb[:T]
        print(f"  Strategy: min_length, truncated to {T} frames")
        
    elif strategy == 'interpolate':
        # Interpolate to match longer sequence (not implemented yet)
        raise NotImplementedError("Interpolation not yet supported")
    
    elif strategy == 'pad':
        # Pad shorter sequence with zeros
        raise NotImplementedError("Padding not yet supported")
    
    else:
        raise ValueError(f"Unknown alignment strategy: {strategy}")
    
    info = {
        'method': strategy,
        'original_lengths': (T_audio, T_visual),
        'aligned_length': T
    }
    
    return audio_aligned, visual_aligned, info


def compute_similarity(audio_emb, visual_emb, metric='cosine'):
    """
    Compute frame-level similarity between audio and visual embeddings
    
    Args:
        audio_emb: (T, D)
        visual_emb: (T, D)
        metric: 'cosine' | 'l2'
    
    Returns:
        similarities: (T,) frame-level similarity scores
    """
    T = audio_emb.shape[0]
    assert visual_emb.shape[0] == T, "Audio and visual must have same length"
    
    print(f"\n=== Computing Similarity ===")
    print(f"Metric: {metric}")
    
    if metric == 'cosine':
        # Compute cosine similarity for each frame pair
        similarities = np.array([
            cosine_similarity(audio_emb[i:i+1], visual_emb[i:i+1])[0, 0]
            for i in range(T)
        ])
    
    elif metric == 'l2':
        # Compute negative L2 distance (higher = more similar)
        similarities = -np.linalg.norm(audio_emb - visual_emb, axis=1)
    
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    print(f"  Similarity shape: {similarities.shape}")
    print(f"  Min: {similarities.min():.4f}, Max: {similarities.max():.4f}")
    print(f"  Mean: {similarities.mean():.4f}, Std: {similarities.std():.4f}")
    
    return similarities


def extract_statistics(similarities, percentiles=[10, 25, 50, 75, 90]):
    """
    Extract statistical features from similarity curve
    These features can be used as input to a classifier
    
    Args:
        similarities: (T,) similarity scores
        percentiles: list of percentile values to compute
    
    Returns:
        stats: dict of statistical features
    """
    stats = {
        'mean': np.mean(similarities),
        'std': np.std(similarities),
        'min': np.min(similarities),
        'max': np.max(similarities),
        'median': np.median(similarities),
        'range': np.max(similarities) - np.min(similarities),
    }
    
    # Add percentiles
    for p in percentiles:
        stats[f'p{p}'] = np.percentile(similarities, p)
    
    # Temporal features
    stats['num_frames'] = len(similarities)
    
    # Low similarity region features (potential anomaly)
    threshold_low = np.mean(similarities) - 2 * np.std(similarities)
    low_sim_frames = similarities < threshold_low
    stats['num_low_sim_frames'] = np.sum(low_sim_frames)
    stats['ratio_low_sim_frames'] = np.sum(low_sim_frames) / len(similarities)
    
    print("\n=== Statistical Features ===")
    for key, val in stats.items():
        print(f"  {key}: {val:.6f}")
    
    return stats


def plot_similarity_curve(similarities, output_path=None, title="Audio-Visual Similarity"):
    """
    Visualize similarity curve over time
    
    Args:
        similarities: (T,) similarity scores
        output_path: If provided, save plot to this path
        title: Plot title
    """
    plt.figure(figsize=(12, 6))
    
    # Main similarity curve
    plt.subplot(2, 1, 1)
    plt.plot(similarities, linewidth=1.5, color='blue', alpha=0.7)
    plt.axhline(y=similarities.mean(), color='red', linestyle='--', label=f'Mean: {similarities.mean():.4f}')
    plt.axhline(y=similarities.mean() - 2*similarities.std(), color='orange', linestyle=':', 
                label=f'Mean - 2*Std: {similarities.mean() - 2*similarities.std():.4f}')
    plt.xlabel('Frame Index')
    plt.ylabel('Cosine Similarity')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Histogram
    plt.subplot(2, 1, 2)
    plt.hist(similarities, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    plt.axvline(x=similarities.mean(), color='red', linestyle='--', label=f'Mean: {similarities.mean():.4f}')
    plt.xlabel('Similarity Score')
    plt.ylabel('Frequency')
    plt.title('Similarity Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved similarity plot to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    """
    Main pipeline for audio-visual comparison
    """
    print("="*60)
    print("Audio-Visual Feature Alignment and Similarity Computation")
    print("="*60)
    
    # Default paths (can be overridden by command-line args)
    audio_npz = "test2_audio_embeddings.npz"
    visual_npz = "test2_visual_embeddings_insightface.npz"
    output_dir = "."
    
    # Check if audio embeddings exist
    if not os.path.exists(audio_npz):
        print(f"\n[WARNING] Audio embeddings not found: {audio_npz}")
        print("Please run speech encoder first:")
        print("  python speech_encoder_anitalker.py")
        
        # Try to generate audio embeddings automatically
        print("\nAttempting to generate audio embeddings...")
        try:
            import speech_encoder_anitalker as speech_enc
            import torch
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            encoder = speech_enc.SpeechMotionEncoder(device=device)
            
            audio_path = "test2_audio.wav"
            if not os.path.exists(audio_path):
                print(f"ERROR: Audio file not found: {audio_path}")
                print("Please extract audio first:")
                print("  python audio_extractor.py")
                sys.exit(1)
            
            print(f"Processing: {audio_path}")
            motion_latent = encoder.process_audio_file(audio_path)
            
            # Save to .npz
            np.savez(audio_npz, embeddings=motion_latent.cpu().detach().numpy())
            print(f"Saved audio embeddings to: {audio_npz}")
            
        except Exception as e:
            print(f"ERROR generating audio embeddings: {e}")
            sys.exit(1)
    
    if not os.path.exists(visual_npz):
        print(f"\nERROR: Visual embeddings not found: {visual_npz}")
        print("Please run InsightFace encoder first:")
        print("  python face_encoder_insightface.py")
        sys.exit(1)
    
    # Load embeddings
    audio_emb, visual_emb = load_embeddings(audio_npz, visual_npz)
    
    # Align embeddings
    audio_aligned, visual_aligned, align_info = align_embeddings(
        audio_emb, visual_emb, strategy='min_length'
    )
    
    # Compute similarity
    similarities = compute_similarity(audio_aligned, visual_aligned, metric='cosine')
    
    # Extract statistics
    stats = extract_statistics(similarities)
    
    # Save statistics
    stats_path = os.path.join(output_dir, "test2_similarity_stats.npz")
    np.savez(stats_path, similarities=similarities, **stats)
    print(f"\nSaved statistics to: {stats_path}")
    
    # Visualize
    plot_path = os.path.join(output_dir, "test2_similarity_curve.png")
    plot_similarity_curve(similarities, output_path=plot_path, 
                          title=f"test2.mp4 - Audio-Visual Similarity (frames={len(similarities)})")
    
    print("\n" + "="*60)
    print("Analysis complete!")
    print("="*60)
    print("\nNext steps:")
    print("  1. Collect real + fake videos")
    print("  2. Extract features for all videos")
    print("  3. Train a lightweight classifier on statistical features")
    print("  4. Evaluate performance (AUC, accuracy, etc.)")


if __name__ == "__main__":
    main()

