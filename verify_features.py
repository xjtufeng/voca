import numpy as np
import os
import sys

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

import matplotlib.pyplot as plt


def verify_embeddings(npz_path):
    """验证提取的特征"""
    if not os.path.exists(npz_path):
        print(f"❌ 文件不存在: {npz_path}")
        return False
    
    data = np.load(npz_path)
    
    embeddings = data['embeddings']
    frame_indices = data['frame_indices']
    timestamps = data['timestamps']
    
    print(f"\n{'='*60}")
    print(f"✅ 特征文件: {npz_path}")
    print(f"{'='*60}")
    print(f"   📊 特征形状: {embeddings.shape}")
    print(f"   🎞️  帧数量: {len(frame_indices)}")
    print(f"   ⏱️  时间范围: {timestamps[0]:.2f}s ~ {timestamps[-1]:.2f}s")
    print(f"   📏 特征维度: {embeddings.shape[1]}")
    print(f"\n   📈 特征统计:")
    print(f"      • 均值: {embeddings.mean():.6f}")
    print(f"      • 标准差: {embeddings.std():.6f}")
    print(f"      • 最小值: {embeddings.min():.6f}")
    print(f"      • 最大值: {embeddings.max():.6f}")
    print(f"      • 中位数: {np.median(embeddings):.6f}")
    
    # 检查是否有异常值
    issues = []
    if np.any(np.isnan(embeddings)):
        issues.append("⚠️  警告: 存在 NaN 值")
    if np.any(np.isinf(embeddings)):
        issues.append("⚠️  警告: 存在 Inf 值")
    if embeddings.std() < 1e-6:
        issues.append("⚠️  警告: 标准差过小，可能是常数特征")
    if embeddings.std() > 1e6:
        issues.append("⚠️  警告: 标准差过大，可能未归一化")
    
    # 检查权重是否随机初始化（全零或接近零）
    if np.abs(embeddings.mean()) < 1e-6 and embeddings.std() < 0.1:
        issues.append("⚠️  警告: 特征接近全零，VGG权重可能未正确加载")
    
    if issues:
        print(f"\n   ⚠️  检测到问题:")
        for issue in issues:
            print(f"      {issue}")
    else:
        print(f"\n   ✅ 特征质量检查通过")
    
    # 计算帧间相似度（余弦相似度）
    if len(embeddings) > 1:
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = cosine_similarity(embeddings[i:i+1], embeddings[i+1:i+2])[0][0]
            similarities.append(sim)
        
        avg_sim = np.mean(similarities)
        print(f"\n   🔗 帧间相似度:")
        print(f"      • 平均相似度: {avg_sim:.4f}")
        print(f"      • 相似度范围: [{min(similarities):.4f}, {max(similarities):.4f}]")
        
        if avg_sim > 0.99:
            print(f"      ⚠️  相似度过高，特征可能缺乏区分性")
        elif avg_sim < 0.5:
            print(f"      ⚠️  相似度过低，帧间变化可能过大")
        else:
            print(f"      ✅ 相似度合理")
    
    return True


def visualize_features(npz_path, output_dir=None):
    """可视化特征"""
    data = np.load(npz_path)
    embeddings = data['embeddings']
    timestamps = data['timestamps']
    
    # 创建可视化
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'特征可视化: {os.path.basename(npz_path)}', fontsize=16)
    
    # 1. 特征热图
    ax1 = axes[0, 0]
    im1 = ax1.imshow(embeddings[:, :100].T, aspect='auto', cmap='viridis')
    ax1.set_title('特征热图 (前100维)')
    ax1.set_xlabel('帧索引')
    ax1.set_ylabel('特征维度')
    plt.colorbar(im1, ax=ax1)
    
    # 2. 特征分布
    ax2 = axes[0, 1]
    ax2.hist(embeddings.flatten(), bins=50, alpha=0.7, edgecolor='black')
    ax2.set_title('特征值分布')
    ax2.set_xlabel('特征值')
    ax2.set_ylabel('频数')
    ax2.grid(True, alpha=0.3)
    
    # 3. 特征均值随时间变化
    ax3 = axes[1, 0]
    feature_means = embeddings.mean(axis=1)
    ax3.plot(timestamps, feature_means, marker='o', linestyle='-', markersize=4)
    ax3.set_title('特征均值随时间变化')
    ax3.set_xlabel('时间 (秒)')
    ax3.set_ylabel('特征均值')
    ax3.grid(True, alpha=0.3)
    
    # 4. 帧间相似度
    ax4 = axes[1, 1]
    if len(embeddings) > 1:
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = cosine_similarity(embeddings[i:i+1], embeddings[i+1:i+2])[0][0]
            similarities.append(sim)
        
        ax4.plot(timestamps[1:], similarities, marker='o', linestyle='-', markersize=4)
        ax4.set_title('帧间余弦相似度')
        ax4.set_xlabel('时间 (秒)')
        ax4.set_ylabel('相似度')
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=0.9, color='r', linestyle='--', alpha=0.5, label='高相似度阈值')
        ax4.legend()
    
    plt.tight_layout()
    
    # 保存图像
    if output_dir:
        vis_path = os.path.join(output_dir, 'feature_visualization.png')
        plt.savefig(vis_path, dpi=150, bbox_inches='tight')
        print(f"\n   💾 可视化已保存: {vis_path}")
    
    return fig


def main():
    """主函数"""
    print("\n" + "="*60)
    print("  🔍 VGG 底脸特征验证")
    print("="*60)
    
    output_dirs = ["test1_output", "test2_output", "test_video_output"]
    
    for output_dir in output_dirs:
        npz_path = os.path.join(output_dir, "vgg_bottom_face_embeddings.npz")
        if os.path.exists(npz_path):
            verify_embeddings(npz_path)
            
            # 尝试可视化（需要sklearn）
            try:
                visualize_features(npz_path, output_dir)
            except ImportError:
                print(f"\n   ℹ️  提示: 安装 scikit-learn 以启用相似度分析")
                print(f"      pip install scikit-learn matplotlib")
            except Exception as e:
                print(f"\n   ⚠️  可视化失败: {e}")
    
    print("\n" + "="*60)
    print("  ✅ 验证完成")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()

