import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
from typing import Tuple, Dict, List
import warnings
import pickle
import json
import pandas as pd
warnings.filterwarnings('ignore')

def _compute_curvature_highdim(traj: np.ndarray) -> np.ndarray:
    """
    在任意维度空间中计算轨迹曲率（基于相邻方向夹角）。
    返回长度为 len(traj)-2 的曲率序列，范围 [0, π]，越大表示转折越剧烈。
    """
    T = len(traj)
    if T < 3:
        return np.array([])

    curv = []
    for i in range(1, T - 1):
        v1 = traj[i]   - traj[i-1]
        v2 = traj[i+1] - traj[i]
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 > 1e-10 and n2 > 1e-10:
            cos_val = np.dot(v1, v2) / (n1 * n2)
            cos_val = np.clip(cos_val, -1.0, 1.0)
            # 定义为 π - 夹角：直线≈0，折返≈π
            curv.append(np.pi - np.arccos(cos_val))
        else:
            curv.append(0.0)
    return np.array(curv)

def compute_cosine_similarity_analysis(positions: np.ndarray, features: np.ndarray) -> Dict:
    """
    计算余弦相似度分析
    - 相邻时刻位移向量 vs 起始到终点位移向量的余弦相似度
    - 分别在高维特征空间(features)和2D空间(positions)计算
    - 分析两个空间的一致性
    """
    n_points = len(positions)
    if n_points < 3:
        return {
            'error': 'Insufficient data points for cosine similarity analysis',
            'n_points': n_points
        }
    
    positions_total_vector = positions[-1] - positions[0]  # 2D空间
    features_total_vector = features[-1] - features[0]     # 高维特征空间
    
    positions_adjacent_vectors = []  
    features_adjacent_vectors = []   
    
    for i in range(1, n_points):
        pos_vec = positions[i] - positions[i-1]
        feature_vec = features[i] - features[i-1]
        positions_adjacent_vectors.append(pos_vec)
        features_adjacent_vectors.append(feature_vec)
    
    positions_cosine_similarities = []  
    features_cosine_similarities = []   
    
    def safe_cosine_similarity(vec1, vec2):
        """安全的余弦相似度计算"""
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 < 1e-10 or norm2 < 1e-10:
            return 0.0
        return np.dot(vec1, vec2) / (norm1 * norm2)
    
    for pos_vec in positions_adjacent_vectors:
        cos_sim = safe_cosine_similarity(pos_vec, positions_total_vector)
        positions_cosine_similarities.append(cos_sim)
    
    for feature_vec in features_adjacent_vectors:
        cos_sim = safe_cosine_similarity(feature_vec, features_total_vector)
        features_cosine_similarities.append(cos_sim)
    
    positions_cosine_similarities = np.array(positions_cosine_similarities)
    features_cosine_similarities = np.array(features_cosine_similarities)
    
    if len(positions_cosine_similarities) > 1 and len(features_cosine_similarities) > 1:
        pearson_consistency = pearsonr(positions_cosine_similarities, features_cosine_similarities)[0]
        spearman_consistency = spearmanr(positions_cosine_similarities, features_cosine_similarities)[0]
        
        if np.isnan(pearson_consistency):
            pearson_consistency = 0.0
        if np.isnan(spearman_consistency):
            spearman_consistency = 0.0
    else:
        pearson_consistency = 0.0
        spearman_consistency = 0.0
    
    cosine_diff = np.abs(positions_cosine_similarities - features_cosine_similarities)
    mean_cosine_diff = np.mean(cosine_diff)
    max_cosine_diff = np.max(cosine_diff)
    
    direction_agreement = np.mean((positions_cosine_similarities > 0) == (features_cosine_similarities > 0))
    
    return {
        'positions_cosine_similarities': positions_cosine_similarities.tolist(),
        'features_cosine_similarities': features_cosine_similarities.tolist(),
        'positions_total_vector': positions_total_vector.tolist(),
        'features_total_vector': features_total_vector.tolist(),
        'positions_adjacent_vectors': [vec.tolist() for vec in positions_adjacent_vectors],
        'features_adjacent_vectors': [vec.tolist() for vec in features_adjacent_vectors],
        'consistency_metrics': {
            'pearson_correlation': float(pearson_consistency),
            'spearman_correlation': float(spearman_consistency),
            'mean_cosine_difference': float(mean_cosine_diff),
            'max_cosine_difference': float(max_cosine_diff),
            'direction_agreement_rate': float(direction_agreement)
        },
        'statistics': {
            'positions_cosine_mean': float(np.mean(positions_cosine_similarities)),
            'positions_cosine_std': float(np.std(positions_cosine_similarities)),
            'features_cosine_mean': float(np.mean(features_cosine_similarities)),
            'features_cosine_std': float(np.std(features_cosine_similarities)),
            'n_time_steps': len(positions_cosine_similarities)
        }
    }

class MotionSemanticConsistency:
    """运动与语义一致性分析器"""
    
    def __init__(self):
        self.results = {}
    
    def direction_consistency(self, positions: np.ndarray, features: np.ndarray) -> float:
        """方向一致性分析"""
        pos_deltas = np.diff(positions, axis=0)  # (n-1, 2)
        feature_deltas = np.diff(features, axis=0)     # (n-1, d)
        
        # 将高维特征变化向量投影到2D空间进行比较
        if feature_deltas.shape[1] > 2:
            pca = PCA(n_components=2)
            feature_deltas_2d = pca.fit_transform(feature_deltas)
        else:
            feature_deltas_2d = feature_deltas
        
        similarities = []
        for i in range(len(pos_deltas)):
            pos_vec = pos_deltas[i].reshape(1, -1)
            feature_vec = feature_deltas_2d[i].reshape(1, -1)
            
            if np.linalg.norm(pos_vec) > 1e-10 and np.linalg.norm(feature_vec) > 1e-10:
                cos_sim = cosine_similarity(pos_vec, feature_vec)[0, 0]
                similarities.append(cos_sim)
        
        return np.mean(similarities) if similarities else 0.0
        
    def distance_consistency(self, positions: np.ndarray, features: np.ndarray) -> Dict[str, float]:
        """
        距离一致性分析
        衡量位置变化距离与特征变化距离的相关性
        """
        n = len(positions)
        
        pos_distances = []
        feature_distances = []
        
        for i in range(1, n):
            pos_dist = np.linalg.norm(positions[i] - positions[i-1])
            feature_dist = np.linalg.norm(features[i] - features[i-1])
            
            pos_distances.append(pos_dist)
            feature_distances.append(feature_dist)
        
        pos_distances = np.array(pos_distances)
        feature_distances = np.array(feature_distances)
        
        pearson_corr = pearsonr(pos_distances, feature_distances)[0] if len(pos_distances) > 1 else 0
        spearman_corr = spearmanr(pos_distances, feature_distances)[0] if len(pos_distances) > 1 else 0
        
        return {
            'pearson': pearson_corr,
            'spearman': spearman_corr,
            'pos_distances': pos_distances,
            'feature_distances': feature_distances
        }
    
    def curvature_consistency(self, positions: np.ndarray, features: np.ndarray) -> dict:
        """
        曲率一致性分析（不做降维）
        - 在各自空间（2D / 高维特征）独立计算曲率序列
        - 比较两条曲率序列的相关性
        """
        pos_curvature = _compute_curvature_highdim(positions)
        feature_curvature = _compute_curvature_highdim(features)

        if len(pos_curvature) > 1 and len(feature_curvature) > 1 and len(pos_curvature) == len(feature_curvature):
            def _safe_corr(x, y, fn):
                if np.allclose(x, x[0]) or np.allclose(y, y[0]):
                    return 0.0
                val = fn(x, y)[0]
                return 0.0 if np.isnan(val) else float(val)

            pearson_corr  = _safe_corr(pos_curvature, feature_curvature, pearsonr)
            spearman_corr = _safe_corr(pos_curvature, feature_curvature, spearmanr)
        else:
            pearson_corr  = 0.0
            spearman_corr = 0.0

        return {
            'pearson': pearson_corr,
            'spearman': spearman_corr,
            'pos_curvature': pos_curvature,
            'feature_curvature': feature_curvature,
        }
    
    def analyze(self, positions: np.ndarray, features: np.ndarray, logits: np.ndarray = None, sample_id: int = None) -> dict:
        """综合分析（包含logits）"""
        direction_score = self.direction_consistency(positions, features)
        distance_results = self.distance_consistency(positions, features)
        curvature_results = self.curvature_consistency(positions, features)
        
        cosine_analysis = compute_cosine_similarity_analysis(positions, features)
        
        def safe_value(x):
            return x if not np.isnan(x) else 0.0
        
        scores = {
            'sample_id': sample_id,
            'direction_consistency': safe_value(direction_score),
            'distance_pearson': safe_value(distance_results['pearson']),
            'distance_spearman': safe_value(distance_results['spearman']),
            'curvature_pearson': safe_value(curvature_results['pearson']),
            'curvature_spearman': safe_value(curvature_results['spearman']),
            'cosine_consistency_pearson': safe_value(cosine_analysis['consistency_metrics']['pearson_correlation']),
            'cosine_consistency_spearman': safe_value(cosine_analysis['consistency_metrics']['spearman_correlation']),
            'cosine_mean_difference': safe_value(cosine_analysis['consistency_metrics']['mean_cosine_difference']),
            'cosine_direction_agreement': safe_value(cosine_analysis['consistency_metrics']['direction_agreement_rate']),
            'detailed_data': {
                'positions': positions.tolist(),
                'features': features.tolist(),
                'logits': logits.tolist() if logits is not None else None,
                'pos_distances': distance_results['pos_distances'].tolist(),
                'feature_distances': distance_results['feature_distances'].tolist(),
                'pos_curvature': curvature_results['pos_curvature'].tolist(),
                'feature_curvature': curvature_results['feature_curvature'].tolist(),
                'cosine_analysis': cosine_analysis
            }
        }
        
        metric_scores = [
            scores['direction_consistency'], 
            scores['distance_pearson'], 
            scores['distance_spearman'], 
            scores['curvature_pearson'], 
            scores['curvature_spearman'],
            scores['cosine_consistency_pearson'],
            scores['cosine_direction_agreement']
        ]
        overall_score = np.mean(metric_scores)
        scores['overall_score'] = overall_score
        
        return scores

def load_trajectory_data(trajectory_dir):
    """加载轨迹分析数据"""
    trajectory_pkl_file = os.path.join(trajectory_dir, "point_trajectories.pkl")
    
    if not os.path.exists(trajectory_pkl_file):
        raise FileNotFoundError(f"轨迹数据文件不存在: {trajectory_pkl_file}")
    
    with open(trajectory_pkl_file, "rb") as f:
        point_trajectories = pickle.load(f)
    
    print(f"成功加载轨迹数据，包含 {len(point_trajectories)} 个数据点")
    return point_trajectories

def load_features_data(trajectory_dir):
    """从轨迹目录加载特征数据"""
    features_pkl_file = os.path.join(trajectory_dir, "all_epoch_features.pkl")
    
    if os.path.exists(features_pkl_file):
        with open(features_pkl_file, "rb") as f:
            all_epoch_features = pickle.load(f)
        print(f"成功加载特征数据，包含 {len(all_epoch_features)} 个epochs")
        return all_epoch_features
    
    print("未找到all_epoch_features.pkl，尝试从单个epoch文件加载...")
    epoch_files = []
    for file in os.listdir(trajectory_dir):
        if file.startswith("epoch_") and file.endswith("_features.npy"):
            epoch_num = int(file.split("_")[1])
            epoch_files.append((epoch_num, file))
    
    if not epoch_files:
        raise FileNotFoundError(f"在 {trajectory_dir} 中未找到特征数据文件")
    
    epoch_files.sort()
    all_epoch_features = {}
    
    for epoch_num, filename in epoch_files:
        file_path = os.path.join(trajectory_dir, filename)
        features = np.load(file_path)
        all_epoch_features[epoch_num] = features
        print(f"加载 Epoch {epoch_num} 特征数据: {features.shape}")
    
    print(f"成功从单个文件加载特征数据，包含 {len(all_epoch_features)} 个epochs")
    return all_epoch_features

def load_logits_data(logits_dir):
    """加载logits数据"""
    logits_pkl_file = os.path.join(logits_dir, "all_epoch_logits.pkl")
    
    if not os.path.exists(logits_pkl_file):
        print(f"警告: Logits数据文件不存在: {logits_pkl_file}")
        return None
    
    with open(logits_pkl_file, "rb") as f:
        all_epoch_logits = pickle.load(f)
    
    print(f"成功加载logits数据，包含 {len(all_epoch_logits)} 个epochs")
    return all_epoch_logits

def align_all_data(point_trajectories, all_epoch_features, all_epoch_logits, selected_idxs):
    """对齐轨迹、特征和logits数据"""
    print("开始对齐轨迹、特征和logits数据...")
    
    available_epochs = set()
    for select_idx in selected_idxs:
        if select_idx in point_trajectories:
            available_epochs.update(point_trajectories[select_idx].keys())
    
    features_epochs = set(all_epoch_features.keys())
    common_epochs = available_epochs.intersection(features_epochs)
    
    if all_epoch_logits is not None:
        logits_epochs = set(all_epoch_logits.keys())
        common_epochs = common_epochs.intersection(logits_epochs)
    
    common_epochs = sorted(common_epochs)
    
    if not common_epochs:
        raise ValueError("轨迹数据、特征数据和logits数据没有共同的epochs")
    
    print(f"找到 {len(common_epochs)} 个共同epochs: {common_epochs[:5]}{'...' if len(common_epochs) > 5 else ''}")
    
    aligned_data = {}
    
    for select_idx in selected_idxs:
        if select_idx not in point_trajectories:
            print(f"警告: select_idx {select_idx} 不在轨迹数据中")
            continue
        
        trajectory = point_trajectories[select_idx]
        
        positions = []
        features = []
        logits = []
        valid_epochs = []
        
        for epoch in common_epochs:
            if epoch not in trajectory:
                continue
            
            if epoch not in all_epoch_features:
                continue
            
            embedding = trajectory[epoch]["embedding"]
            positions.append(embedding)
            
            if "feature" in trajectory[epoch]:
                epoch_feature = trajectory[epoch]["feature"]
                features.append(epoch_feature)
            else:
                epoch_features_data = all_epoch_features[epoch]
                if select_idx < len(epoch_features_data):
                    epoch_feature = epoch_features_data[select_idx]
                    features.append(epoch_feature)
                else:
                    print(f"警告: select_idx {select_idx} 在epoch {epoch} 的特征数据中未找到")
                    continue
            
            if all_epoch_logits is not None and epoch in all_epoch_logits:
                epoch_logits_data = all_epoch_logits[epoch]
                valid_indices = epoch_logits_data["valid_indices"]
                
                if select_idx in valid_indices:
                    logits_index = valid_indices.index(select_idx)
                    epoch_logits = epoch_logits_data["logits"][logits_index]
                    logits.append(epoch_logits)
                else:
                    print(f"警告: select_idx {select_idx} 在epoch {epoch} 的logits数据中未找到")
                    continue
            else:
                logits.append(None)
            
            valid_epochs.append(epoch)
        
        if len(positions) >= 3 and len(features) >= 3:
            aligned_data[select_idx] = {
                "positions": np.array(positions),
                "features": np.array(features),
                "logits": np.array(logits) if all(l is not None for l in logits) else None,
                "epochs": valid_epochs
            }
        else:
            print(f"警告: select_idx {select_idx} 的有效数据点不足 (positions: {len(positions)}, features: {len(features)})")
    
    print(f"成功对齐 {len(aligned_data)} 个数据点的所有数据")
    return aligned_data

def create_individual_sample_visualization(sample_id, sample_data, save_dir):
    """为每个样本创建简化的可视化（只包含2D轨迹和分析图）"""
    sample_viz_dir = os.path.join(save_dir, "individual_samples", f"sample_{sample_id}")
    os.makedirs(sample_viz_dir, exist_ok=True)
    
    positions = sample_data["positions"]
    features = sample_data["features"]
    logits = sample_data["logits"]
    epochs = sample_data["epochs"]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Sample {sample_id} - Analysis', fontsize=16)
    
    ax1 = axes[0, 0]
    ax1.plot(positions[:, 0], positions[:, 1], 'o-', linewidth=2, markersize=8, alpha=0.7)
    ax1.scatter(positions[0, 0], positions[0, 1], color='green', s=150, label='Start', zorder=5)
    ax1.scatter(positions[-1, 0], positions[-1, 1], color='red', s=150, label='End', zorder=5)
    
    for i, epoch in enumerate(epochs):
        if i % max(1, len(epochs)//5) == 0:  
            ax1.annotate(f'E{epoch}', (positions[i, 0], positions[i, 1]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax1.set_title('2D Trajectory')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[0, 1]
    pos_distances = [np.linalg.norm(positions[i] - positions[i-1]) for i in range(1, len(positions))]
    feature_distances = [np.linalg.norm(features[i] - features[i-1]) for i in range(1, len(features))]
    
    transition_epochs = [f'E{epochs[i-1]}-E{epochs[i]}' for i in range(1, len(epochs))]
    x_pos = range(len(pos_distances))
    
    ax2.plot(x_pos, pos_distances, 'o-', label='2D Space', linewidth=2, markersize=6)
    ax2.plot(x_pos, feature_distances, 's-', label='Feature Space', linewidth=2, markersize=6)
    
    if logits is not None:
        logits_distances = [np.linalg.norm(logits[i] - logits[i-1]) for i in range(1, len(logits))]
        ax2.plot(x_pos, logits_distances, '^-', label='Logits Space', linewidth=2, markersize=6)
    
    ax2.set_title('Distance Changes Between Epochs')
    ax2.set_xlabel('Epoch Transition')
    ax2.set_ylabel('Distance')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    if len(transition_epochs) <= 10:
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(transition_epochs, rotation=45, ha='right')
    else:
        step = len(transition_epochs) // 10
        ax2.set_xticks(x_pos[::step])
        ax2.set_xticklabels([transition_epochs[i] for i in range(0, len(transition_epochs), step)], 
                           rotation=45, ha='right')
    
    ax3 = axes[1, 0]
    
    pos_total_vec = positions[-1] - positions[0]
    feature_total_vec = features[-1] - features[0]
    
    pos_cosines = []
    feature_cosines = []
    logits_cosines = []
    
    def safe_cosine_sim(v1, v2):
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-10 or n2 < 1e-10:
            return 0.0
        return np.dot(v1, v2) / (n1 * n2)
    
    for i in range(1, len(positions)):
        pos_vec = positions[i] - positions[i-1]
        feature_vec = features[i] - features[i-1]
        
        pos_cos = safe_cosine_sim(pos_vec, pos_total_vec)
        feature_cos = safe_cosine_sim(feature_vec, feature_total_vec)
        
        pos_cosines.append(pos_cos)
        feature_cosines.append(feature_cos)
        
        if logits is not None:
            logits_total_vec = logits[-1] - logits[0]
            logits_vec = logits[i] - logits[i-1]
            logits_cos = safe_cosine_sim(logits_vec, logits_total_vec)
            logits_cosines.append(logits_cos)
    
    ax3.plot(x_pos, pos_cosines, 'o-', label='2D Space', linewidth=2, markersize=6)
    ax3.plot(x_pos, feature_cosines, 's-', label='Feature Space', linewidth=2, markersize=6)
    
    if logits is not None and logits_cosines:
        ax3.plot(x_pos, logits_cosines, '^-', label='Logits Space', linewidth=2, markersize=6)
    
    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax3.set_title('Cosine Similarity with Total Displacement')
    ax3.set_xlabel('Epoch Transition')
    ax3.set_ylabel('Cosine Similarity')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(-1.1, 1.1)
    
    if len(transition_epochs) <= 10:
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(transition_epochs, rotation=45, ha='right')
    else:
        step = len(transition_epochs) // 10
        ax3.set_xticks(x_pos[::step])
        ax3.set_xticklabels([transition_epochs[i] for i in range(0, len(transition_epochs), step)], 
                           rotation=45, ha='right')
    
    ax4 = axes[1, 1]
    
    correlations = []
    labels = []
    
    if len(pos_distances) > 1:
        corr_pos_feat = pearsonr(pos_distances, feature_distances)[0]
        correlations.append(corr_pos_feat)
        labels.append('2D vs Feature\n(Distance)')
    
    if len(pos_cosines) > 1:
        corr_pos_feat_cos = pearsonr(pos_cosines, feature_cosines)[0]
        correlations.append(corr_pos_feat_cos)
        labels.append('2D vs Feature\n(Cosine)')
    
    if logits is not None:
        if len(logits_distances) > 1:
            corr_pos_logits = pearsonr(pos_distances, logits_distances)[0]
            corr_feat_logits = pearsonr(feature_distances, logits_distances)[0]
            correlations.extend([corr_pos_logits, corr_feat_logits])
            labels.extend(['2D vs Logits\n(Distance)', 'Feature vs Logits\n(Distance)'])
        
        if len(logits_cosines) > 1:
            corr_pos_logits_cos = pearsonr(pos_cosines, logits_cosines)[0]
            corr_feat_logits_cos = pearsonr(feature_cosines, logits_cosines)[0]
            correlations.extend([corr_pos_logits_cos, corr_feat_logits_cos])
            labels.extend(['2D vs Logits\n(Cosine)', 'Feature vs Logits\n(Cosine)'])
    
    correlations = [0.0 if np.isnan(c) else c for c in correlations]
    
    if correlations:
        colors = ['green' if c > 0.5 else 'orange' if c > 0 else 'red' for c in correlations]
        bars = ax4.bar(range(len(correlations)), correlations, color=colors, alpha=0.7)
        ax4.set_xticks(range(len(correlations)))
        ax4.set_xticklabels(labels, rotation=45, ha='right')
        ax4.set_ylabel('Pearson Correlation')
        ax4.set_title('Cross-Space Correlations')
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(-1, 1)
        
        # 添加数值标签
        for i, (bar, corr) in enumerate(zip(bars, correlations)):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{corr:.3f}', ha='center', va='bottom', fontsize=9)
    else:
        ax4.text(0.5, 0.5, 'Insufficient Data\nfor Correlation', 
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Cross-Space Correlations')
    
    plt.tight_layout()
    
    main_viz_file = os.path.join(sample_viz_dir, f'sample_{sample_id}_analysis.png')
    plt.savefig(main_viz_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return sample_viz_dir, main_viz_file

def save_individual_sample_data(sample_id, sample_data, sample_viz_dir):
    """保存每个样本的两个CSV文件"""
    positions = sample_data["positions"]
    features = sample_data["features"]
    logits = sample_data["logits"]
    epochs = sample_data["epochs"]
    
    epoch_data = []
    
    for i, epoch in enumerate(epochs):
        row = {
            'epoch': epoch,
            'position_x': positions[i, 0],
            'position_y': positions[i, 1],
        }
        
        for j in range(features.shape[1]):
            row[f'feature_dim_{j}'] = features[i, j]
        
        if logits is not None:
            for j in range(logits.shape[1]):
                row[f'logits_dim_{j}'] = logits[i, j]
        
        epoch_data.append(row)
    
    epoch_df = pd.DataFrame(epoch_data)
    epoch_csv_file = os.path.join(sample_viz_dir, f'sample_{sample_id}_epoch_data.csv')
    epoch_df.to_csv(epoch_csv_file, index=False, float_format='%.6f')
    
    transition_data = []
    
    def safe_cosine_sim(v1, v2):
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-10 or n2 < 1e-10:
            return 0.0
        return np.dot(v1, v2) / (n1 * n2)
    
    pos_total_vec = positions[-1] - positions[0]
    feature_total_vec = features[-1] - features[0]
    logits_total_vec = None
    if logits is not None:
        logits_total_vec = logits[-1] - logits[0]
    
    for i in range(1, len(epochs)):
        row = {
            'from_epoch': epochs[i-1],
            'to_epoch': epochs[i],
            'transition': f'{epochs[i-1]}-{epochs[i]}'
        }
        
        pos_distance = np.linalg.norm(positions[i] - positions[i-1])
        feature_distance = np.linalg.norm(features[i] - features[i-1])
        row['pos_distance'] = pos_distance
        row['feature_distance'] = feature_distance
        
        if logits is not None:
            logits_distance = np.linalg.norm(logits[i] - logits[i-1])
            row['logits_distance'] = logits_distance
        
        pos_vec = positions[i] - positions[i-1]
        feature_vec = features[i] - features[i-1]
        
        pos_cosine = safe_cosine_sim(pos_vec, pos_total_vec)
        feature_cosine = safe_cosine_sim(feature_vec, feature_total_vec)
        
        row['pos_cosine_with_total'] = pos_cosine
        row['feature_cosine_with_total'] = feature_cosine
        
        if logits is not None and logits_total_vec is not None:
            logits_vec = logits[i] - logits[i-1]
            logits_cosine = safe_cosine_sim(logits_vec, logits_total_vec)
            row['logits_cosine_with_total'] = logits_cosine
        
        row['pos_delta_x'] = pos_vec[0]
        row['pos_delta_y'] = pos_vec[1]
        
        row['feature_vector_norm'] = np.linalg.norm(feature_vec)
        
        feature_dims_to_save = min(5, len(feature_vec))
        for j in range(feature_dims_to_save):
            row[f'feature_delta_dim_{j}'] = feature_vec[j]
        
        if logits is not None:
            logits_vec = logits[i] - logits[i-1]
            row['logits_vector_norm'] = np.linalg.norm(logits_vec)
            
            # logits向量的前几个分量
            logits_dims_to_save = min(5, len(logits_vec))
            for j in range(logits_dims_to_save):
                row[f'logits_delta_dim_{j}'] = logits_vec[j]
        
        transition_data.append(row)
    
    transition_df = pd.DataFrame(transition_data)
    transition_csv_file = os.path.join(sample_viz_dir, f'sample_{sample_id}_transition_data.csv')
    transition_df.to_csv(transition_csv_file, index=False, float_format='%.6f')
    
    np_data = {
        'positions': positions,
        'features': features,
        'logits': logits,
        'epochs': epochs,
        'sample_id': sample_id
    }
    np_file = os.path.join(sample_viz_dir, f'sample_{sample_id}_raw_data.npz')
    if logits is not None:
        np.savez(np_file, **np_data)
    else:
        np_data_no_logits = {k: v for k, v in np_data.items() if k != 'logits'}
        np.savez(np_file, **np_data_no_logits)
    
    return epoch_csv_file, transition_csv_file, np_file

def analyze_consistency_batch(aligned_data):
    """批量分析一致性并创建个体可视化"""
    analyzer = MotionSemanticConsistency()
    all_results = []
    sample_results = {}
    
    print(f"\n开始批量分析 {len(aligned_data)} 个轨迹的一致性...")
    
    for i, (sample_id, data) in enumerate(aligned_data.items()):
        print(f"分析样本 {sample_id} ({i+1}/{len(aligned_data)})")
        
        positions = data["positions"]
        features = data["features"]
        logits = data["logits"]
        
        results = analyzer.analyze(positions, features, logits, sample_id)
        all_results.append(results)
        sample_results[sample_id] = results
    
    return all_results, sample_results

def create_all_individual_visualizations(aligned_data, save_dir):
    """为所有样本创建个体可视化和数据文件"""
    print(f"\n为 {len(aligned_data)} 个样本创建个体可视化和数据文件...")
    
    individual_viz_dir = os.path.join(save_dir, "individual_samples")
    os.makedirs(individual_viz_dir, exist_ok=True)
    
    summary_data = []
    
    for i, (sample_id, data) in enumerate(aligned_data.items()):
        print(f"创建样本 {sample_id} 的可视化和数据文件 ({i+1}/{len(aligned_data)})")
        
        sample_viz_dir, main_viz_file = create_individual_sample_visualization(sample_id, data, save_dir)
        
        epoch_csv_file, transition_csv_file, np_file = save_individual_sample_data(sample_id, data, sample_viz_dir)
        
        summary_data.append({
            'sample_id': sample_id,
            'n_epochs': len(data["epochs"]),
            'start_epoch': data["epochs"][0],
            'end_epoch': data["epochs"][-1],
            'feature_dim': data["features"].shape[1],
            'logits_dim': data["logits"].shape[1] if data["logits"] is not None else None,
            'viz_file': main_viz_file,
            'epoch_csv_file': epoch_csv_file,
            'transition_csv_file': transition_csv_file,
            'np_file': np_file
        })
    
    # 保存摘要
    summary_df = pd.DataFrame(summary_data)
    summary_file = os.path.join(individual_viz_dir, "samples_summary.csv")
    summary_df.to_csv(summary_file, index=False)
    
    print(f"所有个体可视化已保存到: {individual_viz_dir}")
    print(f"摘要文件: {summary_file}")
    
    return individual_viz_dir, summary_file

def calculate_statistics(results_list):
    """计算统计结果"""
    if not results_list:
        print("没有分析结果可以统计")
        return {}
    
    sample_keys = list(results_list[0].keys())
    print(f"分析结果包含的指标: {sample_keys}")
    
    metrics = ['direction_consistency', 'distance_pearson', 'distance_spearman', 
               'curvature_pearson', 'curvature_spearman', 'overall_score',
               'cosine_consistency_pearson', 'cosine_consistency_spearman',
               'cosine_mean_difference', 'cosine_direction_agreement']
    
    stats = {}
    for metric in metrics:
        if metric in sample_keys:  # 只处理存在的键
            values = [r[metric] for r in results_list]
            stats[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'median': np.median(values)
            }
        else:
            print(f"警告: 键 '{metric}' 不存在")
    
    return stats

def print_analysis_results(stats):
    """打印分析结果"""
    print("\n" + "="*80)
    print("运动与语义一致性分析结果（2D空间 vs 高维特征空间 vs Logits空间）")
    print("="*80)
    
    if not stats:
        print("没有可显示的统计结果")
        return
    
    for metric, values in stats.items():
        print(f"\n{metric.upper().replace('_', ' ')}:")
        print(f"  平均值: {values['mean']:.4f}")
        print(f"  标准差: {values['std']:.4f}")
        print(f"  中位数: {values['median']:.4f}")
        print(f"  范围: [{values['min']:.4f}, {values['max']:.4f}]")

def save_analysis_results(stats, results_list, save_dir):
    """保存分析结果"""
    os.makedirs(save_dir, exist_ok=True)
    
    stats_file = os.path.join(save_dir, "motion_semantic_consistency_stats.json")
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)
    
    results_file = os.path.join(save_dir, "motion_semantic_consistency_results.pkl")
    with open(results_file, "wb") as f:
        pickle.dump(results_list, f)
    
    print(f"\n分析结果已保存到:")
    print(f"  - 统计结果: {stats_file}")
    print(f"  - 详细结果: {results_file}")

def load_and_prepare_real_data(trajectory_dir, logits_dir, selected_idxs):
    """加载并准备真实数据用于分析（包含logits）"""
    print("="*80)
    print("加载并准备真实数据用于分析（2D空间 vs 高维特征空间 vs Logits空间）")
    print("="*80)
    
    print("\n1. 加载轨迹数据...")
    point_trajectories = load_trajectory_data(trajectory_dir)
    
    print("\n2. 加载特征数据...")
    all_epoch_features = load_features_data(trajectory_dir)
    
    print("\n3. 加载logits数据...")
    all_epoch_logits = load_logits_data(logits_dir)
    
    print("\n4. 对齐所有数据...")
    aligned_data = align_all_data(point_trajectories, all_epoch_features, all_epoch_logits, selected_idxs)
    
    print(f"\n5. 数据统计信息:")
    print(f"   - 成功处理的数据点数: {len(aligned_data)}")
    print(f"   - Selected indices: {selected_idxs[:10]}{'...' if len(selected_idxs) > 10 else ''}")
    print(f"   - 实际可用的数据点: {list(aligned_data.keys())[:10]}{'...' if len(aligned_data) > 10 else ''}")
    
    if aligned_data:
        sample_data = next(iter(aligned_data.values()))
        epoch_counts = [len(data["epochs"]) for data in aligned_data.values()]
        feature_dims = [data["features"].shape[1] for data in aligned_data.values()]
        
        print(f"   - 每个轨迹的epoch数量: min={min(epoch_counts)}, max={max(epoch_counts)}, avg={np.mean(epoch_counts):.1f}")
        print(f"   - 特征维度: {feature_dims[0]} (所有轨迹应该相同)")
        
        if sample_data["logits"] is not None:
            logits_dims = [data["logits"].shape[1] for data in aligned_data.values() if data["logits"] is not None]
            print(f"   - Logits维度: {logits_dims[0] if logits_dims else 'None'}")
        else:
            print(f"   - Logits维度: None (没有logits数据)")
    
    return aligned_data

def main_analysis(trajectory_dir, logits_dir, selected_idxs, save_dir=None):
    """主分析函数（包含个体可视化）"""
    print("开始运动与语义一致性分析（包含个体样本可视化）...")
    aligned_data = load_and_prepare_real_data(trajectory_dir, logits_dir, selected_idxs)
    
    if not aligned_data:
        print("错误: 没有可用的数据进行分析")
        return None, None, None
    
    if save_dir:
        individual_viz_dir, summary_file = create_all_individual_visualizations(aligned_data, save_dir)
    
    results_list, sample_results = analyze_consistency_batch(aligned_data)
    
    stats = calculate_statistics(results_list)
    
    print_analysis_results(stats)
    
    if save_dir:
        save_analysis_results(stats, results_list, save_dir)
    
    return stats, results_list, aligned_data


if __name__ == "__main__":
    dataset = "cifar10"
    method = "dynavis"
    selected_idxs = list(range(200))
    if dataset == "cifar10":
        content_path = "/home/zicong/data/CIFAR10"
    elif dataset == "fmnist":
        content_path = "/home/zicong/data/fminist_resnet18"
    elif dataset == "backdoor":
        content_path = "/home/zicong/data/backdoor_attack/dynamic/resnet18_MNIST_noise_salt_pepper_0.05_s0_t1"
    elif dataset == "badnet_noise":
        content_path = "/home/zicong/data/BadNet_MNIST_noise_salt_pepper_s0_t0"

    trajectory_dir = os.path.join(content_path, f"Model/{method}_trajectory/trajectory_analysis_{len(selected_idxs)}")
    logits_dir = os.path.join(content_path, f"Model/logits_data")
    save_dir = os.path.join("/home/zicong/DynaVis/result/dynavis_result", f"{dataset}/{len(selected_idxs)}")
    os.makedirs(save_dir, exist_ok=True)
    
    stats, results, aligned_data = main_analysis(
        trajectory_dir=trajectory_dir,
        logits_dir=logits_dir,
        selected_idxs=selected_idxs,
        save_dir=save_dir
    )