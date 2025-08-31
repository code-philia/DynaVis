import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
import pandas as pd
import seaborn as sns
from typing import Dict, List, Tuple
import warnings
import json
warnings.filterwarnings('ignore')

class TrajectoryFidelityAnalyzer:
    """轨迹保真度分析器 - 专注于训练后期logits稳定时期的分析"""
    
    def __init__(self, logits_stability_threshold=0.1, late_training_ratio=0.7):
        """
        Args:
            logits_stability_threshold: logits变化小于此阈值认为是稳定的
            late_training_ratio: 训练后期的比例 (0.7表示后30%的epochs)
        """
        self.logits_stability_threshold = logits_stability_threshold
        self.late_training_ratio = late_training_ratio
        
    def identify_stable_period(self, logits_distances: np.ndarray, epochs: List[int]) -> Tuple[int, int, List[int]]:
        """
        识别logits稳定的后期训练阶段
        
        Returns:
            start_idx: 稳定期开始的索引
            end_idx: 稳定期结束的索引  
            stable_epochs: 稳定期的epoch列表
        """
        late_start_idx = int(len(epochs) * self.late_training_ratio)

        return late_start_idx, len(epochs) - 1, epochs[late_start_idx:]
        
        late_logits_distances = logits_distances[late_start_idx:]
        stable_mask = late_logits_distances < self.logits_stability_threshold
        
        max_length = 0
        max_start = late_start_idx
        current_length = 0
        current_start = late_start_idx
        
        for i, is_stable in enumerate(stable_mask):
            if is_stable:
                if current_length == 0:
                    current_start = late_start_idx + i
                current_length += 1
            else:
                if current_length > max_length:
                    max_length = current_length
                    max_start = current_start
                current_length = 0
        
        if current_length > max_length:
            max_length = current_length
            max_start = current_start
        
        if max_length < 3:  
            return late_start_idx, len(epochs) - 1, epochs[late_start_idx:]
        
        stable_start = max_start
        stable_end = max_start + max_length - 1
        stable_epochs = epochs[stable_start:stable_end + 1]
        
        return stable_start, stable_end, stable_epochs
    
    def analyze_fidelity_in_stable_period(self, sample_data: Dict) -> Dict:
        """分析稳定期内的轨迹保真度"""
        positions = sample_data["positions"]
        features = sample_data["features"] 
        logits = sample_data["logits"]
        epochs = sample_data["epochs"]
        
        pos_distances = np.array([np.linalg.norm(positions[i] - positions[i-1]) 
                                 for i in range(1, len(positions))])
        feature_distances = np.array([np.linalg.norm(features[i] - features[i-1]) 
                                     for i in range(1, len(features))])
        
        if logits is not None:
            logits_distances = np.array([np.linalg.norm(logits[i] - logits[i-1]) 
                                        for i in range(1, len(logits))])
        else:
            logits_distances = np.zeros_like(pos_distances)
        
        stable_start, stable_end, stable_epochs = self.identify_stable_period(
            logits_distances, epochs
        )
        
        if stable_end <= stable_start + 1:
            return {
                'error': 'Insufficient stable period data',
                'stable_period_length': 0
            }
        
        stable_pos_distances = pos_distances[stable_start:stable_end]
        stable_feature_distances = feature_distances[stable_start:stable_end]
        stable_logits_distances = logits_distances[stable_start:stable_end]
        
        stable_pos_cosines, stable_feature_cosines, stable_logits_cosines = self._compute_cosine_similarities(
            positions[stable_start:stable_end+1], 
            features[stable_start:stable_end+1],
            logits[stable_start:stable_end+1] if logits is not None else None
        )
        
        results = {
            'stable_period': {
                'start_epoch': stable_epochs[0],
                'end_epoch': stable_epochs[-1],
                'length': len(stable_epochs),
                'start_idx': stable_start,
                'end_idx': stable_end
            },
            'distance_analysis': self._analyze_distance_fidelity(
                stable_pos_distances, stable_feature_distances, stable_logits_distances
            ),
            'cosine_analysis': self._analyze_cosine_fidelity(
                stable_pos_cosines, stable_feature_cosines, stable_logits_cosines
            ),
            'false_cases': self._detect_false_cases(
                stable_pos_distances, stable_feature_distances, 
                stable_pos_cosines, stable_feature_cosines,
                stable_epochs
            ),
            'raw_data': {
                'stable_epochs': stable_epochs,
                'pos_distances': stable_pos_distances.tolist(),
                'feature_distances': stable_feature_distances.tolist(),
                'logits_distances': stable_logits_distances.tolist(),
                'pos_cosines': stable_pos_cosines,
                'feature_cosines': stable_feature_cosines,
                'logits_cosines': stable_logits_cosines
            }
        }
        
        return results
    
    def _compute_cosine_similarities(self, positions, features, logits=None):
        def safe_cosine_sim(v1, v2):
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 < 1e-10 or n2 < 1e-10:
                return 0.0
            return np.dot(v1, v2) / (n1 * n2)
        
        pos_total_vec = positions[-1] - positions[0]
        feature_total_vec = features[-1] - features[0]
        
        pos_cosines = []
        feature_cosines = []
        logits_cosines = []
        
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
            else:
                logits_cosines.append(0.0)
        
        return pos_cosines, feature_cosines, logits_cosines
    
    def _analyze_distance_fidelity(self, pos_distances, feature_distances, logits_distances):
        """分析距离变化的保真度"""
        pos_feature_corr = pearsonr(pos_distances, feature_distances)[0] if len(pos_distances) > 1 else 0
        pos_logits_corr = pearsonr(pos_distances, logits_distances)[0] if len(pos_distances) > 1 else 0
        
        pos_feature_corr = 0.0 if np.isnan(pos_feature_corr) else pos_feature_corr
        pos_logits_corr = 0.0 if np.isnan(pos_logits_corr) else pos_logits_corr
        
        pos_rank = np.argsort(pos_distances)
        feature_rank = np.argsort(feature_distances)
        
        rank_consistency = spearmanr(pos_distances, feature_distances)[0] if len(pos_distances) > 1 else 0
        rank_consistency = 0.0 if np.isnan(rank_consistency) else rank_consistency
        
        return {
            'pos_feature_correlation': pos_feature_corr,
            'pos_logits_correlation': pos_logits_corr,
            'rank_consistency': rank_consistency,
            'mean_pos_distance': np.mean(pos_distances),
            'mean_feature_distance': np.mean(feature_distances),
            'std_pos_distance': np.std(pos_distances),
            'std_feature_distance': np.std(feature_distances)
        }
    
    def _analyze_cosine_fidelity(self, pos_cosines, feature_cosines, logits_cosines):
        """分析余弦相似度的保真度"""
        pos_cosines = np.array(pos_cosines)
        feature_cosines = np.array(feature_cosines)
        
        cosine_correlation = pearsonr(pos_cosines, feature_cosines)[0] if len(pos_cosines) > 1 else 0
        cosine_correlation = 0.0 if np.isnan(cosine_correlation) else cosine_correlation
        
        direction_agreement = np.mean((pos_cosines > 0) == (feature_cosines > 0))
        
        cosine_diff = np.abs(pos_cosines - feature_cosines)
        
        return {
            'cosine_correlation': cosine_correlation,
            'direction_agreement': direction_agreement,
            'mean_cosine_difference': np.mean(cosine_diff),
            'max_cosine_difference': np.max(cosine_diff),
            'mean_pos_cosine': np.mean(pos_cosines),
            'mean_feature_cosine': np.mean(feature_cosines)
        }
    
    def _detect_false_cases(self, pos_distances, feature_distances, pos_cosines, feature_cosines, epochs):
        """检测False Positive和False Negative案例"""
        
        high_change_threshold = np.percentile(feature_distances, 75)  # 高维空间高变化阈值
        low_change_threshold = np.percentile(feature_distances, 75)   # 高维空间低变化阈值
        
        high_2d_threshold = np.percentile(pos_distances, 75)  # 2D空间高变化阈值
        low_2d_threshold = np.percentile(pos_distances, 75)   # 2D空间低变化阈值
        
        false_positives = []  # 高维空间变化小，但2D空间变化大
        false_negatives = []  # 高维空间变化大，但2D空间变化小
        
        for i in range(len(pos_distances)):
            pos_dist = pos_distances[i]
            feature_dist = feature_distances[i]
            
            # False Positive: 高维变化小，2D变化大
            if feature_dist < low_change_threshold and pos_dist > high_2d_threshold:
                false_positives.append({
                    'epoch_transition': f"{epochs[i]}-{epochs[i+1]}",
                    'index': i,
                    'pos_distance': pos_dist,
                    'feature_distance': feature_dist,
                    'pos_cosine': pos_cosines[i],
                    'feature_cosine': feature_cosines[i],
                    'severity': (pos_dist - feature_dist) / feature_dist if feature_dist > 0 else float('inf')
                })
            
            # False Negative: 高维变化大，2D变化小  
            if feature_dist > high_change_threshold and pos_dist < low_2d_threshold:
                false_negatives.append({
                    'epoch_transition': f"{epochs[i]}-{epochs[i+1]}",
                    'index': i,
                    'pos_distance': pos_dist,
                    'feature_distance': feature_dist,
                    'pos_cosine': pos_cosines[i],
                    'feature_cosine': feature_cosines[i],
                    'severity': (feature_dist - pos_dist) / pos_dist if pos_dist > 0 else float('inf')
                })
        
        return {
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'fp_count': len(false_positives),
            'fn_count': len(false_negatives),
            'total_transitions': len(pos_distances)
        }

def load_and_analyze_samples(individual_samples_dir: str, analyzer: TrajectoryFidelityAnalyzer) -> Dict:
    """加载并分析所有样本"""
    
    summary_file = os.path.join(individual_samples_dir, "samples_summary.csv")
    if not os.path.exists(summary_file):
        raise FileNotFoundError(f"未找到样本摘要文件: {summary_file}")
    
    summary_df = pd.read_csv(summary_file)
    
    results = {}
    all_fidelity_scores = []
    all_false_cases = {'fp': [], 'fn': []}
    
    print(f"开始分析 {len(summary_df)} 个样本的轨迹保真度...")
    
    for idx, row in summary_df.iterrows():
        sample_id = row['sample_id']
        print(f"分析样本 {sample_id} ({idx+1}/{len(summary_df)})")
        
        try:
            sample_data = load_sample_data(individual_samples_dir, sample_id)
            
            fidelity_result = analyzer.analyze_fidelity_in_stable_period(sample_data)
            
            if 'error' not in fidelity_result:
                results[sample_id] = fidelity_result
                
                distance_analysis = fidelity_result['distance_analysis']
                cosine_analysis = fidelity_result['cosine_analysis']
                false_cases = fidelity_result['false_cases']
                
                fidelity_score = {
                    'sample_id': sample_id,
                    'stable_period_length': fidelity_result['stable_period']['length'],
                    'distance_correlation': distance_analysis['pos_feature_correlation'],
                    'cosine_correlation': cosine_analysis['cosine_correlation'],
                    'direction_agreement': cosine_analysis['direction_agreement'],
                    'fp_rate': false_cases['fp_count'] / false_cases['total_transitions'] if false_cases['total_transitions'] > 0 else 0,
                    'fn_rate': false_cases['fn_count'] / false_cases['total_transitions'] if false_cases['total_transitions'] > 0 else 0,
                }
                
                all_fidelity_scores.append(fidelity_score)
                
                all_false_cases['fp'].extend([(sample_id, case) for case in false_cases['false_positives']])
                all_false_cases['fn'].extend([(sample_id, case) for case in false_cases['false_negatives']])
            
            else:
                print(f"样本 {sample_id} 分析失败: {fidelity_result['error']}")
                
        except Exception as e:
            print(f"样本 {sample_id} 加载失败: {e}")
            continue
    
    return {
        'individual_results': results,
        'overall_statistics': all_fidelity_scores,
        'false_cases': all_false_cases
    }

def load_sample_data(individual_samples_dir: str, sample_id: int) -> Dict:
    """加载单个样本的数据"""
    sample_dir = os.path.join(individual_samples_dir, f"sample_{sample_id}")
    
    np_file = os.path.join(sample_dir, f"sample_{sample_id}_raw_data.npz")
    data = np.load(np_file, allow_pickle=True)
    
    return {
        'positions': data['positions'],
        'features': data['features'], 
        'logits': data['logits'] if 'logits' in data else None,
        'epochs': data['epochs'].tolist(),
        'sample_id': sample_id
    }

def generate_comprehensive_report(analysis_results: Dict, save_dir: str):
    """生成综合分析报告"""
    
    overall_stats = analysis_results['overall_statistics']
    false_cases = analysis_results['false_cases']
    
    if not overall_stats:
        print("没有有效的分析结果")
        return
    
    stats_df = pd.DataFrame(overall_stats)
    
    print("\n" + "="*80)
    print("轨迹保真度分析报告 - 训练后期稳定阶段")
    print("="*80)
    
    print(f"\n总体统计 (基于 {len(stats_df)} 个样本):")
    print(f"平均稳定期长度: {stats_df['stable_period_length'].mean():.2f} epochs")
    print(f"距离相关性 (2D vs 高维): {stats_df['distance_correlation'].mean():.4f} ± {stats_df['distance_correlation'].std():.4f}")
    print(f"余弦相似度相关性: {stats_df['cosine_correlation'].mean():.4f} ± {stats_df['cosine_correlation'].std():.4f}")
    print(f"方向一致性: {stats_df['direction_agreement'].mean():.4f} ± {stats_df['direction_agreement'].std():.4f}")
    print(f"False Positive 率: {stats_df['fp_rate'].mean():.4f} ± {stats_df['fp_rate'].std():.4f}")
    print(f"False Negative 率: {stats_df['fn_rate'].mean():.4f} ± {stats_df['fn_rate'].std():.4f}")
    
    stats_file = os.path.join(save_dir, "fidelity_statistics.csv")
    stats_df.to_csv(stats_file, index=False)
    print(f"\n详细统计已保存: {stats_file}")
    
    print(f"\nFalse Cases 分析:")
    print(f"Total False Positives: {len(false_cases['fp'])}")
    print(f"Total False Negatives: {len(false_cases['fn'])}")
    
    if false_cases['fp']:
        fp_data = []
        for sample_id, case in false_cases['fp']:
            case['sample_id'] = sample_id
            fp_data.append(case)
        
        fp_df = pd.DataFrame(fp_data)
        fp_file = os.path.join(save_dir, "false_positives.csv")
        fp_df.to_csv(fp_file, index=False)
        print(f"False Positives 详情已保存: {fp_file}")
        
        top_fp = fp_df.nlargest(5, 'severity')
        print(f"\n最严重的 False Positive 案例:")
        for _, row in top_fp.iterrows():
            print(f"  样本 {row['sample_id']}, {row['epoch_transition']}: "
                  f"2D变化={row['pos_distance']:.4f}, 高维变化={row['feature_distance']:.4f}")
    
    if false_cases['fn']:
        fn_data = []
        for sample_id, case in false_cases['fn']:
            case['sample_id'] = sample_id
            fn_data.append(case)
        
        fn_df = pd.DataFrame(fn_data)
        fn_file = os.path.join(save_dir, "false_negatives.csv")
        fn_df.to_csv(fn_file, index=False)
        print(f"False Negatives 详情已保存: {fn_file}")
        
        top_fn = fn_df.nlargest(5, 'severity')
        print(f"\n最严重的 False Negative 案例:")
        for _, row in top_fn.iterrows():
            print(f"  样本 {row['sample_id']}, {row['epoch_transition']}: "
                  f"2D变化={row['pos_distance']:.4f}, 高维变化={row['feature_distance']:.4f}")
    
    create_fidelity_visualizations(stats_df, false_cases, save_dir)
    
    results_file = os.path.join(save_dir, "complete_analysis_results.json")
    
    def convert_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.int64):
            return int(obj)
        elif isinstance(obj, np.float64):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: convert_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(item) for item in obj]
        else:
            return obj
    
    json_data = convert_for_json(analysis_results)
    
    with open(results_file, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    print(f"\n完整分析结果已保存: {results_file}")

def create_fidelity_visualizations(stats_df: pd.DataFrame, false_cases: Dict, save_dir: str):
    """创建保真度分析可视化"""
    
    # 1. 整体保真度分布
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Trajectory Fidelity Analysis - Late Training Stable Period', fontsize=16)
    
    # 距离相关性分布
    axes[0, 0].hist(stats_df['distance_correlation'], bins=20, alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(stats_df['distance_correlation'].mean(), color='red', linestyle='--', 
                       label=f'Mean: {stats_df["distance_correlation"].mean():.3f}')
    axes[0, 0].set_title('Distance Correlation (2D vs High-D)')
    axes[0, 0].set_xlabel('Correlation')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 余弦相似度相关性分布
    axes[0, 1].hist(stats_df['cosine_correlation'], bins=20, alpha=0.7, edgecolor='black', color='orange')
    axes[0, 1].axvline(stats_df['cosine_correlation'].mean(), color='red', linestyle='--',
                       label=f'Mean: {stats_df["cosine_correlation"].mean():.3f}')
    axes[0, 1].set_title('Cosine Similarity Correlation')
    axes[0, 1].set_xlabel('Correlation')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 方向一致性分布
    axes[0, 2].hist(stats_df['direction_agreement'], bins=20, alpha=0.7, edgecolor='black', color='green')
    axes[0, 2].axvline(stats_df['direction_agreement'].mean(), color='red', linestyle='--',
                       label=f'Mean: {stats_df["direction_agreement"].mean():.3f}')
    axes[0, 2].set_title('Direction Agreement Rate')
    axes[0, 2].set_xlabel('Agreement Rate')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # False Positive率分布
    axes[1, 0].hist(stats_df['fp_rate'], bins=20, alpha=0.7, edgecolor='black', color='red')
    axes[1, 0].axvline(stats_df['fp_rate'].mean(), color='darkred', linestyle='--',
                       label=f'Mean: {stats_df["fp_rate"].mean():.3f}')
    axes[1, 0].set_title('False Positive Rate')
    axes[1, 0].set_xlabel('FP Rate')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # False Negative率分布
    axes[1, 1].hist(stats_df['fn_rate'], bins=20, alpha=0.7, edgecolor='black', color='purple')
    axes[1, 1].axvline(stats_df['fn_rate'].mean(), color='darkviolet', linestyle='--',
                       label=f'Mean: {stats_df["fn_rate"].mean():.3f}')
    axes[1, 1].set_title('False Negative Rate')
    axes[1, 1].set_xlabel('FN Rate')  
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 稳定期长度分布
    axes[1, 2].hist(stats_df['stable_period_length'], bins=20, alpha=0.7, edgecolor='black', color='cyan')
    axes[1, 2].axvline(stats_df['stable_period_length'].mean(), color='darkblue', linestyle='--',
                       label=f'Mean: {stats_df["stable_period_length"].mean():.1f}')
    axes[1, 2].set_title('Stable Period Length')
    axes[1, 2].set_xlabel('Length (epochs)')
    axes[1, 2].set_ylabel('Frequency')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    fidelity_viz_file = os.path.join(save_dir, "fidelity_analysis_overview.png")
    plt.savefig(fidelity_viz_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 相关性散点图
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 距离相关性 vs 余弦相关性
    axes[0].scatter(stats_df['distance_correlation'], stats_df['cosine_correlation'], alpha=0.6)
    axes[0].set_xlabel('Distance Correlation')
    axes[0].set_ylabel('Cosine Correlation')
    axes[0].set_title('Distance vs Cosine Correlation')
    axes[0].grid(True, alpha=0.3)
    
    # 添加对角线
    lims = [
        np.min([axes[0].get_xlim(), axes[0].get_ylim()]),
        np.max([axes[0].get_xlim(), axes[0].get_ylim()]),
    ]
    axes[0].plot(lims, lims, 'k--', alpha=0.5, zorder=0)
    
    # False率对比
    axes[1].scatter(stats_df['fp_rate'], stats_df['fn_rate'], alpha=0.6, color='red')
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('False Negative Rate')
    axes[1].set_title('False Positive vs False Negative Rate')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    correlation_viz_file = os.path.join(save_dir, "correlation_analysis.png")
    plt.savefig(correlation_viz_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"可视化结果已保存:")
    print(f"  - 整体分析: {fidelity_viz_file}")
    print(f"  - 相关性分析: {correlation_viz_file}")

def create_example_visualizations(analysis_results: Dict, save_dir: str, n_examples: int = 3):
    """为最佳和最差的案例以及False cases创建示例可视化"""
    
    stats_df = pd.DataFrame(analysis_results['overall_statistics'])
    
    if len(stats_df) == 0:
        return
    
    # 选择最佳和最差的样本
    stats_df['overall_fidelity'] = (stats_df['distance_correlation'] + 
                                   stats_df['cosine_correlation'] + 
                                   stats_df['direction_agreement']) / 3
    
    best_samples = stats_df.nlargest(n_examples, 'overall_fidelity')
    worst_samples = stats_df.nsmallest(n_examples, 'overall_fidelity')
    
    # 创建示例可视化目录
    examples_dir = os.path.join(save_dir, "fidelity_examples")
    os.makedirs(examples_dir, exist_ok=True)
    
    print(f"\n生成示例可视化:")
    print(f"最佳保真度样本: {best_samples['sample_id'].tolist()}")
    print(f"最差保真度样本: {worst_samples['sample_id'].tolist()}")
    
    # 为每个示例样本创建详细的可视化会需要访问原始数据
    # 这里先创建一个摘要文件
    examples_summary = {
        'best_samples': best_samples.to_dict('records'),
        'worst_samples': worst_samples.to_dict('records'),
        'analysis_summary': {
            'total_samples': len(stats_df),
            'mean_fidelity': stats_df['overall_fidelity'].mean(),
            'std_fidelity': stats_df['overall_fidelity'].std()
        }
    }
    
    summary_file = os.path.join(examples_dir, "examples_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(examples_summary, f, indent=2)
    
    print(f"示例摘要已保存: {summary_file}")

def main_fidelity_analysis(individual_samples_dir: str, save_dir: str = None, late_ratio: float = 0.7):
    """主分析函数"""
    
    if save_dir is None:
        save_dir = os.path.join(os.path.dirname(individual_samples_dir), "fidelity_analysis")
    
    os.makedirs(save_dir, exist_ok=True)
    
    print("开始轨迹保真度分析...")
    print(f"输入目录: {individual_samples_dir}")
    print(f"输出目录: {save_dir}")
    
    # 创建分析器
    analyzer = TrajectoryFidelityAnalyzer(
        logits_stability_threshold=0.1,  # 可根据需要调整
        late_training_ratio=late_ratio  # 分析后30%的训练阶段
    )
    
    try:
        # 分析所有样本
        analysis_results = load_and_analyze_samples(individual_samples_dir, analyzer)
        
        # 生成综合报告
        generate_comprehensive_report(analysis_results, save_dir)
        
        # 创建示例可视化
        create_example_visualizations(analysis_results, save_dir)
        
        print(f"\n轨迹保真度分析完成！结果已保存到: {save_dir}")
        
        return analysis_results
        
    except Exception as e:
        print(f"分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    dataset = "cifar10"
    n_samples = 200
    late_ratio = 0.7 # 分析后30%的训练阶段

    # base_dir 是 trajectory_metric_analyse.py 分析结果保存的目录
    base_dir = f"/home/zicong/DynaVis/result/{dataset}/{n_samples}"
    individual_samples_dir = os.path.join(base_dir, "individual_samples")
    
    save_dir = os.path.join(base_dir, f"fidelity_analysis_{late_ratio}")
    os.makedirs(save_dir, exist_ok=True)
    results = main_fidelity_analysis(individual_samples_dir, save_dir, late_ratio)
    
    if results:
        print("\n分析完成！查看以下文件获取详细结果:")
        print(f"- 统计摘要: {save_dir}/fidelity_statistics.csv")
        print(f"- False Positives: {save_dir}/false_positives.csv")
        print(f"- False Negatives: {save_dir}/false_negatives.csv")
        print(f"- 完整结果: {save_dir}/complete_analysis_results.json")
        print(f"- 可视化图表: {save_dir}/fidelity_analysis_overview.png")