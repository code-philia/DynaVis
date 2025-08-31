import numpy as np
import pandas as pd
import os
from typing import Dict, List

# ---- 单位向量 ----
def _unit_vec(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-12:
        return np.zeros_like(v)
    return v / n

# ---- 1) 片段投影分数（全局方向 & 全局距离归一化）----
def segment_projection_scores(traj: np.ndarray) -> Dict[str, np.ndarray]:
    """
    输入：traj [T, D]，一条轨迹（按 epoch 排序）
    步骤：
      1) 全局方向 g = (x_T - x_1) / ||x_T - x_1||
      2) 相邻片段 Δx_t = x_{t+1} - x_t 的投影 s_t = <Δx_t, g>
      3) 用全局距离 L = ||x_T - x_1|| 归一化
         - 有符号: s_signed_norm[t] = s_t / L
         - 绝对值: s_abs_norm[t]   = |s_t| / L
    返回：
      {"s_signed_norm": [T-1], "s_abs_norm": [T-1], "L": float, "g": [D]}
    """
    T = traj.shape[0]
    if T < 2:
        return {
            "s_signed_norm": np.array([]),
            "s_abs_norm": np.array([]),
            "L": 0.0,
            "g": np.zeros(traj.shape[1], dtype=traj.dtype),
        }
    d_global = traj[-1] - traj[0]
    L = float(np.linalg.norm(d_global))
    g = _unit_vec(d_global)

    deltas = np.diff(traj, axis=0)                   # [T-1, D]
    s = deltas @ g                                   # [T-1] 与全局方向的内积
    if L < 1e-12:
        s_signed_norm = np.zeros(T-1, dtype=traj.dtype)
        s_abs_norm = np.zeros(T-1, dtype=traj.dtype)
    else:
        s_signed_norm = s / L
        s_abs_norm = np.abs(s) / L

    return {"s_signed_norm": s_signed_norm, "s_abs_norm": s_abs_norm, "L": L, "g": g}

# ---- 2) 关键运动掩码 ----
def key_movement_mask(traj: np.ndarray, threshold: float = 0.2, use_abs: bool = True) -> np.ndarray:
    """
    返回长度为 T-1 的布尔掩码，标记哪些相邻 epoch 间隔属于"关键运动"。
    - use_abs=True：使用 |投影|/L（仅关注强度，默认）
    - use_abs=False：使用 有符号投影/L（区分方向，通常更严格）
    """
    scores = segment_projection_scores(traj)
    seq = scores["s_abs_norm"] if use_abs else scores["s_signed_norm"]
    if seq.size == 0:
        return np.array([], dtype=bool)
    return seq >= threshold

# ---- 3) 单样本高低维 FP/FN 评估 ----
def compare_high_low_fpfn(
    high_traj: np.ndarray,
    low_traj: np.ndarray,
    threshold: float = 0.2,
    use_abs: bool = True
) -> Dict[str, object]:
    """
    以"高维"为 GT（ground truth）、"低维"为预测，对单条轨迹进行 FP/FN 评估。
    要求 high_traj 与 low_traj 时间长度一致（同一组 epochs）。
    返回：
      - 计数: TP/FP/FN/TN
      - 指标: precision/recall/f1/accuracy
      - 掩码: gt_mask/pred_mask (长度 T-1)
      - 索引: idx_tp/idx_fp/idx_fn/idx_tn（对应片段的起点索引 0..T-2）
    """
    assert high_traj.shape[0] == low_traj.shape[0], "高低维轨迹的时间长度(epochs)必须一致"
    gt = key_movement_mask(high_traj, threshold=threshold, use_abs=use_abs)
    pr = key_movement_mask(low_traj,  threshold=threshold, use_abs=use_abs)

    TP = int(np.sum((pr == 1) & (gt == 1)))
    TN = int(np.sum((pr == 0) & (gt == 0)))
    FP = int(np.sum((pr == 1) & (gt == 0)))
    FN = int(np.sum((pr == 0) & (gt == 1)))

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    acc       = (TP + TN) / gt.size if gt.size > 0 else 0.0

    return {
        "TP": TP, "FP": FP, "FN": FN, "TN": TN,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(acc),
        "gt_mask": gt,
        "pred_mask": pr,
        "idx_tp": np.where((pr == 1) & (gt == 1))[0],
        "idx_fp": np.where((pr == 1) & (gt == 0))[0],
        "idx_fn": np.where((pr == 0) & (gt == 1))[0],
        "idx_tn": np.where((pr == 0) & (gt == 0))[0],
    }

# ---- 4) 批量评估（保留原本列表输入格式）----
def analyze_fpfn_batch(
    positions_list: List[np.ndarray],
    logits_list: List[np.ndarray],
    threshold: float = 0.2,
    use_abs: bool = True
) -> Dict[str, object]:
    """
    批量评估：以 logits_list(高维) 为 GT，positions_list(低维) 为预测。
    输入：
      - positions_list: N 个 [T, 2] 低维轨迹
      - logits_list:    N 个 [T, D] 高维轨迹
    返回：
      - 总计数: total_TP/FP/FN/TN
      - 宏/微指标: macro_* / micro_*
      - per_sample: 每个样本 compare_high_low_fpfn(...) 的完整结果，便于可视化与误差分析
    """
    assert len(positions_list) == len(logits_list), "positions_list 与 logits_list 数量必须一致"

    total_TP = total_FP = total_FN = total_TN = 0
    macro_prec, macro_rec, macro_f1, macro_acc = [], [], [], []
    per_sample = []

    for pos, logit in zip(positions_list, logits_list):
        res = compare_high_low_fpfn(
            high_traj=logit,
            low_traj=pos,
            threshold=threshold,
            use_abs=use_abs
        )
        per_sample.append(res)
        total_TP += res["TP"]; total_FP += res["FP"]
        total_FN += res["FN"]; total_TN += res["TN"]
        macro_prec.append(res["precision"])
        macro_rec.append(res["recall"])
        macro_f1.append(res["f1"])
        macro_acc.append(res["accuracy"])

    micro_precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0.0
    micro_recall    = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0.0
    micro_f1        = (2 * micro_precision * micro_recall / (micro_precision + micro_recall)) if (micro_precision + micro_recall) > 0 else 0.0
    micro_accuracy  = (total_TP + total_TN) / (total_TP + total_FP + total_FN + total_TN) if (total_TP + total_FP + total_FN + total_TN) > 0 else 0.0

    return {
        "total_TP": int(total_TP), "total_FP": int(total_FP),
        "total_FN": int(total_FN), "total_TN": int(total_TN),
        "macro_precision": float(np.mean(macro_prec) if macro_prec else 0.0),
        "macro_recall": float(np.mean(macro_rec) if macro_rec else 0.0),
        "macro_f1": float(np.mean(macro_f1) if macro_f1 else 0.0),
        "macro_accuracy": float(np.mean(macro_acc) if macro_acc else 0.0),
        "micro_precision": float(micro_precision),
        "micro_recall": float(micro_recall),
        "micro_f1": float(micro_f1),
        "micro_accuracy": float(micro_accuracy),
        "per_sample": per_sample,
    }

# ---- 新增：数据加载和分析函数（支持 late_training_ratio）----
def load_sample_data(individual_samples_dir: str, sample_id: int) -> Dict:
    """加载单个样本的数据"""
    sample_dir = os.path.join(individual_samples_dir, f"sample_{sample_id}")
    
    # 加载numpy数据
    np_file = os.path.join(sample_dir, f"sample_{sample_id}_raw_data.npz")
    data = np.load(np_file, allow_pickle=True)
    
    return {
        'positions': data['positions'],
        'features': data['features'], 
        'logits': data['logits'] if 'logits' in data else None,
        'epochs': data['epochs'].tolist(),
        'sample_id': sample_id
    }

def extract_late_training_period(sample_data: Dict, late_training_ratio: float = 0.7) -> Dict:
    """
    根据 late_training_ratio 提取训练后期的数据
    
    Args:
        sample_data: 样本数据字典
        late_training_ratio: 训练后期比例 (0.7表示后30%的epochs)
    
    Returns:
        提取后期数据的样本字典
    """
    epochs = sample_data['epochs']
    positions = sample_data['positions']
    logits = sample_data['logits']
    
    # 计算训练后期的起始索引
    late_start_idx = int(len(epochs) * late_training_ratio)
    
    # 确保至少有2个数据点用于分析
    if late_start_idx >= len(epochs) - 1:
        late_start_idx = max(0, len(epochs) - 2)
    
    # 提取后期数据
    late_epochs = epochs[late_start_idx:]
    late_positions = positions[late_start_idx:]
    late_logits = logits[late_start_idx:] if logits is not None else None
    
    return {
        'positions': late_positions,
        'logits': late_logits,
        'epochs': late_epochs,
        'sample_id': sample_data['sample_id'],
        'late_start_idx': late_start_idx,
        'original_length': len(epochs),
        'late_length': len(late_epochs)
    }

def load_all_samples_for_fpfn_analysis(individual_samples_dir: str, late_training_ratio: float = 0.7) -> tuple:
    """
    加载所有样本数据，返回 positions_list 和 logits_list
    
    Args:
        individual_samples_dir: 样本数据目录
        late_training_ratio: 训练后期比例 (0.7表示后30%的epochs)
    
    Returns:
        (positions_list, logits_list, sample_ids, late_period_info)
    """
    # 读取样本摘要
    summary_file = os.path.join(individual_samples_dir, "samples_summary.csv")
    if not os.path.exists(summary_file):
        raise FileNotFoundError(f"未找到样本摘要文件: {summary_file}")
    
    summary_df = pd.read_csv(summary_file)
    
    positions_list = []
    logits_list = []
    sample_ids = []
    late_period_info = []
    
    print(f"开始加载 {len(summary_df)} 个样本的数据...")
    print(f"使用训练后期比例: {late_training_ratio} (分析后 {(1-late_training_ratio)*100:.1f}% 的epochs)")
    
    for idx, row in summary_df.iterrows():
        sample_id = row['sample_id']
        print(f"加载样本 {sample_id} ({idx+1}/{len(summary_df)})")
        
        try:
            # 加载完整样本数据
            sample_data = load_sample_data(individual_samples_dir, sample_id)
            
            # 提取训练后期数据
            late_data = extract_late_training_period(sample_data, late_training_ratio)
            
            positions = late_data['positions']  # [T_late, 2]
            logits = late_data['logits']        # [T_late, D]
            
            if logits is not None and len(positions) == len(logits) and len(positions) >= 2:
                positions_list.append(positions)
                logits_list.append(logits)
                sample_ids.append(sample_id)
                
                # 记录后期数据信息
                late_period_info.append({
                    'sample_id': sample_id,
                    'late_start_idx': late_data['late_start_idx'],
                    'original_length': late_data['original_length'],
                    'late_length': late_data['late_length'],
                    'late_start_epoch': late_data['epochs'][0],
                    'late_end_epoch': late_data['epochs'][-1],
                    'late_epochs_range': f"{late_data['epochs'][0]}-{late_data['epochs'][-1]}"
                })
                
                print(f"  -> 原始长度: {late_data['original_length']}, "
                      f"后期长度: {late_data['late_length']}, "
                      f"epochs: {late_data['epochs'][0]}-{late_data['epochs'][-1]}")
            else:
                print(f"样本 {sample_id} 数据不完整或后期数据不足，跳过")
                
        except Exception as e:
            print(f"样本 {sample_id} 加载失败: {e}")
            continue
    
    print(f"成功加载 {len(positions_list)} 个样本的后期数据")
    return positions_list, logits_list, sample_ids, late_period_info

def analyze_samples_fpfn(
    individual_samples_dir: str, 
    threshold: float = 0.2, 
    use_abs: bool = True, 
    late_training_ratio: float = 0.7,
    save_dir: str = None
) -> Dict:
    """
    分析指定目录下所有样本的 FP/FN（支持训练后期数据提取）
    
    Args:
        individual_samples_dir: 样本数据目录
        threshold: 关键运动阈值
        use_abs: 是否使用绝对值
        late_training_ratio: 训练后期比例 (0.7表示后30%的epochs)
        save_dir: 保存结果的目录（可选）
    
    Returns:
        分析结果字典
    """
    print("="*60)
    print("开始 False Positive / False Negative 分析")
    print(f"输入目录: {individual_samples_dir}")
    print(f"阈值: {threshold}, 使用绝对值: {use_abs}")
    print(f"训练后期比例: {late_training_ratio} (分析后 {(1-late_training_ratio)*100:.1f}% 的epochs)")
    print("="*60)
    
    # 加载数据（自动提取训练后期）
    positions_list, logits_list, sample_ids, late_period_info = load_all_samples_for_fpfn_analysis(
        individual_samples_dir, late_training_ratio
    )
    
    if len(positions_list) == 0:
        print("没有有效的样本数据！")
        return {}
    
    # 执行分析
    print(f"\n开始分析 {len(positions_list)} 个样本的后期轨迹...")
    summary = analyze_fpfn_batch(
        positions_list,   # 低维轨迹列表 [N][T_late,2]
        logits_list,      # 高维轨迹列表 [N][T_late,D]
        threshold=threshold,
        use_abs=use_abs
    )
    
    # 添加样本ID和后期数据信息
    summary['sample_ids'] = sample_ids
    summary['n_samples'] = len(sample_ids)
    summary['late_training_ratio'] = late_training_ratio
    summary['late_period_info'] = late_period_info
    
    # 打印结果
    print("\n" + "="*60)
    print("FP/FN 分析结果 (训练后期)")
    print("="*60)
    print(f"总样本数: {summary['n_samples']}")
    print(f"分析的训练后期比例: {late_training_ratio} (后 {(1-late_training_ratio)*100:.1f}% epochs)")
    
    # 打印后期数据统计
    late_lengths = [info['late_length'] for info in late_period_info]
    print(f"后期数据长度统计: 平均 {np.mean(late_lengths):.1f} ± {np.std(late_lengths):.1f} epochs")
    print(f"后期数据长度范围: [{min(late_lengths)}, {max(late_lengths)}]")
    
    print(f"\n总计数 - TP: {summary['total_TP']}, FP: {summary['total_FP']}, FN: {summary['total_FN']}, TN: {summary['total_TN']}")
    print(f"微观指标 - Precision: {summary['micro_precision']:.4f}, Recall: {summary['micro_recall']:.4f}, F1: {summary['micro_f1']:.4f}")
    print(f"宏观指标 - Precision: {summary['macro_precision']:.4f}, Recall: {summary['macro_recall']:.4f}, F1: {summary['macro_f1']:.4f}")
    
    # 分析 per-sample 细节
    fp_counts = [sample['FP'] for sample in summary['per_sample']]
    fn_counts = [sample['FN'] for sample in summary['per_sample']]
    
    print(f"\nPer-sample 统计 (基于后期数据):")
    print(f"平均 FP: {np.mean(fp_counts):.2f} ± {np.std(fp_counts):.2f}")
    print(f"平均 FN: {np.mean(fn_counts):.2f} ± {np.std(fn_counts):.2f}")
    print(f"FP 范围: [{min(fp_counts)}, {max(fp_counts)}]")
    print(f"FN 范围: [{min(fn_counts)}, {max(fn_counts)}]")
    
    # 找出 FP/FN 最多的样本
    max_fp_idx = np.argmax(fp_counts)
    max_fn_idx = np.argmax(fn_counts)
    
    print(f"\nFP 最多的样本: {sample_ids[max_fp_idx]} (FP={fp_counts[max_fp_idx]})")
    print(f"FN 最多的样本: {sample_ids[max_fn_idx]} (FN={fn_counts[max_fn_idx]})")
    
    # 保存结果
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存摘要统计
        import json
        summary_for_save = {k: v for k, v in summary.items() if k != 'per_sample'}
        summary_file = os.path.join(save_dir, f"fpfn_summary_late_{late_training_ratio}.json")
        with open(summary_file, 'w') as f:
            json.dump(summary_for_save, f, indent=2)
        
        # 保存详细的 per-sample 结果
        detailed_results = []
        for i, (sample_id, sample_result, late_info) in enumerate(zip(sample_ids, summary['per_sample'], late_period_info)):
            detailed_results.append({
                'sample_id': sample_id,
                'TP': sample_result['TP'],
                'FP': sample_result['FP'],
                'FN': sample_result['FN'],
                'TN': sample_result['TN'],
                'precision': sample_result['precision'],
                'recall': sample_result['recall'],
                'f1': sample_result['f1'],
                'accuracy': sample_result['accuracy'],
                'fp_indices': sample_result['idx_fp'].tolist(),
                'fn_indices': sample_result['idx_fn'].tolist(),
                'late_period_info': late_info
            })
        
        detailed_file = os.path.join(save_dir, f"fpfn_detailed_results_late_{late_training_ratio}.json")
        with open(detailed_file, 'w') as f:
            json.dump(detailed_results, f, indent=2)
        
        # 保存后期数据统计
        late_stats_df = pd.DataFrame(late_period_info)
        late_stats_file = os.path.join(save_dir, f"late_period_statistics_{late_training_ratio}.csv")
        late_stats_df.to_csv(late_stats_file, index=False)
        
        print(f"\n结果已保存到: {save_dir}")
        print(f"- 摘要: {summary_file}")
        print(f"- 详细结果: {detailed_file}")
        print(f"- 后期数据统计: {late_stats_file}")
    
    return summary

if __name__ == "__main__":
    dataset = "cifar10"
    n_samples = 10
    late_ratio = 0.7  # 分析后30%的训练阶段
    
    base_dir = f"/home/zicong/DynaVis/result/dynavis_result/{dataset}/{n_samples}"
    individual_samples_dir = os.path.join(base_dir, "individual_samples")
    
    save_dir = os.path.join(base_dir, f"fpfn_analysis_late_{late_ratio}")
    
    if os.path.exists(individual_samples_dir):
        results = analyze_samples_fpfn(
            individual_samples_dir=individual_samples_dir,
            threshold=0.1,      # 关键运动阈值，可调整
            use_abs=True,       
            late_training_ratio=late_ratio,
            save_dir=save_dir
        )
        
        if results:
            print("\n分析完成！")
            print(f"分析了 {results['n_samples']} 个样本的后期轨迹")
            print(f"训练后期比例: {late_ratio} (后 {(1-late_ratio)*100:.1f}% epochs)")
            
    else:
        print(f"数据目录不存在: {individual_samples_dir}")
        print("请先运行 analyse.py 生成样本数据")