#!/usr/bin/env python3
"""
Dynavis Simple Evaluation
=========================
给定：
  • 高维序列 X_hd:  [T, N, D]
  • 低维序列 Y_ld:  [T, N, 2]
计算：
  1) 每个 epoch 的 Neighbor Preserving Rate (NPR)
  2) 跨时间的运动一致性（逐样本）：
     - 距离序列的 Pearson/Spearman（低维 vs 高维）
     - 与总位移的余弦一致性（低维 vs 高维）
     - 方向一致率（sign 一致）
输出：CSV/JSON 到 --save_dir。

用法示例：
  python debug3.py --hd_file cifar10_50epochs_500sample.npy \
  --ld_file checkpoints/Y_ld.npz --save_dir checkpoints/out_eval \
  --k 15 --hd_metric euclidean --ld_metric euclidean --t_start 35 --t_end 50

"""
from __future__ import annotations
import argparse
import json
import os
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from scipy.stats import pearsonr, spearmanr
from sklearn.neighbors import NearestNeighbors

# ----------------------------- I/O -----------------------------

def _load_array(file: str, pref_keys: List[str]) -> np.ndarray:
    if file.endswith('.npy'):
        arr = np.load(file)
        return np.asarray(arr)
    elif file.endswith('.npz'):
        data = np.load(file)
        for k in pref_keys:
            if k in data: return np.asarray(data[k])
        # fallback: pick the first array
        if len(data.files) == 0:
            raise ValueError(f"No arrays in npz: {file}")
        return np.asarray(data[data.files[0]])
    else:
        raise ValueError(f"Unsupported file type: {file}")


def load_inputs(hd_file: str, ld_file: str) -> Tuple[np.ndarray, np.ndarray]:
    X_hd = _load_array(hd_file, pref_keys=['X_hd','X','hd'])
    Y_ld = _load_array(ld_file, pref_keys=['Y_ld','Y','ld'])
    if X_hd.ndim != 3 or Y_ld.ndim != 3:
        raise ValueError(f"Expect X_hd [T,N,D] and Y_ld [T,N,2], got {X_hd.shape} and {Y_ld.shape}")
    T1,N1,_ = X_hd.shape
    T2,N2,d2 = Y_ld.shape
    if d2 != 2:
        raise ValueError(f"Y_ld last dim must be 2, got {d2}")
    if T1 != T2 or N1 != N2:
        # 对齐到共同最小 T/N（保守裁剪）
        T = min(T1,T2)
        N = min(N1,N2)
        X_hd = X_hd[:T,:N,:]
        Y_ld = Y_ld[:T,:N,:]
        print(f"[Warn] Shapes mismatched; cropped to X_hd {X_hd.shape}, Y_ld {Y_ld.shape}")
    return X_hd.astype(np.float64), Y_ld.astype(np.float64)

# ----------------------------- NPR -----------------------------

def _knn_indices(X: np.ndarray, k: int, metric: str = 'euclidean') -> np.ndarray:
    """Return neighbor indices (exclude self). X: [N, dim]."""
    X2 = X.reshape(X.shape[0], -1)
    k_eff = min(k+1, X2.shape[0])
    if k_eff <= 1:
        return np.zeros((X2.shape[0], 0), dtype=int)
    nn = NearestNeighbors(n_neighbors=k_eff, metric=metric, n_jobs=-1)
    nn.fit(X2)
    ind = nn.kneighbors(return_distance=False)
    return ind[:,1:k_eff]

from scipy.sparse.csgraph import shortest_path
from sklearn.neighbors import NearestNeighbors, kneighbors_graph

def _knn_indices_euclid(X: np.ndarray, k: int, metric: str = 'euclidean') -> np.ndarray:
    """
    标准欧氏/余弦等度量下的 kNN 索引（不含自身）。
    X: [N, dim]
    """
    X2 = X.reshape(X.shape[0], -1)
    k_eff = min(k + 1, X2.shape[0])   # +1 包含自身，稍后去掉
    if k_eff <= 1:
        return np.zeros((X2.shape[0], 0), dtype=int)
    nn = NearestNeighbors(n_neighbors=k_eff, metric=metric, n_jobs=-1)
    nn.fit(X2)
    ind = nn.kneighbors(return_distance=False)
    return ind[:, 1:k_eff]  # 去自身

def _knn_indices_manifold(X: np.ndarray,
                          k: int,
                          k_graph: Optional[int] = None,
                          base_metric: str = 'euclidean') -> np.ndarray:
    """
    基于“流形最短路”的近邻：
      1) 用 base_metric（一般 'euclidean' 或 'cosine'）构建 k_graph 邻接图（边权=距离）
      2) 在图上跑最短路，得到 N×N 的 geodesic 距离
      3) 对每个点，按最短路距离选 k 个最近邻（不含自身）

    注意：
      - 步骤(2) 需要 O(N^2) 内存；N 很大时需要留意。
      - k_graph 默认取 max(2k, 10)，并截断在 [1, N-1]。
    """
    X2 = X.reshape(X.shape[0], -1)
    N = X2.shape[0]
    if N <= 1 or k <= 0:
        return np.zeros((N, 0), dtype=int)

    if k_graph is None:
        k_graph = int(min(max(2 * k, 10), max(N - 1, 1)))
    else:
        k_graph = int(min(max(k_graph, 1), N - 1))

    # 1) kNN 稀疏图（边权为 base_metric 距离），并对称化
    G = kneighbors_graph(X2, n_neighbors=k_graph, mode='distance', metric=base_metric)
    G = 0.5 * (G + G.T)

    # 2) 最短路（流形距离）
    D = shortest_path(G, directed=False, unweighted=False)  # (N, N)

    # 3) 每行取前 k 个非自身的最短路邻居
    ind_sorted = np.argsort(D, axis=1)
    rows = []
    for i in range(N):
        row = ind_sorted[i]
        row = row[row != i]  # 去掉自身
        rows.append(row[:k])
    return np.vstack(rows)

def compute_epoch_npr_manifold_aware(emb2d: np.ndarray,
                                     feat_hd: np.ndarray,
                                     k: int = 15,
                                     hd_metric: str = 'euclidean',
                                     ld_metric: str = 'euclidean',
                                     hd_manifold_k_graph: Optional[int] = None,
                                     hd_manifold_base_metric: str = 'euclidean'
                                     ) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    统一的 NPR 入口：
      - 若 hd_metric=='manifold'：高维邻居用“流形最短路”距离构造；
      - 否则：高维邻居用 sklearn 支持的普通度量（'euclidean'/'cosine'/...）。
      - 低维邻居仍按 ld_metric（默认欧氏）。

    返回：
      npr: (N,) 每个点的 NPR
      stats: {'mean','std','min','max','median'}
    """
    N = min(len(emb2d), len(feat_hd))
    if N == 0:
        return np.array([]), {'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'median': 0}

    E = np.asarray(emb2d[:N], dtype=float)
    H = np.asarray(feat_hd[:N], dtype=float)

    # 高维近邻
    if hd_metric.lower() == 'manifold':
        knn_hd = _knn_indices_manifold(H, k=k, k_graph=hd_manifold_k_graph, base_metric=hd_manifold_base_metric)
    else:
        knn_hd = _knn_indices_euclid(H, k=k, metric=hd_metric)

    # 低维近邻
    knn_ld = _knn_indices_euclid(E, k=k, metric=ld_metric)

    k_eff = knn_hd.shape[1]
    if k_eff == 0:
        npr = np.zeros(N, dtype=float)
    else:
        npr = np.empty(N, dtype=float)
        for i in range(N):
            a = set(knn_hd[i].tolist())
            b = set(knn_ld[i].tolist())
            npr[i] = len(a & b) / k_eff

    stats = {
        'mean':   float(np.mean(npr) if len(npr) else 0.0),
        'std':    float(np.std(npr) if len(npr) else 0.0),
        'min':    float(np.min(npr) if len(npr) else 0.0),
        'max':    float(np.max(npr) if len(npr) else 0.0),
        'median': float(np.median(npr) if len(npr) else 0.0),
    }
    return npr, stats


def compute_epoch_npr(emb2d: np.ndarray, feat_hd: np.ndarray, k: int = 15,
                      hd_metric: str = 'euclidean', ld_metric: str = 'euclidean') -> Tuple[np.ndarray, Dict[str,float]]:
    N = min(len(emb2d), len(feat_hd))
    if N == 0:
        return np.array([]), {'mean':0,'std':0,'min':0,'max':0,'median':0}
    E = np.asarray(emb2d[:N], dtype=float)
    H = np.asarray(feat_hd[:N], dtype=float)
    knn_hd = _knn_indices(H, k=k, metric=hd_metric)
    knn_ld = _knn_indices(E, k=k, metric=ld_metric)
    k_eff = knn_hd.shape[1]
    if k_eff == 0:
        npr = np.zeros(N, dtype=float)
    else:
        npr = np.empty(N, dtype=float)
        for i in range(N):
            a = set(knn_hd[i].tolist())
            b = set(knn_ld[i].tolist())
            npr[i] = len(a & b) / k_eff
    stats = {
        'mean': float(np.mean(npr) if len(npr) else 0),
        'std': float(np.std(npr) if len(npr) else 0),
        'min': float(np.min(npr) if len(npr) else 0),
        'max': float(np.max(npr) if len(npr) else 0),
        'median': float(np.median(npr) if len(npr) else 0),
    }
    return npr, stats

# ----------------------------- Motion Consistency -----------------------------

def _safe_cos(v1: np.ndarray, v2: np.ndarray) -> float:
    n1 = float(np.linalg.norm(v1)); n2 = float(np.linalg.norm(v2))
    if n1 < 1e-12 or n2 < 1e-12: return 0.0
    return float(np.dot(v1.ravel(), v2.ravel()) / (n1*n2))

def _curvature(traj: np.ndarray) -> np.ndarray:
    """
    任意维轨迹的“曲率”序列（用相邻步方向夹角的 π-反角做近似；直线≈0，折返大）。
    返回长度 T-2 的数组。
    """
    T = len(traj)
    if T < 3:
        return np.array([])
    out = []
    for i in range(1, T - 1):
        v1 = traj[i]   - traj[i-1]
        v2 = traj[i+1] - traj[i]
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 > 1e-12 and n2 > 1e-12:
            cosv = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
            out.append(np.pi - np.arccos(cosv))
        else:
            out.append(0.0)
    return np.asarray(out, dtype=float)

def _safe_corr(x: np.ndarray, y: np.ndarray, fn) -> float:
    """
    在样本太少/常数序列时返回 0，避免 NaN。
    fn: pearsonr 或 spearmanr
    """
    if x is None or y is None:
        return 0.0
    if len(x) < 2 or len(y) < 2 or len(x) != len(y):
        return 0.0
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return 0.0
    try:
        v = fn(x, y)[0]
        return 0.0 if np.isnan(v) else float(v)
    except Exception:
        return 0.0

def per_sample_motion_consistency(Y_ld: np.ndarray, X_hd: np.ndarray) -> Tuple[pd.DataFrame, Dict[str,float]]:
    """
    逐样本跨时间一致性指标（覆盖：距离/曲率/余弦的 Pearson+Spearman、余弦均差、方向一致率）。
    Y_ld: [T,N,2], X_hd: [T,N,D]
    """
    T, N, _ = Y_ld.shape
    rows = []
    for n in tqdm(range(N), desc='Per-sample metrics', leave=False):
        pos  = Y_ld[:, n, :]   # [T,2]
        feat = X_hd[:, n, :]   # [T,D]
        if T < 3:
            continue

        # 1) 距离序列
        pos_d  = np.linalg.norm(pos[1:]  - pos[:-1],  axis=1)
        feat_d = np.linalg.norm(feat[1:] - feat[:-1], axis=1)
        dist_pear  = _safe_corr(pos_d,  feat_d,  pearsonr)
        dist_spear = _safe_corr(pos_d,  feat_d,  spearmanr)
        
        if len(pos_d) >= 2 and len(feat_d) >= 2:
            pos_dd  = np.diff(pos_d)    # (T-2,)
            feat_dd = np.diff(feat_d)
            dist_der_pear  = _safe_corr(pos_dd, feat_dd, pearsonr)
            dist_der_spear = _safe_corr(pos_dd, feat_dd, spearmanr)
        else:
            dist_der_pear = dist_der_spear = 0.0


        # 2) 余弦（相对总位移）
        pos_total  = pos[-1]  - pos[0]
        feat_total = feat[-1] - feat[0]
        pos_cos  = np.array([_safe_cos(pos[i]-pos[i-1],   pos_total)  for i in range(1, T)], dtype=float)
        feat_cos = np.array([_safe_cos(feat[i]-feat[i-1], feat_total) for i in range(1, T)], dtype=float)
        cos_pear  = _safe_corr(pos_cos, feat_cos, pearsonr)      # Cosine Consistency Pearson
        cos_spear = _safe_corr(pos_cos, feat_cos, spearmanr)     # Cosine Consistency Spearman
        cos_mean_diff = float(np.mean(np.abs(pos_cos - feat_cos))) if len(pos_cos) == len(feat_cos) and len(pos_cos) > 0 else 0.0
        dir_agree = float(np.mean((pos_cos > 0) == (feat_cos > 0))) if len(pos_cos) == len(feat_cos) and len(pos_cos) > 0 else 0.0

        # 3) 曲率序列
        pos_c  = _curvature(pos)
        feat_c = _curvature(feat)
        curv_pear  = _safe_corr(pos_c, feat_c, pearsonr)
        curv_spear = _safe_corr(pos_c, feat_c, spearmanr)

        # 4) overall（只平均“越大越好”的项，不把 mean-diff 算进平均）
        overall = float(np.mean([
            dist_pear, dist_spear,
            curv_pear, curv_spear,
            cos_pear,  cos_spear,
            dir_agree
        ]))

        rows.append({
            'sample_id': n,
            # Distance
            'distance_pearson':  dist_pear,
            'distance_spearman': dist_spear,
            # 'dist_derivative_pearson':           dist_der_pear,
            # 'dist_derivative_spearman':          dist_der_spear,
            # Curvature
            # 'curvature_pearson':  curv_pear,
            # 'curvature_spearman': curv_spear,
            # Cosine consistency
            'cosine_consistency_pearson':  cos_pear,
            'cosine_consistency_spearman': cos_spear,
            'cosine_mean_difference':      cos_mean_diff,
            'cosine_direction_agreement':  dir_agree,
            # Overall
            # 'overall_score': overall,
        })

    df = pd.DataFrame(rows)
    stats = {}
    if len(df) > 0:
        keys = [
            'distance_pearson',
            'distance_spearman',
            # 'dist_derivative_pearson',
            # 'dist_derivative_spearman',
            # 'curvature_pearson',
            # 'curvature_spearman',
            'cosine_consistency_pearson',
            'cosine_consistency_spearman',
            'cosine_mean_difference',
            'cosine_direction_agreement',
            # 'overall_score'
        ]
        for k in keys:
            vals = df[k].values
            stats[k] = {
                'mean':   float(np.mean(vals)),
                'std':    float(np.std(vals)),
                'min':    float(np.min(vals)),
                'max':    float(np.max(vals)),
                'median': float(np.median(vals)),
            }
    return df, stats

from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr

def _knn_indices_fast(X: np.ndarray, k: int, metric: str) -> np.ndarray:
    """一次性计算 kNN 索引，去掉自身"""
    use_metric = 'cosine' if metric == 'cosine' else 'euclidean'
    nn = NearestNeighbors(n_neighbors=k+1, metric=use_metric)
    nn.fit(X)
    _, idx = nn.kneighbors(X)
    return idx[:, 1:]  # 去掉自己

def _sign_with_eps(x: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    s = np.zeros_like(x, dtype=int)
    s[x >  eps] =  1
    s[x < -eps] = -1
    return s

def _safe_corr_vec(x: np.ndarray, y: np.ndarray):
    """保持和原本逻辑一致的 Pearson / Spearman"""
    if len(x) < 2 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return 0.0, 0.0
    px = pearsonr(x, y)[0]
    sx = spearmanr(x, y)[0]
    if np.isnan(px): px = 0.0
    if np.isnan(sx): sx = 0.0
    return float(px), float(sx)

def neighbor_change_preservation_one_pair_fast(
    X_a: np.ndarray, X_b: np.ndarray,
    Y_a: np.ndarray, Y_b: np.ndarray,
    k: int = 15,
    hd_metric: str = 'euclidean',
    ld_metric: str = 'euclidean',
    eps: float = 1e-9,
):
    """只计算 sign_agreement / pearson / spearman 三个指标"""
    N = X_a.shape[0]

    knn_a = _knn_indices_fast(X_a, k=k, metric=hd_metric)
    knn_b = _knn_indices_fast(X_b, k=k, metric=hd_metric)

    sign_vals, pear_vals, spear_vals = [], [], []

    for i in range(N):
        nbrs = np.unique(np.concatenate([knn_a[i], knn_b[i]], axis=0))
        if nbrs.size == 0:
            continue

        # 高维距离
        da_hd = np.linalg.norm(X_a[nbrs] - X_a[i], axis=1) if hd_metric == 'euclidean' \
            else 1.0 - (X_a[nbrs] @ X_a[i]) / (
                np.linalg.norm(X_a[nbrs], axis=1) * (np.linalg.norm(X_a[i]) + 1e-12))
        db_hd = np.linalg.norm(X_b[nbrs] - X_b[i], axis=1) if hd_metric == 'euclidean' \
            else 1.0 - (X_b[nbrs] @ X_b[i]) / (
                np.linalg.norm(X_b[nbrs], axis=1) * (np.linalg.norm(X_b[i]) + 1e-12))
        delta_hd = db_hd - da_hd

        # 低维距离
        da_ld = np.linalg.norm(Y_a[nbrs] - Y_a[i], axis=1) if ld_metric == 'euclidean' \
            else 1.0 - (Y_a[nbrs] @ Y_a[i]) / (
                np.linalg.norm(Y_a[nbrs], axis=1) * (np.linalg.norm(Y_a[i]) + 1e-12))
        db_ld = np.linalg.norm(Y_b[nbrs] - Y_b[i], axis=1) if ld_metric == 'euclidean' \
            else 1.0 - (Y_b[nbrs] @ Y_b[i]) / (
                np.linalg.norm(Y_b[nbrs], axis=1) * (np.linalg.norm(Y_b[i]) + 1e-12))
        delta_ld = db_ld - da_ld

        # sign 一致率
        s_hd = _sign_with_eps(delta_hd, eps=eps)
        s_ld = _sign_with_eps(delta_ld, eps=eps)
        mask = (s_hd != 0) | (s_ld != 0)
        sign_agree = float(np.mean(s_hd[mask] == s_ld[mask])) if np.any(mask) else 0.0

        # 相关系数
        pear, spear = _safe_corr_vec(delta_hd, delta_ld)

        sign_vals.append(sign_agree)
        pear_vals.append(pear)
        spear_vals.append(spear)

    df = pd.DataFrame({
        'sample_id': np.arange(len(sign_vals)),
        'sign_agreement': sign_vals,
        'delta_pearson': pear_vals,
        'delta_spearman': spear_vals,
    })

    summary = {}
    for col in ['sign_agreement', 'delta_pearson', 'delta_spearman']:
        v = df[col].values
        summary[f"{col}_mean"] = {
            'mean': float(np.mean(v)),
            'std': float(np.std(v)),
            'min': float(np.min(v)),
            'max': float(np.max(v)),
            'median': float(np.median(v)),
        }
    return df, summary

def neighbor_change_preservation_over_time_fast(
    X_hd: np.ndarray, Y_ld: np.ndarray,
    k: int = 15,
    hd_metric: str = 'euclidean',
    ld_metric: str = 'euclidean',
    mode: str = 'consecutive',
    eps: float = 1e-9,
    save_dir: str = None
):
    """整体评估，只保留 sign/pearson/spearman 三个指标"""
    T, N, _ = X_hd.shape
    pair_list = [(0, T-1)] if mode == 'endpoints' else [(t-1, t) for t in range(1, T)]

    all_pair_stats = []
    for (a, b) in tqdm(pair_list, desc="Neighbor-change eval"):
        _, stat_pair = neighbor_change_preservation_one_pair_fast(
            X_a=X_hd[a], X_b=X_hd[b],
            Y_a=Y_ld[a], Y_b=Y_ld[b],
            k=k, hd_metric=hd_metric, ld_metric=ld_metric, eps=eps
        )
        all_pair_stats.append({'epoch_a': a, 'epoch_b': b, **{k_: v['mean'] for k_, v in stat_pair.items()}})

    df_stats = pd.DataFrame(all_pair_stats)
    return df_stats


def parse_args():
    p = argparse.ArgumentParser(description='Simple evaluation: NPR + motion consistency.')
    p.add_argument('--hd_file', type=str, required=True, help='高维文件（.npy 或 .npz）')
    p.add_argument('--ld_file', type=str, required=True, help='低维文件（.npy 或 .npz）')
    p.add_argument('--save_dir', type=str, required=True, help='输出目录')
    p.add_argument('--k', type=int, default=15, help='NPR 的 k 近邻')
    p.add_argument('--hd_metric', type=str, default='euclidean', choices=['euclidean','cosine'])
    p.add_argument('--ld_metric', type=str, default='euclidean', choices=['euclidean','cosine'])
    p.add_argument('--t_start', type=int, default=35, help='起始 epoch（含）')
    p.add_argument('--t_end',   type=int, default=50, help='结束 epoch（含）')
    p.add_argument('--hd_k_graph', type=int, default=None,
                help="若 hd_metric='manifold'，构图时的 k（默认会用 max(2k,10) 截断到 [1, N-1]）")
    p.add_argument('--hd_base_metric', type=str, default='euclidean',
                choices=['euclidean','cosine'],
                help="若 hd_metric='manifold'，构图边权的底层度量")
    p.add_argument('--chg_mode', type=str, default='consecutive',
                   choices=['consecutive', 'endpoints'],
                   help="邻居距离变化评估的时间配对方式：连续 (t-1,t) 或首尾 (start,end)")
    p.add_argument('--chg_knn_from', type=str, default='a',
                   choices=['a','b'],
                   help="参考 kNN 来自配对中的哪个 epoch（a=早期，b=后期）")
    p.add_argument('--chg_eps', type=float, default=1e-9,
                   help="判断变近/变远的零阈值")
    return p.parse_args()

def slice_time_range(X_hd: np.ndarray, Y_ld: np.ndarray, t_start: int | None, t_end: int | None):
    """
    返回裁剪后的 (X_sel, Y_sel, labels)，labels 是原始 epoch 编号（用于命名/打印）
    """
    T = X_hd.shape[0]
    if t_start is None and t_end is None:
        return X_hd, Y_ld, np.arange(T)
    if t_start is None: t_start = 0
    if t_end   is None: t_end   = T - 1
    # 合法化 & 包含式区间
    t_start = max(0, int(t_start))
    t_end   = min(T - 1, int(t_end))
    if t_start > t_end:
        raise ValueError(f"Invalid time window: [{t_start}, {t_end}]")
    labels = np.arange(T)[t_start:t_end+1]
    return X_hd[t_start:t_end+1], Y_ld[t_start:t_end+1], labels

def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    X_hd, Y_ld = load_inputs(args.hd_file, args.ld_file)
    X_hd, Y_ld, epoch_labels = slice_time_range(X_hd, Y_ld, args.t_start, args.t_end)
    T, N, D = X_hd.shape
    print(f"[Info] Loaded: X_hd {X_hd.shape}, Y_ld {Y_ld.shape}")

    # 1) NPR per epoch
    per_epoch_std = {}
    per_epoch_mani = {}
    rows_std, rows_mani = [], []

    for e in tqdm(range(T), desc='NPR per epoch'):
        # --- 原版 NPR ---
        npr_std, s_std = compute_epoch_npr(
            emb2d=Y_ld[e], feat_hd=X_hd[e],
            k=args.k, hd_metric=args.hd_metric,
            ld_metric=args.ld_metric
        )
        per_epoch_std[e] = {'mean': s_std['mean'], 'std': s_std['std'],
                            'min': s_std['min'], 'max': s_std['max'],
                            'median': s_std['median'], 'N': int(len(npr_std)),
                            'k': int(min(args.k, max(0, len(npr_std)-1)))}
        pd.DataFrame({'index': np.arange(len(npr_std)), 'npr': npr_std}).to_csv(
            os.path.join(args.save_dir, f'npr_epoch_{e}_std.csv'),
            index=False, float_format='%.6f'
        )
        rows_std.append({'epoch': e, **per_epoch_std[e]})

        # --- 流形 NPR（只有在要求“同时计算”或hd_metric=manifold时才跑） ---
        npr_mani, s_mani = compute_epoch_npr_manifold_aware(
            emb2d=Y_ld[e],
            feat_hd=X_hd[e],
            k=args.k,
            hd_metric='manifold',
            ld_metric=args.ld_metric,
            hd_manifold_k_graph=args.hd_k_graph,
            hd_manifold_base_metric=args.hd_base_metric
        )
        per_epoch_mani[e] = {'mean': s_mani['mean'], 'std': s_mani['std'],
                            'min': s_mani['min'], 'max': s_mani['max'],
                            'median': s_mani['median'], 'N': int(len(npr_mani)),
                            'k': int(min(args.k, max(0, len(npr_mani)-1)))}
        pd.DataFrame({'index': np.arange(len(npr_mani)), 'npr': npr_mani}).to_csv(
            os.path.join(args.save_dir, f'npr_epoch_{e}_manifold.csv'),
            index=False, float_format='%.6f'
        )
        rows_mani.append({'epoch': e, **per_epoch_mani[e]})


    # summary
    df_std = pd.DataFrame(rows_std)
    df_std.to_csv(os.path.join(args.save_dir, 'npr_summary_std.csv'), index=False)
    with open(os.path.join(args.save_dir, 'npr_summary_std.json'), 'w') as f:
        json.dump(per_epoch_std, f, indent=2)

    means = [s['mean'] for s in per_epoch_std.values()]
    overall_unweighted_std = float(np.mean(means))
    total_N_std = sum(s['N'] for s in per_epoch_std.values())
    overall_weighted_std = float(sum(per_epoch_std[e]['mean'] * per_epoch_std[e]['N']
                                     for e in per_epoch_std) / max(1, total_N_std))
    print("\n===== NPR Overall (STD metric) =====")
    print(f"Unweighted mean of epoch means: {overall_unweighted_std:.4f}")
    print(f"Weighted by N (pooled overall): {overall_weighted_std:.4f}")

    # 流形汇总
    df_mani = pd.DataFrame(rows_mani)
    df_mani.to_csv(os.path.join(args.save_dir, 'npr_summary_manifold.csv'), index=False)
    with open(os.path.join(args.save_dir, 'npr_summary_manifold.json'), 'w') as f:
        json.dump(per_epoch_mani, f, indent=2)

    means_m = [s['mean'] for s in per_epoch_mani.values()]
    overall_unweighted_m = float(np.mean(means_m))
    total_N_m = sum(s['N'] for s in per_epoch_mani.values())
    overall_weighted_m = float(sum(per_epoch_mani[e]['mean'] * per_epoch_mani[e]['N']
                                   for e in per_epoch_mani) / max(1, total_N_m))
    print("\n===== NPR Overall (MANIFOLD metric) =====")
    print(f"Unweighted mean of epoch means: {overall_unweighted_m:.4f}")
    print(f"Weighted by N (pooled overall): {overall_weighted_m:.4f}")


    # 2) Motion consistency per sample over T
    df_samples, stats = per_sample_motion_consistency(Y_ld, X_hd)
    df_samples.to_csv(os.path.join(args.save_dir, 'per_sample_metrics.csv'), index=False, float_format='%.6f')
    payload = {'per_sample': stats}
    # if npr_overall is not None:
    #     payload['npr_overall'] = npr_overall
    with open(os.path.join(args.save_dir, 'metrics_summary.json'), 'w') as f:
        json.dump(payload, f, indent=2)

    print("\n===== Motion–Semantic Consistency Summary =====")
    for k, v in stats.items():
        print(f"{k:>30s} | mean={v['mean']:.4f} std={v['std']:.4f} median={v['median']:.4f} min={v['min']:.4f} max={v['max']:.4f}")

    print(f"\n[Done] Saved outputs to {args.save_dir}")
    
    # 3) Neighbor-change preservation
    df_stats = neighbor_change_preservation_over_time_fast(
        X_hd, Y_ld,
        k=args.k,
        hd_metric=args.hd_metric,
        ld_metric=args.ld_metric,
        mode=args.chg_mode,
        eps=args.chg_eps,
        save_dir=args.save_dir
    )

    # 保存 pair 级别结果
    df_stats.to_csv(os.path.join(args.save_dir, 'neighbor_change_summary_pairs.csv'),
                    index=False, float_format='%.6f')

    # 额外整体统计（全局平均）
    chg_summary = {
        col: {
            'mean': float(df_stats[col].mean()),
            'std': float(df_stats[col].std()),
            'min': float(df_stats[col].min()),
            'max': float(df_stats[col].max()),
            'median': float(df_stats[col].median()),
        }
        for col in ['sign_agreement_mean', 'delta_pearson_mean', 'delta_spearman_mean']
    }

    with open(os.path.join(args.save_dir, 'neighbor_change_overall.json'), 'w') as f:
        json.dump(chg_summary, f, indent=2)

    print("\n===== Neighbor-change Preservation (fast) =====")
    for k, v in chg_summary.items():
        print(f"{k:>28s} | mean={v['mean']:.4f} std={v['std']:.4f} "
            f"median={v['median']:.4f} min={v['min']:.4f} max={v['max']:.4f}")




if __name__ == '__main__':
    main()

# python debug3.py --hd_file /inspire/hdd/global_user/liuyiming-240108540153/training_dynamic/selected_subset/cifar10_50epochs_500sample.npy --ld_file /inspire/hdd/global_user/liuyiming-240108540153/training_dynamic/selected_subset/checkpoints/test/Y_ld.npz --save_dir /inspire/hdd/global_user/liuyiming-240108540153/training_dynamic/selected_subset/checkpoints/out_eval

# python debug3.py --hd_file /inspire/hdd/global_user/liuyiming-240108540153/training_dynamic/image_models/ResNet-CIFAR100/Classification-normal/selected_subset/stacked_train_embeddings.npy --ld_file /inspire/hdd/global_user/liuyiming-240108540153/training_dynamic/image_models/ResNet-CIFAR100/Classification-normal/selected_subset/checkpoints/Y_ld_robust.npz --save_dir /inspire/hdd/global_user/liuyiming-240108540153/training_dynamic/image_models/ResNet-CIFAR100/Classification-normal/selected_subset/out_eval/dynavis

# python debug3.py --hd_file /inspire/hdd/global_user/liuyiming-240108540153/training_dynamic/image_models/ResNet-CIFAR100/Classification-normal/selected_subset/stacked_train_embeddings.npy --ld_file /inspire/hdd/global_user/liuyiming-240108540153/training_dynamic/image_models/ResNet-CIFAR100/Classification-normal/selected_subset/DVI_results/sample_idx_full/saved_data/emb_dvi.npy --save_dir /inspire/hdd/global_user/liuyiming-240108540153/training_dynamic/image_models/ResNet-CIFAR100/Classification-normal/selected_subset/out_eval/dvi

# python debug3.py --hd_file /inspire/hdd/global_user/liuyiming-240108540153/training_dynamic/image_models/ResNet-CIFAR100/Classification-normal/selected_subset/stacked_train_embeddings.npy --ld_file /inspire/hdd/global_user/liuyiming-240108540153/training_dynamic/image_models/ResNet-CIFAR100/Classification-normal/selected_subset/TimeVis+_results/sample_idx_full/saved_data/emb_timevis.npy --save_dir /inspire/hdd/global_user/liuyiming-240108540153/training_dynamic/image_models/ResNet-CIFAR100/Classification-normal/selected_subset/out_eval/timevis

# python train_jvp_modi.py --norm_mode robust --use_calib --use_sat --use_rank