#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Per-sample motion consistency metrics for DynaVis.

For each sample (trajectory across epochs), we compute:
  - Step-length Pearson / Spearman (low vs. high dimensional)
  - Cosine similarity to total displacement (Pearson / Spearman)
  - Direction agreement rate (sign of cosine values)
  - Top-3 step-length recall and ordering consistency
"""

from __future__ import annotations
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from scipy.stats import pearsonr, spearmanr


# ----------------------------------------------------------------------
# Basic helpers
# ----------------------------------------------------------------------

def _safe_cos(v1: np.ndarray, v2: np.ndarray) -> float:
    """Cosine similarity with guard for near-zero vectors."""
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 < 1e-12 or n2 < 1e-12:
        return 0.0
    return float(np.dot(v1.ravel(), v2.ravel()) / (n1 * n2))


def _safe_corr(x: np.ndarray, y: np.ndarray, fn) -> float:
    """
    Safe wrapper for Pearson/Spearman correlation.

    Returns 0.0 if:
      - length < 2
      - all values are (almost) constant
      - any error occurs or result is NaN
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


def _topk_indices_desc(x: np.ndarray, k: int) -> np.ndarray:
    """
    Return indices of the top-k values in descending order (stable).
    """
    k = min(k, x.shape[0])
    if k <= 0:
        return np.array([], dtype=int)
    part = np.argpartition(-x, k - 1)[:k]
    return part[np.argsort(-x[part], kind="stable")]


def _order_agreement_ratio(ref_vals: np.ndarray, cmp_vals: np.ndarray) -> float:
    """
    For 3 elements, compute the fraction of pairwise orderings that agree.

    For indices (0,1,2) we check 3 pairs: (0,1), (0,2), (1,2).
    We count how many times the '>' relation is preserved from ref_vals to
    cmp_vals. Ties (both zero) are treated as agreement; mixed ties are treated
    as disagreement.
    """
    if ref_vals.shape[0] != 3 or cmp_vals.shape[0] != 3:
        return 0.0

    pairs = [(0, 1), (0, 2), (1, 2)]
    agree = 0
    for i, j in pairs:
        ref_sign = np.sign(ref_vals[i] - ref_vals[j])
        cmp_sign = np.sign(cmp_vals[i] - cmp_vals[j])

        if ref_sign == 0 and cmp_sign == 0:
            agree += 1
        elif ref_sign == 0 or cmp_sign == 0:
            # one side tied, the other not -> treat as disagreement
            pass
        elif np.sign(ref_sign) == np.sign(cmp_sign):
            agree += 1

    return agree / 3.0


# ----------------------------------------------------------------------
# Main metric: per-sample motion consistency
# ----------------------------------------------------------------------

def per_sample_motion_consistency(
    Y_ld: np.ndarray,
    X_hd: np.ndarray,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    """
    Compute per-sample motion consistency metrics across time.

    Args:
      Y_ld: [T, N, 2] low-dimensional trajectories
      X_hd: [T, N, D] high-dimensional trajectories

    Returns:
      df:    DataFrame with per-sample metrics
      stats: summary statistics over all samples
    """
    T, N, _ = Y_ld.shape
    rows = []

    for n in tqdm(range(N), desc="Per-sample metrics", leave=False):
        pos = Y_ld[:, n, :]    # [T, 2]
        feat = X_hd[:, n, :]   # [T, D]
        if T < 3:
            continue

        # 1) Step lengths (per epoch difference)
        pos_d = np.linalg.norm(pos[1:] - pos[:-1], axis=1)        # length T-1
        feat_d = np.linalg.norm(feat[1:] - feat[:-1], axis=1)

        dist_pear = _safe_corr(pos_d, feat_d, pearsonr)
        dist_spear = _safe_corr(pos_d, feat_d, spearmanr)

        # 2) Cosine similarity to total displacement
        pos_total = pos[-1] - pos[0]
        feat_total = feat[-1] - feat[0]

        pos_cos = np.array(
            [_safe_cos(pos[i] - pos[i - 1], pos_total) for i in range(1, T)],
            dtype=float,
        )
        feat_cos = np.array(
            [_safe_cos(feat[i] - feat[i - 1], feat_total) for i in range(1, T)],
            dtype=float,
        )

        cos_pear = _safe_corr(pos_cos, feat_cos, pearsonr)
        cos_spear = _safe_corr(pos_cos, feat_cos, spearmanr)
        cos_mean_diff = float(
            np.mean(np.abs(pos_cos - feat_cos))
        ) if len(pos_cos) == len(feat_cos) and len(pos_cos) > 0 else 0.0
        dir_agree = float(
            np.mean((pos_cos > 0) == (feat_cos > 0))
        ) if len(pos_cos) == len(feat_cos) and len(pos_cos) > 0 else 0.0

        # 3) Top-3 distance recall and order agreement
        if len(feat_d) >= 3 and len(pos_d) >= 3:
            idx3_hd = _topk_indices_desc(feat_d, 3)  # high-dim is reference
            idx3_ld = _topk_indices_desc(pos_d, 3)

            # (a) Top-3 distance recall: how many of hd top3 steps
            #     also appear in ld top3 (Recall@3).
            top3_recall = len(set(idx3_hd.tolist()) & set(idx3_ld.tolist())) / 3.0

            # (b) Top-3 order agreement: restrict to the same 3 steps (hd indices)
            hd_vals = feat_d[idx3_hd]
            ld_vals = pos_d[idx3_hd]
            top3_order = _order_agreement_ratio(hd_vals, ld_vals)
        else:
            top3_recall = 0.0
            top3_order = 0.0

        rows.append(
            {
                "sample_id": n,
                # Distance series
                "distance_pearson": dist_pear,
                "distance_spearman": dist_spear,
                # Cosine consistency
                "cosine_consistency_pearson": cos_pear,
                "cosine_consistency_spearman": cos_spear,
                "cosine_mean_difference": cos_mean_diff,
                "cosine_direction_agreement": dir_agree,
                # Top-3 metrics
                "top3_distance_recall": top3_recall,
                "top3_order_agreement": top3_order,
            }
        )

    df = pd.DataFrame(rows)
    stats: Dict[str, Dict[str, float]] = {}

    if len(df) > 0:
        keys = [
            "distance_pearson",
            "distance_spearman",
            "cosine_consistency_pearson",
            "cosine_consistency_spearman",
            "cosine_mean_difference",
            "cosine_direction_agreement",
            "top3_distance_recall",
            "top3_order_agreement",
        ]
        for k in keys:
            vals = df[k].values
            stats[k] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
                "median": float(np.median(vals)),
            }

    return df, stats
