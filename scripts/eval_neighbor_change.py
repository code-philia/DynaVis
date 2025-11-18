#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neighbor-distance change preservation for DynaVis.

We compare how neighbor distances change between two epochs in:
  - high-dimensional space
  - low-dimensional visualization space

For each pair of epochs (a, b) we compute:
  - sign agreement of distance changes (closer / farther)
  - Pearson / Spearman between high- and low-dim distance deltas
"""

from __future__ import annotations
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from scipy.stats import pearsonr, spearmanr
from sklearn.neighbors import NearestNeighbors


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _knn_indices_fast(X: np.ndarray, k: int, metric: str) -> np.ndarray:
    """
    Fast kNN using scikit-learn.

    metric:
      - 'euclidean'
      - 'cosine'
    """
    use_metric = "cosine" if metric == "cosine" else "euclidean"
    nn = NearestNeighbors(n_neighbors=k + 1, metric=use_metric)
    nn.fit(X)
    _, idx = nn.kneighbors(X)
    return idx[:, 1:]


def _sign_with_eps(x: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """
    Map real numbers to {-1, 0, +1} with a deadzone around zero of width eps.
    """
    s = np.zeros_like(x, dtype=int)
    s[x > eps] = 1
    s[x < -eps] = -1
    return s


def _safe_corr_vec(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """
    Safe Pearson/Spearman for 1D vectors (used for distance deltas).
    """
    if len(x) < 2 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return 0.0, 0.0

    px = pearsonr(x, y)[0]
    sx = spearmanr(x, y)[0]
    if np.isnan(px):
        px = 0.0
    if np.isnan(sx):
        sx = 0.0
    return float(px), float(sx)


# ----------------------------------------------------------------------
# Core metrics
# ----------------------------------------------------------------------

def neighbor_change_preservation_one_pair_fast(
    X_a: np.ndarray,
    X_b: np.ndarray,
    Y_a: np.ndarray,
    Y_b: np.ndarray,
    k: int = 15,
    hd_metric: str = "euclidean",
    ld_metric: str = "euclidean",
    eps: float = 1e-9,
):
    """
    Evaluate neighbor-distance change preservation between two epochs (a, b).

    Args:
      X_a, X_b: high-dim embeddings at epochs a and b, shape [N, D]
      Y_a, Y_b: low-dim embeddings at epochs a and b, shape [N, 2]
      k:        number of neighbors for defining local neighborhood
      hd_metric / ld_metric: 'euclidean' or 'cosine'
      eps:      threshold for deciding "closer / farther / unchanged"

    Returns:
      df: DataFrame with per-sample sign agreement and correlations
      summary: dict with mean/std/min/max/median for each metric
    """
    N = X_a.shape[0]

    knn_a = _knn_indices_fast(X_a, k=k, metric=hd_metric)
    knn_b = _knn_indices_fast(X_b, k=k, metric=hd_metric)

    sign_vals: List[float] = []
    pear_vals: List[float] = []
    spear_vals: List[float] = []

    for i in range(N):
        # union of neighbors at a and b
        nbrs = np.unique(np.concatenate([knn_a[i], knn_b[i]], axis=0))
        if nbrs.size == 0:
            continue

        # High-dimensional distances
        if hd_metric == "euclidean":
            da_hd = np.linalg.norm(X_a[nbrs] - X_a[i], axis=1)
            db_hd = np.linalg.norm(X_b[nbrs] - X_b[i], axis=1)
        else:  # cosine distance
            da_hd = 1.0 - (X_a[nbrs] @ X_a[i]) / (
                np.linalg.norm(X_a[nbrs], axis=1) * (np.linalg.norm(X_a[i]) + 1e-12)
            )
            db_hd = 1.0 - (X_b[nbrs] @ X_b[i]) / (
                np.linalg.norm(X_b[nbrs], axis=1) * (np.linalg.norm(X_b[i]) + 1e-12)
            )
        delta_hd = db_hd - da_hd

        # Low-dimensional distances
        if ld_metric == "euclidean":
            da_ld = np.linalg.norm(Y_a[nbrs] - Y_a[i], axis=1)
            db_ld = np.linalg.norm(Y_b[nbrs] - Y_b[i], axis=1)
        else:
            da_ld = 1.0 - (Y_a[nbrs] @ Y_a[i]) / (
                np.linalg.norm(Y_a[nbrs], axis=1) * (np.linalg.norm(Y_a[i]) + 1e-12)
            )
            db_ld = 1.0 - (Y_b[nbrs] @ Y_b[i]) / (
                np.linalg.norm(Y_b[nbrs], axis=1) * (np.linalg.norm(Y_b[i]) + 1e-12)
            )
        delta_ld = db_ld - da_ld

        # Sign agreement of distance changes
        s_hd = _sign_with_eps(delta_hd, eps=eps)
        s_ld = _sign_with_eps(delta_ld, eps=eps)
        mask = (s_hd != 0) | (s_ld != 0)
        sign_agree = float(np.mean(s_hd[mask] == s_ld[mask])) if np.any(mask) else 0.0

        # Correlations of distance deltas
        pear, spear = _safe_corr_vec(delta_hd, delta_ld)

        sign_vals.append(sign_agree)
        pear_vals.append(pear)
        spear_vals.append(spear)

    df = pd.DataFrame(
        {
            "sample_id": np.arange(len(sign_vals)),
            "sign_agreement": sign_vals,
            "delta_pearson": pear_vals,
            "delta_spearman": spear_vals,
        }
    )

    summary: Dict[str, Dict[str, float]] = {}
    for col in ["sign_agreement", "delta_pearson", "delta_spearman"]:
        v = df[col].values
        summary[f"{col}_mean"] = {
            "mean": float(np.mean(v)),
            "std": float(np.std(v)),
            "min": float(np.min(v)),
            "max": float(np.max(v)),
            "median": float(np.median(v)),
        }
    return df, summary


def neighbor_change_preservation_over_time_fast(
    X_hd: np.ndarray,
    Y_ld: np.ndarray,
    k: int = 15,
    hd_metric: str = "euclidean",
    ld_metric: str = "euclidean",
    mode: str = "consecutive",
    eps: float = 1e-9,
):
    """
    Aggregate neighbor-change preservation over multiple epoch pairs.

    Args:
      X_hd, Y_ld: full trajectories [T, N, D] and [T, N, 2]
      k:          neighbors for local neighborhoods
      hd_metric / ld_metric: 'euclidean' or 'cosine'
      mode:
        - 'consecutive': evaluate all (t-1, t) pairs
        - 'endpoints':  only evaluate (0, T-1)
      eps:        threshold for sign of distance changes

    Returns:
      df_stats: DataFrame with mean metrics for each epoch pair
    """
    T, N, _ = X_hd.shape
    if mode == "endpoints":
        pair_list = [(0, T - 1)]
    else:
        pair_list = [(t - 1, t) for t in range(1, T)]

    all_pair_stats = []
    for (a, b) in tqdm(pair_list, desc="Neighbor-change eval"):
        _, stat_pair = neighbor_change_preservation_one_pair_fast(
            X_a=X_hd[a],
            X_b=X_hd[b],
            Y_a=Y_ld[a],
            Y_b=Y_ld[b],
            k=k,
            hd_metric=hd_metric,
            ld_metric=ld_metric,
            eps=eps,
        )
        all_pair_stats.append(
            {
                "epoch_a": a,
                "epoch_b": b,
                **{k_: v["mean"] for k_, v in stat_pair.items()},
            }
        )

    df_stats = pd.DataFrame(all_pair_stats)
    return df_stats
