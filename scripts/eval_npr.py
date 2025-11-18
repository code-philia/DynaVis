#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neighbor Preserving Rate (NPR) evaluation for DynaVis.

Provides:
  - per-epoch NPR computation (Euclidean / cosine)
  - optional manifold-aware NPR (geodesic distances on kNN graph)
"""

from __future__ import annotations
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy.sparse.csgraph import shortest_path
from sklearn.neighbors import kneighbors_graph


# ----------------------------------------------------------------------
# Basic kNN helpers
# ----------------------------------------------------------------------

def _knn_indices_euclid(X: np.ndarray, k: int, metric: str = "euclidean") -> np.ndarray:
    """
    Compute k nearest neighbors in the given metric (excluding self).

    Args:
      X: [N, D] data
      k: number of neighbors
      metric: 'euclidean' or 'cosine'

    Returns:
      indices: [N, k] neighbor indices
    """
    X2 = X.reshape(X.shape[0], -1)
    N = X2.shape[0]
    k_eff = min(k + 1, N)
    if k_eff <= 1:
        return np.zeros((N, 0), dtype=int)

    nn = NearestNeighbors(n_neighbors=k_eff, metric=metric, n_jobs=-1)
    nn.fit(X2)
    ind = nn.kneighbors(return_distance=False)
    # drop self
    return ind[:, 1:k_eff]


def _knn_indices_manifold(
    X: np.ndarray,
    k: int,
    k_graph: Optional[int] = None,
    base_metric: str = "euclidean",
) -> np.ndarray:
    """
    kNN based on manifold (geodesic) distances approximated by a kNN graph.

    Steps:
      1) Build a symmetric k_graph-NN graph in base_metric.
      2) Run shortest_path on the graph (weighted).
      3) For each node, sort by geodesic distance and take top-k neighbors.
    """
    X2 = X.reshape(X.shape[0], -1)
    N = X2.shape[0]
    if N <= 1 or k <= 0:
        return np.zeros((N, 0), dtype=int)

    if k_graph is None:
        # Slightly over-connected graph by default
        k_graph = int(min(max(2 * k, 10), max(N - 1, 1)))
    else:
        k_graph = int(min(max(k_graph, 1), N - 1))

    G = kneighbors_graph(X2, n_neighbors=k_graph, mode="distance", metric=base_metric)
    # make it symmetric; this is important for shortest_path
    G = 0.5 * (G + G.T)

    D = shortest_path(G, directed=False, unweighted=False)
    ind_sorted = np.argsort(D, axis=1)

    rows = []
    for i in range(N):
        row = ind_sorted[i]
        row = row[row != i]          # drop self
        rows.append(row[:k])
    return np.vstack(rows)


# ----------------------------------------------------------------------
# NPR computation
# ----------------------------------------------------------------------

def compute_epoch_npr(
    emb2d: np.ndarray,
    feat_hd: np.ndarray,
    k: int = 15,
    hd_metric: str = "euclidean",
    ld_metric: str = "euclidean",
    hd_manifold_k_graph: Optional[int] = None,
    hd_manifold_base_metric: str = "euclidean",
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Compute Neighbor Preserving Rate (NPR) for a single epoch.

    Args:
      emb2d:   low-dimensional embeddings [N, 2]
      feat_hd: high-dimensional embeddings [N, D]
      k:       number of neighbors
      hd_metric:
        - 'euclidean' or 'cosine' (direct metric in high-dim)
        - 'manifold'   (use graph-geodesic distances in high-dim)
      ld_metric: 'euclidean' or 'cosine' for the visualization space
      hd_manifold_k_graph: k used when building the high-dim kNN graph
      hd_manifold_base_metric: base edge metric for the kNN graph

    Returns:
      npr:   [N] per-sample NPR values
      stats: summary statistics (mean/std/min/max/median)
    """
    N = min(len(emb2d), len(feat_hd))
    if N == 0:
        return np.array([]), {"mean": 0, "std": 0, "min": 0, "max": 0, "median": 0}

    E = np.asarray(emb2d[:N], dtype=float)
    H = np.asarray(feat_hd[:N], dtype=float)

    hd_metric_lower = hd_metric.lower()
    if hd_metric_lower == "manifold":
        knn_hd = _knn_indices_manifold(
            H,
            k=k,
            k_graph=hd_manifold_k_graph,
            base_metric=hd_manifold_base_metric,
        )
    else:
        # 'euclidean' or 'cosine'
        knn_hd = _knn_indices_euclid(H, k=k, metric=hd_metric_lower)

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
        "mean":   float(np.mean(npr) if len(npr) else 0.0),
        "std":    float(np.std(npr) if len(npr) else 0.0),
        "min":    float(np.min(npr) if len(npr) else 0.0),
        "max":    float(np.max(npr) if len(npr) else 0.0),
        "median": float(np.median(npr) if len(npr) else 0.0),
    }
    return npr, stats
