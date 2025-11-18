#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
I/O helpers for DynaVis evaluation.

Responsible for:
  - loading high- and low-dimensional trajectories
  - basic shape checking and cropping
  - slicing a time window [t_start, t_end]
"""

from __future__ import annotations
from typing import List, Tuple
import numpy as np


def _load_array(file: str, pref_keys: List[str]) -> np.ndarray:
    """
    Load an array from .npy or .npz.

    For .npz, try preferred keys in order; if none match, fall back to the first
    array in the archive.
    """
    if file.endswith(".npy"):
        arr = np.load(file)
        return np.asarray(arr)

    if file.endswith(".npz"):
        data = np.load(file)
        # try preferred keys
        for k in pref_keys:
            if k in data:
                return np.asarray(data[k])
        # fallback: first entry
        if len(data.files) == 0:
            raise ValueError(f"No arrays stored in npz: {file}")
        return np.asarray(data[data.files[0]])

    raise ValueError(f"Unsupported file type: {file}")


def load_inputs(hd_file: str, ld_file: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load high-dimensional (X_hd) and low-dimensional (Y_ld) trajectories.

    Expected shapes:
      X_hd: [T, N, D]
      Y_ld: [T, N, 2]

    If shapes mismatch in T or N, we crop to the common minimum.
    """
    X_hd = _load_array(hd_file, pref_keys=["X_hd", "X", "hd"])
    Y_ld = _load_array(ld_file, pref_keys=["Y_ld", "Y", "ld"])

    if X_hd.ndim != 3 or Y_ld.ndim != 3:
        raise ValueError(
            f"Expect X_hd [T,N,D] and Y_ld [T,N,2], "
            f"got {X_hd.shape} and {Y_ld.shape}"
        )

    T1, N1, _ = X_hd.shape
    T2, N2, d2 = Y_ld.shape
    if d2 != 2:
        raise ValueError(f"Y_ld last dim must be 2, got {d2}")

    if T1 != T2 or N1 != N2:
        T = min(T1, T2)
        N = min(N1, N2)
        X_hd = X_hd[:T, :N, :]
        Y_ld = Y_ld[:T, :N, :]
        print(f"[Warn] Shapes mismatched; cropped to X_hd {X_hd.shape}, Y_ld {Y_ld.shape}")

    return X_hd.astype(np.float64), Y_ld.astype(np.float64)


def slice_time_range(
    X_hd: np.ndarray,
    Y_ld: np.ndarray,
    t_start: int | None,
    t_end: int | None,
):
    """
    Slice trajectories in the time dimension.

    Args:
      X_hd, Y_ld: full trajectories, shape [T, N, D] and [T, N, 2]
      t_start: inclusive start index (0-based). If None, use 0.
      t_end:   inclusive end index (0-based). If None, use T-1.

    Returns:
      X_slice, Y_slice, epoch_labels (np.arange of selected epochs).
    """
    T = X_hd.shape[0]
    if t_start is None and t_end is None:
        return X_hd, Y_ld, np.arange(T)

    if t_start is None:
        t_start = 0
    if t_end is None:
        t_end = T - 1

    t_start = max(0, int(t_start))
    t_end = min(T - 1, int(t_end))
    if t_start > t_end:
        raise ValueError(f"Invalid time window: [{t_start}, {t_end}]")

    labels = np.arange(T)[t_start : t_end + 1]
    return X_hd[t_start : t_end + 1], Y_ld[t_start : t_end + 1], labels
