#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DynaVis evaluation entry point.

This script ties everything together:
  1) Per-epoch NPR (standard / manifold-aware)
  2) Per-sample motion–semantic consistency
  3) Neighbor-distance change preservation

Example:
  python scripts/evaluation.py \
      --hd_file path/to/X_hd.npy \
      --ld_file path/to/Y_ld.npy \
      --save_dir results/dynavis_eval \
      --k 15 \
      --hd_metric euclidean \
      --ld_metric euclidean \
      --t_start 85 --t_end 100
"""

from __future__ import annotations
import argparse
import json
import os

import numpy as np
import pandas as pd

from scripts.eval_io import load_inputs, slice_time_range
from scripts.eval_npr import compute_epoch_npr
from scripts.eval_motion import per_sample_motion_consistency
from scripts.eval_neighbor_change import neighbor_change_preservation_over_time_fast


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="DynaVis evaluation: NPR + motion consistency + neighbor-change."
    )

    p.add_argument("--hd_file", type=str, required=True,
                   help="High-dimensional trajectory file (.npy or .npz).")
    p.add_argument("--ld_file", type=str, required=True,
                   help="Low-dimensional trajectory file (.npy or .npz).")
    p.add_argument("--save_dir", type=str, required=True,
                   help="Output directory for CSV/JSON results.")

    # NPR-related arguments
    p.add_argument("--k", type=int, default=15,
                   help="Number of neighbors k for NPR and neighbor-change.")
    p.add_argument(
        "--hd_metric",
        type=str,
        default="euclidean",
        choices=["euclidean", "cosine", "manifold"],
        help="Metric for high-dimensional NPR: euclidean/cosine/manifold.",
    )
    p.add_argument(
        "--ld_metric",
        type=str,
        default="euclidean",
        choices=["euclidean", "cosine"],
        help="Metric for low-dimensional NPR.",
    )
    p.add_argument(
        "--hd_k_graph",
        type=int,
        default=None,
        help=(
            "If hd_metric='manifold', k for building the kNN graph "
            "(default uses max(2k,10) clipped to [1, N-1])."
        ),
    )
    p.add_argument(
        "--hd_base_metric",
        type=str,
        default="euclidean",
        choices=["euclidean", "cosine"],
        help="Base edge metric when hd_metric='manifold'.",
    )

    # Time window
    p.add_argument(
        "--t_start",
        type=int,
        default=None,
        help="Start epoch index (inclusive, 0-based). If None, use 0.",
    )
    p.add_argument(
        "--t_end",
        type=int,
        default=None,
        help="End epoch index (inclusive, 0-based). If None, use T-1.",
    )

    # Neighbor-change options
    p.add_argument(
        "--chg_mode",
        type=str,
        default="consecutive",
        choices=["consecutive", "endpoints"],
        help="How to pair epochs for neighbor-change evaluation.",
    )
    p.add_argument(
        "--chg_eps",
        type=float,
        default=1e-9,
        help="Epsilon for deciding the sign of distance changes.",
    )

    return p.parse_args()


# ----------------------------------------------------------------------
# Main driver
# ----------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    # 0) Load and slice trajectories
    X_hd, Y_ld = load_inputs(args.hd_file, args.ld_file)
    print("[Info] Loaded from disk:", X_hd.shape, Y_ld.shape)

    X_hd, Y_ld, epoch_labels = slice_time_range(X_hd, Y_ld, args.t_start, args.t_end)
    T, N, D = X_hd.shape
    print(f"[Info] After slicing: X_hd {X_hd.shape}, Y_ld {Y_ld.shape}")
    print(f"[Info] Evaluating epochs: {epoch_labels.tolist()}")

    # 1) NPR per epoch ---------------------------------------------------
    print("\n===== 1) NPR per epoch =====")
    per_epoch_stats = {}
    rows_npr = []

    for local_e in range(T):
        global_e = int(epoch_labels[local_e])

        npr_vals, s = compute_epoch_npr(
            emb2d=Y_ld[local_e],
            feat_hd=X_hd[local_e],
            k=args.k,
            hd_metric=args.hd_metric,
            ld_metric=args.ld_metric,
            hd_manifold_k_graph=args.hd_k_graph,
            hd_manifold_base_metric=args.hd_base_metric,
        )

        per_epoch_stats[global_e] = {
            "mean": s["mean"],
            "std": s["std"],
            "min": s["min"],
            "max": s["max"],
            "median": s["median"],
            "N": int(len(npr_vals)),
            "k": int(min(args.k, max(0, len(npr_vals) - 1))),
        }

        # save per-epoch NPR distribution
        df_epoch = pd.DataFrame(
            {"index": np.arange(len(npr_vals)), "npr": npr_vals}
        )
        df_epoch.to_csv(
            os.path.join(args.save_dir, f"npr_epoch_{global_e}.csv"),
            index=False,
            float_format="%.6f",
        )

        row_summary = {"epoch": global_e, **per_epoch_stats[global_e]}
        rows_npr.append(row_summary)

    # save NPR summary
    df_npr_summary = pd.DataFrame(rows_npr).sort_values("epoch")
    df_npr_summary.to_csv(
        os.path.join(args.save_dir, "npr_summary.csv"),
        index=False,
        float_format="%.6f",
    )
    with open(os.path.join(args.save_dir, "npr_summary.json"), "w") as f:
        json.dump(per_epoch_stats, f, indent=2)

    # global NPR aggregation
    means = [s["mean"] for s in per_epoch_stats.values()]
    overall_unweighted = float(np.mean(means))
    total_N = sum(s["N"] for s in per_epoch_stats.values())
    overall_weighted = float(
        sum(s["mean"] * s["N"] for s in per_epoch_stats.values())
        / max(1, total_N)
    )

    print(f"Unweighted mean of epoch means: {overall_unweighted:.4f}")
    print(f"Weighted by sample count (pooled): {overall_weighted:.4f}")

    # 2) Per-sample motion consistency -----------------------------------
    print("\n===== 2) Motion–semantic consistency (per sample) =====")
    df_samples, stats_motion = per_sample_motion_consistency(Y_ld, X_hd)

    df_samples.to_csv(
        os.path.join(args.save_dir, "per_sample_metrics.csv"),
        index=False,
        float_format="%.6f",
    )
    with open(os.path.join(args.save_dir, "motion_summary.json"), "w") as f:
        json.dump({"per_sample": stats_motion}, f, indent=2)

    print("[Info] Motion consistency summary:")
    for k, v in stats_motion.items():
        print(
            f"{k:>30s} | "
            f"mean={v['mean']:.4f} std={v['std']:.4f} "
            f"median={v['median']:.4f} min={v['min']:.4f} max={v['max']:.4f}"
        )

    # 3) Neighbor-distance change preservation ---------------------------
    print("\n===== 3) Neighbor-distance change preservation =====")
    # If hd_metric is 'manifold', neighbor-change still uses Euclidean
    hd_metric_nc = args.hd_metric if args.hd_metric != "manifold" else "euclidean"

    df_nc_pairs = neighbor_change_preservation_over_time_fast(
        X_hd,
        Y_ld,
        k=args.k,
        hd_metric=hd_metric_nc,
        ld_metric=args.ld_metric,
        mode=args.chg_mode,
        eps=args.chg_eps,
    )

    df_nc_pairs.to_csv(
        os.path.join(args.save_dir, "neighbor_change_pairs.csv"),
        index=False,
        float_format="%.6f",
    )

    nc_summary = {
        col: {
            "mean": float(df_nc_pairs[col].mean()),
            "std": float(df_nc_pairs[col].std()),
            "min": float(df_nc_pairs[col].min()),
            "max": float(df_nc_pairs[col].max()),
            "median": float(df_nc_pairs[col].median()),
        }
        for col in [
            "sign_agreement_mean",
            "delta_pearson_mean",
            "delta_spearman_mean",
        ]
    }

    with open(
        os.path.join(args.save_dir, "neighbor_change_overall.json"), "w"
    ) as f:
        json.dump(nc_summary, f, indent=2)

    print("[Info] Neighbor-change summary:")
    for k, v in nc_summary.items():
        print(
            f"{k:>28s} | "
            f"mean={v['mean']:.4f} std={v['std']:.4f} "
            f"median={v['median']:.4f} min={v['min']:.4f} max={v['max']:.4f}"
        )

    print(f"\n[Done] All outputs saved to: {args.save_dir}")


if __name__ == "__main__":
    main()
