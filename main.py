#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse
import torch

from scripts.hparams import HParams
from scripts.train_motion import main as train_motion_main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DynaVis-style motion visualization training script."
    )

    # Basic dims & training schedule
    parser.add_argument("--D", type=int, default=472)
    parser.add_argument("--d", type=int, default=2)
    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--lr_ae", type=float, default=1e-3)
    parser.add_argument("--lr_joint", type=float, default=5e-4)
    parser.add_argument("--epochs_ae", type=int, default=20)
    parser.add_argument("--epochs_joint", type=int, default=20)

    # Loss weights
    parser.add_argument("--lambda_rec", type=float, default=1.0)
    parser.add_argument("--lambda_dir", type=float, default=4.0)
    parser.add_argument("--lambda_rank", type=float, default=0.8)

    # Direction consistency
    parser.add_argument("--dir_windows", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--dir_betas", type=float, nargs="+", default=[0.5, 0.5, 0.5])
    parser.add_argument("--dir_min_step_norm", type=float, default=1e-4)

    # Ranking margins
    parser.add_argument("--rank_margin_top_order", type=float, default=0.02)
    parser.add_argument("--rank_margin_top_vs_rest", type=float, default=0.01)

    # KL / temperature schedule
    parser.add_argument("--kl_weight", type=float, default=0.5)
    parser.add_argument("--kl_tau_start", type=float, default=0.7)
    parser.add_argument("--kl_tau_end", type=float, default=0.25)
    parser.add_argument(
        "--kl_weighted_by_p",
        action="store_true",
        default=True,
        help="Enable step-wise weighting of KL by target distribution p.",
    )

    # Training details
    parser.add_argument("--warmup_epochs", type=int, default=16)
    parser.add_argument("--grad_clip", type=float, default=0.5)

    # L2 regularization toggle (this is your new hyper-parameter)
    parser.add_argument(
        "--use_l2",
        action="store_true",
        default=False,
        help="Enable L2 regularization on encoder/decoder in both Stage 1 and Stage 2.",
    )

    # Normalization
    parser.add_argument(
        "--norm_mode",
        type=str,
        default="robust",
        choices=["global", "anchor0", "robust", "per_epoch", "center_only"],
    )
    parser.add_argument("--std_clip_low", type=float, default=1e-8)
    parser.add_argument("--std_clip_high", type=float, default=0.0)

    # Paths
    parser.add_argument(
        "--data_path",
        type=str,
        default=(
            "/inspire/hdd/global_user/"
            "liuyiming-240108540153/training_dynamic/object_detection/"
            "voc/selected_subset/stacked_train_embeddings.npy"
        ),
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default=(
            "/inspire/hdd/global_user/"
            "liuyiming-240108540153/training_dynamic/object_detection/"
            "voc/selected_subset/simple_motion_ckpts"
        ),
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    h = HParams(
        D=args.D,
        d=args.d,
        bs=args.bs,
        lr_ae=args.lr_ae,
        lr_joint=args.lr_joint,
        epochs_ae=args.epochs_ae,
        epochs_joint=args.epochs_joint,
        lambda_rec=args.lambda_rec,
        lambda_dir=args.lambda_dir,
        lambda_rank=args.lambda_rank,
        dir_windows=tuple(args.dir_windows),
        dir_betas=tuple(args.dir_betas),
        dir_min_step_norm=args.dir_min_step_norm,
        rank_margin_top_order=args.rank_margin_top_order,
        rank_margin_top_vs_rest=args.rank_margin_top_vs_rest,
        kl_weight=args.kl_weight,
        kl_tau_start=args.kl_tau_start,
        kl_tau_end=args.kl_tau_end,
        kl_weighted_by_p=args.kl_weighted_by_p,
        warmup_epochs=args.warmup_epochs,
        grad_clip=args.grad_clip,
        norm_mode=args.norm_mode,
        std_clip_low=args.std_clip_low,
        std_clip_high=args.std_clip_high,
        data_path=args.data_path,
        ckpt_dir=args.ckpt_dir,
        device="cuda" if torch.cuda.is_available() else "cpu",
        use_l2=args.use_l2,
    )

    train_motion_main(h)
