#!/usr/bin/env python3
from __future__ import annotations
import argparse, os, math, random
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ===================== 全局加速设置 =====================
torch.backends.cudnn.benchmark = True
try:
    torch.set_float32_matmul_precision('high')  # 允许 TF32
except Exception:
    pass

def maybe_compile(m):
    try:
        return torch.compile(m)
    except Exception:
        return m

def amp_capabilities():
    # bfloat16优先；若不可用，退回fp16+GradScaler；都不行则禁用AMP
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available()
    return use_bf16, use_fp16

USE_BF16, USE_FP16 = amp_capabilities()
AMP_DTYPE = torch.bfloat16 if USE_BF16 else (torch.float16 if USE_FP16 else None)
SCALER = None if (AMP_DTYPE is None or AMP_DTYPE == torch.bfloat16) else torch.cuda.amp.GradScaler()

# ===================== 基础工具 =====================
def seed_all(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def exists(x): return x is not None

@dataclass
class HParams:
    D: int = 20; d: int = 2; T: int = 64; N: int = 512; bs: int = 1024
    lr: float = 2e-3
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    lambda_push: float = 1.0; lambda_pull: float = 1.0; lambda_rec: float = 1.0
    lip_reg: float = 0.0
    # 规范化
    norm_mode: str = "robust"
    std_clip_low: float = 1e-8
    std_clip_high: float = 0.0
    # 副损
    use_calib: bool = False
    use_sat: bool = False
    use_rank: bool = False
    lambda_calib: float = 0.3
    lambda_sat: float = 0.05
    lambda_rank: float = 0.2
    sat_margin: float = 0.95
    rank_margin: float = 0.05
    rank_pairs: int = 256

# ===================== 模型 & JVP =====================
def time_embed(t: torch.Tensor, dim: int = 32) -> torch.Tensor:
    device = t.device; half = dim // 2
    freqs = torch.exp(torch.linspace(math.log(1e-4), math.log(1.0), steps=half, device=device))
    ang = t[:, None] * freqs[None, :] * 2 * math.pi
    emb = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)
    if dim % 2 == 1: emb = F.pad(emb, (0, 1))
    return emb

def make_mlp(in_dim: int, out_dim: int, hidden: int = 256, depth: int = 3, act: str = 'silu') -> nn.Sequential:
    Act = {'relu': nn.ReLU, 'gelu': nn.GELU, 'silu': nn.SiLU, 'tanh': nn.Tanh}[act]
    layers=[]; d=in_dim
    for _ in range(depth): layers += [nn.Linear(d, hidden), Act()]; d=hidden
    layers += [nn.Linear(d, out_dim)]
    return nn.Sequential(*layers)

# 原逐样本JVP（保留作后备）
def jvp_wrt_x_with_t(func_xt, x: torch.Tensor, v: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    from torch.autograd.functional import jvp
    outs=[]
    for xb, vb, tb in zip(x, v, t):
        tb = tb.view(1)
        _, j = jvp(lambda _x: func_xt(_x.unsqueeze(0), tb).squeeze(0), (xb,), (vb,), create_graph=True)
        outs.append(j)
    return torch.stack(outs, dim=0)

# 更快的批量JVP（可用则优先）
def jvp_wrt_x_with_t_fast(func_xt, x: torch.Tensor, v: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    try:
        from torch.func import jvp as func_jvp
    except Exception:
        return jvp_wrt_x_with_t(func_xt, x, v, t)

    def f_wrapped(x_):
        return func_xt(x_, t)
    # func_jvp 对整批做 JVP
    _, jvp_out = func_jvp(f_wrapped, (x,), (v,))
    return jvp_out

class Encoder(nn.Module):
    def __init__(self, dim_x: int, dim_y: int, t_dim: int = 32, hidden: int = 256, depth: int = 3):
        super().__init__(); self.net = make_mlp(dim_x + t_dim, dim_y, hidden=hidden, depth=depth); self.t_dim = t_dim
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, time_embed(t, self.t_dim)], dim=-1))

class Decoder(nn.Module):
    def __init__(self, dim_y: int, dim_x: int, t_dim: int = 32, hidden: int = 256, depth: int = 3):
        super().__init__(); self.net = make_mlp(dim_y + t_dim, dim_x, hidden=hidden, depth=depth); self.t_dim = t_dim
    def forward(self, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([y, time_embed(t, self.t_dim)], dim=-1))

class LowDimVF(nn.Module):
    def __init__(self, dim_y: int, t_dim: int = 32, hidden: int = 256, depth: int = 3):
        super().__init__(); self.net = make_mlp(dim_y + t_dim, dim_y, hidden=hidden, depth=depth); self.t_dim = t_dim
    def forward(self, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([y, time_embed(t, self.t_dim)], dim=-1))

# ===================== 数据与差分 =====================
def finite_diff_xdot(X: np.ndarray, t: np.ndarray) -> np.ndarray:
    T,N,D = X.shape; Xdot = np.zeros_like(X, dtype=np.float32)
    for k in range(1, T-1):
        dt = (t[k+1] - t[k-1]); Xdot[k] = (X[k+1] - X[k-1]) / dt
    Xdot[0]  = (X[1]  - X[0])  / (t[1]  - t[0])
    Xdot[-1] = (X[-1] - X[-2]) / (t[-1] - t[-2])
    return Xdot

def load_or_generate(path: str):
    arr = np.load(path)
    X = arr.astype(np.float32) if isinstance(arr, np.ndarray) else arr["X"].astype(np.float32)
    t = np.arange(X.shape[0], dtype=np.float32)
    Xdot = finite_diff_xdot(X, t)
    print(f"Loaded: X{X.shape}, t{t.shape}, Xdot{Xdot.shape}")
    return X, t, Xdot

# ===================== 标准化 =====================
def _clip_std(std: np.ndarray, low: float, high: float) -> np.ndarray:
    std = np.maximum(std, low)
    if high and high > 0: std = np.minimum(std, high)
    return std

def compute_stats_for_norm(X_raw: np.ndarray, t_raw: np.ndarray,
                           mode: str = "robust",
                           std_clip_low: float = 1e-8,
                           std_clip_high: float = 0.0) -> Dict[str, np.ndarray]:
    T,N,D = X_raw.shape
    stats: Dict[str, np.ndarray|str|float] = {}
    stats["mode"] = np.array(mode)
    stats["t_min"] = np.array(float(np.min(t_raw)))
    stats["t_max"] = np.array(float(np.max(t_raw)))

    if mode == "per_epoch":
        mean_TD = X_raw.mean(axis=1)
        std_TD  = _clip_std(X_raw.std(axis=1), std_clip_low, std_clip_high)
        stats["mean_per_epoch"] = mean_TD.astype(np.float32)
        stats["std_per_epoch"]  = std_TD.astype(np.float32)
    elif mode == "anchor0":
        mean = X_raw[0].mean(axis=0)
        std  = _clip_std(X_raw[0].std(axis=0), std_clip_low, std_clip_high)
        stats["mean"] = mean.astype(np.float32); stats["std"]  = std.astype(np.float32)
    elif mode == "robust":
        q25 = np.percentile(X_raw, 25, axis=(0,1))
        q75 = np.percentile(X_raw, 75, axis=(0,1))
        median = np.median(X_raw, axis=(0,1))
        iqr = q75 - q25
        robust_std = _clip_std(iqr / 1.349, std_clip_low, std_clip_high)
        stats["mean"] = median.astype(np.float32); stats["std"]  = robust_std.astype(np.float32)
    elif mode == "center_only":
        mean = X_raw.mean(axis=(0,1))
        stats["mean"] = mean.astype(np.float32); stats["std"]  = np.ones_like(mean, dtype=np.float32)
    else:
        mean = X_raw.mean(axis=(0,1))
        std  = _clip_std(X_raw.std(axis=(0,1)), std_clip_low, std_clip_high)
        stats["mean"] = mean.astype(np.float32); stats["std"]  = std.astype(np.float32)
    return stats

def apply_normalization_with_stats(X_raw: np.ndarray, Xdot_raw: np.ndarray, t_raw: np.ndarray,
                                   stats: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mode = str(stats["mode"])
    t_min = float(stats["t_min"]); t_max = float(stats["t_max"])
    den = (t_max - t_min) if (t_max > t_min) else 1.0
    t = ((t_raw - t_min) / den).astype(np.float32)

    if mode == "per_epoch":
        mean_TD = stats["mean_per_epoch"].astype(np.float32)
        std_TD  = stats["std_per_epoch"].astype(np.float32)
        X    = ((X_raw - mean_TD[:, None, :]) / std_TD[:, None, :]).astype(np.float32)
        Xdot = (Xdot_raw / std_TD[:, None, :]).astype(np.float32)
    else:
        mean = stats["mean"].astype(np.float32)
        std  = stats["std"].astype(np.float32)
        X    = ((X_raw - mean) / std).astype(np.float32)
        Xdot = (Xdot_raw / std).astype(np.float32)
    return X, Xdot, t.astype(np.float32)

# ===================== 数据集 =====================
class PairDataset(Dataset):
    def __init__(self, X: np.ndarray, t: np.ndarray, Xdot: Optional[np.ndarray] = None):
        assert X.ndim == 3 and t.ndim == 1 and X.shape[0] == t.shape[0]
        self.X = X.astype(np.float32); self.t = t.astype(np.float32)
        if Xdot is None: Xdot = finite_diff_xdot(X, t)
        self.Xdot = Xdot.astype(np.float32)
        self.T, self.N, self.D = X.shape
        self.idxs = [(k, n) for k in range(self.T) for n in range(self.N)]
    def __len__(self): return len(self.idxs)
    def __getitem__(self, idx):
        k, n = self.idxs[idx]
        return {'x_t': torch.from_numpy(self.X[k, n]),
                'x_dot': torch.from_numpy(self.Xdot[k, n]),
                't': torch.tensor(self.t[k], dtype=torch.float32),
                'n': torch.tensor(n, dtype=torch.long)}

# ===================== 端点缓存 =====================
@dataclass
class EndpointCache:
    X0: torch.Tensor  # [N, D]
    XT: torch.Tensor  # [N, D]
    t0: float
    tT: float

def build_endpoint_cache(X: np.ndarray, t: np.ndarray, device: str) -> EndpointCache:
    X0 = torch.from_numpy(X[0]).to(device)
    XT = torch.from_numpy(X[-1]).to(device)
    t0 = float(t[0]); tT = float(t[-1])
    return EndpointCache(X0=X0, XT=XT, t0=t0, tT=tT)

# ===================== 副损与工具 =====================
def cosine(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    a_n = F.normalize(a, dim=-1, eps=eps)
    b_n = F.normalize(b, dim=-1, eps=eps)
    return (a_n * b_n).sum(dim=-1)

def pairwise_rank_loss(
    c_lo: torch.Tensor,
    c_hi: torch.Tensor,
    num_pairs: int = 256,
    margin: float = 0.05,
    min_gap: float = 0.02,
    hard_ratio: float = 0.5,
) -> torch.Tensor:
    device = c_lo.device
    B = c_lo.numel()
    if B < 2:
        return torch.zeros((), device=device)
    c_lo = c_lo.clamp(-1.0, 1.0)
    c_hi = c_hi.clamp(-1.0, 1.0)
    k_hard = int(num_pairs * hard_ratio); k_rand = num_pairs - k_hard
    order = torch.argsort(c_hi, dim=0)
    m = max(1, min(B // 4, k_hard))
    bottom_idx = order[:m]; top_idx = order[-m:]
    if top_idx.numel() != bottom_idx.numel():
        m = min(top_idx.numel(), bottom_idx.numel())
        top_idx = top_idx[-m:]; bottom_idx = bottom_idx[:m]
    p_hard = top_idx; q_hard = bottom_idx
    if k_rand > 0:
        cand = max(2 * k_rand, 1)
        idx = torch.randint(0, B, (2 * cand,), device=device)
        p_all = idx[:cand]; q_all = idx[cand:]
        diff = (c_hi[p_all] - c_hi[q_all])
        mask = diff.abs() > min_gap
        if mask.any():
            p_rand = p_all[mask][:k_rand]; q_rand = q_all[mask][:k_rand]
        else:
            need = k_rand
            rep_top = top_idx.repeat((need + m - 1) // m)[:need]
            rep_bot = bottom_idx.repeat((need + m - 1) // m)[:need]
            p_rand, q_rand = rep_top, rep_bot
    else:
        p_rand = torch.empty(0, dtype=torch.long, device=device)
        q_rand = torch.empty(0, dtype=torch.long, device=device)
    p = torch.cat([p_hard, p_rand], dim=0); q = torch.cat([q_hard, q_rand], dim=0)
    if p.numel() == 0: return torch.zeros((), device=device)
    s = c_hi[p] - c_hi[q]; sign = torch.sign(s)
    mask = (sign != 0).float()
    d_lo = c_lo[p] - c_lo[q]
    loss_vec = F.relu(margin - sign * d_lo) * mask
    denom = mask.sum().clamp_min(1.0)
    return loss_vec.sum() / denom

# ===================== Loss / 训练阶段 =====================
def joint_losses(
    f: Encoder, g: Decoder | None, W: LowDimVF, batch,
    lambda_push=1.0, lambda_pull=1.0, lambda_rec=1.0, lip_reg=0.0,
    ep_cache: Optional[EndpointCache] = None,
    use_calib: bool = False, lambda_calib: float = 0.3,
    use_sat: bool = False, lambda_sat: float = 0.05, sat_margin: float = 0.95,
    use_rank: bool = False, lambda_rank: float = 0.2, rank_pairs: int = 256, rank_margin: float = 0.05,
    **_
):
    x_t = batch['x_t']; t = batch['t']; Vx = batch['x_dot']; n_idx = batch['n'].long()

    # AMP 区域下做前向与JVP
    if AMP_DTYPE is None:
        y = f(x_t, t)
        JfV = jvp_wrt_x_with_t_fast(f, x_t, Vx, t)
        Wy  = W(y, t)
        L_push = F.mse_loss(JfV, Wy)
        L_pull = torch.tensor(0.0, device=x_t.device); L_rec = torch.tensor(0.0, device=x_t.device)
        if exists(g):
            JgW = jvp_wrt_x_with_t_fast(g, y,  Wy, t)
            L_pull = F.mse_loss(Vx, JgW)
            L_rec  = F.mse_loss(g(y, t), x_t)
    else:
        with torch.cuda.amp.autocast(dtype=AMP_DTYPE):
            y = f(x_t, t)
            JfV = jvp_wrt_x_with_t_fast(f, x_t, Vx, t)
            Wy  = W(y, t)
            L_push = F.mse_loss(JfV, Wy)
            L_pull = torch.tensor(0.0, device=x_t.device); L_rec = torch.tensor(0.0, device=x_t.device)
            if exists(g):
                JgW = jvp_wrt_x_with_t_fast(g, y,  Wy, t)
                L_pull = F.mse_loss(Vx, JgW)
                L_rec  = F.mse_loss(g(y, t), x_t)

    L_lip = torch.tensor(0.0, device=x_t.device)
    if lip_reg > 0:
        u  = torch.randn_like(x_t)
        Ju = jvp_wrt_x_with_t_fast(f, x_t, u, t)
        L_lip = (Ju.norm(dim=-1) / (u.norm(dim=-1) + 1e-8)).mean()

    L_calib = torch.tensor(0.0, device=x_t.device)
    L_sat   = torch.tensor(0.0, device=x_t.device)
    L_rank  = torch.tensor(0.0, device=x_t.device)

    if (use_calib or use_sat or use_rank) and (ep_cache is not None):
        X0_n = ep_cache.X0[n_idx]
        XT_n = ep_cache.XT[n_idx]
        t0   = torch.full((x_t.size(0),), ep_cache.t0, device=x_t.device, dtype=t.dtype)
        tT   = torch.full((x_t.size(0),), ep_cache.tT, device=x_t.device, dtype=t.dtype)
        with torch.no_grad():
            y0 = f(X0_n, t0); yT = f(XT_n, tT)
            e_lo = (yT - y0)
        e_hi = (XT_n - X0_n)
        c_hi = cosine(Vx, e_hi)
        c_lo = cosine(Wy, e_lo)
        if use_calib:
            L_calib = F.smooth_l1_loss(c_lo, c_hi)
        if use_sat:
            L_sat = F.relu(c_lo.abs() - sat_margin).pow(2).mean()
        if use_rank:
            L_rank = pairwise_rank_loss(c_lo, c_hi, num_pairs=rank_pairs, margin=rank_margin)

    loss = (lambda_push*L_push + lambda_pull*L_pull + lambda_rec*L_rec + lip_reg*L_lip
            + (lambda_calib * L_calib if use_calib else 0.0)
            + (lambda_sat   * L_sat   if use_sat   else 0.0)
            + (lambda_rank  * L_rank  if use_rank  else 0.0))

    return {
        'loss': loss,
        'L_push': L_push.detach(), 'L_pull': L_pull.detach(),
        'L_rec': L_rec.detach(),   'L_lip':  L_lip.detach(),
        'L_calib': L_calib.detach(), 'L_sat': L_sat.detach(), 'L_rank': L_rank.detach()
    }

def lipschitz_penalty(f, x, t, subset: int | None = 512):
    if subset is not None and x.size(0) > subset:
        idx = torch.randperm(x.size(0), device=x.device)[:subset]
        x = x[idx]; t = t[idx]
    u  = torch.randn_like(x); Ju = jvp_wrt_x_with_t_fast(f, x, u, t)
    return (Ju.norm(dim=-1) / (u.norm(dim=-1) + 1e-8)).mean()

# ===================== 三阶段（优化版） =====================
def _adamw_fused_supported():
    try:
        return torch.cuda.is_available()
    except Exception:
        return False

def stage1_pretrain_ae(f, g, dl, device, epochs=8, lr_f=5e-4, lr_g=1e-3,
                       lip_max=1e-3, lip_warmup=3, lip_every=2, lip_subset=256):
    f = maybe_compile(f); g = maybe_compile(g)
    opt = torch.optim.AdamW([
        {'params': f.parameters(), 'lr': lr_f},
        {'params': g.parameters(), 'lr': lr_g},
    ], weight_decay=0.0, fused=_adamw_fused_supported())
    f.train(); g.train()
    for epoch in range(1, epochs+1):
        lam_lip = lip_max * min(1.0, epoch / max(1, lip_warmup))
        acc = {'rec':0.0, 'lip':0.0, 'n':0}
        pbar = tqdm(dl, desc=f'[AE] epoch {epoch}/{epochs}')
        for step, batch in enumerate(pbar):
            for k in batch: batch[k] = batch[k].to(device, non_blocking=True)
            x_t, t = batch['x_t'], batch['t']
            if AMP_DTYPE is None:
                y = f(x_t, t); L_rec = F.mse_loss(g(y, t), x_t)
                if lam_lip > 0 and (step % lip_every == 0):
                    L_lip = lipschitz_penalty(f, x_t, t, subset=lip_subset)
                else: L_lip = torch.tensor(0.0, device=device)
                lam_lip_effective = 0.0  # 你原先就置0，这里保持一致
                loss = L_rec + lam_lip_effective * L_lip
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            else:
                with torch.cuda.amp.autocast(dtype=AMP_DTYPE):
                    y = f(x_t, t); L_rec = F.mse_loss(g(y, t), x_t)
                    if lam_lip > 0 and (step % lip_every == 0):
                        L_lip = lipschitz_penalty(f, x_t, t, subset=lip_subset)
                    else: L_lip = torch.tensor(0.0, device=device)
                    lam_lip_effective = 0.0
                    loss = L_rec + lam_lip_effective * L_lip
                opt.zero_grad(set_to_none=True)
                if SCALER is None: loss.backward(); opt.step()
                else: SCALER.scale(loss).backward(); SCALER.step(opt); SCALER.update()

            acc['rec'] += L_rec.item(); acc['lip'] += L_lip.item(); acc['n'] += 1
            pbar.set_postfix(rec=acc['rec']/acc['n'])
        print(f"[AE] epoch {epoch}/{epochs}  L_rec={acc['rec']/acc['n']:.5f}  L_lip={acc['lip']/max(1,acc['n']):.5f}  λ_lip={lam_lip_effective:g}")

def stage2_train_W(f, g, W, dl, device, epochs=4, lr_W=2e-3):
    for p in f.parameters(): p.requires_grad = False
    for p in g.parameters(): p.requires_grad = False
    for p in W.parameters(): p.requires_grad = True
    f = maybe_compile(f); W = maybe_compile(W)
    opt = torch.optim.AdamW(W.parameters(), lr=lr_W, weight_decay=0.0, fused=_adamw_fused_supported())

    for epoch in range(1, epochs+1):
        W.train()
        acc_push, nstep = 0.0, 0
        pbar = tqdm(dl, desc=f'[W] epoch {epoch}/{epochs}')
        for batch in pbar:
            for k in batch: batch[k] = batch[k].to(device, non_blocking=True)
            x_t, t, Vx = batch['x_t'], batch['t'], batch['x_dot']

            if AMP_DTYPE is None:
                y = f(x_t, t)
                target_y_dot = jvp_wrt_x_with_t_fast(f, x_t, Vx, t)
                pred_y_dot   = W(y, t)
                L_push = F.mse_loss(pred_y_dot, target_y_dot)
                opt.zero_grad(set_to_none=True); L_push.backward(); opt.step()
            else:
                with torch.cuda.amp.autocast(dtype=AMP_DTYPE):
                    y = f(x_t, t)
                    target_y_dot = jvp_wrt_x_with_t_fast(f, x_t, Vx, t)
                    pred_y_dot   = W(y, t)
                    L_push = F.mse_loss(pred_y_dot, target_y_dot)
                opt.zero_grad(set_to_none=True)
                if SCALER is None: L_push.backward(); opt.step()
                else: SCALER.scale(L_push).backward(); SCALER.step(opt); SCALER.update()

            acc_push += L_push.item(); nstep += 1
            pbar.set_postfix(push=acc_push/nstep)
        print(f"[W] epoch {epoch}/{epochs}  L_push={acc_push/nstep:.5f}")

def stage3_joint_finetune(
    f, g, W, dl, device, ep_cache: EndpointCache,
    epochs=18, base_lrs=(2e-4, 5e-4, 1e-3),
    lambdas=(1.0, 1.0, 1.0),
    warmup=6, lip_max=1e-3, lip_warmup=6,
    use_calib=False, use_sat=False, use_rank=False,
    lambda_calib=0.3, lambda_sat=0.05, lambda_rank=0.2,
    sat_margin=0.95, rank_pairs=256, rank_margin=0.05,
    grad_accum_steps=1
):
    lr_f, lr_g, lr_W = base_lrs
    lam_push_t, lam_pull_t, lam_rec = lambdas
    for p in f.parameters(): p.requires_grad = True
    for p in g.parameters(): p.requires_grad = True
    for p in W.parameters(): p.requires_grad = True

    f = maybe_compile(f); g = maybe_compile(g); W = maybe_compile(W)
    opt = torch.optim.AdamW([
        {'params': f.parameters(), 'lr': lr_f},
        {'params': g.parameters(), 'lr': lr_g},
        {'params': W.parameters(), 'lr': lr_W},
    ], weight_decay=0.0, fused=_adamw_fused_supported())

    for epoch in range(1, epochs+1):
        scale = min(1.0, epoch / max(1, warmup))
        lam_push = lam_push_t * scale
        lam_pull = lam_pull_t * scale
        lam_lip  = 0.0  # 你原代码已置0，保持一致

        f.train(); g.train(); W.train()
        meters = {k:0.0 for k in ['push','pull','rec','lip','calib','sat','rank']}
        nstep = 0

        pbar = tqdm(dl, desc=f'[Joint] epoch {epoch}/{epochs}')
        opt.zero_grad(set_to_none=True)
        for step, batch in enumerate(pbar, start=1):
            for k in batch: batch[k] = batch[k].to(device, non_blocking=True)

            if AMP_DTYPE is None:
                outs = joint_losses(
                    f, g, W, batch,
                    lambda_push=lam_push, lambda_pull=lam_pull, lambda_rec=lam_rec, lip_reg=lam_lip,
                    ep_cache=ep_cache,
                    use_calib=use_calib, lambda_calib=lambda_calib,
                    use_sat=use_sat, lambda_sat=lambda_sat, sat_margin=sat_margin,
                    use_rank=use_rank, lambda_rank=lambda_rank, rank_pairs=rank_pairs, rank_margin=rank_margin,
                    current_epoch=epoch
                )
                loss = outs['loss'] / grad_accum_steps
                loss.backward()
            else:
                with torch.cuda.amp.autocast(dtype=AMP_DTYPE):
                    outs = joint_losses(
                        f, g, W, batch,
                        lambda_push=lam_push, lambda_pull=lam_pull, lambda_rec=lam_rec, lip_reg=lam_lip,
                        ep_cache=ep_cache,
                        use_calib=use_calib, lambda_calib=lambda_calib,
                        use_sat=use_sat, lambda_sat=lambda_sat, sat_margin=sat_margin,
                        use_rank=use_rank, lambda_rank=lambda_rank, rank_pairs=rank_pairs, rank_margin=rank_margin,
                        current_epoch=epoch
                    )
                    loss = outs['loss'] / grad_accum_steps
                if SCALER is None: loss.backward()
                else: SCALER.scale(loss).backward()

            if step % grad_accum_steps == 0:
                if SCALER is None:
                    opt.step()
                else:
                    SCALER.step(opt); SCALER.update()
                opt.zero_grad(set_to_none=True)

            for k in meters: meters[k] += outs[f"L_{k}"].item()
            nstep += 1
            pbar.set_postfix({k: meters[k]/nstep for k in meters})

        print(f"[Joint] epoch {epoch}/{epochs} | λpush={lam_push:g} λpull={lam_pull:g} λrec={lam_rec:g} "
              + " ".join([f"L_{k}={meters[k]/nstep:.5f}" for k in meters]))

# ===================== 推理导出（原样） =====================
@torch.no_grad()
def save_predicted_Y_raw(
    f,
    X_raw: np.ndarray,
    t_raw: np.ndarray,
    device: str,
    out_npz: str,
    per_epoch_dir: str,
    batch_size: int,
    stats_path: str,
    assume_normalized: bool = False,
    clip_t: bool = False,
):
    f.eval()
    if not assume_normalized and os.path.exists(stats_path):
        stats = np.load(stats_path, allow_pickle=True); stats = {k: stats[k] for k in stats.files}
        Xdot_raw = finite_diff_xdot(X_raw, t_raw)
        X, _, t = apply_normalization_with_stats(X_raw, Xdot_raw, t_raw, stats)
        if clip_t: t = np.clip(t, 0.0, 1.0)
    else:
        X = X_raw.astype(np.float32); t = t_raw.astype(np.float32)

    T,N,D = X.shape
    dummy_x = torch.from_numpy(X[0, :1]).to(device)
    dummy_t = torch.tensor([float(t[0])], device=device)
    d = int(f(dummy_x, dummy_t).shape[-1])
    Y = np.empty((T, N, d), dtype=np.float32)

    for k in range(T):
        xk = torch.from_numpy(np.ascontiguousarray(X[k])).to(device)
        tk = torch.full((N,), float(t[k]), device=device)
        for s in range(0, N, batch_size):
            xe = xk[s:s+batch_size]; te = tk[s:s+batch_size]
            if AMP_DTYPE is None:
                Y[k, s:s+batch_size] = f(xe, te).cpu().numpy().astype(np.float32)
            else:
                with torch.cuda.amp.autocast(dtype=AMP_DTYPE):
                    Y[k, s:s+batch_size] = f(xe, te).float().cpu().numpy().astype(np.float32)
        if per_epoch_dir:
            os.makedirs(per_epoch_dir, exist_ok=True)
            np.save(os.path.join(per_epoch_dir, f"epoch_{k+1}_embedding.npy"), Y[k])

    os.makedirs(os.path.dirname(out_npz) or ".", exist_ok=True)
    np.savez(out_npz, Y_ld=Y, t=t.astype(np.float32))
    print(f"[Saved] low-dim embeddings: {out_npz}")
    if per_epoch_dir: print(f"[Saved] per-epoch embeddings → {per_epoch_dir}")
    return Y

# ===================== 可视化（可选） =====================
def visualize(f: Encoder, W: LowDimVF, X: np.ndarray, t: np.ndarray, out='viz_lowdim.png'):
    import matplotlib.pyplot as plt
    f.eval(); W.eval(); device = next(f.parameters()).device
    with torch.no_grad():
        T,N,D = X.shape
        d = f(torch.zeros(1, D, device=device), torch.zeros(1, device=device)).shape[-1]
        Y = np.zeros((T, N, d), dtype=np.float32)
        for k in range(T):
            xk = torch.from_numpy(X[k]).to(device)
            tk = torch.full((N,), float(t[k]), device=device)
            if AMP_DTYPE is None:
                Y[k] = f(xk, tk).cpu().numpy()
            else:
                with torch.cuda.amp.autocast(dtype=AMP_DTYPE):
                    Y[k] = f(xk, tk).float().cpu().numpy()
    if Y.shape[-1] == 2:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6,6))
        for n in range(min(100, Y.shape[1])):
            plt.plot(Y[:, n, 0], Y[:, n, 1], alpha=0.35, linewidth=1)
        plt.title('Projected trajectories (2D)'); plt.xlabel('y1'); plt.ylabel('y2')
        plt.axis('equal'); plt.tight_layout(); plt.savefig(out, dpi=160); plt.close()
    else:
        plt.figure(figsize=(6,6))
        plt.scatter(Y[:,:,0].flatten(), Y[:,:,1].flatten(), s=3, alpha=0.3)
        plt.title('Projected samples (show first two)'); plt.xlabel('y1'); plt.ylabel('y2')
        plt.axis('equal'); plt.tight_layout(); plt.savefig(out, dpi=160); plt.close()

# ===================== 主流程 =====================
def train(args: HParams):
    seed_all(0)
    data_path = "/inspire/hdd/global_user/liuyiming-240108540153/training_dynamic/text_models/TextRCNN-THUCNews/Classification/selected_subset/stacked_train_embeddings.npy"
    X_raw, t_raw, Xdot_raw = load_or_generate(path=data_path)

    ckpt_dir = "/inspire/hdd/global_user/liuyiming-240108540153/training_dynamic/text_models/TextRCNN-THUCNews/Classification/selected_subset/checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)
    stats = compute_stats_for_norm(X_raw, t_raw, mode=args.norm_mode,
                                   std_clip_low=args.std_clip_low, std_clip_high=args.std_clip_high)
    stats_path = os.path.join(ckpt_dir, f"norm_stats_{args.norm_mode}.npz")
    np.savez(stats_path, **stats)
    X, Xdot, t = apply_normalization_with_stats(X_raw, Xdot_raw, t_raw, stats)

    # DataLoader（更快的参数）
    num_workers = max(2, os.cpu_count() // 2) if torch.cuda.is_available() else 0
    ds = PairDataset(X, t, Xdot)
    dl = DataLoader(
        ds, batch_size=args.bs, shuffle=True, drop_last=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=(num_workers > 0), prefetch_factor=(4 if num_workers > 0 else None)
    )

    # 模型
    f = Encoder(args.D, args.d).to(args.device)
    g = Decoder(args.d, args.D).to(args.device)
    W = LowDimVF(args.d).to(args.device)

    # 端点缓存
    ep_cache = build_endpoint_cache(X, t, device=args.device)

    # 三阶段训练（与原逻辑一致）
    stage1_pretrain_ae(f, g, dl, args.device,
                       epochs=8, lr_f=5e-4, lr_g=1e-3,
                       lip_max=1e-3, lip_warmup=3, lip_every=2, lip_subset=256)
    stage2_train_W(f, g, W, dl, args.device, epochs=4, lr_W=2e-3)
    stage3_joint_finetune(
        f, g, W, dl, args.device, ep_cache,
        epochs=18, base_lrs=(2e-4, 5e-4, 1e-3),
        lambdas=(args.lambda_push, args.lambda_pull, args.lambda_rec),
        warmup=6, lip_max=1e-3, lip_warmup=6,
        use_calib=args.use_calib, use_sat=args.use_sat, use_rank=args.use_rank,
        lambda_calib=args.lambda_calib, lambda_sat=args.lambda_sat, lambda_rank=args.lambda_rank,
        sat_margin=args.sat_margin, rank_pairs=args.rank_pairs, rank_margin=args.rank_margin
    )

    # 导出 / 可视化
    save_predicted_Y_raw(
        f=f, X_raw=X_raw, t_raw=t_raw, device=args.device,
        out_npz=os.path.join(ckpt_dir, f"Y_ld_{args.norm_mode}.npz"),
        per_epoch_dir=os.path.join(ckpt_dir, f"embeddings_{args.norm_mode}"),
        batch_size=2048, stats_path=stats_path, assume_normalized=False, clip_t=False
    )
    try:
        visualize(f, W, X, t, out=os.path.join(ckpt_dir, f'viz_lowdim_{args.norm_mode}.png'))
    except Exception as e:
        print("Visualization failed:", e)

# ===================== CLI =====================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--D', type=int, default=812)
    parser.add_argument('--d', type=int, default=2)
    parser.add_argument('--T', type=int, default=50)
    parser.add_argument('--N', type=int, default=500)
    parser.add_argument('--bs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--lambda_push', type=float, default=1.0)
    parser.add_argument('--lambda_pull', type=float, default=1.0)
    parser.add_argument('--lambda_rec', type=float, default=1.0)
    parser.add_argument('--lip_reg', type=float, default=1e-3)
    parser.add_argument('--norm_mode', type=str, default='robust',
                        choices=['global','anchor0','robust','per_epoch','center_only'])
    parser.add_argument('--std_clip_low', type=float, default=1e-8)
    parser.add_argument('--std_clip_high', type=float, default=0.0)
    parser.add_argument('--use_calib', action='store_true')
    parser.add_argument('--use_sat', action='store_true')
    parser.add_argument('--use_rank', action='store_true')
    parser.add_argument('--lambda_calib', type=float, default=0.3)
    parser.add_argument('--lambda_sat', type=float, default=0.05)
    parser.add_argument('--lambda_rank', type=float, default=0.2)
    parser.add_argument('--sat_margin', type=float, default=0.95)
    parser.add_argument('--rank_margin', type=float, default=0.05)
    parser.add_argument('--rank_pairs', type=int, default=256)

    args = parser.parse_args()

    h = HParams(D=args.D, d=args.d, T=args.T, N=args.N, bs=args.bs, lr=args.lr,
                lambda_push=args.lambda_push, lambda_pull=args.lambda_pull,
                lambda_rec=args.lambda_rec, lip_reg=args.lip_reg,
                device='cuda' if torch.cuda.is_available() else 'cpu',
                norm_mode=args.norm_mode, std_clip_low=args.std_clip_low, std_clip_high=args.std_clip_high,
                use_calib=args.use_calib, use_sat=args.use_sat, use_rank=args.use_rank,
                lambda_calib=args.lambda_calib, lambda_sat=args.lambda_sat, lambda_rank=args.lambda_rank,
                sat_margin=args.sat_margin, rank_pairs=args.rank_pairs, rank_margin=args.rank_margin)
    train(h)


# python train_jvp_modi.py --norm_mode robust --use_calib --use_sat --use_rank