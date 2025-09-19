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
    # 新增：三个对齐/排序副损的权重与开关
    use_calib: bool = False
    use_sat: bool = False
    use_rank: bool = False
    lambda_calib: float = 0.3   # 相对 push 的建议：先设 0.3×push
    lambda_sat: float = 0.05    # 反饱和
    lambda_rank: float = 0.2    # 排序一致
    sat_margin: float = 0.95    # |c_lo| 超过此阈值才惩罚
    rank_margin: float = 0.05   # pairwise hinge margin
    rank_pairs: int = 256       # 每步采样多少对 pair 计算排序损失

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

def jvp_wrt_x_with_t(func_xt, x: torch.Tensor, v: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    from torch.autograd.functional import jvp
    outs=[]
    for xb, vb, tb in zip(x, v, t):
        tb = tb.view(1)
        _, j = jvp(lambda _x: func_xt(_x.unsqueeze(0), tb).squeeze(0), (xb,), (vb,), create_graph=True)
        outs.append(j)
    return torch.stack(outs, dim=0)

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
        mean_TD = X_raw.mean(axis=1)          # [T,D]
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
    else: # global
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
        mean_TD = stats["mean_per_epoch"].astype(np.float32)  # [T,D]
        std_TD  = stats["std_per_epoch"].astype(np.float32)   # [T,D]
        X    = ((X_raw - mean_TD[:, None, :]) / std_TD[:, None, :]).astype(np.float32)
        Xdot = (Xdot_raw / std_TD[:, None, :]).astype(np.float32)
    else:
        mean = stats["mean"].astype(np.float32)  # [D]
        std  = stats["std"].astype(np.float32)   # [D]
        X    = ((X_raw - mean) / std).astype(np.float32)
        Xdot = (Xdot_raw / std).astype(np.float32)
    return X, Xdot, t.astype(np.float32)

# ===================== 数据集（原始：不含 neighbor） =====================
class PairDataset(Dataset):
    """
    返回:
      - x_t: [D]
      - x_dot: [D]
      - t: scalar
      - n: 该样本的轨迹编号（用于查参考向量）
    """
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

# ===================== 三项新损需要的端点缓存 =====================
@dataclass
class EndpointCache:
    X0: torch.Tensor  # [N, D]  标准化后的起点
    XT: torch.Tensor  # [N, D]  标准化后的终点
    t0: float
    tT: float

def build_endpoint_cache(X: np.ndarray, t: np.ndarray, device: str) -> EndpointCache:
    # 注意：X, t 已经是标准化后的
    X0 = torch.from_numpy(X[0]).to(device)      # [N, D]
    XT = torch.from_numpy(X[-1]).to(device)     # [N, D]
    t0 = float(t[0]); tT = float(t[-1])
    return EndpointCache(X0=X0, XT=XT, t0=t0, tT=tT)

# ===================== 新增：三项副损（在 joint_losses 内调用） =====================
def cosine(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    a_n = F.normalize(a, dim=-1, eps=eps)
    b_n = F.normalize(b, dim=-1, eps=eps)
    return (a_n * b_n).sum(dim=-1)

def pairwise_rank_loss(
    c_lo: torch.Tensor,
    c_hi: torch.Tensor,
    num_pairs: int = 256,
    margin: float = 0.05,
    min_gap: float = 0.02,       # ★ 新增：高维分数至少相差这么多才算有效约束
    hard_ratio: float = 0.5,     # ★ 新增：一半从“最容易分”的top-bottom挑，另一半随机采样过滤
) -> torch.Tensor:
    """
    更稳的 Spearman 代理：
    - 一半 pairs 用“极端对”（高维分数最高 vs 最低）保证 sign != 0；
    - 另一半随机 pairs，但过滤掉 |c_hi_i - c_hi_j| <= min_gap 的“无效对”；
    - 用 pairwise hinge: [margin - sign(c_hi_i - c_hi_j) * (c_lo_i - c_lo_j)]_+ 。
    """
    device = c_lo.device
    B = c_lo.numel()
    if B < 2:
        return torch.zeros((), device=device)

    # 保底 clamp，避免数值偶发 >1
    c_lo = c_lo.clamp(-1.0, 1.0)
    c_hi = c_hi.clamp(-1.0, 1.0)

    # ---- 1) “极端对”：top vs bottom，确保有有效对 ----
    k_hard = int(num_pairs * hard_ratio)
    k_rand = num_pairs - k_hard

    # 排序索引（按高维评分）
    order = torch.argsort(c_hi, dim=0)     # 升序
    # 两端各取 m 个
    m = max(1, min(B // 4, k_hard))        # 取四分之一或不超过需求
    bottom_idx = order[:m]
    top_idx    = order[-m:]

    # 和尚配对（长度一致）
    if top_idx.numel() != bottom_idx.numel():
        m = min(top_idx.numel(), bottom_idx.numel())
        top_idx = top_idx[-m:]
        bottom_idx = bottom_idx[:m]

    p_hard = top_idx
    q_hard = bottom_idx

    # ---- 2) 随机对 + 过滤小差值 ----
    # 随机采到 2*k_rand 个，然后过滤掉 |diff|<=min_gap，再截断到 k_rand
    if k_rand > 0:
        # 至少采足量候选
        cand = max(2 * k_rand, 1)
        idx = torch.randint(0, B, (2 * cand,), device=device)
        p_all = idx[:cand]
        q_all = idx[cand:]

        diff = (c_hi[p_all] - c_hi[q_all])
        mask = diff.abs() > min_gap
        if mask.any():
            p_rand = p_all[mask][:k_rand]
            q_rand = q_all[mask][:k_rand]
        else:
            # 兜底：如果全都太接近，就退化成再拿一些极端对或直接用 top/bottom 循环
            need = k_rand
            rep_top = top_idx.repeat((need + m - 1) // m)[:need]
            rep_bot = bottom_idx.repeat((need + m - 1) // m)[:need]
            p_rand, q_rand = rep_top, rep_bot
    else:
        p_rand = torch.empty(0, dtype=torch.long, device=device)
        q_rand = torch.empty(0, dtype=torch.long, device=device)

    # ---- 3) 拼接所有 pairs ----
    p = torch.cat([p_hard, p_rand], dim=0)
    q = torch.cat([q_hard, q_rand], dim=0)
    if p.numel() == 0:
        return torch.zeros((), device=device)

    # ---- 4) 计算 hinge 排序损失 ----
    # 高维的排序方向
    s = c_hi[p] - c_hi[q]
    sign = torch.sign(s)  # {-1, 0, 1}
    # 过滤相等（或近似相等）的对（极少数情况）
    mask = (sign != 0).float()
    # 低维的差
    d_lo = c_lo[p] - c_lo[q]
    # hinge:  希望 sign * d_lo >= margin
    loss_vec = F.relu(margin - sign * d_lo) * mask
    denom = mask.sum().clamp_min(1.0)
    return loss_vec.sum() / denom

# ===================== Loss / 训练阶段 =====================
def joint_losses(
    f: Encoder, g: Decoder | None, W: LowDimVF, batch,
    lambda_push=1.0, lambda_pull=1.0, lambda_rec=1.0, lip_reg=0.0,
    # 新增：端点缓存 & 三项副损的超参
    ep_cache: Optional[EndpointCache] = None,
    use_calib: bool = False, lambda_calib: float = 0.3,
    use_sat: bool = False, lambda_sat: float = 0.05, sat_margin: float = 0.95,
    use_rank: bool = False, lambda_rank: float = 0.2, rank_pairs: int = 256, rank_margin: float = 0.05,
    **_
):
    x_t = batch['x_t']; t = batch['t']; Vx = batch['x_dot']; n_idx = batch['n'].long()
    # 主干：push / pull / rec / lip
    y = f(x_t, t)                         # [B, d]
    JfV = jvp_wrt_x_with_t(f, x_t, Vx, t) # [B, d]  目标低维速度
    Wy  = W(y, t)                         # [B, d]  预测低维速度
    L_push = F.mse_loss(JfV, Wy)

    L_pull = torch.tensor(0.0, device=x_t.device); L_rec = torch.tensor(0.0, device=x_t.device)
    if exists(g):
        JgW = jvp_wrt_x_with_t(g, y,  Wy, t)
        L_pull = F.mse_loss(Vx, JgW)
        L_rec  = F.mse_loss(g(y, t), x_t)

    L_lip = torch.tensor(0.0, device=x_t.device)
    if lip_reg > 0:
        u  = torch.randn_like(x_t); Ju = jvp_wrt_x_with_t(f, x_t, u, t)
        L_lip = (Ju.norm(dim=-1) / (u.norm(dim=-1) + 1e-8)).mean()

    # ===== 三项副损（可选）=====
    L_calib = torch.tensor(0.0, device=x_t.device)
    L_sat   = torch.tensor(0.0, device=x_t.device)
    L_rank  = torch.tensor(0.0, device=x_t.device)

    if (use_calib or use_sat or use_rank) and (ep_cache is not None):
        # 端点连线（高维 & 低维）
        X0_n = ep_cache.X0[n_idx]     # [B, D]
        XT_n = ep_cache.XT[n_idx]     # [B, D]
        t0   = torch.full((x_t.size(0),), ep_cache.t0, device=x_t.device, dtype=t.dtype)
        tT   = torch.full((x_t.size(0),), ep_cache.tT, device=x_t.device, dtype=t.dtype)

        # 高维参考方向（起终点连线）
        e_hi = (XT_n - X0_n)          # [B, D]
        # 低维参考方向：用当前 f 编码端点
        with torch.no_grad():  # 参考向量不回传梯度，避免额外开销（也可去掉 no_grad）
            y0 = f(X0_n, t0)          # [B, d]
            yT = f(XT_n, tT)          # [B, d]
            e_lo = (yT - y0)          # [B, d]

        # 构造高/低维余弦分数
        c_hi = cosine(Vx, e_hi)       # 高维：速度 vs 端点连线
        c_lo = cosine(Wy, e_lo)       # 低维：预测速度 vs 端点连线（低维端点由 f 现算）

        if use_calib:
            # 方向回归校准（Huber 更稳）
            L_calib = F.smooth_l1_loss(c_lo, c_hi)

        if use_sat:
            # 反饱和（只对 |c_lo|>sat_margin 惩罚，避免到 ±1 附近堆积）
            L_sat = F.relu(c_lo.abs() - sat_margin).pow(2).mean()

        if use_rank:
            # 排序一致（pairwise hinge Spearman 代理）
            L_rank = pairwise_rank_loss(c_lo, c_hi, num_pairs=rank_pairs, margin=rank_margin)

    # 总损
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
    u  = torch.randn_like(x); Ju = jvp_wrt_x_with_t(f, x, u, t)
    return (Ju.norm(dim=-1) / (u.norm(dim=-1) + 1e-8)).mean()

def stage1_pretrain_ae(f, g, dl, device, epochs=8, lr_f=5e-4, lr_g=1e-3,
                       lip_max=1e-3, lip_warmup=3, lip_every=2, lip_subset=256):
    opt = torch.optim.AdamW([
        {'params': f.parameters(), 'lr': lr_f},
        {'params': g.parameters(), 'lr': lr_g},
    ], weight_decay=0.0)
    f.train(); g.train()
    for epoch in range(1, epochs+1):
        lam_lip = lip_max * min(1.0, epoch / max(1, lip_warmup))
        acc = {'rec':0.0, 'lip':0.0, 'n':0}
        for step, batch in enumerate(tqdm(dl, desc=f'[AE] epoch {epoch}/{epochs}')):
            for k in batch: batch[k] = batch[k].to(device)
            x_t, t = batch['x_t'], batch['t']
            y = f(x_t, t); L_rec = F.mse_loss(g(y, t), x_t)
            if lam_lip > 0 and (step % lip_every == 0):
                L_lip = lipschitz_penalty(f, x_t, t, subset=lip_subset)
            else: L_lip = torch.tensor(0.0, device=device)
            lam_lip = 0
            loss = L_rec + lam_lip * L_lip
            opt.zero_grad(); loss.backward(); opt.step()
            acc['rec'] += L_rec.item(); acc['lip'] += L_lip.item(); acc['n'] += 1
        print(f"[AE] epoch {epoch}/{epochs}  L_rec={acc['rec']/acc['n']:.5f}  L_lip={acc['lip']/max(1,acc['n']):.5f}  λ_lip={lam_lip:g}")

def stage2_train_W(f, g, W, dl, device, epochs=4, lr_W=2e-3):
    for p in f.parameters(): p.requires_grad = False
    for p in g.parameters(): p.requires_grad = False
    W.train()
    opt = torch.optim.AdamW(W.parameters(), lr=lr_W, weight_decay=0.0)
    for epoch in range(1, epochs+1):
        acc = {'push':0.0, 'n':0}
        for batch in tqdm(dl, desc=f'[W] epoch {epoch}/{epochs}'):
            for k in batch: batch[k] = batch[k].to(device)
            x_t, t, Vx = batch['x_t'], batch['t'], batch['x_dot']
            y = f(x_t, t)
            target_y_dot = jvp_wrt_x_with_t(f, x_t, Vx, t)
            pred_y_dot   = W(y, t)
            L_push = F.mse_loss(pred_y_dot, target_y_dot)
            opt.zero_grad(); L_push.backward(); opt.step()
            acc['push'] += L_push.item(); acc['n'] += 1
        print(f"[W] epoch {epoch}/{epochs}  L_push={acc['push']/acc['n']:.5f}")

def stage3_joint_finetune(
    f, g, W, dl, device, ep_cache: EndpointCache,
    epochs=18, base_lrs=(2e-4, 5e-4, 1e-3),
    lambdas=(1.0, 1.0, 1.0),
    warmup=6, lip_max=1e-3, lip_warmup=6,
    # 副损的开关/权重
    use_calib=False, use_sat=False, use_rank=False,
    lambda_calib=0.3, lambda_sat=0.05, lambda_rank=0.2,
    sat_margin=0.95, rank_pairs=256, rank_margin=0.05
):
    lr_f, lr_g, lr_W = base_lrs
    lam_push_t, lam_pull_t, lam_rec = lambdas
    for p in f.parameters(): p.requires_grad = True
    for p in g.parameters(): p.requires_grad = True
    for p in W.parameters(): p.requires_grad = True
    opt = torch.optim.AdamW([
        {'params': f.parameters(), 'lr': lr_f},
        {'params': g.parameters(), 'lr': lr_g},
        {'params': W.parameters(), 'lr': lr_W},
    ], weight_decay=0.0)

    for epoch in range(1, epochs+1):
        scale = min(1.0, epoch / max(1, warmup))
        lam_push = lam_push_t * scale
        lam_pull = lam_pull_t * scale
        lam_lip  = lip_max * min(1.0, epoch / max(1, lip_warmup))
        lam_lip = 0
        f.train(); g.train(); W.train()
        acc = {'push':0.0, 'pull':0.0, 'rec':0.0, 'lip':0.0, 'calib':0.0, 'sat':0.0, 'rank':0.0, 'n':0}
        for batch in tqdm(dl, desc=f'[Joint] epoch {epoch}/{epochs}'):
            for k in batch: batch[k] = batch[k].to(device)
            outs = joint_losses(
                f, g, W, batch,
                lambda_push=lam_push, lambda_pull=lam_pull, lambda_rec=lam_rec, lip_reg=lam_lip,
                ep_cache=ep_cache,
                use_calib=use_calib, lambda_calib=lambda_calib,
                use_sat=use_sat, lambda_sat=lambda_sat, sat_margin=sat_margin,
                use_rank=use_rank, lambda_rank=lambda_rank, rank_pairs=rank_pairs, rank_margin=rank_margin,
                current_epoch=epoch
            )
            loss = outs['loss']
            opt.zero_grad(); loss.backward(); opt.step()
            acc['push'] += outs['L_push'].item()
            acc['pull'] += outs['L_pull'].item()
            acc['rec']  += outs['L_rec'].item()
            acc['lip']  += outs['L_lip'].item()
            acc['calib']+= outs['L_calib'].item()
            acc['sat']  += outs['L_sat'].item()
            acc['rank'] += outs['L_rank'].item()
            acc['n']    += 1
        print(f"[Joint] epoch {epoch}/{epochs} | λpush={lam_push:g} λpull={lam_pull:g} λrec={lam_rec:g} λlip={lam_lip:g} "
              f"| L_push={acc['push']/acc['n']:.5f} L_pull={acc['pull']/acc['n']:.5f} "
              f"L_rec={acc['rec']/acc['n']:.5f} L_lip={acc['lip']/acc['n']:.5f} "
              f"L_calib={acc['calib']/acc['n']:.5f} L_sat={acc['sat']/acc['n']:.5f} L_rank={acc['rank']/acc['n']:.5f}")

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
            Y[k, s:s+batch_size] = f(xe, te).cpu().numpy().astype(np.float32)
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
            Y[k] = f(xk, tk).cpu().numpy()
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
    data_path = "/inspire/hdd/global_user/liuyiming-240108540153/training_dynamic/image_models/ResNet-TinyImageNet/Classification-normal/selected_subset/stacked_train_embeddings.npy"
    X_raw, t_raw, Xdot_raw = load_or_generate(path=data_path)

    # 计算并应用规范化（默认 robust）
    ckpt_dir = "/inspire/hdd/global_user/liuyiming-240108540153/training_dynamic/image_models/ResNet-TinyImageNet/Classification-normal/selected_subset/checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)
    stats = compute_stats_for_norm(X_raw, t_raw, mode=args.norm_mode,
                                   std_clip_low=args.std_clip_low, std_clip_high=args.std_clip_high)
    stats_path = os.path.join(ckpt_dir, f"norm_stats_{args.norm_mode}.npz")
    np.savez(stats_path, **stats)
    X, Xdot, t = apply_normalization_with_stats(X_raw, Xdot_raw, t_raw, stats)

    # DataLoader：原始 PairDataset（不使用 neighbor）
    ds = PairDataset(X, t, Xdot)
    dl = DataLoader(ds, batch_size=args.bs, shuffle=True, drop_last=False, num_workers=0, pin_memory=True)

    # 模型
    f = Encoder(args.D, args.d).to(args.device)
    g = Decoder(args.d, args.D).to(args.device)
    W = LowDimVF(args.d).to(args.device)

    # 端点缓存（标准化后）
    ep_cache = build_endpoint_cache(X, t, device=args.device)

    # 三阶段训练
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
    parser.add_argument('--D', type=int, default=512)
    parser.add_argument('--d', type=int, default=2)
    parser.add_argument('--T', type=int, default=50)
    parser.add_argument('--N', type=int, default=500)
    parser.add_argument('--bs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--lambda_push', type=float, default=1.0)
    parser.add_argument('--lambda_pull', type=float, default=1.0)
    parser.add_argument('--lambda_rec', type=float, default=1.0)
    parser.add_argument('--lip_reg', type=float, default=1e-3)
    # 规范化
    parser.add_argument('--norm_mode', type=str, default='robust',
                        choices=['global','anchor0','robust','per_epoch','center_only'])
    parser.add_argument('--std_clip_low', type=float, default=1e-8)
    parser.add_argument('--std_clip_high', type=float, default=0.0)
    # 三项副损开关/权重
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
