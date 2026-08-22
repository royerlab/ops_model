"""CellFlow-style conditional flow matching in CellDINO space (optional direction method).

Replaces the linear mean-diff axis (d = μ_KD − μ_ctrl) with a learned velocity field that
transports the control embedding distribution → the target (KD) distribution — a rectified /
conditional flow-matching model. Traversal = Euler-integrate the ODE from a control cell's
embedding; decode each step with the frozen DiffAE. Distribution-aware and nonlinear, so it
captures multimodal / off-centroid phenotypes a single mean vector can't.

Ref: CellFlow (bioRxiv 2025.04.11.648220); Flow Matching Guide (arXiv 2412.06264).
Deterministic given `seed` (fixed pairing/time sampling) so traversals stay reproducible.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment


class FlowNet(nn.Module):
    """Velocity field v_θ(x, t): CellDINO-dim in, CellDINO-dim out, time appended."""

    def __init__(self, dim: int, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x, t):                          # x:(B,dim) t:(B,)
        return self.net(torch.cat([x, t[:, None]], dim=1))


def _sinkhorn_col_sample(a: torch.Tensor, b: torch.Tensor, eps: float, n_iter: int,
                         g: torch.Generator) -> torch.Tensor:
    """Entropic OT plan (log-domain Sinkhorn, uniform marginals) between rows of a (Na,dim)
    and b (Nb,dim); returns, for each row of a, a column index of b sampled proportional to
    that row of the plan (the "soft" analogue of Hungarian's hard assignment). Uniform
    marginals mean the row-softmax of (f_i + g_j - cost_ij)/eps is exactly the normalized
    plan row regardless of the (constant, canceling) marginal weights.
    `eps` is a RELATIVE regularization strength (fraction of the cost matrix's own mean),
    not an absolute value -- raw CellDINO squared-distances run into the hundreds/thousands,
    so a fixed small eps (e.g. 0.05) overflows the log-domain updates (standard Sinkhorn
    stability practice: scale eps by the cost scale, e.g. POT's `reg * C.max()` convention)."""
    cost = torch.cdist(a, b).pow(2)
    eps = eps * cost.mean().clamp_min(1e-6)
    Na, Nb = cost.shape
    log_mu, log_nu = -math.log(Na), -math.log(Nb)
    f = torch.zeros(Na, device=cost.device)
    h = torch.zeros(Nb, device=cost.device)
    for _ in range(n_iter):
        # closed-form replacement each pass (f/h only enter as outer multiplicative factors
        # in the row/col marginal conditions) -- NOT an additive/accumulating update.
        f = eps * (log_mu - torch.logsumexp((h[None, :] - cost) / eps, dim=1))
        h = eps * (log_nu - torch.logsumexp((f[:, None] - cost) / eps, dim=0))
    row_probs = ((f[:, None] + h[None, :] - cost) / eps).softmax(dim=1)
    return torch.multinomial(row_probs, 1, generator=g).squeeze(1)


def train_flow(embs, labels, dev, steps=2000, bs=256, lr=1e-3, hidden=512, seed=0,
               coupling="independent", ot_batch_size=2048, ot_resample_every=50,
               sinkhorn_eps=0.05, sinkhorn_iters=100):
    """Conditional flow matching, control(label 0) → KD(label 1).
    x_t = (1-t)·x0 + t·x1, regress v_θ(x_t,t) to the straight-line velocity (x1 − x0).

    coupling:
      - 'independent' (default, unchanged): x0/x1 minibatches paired at random. Straight
        lines from unrelated pairs cross in embedding space, so the field the network
        fits is the (smoothed, high-variance) average of conflicting targets.
      - 'ot': pair x0/x1 within each minibatch via EXACT optimal transport (Hungarian
        assignment on squared-Euclidean cost, `scipy.optimize.linear_sum_assignment`)
        before building the interpolant — the OT-CFM fix (Tong et al. 2023 / CellFlow)
        for crossing-path noise. Deterministic given the minibatch draw (no entropic-OT
        epsilon to tune), no new dependency beyond scipy.
      - 'ot_sinkhorn': the literature-grounded fix for 'ot's remaining gap (2026-08-21
        build-log entry: Tong et al.'s own batch-size ablation shows bs=1 is IDENTICAL to
        independent coupling, and harder multimodal targets need larger batches than the
        SGD gradient batch can afford; CellFlow itself never uses exact/Hungarian OT on the
        gradient batch, always entropic/Sinkhorn on a larger pool). Every
        `ot_resample_every` steps, draws a fresh `ot_batch_size` sample (bigger than `bs`,
        cheap since Sinkhorn is O(b^2) not Hungarian's O(b^3)), solves an entropic OT plan
        (`_sinkhorn_col_sample`), and caches that paired pool; SGD steps in between just
        re-subsample `bs`-sized minibatches FROM the cached pairing — decoupling the
        OT-pairing batch from the gradient batch per Cheng & Schwing 2503.10636's
        "oversampling" fix.
    """
    torch.manual_seed(seed); np.random.seed(seed)
    x0 = torch.as_tensor(embs[labels == 0], dtype=torch.float32, device=dev)
    x1 = torch.as_tensor(embs[labels == 1], dtype=torch.float32, device=dev)
    net = FlowNet(embs.shape[1], hidden).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    g = torch.Generator(device=dev).manual_seed(seed)
    n0, n1 = len(x0), len(x1)
    paired_a = paired_b = None
    for step in range(steps):
        if coupling == "ot_sinkhorn" and step % ot_resample_every == 0:
            ob_n = min(ot_batch_size, n0)
            oa = x0[torch.randint(0, n0, (ob_n,), generator=g, device=dev)]
            ob = x1[torch.randint(0, n1, (min(ot_batch_size, n1),), generator=g, device=dev)]
            col = _sinkhorn_col_sample(oa, ob, sinkhorn_eps, sinkhorn_iters, g)
            paired_a, paired_b = oa, ob[col]
        if coupling == "ot_sinkhorn":
            idx = torch.randint(0, len(paired_a), (bs,), generator=g, device=dev)
            a, b = paired_a[idx], paired_b[idx]
        else:
            a = x0[torch.randint(0, n0, (bs,), generator=g, device=dev)]
            b = x1[torch.randint(0, n1, (bs,), generator=g, device=dev)]
            if coupling == "ot":
                cost = torch.cdist(a, b).pow(2).detach().cpu().numpy()
                _, col = linear_sum_assignment(cost)
                b = b[col]
        t = torch.rand(bs, generator=g, device=dev)
        xt = (1 - t)[:, None] * a + t[:, None] * b
        loss = ((net(xt, t) - (b - a)) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    return net


@torch.no_grad()
def integrate_flow(net, z0, dev, n_record=10, t_max=1.0, n_sub=None):
    """Euler-integrate dz/dt = v_θ(z,t) from t=0→t_max; record n_record+1 evenly-spaced
    points (incl. start). t_max>1 = OVERSHOOT past the KD manifold (t>1 is extrapolated).
    Substep count scales with t_max to keep step size ~constant. Returns (n_record+1, dim)."""
    if n_sub is None:
        n_sub = max(n_record, int(round(50 * t_max)))
    z = z0.clone().to(dev)
    dt = t_max / n_sub
    every = max(1, n_sub // n_record)
    traj = [z.clone()]
    for k in range(n_sub):
        t = torch.full((z.shape[0],), k * dt, device=dev)
        z = z + dt * net(z, t)
        if (k + 1) % every == 0:
            traj.append(z.clone())
    return torch.cat(traj, dim=0)


@torch.no_grad()
def integrate_flow_bidir(net, z0, dev, n_record=10, t_max=1.0, n_sub=None):
    """Three-way: forward control→KD to t_max, plus a backward 'anti-KD' arm (step AGAINST
    the field at t≈0). t_max>1 overshoots both extremes (analogous to mean-diff α>1).
    Returns (2·n_record+1, dim): anti_extreme … NTC(center) … KD_extreme."""
    if n_sub is None:
        n_sub = max(n_record, int(round(50 * t_max)))
    z0 = z0.clone().to(dev)
    dt = t_max / n_sub
    every = max(1, n_sub // n_record)
    z, fwd = z0.clone(), []
    for k in range(n_sub):
        t = torch.full((z.shape[0],), k * dt, device=dev)
        z = z + dt * net(z, t)
        if (k + 1) % every == 0:
            fwd.append(z.clone())
    z, bwd = z0.clone(), []
    t0 = torch.zeros(z0.shape[0], device=dev)
    for k in range(n_sub):
        z = z - dt * net(z, t0)                      # opposite the control→KD velocity
        if (k + 1) % every == 0:
            bwd.append(z.clone())
    return torch.cat(list(reversed(bwd)) + [z0.clone()] + fwd, dim=0)


@torch.no_grad()
def _integrate_endpoint_batched(net, z0: torch.Tensor, dev, t_max: float = 1.0, n_sub: int | None = None) -> torch.Tensor:
    """Euler-integrate to t_max and return ONLY the endpoint, for an arbitrary batch size
    (unlike `integrate_flow`'s multi-record `torch.cat`, which conflates the batch and time
    dims for batch>1 -- fine there since every existing caller uses batch=1, but reflow needs
    to integrate a whole population at once). Kept local rather than reusing
    `flow_field_compute.integrate_path` to avoid a circular import (that module imports
    FlowNet FROM this one)."""
    if n_sub is None:
        n_sub = max(1, int(round(50 * t_max)))
    z = z0.clone().to(dev)
    dt = t_max / n_sub
    for k in range(n_sub):
        t = torch.full((z.shape[0],), k * dt, device=dev)
        z = z + dt * net(z, t)
    return z


def reflow_train(net_old, x0_pool: np.ndarray, dev, steps: int = 2000, bs: int = 256,
                 lr: float = 1e-3, hidden: int = 512, seed: int = 0, t_max: float = 1.0) -> FlowNet:
    """Rectified-flow 'reflow' (Liu, Gong & Liu, arXiv 2209.03003): retrain a FRESH FlowNet on
    (x0, x1'=net_old(x0)) pairs, where x1' is `net_old`'s OWN deterministic t_max-endpoint for
    that x0 -- not the original ambiguous real-data pairing. These pairs are exact and
    non-crossing by construction (every x0 has exactly one x1'), which is what provably
    straightens the ODE (reduces trajectory curvature) without changing the t=1 marginal.
    Needs a DIFFERENT training loop from `train_flow`: pairing must be preserved by a SHARED
    index per minibatch draw, not resampled independently per side."""
    torch.manual_seed(seed); np.random.seed(seed)
    x0 = torch.as_tensor(x0_pool, dtype=torch.float32, device=dev)
    x1 = _integrate_endpoint_batched(net_old, x0, dev, t_max=t_max)
    net = FlowNet(x0.shape[1], hidden).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    g = torch.Generator(device=dev).manual_seed(seed)
    n = len(x0)
    for _ in range(steps):
        idx = torch.randint(0, n, (bs,), generator=g, device=dev)
        a, b = x0[idx], x1[idx]
        t = torch.rand(bs, generator=g, device=dev)
        xt = (1 - t)[:, None] * a + t[:, None] * b
        loss = ((net(xt, t) - (b - a)) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    return net
