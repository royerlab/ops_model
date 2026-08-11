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

import numpy as np
import torch
import torch.nn as nn


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


def train_flow(embs, labels, dev, steps=2000, bs=256, lr=1e-3, hidden=512, seed=0):
    """Conditional flow matching, control(label 0) → KD(label 1). Independent coupling:
    x_t = (1-t)·x0 + t·x1, regress v_θ(x_t,t) to the straight-line velocity (x1 − x0)."""
    torch.manual_seed(seed); np.random.seed(seed)
    x0 = torch.as_tensor(embs[labels == 0], dtype=torch.float32, device=dev)
    x1 = torch.as_tensor(embs[labels == 1], dtype=torch.float32, device=dev)
    net = FlowNet(embs.shape[1], hidden).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    g = torch.Generator(device=dev).manual_seed(seed)
    n0, n1 = len(x0), len(x1)
    for _ in range(steps):
        a = x0[torch.randint(0, n0, (bs,), generator=g, device=dev)]
        b = x1[torch.randint(0, n1, (bs,), generator=g, device=dev)]
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
