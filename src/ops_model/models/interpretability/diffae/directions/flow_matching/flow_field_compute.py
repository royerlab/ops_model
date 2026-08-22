"""SLURM-shardable compute unit for the multi-perturbation flow-field landscape (a coding_exps
visualization; see coding_exps/diffex/ot_cfm_test/flow_field_landscape.py). Lives in the
package — not coding_exps — purely so a submitit worker can unpickle it by import path on the
compute node; a bare script isn't importable there (see ot_cfm_test.py's build log entry for
the same issue). No plotting dependency here, only torch/numpy — cheap, CPU-only.
"""
from __future__ import annotations

import os

# Must precede numpy/torch import to take effect (BLAS thread pools init at import time) —
# a prior unclipped/unthreaded run of a sibling script caused a 12-CPU-hour runaway on this
# shared login node. compute_shard() also calls torch.set_num_threads at runtime as a second
# line of defense, since env vars only help if THIS module is the first to import torch/numpy
# in the process.
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import glob
import json
from pathlib import Path

import numpy as np
import torch

from .flow import FlowNet

FLOW_FIELDS_ROOT = "/hpc/projects/icd.fast.ops/models/diffex/flow_fields"
REFERENCE_CACHE = "/hpc/mydata/gav.sturm/ops_mono/coding_exps/diffex/ot_cfm_test/sweep_v2/HSPA5/cache"


def ntc_anchor() -> np.ndarray:
    embs = np.load(f"{REFERENCE_CACHE}/celldino.npz")["features"]
    labels = np.load(f"{REFERENCE_CACHE}/crops.npz")["labels"]
    return embs[labels == 0].mean(0).astype(np.float32)


def ntc_probes(k: int = 6, seed: int = 0) -> np.ndarray:
    """k REAL individual NTC cell embeddings (not the population mean) — the same k cells for
    every target, so pushing them through each target's OWN trained field and comparing the
    resulting spread is a fair, real (not synthetic) measure of variance: a tight fan means the
    field predicts a consistent effect across real cells, a wide one means real cell-to-cell
    heterogeneity (or model uncertainty) actually matters for that target."""
    embs = np.load(f"{REFERENCE_CACHE}/celldino.npz")["features"]
    labels = np.load(f"{REFERENCE_CACHE}/crops.npz")["labels"]
    ntc = embs[labels == 0]
    idx = np.random.default_rng(seed).choice(len(ntc), size=k, replace=False)
    return ntc[idx].astype(np.float32)  # (k, dim)


def load_net(out_dir: Path) -> FlowNet:
    sd = torch.load(out_dir / "flow_ot.pt", map_location="cpu")
    dim = sd["net.0.weight"].shape[1] - 1
    hidden = sd["net.0.weight"].shape[0]
    net = FlowNet(dim, hidden)
    net.load_state_dict(sd)
    net.eval()
    return net


@torch.no_grad()
def integrate_path(net: FlowNet, z0: torch.Tensor, n_sub: int, t_max: float) -> np.ndarray:
    """Euler-integrate net's OWN trained field from t=0 to t_max, from a batch of K start
    points (z0: (K,dim), integrated together — FlowNet batches over dim 0 naturally). Returns
    the full path (n_sub+1, K, dim), not just the endpoint, so curvature AND the spread across
    the K starting cells are both visible."""
    z = z0.clone()
    dt = t_max / n_sub
    path = [z.clone()]
    for k in range(n_sub):
        t = torch.full((z.shape[0],), k * dt, dtype=torch.float32)
        z = z + dt * net(z, t)
        path.append(z.clone())
    return torch.stack(path, dim=0).numpy()


def compute_shard(grain: str, shard_idx: int, n_shards: int, shard_dir: str,
                  n_sub: int = 30, t_max: float = 1.0, k_probes: int = 6) -> str:
    """Load+integrate every `n_shards`-th target's checkpoint (interleaved sharding, no
    manifest file needed). CPU-only, cheap (tiny nets) — sharding this is about never running
    ~1000 sequential checkpoint loads as one long loop on a shared login node, not raw speed.
    Integrates k_probes REAL NTC cells (not just their mean) through each target's own field,
    so the saved paths capture real cell-to-cell variance, not a single point estimate."""
    import torch as _torch
    _torch.set_num_threads(4)
    ntc_mean = ntc_anchor()
    probes = ntc_probes(k=k_probes)
    ckpts = sorted(glob.glob(f"{FLOW_FIELDS_ROOT}/{grain}/*/flow_ot.pt"))[shard_idx::n_shards]
    probes_t = torch.as_tensor(probes, dtype=torch.float32)  # (K,dim)
    rows, paths = [], []
    for p in ckpts:
        out_dir = Path(p).parent
        meta = json.loads((out_dir / "metrics.json").read_text())
        net = load_net(out_dir)
        paths.append(integrate_path(net, probes_t, n_sub=n_sub, t_max=t_max))  # (S+1,K,dim)
        rows.append({"target": meta["target"], "grain": grain,
                    "flow_advantage": meta.get("flow_advantage_ko_delta_ratio")})
    out = Path(shard_dir) / f"{grain}_shard{shard_idx:03d}_of_{n_shards}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, targets=[r["target"] for r in rows], grain=[r["grain"] for r in rows],
            flow_advantage=[r["flow_advantage"] for r in rows],
            paths=np.stack(paths), ntc=ntc_mean)  # paths: (N_shard, S+1, K, dim)
    print(f"[{grain} shard {shard_idx}/{n_shards}] {len(rows)} targets x {k_probes} probes -> {out}")
    return str(out)
