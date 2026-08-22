"""A/B test: does decoupled-batch Sinkhorn OT coupling (`coupling="ot_sinkhorn"`, added to
flow.py per the 2026-08-21 literature review -- Tong et al. 2302.00482's batch-size ablation +
Cheng & Schwing 2503.10636's oversampling fix + CellFlow's actual Sinkhorn-not-Hungarian
recipe) improve on the existing exact-Hungarian `coupling="ot"` at recovering real bimodal
structure? Same methodology as the 2026-08-20 raw-embedding audit: train both couplings on the
same cached real embeddings, integrate to t_max=1 (canonical, no overshoot) on ALL real NTC
cells, classify via nearest-real-centroid (NTC / cluster-A / cluster-B). CPU-only, no
DiffAE/decode needed -- this tests the field itself at its trained t=1, not the generation
pipeline.

    python -m ops_model.models.interpretability.diffae.directions.sinkhorn_ablation --gene TUBGCP6 --out-dir /path
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import numpy as np
import torch
from sklearn.cluster import KMeans

from ..flow_matching.flow import train_flow
from ..flow_matching.flow_field_compute import integrate_path

ROOT = "/hpc/projects/icd.fast.ops/models/diffex/flow_fields/geneKO"


def _classify_nearest(pts: np.ndarray, c_ntc: np.ndarray, c_a: np.ndarray, c_b: np.ndarray) -> dict:
    d = np.stack([np.linalg.norm(pts - c[None], axis=1) for c in (c_ntc, c_a, c_b)], axis=1)
    lab = np.argmin(d, axis=1)
    n = len(lab)
    return {"NTC": float((lab == 0).sum() / n), "A": float((lab == 1).sum() / n), "B": float((lab == 2).sum() / n)}


def run_sinkhorn_ablation(gene: str, out_dir: str, steps: int = 2000, seed: int = 0,
                          kd_top_k: int | None = None) -> dict:
    """kd_top_k: if set, restrict the KD side used for OT PAIRING/training to the top-k
    rank-ordered cells (row 0 = best rank -- see data.py::_top_cells, `.sort_values("rank")`,
    order preserved through materialize_crops's shuffle=False loader) -- testing whether a
    purer phenotype population (less penetrance/ambiguity noise) helps the coupling find the
    real bimodal structure. Evaluation centroids (c_a/c_b) always come from the FULL kd
    population -- the ground truth being tested against doesn't change with kd_top_k."""
    torch.set_num_threads(4)
    dev = torch.device("cpu")
    cache = f"{ROOT}/{gene}/cache"
    feats = np.load(f"{cache}/celldino.npz")["features"].astype(np.float32)
    labels = np.load(f"{cache}/crops.npz")["labels"]
    kd, ntc = feats[labels == 1], feats[labels == 0]

    km = KMeans(n_clusters=2, n_init=4, random_state=0).fit(kd)
    c_a, c_b = kd[km.labels_ == 0].mean(0), kd[km.labels_ == 1].mean(0)
    c_ntc = ntc.mean(0)
    ntc_t = torch.as_tensor(ntc, dtype=torch.float32)

    kd_train = kd[:kd_top_k] if kd_top_k else kd
    train_embs = np.vstack([ntc, kd_train])
    train_labels = np.array([0] * len(ntc) + [1] * len(kd_train))

    result = {"gene": gene, "n_kd": int(len(kd)), "n_ntc": int(len(ntc)), "kd_top_k": kd_top_k or len(kd)}
    for coupling in ["ot", "ot_sinkhorn"]:
        net = train_flow(train_embs, train_labels, dev, steps=steps, seed=seed, coupling=coupling)
        out = integrate_path(net, ntc_t, n_sub=30, t_max=1.0)[-1]
        result[coupling] = _classify_nearest(out, c_ntc, c_a, c_b)
        print(f"[{gene}] kd_top_k={result['kd_top_k']} {coupling} -> {result[coupling]}")

    out_path = Path(out_dir); out_path.mkdir(parents=True, exist_ok=True)
    (out_path / f"sinkhorn_ablation_{gene}_k{result['kd_top_k']}.json").write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Hungarian-OT vs Sinkhorn-OT coupling A/B test")
    ap.add_argument("--gene", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--kd-top-k", type=int, default=None)
    args = ap.parse_args()
    run_sinkhorn_ablation(args.gene, args.out_dir, steps=args.steps, kd_top_k=args.kd_top_k)


if __name__ == "__main__":
    main()
