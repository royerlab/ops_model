"""Does one rectified-flow 'reflow' pass (Liu et al. 2209.03003, `flow.py::reflow_train`) on
top of an already-trained OT-coupled field improve real-bimodal-balance recovery? Reflow
retrains on the field's OWN (x0, endpoint) pairs -- exact and non-crossing by construction --
which provably straightens the ODE without changing the t=1 marginal. Untested whether
straightening also helps recover the true cluster balance; this checks it directly, same
nearest-real-centroid methodology as sinkhorn_ablation.py, before vs. after reflow.

    python -m ops_model.models.interpretability.diffae.directions.reflow_ablation --gene TUBGCP6 --out-dir /path
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

from ..flow_matching.flow import reflow_train, train_flow
from ..flow_matching.flow_field_compute import integrate_path
from .sinkhorn_ablation import ROOT, _classify_nearest


def run_reflow_ablation(gene: str, out_dir: str, steps: int = 2000, seed: int = 0) -> dict:
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

    result = {"gene": gene, "n_kd": int(len(kd)), "n_ntc": int(len(ntc))}
    for coupling in ["ot", "ot_sinkhorn"]:
        net_old = train_flow(feats, labels, dev, steps=steps, seed=seed, coupling=coupling)
        before = _classify_nearest(integrate_path(net_old, ntc_t, n_sub=30, t_max=1.0)[-1], c_ntc, c_a, c_b)

        net_new = reflow_train(net_old, ntc, dev, steps=steps, seed=seed, t_max=1.0)
        after = _classify_nearest(integrate_path(net_new, ntc_t, n_sub=30, t_max=1.0)[-1], c_ntc, c_a, c_b)

        result[coupling] = {"before_reflow": before, "after_reflow": after}
        print(f"[{gene}] {coupling} before={before} after_reflow={after}")

    out_path = Path(out_dir); out_path.mkdir(parents=True, exist_ok=True)
    (out_path / f"reflow_ablation_{gene}.json").write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Before/after reflow bimodal-balance recovery test")
    ap.add_argument("--gene", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--steps", type=int, default=2000)
    args = ap.parse_args()
    run_reflow_ablation(args.gene, args.out_dir, steps=args.steps)


if __name__ == "__main__":
    main()
