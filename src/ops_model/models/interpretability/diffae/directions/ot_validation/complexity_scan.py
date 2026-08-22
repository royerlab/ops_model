"""Generalized multi-modality scan: replaces the earlier `bimodality_scan.py`'s fixed-k=2
KMeans-separation assumption (which can only ever find genes that already fit a clean 2-cluster
model -- exactly the easy case a 2-vector piecewise mean_diff already handles) with a proper
model-selection approach. Per gene: PCA the real KD population to a tractable dimensionality,
fit a Gaussian mixture for k=1..5, pick the best k by BIC. Flags genes whose real phenotype is
genuinely >2-modal (or where k=2 wins but very unevenly) -- the actual regime to test whether
OT-CFM's continuous, non-discrete field has an edge that a small, fixed number of mean_diff
vectors doesn't.

    python -m ops_model.models.interpretability.diffae.directions.complexity_scan \
        --shard-idx 0 --n-shards 40 --shard-dir /path
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import numpy as np
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture


def _best_k(kd: np.ndarray, pca_dim: int = 20, k_max: int = 5, seed: int = 0) -> dict:
    n_comp = min(pca_dim, kd.shape[0] - 1, kd.shape[1])
    kd_pca = PCA(n_components=n_comp, random_state=seed).fit_transform(kd)
    bics = []
    for k in range(1, k_max + 1):
        gm = GaussianMixture(n_components=k, covariance_type="diag", random_state=seed, reg_covar=1e-3)
        gm.fit(kd_pca)
        bics.append(float(gm.bic(kd_pca)))
    best_k = int(np.argmin(bics)) + 1
    result = {"bics": bics, "best_k": best_k, "bic_gain_over_k1": bics[0] - min(bics)}
    if best_k >= 2:
        gm = GaussianMixture(n_components=best_k, covariance_type="diag", random_state=seed, reg_covar=1e-3).fit(kd_pca)
        counts = np.bincount(gm.predict(kd_pca), minlength=best_k)
        result["cluster_sizes"] = counts.tolist()
        result["balance_min_max"] = float(counts.min() / counts.max())
    return result


def compute_shard(shard_idx: int, n_shards: int, shard_dir: str, pca_dim: int = 20, k_max: int = 5,
                  grain: str = "geneKO") -> str:
    root = f"/hpc/projects/icd.fast.ops/models/diffex/flow_fields/{grain}"
    gene_dirs = sorted(glob.glob(f"{root}/*"))[shard_idx::n_shards]
    rows = []
    for gd in gene_dirs:
        gene = os.path.basename(gd)
        cache = f"{gd}/cache"
        try:
            feats = np.load(f"{cache}/celldino.npz")["features"].astype(np.float32)
            labels = np.load(f"{cache}/crops.npz")["labels"]
        except Exception:
            continue
        kd = feats[labels == 1]
        if len(kd) < 20:
            continue
        r = _best_k(kd, pca_dim=pca_dim, k_max=k_max)
        r["gene"] = gene
        r["n_kd"] = int(len(kd))
        rows.append(r)
    out = Path(shard_dir) / f"complexity_shard{shard_idx:03d}_of_{n_shards}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))
    print(f"[shard {shard_idx}/{n_shards}] {len(rows)} genes -> {out}")
    return str(out)


def main() -> None:
    ap = argparse.ArgumentParser(description="Multi-modality (GMM/BIC) scan, sharded")
    ap.add_argument("--shard-idx", type=int, required=True)
    ap.add_argument("--n-shards", type=int, required=True)
    ap.add_argument("--shard-dir", required=True)
    ap.add_argument("--grain", default="geneKO", choices=["geneKO", "complex"])
    args = ap.parse_args()
    compute_shard(args.shard_idx, args.n_shards, args.shard_dir, grain=args.grain)


if __name__ == "__main__":
    main()
