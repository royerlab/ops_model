"""Generalized (K>2) version of mixture_flow_test.py, using the real best_k from
complexity_scan.py's BIC model selection instead of a forced KMeans(k=2) -- TUBGCP6 and KIF11
turned out to be best_k=5 and best_k=3 respectively, not 2, so today's earlier mixture-of-flows
test was itself run on an oversimplified split. Clusters assigned via GaussianMixture (same
method complexity_scan used to pick best_k, for consistency) on PCA-reduced KD embeddings.
For each of the K real sub-clusters: train ONE flow (NTC -> that cluster only) and compute ONE
mean_diff vector (NTC centroid -> that cluster centroid), then check how purely each routes the
full real NTC population toward ITS OWN cluster (nearest-of-(K+1)-centroids classification).

    python -m ops_model.models.interpretability.diffae.directions.mixture_flow_multik --gene TUBGCP6 --k 5 --out-dir /path
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
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

from ..flow_matching.flow import train_flow
from ..flow_matching.flow_field_compute import integrate_path

ROOT = "/hpc/projects/icd.fast.ops/models/diffex/flow_fields/geneKO"


def _classify_nearest_k(pts: np.ndarray, centroids: list[np.ndarray]) -> list[float]:
    d = np.stack([np.linalg.norm(pts - c[None], axis=1) for c in centroids], axis=1)
    lab = np.argmin(d, axis=1)
    n = len(lab)
    return [float((lab == i).sum() / n) for i in range(len(centroids))]


def run_mixture_flow_multik(gene: str, k: int, out_dir: str, steps: int = 2000, seed: int = 0,
                            pca_dim: int = 20) -> dict:
    torch.set_num_threads(4)
    dev = torch.device("cpu")
    cache = f"{ROOT}/{gene}/cache"
    feats = np.load(f"{cache}/celldino.npz")["features"].astype(np.float32)
    labels = np.load(f"{cache}/crops.npz")["labels"]
    kd, ntc = feats[labels == 1], feats[labels == 0]
    ntc_t = torch.as_tensor(ntc, dtype=torch.float32)

    n_comp = min(pca_dim, kd.shape[0] - 1, kd.shape[1])
    kd_pca = PCA(n_components=n_comp, random_state=seed).fit_transform(kd)
    gm = GaussianMixture(n_components=k, covariance_type="diag", random_state=seed, reg_covar=1e-3).fit(kd_pca)
    cluster_lab = gm.predict(kd_pca)

    c_ntc = ntc.mean(0)
    clusters = [kd[cluster_lab == i] for i in range(k)]
    centroids = [c_ntc] + [c.mean(0) for c in clusters]  # index 0 = NTC, 1..k = clusters

    result = {"gene": gene, "k": k, "n_ntc": int(len(ntc)),
             "cluster_sizes": [int(len(c)) for c in clusters], "mean_diff": {}, "ot": {}}
    for i, c in enumerate(clusters):
        c_i = centroids[i + 1]
        d_vec = c_i - c_ntc
        gap = float(np.linalg.norm(d_vec))
        out_md = ntc + gap * (d_vec / gap)[None, :]
        result["mean_diff"][f"cluster_{i}"] = _classify_nearest_k(out_md, centroids)

        embs_i = np.vstack([ntc, c]); labels_i = np.array([0] * len(ntc) + [1] * len(c))
        net_i = train_flow(embs_i, labels_i, dev, steps=steps, seed=seed, coupling="ot")
        out_ot = integrate_path(net_i, ntc_t, n_sub=30, t_max=1.0)[-1]
        result["ot"][f"cluster_{i}"] = _classify_nearest_k(out_ot, centroids)
        print(f"[{gene} k={k}] cluster_{i} (n={len(c)}): mean_diff={result['mean_diff'][f'cluster_{i}']} "
             f"ot={result['ot'][f'cluster_{i}']}")

    out_path = Path(out_dir); out_path.mkdir(parents=True, exist_ok=True)
    (out_path / f"mixture_multik_{gene}_k{k}.json").write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Multi-K mixture-of-flows vs per-cluster mean_diff")
    ap.add_argument("--gene", required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--steps", type=int, default=2000)
    args = ap.parse_args()
    run_mixture_flow_multik(args.gene, args.k, args.out_dir, steps=args.steps)


if __name__ == "__main__":
    main()
