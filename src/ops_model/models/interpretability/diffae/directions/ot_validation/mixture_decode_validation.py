"""Decode-validated version of mixture_flow_multik.py: does the per-cluster OT-vs-mean_diff
routing-purity advantage found in raw embedding space survive actual DiffAE decode + CellDINO
re-embedding? Same lesson as the earlier pool_shape_recovery.py correction (2026-08-20/21 build
log): raw embedding-space math can look clean and still reverse once real image generation is
in the loop -- required before trusting the multi-K mixture-of-flows result at scale.

Pushes a shared pool of real NTC cells through EVERY cluster's mean_diff vector and trained
OT flow, decodes each through the frozen DiffAE, re-embeds via CellDINO, and classifies against
the real (K+1)-way centroids (NTC + each discovered cluster) -- same methodology as
mixture_flow_multik.py, just with the generation pipeline in the loop.

    python -m ops_model.models.interpretability.diffae.directions.mixture_decode_validation \
        --gene HAUS6 --k 4 --out-dir /path
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

from ...classifier.celldino_features import embed_crops
from ..config import DirConfig
from ..flow_matching.flow import train_flow
from ..flow_matching.flow_field_compute import integrate_path
from .mixture_flow_multik import _classify_nearest_k
from ..traverse import _sample_guided, load_diffae


def _decode_and_reencode(diffae, null_base, embs: np.ndarray, xT_list, w: float, cfg) -> np.ndarray:
    dev = xT_list[0].device
    imgs = []
    for i in range(len(embs)):
        z = torch.as_tensor(embs[i:i + 1], dtype=torch.float32, device=dev)
        img = _sample_guided(diffae, xT_list[i].clone(), z, null_base, w, cfg)
        imgs.append(img.cpu().numpy()[0, 0])
    flat = np.stack(imgs)[:, None].astype(np.float32)
    return embed_crops(flat, cfg, cache_path=None)


def run_mixture_decode_validation(gene: str, k: int, out_dir: str, n_pool: int = 20,
                                  steps: int = 2000, w: float = 1.0, seed: int = 0,
                                  grain: str = "geneKO") -> dict:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = f"/hpc/projects/icd.fast.ops/models/diffex/flow_fields/{grain}"
    cache = f"{root}/{gene}/cache"
    feats = np.load(f"{cache}/celldino.npz")["features"].astype(np.float32)
    labels = np.load(f"{cache}/crops.npz")["labels"]
    kd, ntc = feats[labels == 1], feats[labels == 0]

    n_comp = min(20, kd.shape[0] - 1, kd.shape[1])
    kd_pca = PCA(n_components=n_comp, random_state=seed).fit_transform(kd)
    gm = GaussianMixture(n_components=k, covariance_type="diag", random_state=seed, reg_covar=1e-3).fit(kd_pca)
    cluster_lab = gm.predict(kd_pca)

    c_ntc = ntc.mean(0)
    clusters = [kd[cluster_lab == i] for i in range(k)]
    centroids = [c_ntc] + [c.mean(0) for c in clusters]

    rng = np.random.default_rng(seed)
    pool_idx = rng.choice(len(ntc), size=min(n_pool, len(ntc)), replace=False)
    pool = ntc[pool_idx]

    cfg = DirConfig(grain=grain, target=gene, device="cuda")
    diffae = load_diffae(cfg, dev)
    null_base = diffae.null_emb.detach()[None].to(dev)
    H = cfg.crop_size
    xT_list = [torch.randn(1, 1, H, H, generator=torch.Generator(device=dev).manual_seed(1234 + i),
                           device=dev) for i in range(len(pool))]

    decoded_null = _decode_and_reencode(diffae, null_base, pool, xT_list, w, cfg)

    result = {"gene": gene, "k": k, "n_pool": int(len(pool)),
             "cluster_sizes": [int(len(c)) for c in clusters],
             "decoded_null_classify": _classify_nearest_k(decoded_null, centroids),
             "mean_diff": {}, "ot": {}}

    net_dev = torch.device("cpu")
    for i, c in enumerate(clusters):
        c_i = centroids[i + 1]
        d_vec = c_i - c_ntc
        gap = float(np.linalg.norm(d_vec))
        md_end = pool + gap * (d_vec / gap)[None, :]
        decoded_md = _decode_and_reencode(diffae, null_base, md_end, xT_list, w, cfg)
        result["mean_diff"][f"cluster_{i}"] = _classify_nearest_k(decoded_md, centroids)

        embs_i = np.vstack([ntc, c]); labels_i = np.array([0] * len(ntc) + [1] * len(c))
        net_i = train_flow(embs_i, labels_i, net_dev, steps=steps, seed=seed, coupling="ot")
        fo_end = integrate_path(net_i, torch.as_tensor(pool, dtype=torch.float32), n_sub=30, t_max=1.0)[-1]
        decoded_fo = _decode_and_reencode(diffae, null_base, fo_end, xT_list, w, cfg)
        result["ot"][f"cluster_{i}"] = _classify_nearest_k(decoded_fo, centroids)

        print(f"[{gene} k={k}] cluster_{i} (n={len(c)}): mean_diff={result['mean_diff'][f'cluster_{i}']} "
             f"ot={result['ot'][f'cluster_{i}']}")

    out_path = Path(out_dir); out_path.mkdir(parents=True, exist_ok=True)
    (out_path / f"decode_multik_{gene}_k{k}.json").write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Decode-validated multi-K mixture-of-flows vs mean_diff")
    ap.add_argument("--gene", required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-pool", type=int, default=20)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--w", type=float, default=1.0)
    ap.add_argument("--grain", default="geneKO", choices=["geneKO", "complex"])
    args = ap.parse_args()
    run_mixture_decode_validation(args.gene, args.k, args.out_dir, n_pool=args.n_pool, steps=args.steps,
                                  w=args.w, grain=args.grain)


if __name__ == "__main__":
    main()
