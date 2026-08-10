"""Build v5-multibag per-experiment PCA h5ads for the sliding-window sweep.

Reads the existing per-exp PCA h5ads at
``expansion_v1/per_experiment_v5_pca/`` and writes copies to
``expansion_v1_multibag/per_experiment_v5_pca/`` with the ``gko/ebionly/ebifb``
rank columns overwritten with multi-bag SHAP-derived values (from the
sidecars built by ``build_multibag_sidecars.py``).

Same column names as v5 → the sweep script picks up the multi-bag ranks
automatically when ``V5_EXPANSION_SUBDIR=expansion_v1_multibag`` is set.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

SRC_ROOT = Path("/hpc/projects/icd.fast.ops/models/alex_lin_attention/v5/expansion_v1")
DST_ROOT = Path("/hpc/projects/icd.fast.ops/models/alex_lin_attention/v5/expansion_v1_multibag")
SRC_PCA = SRC_ROOT / "per_experiment_v5_pca"
DST_PCA = DST_ROOT / "per_experiment_v5_pca"

HEADS = ["gko", "ebionly", "ebifb"]


def _load_sidecar(head: str) -> pd.DataFrame:
    """Return sidecar w/ cols (experiment, well, seg, rank, gene) — one row/cell.

    ~0.06% of cells appear under two genes' rankings (barcode-mismatch /
    dual-guide crosstalk). Dedup by keeping the lowest (best) rank per
    (exp, well, seg) — matches attach_set_accuracy_rank's pattern.
    """
    p = SRC_ROOT / f"per_experiment_v5_multibag_{head}.parquet"
    df = pd.read_parquet(p, columns=[
        "experiment", "well", "segmentation_id",
        f"v5m_{head}_gene", f"v5m_{head}_rank",
    ])
    df["well"] = df["well"].astype(str)
    df = (df.sort_values(["experiment", "well", "segmentation_id", f"v5m_{head}_rank"])
            .drop_duplicates(subset=["experiment", "well", "segmentation_id"], keep="first"))
    return df


def attach_one(experiment: str) -> str:
    src = SRC_PCA / f"{experiment}.h5ad"
    dst = DST_PCA / f"{experiment}.h5ad"
    DST_PCA.mkdir(parents=True, exist_ok=True)
    if not src.exists():
        return f"MISSING: {src}"

    t0 = time.time()
    a = ad.read_h5ad(src)
    obs = a.obs.copy()
    obs["well"] = obs["well"].astype(str)
    n = len(obs)

    for head in HEADS:
        sc = _load_sidecar(head)
        sc = sc[sc["experiment"] == experiment].drop(columns=["experiment"])
        merged = obs.merge(
            sc, on=["well", "segmentation_id"], how="left", validate="m:1",
        )
        rank_vals = merged[f"v5m_{head}_rank"].to_numpy(dtype=np.float64)
        gene_vals = merged[f"v5m_{head}_gene"].astype(str).fillna("").values
        rank_type = np.where(np.isfinite(rank_vals), "top", "")

        # NTC cells are in the sidecar with gene='NTC' — mark them explicitly.
        ntc = (gene_vals == "NTC")
        rank_type = np.where(ntc, "NTC", rank_type)

        a.obs[f"{head}_rank"]      = rank_vals
        a.obs[f"{head}_gene"]      = pd.Categorical(gene_vals)
        a.obs[f"{head}_rank_type"] = pd.Categorical(rank_type)

        matched = int(np.isfinite(rank_vals).sum())
        print(f"  [{experiment}] {head}: matched {matched:,}/{n:,} ({matched/n*100:.1f}%)")

    a.write_h5ad(dst)
    return f"OK: {experiment} ({time.time()-t0:.1f}s)"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", help="Single experiment to process")
    ap.add_argument("--all", action="store_true", help="Process all experiments")
    args = ap.parse_args()
    if args.experiment:
        print(attach_one(args.experiment))
    elif args.all:
        exps = sorted(p.stem for p in SRC_PCA.glob("*.h5ad"))
        for e in exps:
            print(attach_one(e))
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
