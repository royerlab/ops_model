"""Multibag pick montages for the fluorescence panels C/D — one KO + one NTC montage per panel column,
using each column's ACTUAL panel marker (GENE_COLS/COMPLEX_COLS) and the multibag SHAP rankings
(fluor_shap/{geneKO,complex}). Fixes the montage↔panel marker mismatch (e.g. TOMM20 = Mitochondria_TOMM20,
not ChromaLIVE). Crops the marker channel + blue seg mask; rank/cell-key/shap annotated.

Run (SLURM): python fluor_panel_montages.py --submit
"""
import sys

import pandas as pd

from _setacc_common import GENE_COLS, COMPLEX_COLS, _materialize, slugify
from fluor_shap_montages import render_montage   # reuses OUT=figure4_shap_montages + seg overlay

R = "/hpc/projects/icd.fast.ops/models/diffex/viewer_assets_v5/_rankings/fluor_shap"
COLS = GENE_COLS + COMPLEX_COLS
N = 100


def _df(mc, block, cls):
    sub = "complex" if (block == "complexes" and cls != "NTC") else "geneKO"   # NTC always from the geneKO parquet
    d = pd.read_parquet(f"{R}/{sub}/{slugify(mc)}.parquet")
    d = d[d["gene"].astype(str) == str(cls)]
    if "rank_type" in d.columns:
        d = d[d["rank_type"] == "top"]
    return d.sort_values("rank").reset_index(drop=True)


def col_montages(col):
    for tag, cls in (("KO", col["key"]), ("NTC", "NTC")):
        d = _df(col["mc"], col["block"], cls).head(N)
        if d.empty:
            print(f"skip {tag} {col['top_label']} ({col['mc']}): no rows", flush=True); continue
        raw, recs = _materialize(d, col["mc"], col["ch"], cls)
        lbl = col["top_label"].replace("\n", " ")
        render_montage(raw, recs, f"{tag}  {lbl}  ({col['mc']})  multibag SHAP",
                       f"fluorpanel_{tag}_{col['slug']}")


def main(slugs=None):
    want = set(slugs) if slugs else None
    for c in COLS:
        if want and c["slug"] not in want:
            continue
        col_montages(c)


def _job(slug):
    import os
    os.environ.setdefault("OPS_DIFFEX_ASSETS", "viewer_assets_v5")
    main([slug])


def submit(slugs=None):
    import os
    import pathlib
    from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs
    figdir = str(pathlib.Path(__file__).resolve().parent)
    os.environ["PYTHONPATH"] = figdir + os.pathsep + os.environ.get("PYTHONPATH", "")
    os.environ.setdefault("OPS_DIFFEX_ASSETS", "viewer_assets_v5")
    ss = slugs or [c["slug"] for c in COLS]
    jobs = [{"name": f"flpanel_{s[:16]}", "func": _job, "kwargs": {"slug": s}} for s in ss]
    submit_parallel_jobs(jobs, experiment="diffex_flpanel",
                         slurm_params={"slurm_partition": "cpu", "cpus_per_task": 8, "mem_gb": 64, "timeout_min": 90},
                         log_dir="diffex_flpanel", wait_for_completion=False)


if __name__ == "__main__":
    submit(sys.argv[2:] or None) if (len(sys.argv) > 1 and sys.argv[1] == "--submit") else main(sys.argv[1:] or None)
