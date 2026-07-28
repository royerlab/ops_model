"""Split Alex's new shap_screen fluor ranking (one combined CSV, 55 channels, per-cell, robust bin-size
ranking) into the per-marker parquets the v5 traversal build consumes (FRP_DIR/<slug>.parquet), in the
SAME schema as the old qualifying rankings so build_marker is a drop-in.

  new CSV cols:  gene, channel_name, rank, shap, ..., experiment, well, x_pheno, y_pheno, segmentation_id
  old schema:    channel_name, gene, rank, pma_attention, experiment, well, x_pheno, y_pheno, segmentation, rank_type

  python -m ops_model.models.attention.diffex.viewer.build_fluor_shap_rankings          # local (needs ~64GB)
  python -m ops_model.models.attention.diffex.viewer.build_fluor_shap_rankings --submit # SLURM cpu, mem 96
"""
from __future__ import annotations

import argparse
import os

import pandas as pd

from ..classifier.config import slugify

CSV = ("/hpc/projects/icd.fast.ops/models/alex_lin_attention/v5/multi_rank/"
       "shap_screen/shap_screen_fluor_all.csv")
OUT_DIR = "/hpc/projects/icd.fast.ops/models/diffex/viewer_assets_v5/_rankings/fluor_shap/geneKO"


def build():
    os.makedirs(OUT_DIR, exist_ok=True)
    use = ["gene", "channel_name", "rank", "shap", "experiment", "well", "x_pheno", "y_pheno", "segmentation_id"]
    print(f"[shap-rank] reading {CSV} ...", flush=True)
    df = pd.read_csv(CSV, usecols=use)
    df = df.rename(columns={"shap": "pma_attention", "segmentation_id": "segmentation"})
    df["rank_type"] = "top"
    cols = ["channel_name", "gene", "rank", "pma_attention", "experiment", "well",
            "x_pheno", "y_pheno", "segmentation", "rank_type"]
    df = df[cols]
    print(f"[shap-rank] {len(df):,} rows, {df.channel_name.nunique()} channels", flush=True)
    n = 0
    for ch, sub in df.groupby("channel_name"):
        p = f"{OUT_DIR}/{slugify(ch)}.parquet"
        sub.reset_index(drop=True).to_parquet(p)
        n += 1
        print(f"  [{n:2d}] {slugify(ch):40s} {len(sub):>8,} rows ({sub.gene.nunique()} classes) -> {os.path.basename(p)}",
              flush=True)
    print(f"[shap-rank] wrote {n} per-marker parquets -> {OUT_DIR}")
    return {"markers": n, "rows": int(len(df))}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--submit", action="store_true")
    a = ap.parse_args()
    if a.submit:
        from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs
        submit_parallel_jobs(
            jobs_to_submit=[{"name": "fluor_shap_split", "func": build, "kwargs": {}}],
            experiment="diffex_shaprank",
            slurm_params={"slurm_partition": "cpu", "cpus_per_task": 8, "mem_gb": 96, "timeout_min": 60},
            log_dir="diffex_shaprank", wait_for_completion=False)
    else:
        build()
