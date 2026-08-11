"""Split Alex's new shap_screen fluor ranking (one combined CSV, 55 channels, per-cell, robust bin-size
ranking) into the per-marker parquets the v5 traversal build consumes (FRP_DIR/<slug>.parquet), in the
SAME schema as the old qualifying rankings so build_marker is a drop-in.

  new CSV cols:  gene, channel_name, rank, shap, ..., experiment, well, x_pheno, y_pheno, segmentation_id
  old schema:    channel_name, gene, rank, pma_attention, experiment, well, x_pheno, y_pheno, segmentation, rank_type

  python -m ops_model.models.interpretability.toolkit.viewer.build_fluor_shap_rankings          # local (needs ~64GB)
  python -m ops_model.models.interpretability.toolkit.viewer.build_fluor_shap_rankings --submit # SLURM cpu, mem 96
"""
from __future__ import annotations

import argparse
import os

import pandas as pd

from ops_model.models.interpretability.diffae.classifier.config import slugify

CSV = ("/hpc/projects/icd.fast.ops/models/alex_lin_attention/v5/multi_rank/"
       "shap_screen/shap_screen_fluor_all.csv")
OUT_DIR = "/hpc/projects/icd.fast.ops/models/diffex/viewer_assets_v5/_rankings/fluor_shap/geneKO"

# --- fluor COMPLEX (EBI) ranking: per-GENE cells → pool member genes into each complex ---
EBI_CSV = ("/hpc/projects/icd.fast.ops/models/alex_lin_attention/v5/multi_rank/"
           "shap_screen_ebi_fluor_all.csv")
# the shap CSV uses OLD gene symbols (RARS/DARS/MARS/… not RARS1/…), so the gene→complex map MUST come from the
# old-gene-names yaml — the updated one silently drops the aminoacyl-tRNA-synthetase complex (11 genes).
EBI_YAML = "/hpc/projects/icd.fast.ops/configs/gene_clusters/EBI_complexes_v1_old_gene_names.yaml"
CX_OUT = "/hpc/projects/icd.fast.ops/models/diffex/viewer_assets_v5/_rankings/fluor_shap/complex"
TOP_N_CX = 500        # top cells per (channel, complex) for a robust centroid (display cap is separate)


def _gene_to_complex():
    import yaml
    y = yaml.safe_load(open(EBI_YAML)) or {}
    m = {}
    for _, v in y.items():
        for g in v.get("genes", []):
            m[str(g)] = v["name"]
    return m


def build_complex():
    """Pool the per-gene EBI shap cells into per-complex centroid rankings (gene col = complex name), same
    schema build_marker_complex consumes. FAIL LOUD if any non-NTC gene has no complex."""
    os.makedirs(CX_OUT, exist_ok=True)
    g2c = _gene_to_complex()
    use = ["gene", "channel_name", "rank", "shap", "experiment", "well", "x_pheno", "y_pheno", "segmentation_id"]
    print(f"[shap-cx] reading {EBI_CSV} ...", flush=True)
    df = pd.read_csv(EBI_CSV, usecols=use)
    df = df[~df["gene"].astype(str).str.startswith("NTC")]          # NTC = control anchor, not a complex target
    df["complex"] = df["gene"].astype(str).map(g2c)
    unmapped = sorted(df.loc[df["complex"].isna(), "gene"].astype(str).unique())
    if unmapped:                                                    # no good reason a screened gene lacks a complex
        raise ValueError(f"[shap-cx] {len(unmapped)} gene(s) unmapped in {os.path.basename(EBI_YAML)}: {unmapped}")
    n = 0
    for ch, gch in df.groupby("channel_name"):
        parts = []
        for cx, gcx in gch.groupby("complex"):
            g = (gcx.drop_duplicates(["experiment", "well", "x_pheno", "y_pheno"])   # one row per cell
                    .sort_values("shap", ascending=False).head(TOP_N_CX).copy())     # pool members, re-rank by score
            g["rank"] = range(1, len(g) + 1)
            g["gene"] = cx                                                           # grouping key → complex name
            parts.append(g)
        o = (pd.concat(parts, ignore_index=True)[["channel_name", "gene", "rank", "shap", "experiment", "well",
                                                  "x_pheno", "y_pheno", "segmentation_id"]]
             .rename(columns={"segmentation_id": "segmentation", "shap": "pma_attention"}))
        o["rank_type"] = "top"; o["predicted_class"] = o["gene"]
        p = f"{CX_OUT}/{slugify(ch)}.parquet"; o.to_parquet(p)
        n += 1
        print(f"  [{n:2d}] {slugify(ch):40s} {o.gene.nunique()} complexes, {len(o):>6,} cells -> {os.path.basename(p)}",
              flush=True)
    print(f"[shap-cx] wrote {n} per-marker complex parquets -> {CX_OUT}")
    return {"markers": n}


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
    ap.add_argument("--complex", action="store_true", help="build the EBI complex rankings instead of geneKO")
    a = ap.parse_args()
    fn = build_complex if a.complex else build
    if a.submit:
        from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs
        submit_parallel_jobs(
            jobs_to_submit=[{"name": "fluor_shap_cx" if a.complex else "fluor_shap_split", "func": fn, "kwargs": {}}],
            experiment="diffex_shaprank",
            slurm_params={"slurm_partition": "cpu", "cpus_per_task": 8, "mem_gb": 96, "timeout_min": 60},
            log_dir="diffex_shaprank", wait_for_completion=False)
    else:
        fn()
