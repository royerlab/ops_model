"""Split Alex's new shap_screen PHASE rankings into the pma_v5 parquet format the phase traversal build
consumes (single Phase2D channel → one parquet each). NON-DESTRUCTIVE: writes to pma_shap_phase_* (the
production pma_v5_phase_* are left in place) so we can validate before repointing GRAINS.

  geneKO  → pma_shap_phase_geneKO.parquet   (schema: gene, experiment, well, x_pheno, y_pheno, segmentation,
                                             pma_attention, rank, rank_type)
  complex → pma_shap_phase_complex.parquet  (adds predicted_class=complex, gene=member gene; EBI-pooled)

  python -m ops_model.models.interpretability.diffex.viewer.build_phase_shap_rankings --geneko  --submit
  python -m ops_model.models.interpretability.diffex.viewer.build_phase_shap_rankings --complex --submit
"""
from __future__ import annotations

import argparse
import os

import pandas as pd

M = "/hpc/projects/icd.fast.ops/models/alex_lin_attention/v5/multi_rank"
GENEKO_CSV = f"{M}/shap_screen_phase_all.csv"
EBI_CSV = f"{M}/shap_screen_ebi_phase_all.csv"
# the shap CSVs use OLD gene symbols → old-names yaml (the updated one silently drops the tRNA-synthetase complex)
EBI_YAML = "/hpc/projects/icd.fast.ops/configs/gene_clusters/EBI_complexes_v1_old_gene_names.yaml"
RANK = "/hpc/projects/icd.fast.ops/models/diffex/viewer_assets_v5/_rankings"
GENEKO_OUT = f"{RANK}/pma_shap_phase_geneKO.parquet"
COMPLEX_OUT = f"{RANK}/pma_shap_phase_complex.parquet"
TOP_N_CX = 500


def _gene_to_complex():
    import yaml
    y = yaml.safe_load(open(EBI_YAML)) or {}
    return {str(g): v["name"] for _, v in y.items() for g in v.get("genes", [])}


def build_geneko():
    use = ["gene", "rank", "shap", "experiment", "well", "x_pheno", "y_pheno", "segmentation_id"]
    print(f"[phase-shap] reading {GENEKO_CSV} ...", flush=True)
    df = (pd.read_csv(GENEKO_CSV, usecols=use)
          .rename(columns={"shap": "pma_attention", "segmentation_id": "segmentation"}))
    df["rank_type"] = "top"
    df = df[["gene", "experiment", "well", "x_pheno", "y_pheno", "segmentation", "pma_attention", "rank", "rank_type"]]
    df.reset_index(drop=True).to_parquet(GENEKO_OUT)
    print(f"[phase-shap] geneKO: {len(df):,} rows, {df.gene.nunique()} classes -> {GENEKO_OUT}")
    return {"rows": int(len(df)), "classes": int(df.gene.nunique())}


def build_complex():
    g2c = _gene_to_complex()
    use = ["gene", "rank", "shap", "experiment", "well", "x_pheno", "y_pheno", "segmentation_id"]
    print(f"[phase-shap] reading {EBI_CSV} ...", flush=True)
    df = pd.read_csv(EBI_CSV, usecols=use)
    df = df[~df["gene"].astype(str).str.startswith("NTC")]
    df["complex"] = df["gene"].astype(str).map(g2c)
    unmapped = sorted(df.loc[df["complex"].isna(), "gene"].astype(str).unique())
    if unmapped:                                            # no good reason a screened gene lacks a complex
        raise ValueError(f"[phase-shap] {len(unmapped)} gene(s) unmapped in {os.path.basename(EBI_YAML)}: {unmapped}")
    parts = []
    for cx, gcx in df.groupby("complex"):
        g = (gcx.drop_duplicates(["experiment", "well", "x_pheno", "y_pheno"])
                .sort_values("shap", ascending=False).head(TOP_N_CX).copy())
        g["rank"] = range(1, len(g) + 1)
        g["predicted_class"] = cx
        parts.append(g)
    o = (pd.concat(parts, ignore_index=True)
         .rename(columns={"shap": "pma_attention", "segmentation_id": "segmentation"})
         [["predicted_class", "gene", "experiment", "well", "segmentation", "x_pheno", "y_pheno",
           "pma_attention", "rank"]])
    o["rank_type"] = "top"
    o.to_parquet(COMPLEX_OUT)
    print(f"[phase-shap] complex: {len(o):,} rows, {o.predicted_class.nunique()} complexes -> {COMPLEX_OUT}")
    return {"rows": int(len(o)), "complexes": int(o.predicted_class.nunique())}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--geneko", action="store_true")
    ap.add_argument("--complex", action="store_true")
    ap.add_argument("--submit", action="store_true")
    a = ap.parse_args()
    fns = ([build_geneko] if a.geneko else []) + ([build_complex] if a.complex else [])
    if not fns:
        ap.error("pass --geneko and/or --complex")
    if a.submit:
        from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs
        submit_parallel_jobs(
            jobs_to_submit=[{"name": f"phase_shap_{fn.__name__}", "func": fn, "kwargs": {}} for fn in fns],
            experiment="diffex_shaprank",
            slurm_params={"slurm_partition": "cpu", "cpus_per_task": 8, "mem_gb": 200, "timeout_min": 90},
            log_dir="diffex_shaprank", wait_for_completion=False)
    else:
        for fn in fns:
            fn()
