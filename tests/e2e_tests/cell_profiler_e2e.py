"""End-to-end test for the CellProfiler feature-extraction pipeline.

Self-contained script (run directly, not via pytest). It exercises the full
production path for one minimal example:

    1. subset a real per-well link CSV to one cell per gene KO, saved to a tmp dir
    2. point an inline config at that tmp dir (link_csv_dir) + tmp output_dir
    3. run extraction normally (extract_cp_features_parallel, run locally over the
       whole minimal subset rather than via SLURM array jobs) -> cp_features.csv
    4. convert to AnnData (process_features_csv, CellProfiler branch: split by
       reporter) -> per-reporter cell/guide/gene .h5ad
    5. verify the outputs at each step

Unlike the embedding extractors, CellProfiler extraction is index-based and
reads its labels from a DataFrame, so the subsetted link CSV is loaded through
OpsDataManager (link_csv_dir) to build that DataFrame. Crops/measurements come
from the real phenotyping_v3.zarr; only the link CSV is subset. All outputs go
to a fresh tmp dir (printed at the end for inspection).

Requires a GPU node (granularity is computed on GPU workers).

Run with:
    uv run python tests/e2e_tests/cell_profiler_e2e.py
"""

import tempfile
from pathlib import Path

import pandas as pd
import yaml

from ops_model.data import data_loader
from ops_model.features.cp_extraction import extract_cp_features_parallel
from ops_model.features.processing_common import process_features_csv

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

EXPERIMENT = "ops0031_20250424"
WELL = "A/1/0"
WELL_PREFIX = WELL[0] + WELL[2]  # "A/1/0" -> "A1"
# Multi-channel so the CellProfiler reporter-split branch is exercised.
OUT_CHANNELS = ["Phase2D"]
GUIDE_COL = "sgRNA"
MAX_GENES = 32  # cap (one cell per gene) to keep the run fast
NUM_WORKERS = 2  # CPU extraction workers (tiny dataset; keep small)

ASSEMBLY_DIR = Path(f"/hpc/projects/icd.fast.ops/{EXPERIMENT}/3-assembly")
SOURCE_LINK_CSV = ASSEMBLY_DIR / f"{WELL_PREFIX}_linked_pheno_iss.csv"
PHENOTYPING_ZARR = ASSEMBLY_DIR / "phenotyping_v3.zarr"


def build_config(link_dir: Path, output_dir: Path) -> dict:
    """Inline config; mirrors experiments/embedding/configs/cell-profiler/*.yml."""
    return {
        "model_type": "cellprofiler",
        "dataset_type": "basic",
        "data_manager": {
            "experiments": {EXPERIMENT: [WELL]},
            "batch_size": 1,
            "data_split": [1, 0, 0],  # CP processes the train split
            "out_channels": OUT_CHANNELS,
            "initial_yx_patch_size": [256, 256],
            "final_yx_patch_size": [128, 128],
            "num_workers": 0,
            "link_csv_dir": str(link_dir),
        },
        "output_dir": str(output_dir),
        "cell_type": "A549",
        "processing": {},
        # Skip PCA/UMAP: they'd fail on a tiny one-cell-per-gene dataset.
        "aggregation": {
            "guide_level": {"compute_embeddings": False},
            "gene_level": {"compute_embeddings": False},
        },
    }


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------


def subsample_link_csv(link_dir: Path) -> int:
    """Copy the real link CSV into link_dir, keeping one cell per gene KO.

    Returns the number of genes (== number of cells kept).
    """
    assert SOURCE_LINK_CSV.exists(), f"source link CSV not found: {SOURCE_LINK_CSV}"
    assert PHENOTYPING_ZARR.exists(), f"phenotyping zarr not found: {PHENOTYPING_ZARR}"

    df = pd.read_csv(SOURCE_LINK_CSV)
    gene_col = "gene_name" if "gene_name" in df.columns else "Gene name"
    assert gene_col in df.columns, f"no gene column in {SOURCE_LINK_CSV}"

    # Mirror the loader's basic QC so kept cells survive into a batch.
    df = df.dropna(subset=["segmentation_id", gene_col])

    one_per_gene = df.groupby(gene_col, sort=True).head(1)
    genes = one_per_gene[gene_col].unique()[:MAX_GENES]
    subsampled = one_per_gene[one_per_gene[gene_col].isin(genes)].reset_index(drop=True)

    n_genes = subsampled[gene_col].nunique()
    assert n_genes >= 2, "need at least a couple of genes for the test"
    assert len(subsampled) == n_genes, "expected exactly one cell per gene"

    out_csv = link_dir / f"{WELL_PREFIX}_linked_pheno_iss.csv"
    subsampled.to_csv(out_csv, index=False)
    assert out_csv.exists()
    print(f"[1] Subsampled link CSV: {n_genes} cells (1/gene) -> {out_csv}")
    return n_genes


def run_extraction(link_dir: Path, output_dir: Path) -> Path:
    """Run CellProfiler extraction locally over the whole minimal subset."""
    # Build the labels DataFrame from the subsetted link CSV (via link_csv_dir).
    dm = data_loader.OpsDataManager(
        experiments={EXPERIMENT: [WELL]},
        batch_size=1,
        data_split=(1, 0, 0),
        out_channels=OUT_CHANNELS,
        initial_yx_patch_size=(256, 256),
        link_csv_dir=str(link_dir),
        verbose=False,
        guide_col=GUIDE_COL,
    )
    dm.construct_dataloaders(num_workers=0, dataset_type="cell_profile")
    labels_df = dm.train_loader.dataset.labels_df
    indices = list(range(len(labels_df)))
    del dm  # workers open their own zarr handles

    results_df = extract_cp_features_parallel(
        experiment_dict={EXPERIMENT: [WELL]},
        indices=indices,
        out_channels=OUT_CHANNELS,
        num_workers=NUM_WORKERS,
        labels_df=labels_df,
        guide_col=GUIDE_COL,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    cp_csv = output_dir / "cp_features.csv"
    results_df.to_csv(cp_csv, index=False)
    print(f"[3] Extracted CP features -> {cp_csv}")
    return cp_csv


def verify_feature_csv(cp_csv: Path, n_genes: int) -> int:
    """Verify the extracted CP feature CSV. Returns the cell count."""
    assert cp_csv.exists(), f"feature CSV not produced: {cp_csv}"
    feats = pd.read_csv(cp_csv)
    n_cells = len(feats)
    assert 0 < n_cells <= n_genes, f"unexpected cell count {n_cells} (genes={n_genes})"

    meta_cols = {"label_int", "label_str", GUIDE_COL, "well", "experiment"}
    for col in meta_cols:
        assert col in feats.columns, f"missing metadata column {col}"
    # CellProfiler features are variable-width; just confirm there are some.
    feature_cols = [c for c in feats.columns if c not in meta_cols and "position" not in c]
    assert len(feature_cols) > 0, "no CellProfiler feature columns found"

    print(f"[3] Feature CSV OK: {n_cells} cells x {len(feature_cols)} CP features -> {cp_csv}")
    return n_cells


def verify_anndata(cp_csv: Path) -> None:
    """Verify the per-reporter AnnData outputs from process_features_csv."""
    import anndata as ad

    anndata_dir = cp_csv.parent / "anndata_objects"
    produced = sorted(anndata_dir.glob("features_processed_*.h5ad"))
    assert produced, f"no features_processed_*.h5ad written in {anndata_dir}"

    print(f"[4] AnnData OK: {len(produced)} reporter file(s)")
    for path in produced:
        adata = ad.read_h5ad(path)
        assert adata.n_obs > 0, f"empty cell-level AnnData: {path}"
        assert adata.n_vars > 0, f"no features in: {path}"
        print(f"      {path.name}: {adata.n_obs} x {adata.n_vars}")


def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="cell_profiler_e2e_"))
    link_dir = tmp / "link_csvs"
    link_dir.mkdir(parents=True, exist_ok=True)
    output_dir = tmp / "cell-profiler"

    print(f"Working dir: {tmp}\n")

    # 1. subset + 2. config -> tmp
    n_genes = subsample_link_csv(link_dir)
    config = build_config(link_dir, output_dir)
    config_path = tmp / "config.yml"
    config_path.write_text(yaml.safe_dump(config))
    print(f"[2] Wrote config -> {config_path}")

    # 3. run extraction
    cp_csv = run_extraction(link_dir, output_dir)
    n_cells = verify_feature_csv(cp_csv, n_genes)

    # 4. CSV -> AnnData (CellProfiler branch: split by reporter) + verify
    process_features_csv(str(cp_csv), config_path=str(config_path))
    verify_anndata(cp_csv)

    print(f"\n✓ CellProfiler e2e PASSED ({n_cells} cells). Outputs under: {tmp}")


if __name__ == "__main__":
    main()
