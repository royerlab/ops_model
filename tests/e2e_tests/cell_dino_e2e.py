"""End-to-end test for the Cell-DINO feature-extraction pipeline.

Self-contained script (run directly, not via pytest). It exercises the full
production path for one minimal example:

    1. subset a real per-well link CSV to one cell per gene KO, saved to a tmp dir
    2. point an inline config at that tmp dir (link_csv_dir) + tmp output_dir
    3. run extraction normally (extract_cell_dino_features) -> feature CSV
    4. convert to AnnData (process_features_csv) -> cell/guide/gene .h5ad
    5. verify the outputs at each step

Crops are read from the real phenotyping_v3.zarr at the subsampled bboxes, so
only the link CSV is subset; nothing else is mutated. All outputs go to a fresh
tmp dir (printed at the end for inspection).

Requires a GPU node with the Cell-DINO checkpoint on disk.

Run with:
    uv run python tests/e2e_tests/cell_dino_e2e.py
"""

import tempfile
from pathlib import Path

import pandas as pd
import yaml

from ops_model.models.cell_dino import extract_cell_dino_features
from ops_model.features.processing_common import process_features_csv

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

EXPERIMENT = "ops0031_20250424"
WELL = "A/1/0"
WELL_PREFIX = WELL[0] + WELL[2]  # "A/1/0" -> "A1"
CHANNEL = "Phase2D"
EXPECTED_DIM = 1024  # Cell-DINO ViT-L/16 embedding width
MAX_GENES = 32  # cap (one cell per gene) to keep the run fast

ASSEMBLY_DIR = Path(f"/hpc/projects/icd.fast.ops/{EXPERIMENT}/3-assembly")
SOURCE_LINK_CSV = ASSEMBLY_DIR / f"{WELL_PREFIX}_linked_pheno_iss.csv"
PHENOTYPING_ZARR = ASSEMBLY_DIR / "phenotyping_v3.zarr"


def build_config(link_dir: Path, output_dir: Path) -> dict:
    """Inline config; mirrors experiments/embedding/configs/cell_dino/ops0031_cell_dino.yml."""
    return {
        "model_type": "cell_dino",
        "embedding_type": "cell_dino",
        "dataset_type": "basic",
        "data_manager": {
            "experiments": {EXPERIMENT: [WELL]},
            "batch_size": 8,
            "data_split": [0, 0, 1],  # everything in the test loader
            "out_channels": [CHANNEL],
            "initial_yx_patch_size": [256, 256],
            "final_yx_patch_size": [128, 128],
            "num_workers": 0,
            "link_csv_dir": str(link_dir),
        },
        "output_dir": str(output_dir),
        "cell_type": "A549",
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


def verify_feature_csv(feature_csv: Path, n_genes: int) -> int:
    """Verify the extracted feature CSV. Returns the cell count."""
    assert feature_csv.exists(), f"feature CSV not produced: {feature_csv}"
    feats = pd.read_csv(feature_csv)
    n_cells = len(feats)
    assert 0 < n_cells <= n_genes, f"unexpected cell count {n_cells} (genes={n_genes})"

    feature_cols = [c for c in feats.columns if str(c).isdigit()]
    assert (
        len(feature_cols) == EXPECTED_DIM
    ), f"expected {EXPECTED_DIM} feature dims, got {len(feature_cols)}"
    for col in ("label_int", "label_str", "sgRNA", "experiment", "well"):
        assert col in feats.columns, f"missing metadata column {col}"

    print(f"[3] Feature CSV OK: {n_cells} cells x {len(feature_cols)} dims -> {feature_csv}")
    return n_cells


def verify_anndata(feature_csv: Path, n_cells: int) -> None:
    """Verify the AnnData outputs from process_features_csv."""
    import anndata as ad

    anndata_dir = feature_csv.parent / "anndata_objects"
    produced = list(anndata_dir.glob("features_processed_*.h5ad"))
    assert produced, f"no features_processed_*.h5ad written in {anndata_dir}"

    reloaded = ad.read_h5ad(produced[0])
    assert reloaded.n_obs == n_cells, "AnnData cell count != feature CSV rows"
    assert reloaded.n_vars == EXPECTED_DIM, "AnnData feature width mismatch"
    print(f"[4] AnnData OK: {reloaded.n_obs} x {reloaded.n_vars} -> {produced[0]}")


def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="cell_dino_e2e_"))
    link_dir = tmp / "link_csvs"
    link_dir.mkdir(parents=True, exist_ok=True)
    output_dir = tmp / "features"

    print(f"Working dir: {tmp}\n")

    # 1. subset + 2. config -> tmp
    n_genes = subsample_link_csv(link_dir)
    config = build_config(link_dir, output_dir)
    config_path = tmp / "config.yml"
    config_path.write_text(yaml.safe_dump(config))
    print(f"[2] Wrote config -> {config_path}")

    # 3. run extraction
    extract_cell_dino_features(config=config)
    feature_csv = output_dir / f"cell_dino_features_{CHANNEL}.csv"
    n_cells = verify_feature_csv(feature_csv, n_genes)

    # 4. CSV -> AnnData + verify
    process_features_csv(str(feature_csv), config_path=str(config_path))
    verify_anndata(feature_csv, n_cells)

    print(f"\n✓ Cell-DINO e2e PASSED. Outputs under: {tmp}")


if __name__ == "__main__":
    main()
