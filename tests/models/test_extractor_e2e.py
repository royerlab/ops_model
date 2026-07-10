"""End-to-end smoke test for the pretrained feature extractors.

Exercises the full inference path for ``cell_dino``, ``dinov3``, and ``subcell``
against real data:

  1. copy a real per-well link CSV into a tmp dir and subsample it to one cell
     per gene knockout,
  2. run ``extract_*_features`` (crops are pulled from the real
     ``phenotyping_v3.zarr`` at the subsampled bboxes),
  3. convert the resulting feature CSV to AnnData via ``process_features_csv``.

Outputs at each step are verified, and everything is written under pytest's
``tmp_path`` — only the link CSV / zarr / checkpoints are read.

This is a GPU integration test, not a unit test: it needs CUDA and the model
checkpoints on disk. The whole module skips without a GPU, and each model
parameter skips if its checkpoint is missing. It is marked ``slow`` so it is
deselected from the default suite.

Run with:
    uv run pytest tests/models/test_extractor_e2e.py -m slow -v
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pandas as pd
import pytest
import torch
import yaml

from ops_model.models.cell_dino import extract_cell_dino_features
from ops_model.models.dinov3 import extract_dinov3_features
from ops_model.models.subcell import extract_subcell_features
from ops_model.features.processing_common import process_features_csv

# Integration test: heavy + needs a GPU, so keep it out of the default run and
# don't let incidental torch/monai/anndata warnings fail it (the suite treats
# warnings as errors globally).
pytestmark = [pytest.mark.slow, pytest.mark.filterwarnings("ignore")]

# --- Fixed test data (read-only) -------------------------------------------
EXPERIMENT = "ops0031_20250424"
WELL = "A/1/0"
WELL_PREFIX = WELL[0] + WELL[2]  # "A/1/0" -> "A1"
ASSEMBLY_DIR = Path(f"/hpc/projects/icd.fast.ops/{EXPERIMENT}/3-assembly")
SOURCE_LINK_CSV = ASSEMBLY_DIR / f"{WELL_PREFIX}_linked_pheno_iss.csv"
PHENOTYPING_ZARR = ASSEMBLY_DIR / "phenotyping_v3.zarr"

# Cap genes purely to bound runtime; one cell is kept per gene up to this many.
MAX_GENES = 32

_CKPT_ROOT = Path("/hpc/projects/icd.fast.ops/models/model_checkpoints")


@dataclass
class ModelSpec:
    name: str
    extract_fn: Callable
    checkpoint: Path
    out_channels: list  # channels passed to the extractor config
    expected_dim: int  # embedding width
    extra_data_manager: dict = field(default_factory=dict)


MODEL_SPECS = [
    ModelSpec(
        name="cell_dino",
        extract_fn=extract_cell_dino_features,
        checkpoint=_CKPT_ROOT
        / "cell_dino/channel_adaptive_dino_vitl16_pretrain_cells-ef7c17ff.pth",
        out_channels=["Phase2D"],
        expected_dim=1024,
    ),
    ModelSpec(
        name="dinov3",
        extract_fn=extract_dinov3_features,
        checkpoint=_CKPT_ROOT
        / "dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
        out_channels=["Phase2D"],
        expected_dim=1024,
    ),
    ModelSpec(
        name="subcell",
        extract_fn=extract_subcell_features,
        checkpoint=_CKPT_ROOT / "subcell/bg/DNA-Protein_MAE-CellS-ProtS-Pool.pth",
        # SubCell needs [DNA, protein]; ops0031's nucleus channel is a prediction.
        out_channels=["nuclei_prediction", "GFP"],
        expected_dim=1536,
    ),
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def _require_gpu():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device available — extractors require a GPU")


@pytest.fixture(scope="session")
def _require_source_data():
    if not SOURCE_LINK_CSV.exists():
        pytest.skip(f"source link CSV not found: {SOURCE_LINK_CSV}")
    if not PHENOTYPING_ZARR.exists():
        pytest.skip(f"phenotyping zarr not found: {PHENOTYPING_ZARR}")


@pytest.fixture(scope="session")
def subsampled_link_dir(tmp_path_factory, _require_source_data):
    """Copy the real link CSV into tmp, keep one cell per gene KO, save it.

    Returns (link_dir, n_genes). The file keeps the
    ``{well_prefix}_linked_pheno_iss.csv`` name so OpsDataManager picks it up via
    ``link_csv_dir``.
    """
    df = pd.read_csv(SOURCE_LINK_CSV)

    gene_col = "gene_name" if "gene_name" in df.columns else "Gene name"
    assert gene_col in df.columns, f"no gene column in {SOURCE_LINK_CSV}"

    # Mirror the loader's basic QC so the kept cells actually survive to a batch.
    df = df.dropna(subset=["segmentation_id"])
    df = df.dropna(subset=[gene_col])

    # One cell per gene KO, capped for runtime.
    one_per_gene = df.groupby(gene_col, sort=True).head(1)
    genes = one_per_gene[gene_col].unique()[:MAX_GENES]
    subsampled = one_per_gene[one_per_gene[gene_col].isin(genes)].reset_index(drop=True)

    n_genes = subsampled[gene_col].nunique()
    assert n_genes >= 2, "need at least a couple of genes for the test"
    assert len(subsampled) == n_genes, "expected exactly one cell per gene"

    link_dir = tmp_path_factory.mktemp("link_csvs")
    out_csv = link_dir / f"{WELL_PREFIX}_linked_pheno_iss.csv"
    subsampled.to_csv(out_csv, index=False)

    # Step 1 verification: subsampled CSV exists and is 1 row per gene.
    assert out_csv.exists()
    reloaded = pd.read_csv(out_csv)
    assert len(reloaded) == n_genes
    assert reloaded[gene_col].nunique() == n_genes

    return link_dir, n_genes


@pytest.fixture(autouse=True)
def _free_gpu_memory():
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_config(spec: ModelSpec, link_dir: Path, output_dir: Path) -> dict:
    return {
        "model_type": spec.name,
        "embedding_type": spec.name,
        "dataset_type": "basic",
        "data_manager": {
            "experiments": {EXPERIMENT: [WELL]},
            "batch_size": 8,
            "data_split": [0, 0, 1],  # everything in the test loader
            "out_channels": spec.out_channels,
            "initial_yx_patch_size": [256, 256],
            "final_yx_patch_size": [128, 128],
            "num_workers": 0,
            "link_csv_dir": str(link_dir),
            **spec.extra_data_manager,
        },
        "output_dir": str(output_dir),
        "cell_type": "A549",
        # Skip PCA/UMAP — they'd fail on a tiny one-cell-per-gene dataset.
        "aggregation": {
            "guide_level": {"compute_embeddings": False},
            "gene_level": {"compute_embeddings": False},
        },
    }


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("spec", MODEL_SPECS, ids=lambda s: s.name)
def test_extractor_end_to_end(spec, subsampled_link_dir, tmp_path, _require_gpu):
    if not spec.checkpoint.exists():
        pytest.skip(f"{spec.name} checkpoint missing: {spec.checkpoint}")

    link_dir, n_genes = subsampled_link_dir
    output_dir = tmp_path / "features"
    config = _build_config(spec, link_dir, output_dir)

    config_path = tmp_path / "config.yml"
    config_path.write_text(yaml.safe_dump(config))

    # --- Step 2: run inference -> feature CSV --------------------------------
    spec.extract_fn(config=config)

    name_channel = spec.out_channels[-1]  # protein channel for subcell, the sole channel otherwise
    feature_csv = output_dir / f"{spec.name}_features_{name_channel}.csv"
    assert feature_csv.exists(), f"feature CSV not produced: {feature_csv}"

    feats = pd.read_csv(feature_csv)
    n_cells = len(feats)
    assert 0 < n_cells <= n_genes, f"unexpected cell count {n_cells} (genes={n_genes})"

    # Feature columns are the integer-named ones written from the embedding array.
    feature_cols = [c for c in feats.columns if str(c).isdigit()]
    assert len(feature_cols) == spec.expected_dim, (
        f"{spec.name}: expected {spec.expected_dim} feature dims, "
        f"got {len(feature_cols)}"
    )
    for meta_col in ("label_int", "label_str", "sgRNA", "experiment", "well"):
        assert meta_col in feats.columns, f"missing metadata column {meta_col}"

    # --- Step 3: feature CSV -> AnnData --------------------------------------
    adata = process_features_csv(str(feature_csv), config_path=str(config_path))

    anndata_dir = feature_csv.parent / "anndata_objects"
    produced = list(anndata_dir.glob("features_processed_*.h5ad"))
    assert produced, f"no features_processed_*.h5ad written in {anndata_dir}"

    assert adata.n_obs == n_cells, "AnnData cell count != feature CSV rows"
    assert adata.n_vars == spec.expected_dim, "AnnData feature width mismatch"

    # The saved cell-level object loads back and matches.
    import anndata as ad

    reloaded = ad.read_h5ad(produced[0])
    assert reloaded.n_obs == n_cells
    assert reloaded.n_vars == spec.expected_dim
