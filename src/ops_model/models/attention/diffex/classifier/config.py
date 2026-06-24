"""Config for the DiffEx single-cell classifier PoC (options B and C).

One classifier that scores top-attention cells, giving DiffEx its differentiable
image->class-k signal. PoC = binary {gene}-vs-rest on phase crops.
"""
from __future__ import annotations

from dataclasses import dataclass

_V4 = "/hpc/projects/icd.fast.ops/models/alex_lin_attention/v4"

# Phase per-cell attention exports (Alex v4). Same schema:
#   gene, channel(=Phase2D), model_confidence, predicted_class, experiment,
#   well, segmentation, x_pheno, y_pheno, pma_attention, rank, rank_type.
# geneKO: class is `gene`. complex (EBI): class is `predicted_class` (complex name).
PMA_PHASE_GENEKO = f"{_V4}/pma_phase_cells_v2_all.parquet"
PMA_PHASE_EBI = f"{_V4}/pma_phase_cells_ebi_all.parquet"

GRAINS = {
    "geneKO": {"parquet": PMA_PHASE_GENEKO, "class_col": "gene"},
    "complex": {"parquet": PMA_PHASE_EBI, "class_col": "predicted_class"},
}

# Default output root; per-run results land under <root>/<grain>/<class-slug>/.
DEFAULT_OUT_ROOT = "/hpc/projects/icd.fast.ops/models/diffex"


def slugify(name: str) -> str:
    """Filesystem-safe tag for a class name (complex names have spaces/slashes)."""
    return "".join(c if c.isalnum() else "_" for c in str(name)).strip("_")


@dataclass
class Config:
    # ---- task / data (locked with Gav 2026-06-16) ----
    gene: str = "HSPA5"          # positive class VALUE (a gene or a complex name)
    class_col: str = "gene"      # parquet column to match `gene` against ("gene" | "predicted_class")
    n_per_class: int = 1000      # top-N attention cells per class
    neg_rank_max: int = 5        # negatives = top-`neg_rank_max` cells of OTHER genes (distinct)
    crop_size: int = 160         # single-cell crop (px)
    channel: str = "Phase2D"     # phase-only PoC
    mask_cell: bool = False      # no seg mask — usually better (keeps full crop context)

    # ---- train/val/test split (grouped by experiment by default) ----
    split_mode: str = "experiment"  # "experiment" (grouped, confound guard) | "random"
    val_fraction: float = 0.15      # model selection (best epoch)
    test_fraction: float = 0.15     # clean held-out number (never used for selection)
    seed: int = 0

    # ---- training ----
    epochs: int = 30
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    device: str = "cuda"
    num_workers: int = 0  # crop materialization is a one-time zarr read; 0 = fork-safe

    # ---- paths ----
    pma_parquet: str = PMA_PHASE_GENEKO
