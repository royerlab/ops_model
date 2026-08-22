"""
Scipt to store paths to complete datasets

"""

import os
from pathlib import Path


class OpsPaths:
    @staticmethod
    def _resolve_base() -> Path:
        """Base output dir, overridable via the OPS_OUTPUT_BASE_DIR env var."""
        return Path(
            os.environ.get(
                "OPS_OUTPUT_BASE_DIR",
                "/hpc/projects/icd.fast.ops",
            )
        )

    @staticmethod
    def _resolve_model_base() -> Path:
        """Base dir for read-only model assets (checkpoints).

        Anchored to the canonical data root (OPS_BASE_PATH), NOT the output dir:
        models are inputs and must not follow OPS_OUTPUT_BASE_DIR, which
        redirects to rerun/research trees where the checkpoints don't exist.
        Override with OPS_MODELS_BASE_DIR if models live elsewhere.
        """
        return Path(
            os.environ.get(
                "OPS_MODELS_BASE_DIR",
                os.environ.get("OPS_BASE_PATH", "/hpc/projects/icd.fast.ops"),
            )
        )

    @classmethod
    def model_checkpoints_dir(cls) -> Path:
        """Root directory holding all model checkpoints (read-only inputs)."""
        return cls._resolve_model_base() / "models" / "model_checkpoints"

    @classmethod
    def checkpoint(cls, *parts: str) -> Path:
        """Path to a checkpoint under the model_checkpoints root.

        e.g. ``OpsPaths.checkpoint("dinov3", "dinov3_vitl16_pretrain_....pth")``.
        Not experiment-specific, so callable without instantiating.
        """
        return cls.model_checkpoints_dir().joinpath(*parts)

    @classmethod
    def slurm_log_dir(cls, model: str) -> Path:
        """SLURM log directory for a given model's batch jobs."""
        return cls._resolve_base() / "models" / "logs" / model / "slurm_logs"

    def __init__(self, experiment: str, well: str = None):
        self.experiment = experiment
        if well is not None:
            self.well_prefix = self.reformat_well_name(well)
        else:
            self.well_prefix = None

        # Allow override of base directory via environment variable
        self.base = self._resolve_base()

        # Allow override of fast_ops base directory (defaults to base)
        fast_base = Path(
            os.environ.get("OPS_FAST_OUTPUT_BASE_DIR", str(self.base))
        )

        self.stores = {
            "phenotyping": self.base / self.experiment / "3-assembly/phenotyping.zarr",
            "phenotyping_v3": fast_base
            / self.experiment
            / "3-assembly/phenotyping_v3.zarr",
            # Sibling v3 store holding the 7 raw brightfield z-slices as channels
            # (labels symlinked from phenotyping_v3). Built by run_bf_titration_pipeline
            # and read by Cell-DINO inference for the per-slice titration comparison.
            "bf_slices_assembled_v3": fast_base
            / self.experiment
            / "3-assembly/bf_slices_assembled_v3.zarr",
            # Per-experiment denoised fluor-marker v3 store (labels symlinked from
            # phenotyping_v3). Built by run_fluor_denoise_titration_pipeline and read
            # by Cell-DINO for the denoised-vs-raw marker titration. Glob-resolved
            # since the marker is in the name but each experiment has exactly one.
            "denoise_fluor_assembled_v3": next(
                iter(sorted((fast_base / self.experiment
                             / "1-preprocess/live_imaging/reconstruction").glob(
                    "phenotyping_fluor_2d_denoise_*_assembled_v3.zarr"))),
                fast_base / self.experiment / "1-preprocess/live_imaging/reconstruction"
                / "phenotyping_fluor_2d_denoise_assembled_v3.zarr"),
        }

        self.embeddings = {
            "cell_profiler": self.base
            / self.experiment
            / "3-assembly"
            / f"cell-profiler/cellprofiler_features.csv",
        }

        self.links = {
            "original": self.base
            / self.experiment
            / "3-assembly"
            / f"{self.well_prefix}_linked_pheno_iss.csv",
            # Second (post-fixation) imaging pass linked to ISS in its own right.
            # Carries fixed_bbox / fixed_cell_seg_id, so labels.py resolves the mask
            # to cell_seg_fixed rather than the live cell_seg.
            "fixed": self.base
            / self.experiment
            / "3-assembly"
            / f"{self.well_prefix}_linked_pheno_iss_fixed.csv",
            "training": self.base
            / "models"
            / "link_csvs"
            / self.experiment
            / f"{self.well_prefix}_linked_pheno_iss.csv",
        }

        self.other = {
            "gene_library": "/hpc/projects/intracellular_dashboard/ops/configs/annotated_guide_library_123-UpdateJuly28_2025.csv",
        }

    def reformat_well_name(self, well: str) -> str:
        assert self.validate_well_name(well), f"Invalid well name format: {well}"
        return well[0] + well[2]

    def validate_well_name(self, well: str) -> bool:
        if len(well.split("/")) != 3:
            return False
        elif well[0] not in "ABC":
            return False
        elif not well[2].isdigit():
            return False
        elif not well[4].isdigit():
            return False
        return True

# Column names each imaging pass contributes to a link CSV. The fixed-pass CSV is a
# SUPERSET: it carries the live columns (bbox / segmentation_id / cell_seg) alongside
# its own, exactly as the cell-painting CSV carries both pheno and cp columns. So one
# file serves both passes and only the column choice differs per channel.
# x/y are the cell-position columns feature extraction reads as x_pheno/y_pheno.
# The live CSV names them that already; the fixed CSV (cell-painting-style schema)
# has x_pheno_centroid for the LIVE cell plus x_fixed for the fixed one -- a
# fixed-pass feature must use the fixed centroid.
_PASS_COLUMNS = {
    "live":  {"bbox": "bbox",       "seg_id": "segmentation_id",   "mask": "cell_seg",
              "x": "x_pheno", "y": "y_pheno"},
    "fixed": {"bbox": "fixed_bbox", "seg_id": "fixed_cell_seg_id", "mask": "cell_seg_fixed",
              "x": "x_fixed", "y": "y_fixed"},
}


def pass_for_channels(channels) -> str:
    """"fixed" when every real channel is from the reimage pass, else "live"."""
    chans = [channels] if isinstance(channels, str) else list(channels or [])
    real = [c for c in chans if c not in ("all", "random")]
    return "fixed" if real and all(str(c).endswith("_fixed") for c in real) else "live"


def link_csv_for_channels(exp_name, well, channels=None, link_csv_dir=None):
    """Link CSV for a well, preferring the fixed-pass superset when it exists.

    Only a fixed-channel job switches file. The fixed CSV does carry the live columns,
    but it has a row only where a fixed cell was found, so reading it for a LIVE job
    silently drops the live-only cells (12,129 of 350,109 on ops0185 A/3). Each pass
    therefore reads the CSV that is complete for it.
    """
    from pathlib import Path

    if link_csv_dir is not None:
        prefix = well[0] + well[2]
        return Path(link_csv_dir) / f"{prefix}_linked_pheno_iss.csv"

    paths = OpsPaths(exp_name, well=well)
    if pass_for_channels(channels) != "fixed":
        return paths.links["original"]
    fixed = Path(paths.links["fixed"])
    if not fixed.exists():
        raise FileNotFoundError(
            f"Channels {channels} are from the reimage pass but its link CSV is "
            f"missing: {fixed}. Run the link step with --mode fixed first; the live "
            f"CSV has no fixed_bbox/cell_seg_fixed, so patches would be cut with the "
            f"live mask."
        )
    return fixed


def select_pass_columns(df, channels):
    """Point the generic bbox / segmentation_id / mask_label columns at one pass.

    A fixed-channel job must be cut with the reimage pass's own geometry: cells move
    and shrink on fixation, so live bboxes and the live cell_seg would sample the
    wrong pixels. Returns df unchanged for live channels (or when the fixed columns
    are absent, i.e. an experiment with no reimage pass).
    """
    which = pass_for_channels(channels)
    cols = _PASS_COLUMNS[which]
    if which == "live":
        df["mask_label"] = df.get("mask_label", cols["mask"])
        return df
    missing = [c for c in (cols["bbox"], cols["seg_id"]) if c not in df.columns]
    if missing:
        raise KeyError(
            f"Channels {list(channels)} are from the reimage pass but the link CSV "
            f"lacks {missing}. Regenerate the link with the fixed pass included -- "
            f"using the live columns would cut patches with the live mask."
        )
    df = df.copy()
    df["bbox"] = df[cols["bbox"]]
    df["segmentation_id"] = df[cols["seg_id"]]
    df["mask_label"] = cols["mask"]
    # Extraction reads x_pheno / y_pheno; point them at this pass's centroid.
    for axis in ("x", "y"):
        src = cols[axis]
        if src in df.columns:
            df[f"{axis}_pheno"] = df[src]
    return df
