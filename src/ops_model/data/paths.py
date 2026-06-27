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

    @classmethod
    def model_checkpoints_dir(cls) -> Path:
        """Root directory holding all model checkpoints."""
        return cls._resolve_base() / "models" / "model_checkpoints"

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
