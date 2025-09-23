"""
Scipt to store paths to complete datasets

"""

from pathlib import Path


class OpsPaths:
    def __init__(self, experiment: str, well: str = None):
        self.experiment = experiment

        self.base = Path("/hpc/projects/intracellular_dashboard/ops")

        self.phenotyping = (
            self.base / self.experiment / "3-assembly/phenotyping_og.zarr"
        )

        # ensure that well is a valid format
        if well is not None:
            assert len(well) == 2 and "/" not in well
        self.links = (
            self.base
            / self.experiment
            / "3-assembly"
            / f"{well}_linked_pheno_iss_prob.csv"
        )

        self.cell_profiler_out = (
            self.base
            / self.experiment
            / "3-assembly"
            / f"cell-profiler/cellprofiler_features.csv"
        )
