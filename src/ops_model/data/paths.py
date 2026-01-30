"""
Scipt to store paths to complete datasets

"""

from pathlib import Path


class OpsPaths:
    def __init__(self, experiment: str, well: str = None):
        self.experiment = experiment
        if well is not None:
            self.well_prefix = self.reformat_well_name(well)
        else:
            self.well_prefix = None

        self.base = Path("/hpc/projects/intracellular_dashboard/ops")

        self.stores = {
            "phenotyping": self.base / self.experiment / "3-assembly/phenotyping.zarr",
            "phenotyping_v3": Path("/hpc/projects/icd.fast.ops")
            / self.experiment
            / "3-assembly/phenotyping_v3.zarr",
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
