# %%
import shutil
from pathlib import Path

from ops_model.data.paths import OpsPaths

# For a list of experiments, copy linked_pheno_iss.csv files to a shared folder for model training

# Configuration
experiments = [
    "ops0015_20250213",
    "ops0016_20250220",
    "ops0031_20250424",
    "ops0032_20250428",
    "ops0033_20250429",
    "ops0035_20250501",
    "ops0036_20250505",
    "ops0037_20250506",
    "ops0038_20250514",
    "ops0044_20250602",
    "ops0045_20250603",
    "ops0046_20250611",
    "ops0048_20250616",
    "ops0049_20250626",
    "ops0052_20250702",
    "ops0054_20250710",
    "ops0055_20250715",
    "ops0056_20250721",
    "ops0057_20250722",
    "ops0058_20250805",
    "ops0059_20250804",
    "ops0062_20250729",
    "ops0063_20250731",
    "ops0064_20250811",
    "ops0065_20250812",
    "ops0066_20250820",
    "ops0068_20250901",
    "ops0069_20250902",
    "ops0070_20250908",
    "ops0071_20250828",
    "ops0076_20250917",
    "ops0078_20250923",
    "ops0079_20250916",
    "ops0081_20250924",
    "ops0084_20251022",
    "ops0085_20251118",
    "ops0086_20250922",
    "ops0089_20251119",
    "ops0090_20251120",
    "ops0091_20251117",
    "ops0092_20251027",
    "ops0097_20251023",
    "ops0098_20251113",
    "ops0100_20251218",
    "ops0101_20251211",
    "ops0102_20251210",
    "ops0103_20251216",
    "ops0104_20251215",
    "ops0105_20260106",
    "ops0106_20251204",
    "ops0107_20251208",
    "ops0108_20251209",
    "ops0110_20260108",
    "ops0113_20251219",
    "ops0114_20260112",
    "ops0115_20260121",
    "ops0116_20260120",
    # Add more experiment names here
]
wells = ["A/1/0", "A/2/0", "A/3/0"]

# Source directory where experiment folders are located
source_base_path = Path("/hpc/projects/intracellular_dashboard/ops")

# Destination directory for copied files
destination_path = Path("/hpc/mydata/alexander.hillsley/ops/training_data")

# Create destination directory if it doesn't exist
destination_path.mkdir(parents=True, exist_ok=True)

# For each experiment in list of experiments
for experiment in experiments:
    print(f"Processing experiment: {experiment}")
    for well in wells:
        print(f" - Processing well: {well}")
        path_obj = OpsPaths(experiment=experiment, well=well)

        # Create a subdir at destination_path with experiment name
        experiment_dest_dir = path_obj.links["training"].parent
        experiment_dest_dir.mkdir(parents=True, exist_ok=True)

        # Source path for this experiment's linked_pheno_iss.csv
        # Adjust this path structure based on your actual directory layout
        source_file = path_obj.links["original"]

        # Check if source file exists
        if source_file.exists():
            # Copy the linked_pheno_iss.csv file to the new subdir
            dest_file = path_obj.links["training"]
            shutil.copy2(source_file, dest_file)
            print(f"   Copied {source_file} to {dest_file}")
        else:
            print(f"   File not found: {source_file}")

print("\nCopying complete!")

# %%
