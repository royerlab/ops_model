# %%
import shutil
from pathlib import Path

from ops_model.data.paths import OpsPaths

# For a list of experiments, copy linked_pheno_iss.csv files to a shared folder for model training

# Configuration
experiments = [
    "ops0031_20250424",
    "ops0045_20250603",
    "ops0053_20250709",
    "ops0056_20250721",
    "ops0057_20250722",
    "ops0062_20250729",
    "ops0063_20250731",
    "ops0064_20250811",
    "ops0065_20250812",
    "ops0068_20250910",
    "ops0071_20250828",
    "ops0076_20250911",
    "ops0079_20250916",
    "ops0089_20251119",
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
