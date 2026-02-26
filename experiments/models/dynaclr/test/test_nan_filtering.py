"""
Test script to identify and filter crops with NaN/invalid data.
Run this BEFORE training to clean your labels_df.
"""

# %%
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from ops_model.data.qc.qc_labels import filter_invalid_crops, filter_small_bboxes
from ops_model.data.paths import OpsPaths

CONFIG_PATH = Path(
    "/home/eduardo.hirata/repos/ops_model/experiments/models/dynaclr/bag_of_channels/train_bagofchannels.yml"
)

print("=" * 80)
print("TESTING INVALID CROP FILTERING")
print("=" * 80)

# Load config
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

# Load labels
labels_df = pd.read_csv(config["data_manager"]["labels_df_path"])
print(f"\n## Initial Dataset")
print(f"- Total crops: {len(labels_df)}")

# Initialize stores (matching data_loader.py pattern)
import zarr

experiments = labels_df["store_key"].unique().tolist()
stores = {}
for exp in experiments:
    ops_path = OpsPaths(experiment=exp)
    store_path = ops_path.stores["phenotyping_v3"]

    if store_path.exists():
        stores[exp] = zarr.open_group(str(store_path), mode="r")
        print(f"✓ Loaded store for {exp}")
    else:
        print(f"⚠️  Warning: Store not found for {exp} at {store_path}")

print(f"\n## Running QC Filters")

# Filter 1: Small bboxes
print(f"\n### Filter 1: Small Bounding Boxes")
labels_df, num_removed = filter_small_bboxes(labels_df, threshold=10)
print(f"- Removed {num_removed} crops with bbox < 10 pixels")
print(f"- Remaining: {len(labels_df)}")

# Filter 2: Invalid/NaN crops
print(f"\n### Filter 2: Invalid/NaN Data")
print(f"- Checking for crops with >90% NaN pixels...")
print(f"- This may take a few minutes...")

labels_df, num_removed = filter_invalid_crops(
    labels_df,
    stores=stores,
    nan_threshold=0.9,  # Remove if >90% NaN
    check_sample_size=None,  # Check all crops (can set to 1000 for speed test)
)
print(f"- Removed {num_removed} crops with invalid data")
print(f"- Remaining: {len(labels_df)}")

# Save filtered labels
output_path = config["data_manager"]["labels_df_path"].replace(
    ".csv", "_qc_filtered.csv"
)
labels_df.to_csv(output_path, index=False)

print(f"\n## Summary")
print(f"- Final dataset size: {len(labels_df)}")
print(f"- Saved to: {output_path}")
print(f"\n✅ Update your config to use this filtered CSV:")
print(f"```yaml")
print(f"data_manager:")
print(f"  labels_df_path: {output_path}")
print(f"```")

# Close stores
for store in stores.values():
    store.close()

print("\n" + "=" * 80)

# %%
