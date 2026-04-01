"""
Demo: Initialize OpsDataManager, filter labels_df to specific perturbations,
and construct dataloaders.
"""

from monai.transforms import CenterSpatialCropd, Compose, SpatialPadd, ToTensord

from ops_model.data import data_loader

# ---------------------------------------------------------------------------
# 1. Configure experiments
#    experiments: {experiment_name: [list of wells]}
# ---------------------------------------------------------------------------
WELLS = ["A/1/0", "A/2/0", "A/3/0"]

experiments = {
    # "ops0031_20250424": WELLS,
    # "ops0033_20250429": WELLS,
    # "ops0035_20250501": WELLS,
    # "ops0036_20250505": WELLS,
    # "ops0037_20250506": WELLS,
    # "ops0038_20250514": WELLS,
    # "ops0045_20250603": WELLS,
    # "ops0046_20250611": WELLS,
    # "ops0048_20250616": WELLS,
    # "ops0049_20250626": WELLS,
    # "ops0052_20250702": WELLS,
    # "ops0053_20250709": WELLS,
    # "ops0054_20250710": WELLS,
    # "ops0055_20250715": WELLS,
    # "ops0056_20250721": WELLS,
    # "ops0057_20250722": WELLS,
    # "ops0058_20250805": WELLS,
    # "ops0059_20250804": WELLS,
    # "ops0062_20250729": WELLS,
    # "ops0063_20250731": WELLS,
    # "ops0064_20250811": WELLS,
    # "ops0065_20250812": WELLS,
    # "ops0066_20250820": WELLS,
    # "ops0068_20250901": WELLS,
    # "ops0069_20250902": WELLS,
    # "ops0070_20250908": WELLS,
    # "ops0071_20250828": WELLS,
    # "ops0076_20250917": WELLS,
    # "ops0078_20250923": WELLS,
    # "ops0079_20250916": WELLS,
    # "ops0081_20250924": WELLS,
    # "ops0084_20251022": WELLS,
    # "ops0085_20251118": WELLS,
    # "ops0086_20250922": WELLS,
    # "ops0089_20251119": WELLS,
    # "ops0090_20251120": WELLS,
    # "ops0091_20251117": WELLS,
    # "ops0092_20251027": WELLS,
    # "ops0097_20251023": WELLS,
    # "ops0098_20251113": WELLS,
    # "ops0100_20251218": WELLS,
    # "ops0101_20251211": WELLS,
    # "ops0102_20251210": WELLS,
    # "ops0103_20251216": WELLS,
    # "ops0104_20251215": WELLS,
    # "ops0105_20260106": WELLS,
    # "ops0106_20251204": WELLS,
    # "ops0107_20251208": WELLS,
    # "ops0108_20251209": WELLS,
    # "ops0110_20260108": WELLS,
    # "ops0113_20251219": WELLS,
    # "ops0114_20260112": WELLS,
    # "ops0115_20260121": WELLS,
    # "ops0116_20260120": WELLS,
    # "ops0117_20260128": WELLS,
    # "ops0118_20260129": WELLS,
    # "ops0119_20260203": WELLS,
    # "ops0120_20260204": WELLS,
    "ops0121_20260210": WELLS,
    "ops0122_20260211": WELLS,
    "ops0124_20260218": WELLS,
}

# ---------------------------------------------------------------------------
# 2. Initialize OpsDataManager
# ---------------------------------------------------------------------------
dm = data_loader.OpsDataManager(
    experiments=experiments,
    batch_size=32,
    data_split=(0.9, 0.05, 0.05),
    out_channels=["Phase2D"],
    initial_yx_patch_size=(256, 256),
    final_yx_patch_size=(128, 128),
    verbose=False,
)

# ---------------------------------------------------------------------------
# 3. Load all labels and filter to a specific list of perturbations
# ---------------------------------------------------------------------------
labels_df = dm.get_labels()

perturbations_of_interest = ["RPL26", "NTC"] # warning there is a huge class imbalance in favor of NTCs
filtered_df = labels_df[labels_df["gene_name"].isin(perturbations_of_interest)].copy()
filtered_df = filtered_df.reset_index(drop=True)

print(f"Total cells before filtering: {len(labels_df)}")
print(f"Total cells after filtering:  {len(filtered_df)}")
print(f"Perturbations retained: {filtered_df['gene_name'].unique().tolist()}")

# ---------------------------------------------------------------------------
# 4. Construct dataloaders from the filtered labels_df
# ---------------------------------------------------------------------------
dm.construct_dataloaders(
    labels_df=filtered_df,
    num_workers=4,
    dataset_type="basic",
    basic_kwargs={
        "cell_masks": True, # do you want to mask out neighboring cells?
        "transform": Compose(
            [
                SpatialPadd(
                    keys=["data", "mask"],
                    spatial_size=dm.initial_yx_patch_size,
                ),
                CenterSpatialCropd(
                    keys=["data", "mask"],
                    roi_size=dm.final_yx_patch_size,
                ),
                ToTensord(
                    keys=["data", "mask"],
                ),
            ]
        ),
    },
)

print(f"Train batches: {len(dm.train_loader)}")
print(f"Val batches:   {len(dm.val_loader)}")
print(f"Test batches:  {len(dm.test_loader)}")

# ---------------------------------------------------------------------------
# 5. Iterate over a loader
# ---------------------------------------------------------------------------
for batch in dm.test_loader:
    print("data shape:", batch["data"].shape)
    print("gene_label:", batch["gene_label"])
    break
