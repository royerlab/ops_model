"""
Script to inspect and visualize batches from the DynaCLR dataloader.
Run cells interactively to visualize different batches.
"""

# %%
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from ops_model.data import data_loader

# %%
CONFIG_PATH = Path(
    "/home/eduardo.hirata/repos/ops_model/experiments/models/dynaclr/phase_only/dynaclr_phase_only.yml"
)
BATCH_SIZE = 64
SPLIT = "train"

# %%
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

config["data_manager"]["batch_size"] = BATCH_SIZE

labels_path = config["data_manager"]["labels_df_path"]
if labels_path.endswith(".parquet"):
    labels_df = pd.read_parquet(labels_path)
else:
    labels_df = pd.read_csv(labels_path, low_memory=False)
print(f"Input rows: {len(labels_df)}")
if "sample_weight" in labels_df.columns:
    labels_df.pop("sample_weight")

experiments = labels_df["store_key"].unique().tolist()
experiment_dict = {a: [] for a in experiments}

data_manager = data_loader.OpsDataManager(
    experiments=experiment_dict,
    batch_size=config["data_manager"]["batch_size"],
    data_split=tuple(config["data_manager"]["data_split"]),
    out_channels=None,
    initial_yx_patch_size=tuple(config["data_manager"]["initial_yx_patch_size"]),
    final_yx_patch_size=tuple(config["data_manager"]["final_yx_patch_size"]),
)

data_manager.construct_dataloaders(
    num_workers=config["data_manager"]["num_workers"],
    labels_df=labels_df,
    balanced_sampling=config["data_manager"].get("balanced_sampling", False),
    balance_col="reporter",
    dataset_type=config["dataset_type"],
    contrastive_kwargs=config["data_manager"].get("contrastive_kwargs"),
)

# Check transform configuration
contrastive_kwargs = config["data_manager"].get("contrastive_kwargs", {})
print(f"\n## Transform Configuration")
if "transforms" in contrastive_kwargs:
    print(
        f"✓ Using transforms from config ({len(contrastive_kwargs['transforms'])} total):"
    )
    for i, t in enumerate(contrastive_kwargs["transforms"]):
        transform_name = t["class_path"].split(".")[-1]
        init_args = t.get("init_args", {})
        prob = init_args.get("prob", 1.0)
        print(f"  {i + 1}. {transform_name}", end="")
        if prob < 1.0:
            print(f" (prob={prob})", end="")
        print()
else:
    print(f"→ Using default hardcoded transforms")

dataloader = data_manager.train_loader
data_iter = iter(dataloader)

# Get the reverse lookup table to convert integer labels back to gene names
label_str_lut = data_manager.int_label_lut

train_dataset_size = len(data_manager.train_loader.dataset)
val_dataset_size = len(data_manager.val_loader.dataset)
test_dataset_size = len(data_manager.test_loader.dataset)
total_dataset_size = train_dataset_size + val_dataset_size + test_dataset_size

print(f"\n## Dataloader Info")
print(f"  Train loader batches: {len(data_manager.train_loader)}")
print(f"  Train dataset size: {train_dataset_size}")
print(f"  Val dataset size: {val_dataset_size}")
print(f"  Test dataset size: {test_dataset_size}")
print(f"  Total dataset size: {total_dataset_size}")
print(f"  Data split ratio: {config['data_manager']['data_split']}")
print(
    f"\nExpected train samples: {int(len(labels_df) * config['data_manager']['data_split'][0])}"
)
print(f"Actual train samples: {train_dataset_size}")
print(
    f"Data loss: {len(labels_df) - total_dataset_size} cells ({100 * (len(labels_df) - total_dataset_size) / len(labels_df):.1f}%)"
)

# %%
# Re-run this cell to visualize a new batch (anchor vs positive side by side)
batch = next(data_iter)

anchor_images = batch["anchor"].numpy()
positive_images = batch["positive"].numpy()
gene_labels = batch["gene_label"]
marker_labels = batch["marker_label"]
crop_info = batch["crop_info"]

print(
    f"\nPositive source mode: {config['data_manager']['contrastive_kwargs']['positive_source']}"
)
print(f"\nPair verification for first 5 samples:")
for i in range(min(5, len(gene_labels))):
    anchor_gene_idx = gene_labels[i]["anchor"]
    positive_gene_idx = gene_labels[i]["positive"]
    anchor_gene = label_str_lut[anchor_gene_idx]
    positive_gene = label_str_lut[positive_gene_idx]
    anchor_idx = crop_info[i]["anchor"]["total_index"]
    positive_idx = crop_info[i]["positive"]["total_index"]
    anchor_sgrna = crop_info[i]["anchor"]["sgRNA"]
    positive_sgrna = crop_info[i]["positive"]["sgRNA"]
    same_cell = anchor_idx == positive_idx
    same_sgrna = anchor_sgrna == positive_sgrna

    print(
        f"  Sample {i}: anchor={anchor_gene} (idx={anchor_idx}, sgRNA={anchor_sgrna[:10]}...)"
    )
    print(
        f"            positive={positive_gene} (idx={positive_idx}, sgRNA={positive_sgrna[:10]}...)"
    )
    print(f"            same_cell={same_cell}, same_sgRNA={same_sgrna}")
    print()

print(f"\nNormalization check for first sample:")
print(f"Channel: {marker_labels[0][0]}")
print(
    f"Anchor - min: {anchor_images[0].min():.3f}, max: {anchor_images[0].max():.3f}, mean: {anchor_images[0].mean():.3f}, std: {anchor_images[0].std():.3f}"
)
print(
    f"Positive - min: {positive_images[0].min():.3f}, max: {positive_images[0].max():.3f}, mean: {positive_images[0].mean():.3f}, std: {positive_images[0].std():.3f}"
)

num_samples = 10
fig, axes = plt.subplots(2, num_samples, figsize=(20, 4))

for i in range(num_samples):
    for row, (img_data, img_type) in enumerate(
        [(anchor_images[i], "anchor"), (positive_images[i], "positive")]
    ):
        img = img_data
        channel_idx = None
        if img.ndim == 4:
            channel_idx = img.shape[1] // 2
            img = img[0, channel_idx]
        elif img.ndim == 3:
            channel_idx = 0
            img = img[0]

        axes[row, i].imshow(img, cmap="gray")
        channel_name = marker_labels[i][0] if marker_labels[i] else "unknown"
        gene_name = label_str_lut[gene_labels[i][img_type]]
        title = f"{gene_name}\n{img_type}: {channel_name}"
        axes[row, i].set_title(title, fontsize=8)
        axes[row, i].set_xticks([])
        axes[row, i].set_yticks([])

plt.tight_layout()
plt.show()

# %%
