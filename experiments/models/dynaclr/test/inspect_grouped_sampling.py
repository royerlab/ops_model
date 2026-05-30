"""
Test script to verify GroupedBatchSampler + perturbation_n_reporter.
Checks that:
1. All samples in a batch share the same reporter
2. Anchor and positive share the same gene AND reporter
3. Anchor and positive are different cells
"""

# %%
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

from ops_model.data import data_loader

# %%
CONFIG_PATH = Path(
    "/home/eduardo.hirata/repos/ops_model/experiments/models/dynaclr/bag_of_channels/train_bagofchannels.yml"
)
BATCH_SIZE = 64
NUM_BATCHES_TO_CHECK = 10

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

# Override config to use grouped sampling + perturbation_n_reporter
contrastive_kwargs = config["data_manager"].get("contrastive_kwargs", {})
contrastive_kwargs["positive_source"] = "perturbation_n_reporter"

data_manager.construct_dataloaders(
    num_workers=0,
    labels_df=labels_df,
    balanced_sampling=False,
    balance_col=config["data_manager"].get("balance_col", "reporter"),
    grouped_sampling=True,
    grouped_sampling_val=True,
    group_col="reporter",
    dataset_type=config["dataset_type"],
    contrastive_kwargs=contrastive_kwargs,
)

label_str_lut = data_manager.int_label_lut
print(f"Positive source: {contrastive_kwargs['positive_source']}")
print(f"Train batches: {len(data_manager.train_loader)}")
print(f"Val batches: {len(data_manager.val_loader)}")

# %%
print(f"\n## Checking {NUM_BATCHES_TO_CHECK} train batches\n")

all_passed = True
reporter_counts = Counter()

for batch_idx, batch in enumerate(data_manager.train_loader):
    if batch_idx >= NUM_BATCHES_TO_CHECK:
        break

    crop_info = batch["crop_info"]
    gene_labels = batch["gene_label"]
    bs = len(crop_info)

    anchor_reporters = [crop_info[i]["anchor"]["reporter"] for i in range(bs)]
    positive_reporters = [crop_info[i]["positive"]["reporter"] for i in range(bs)]
    anchor_genes = [label_str_lut[gene_labels[i]["anchor"]] for i in range(bs)]
    positive_genes = [label_str_lut[gene_labels[i]["positive"]] for i in range(bs)]
    anchor_ids = [crop_info[i]["anchor"]["total_index"] for i in range(bs)]
    positive_ids = [crop_info[i]["positive"]["total_index"] for i in range(bs)]

    unique_anchor_reporters = set(anchor_reporters)
    all_same_reporter_in_batch = len(unique_anchor_reporters) == 1
    all_pairs_same_gene = all(a == p for a, p in zip(anchor_genes, positive_genes))
    all_pairs_same_reporter = all(
        a == p for a, p in zip(anchor_reporters, positive_reporters)
    )
    any_different_cell = any(a != p for a, p in zip(anchor_ids, positive_ids))

    batch_reporter = list(unique_anchor_reporters)[0] if all_same_reporter_in_batch else "MIXED"
    reporter_counts[batch_reporter] += 1

    gene_dist = Counter(anchor_genes)

    status = "PASS" if (all_same_reporter_in_batch and all_pairs_same_gene and all_pairs_same_reporter) else "FAIL"
    if status == "FAIL":
        all_passed = False

    print(f"Batch {batch_idx}: [{status}] reporter={batch_reporter}, "
          f"genes={dict(gene_dist)}, "
          f"pairs_same_gene={all_pairs_same_gene}, "
          f"pairs_same_reporter={all_pairs_same_reporter}, "
          f"has_different_cells={any_different_cell}")

print(f"\n## Summary")
print(f"All batches passed: {all_passed}")
print(f"Reporters sampled: {dict(reporter_counts)}")

# %%
# Detailed pair inspection + visualization
import matplotlib.pyplot as plt

batch = next(iter(data_manager.train_loader))
anchor_images = batch["anchor"].numpy()
positive_images = batch["positive"].numpy()
crop_info = batch["crop_info"]
gene_labels = batch["gene_label"]

batch_reporter = crop_info[0]["anchor"]["reporter"]
print(f"\n## Batch reporter: {batch_reporter}\n")

for i in range(min(8, len(crop_info))):
    a_gene = label_str_lut[gene_labels[i]["anchor"]]
    p_gene = label_str_lut[gene_labels[i]["positive"]]
    a_rep = crop_info[i]["anchor"]["reporter"]
    p_rep = crop_info[i]["positive"]["reporter"]
    a_idx = crop_info[i]["anchor"]["total_index"]
    p_idx = crop_info[i]["positive"]["total_index"]

    print(f"  [{i}] anchor: {a_gene:10s} | {a_rep:30s} | idx={a_idx}")
    print(f"       positive: {p_gene:10s} | {p_rep:30s} | idx={p_idx}")
    print(f"       same_gene={a_gene == p_gene}, same_reporter={a_rep == p_rep}, different_cell={a_idx != p_idx}")
    print()

# %%
# Visualize anchor (top) vs positive (bottom) — all same reporter, same gene per column
num_samples = min(10, len(crop_info))
fig, axes = plt.subplots(2, num_samples, figsize=(20, 4))

for i in range(num_samples):
    for row, (img_data, img_type) in enumerate(
        [(anchor_images[i], "anchor"), (positive_images[i], "positive")]
    ):
        img = img_data
        if img.ndim == 4:
            img = img[0, img.shape[1] // 2]
        elif img.ndim == 3:
            img = img[0]

        axes[row, i].imshow(img, cmap="gray")
        gene = label_str_lut[gene_labels[i][img_type]]
        rep = crop_info[i][img_type]["reporter"]
        axes[row, i].set_title(f"{gene}\n{img_type}", fontsize=8)
        axes[row, i].set_xticks([])
        axes[row, i].set_yticks([])

fig.suptitle(f"Reporter: {batch_reporter} | Top=anchor, Bottom=positive (same gene, different cell)", fontsize=10)
plt.tight_layout()
plt.show()

# %%
# Visualize multiple batches to see different reporters
fig, axes = plt.subplots(4, 8, figsize=(16, 8))

for batch_idx, batch in enumerate(data_manager.train_loader):
    if batch_idx >= 4:
        break
    crop_info = batch["crop_info"]
    gene_labels = batch["gene_label"]
    images = batch["anchor"].numpy()
    reporter = crop_info[0]["anchor"]["reporter"]

    for col in range(8):
        img = images[col]
        if img.ndim == 4:
            img = img[0, img.shape[1] // 2]
        elif img.ndim == 3:
            img = img[0]

        gene = label_str_lut[gene_labels[col]["anchor"]]
        axes[batch_idx, col].imshow(img, cmap="gray")
        axes[batch_idx, col].set_title(gene, fontsize=7)
        axes[batch_idx, col].set_xticks([])
        axes[batch_idx, col].set_yticks([])

    axes[batch_idx, 0].set_ylabel(reporter[:15], fontsize=8, rotation=0, labelpad=60)

fig.suptitle("4 batches — each row is one reporter, columns are different gene KOs", fontsize=10)
plt.tight_layout()
plt.show()

# %%
