"""
Classifier-based batch correction for DynaCLR embeddings.

Trains a gene classifier with BatchNorm on (experiment x gene)-level
aggregated embeddings. The penultimate layer provides batch-corrected
gene representations because BatchNorm normalizes activations across
experiments. Compares raw vs corrected metrics.
"""

# %%
import sys
import time
from pathlib import Path

import anndata as ad
import lightning as L
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from evaluate_map import (
    aggregate_by_experiment_gene,
    compute_batch_silhouette,
    compute_silhouette_by_complex,
    load_and_prepare,
    plot_umap_by_complex,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from ops_model.features.anndata_utils import compute_embeddings
from ops_model.models.eval_mlp import (
    AggregatedClassifierDataset,
    LitBatchCorrectionClassifier,
    extract_corrected_embeddings,
)

# ── Paths ──────────────────────────────────────────────────────────────
ZARR_PATH = (
    "/hpc/projects/intracellular_dashboard/ops/models/logs/dynaclr/"
    "ops_bag_of_channels_v1/version_0/predict/"
    "dynaclr_embeddings_bagofchannels_v1_ckpt265.zarr"
)
LABELS_PATH = (
    "/home/eduardo.hirata/repos/ops_model/experiments/models/dynaclr/"
    "test/labels_testset_10complex_n_NTC_v2_filtered.parquet"
)
SAVE_DIR = Path(
    "/hpc/projects/intracellular_dashboard/ops/models/logs/dynaclr/"
    "ops_bag_of_channels_v1/version_0/predict/batch_correction"
)
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# ── Hyperparameters ────────────────────────────────────────────────────
MAX_EPOCHS = 200
BATCH_SIZE = 64
LR = 1e-3
WEIGHT_DECAY = 1e-2
PATIENCE = 20
VAL_SPLIT = 0.15
SEED = 42

# ── 1) Load zarr + join sgRNA from labels ──────────────────────────────
print("=" * 60)
print("STEP 1: Loading data")
print("=" * 60)
adata = load_and_prepare(ZARR_PATH, LABELS_PATH)
print(f"Loaded {adata.shape[0]:,} cells, {adata.shape[1]} features")

# ── 2) Aggregate cell-level → (experiment x gene) ─────────────────────
print("\n" + "=" * 60)
print("STEP 2: Aggregating to (experiment x gene) level")
print("=" * 60)
adata_agg = aggregate_by_experiment_gene(adata)
print(f"Aggregated shape: {adata_agg.shape}")

# ── 3) Compute raw metrics (before correction) ────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Computing raw metrics (before correction)")
print("=" * 60)

n_pca = min(128, adata_agg.shape[0] - 1, adata_agg.shape[1] - 1)
sc.pp.pca(adata_agg, n_comps=n_pca)
raw_batch_sil = compute_batch_silhouette(
    adata_agg, embedding_key="X_pca", sample_size=adata_agg.shape[0]
)

# Gene-level raw (for complex silhouette)
perturbations = adata_agg.obs["perturbation"].values
unique_genes = np.unique(perturbations)
gene_raw = []
for gene in unique_genes:
    mask = perturbations == gene
    gene_raw.append(np.asarray(adata_agg.X[mask]).mean(axis=0))

adata_gene_raw = ad.AnnData(X=np.array(gene_raw))
adata_gene_raw.obs["perturbation"] = list(unique_genes)
adata_gene_raw.obs_names = list(unique_genes)

n_pca_gene = min(128, adata_gene_raw.shape[0] - 1, adata_gene_raw.shape[1] - 1)
adata_gene_raw = compute_embeddings(
    adata_gene_raw,
    n_pca_components=n_pca_gene,
    n_neighbors=min(15, adata_gene_raw.shape[0] - 1),
    compute_pca=True,
    compute_umap=True,
    compute_phate=False,
)
raw_complex_sil = compute_silhouette_by_complex(
    adata_gene_raw, embedding_key="X_pca"
)

# ── 4) Prepare training data ──────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: Preparing training data")
print("=" * 60)

# Exclude NTC from training targets
train_mask = adata_agg.obs["perturbation"] != "NTC"
adata_train = adata_agg[train_mask].copy()

# Create integer labels for non-NTC perturbations
genes_for_training = sorted(adata_train.obs["perturbation"].unique())
gene_to_int = {g: i for i, g in enumerate(genes_for_training)}
num_classes = len(gene_to_int)
labels = np.array([gene_to_int[g] for g in adata_train.obs["perturbation"]])

print(f"Training samples: {adata_train.shape[0]}")
print(f"Number of gene classes: {num_classes}")

# Split by experiment (prevents leakage)
rng = np.random.RandomState(SEED)
experiments = np.unique(adata_train.obs["experiment"].values)
rng.shuffle(experiments)
n_val = max(1, int(len(experiments) * VAL_SPLIT))
val_experiments = set(experiments[:n_val])
train_experiments = set(experiments[n_val:])

train_idx = np.array([
    i for i, e in enumerate(adata_train.obs["experiment"].values)
    if e in train_experiments
])
val_idx = np.array([
    i for i, e in enumerate(adata_train.obs["experiment"].values)
    if e in val_experiments
])

X_all = np.asarray(adata_train.X)
train_dataset = AggregatedClassifierDataset(X_all[train_idx], labels[train_idx])
val_dataset = AggregatedClassifierDataset(X_all[val_idx], labels[val_idx])

print(f"Train: {len(train_dataset)} samples ({len(train_experiments)} experiments)")
print(f"Val:   {len(val_dataset)} samples ({len(val_experiments)} experiments)")
print(f"Val experiments: {sorted(val_experiments)}")

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
)

# ── 5) Train classifier ───────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: Training batch correction classifier")
print("=" * 60)

model = LitBatchCorrectionClassifier(
    num_classes=num_classes,
    input_dim=adata_agg.shape[1],
    lr=LR,
    weight_decay=WEIGHT_DECAY,
)

logger = TensorBoardLogger(save_dir=str(SAVE_DIR), name="batch_correction")

callbacks = [
    EarlyStopping(monitor="loss/val", patience=PATIENCE, mode="min"),
    ModelCheckpoint(
        monitor="loss/val",
        mode="min",
        save_top_k=1,
        filename="best-{epoch}-{acc/val:.3f}",
    ),
]

trainer = L.Trainer(
    max_epochs=MAX_EPOCHS,
    accelerator="auto",
    devices=1,
    logger=logger,
    callbacks=callbacks,
    enable_progress_bar=True,
)

t0 = time.time()
trainer.fit(model, train_loader, val_loader)
train_time = time.time() - t0
print(f"Training completed in {train_time:.1f}s")

# Load best checkpoint
best_ckpt = trainer.checkpoint_callback.best_model_path
if best_ckpt:
    print(f"Loading best checkpoint: {best_ckpt}")
    model = LitBatchCorrectionClassifier.load_from_checkpoint(best_ckpt)

# Get final val accuracy
val_results = trainer.validate(model, val_loader, verbose=False)
val_acc = val_results[0].get("acc/val", float("nan"))
val_loss = val_results[0].get("loss/val", float("nan"))
print(f"Best val accuracy: {val_acc:.3f}")
print(f"Best val loss: {val_loss:.4f}")

# ── 6) Extract corrected embeddings ───────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6: Extracting batch-corrected embeddings")
print("=" * 60)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
adata_corrected_expgene, adata_corrected_gene = extract_corrected_embeddings(
    model, adata_agg
)

print(f"Corrected (exp x gene): {adata_corrected_expgene.shape}")
print(f"Corrected (gene): {adata_corrected_gene.shape}")

# ── 7) Compute corrected metrics ──────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 7: Computing corrected metrics")
print("=" * 60)

# Batch silhouette on corrected (exp x gene) level
n_pca_corr = min(
    128,
    adata_corrected_expgene.shape[0] - 1,
    adata_corrected_expgene.shape[1] - 1,
)
sc.pp.pca(adata_corrected_expgene, n_comps=n_pca_corr)
corrected_batch_sil = compute_batch_silhouette(
    adata_corrected_expgene,
    embedding_key="X_pca",
    sample_size=adata_corrected_expgene.shape[0],
)

# Complex silhouette on corrected gene level
n_pca_gene_corr = min(
    128,
    adata_corrected_gene.shape[0] - 1,
    adata_corrected_gene.shape[1] - 1,
)
adata_corrected_gene = compute_embeddings(
    adata_corrected_gene,
    n_pca_components=n_pca_gene_corr,
    n_neighbors=min(15, adata_corrected_gene.shape[0] - 1),
    compute_pca=True,
    compute_umap=True,
    compute_phate=False,
)
corrected_complex_sil = compute_silhouette_by_complex(
    adata_corrected_gene, embedding_key="X_pca"
)

# ── 8) UMAP plots ─────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 8: Generating UMAP plots")
print("=" * 60)

plot_umap_by_complex(
    adata_gene_raw,
    save_path=str(SAVE_DIR / "umap_raw_gene.png"),
)
plot_umap_by_complex(
    adata_corrected_gene,
    save_path=str(SAVE_DIR / "umap_corrected_gene.png"),
)

# Also do UMAP colored by experiment on (exp x gene) level
adata_corrected_expgene = compute_embeddings(
    adata_corrected_expgene,
    n_pca_components=n_pca_corr,
    n_neighbors=min(15, adata_corrected_expgene.shape[0] - 1),
    compute_pca=True,
    compute_umap=True,
    compute_phate=False,
)

# Save corrected embeddings
adata_corrected_gene.write_zarr(str(SAVE_DIR / "corrected_gene_embeddings.zarr"))
adata_corrected_expgene.write_zarr(
    str(SAVE_DIR / "corrected_expgene_embeddings.zarr")
)

# ── 9) Print comparison table ─────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 9: Results summary")
print("=" * 60)

print("\n## Batch Correction Results\n")
print("| Metric | Raw | Corrected |")
print("|--------|-----|-----------|")
print(f"| Batch silhouette (experiment) | {raw_batch_sil:.3f} | {corrected_batch_sil:.3f} |")
print(f"| Complex silhouette | {raw_complex_sil['overall']:.3f} | {corrected_complex_sil['overall']:.3f} |")
print(f"| Classifier val accuracy | - | {100 * val_acc:.1f}% |")
print(f"| Classifier val loss | - | {val_loss:.4f} |")

batch_sil_delta = corrected_batch_sil - raw_batch_sil
complex_sil_delta = corrected_complex_sil["overall"] - raw_complex_sil["overall"]
print(f"\n**Batch silhouette delta**: {batch_sil_delta:+.3f} ({'improved' if batch_sil_delta < 0 else 'worsened'})")
print(f"**Complex silhouette delta**: {complex_sil_delta:+.3f} ({'improved' if complex_sil_delta > 0 else 'worsened'})")

if raw_complex_sil["per_complex"] and corrected_complex_sil["per_complex"]:
    print("\n### Per-complex silhouette\n")
    print("| Complex | Raw | Corrected | Delta |")
    print("|---------|-----|-----------|-------|")
    for cpx in sorted(raw_complex_sil["per_complex"].keys()):
        raw_s = raw_complex_sil["per_complex"].get(cpx, float("nan"))
        corr_s = corrected_complex_sil["per_complex"].get(cpx, float("nan"))
        delta = corr_s - raw_s
        print(f"| {cpx} | {raw_s:.3f} | {corr_s:.3f} | {delta:+.3f} |")

print(f"\n### Training details\n")
print(f"- Samples: {len(train_dataset)} train / {len(val_dataset)} val")
print(f"- Classes: {num_classes}")
print(f"- Architecture: {model.hparams.input_dim} → {model.hparams.hidden_dims} → {num_classes}")
print(f"- Training time: {train_time:.1f}s")
print(f"- Best checkpoint: {best_ckpt}")
print(f"\nResults saved to `{SAVE_DIR}`")

# %%
