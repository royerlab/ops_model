"""
Compute mAP (phenotypic activity & distinctiveness) and plot UMAP per complex
for DynaCLR prediction embeddings.

Uses copairs directly for mAP computation and existing anndata_utils for
aggregation + dimensionality reduction.
"""

import time

import anndata as ad
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc
from copairs import map as copairs_map
from sklearn.metrics import silhouette_score, silhouette_samples

from ops_model.features.anndata_utils import compute_embeddings

COMPLEX_GROUPS = {
    "CCT": ["CCT3", "CCT4", "CCT6A", "CCT7", "TCP1"],
    "COP": ["COPA", "COPB1", "COPB2", "COPZ1"],
    "EIF3": ["EIF3A", "EIF3B", "EIF3L", "EIF3M"],
    "HAUS": ["HAUS4", "HAUS5", "HAUS6"],
    "LSM": ["LSM4", "LSM5", "LSM6"],
    "MED": ["MED6", "MED18", "MED21"],
    "POLR3": ["POLR3A", "POLR3B", "POLR3E", "POLR3F"],
    "SNRPD": ["SNRPD1", "SNRPD2", "SNRPD3"],
    "CBWD": ["CBWD1", "CBWD2", "CBWD3"],
}


def assign_complex(gene, groups=COMPLEX_GROUPS):
    for complex_name, members in groups.items():
        if gene in members:
            return complex_name
    return "other"


def load_and_prepare(adata_zarr_path, labels_path):
    """
    Load zarr predictions and join sgRNA/label_str from labels dataframe.

    The zarr obs index contains total_index values (as strings) that map
    back to the labels dataframe.
    """
    t0 = time.time()
    print("Loading zarr...", end=" ", flush=True)
    adata = ad.read_zarr(adata_zarr_path)
    print(f"{adata.shape} ({time.time() - t0:.1f}s)")

    t0 = time.time()
    print("Loading labels + joining sgRNA...", end=" ", flush=True)
    if labels_path.endswith(".parquet"):
        labels_df = pd.read_parquet(labels_path)
    else:
        labels_df = pd.read_csv(labels_path, low_memory=False)

    labels_lookup = labels_df.set_index(
        labels_df["total_index"].astype(str)
    )["sgRNA"]
    adata.obs["sgRNA"] = adata.obs_names.map(labels_lookup)
    adata.obs["label_str"] = adata.obs["perturbation"]

    n_missing = adata.obs["sgRNA"].isna().sum()
    if n_missing > 0:
        print(f"\n  WARNING: {n_missing} cells without sgRNA match, dropping")
        adata = adata[~adata.obs["sgRNA"].isna()].copy()

    print(f"{adata.obs['perturbation'].nunique()} perturbations, "
          f"{adata.obs['sgRNA'].nunique()} sgRNAs ({time.time() - t0:.1f}s)")
    return adata


def _fast_groupby_mean(X, group_labels):
    """
    Fast numpy groupby-mean. Avoids creating a 7M x 768 pandas DataFrame.

    Parameters
    ----------
    X : np.ndarray
        (n_cells, n_features) array
    group_labels : array-like
        Group label per cell (e.g. sgRNA or gene name)

    Returns
    -------
    X_agg : np.ndarray
        (n_groups, n_features)
    unique_labels : np.ndarray
        Ordered group labels matching rows of X_agg
    """
    codes, unique_labels = pd.factorize(group_labels, sort=True)
    n_groups = len(unique_labels)
    X_agg = np.zeros((n_groups, X.shape[1]), dtype=np.float64)
    counts = np.zeros(n_groups, dtype=np.int64)
    np.add.at(X_agg, codes, X)
    np.add.at(counts, codes, 1)
    X_agg /= counts[:, None]
    return X_agg, unique_labels


def aggregate(adata, level):
    """
    Aggregate cell-level to guide/gene level with PCA + UMAP.

    Uses pure numpy groupby-mean instead of pandas to handle 7M+ cells fast.
    """
    group_col = "sgRNA" if level == "guide" else "label_str"

    t0 = time.time()
    print(f"Aggregating to {level} level...", end=" ", flush=True)
    X_agg, unique_labels = _fast_groupby_mean(
        np.asarray(adata.X), adata.obs[group_col].values
    )
    print(f"{X_agg.shape[0]} {level}s x {X_agg.shape[1]} features ({time.time() - t0:.1f}s)")

    adata_agg = ad.AnnData(X=X_agg)
    adata_agg.var_names = adata.var_names
    adata_agg.obs[group_col] = unique_labels
    adata_agg.obs_names = [str(x) for x in unique_labels]

    # Add perturbation column
    if level == "guide":
        sgRNA_to_gene = (
            adata.obs.groupby("sgRNA")["perturbation"].first().to_dict()
        )
        adata_agg.obs["perturbation"] = (
            adata_agg.obs["sgRNA"].map(sgRNA_to_gene)
        )
    else:
        adata_agg.obs["perturbation"] = adata_agg.obs["label_str"]

    t0 = time.time()
    print(f"Computing PCA + UMAP...", end=" ", flush=True)
    adata_agg = compute_embeddings(
        adata_agg,
        n_pca_components=min(128, adata_agg.shape[0] - 1),
        n_neighbors=15,
        compute_pca=True,
        compute_umap=True,
        compute_phate=False,
    )
    print(f"done ({time.time() - t0:.1f}s)")
    return adata_agg


# ---------------------------------------------------------------------------
# copairs mAP (following Alex's pattern from ops_model/post_process/map/map.py)
# ---------------------------------------------------------------------------

def _adata_to_copairs_df(adata):
    """Convert aggregated AnnData to DataFrame for copairs."""
    if hasattr(adata.X, "toarray"):
        features_df = pd.DataFrame(
            adata.X.toarray(), index=adata.obs_names, columns=adata.var_names
        )
    else:
        features_df = pd.DataFrame(
            adata.X, index=adata.obs_names, columns=adata.var_names
        )
    return pd.concat([adata.obs, features_df], axis=1)


def phenotypic_activity(adata_guide):
    """
    Compute phenotypic activity mAP at guide level.

    Positive pairs: same perturbation, different sgRNA.
    Negative pairs: different from NTC.
    """
    t0 = time.time()
    print("Computing phenotypic activity...", end=" ", flush=True)
    df = _adata_to_copairs_df(adata_guide)
    df["is_NTC"] = df["perturbation"] == "NTC"

    meta_cols = [c for c in ["sgRNA", "n_cells", "perturbation", "is_NTC"]
                 if c in df.columns]
    feat_cols = [c for c in df.columns if c not in meta_cols]

    results = copairs_map.average_precision(
        df[meta_cols],
        np.asarray(df[feat_cols]),
        pos_sameby=["perturbation"],
        pos_diffby=["sgRNA"],
        neg_sameby=[],
        neg_diffby=["is_NTC"],
    )
    activity_map = copairs_map.mean_average_precision(
        results, sameby=["perturbation"],
        null_size=1_000_000, threshold=0.05, seed=0,
    )
    activity_map["-log10(p-value)"] = -activity_map["corrected_p_value"].apply(np.log10)
    print(f"done ({time.time() - t0:.1f}s)")
    return activity_map


def phenotypic_distinctiveness(adata_guide, activity_map):
    """
    Compute phenotypic distinctiveness mAP for active perturbations only.

    Positive pairs: same perturbation, different sgRNA.
    Negative pairs: different perturbation.
    """
    t0 = time.time()
    active = activity_map[activity_map["below_corrected_p"]]["perturbation"].tolist()
    print(f"Computing distinctiveness ({len(active)} active)...", end=" ", flush=True)

    adata_filt = adata_guide[adata_guide.obs["perturbation"].isin(active)].copy()
    df = _adata_to_copairs_df(adata_filt)

    meta_cols = [c for c in ["sgRNA", "n_cells", "perturbation"] if c in df.columns]
    feat_cols = [c for c in df.columns if c not in meta_cols]

    results = copairs_map.average_precision(
        df[meta_cols],
        np.asarray(df[feat_cols]),
        pos_sameby=["perturbation"],
        pos_diffby=["sgRNA"],
        neg_sameby=[],
        neg_diffby=["perturbation"],
    )
    dist_map = copairs_map.mean_average_precision(
        results, sameby=["perturbation"],
        null_size=1_000_000, threshold=0.05, seed=0,
    )
    dist_map["-log10(p-value)"] = -dist_map["corrected_p_value"].apply(np.log10)
    print(f"done ({time.time() - t0:.1f}s)")
    return dist_map


# ---------------------------------------------------------------------------
# Silhouette & batch diagnostics
# ---------------------------------------------------------------------------

def compute_silhouette_by_complex(adata_gene, embedding_key="X_pca"):
    """
    Compute silhouette score on gene-level embeddings grouped by complex.

    Parameters
    ----------
    adata_gene : ad.AnnData
        Gene-level aggregated AnnData with PCA computed.
    embedding_key : str
        Key in obsm to use for distance computation.

    Returns
    -------
    dict
        Overall silhouette score and per-complex mean silhouette.
    """
    t0 = time.time()
    print("Computing complex silhouette...", end=" ", flush=True)

    labels = (
        adata_gene.obs["perturbation"]
        .astype(str)
        .map(assign_complex)
        .values
    )
    # Exclude "other" (NTC, etc.) — they have no expected cluster
    mask = labels != "other"
    if mask.sum() < 3:
        print("too few labeled genes, skipping")
        return {"overall": float("nan"), "per_complex": {}}

    X = adata_gene.obsm[embedding_key][mask]
    labels_filt = labels[mask]

    overall = silhouette_score(X, labels_filt, metric="euclidean")
    sample_scores = silhouette_samples(X, labels_filt, metric="euclidean")

    per_complex = {}
    for cpx in sorted(set(labels_filt)):
        cpx_mask = labels_filt == cpx
        per_complex[cpx] = float(np.mean(sample_scores[cpx_mask]))

    print(f"overall={overall:.3f} ({time.time() - t0:.1f}s)")
    return {"overall": overall, "per_complex": per_complex}


def compute_batch_silhouette(
    adata, embedding_key="X_pca", sample_size=10000, seed=42
):
    """
    Compute silhouette score using experiment as label (cell-level).

    High score means batch effects dominate the embedding space.
    Low/negative score means biology dominates. We want this LOW.

    Parameters
    ----------
    adata : ad.AnnData
        Cell-level AnnData with PCA computed and 'experiment' in obs.
    embedding_key : str
        Key in obsm to use.
    sample_size : int
        Subsample for speed (silhouette is O(n^2)).
    seed : int
        Random seed for subsampling.

    Returns
    -------
    float
        Batch silhouette score.
    """
    t0 = time.time()
    print("Computing batch silhouette...", end=" ", flush=True)

    if "experiment" not in adata.obs.columns:
        print("no 'experiment' column, skipping")
        return float("nan")

    n = adata.shape[0]
    if n > sample_size:
        rng = np.random.RandomState(seed)
        idx = rng.choice(n, sample_size, replace=False)
    else:
        idx = np.arange(n)

    X = adata.obsm[embedding_key][idx]
    labels = adata.obs["experiment"].values[idx]

    # Need at least 2 experiments with 2+ cells each
    unique, counts = np.unique(labels, return_counts=True)
    valid = unique[counts >= 2]
    if len(valid) < 2:
        print("too few experiments, skipping")
        return float("nan")

    # Filter to valid experiments
    valid_mask = np.isin(labels, valid)
    X = X[valid_mask]
    labels = labels[valid_mask]

    score = silhouette_score(X, labels, metric="euclidean")
    print(f"{score:.3f} ({time.time() - t0:.1f}s)")
    return score


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_umap_by_complex(adata_gene, save_path=None):
    adata_gene.obs["complex"] = pd.Categorical(
        adata_gene.obs["perturbation"].astype(str).map(assign_complex)
    )
    adata_gene.obs["perturbation"] = pd.Categorical(
        adata_gene.obs["perturbation"].astype(str)
    )

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    sc.pl.umap(adata_gene, color="complex",
               title="UMAP by protein complex", size=1000,
               ax=axes[0], show=False)
    sc.pl.umap(adata_gene, color="perturbation",
               title="UMAP by gene", size=1000,
               ax=axes[1], show=False)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def aggregate_by_experiment_gene(adata):
    """
    Aggregate cell-level embeddings to (experiment x perturbation) level.

    Groups by (experiment, perturbation) using fast numpy groupby-mean,
    producing ~1700 rows (53 experiments x 33 genes) with 768 features.

    Parameters
    ----------
    adata : ad.AnnData
        Cell-level AnnData with 'experiment' and 'perturbation' in obs.

    Returns
    -------
    ad.AnnData
        Aggregated AnnData with obs columns: experiment, perturbation, n_cells.
    """
    t0 = time.time()
    print("Aggregating to (experiment x gene) level...", end=" ", flush=True)

    exp = adata.obs["experiment"].astype(str).values
    pert = adata.obs["perturbation"].astype(str).values
    combo = np.array([f"{e}||{p}" for e, p in zip(exp, pert)])

    X_agg, unique_combos = _fast_groupby_mean(np.asarray(adata.X), combo)

    experiments = np.array([c.split("||")[0] for c in unique_combos])
    perturbations = np.array([c.split("||")[1] for c in unique_combos])

    combo_to_count = dict(zip(*np.unique(combo, return_counts=True)))
    n_cells = np.array([combo_to_count[c] for c in unique_combos])

    adata_agg = ad.AnnData(X=X_agg.astype(np.float32))
    adata_agg.var_names = adata.var_names
    adata_agg.obs["experiment"] = experiments
    adata_agg.obs["perturbation"] = perturbations
    adata_agg.obs["n_cells"] = n_cells
    adata_agg.obs_names = [f"{e}__{p}" for e, p in zip(experiments, perturbations)]

    print(
        f"{adata_agg.shape[0]} (exp x gene) combinations, "
        f"{len(np.unique(experiments))} experiments, "
        f"{len(np.unique(perturbations))} perturbations ({time.time() - t0:.1f}s)"
    )
    return adata_agg


def plot_map_volcano(map_df, title, save_path=None):
    """Volcano plot of mAP results colored by complex."""
    map_df = map_df.copy()
    map_df["complex"] = map_df["perturbation"].apply(assign_complex)

    fig, ax = plt.subplots(figsize=(10, 7))

    nonsig = map_df[~map_df["below_corrected_p"]]
    sig = map_df[map_df["below_corrected_p"]]

    ax.scatter(nonsig["mean_average_precision"], nonsig["-log10(p-value)"],
               c="lightgrey", s=40, alpha=0.7, edgecolors="grey",
               linewidths=0.5, label="Not significant")

    complexes = sorted(sig["complex"].unique())
    cmap = plt.cm.get_cmap("tab10", len(complexes))
    for i, cpx in enumerate(complexes):
        mask = sig["complex"] == cpx
        ax.scatter(sig.loc[mask, "mean_average_precision"],
                   sig.loc[mask, "-log10(p-value)"],
                   c=[cmap(i)], s=60, label=cpx,
                   edgecolors="black", linewidths=0.5)
        for _, row in sig[mask].iterrows():
            ax.annotate(row["perturbation"],
                        (row["mean_average_precision"], row["-log10(p-value)"]),
                        fontsize=7, ha="left", va="bottom",
                        xytext=(3, 3), textcoords="offset points")

    ax.axhline(-np.log10(0.05), color="black", linestyle="--", alpha=0.5)
    active_ratio = map_df["below_corrected_p"].mean()
    ax.set_title(title)
    ax.set_xlabel("mAP")
    ax.set_ylabel("-log10(p-value)")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    ax.text(0.02, 0.98, f"Significant: {100 * active_ratio:.1f}%",
            transform=ax.transAxes, va="top", fontsize=10, fontweight="bold")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
