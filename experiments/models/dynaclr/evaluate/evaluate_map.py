"""
Compute mAP (phenotypic activity & distinctiveness) and plot UMAP per complex
for DynaCLR prediction embeddings.

Uses copairs directly for mAP computation and existing anndata_utils for
aggregation + dimensionality reduction.
"""

import time
from pathlib import Path

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

    labels_indexed = labels_df.set_index(
        labels_df["total_index"].astype(str)
    )

    # Join columns from labels that are missing in the zarr obs
    join_cols = ["sgRNA", "reporter", "channel", "well"]
    for col in join_cols:
        if col not in adata.obs.columns:
            adata.obs[col] = adata.obs_names.map(labels_indexed[col])

    adata.obs["label_str"] = adata.obs["perturbation"]

    n_missing = adata.obs["sgRNA"].isna().sum()
    if n_missing > 0:
        print(f"\n  WARNING: {n_missing} cells without sgRNA match, dropping")
        adata = adata[~adata.obs["sgRNA"].isna()].copy()

    print(
        f"{adata.obs['perturbation'].nunique()} perturbations, "
        f"{adata.obs['sgRNA'].nunique()} sgRNAs, "
        f"{adata.obs['reporter'].nunique()} reporters ({time.time() - t0:.1f}s)"
    )
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


def aggregate_guides_per_reporter(adata):
    """
    Aggregate cell-level embeddings to (sgRNA x reporter) guide level.

    For each reporter, groups cells by sgRNA and computes the mean embedding.
    Returns an AnnData where each row is one (sgRNA, reporter) combination.

    Parameters
    ----------
    adata : ad.AnnData
        Cell-level AnnData with 'sgRNA', 'perturbation', and 'reporter' in obs.

    Returns
    -------
    ad.AnnData
        Guide-level AnnData with obs columns: sgRNA, perturbation, reporter,
        organelle, n_cells.
    """
    t0 = time.time()
    print("Aggregating to (sgRNA x reporter) guide level...", end=" ", flush=True)

    sgrna = adata.obs["sgRNA"].astype(str).values
    rep = adata.obs["reporter"].astype(str).values
    combo = np.array([f"{s}||{r}" for s, r in zip(sgrna, rep)])

    X_agg, unique_combos = _fast_groupby_mean(np.asarray(adata.X), combo)

    sgrnas = np.array([c.split("||")[0] for c in unique_combos])
    reporters = np.array([c.split("||")[1] for c in unique_combos])

    combo_to_count = dict(zip(*np.unique(combo, return_counts=True)))
    n_cells = np.array([combo_to_count[c] for c in unique_combos])

    # Map sgRNA -> perturbation
    sgrna_to_gene = (
        adata.obs.groupby("sgRNA")["perturbation"].first().to_dict()
    )

    adata_agg = ad.AnnData(X=X_agg.astype(np.float32))
    adata_agg.var_names = adata.var_names
    adata_agg.obs["sgRNA"] = sgrnas
    adata_agg.obs["reporter"] = reporters
    adata_agg.obs["perturbation"] = [sgrna_to_gene.get(s, "unknown") for s in sgrnas]
    adata_agg.obs["organelle"] = [
        REPORTER_ORGANELLE.get(r, "unknown") for r in reporters
    ]
    adata_agg.obs["n_cells"] = n_cells
    adata_agg.obs_names = [f"{s}__{r}" for s, r in zip(sgrnas, reporters)]

    print(
        f"{adata_agg.shape[0]} (sgRNA x reporter) combos, "
        f"{len(np.unique(reporters))} reporters ({time.time() - t0:.1f}s)"
    )
    return adata_agg


def phenotypic_activity_per_reporter(adata_guide_per_reporter):
    """
    Compute phenotypic activity mAP separately for each reporter.

    Parameters
    ----------
    adata_guide_per_reporter : ad.AnnData
        (sgRNA x reporter) guide-level AnnData from
        ``aggregate_guides_per_reporter``.

    Returns
    -------
    pd.DataFrame
        Per-reporter summary: reporter, organelle, n_significant, pct_significant,
        mean_mAP, median_mAP.
    dict
        Full mAP results keyed by reporter name.
    """
    t0 = time.time()
    reporters = sorted(adata_guide_per_reporter.obs["reporter"].unique())
    print(f"Computing per-reporter phenotypic activity ({len(reporters)} reporters)...")

    all_results = {}
    summary_rows = []

    for rep in reporters:
        mask = adata_guide_per_reporter.obs["reporter"] == rep
        adata_rep = adata_guide_per_reporter[mask].copy()

        # Need at least 2 perturbations with 2+ sgRNAs
        pert_counts = adata_rep.obs.groupby("perturbation")["sgRNA"].nunique()
        valid_perts = pert_counts[pert_counts >= 2].index.tolist()
        if len(valid_perts) < 3:
            print(f"  {rep}: too few valid perturbations ({len(valid_perts)}), skipping")
            continue

        adata_rep = adata_rep[adata_rep.obs["perturbation"].isin(valid_perts)].copy()
        df = _adata_to_copairs_df(adata_rep)
        df["is_NTC"] = df["perturbation"] == "NTC"

        meta_cols = [c for c in ["sgRNA", "n_cells", "perturbation", "reporter",
                                  "organelle", "is_NTC"] if c in df.columns]
        feat_cols = [c for c in df.columns if c not in meta_cols]

        try:
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
            activity_map["reporter"] = rep

            all_results[rep] = activity_map

            n_sig = activity_map["below_corrected_p"].sum()
            n_total = len(activity_map)
            sig_only = activity_map[activity_map["below_corrected_p"]]
            summary_rows.append({
                "reporter": rep,
                "organelle": REPORTER_ORGANELLE.get(rep, "unknown"),
                "n_perturbations": n_total,
                "n_significant": int(n_sig),
                "pct_significant": 100 * n_sig / n_total if n_total > 0 else 0,
                "mean_mAP": float(activity_map["mean_average_precision"].mean()),
                "median_mAP": float(activity_map["mean_average_precision"].median()),
                "mean_mAP_significant": float(sig_only["mean_average_precision"].mean()) if len(sig_only) > 0 else 0,
            })
            print(f"  {rep}: {n_sig}/{n_total} significant, mean mAP={summary_rows[-1]['mean_mAP']:.3f}")
        except Exception as e:
            print(f"  {rep}: ERROR - {e}")
            continue

    df_summary = pd.DataFrame(summary_rows)
    if len(df_summary) > 0:
        df_summary = df_summary.sort_values("pct_significant", ascending=False)

    print(f"Per-reporter activity done ({time.time() - t0:.1f}s)")
    return df_summary, all_results


def phenotypic_distinctiveness_per_reporter(adata_guide_per_reporter, activity_per_reporter):
    """
    Compute phenotypic distinctiveness mAP separately for each reporter.

    Parameters
    ----------
    adata_guide_per_reporter : ad.AnnData
        (sgRNA x reporter) guide-level AnnData.
    activity_per_reporter : dict
        Full activity results keyed by reporter, from
        ``phenotypic_activity_per_reporter``.

    Returns
    -------
    pd.DataFrame
        Per-reporter summary of distinctiveness.
    dict
        Full distinctiveness results keyed by reporter name.
    """
    t0 = time.time()
    reporters = sorted(activity_per_reporter.keys())
    print(f"Computing per-reporter distinctiveness ({len(reporters)} reporters)...")

    all_results = {}
    summary_rows = []

    for rep in reporters:
        act_map = activity_per_reporter[rep]
        active = act_map[act_map["below_corrected_p"]]["perturbation"].tolist()
        if len(active) < 3:
            print(f"  {rep}: too few active perturbations ({len(active)}), skipping")
            continue

        mask = (
            (adata_guide_per_reporter.obs["reporter"] == rep)
            & (adata_guide_per_reporter.obs["perturbation"].isin(active))
        )
        adata_rep = adata_guide_per_reporter[mask].copy()

        df = _adata_to_copairs_df(adata_rep)
        meta_cols = [c for c in ["sgRNA", "n_cells", "perturbation", "reporter",
                                  "organelle"] if c in df.columns]
        feat_cols = [c for c in df.columns if c not in meta_cols]

        try:
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
            dist_map["reporter"] = rep

            all_results[rep] = dist_map

            n_sig = dist_map["below_corrected_p"].sum()
            n_total = len(dist_map)
            sig_only = dist_map[dist_map["below_corrected_p"]]
            summary_rows.append({
                "reporter": rep,
                "organelle": REPORTER_ORGANELLE.get(rep, "unknown"),
                "n_active": n_total,
                "n_distinctive": int(n_sig),
                "pct_distinctive": 100 * n_sig / n_total if n_total > 0 else 0,
                "mean_mAP": float(dist_map["mean_average_precision"].mean()),
                "median_mAP": float(dist_map["mean_average_precision"].median()),
            })
            print(f"  {rep}: {n_sig}/{n_total} distinctive, mean mAP={summary_rows[-1]['mean_mAP']:.3f}")
        except Exception as e:
            print(f"  {rep}: ERROR - {e}")
            continue

    df_summary = pd.DataFrame(summary_rows)
    if len(df_summary) > 0:
        df_summary = df_summary.sort_values("pct_distinctive", ascending=False)

    print(f"Per-reporter distinctiveness done ({time.time() - t0:.1f}s)")
    return df_summary, all_results


def plot_per_reporter_map_summary(df_activity, df_distinctiveness=None, save_dir=None):
    """
    Bar chart of per-reporter mAP (activity and optionally distinctiveness).

    Parameters
    ----------
    df_activity : pd.DataFrame
        Per-reporter activity summary from ``phenotypic_activity_per_reporter``.
    df_distinctiveness : pd.DataFrame, optional
        Per-reporter distinctiveness summary.
    save_dir : str or Path, optional
        Directory to save figure.
    """
    save_dir = Path(save_dir) if save_dir else None

    has_dist = df_distinctiveness is not None and len(df_distinctiveness) > 0
    n_panels = 2 if has_dist else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(8 * n_panels, 10))
    if n_panels == 1:
        axes = [axes]

    # Sort by organelle then by mean_mAP within organelle
    df_act = df_activity.copy()
    df_act["org_idx"] = df_act["organelle"].map(
        {o: i for i, o in enumerate(_ORGANELLE_ORDER)}
    ).fillna(99)
    df_act = df_act.sort_values(["org_idx", "mean_mAP"], ascending=[True, False])

    # Color by organelle
    organelle_cmap = plt.cm.get_cmap("tab20", len(_ORGANELLE_ORDER))
    org_colors = {o: organelle_cmap(i) for i, o in enumerate(_ORGANELLE_ORDER)}

    colors = [org_colors.get(o, "grey") for o in df_act["organelle"]]
    axes[0].barh(range(len(df_act)), df_act["mean_mAP"].values, color=colors)
    axes[0].set_yticks(range(len(df_act)))
    axes[0].set_yticklabels(df_act["reporter"].values, fontsize=7)
    axes[0].set_xlabel("Mean mAP (activity)")
    axes[0].set_title(f"Phenotypic Activity per Reporter\n(% significant shown)")
    axes[0].invert_yaxis()
    for i, (_, row) in enumerate(df_act.iterrows()):
        axes[0].text(
            row["mean_mAP"] + 0.01, i,
            f"{row['pct_significant']:.0f}%",
            va="center", fontsize=6,
        )

    if has_dist:
        df_dist = df_distinctiveness.copy()
        df_dist["org_idx"] = df_dist["organelle"].map(
            {o: i for i, o in enumerate(_ORGANELLE_ORDER)}
        ).fillna(99)
        df_dist = df_dist.sort_values(["org_idx", "mean_mAP"], ascending=[True, False])

        colors_d = [org_colors.get(o, "grey") for o in df_dist["organelle"]]
        axes[1].barh(range(len(df_dist)), df_dist["mean_mAP"].values, color=colors_d)
        axes[1].set_yticks(range(len(df_dist)))
        axes[1].set_yticklabels(df_dist["reporter"].values, fontsize=7)
        axes[1].set_xlabel("Mean mAP (distinctiveness)")
        axes[1].set_title(f"Phenotypic Distinctiveness per Reporter\n(% distinctive shown)")
        axes[1].invert_yaxis()
        for i, (_, row) in enumerate(df_dist.iterrows()):
            axes[1].text(
                row["mean_mAP"] + 0.01, i,
                f"{row['pct_distinctive']:.0f}%",
                va="center", fontsize=6,
            )

    # Legend for organelle colors
    handles = [
        plt.Line2D([0], [0], marker="s", color="w",
                    markerfacecolor=org_colors.get(o, "grey"),
                    markersize=8, label=o)
        for o in _ORGANELLE_ORDER if o in df_act["organelle"].values
    ]
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=7,
               bbox_to_anchor=(0.5, -0.05))

    plt.tight_layout()
    if save_dir:
        fig.savefig(save_dir / "map_per_reporter.png", dpi=150, bbox_inches="tight")
    plt.show()


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


# ---------------------------------------------------------------------------
# Reporter / organelle groupings (using names as they appear in labels_df)
# ---------------------------------------------------------------------------

REPORTER_ORGANELLE = {
    # ER
    "SEC61B": "ER",
    "5xUPRE": "ER",
    "NCLN": "ER",
    # ER/Golgi transport
    "COPE": "ER/Golgi",
    "COP-II": "ER/Golgi",
    "VAPA": "ER/Golgi",
    "VAMP3": "ER/Golgi",
    # Mitochondria
    "TOMM70A": "Mitochondria",
    "TOMM20": "Mitochondria",
    "ChromaLIVE_561_excitation": "Mitochondria",
    "ChromaLIVE_488_excitation": "Mitochondria",
    # Autophagy
    "MAP1LC3B": "Autophagy",
    "ATG101": "Autophagy",
    # Endosome / Lysosome
    "EEA1": "Endo/Lyso",
    "VPS35": "Endo/Lyso",
    "RAB7A": "Endo/Lyso",
    "LAMP1": "Endo/Lyso",
    "LysoTracker_live-cell_dye": "Endo/Lyso",
    "pHrodo-dextran_Live_Cell_Dye": "Endo/Lyso",
    # Plasma membrane
    "SLC3A2": "Plasma Membrane",
    "ATP1B3": "Plasma Membrane",
    # Cytoskeleton
    "CLTA": "Cytoskeleton",
    "VIM": "Cytoskeleton",
    "MAP4": "Cytoskeleton",
    "FastAct_SPY555_Live_Cell_Dye": "Cytoskeleton",
    # Nucleus
    "H2B_(TOMM20-combo)": "Nucleus",
    "MKI67": "Nucleus",
    "FBL": "Nucleus",
    "NPM3": "Nucleus",
    "SRRM2": "Nucleus",
    # Stress / Proteostasis
    "HSPA1B": "Stress",
    "G3BP1": "Stress",
    "PSMB7": "Stress",
    "CellROX_live-cell_dye": "Stress",
    "FeRhoNox_live-cell_dye": "Stress",
    "CellEvent-Caspase_live-cell_dye": "Stress",
    # Lipid / Metabolic
    "BODIPY_live_cell_dye": "Lipid",
    "PLIN2": "Lipid",
    "Peroxi_SPY650_live_cell_dye": "Lipid",
    # Label-free
    "Phase": "Label-free",
}

# Canonical column order for heatmaps: organelle groups
_ORGANELLE_ORDER = [
    "ER", "ER/Golgi", "Mitochondria", "Autophagy", "Endo/Lyso",
    "Plasma Membrane", "Cytoskeleton", "Nucleus", "Stress", "Lipid",
    "Label-free",
]


def _sort_reporters_by_organelle(reporters):
    """Sort reporter names by organelle group for consistent ordering."""
    def _key(r):
        org = REPORTER_ORGANELLE.get(r, "zzz")
        org_idx = _ORGANELLE_ORDER.index(org) if org in _ORGANELLE_ORDER else 99
        return (org_idx, r)
    return sorted(reporters, key=_key)


# ---------------------------------------------------------------------------
# Marker-level analysis
# ---------------------------------------------------------------------------

def aggregate_by_perturbation_reporter(adata):
    """
    Aggregate cell-level embeddings to (perturbation x reporter) level.

    Parameters
    ----------
    adata : ad.AnnData
        Cell-level AnnData with 'perturbation' and 'reporter' in obs.

    Returns
    -------
    ad.AnnData
        Aggregated AnnData with obs columns: perturbation, reporter,
        organelle, complex, n_cells.
    """
    t0 = time.time()
    print("Aggregating to (perturbation x reporter) level...", end=" ", flush=True)

    pert = adata.obs["perturbation"].astype(str).values
    rep = adata.obs["reporter"].astype(str).values
    combo = np.array([f"{p}||{r}" for p, r in zip(pert, rep)])

    X_agg, unique_combos = _fast_groupby_mean(np.asarray(adata.X), combo)

    perturbations = np.array([c.split("||")[0] for c in unique_combos])
    reporters = np.array([c.split("||")[1] for c in unique_combos])

    combo_to_count = dict(zip(*np.unique(combo, return_counts=True)))
    n_cells = np.array([combo_to_count[c] for c in unique_combos])

    adata_agg = ad.AnnData(X=X_agg.astype(np.float32))
    adata_agg.var_names = adata.var_names
    adata_agg.obs["perturbation"] = perturbations
    adata_agg.obs["reporter"] = reporters
    adata_agg.obs["organelle"] = [
        REPORTER_ORGANELLE.get(r, "unknown") for r in reporters
    ]
    adata_agg.obs["complex"] = [assign_complex(p) for p in perturbations]
    adata_agg.obs["n_cells"] = n_cells
    adata_agg.obs_names = [
        f"{p}__{r}" for p, r in zip(perturbations, reporters)
    ]

    print(
        f"{adata_agg.shape[0]} combos, "
        f"{len(np.unique(perturbations))} perturbations x "
        f"{len(np.unique(reporters))} reporters ({time.time() - t0:.1f}s)"
    )
    return adata_agg


def marker_response_profile(adata_pr, normalize_to_ntc=True):
    """
    Compute cosine distance from NTC for each (perturbation x reporter).

    Parameters
    ----------
    adata_pr : ad.AnnData
        (perturbation x reporter) aggregated AnnData from
        ``aggregate_by_perturbation_reporter``.
    normalize_to_ntc : bool
        Reserved for future use (e.g. within-reporter z-scoring).

    Returns
    -------
    pd.DataFrame
        Pivot table (perturbation x reporter) of cosine distances to NTC.
        Columns sorted by organelle group.
    pd.DataFrame
        Long-form table with perturbation, reporter, organelle, complex,
        cosine_dist_to_ntc, n_cells.
    """
    from sklearn.metrics.pairwise import cosine_distances

    t0 = time.time()
    print("Computing marker response profiles...", end=" ", flush=True)

    X = np.asarray(adata_pr.X)
    obs = adata_pr.obs

    # Build NTC reference per reporter
    ntc_mask = obs["perturbation"] == "NTC"
    ntc_reporters = obs.loc[ntc_mask, "reporter"].values
    ntc_X = X[ntc_mask.values]
    ntc_lookup = {r: ntc_X[i] for i, r in enumerate(ntc_reporters)}

    rows = []
    for i in range(len(obs)):
        pert = obs["perturbation"].iloc[i]
        if pert == "NTC":
            continue
        rep = obs["reporter"].iloc[i]
        if rep not in ntc_lookup:
            continue
        emb = X[i].reshape(1, -1)
        ntc_emb = ntc_lookup[rep].reshape(1, -1)
        dist = cosine_distances(emb, ntc_emb)[0, 0]
        rows.append({
            "perturbation": pert,
            "reporter": rep,
            "organelle": REPORTER_ORGANELLE.get(rep, "unknown"),
            "complex": assign_complex(pert),
            "cosine_dist_to_ntc": dist,
            "n_cells": obs["n_cells"].iloc[i],
        })

    df_long = pd.DataFrame(rows)

    # Pivot to (perturbation x reporter) matrix
    df_pivot = df_long.pivot(
        index="perturbation", columns="reporter", values="cosine_dist_to_ntc"
    )
    # Sort columns by organelle group
    sorted_cols = _sort_reporters_by_organelle(df_pivot.columns.tolist())
    df_pivot = df_pivot[sorted_cols]

    print(f"{df_pivot.shape} ({time.time() - t0:.1f}s)")
    return df_pivot, df_long


def within_complex_marker_consistency(df_pivot):
    """
    Compute correlation of marker response profiles within each complex.

    For each complex, computes the mean pairwise Pearson correlation of the
    40-dim response vectors (cosine distance to NTC per reporter) among
    complex members. High correlation means the model captures that complex
    members affect the same organelles similarly.

    Parameters
    ----------
    df_pivot : pd.DataFrame
        (perturbation x reporter) pivot from ``marker_response_profile``.

    Returns
    -------
    pd.DataFrame
        Per-complex: complex, n_members, mean_pairwise_corr, member_genes.
    """
    t0 = time.time()
    print("Computing within-complex marker consistency...", end=" ", flush=True)

    results = []
    for cpx, members in COMPLEX_GROUPS.items():
        present = [g for g in members if g in df_pivot.index]
        if len(present) < 2:
            continue
        profiles = df_pivot.loc[present].values
        # Pairwise Pearson correlation
        corr_matrix = np.corrcoef(profiles)
        # Upper triangle (exclude diagonal)
        triu_idx = np.triu_indices(len(present), k=1)
        pairwise_corrs = corr_matrix[triu_idx]
        results.append({
            "complex": cpx,
            "n_members": len(present),
            "mean_pairwise_corr": float(np.mean(pairwise_corrs)),
            "min_pairwise_corr": float(np.min(pairwise_corrs)),
            "max_pairwise_corr": float(np.max(pairwise_corrs)),
            "member_genes": ", ".join(present),
        })

    df = pd.DataFrame(results).sort_values("mean_pairwise_corr", ascending=False)
    print(f"done ({time.time() - t0:.1f}s)")
    return df


def plot_marker_response_heatmap(df_pivot, save_path=None):
    """
    Heatmap of cosine distance to NTC: rows = perturbations (clustered by
    complex), columns = reporters (grouped by organelle).

    Parameters
    ----------
    df_pivot : pd.DataFrame
        (perturbation x reporter) pivot from ``marker_response_profile``.
    save_path : str, optional
        Path to save figure.
    """
    # Sort rows by complex, then alphabetically within complex
    row_order = []
    for cpx in sorted(COMPLEX_GROUPS.keys()):
        members = [g for g in COMPLEX_GROUPS[cpx] if g in df_pivot.index]
        row_order.extend(sorted(members))
    # Add any remaining genes not in a complex
    remaining = [g for g in df_pivot.index if g not in row_order]
    row_order.extend(sorted(remaining))
    df_plot = df_pivot.loc[row_order]

    # Color bar for organelle groups on columns
    col_organelles = [REPORTER_ORGANELLE.get(r, "unknown") for r in df_plot.columns]

    fig, ax = plt.subplots(figsize=(20, 10))
    im = ax.imshow(df_plot.values, aspect="auto", cmap="YlOrRd", interpolation="none")

    ax.set_xticks(range(len(df_plot.columns)))
    ax.set_xticklabels(df_plot.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(df_plot.index)))
    ax.set_yticklabels(df_plot.index, fontsize=8)

    # Add organelle group separators on x-axis
    prev_org = col_organelles[0]
    for i, org in enumerate(col_organelles):
        if org != prev_org:
            ax.axvline(i - 0.5, color="black", linewidth=0.8)
            prev_org = org

    # Add complex group separators on y-axis
    row_complexes = [assign_complex(g) for g in row_order]
    prev_cpx = row_complexes[0]
    for i, cpx in enumerate(row_complexes):
        if cpx != prev_cpx:
            ax.axhline(i - 0.5, color="black", linewidth=0.8)
            prev_cpx = cpx

    cbar = plt.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label("Cosine distance to NTC", fontsize=10)
    ax.set_title("Marker Response Profile: Cosine Distance to NTC", fontsize=12)
    ax.set_xlabel("Reporter (grouped by organelle)")
    ax.set_ylabel("Perturbation (grouped by complex)")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_umap_perturbation_reporter(adata_pr, save_dir=None):
    """
    UMAP of (perturbation x reporter) embeddings colored by organelle,
    perturbation, and complex.

    Each point is one (perturbation, reporter) mean embedding. Within each
    perturbation cluster the markers should separate — showing the model
    captures organelle-level biology per perturbation.

    Parameters
    ----------
    adata_pr : ad.AnnData
        (perturbation x reporter) aggregated AnnData from
        ``aggregate_by_perturbation_reporter``. Must have PCA/UMAP computed.
    save_dir : str or Path, optional
        Directory to save figures.
    """
    save_dir = Path(save_dir) if save_dir else None

    for col in ["organelle", "reporter", "perturbation", "complex"]:
        adata_pr.obs[col] = pd.Categorical(adata_pr.obs[col].astype(str))

    fig, axes = plt.subplots(2, 2, figsize=(22, 18))

    sc.pl.umap(
        adata_pr, color="organelle", size=120,
        title="Colored by organelle group", ax=axes[0, 0], show=False,
    )
    sc.pl.umap(
        adata_pr, color="complex", size=120,
        title="Colored by protein complex", ax=axes[0, 1], show=False,
    )
    sc.pl.umap(
        adata_pr, color="perturbation", size=120,
        title="Colored by perturbation", ax=axes[1, 0], show=False,
        legend_loc="on data", legend_fontsize=5, legend_fontoutline=1,
    )
    sc.pl.umap(
        adata_pr, color="reporter", size=120,
        title="Colored by reporter", ax=axes[1, 1], show=False,
    )

    plt.tight_layout()
    if save_dir:
        fig.savefig(
            save_dir / "umap_perturbation_x_reporter.png",
            dpi=150, bbox_inches="tight",
        )
    plt.show()

    # Per-complex detail: one UMAP per complex highlighting its members
    complexes_with_members = {
        cpx: [g for g in members if g in adata_pr.obs["perturbation"].values]
        for cpx, members in COMPLEX_GROUPS.items()
    }
    complexes_with_members = {
        k: v for k, v in complexes_with_members.items() if len(v) >= 2
    }

    n_cpx = len(complexes_with_members)
    if n_cpx == 0:
        return

    ncols = 3
    nrows = (n_cpx + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 6 * nrows))
    axes = np.atleast_2d(axes)

    for idx, (cpx, members) in enumerate(sorted(complexes_with_members.items())):
        ax = axes[idx // ncols, idx % ncols]

        # Subset to complex members + NTC
        mask = adata_pr.obs["perturbation"].isin(members + ["NTC"])
        adata_sub = adata_pr[mask].copy()
        adata_sub.obs["perturbation"] = pd.Categorical(
            adata_sub.obs["perturbation"].astype(str)
        )
        adata_sub.obs["organelle"] = pd.Categorical(
            adata_sub.obs["organelle"].astype(str)
        )

        sc.pl.umap(
            adata_sub, color="organelle", size=200,
            title=f"{cpx} complex ({', '.join(members)})",
            ax=ax, show=False,
        )

    # Hide empty axes
    for idx in range(n_cpx, nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    plt.tight_layout()
    if save_dir:
        fig.savefig(
            save_dir / "umap_per_complex_by_organelle.png",
            dpi=150, bbox_inches="tight",
        )
    plt.show()


def reporter_silhouette_within_perturbation(
    adata, sample_per_gene=5000, min_reporters=5, seed=42
):
    """
    For each perturbation, compute silhouette score using reporter as label.

    Measures whether the embedding preserves organelle-level structure within
    a perturbation — cells imaged with different reporters should have
    distinguishable embeddings if the model learns organelle biology.

    Parameters
    ----------
    adata : ad.AnnData
        Cell-level AnnData with 'perturbation' and 'reporter' in obs.
    sample_per_gene : int
        Max cells to sample per perturbation (silhouette is O(n^2)).
    min_reporters : int
        Skip perturbations with fewer unique reporters.
    seed : int
        Random seed for subsampling.

    Returns
    -------
    pd.DataFrame
        Per-perturbation silhouette scores with complex assignment.
    """
    t0 = time.time()
    print("Computing reporter silhouette within perturbations...", end=" ", flush=True)

    rng = np.random.RandomState(seed)
    results = []

    for pert in adata.obs["perturbation"].unique():
        mask = adata.obs["perturbation"] == pert
        adata_sub = adata[mask]
        reporters = adata_sub.obs["reporter"].values
        unique_reps = np.unique(reporters)

        if len(unique_reps) < min_reporters:
            continue

        n = adata_sub.shape[0]
        if n > sample_per_gene:
            idx = rng.choice(n, sample_per_gene, replace=False)
        else:
            idx = np.arange(n)

        X = np.asarray(adata_sub.X[idx])
        labels = reporters[idx]

        # Filter to reporters with >= 2 cells in this sample
        unique, counts = np.unique(labels, return_counts=True)
        valid = set(unique[counts >= 2])
        if len(valid) < 2:
            continue
        valid_mask = np.array([lab in valid for lab in labels])
        X = X[valid_mask]
        labels = labels[valid_mask]

        score = silhouette_score(X, labels, metric="cosine")
        results.append({
            "perturbation": pert,
            "complex": assign_complex(pert),
            "reporter_silhouette": float(score),
            "n_cells_sampled": len(labels),
            "n_reporters": len(valid),
        })

    df = pd.DataFrame(results)
    if len(df) > 0:
        df = df.sort_values("reporter_silhouette", ascending=False)
    print(f"{len(df)} perturbations ({time.time() - t0:.1f}s)")
    return df


def plot_umap_per_reporter(adata_pr, save_dir=None):
    """
    For each reporter, plot a UMAP of all perturbations colored by complex.

    Each subplot shows one reporter's view: 33 points (one per perturbation),
    colored by protein complex. If complex members cluster together within a
    single reporter, the model captures perturbation biology at that
    organelle readout.

    Parameters
    ----------
    adata_pr : ad.AnnData
        (perturbation x reporter) aggregated AnnData with UMAP computed.
    save_dir : str or Path, optional
        Directory to save figures.
    """
    save_dir = Path(save_dir) if save_dir else None

    reporters = sorted(adata_pr.obs["reporter"].unique())
    n_rep = len(reporters)
    ncols = 5
    nrows = (n_rep + ncols - 1) // ncols

    # Build a consistent color map for complexes across all subplots
    all_complexes = sorted(
        adata_pr.obs["complex"].unique(),
        key=lambda x: (x == "other", x),
    )
    cmap = plt.cm.get_cmap("tab10", len(all_complexes))
    complex_colors = {cpx: cmap(i) for i, cpx in enumerate(all_complexes)}

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.atleast_2d(axes)

    umap_coords = adata_pr.obsm["X_umap"]

    for idx, rep in enumerate(reporters):
        ax = axes[idx // ncols, idx % ncols]
        mask = adata_pr.obs["reporter"] == rep
        obs_sub = adata_pr.obs[mask]
        coords = umap_coords[mask.values]

        organelle = REPORTER_ORGANELLE.get(rep, "unknown")

        for cpx in all_complexes:
            cpx_mask = obs_sub["complex"].values == cpx
            if cpx_mask.sum() == 0:
                continue
            ax.scatter(
                coords[cpx_mask, 0],
                coords[cpx_mask, 1],
                c=[complex_colors[cpx]],
                s=50,
                label=cpx,
                edgecolors="black",
                linewidths=0.3,
                alpha=0.8,
            )
            # Label perturbation names for non-NTC/non-other
            if cpx != "other":
                for j in np.where(cpx_mask)[0]:
                    ax.annotate(
                        obs_sub["perturbation"].iloc[j],
                        (coords[j, 0], coords[j, 1]),
                        fontsize=4,
                        alpha=0.6,
                        ha="left",
                        va="bottom",
                        xytext=(2, 2),
                        textcoords="offset points",
                    )

        ax.set_title(f"{rep}\n({organelle})", fontsize=8, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])

    # Hide empty axes
    for idx in range(n_rep, nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    # Shared legend
    handles = [
        plt.Line2D(
            [0], [0],
            marker="o",
            color="w",
            markerfacecolor=complex_colors[cpx],
            markersize=8,
            markeredgecolor="black",
            markeredgewidth=0.3,
            label=cpx,
        )
        for cpx in all_complexes
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(all_complexes),
        fontsize=8,
        frameon=True,
        bbox_to_anchor=(0.5, -0.02),
    )

    fig.suptitle(
        "Per-reporter UMAP: perturbations colored by complex",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    if save_dir:
        fig.savefig(
            save_dir / "umap_per_reporter.png",
            dpi=150,
            bbox_inches="tight",
        )
    plt.show()


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
