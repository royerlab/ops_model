"""Gene-level embedding evaluator."""

from __future__ import annotations

import warnings
import yaml
import numpy as np
import pandas as pd
import anndata as ad
from sklearn.metrics import silhouette_score

from ops_model.post_process.map.map import (
    phenotypic_consistency_manual_annotation,
    phenotypic_consistency_corum,
)
from ops_model.post_process.anndata_processing.anndata_validator import AnndataValidator
from ops_model.eval.metrics import mean_cosine_sim_within_groups

MANUAL_ANNOTATION_YAML_PATH = (
    "/hpc/projects/icd.ops/configs/gene_clusters/chad_positive_controls_v4.yml"
)


def _load_gene_clusters() -> dict:
    with open(MANUAL_ANNOTATION_YAML_PATH) as f:
        return yaml.safe_load(f)


def evaluate_gene_level(
    adata: ad.AnnData, activity_map: pd.DataFrame | None = None
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Evaluate gene-level embeddings and return a flat dict of scalar metrics.

    Parameters
    ----------
    adata : AnnData
        Gene-level AnnData passing the gene-level schema. Required .obs columns:
        ``perturbation``, ``n_cells``, ``guides``, ``n_experiments``.
        Required .uns: ``aggregation_method``.
    activity_map : DataFrame, optional
        Per-perturbation mAP results from a prior guide-level evaluation
        (the second return value of ``evaluate_guide_level``). Used to filter
        consistency metrics to active genes only. If not provided, all genes
        are assumed active and a warning is printed.

    Returns
    -------
    metrics : dict
        Flat dict of scalar metrics. Keys:
        ``pct_complexes_significant_manual``, ``mean_map_complexes_manual``,
        ``pct_complexes_significant_corum``, ``mean_map_complexes_corum``,
        ``mean_cosine_sim_within_complex``, ``silhouette_within_complex``.
    consistency_corum_map : DataFrame
        Per-complex mAP results from ``phenotypic_consistency_corum``.
    consistency_manual_map : DataFrame
        Per-complex mAP results from ``phenotypic_consistency_manual_annotation``.
    """
    # 1. Validate
    # debug: 
    print("Validating gene-level AnnData...")
    print(adata.obs.keys())

    AnndataValidator().validate(adata, level="gene", strict=False)

    # 2. Activity map — use provided or assume all genes active
    if activity_map is None:
        warnings.warn(
            "No activity_map provided to evaluate_gene_level. "
            "All genes will be treated as active. "
            "For accurate results, pass the activity_map returned by evaluate_guide_level.",
            UserWarning,
            stacklevel=2,
        )
        activity_map = pd.DataFrame(
            {
                "perturbation": adata.obs["perturbation"].unique(),
                "below_corrected_p": True,
                "mean_average_precision": float("nan"),
                "corrected_p_value": float("nan"),
            }
        )

    # 3. Phenotypic consistency -- manual annotation
    consistency_manual_map, consistency_manual_ratio = (
        phenotypic_consistency_manual_annotation(adata, activity_map, plot_results=False)
    )
    pct_complexes_significant_manual = float(consistency_manual_ratio)
    mean_map_complexes_manual = float(
        consistency_manual_map["mean_average_precision"].mean()
    )

    # 4. Phenotypic consistency -- CORUM
    consistency_corum_map, consistency_corum_ratio = phenotypic_consistency_corum(
        adata, activity_map, plot_results=False
    )
    pct_complexes_significant_corum = float(consistency_corum_ratio)
    mean_map_complexes_corum = float(
        consistency_corum_map["mean_average_precision"].mean()
    )

    # 5. Within-complex cosine similarity
    gene_clusters = _load_gene_clusters()
    perturbation_to_idx = {p: i for i, p in enumerate(adata.obs["perturbation"])}
    groups = [
        [perturbation_to_idx[g] for g in c["genes"] if g in perturbation_to_idx]
        for c in gene_clusters.values()
    ]
    mean_cosine_sim_within_complex = mean_cosine_sim_within_groups(adata, groups)

    # 6. Within-complex silhouette score (first complex wins for genes in multiple)
    perturbation_to_complex: dict[str, str] = {}
    for cluster_name, cluster_data in gene_clusters.items():
        for gene in cluster_data["genes"]:
            if gene not in perturbation_to_complex:
                perturbation_to_complex[gene] = cluster_name

    labels_all = [
        perturbation_to_complex.get(p) for p in adata.obs["perturbation"]
    ]
    valid_mask = np.array([lbl is not None for lbl in labels_all])
    X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    X_labeled = X[valid_mask]
    labels_valid = [lbl for lbl, v in zip(labels_all, valid_mask) if v]
    silhouette_within_complex = (
        float(silhouette_score(X_labeled, labels_valid))
        if len(set(labels_valid)) >= 2
        else float("nan")
    )

    metrics = {
        "pct_complexes_significant_manual": pct_complexes_significant_manual,
        "mean_map_complexes_manual": mean_map_complexes_manual,
        "pct_complexes_significant_corum": pct_complexes_significant_corum,
        "mean_map_complexes_corum": mean_map_complexes_corum,
        "mean_cosine_sim_within_complex": mean_cosine_sim_within_complex,
        "silhouette_within_complex": silhouette_within_complex,
    }
    return metrics, consistency_corum_map, consistency_manual_map
