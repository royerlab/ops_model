"""Guide-level embedding evaluator."""

from __future__ import annotations

import yaml
import numpy as np
import pandas as pd
import anndata as ad
from sklearn.metrics import silhouette_score

from ops_model.post_process.map.map import (
    phenotypic_activity_assesment,
    phenotypic_distinctivness,
)
from ops_model.post_process.anndata_processing.anndata_validator import AnndataValidator
from ops_model.eval.metrics import mean_cosine_sim_within_groups

POS_CONTROLS_YAML_PATH = (
    "/hpc/projects/icd.ops/configs/gene_clusters/chad_positive_controls_v4.yml"
)


def _load_pos_controls() -> dict:
    with open(POS_CONTROLS_YAML_PATH) as f:
        return yaml.safe_load(f)


def evaluate_guide_level(adata: ad.AnnData) -> tuple[dict, pd.DataFrame]:
    """Evaluate guide-level embeddings and return metrics and the activity map.

    Parameters
    ----------
    adata : AnnData
        Guide-level AnnData passing the guide-level schema. Required .obs columns:
        ``perturbation``, ``sgRNA``, ``n_cells``. Required .uns: ``aggregation_method``.

    Returns
    -------
    metrics : dict
        Flat dict of scalar metrics. Keys:
        ``pct_perturbations_active``, ``mean_map_active``,
        ``pct_pos_controls_active``, ``mean_map_pos_controls``,
        ``pct_perturbations_distinct``, ``mean_map_distinct``,
        ``mean_cosine_sim_within_gene``, ``silhouette_within_gene``.
    activity_map : DataFrame
        Per-perturbation mAP results from ``phenotypic_activity_assesment``.
        Pass this to ``evaluate_gene_level`` to filter by active genes.
    """
    # 1. Validate
    AnndataValidator().validate(adata, level="guide", strict=False)

    # 2. Phenotypic activity
    activity_map, active_ratio = phenotypic_activity_assesment(adata, plot_results=False)
    pct_perturbations_active = float(active_ratio)
    mean_map_active = float(
        activity_map[activity_map["below_corrected_p"]]["mean_average_precision"].mean()
    )

    # 3. Positive controls
    pos_controls = _load_pos_controls()
    pos_control_genes = {
        gene for cluster in pos_controls.values() for gene in cluster["genes"]
    }
    pos_control_map = activity_map[activity_map["perturbation"].isin(pos_control_genes)]
    pct_pos_controls_active = float(pos_control_map["below_corrected_p"].mean())
    mean_map_pos_controls = float(pos_control_map["mean_average_precision"].mean())

    # 4. Phenotypic distinctiveness
    distinctiveness_map, distinctive_ratio = phenotypic_distinctivness(
        adata, activity_map, plot_results=False
    )
    pct_perturbations_distinct = float(distinctive_ratio)
    mean_map_distinct = float(
        distinctiveness_map[distinctiveness_map["below_corrected_p"]][
            "mean_average_precision"
        ].mean()
    )

    # 5. Within-perturbation cosine similarity
    groups = [
        list(np.where(adata.obs["perturbation"] == p)[0])
        for p in adata.obs["perturbation"].unique()
    ]
    mean_cosine_sim_within_gene = mean_cosine_sim_within_groups(adata, groups)

    # 6. Within-perturbation silhouette score
    X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    labels = adata.obs["perturbation"].values
    silhouette_within_gene = (
        float(silhouette_score(X, labels))
        if len(np.unique(labels)) >= 2
        else float("nan")
    )

    metrics = {
        "pct_perturbations_active": pct_perturbations_active,
        "mean_map_active": mean_map_active,
        "pct_pos_controls_active": pct_pos_controls_active,
        "mean_map_pos_controls": mean_map_pos_controls,
        "pct_perturbations_distinct": pct_perturbations_distinct,
        "mean_map_distinct": mean_map_distinct,
        "mean_cosine_sim_within_gene": mean_cosine_sim_within_gene,
        "silhouette_within_gene": silhouette_within_gene,
    }
    return metrics, activity_map
