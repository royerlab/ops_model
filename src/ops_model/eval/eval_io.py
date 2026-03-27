"""Save evaluation outputs (CSVs and plots) to disk."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from ops_utils.analysis.map_scores import plot_map_scatter


def save_guide_eval(
    activity_map: pd.DataFrame,
    distinctiveness_map: pd.DataFrame,
    metrics: dict,
    output_dir: str | Path,
) -> None:
    """Save guide-level evaluation outputs to output_dir.

    Writes:
    - ``phenotypic_activity.csv``
    - ``phenotypic_distinctiveness.csv``
    - ``map_activity_distinctiveness.png``

    Parameters
    ----------
    activity_map : DataFrame
        Per-perturbation mAP results from ``phenotypic_activity_assesment``.
    distinctiveness_map : DataFrame
        Per-perturbation mAP results from ``phenotypic_distinctivness``.
    metrics : dict
        Scalar metrics returned by ``evaluate_guide_level`` (used for plot titles).
    output_dir : str or Path
        Directory to write outputs into (created if it does not exist).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    activity_map.to_csv(output_dir / "phenotypic_activity.csv", index=False)
    distinctiveness_map.to_csv(output_dir / "phenotypic_distinctiveness.csv", index=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    plot_map_scatter(ax1, activity_map, "Activity", metrics["pct_perturbations_active"])
    plot_map_scatter(ax2, distinctiveness_map, "Distinctiveness", metrics["pct_perturbations_distinct"])
    fig.tight_layout()
    fig.savefig(output_dir / "map_activity_distinctiveness.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_gene_eval(
    consistency_corum_map: pd.DataFrame,
    consistency_manual_map: pd.DataFrame,
    metrics: dict,
    output_dir: str | Path,
) -> None:
    """Save gene-level evaluation outputs to output_dir.

    Writes:
    - ``phenotypic_consistency_corum.csv``
    - ``phenotypic_consistency_manual.csv``
    - ``map_consistency.png``

    Parameters
    ----------
    consistency_corum_map : DataFrame
        Per-complex mAP results from ``phenotypic_consistency_corum``.
    consistency_manual_map : DataFrame
        Per-complex mAP results from ``phenotypic_consistency_manual_annotation``.
    metrics : dict
        Scalar metrics returned by ``evaluate_gene_level`` (used for plot titles).
    output_dir : str or Path
        Directory to write outputs into (created if it does not exist).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    consistency_corum_map.to_csv(output_dir / "phenotypic_consistency_corum.csv", index=False)
    consistency_manual_map.to_csv(output_dir / "phenotypic_consistency_manual.csv", index=False)

    corum_entity_col = "complex_id" if "complex_id" in consistency_corum_map.columns else consistency_corum_map.columns[0]
    chad_entity_col = "complex_num" if "complex_num" in consistency_manual_map.columns else consistency_manual_map.columns[0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    plot_map_scatter(ax1, consistency_corum_map, "Consistency (CORUM)", metrics["pct_complexes_significant_corum"])
    plot_map_scatter(ax2, consistency_manual_map, "Consistency (CHAD)", metrics["pct_complexes_significant_manual"])
    fig.tight_layout()
    fig.savefig(output_dir / "map_consistency.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
