"""
Phenotypic Activity Assessment using copairs mAP (mean Average Precision).

Backward-compatible re-export wrapper. All implementations now live in
``ops_utils.analysis`` for cross-pipeline reuse.
"""

# Re-export all shared mAP functions from ops_utils (single source of truth)
from ops_utils.analysis.map_scores import (  # noqa: F401
    _compute_single_complex_map,
    adata_to_copairs_df,
    compute_auc_score,
    compute_threshold_sweep_auc,
    phenotypic_activity_assesment,
    phenotypic_distinctivness,
    phenotypic_consistency_corum,
    phenotypic_consistency_manual_annotation,
    map_main,
)

# Re-export UMAP visualization from ops_utils
from ops_utils.analysis.map_umap import metric_umap  # noqa: F401
