import logging
from contextlib import contextmanager
from typing import Tuple, List, Dict
import anndata as ad

from .config_handler import CombinationConfig
from ops_model.features.anndata_utils import concatenate_experiments_comprehensive

# Initialize logger
logger = logging.getLogger(__name__)


@contextmanager
def temp_log_level(level, module_names):
    """Temporarily set the logging level for a list of modules."""
    original_levels = {}
    for name in module_names:
        module_logger = logging.getLogger(name)
        original_levels[name] = module_logger.level
        module_logger.setLevel(level)
    try:
        yield
    finally:
        for name, original_level in original_levels.items():
            logging.getLogger(name).setLevel(original_level)


class ComprehensiveCombiner:
    """
    Orchestrates the comprehensive combination of experiments by wrapping
    the core logic in `anndata_utils.concatenate_experiments_comprehensive`.
    """

    def __init__(self, config: CombinationConfig):
        """
        Initializes the combiner with a validated configuration.
        """
        self.config = config

    def combine(self) -> Tuple[ad.AnnData, ad.AnnData]:
        """
        Execute the full combination pipeline by calling the utility function.
        """
        logger.info("Starting comprehensive combination process...")

        # Prepare arguments for the utility function from the config
        # Note: Currently, the same embedding config is used for both guide and gene levels
        # in concatenate_experiments_comprehensive. Using gene_level config since that's
        # typically more important for downstream analysis.
        embedding_config = self.config.embeddings.get("gene_level")
        if embedding_config is None:
            # Fallback to guide_level if gene_level missing
            embedding_config = self.config.embeddings.get("guide_level")
        if embedding_config is None:
            # Final fallback if both missing (shouldn't happen)
            from .config_handler import EmbeddingConfig

            embedding_config = EmbeddingConfig()

        # Convert experiments_channels from Dict[str, List[str]] to List[Tuple[str, str]]
        experiments_channels_list = [
            (exp, ch)
            for exp, channels in (self.config.experiments_channels or {}).items()
            for ch in channels
        ]

        if not experiments_channels_list:
            raise ValueError("No experiment/channel pairs were found to combine.")

        # Temporarily silence verbose logs from underlying libraries
        with temp_log_level(logging.WARNING, ["scanpy", "umap"]):
            adata_guide, adata_gene = concatenate_experiments_comprehensive(
                experiments_channels=experiments_channels_list,
                feature_type=self.config.feature_type,
                base_dir=self.config.base_dir,
                feature_dir=self.config.feature_dir,
                recompute_embeddings=embedding_config.compute_embeddings,
                n_pca_components=embedding_config.n_pca_components,
                n_umap_neighbors=embedding_config.n_neighbors,
                compute_pca=embedding_config.pca,
                compute_umap=embedding_config.umap,
                compute_phate=embedding_config.phate,
                normalize_on_pooling=self.config.normalization.get(
                    "normalize_on_pooling", True
                ),
                normalize_on_controls=self.config.normalization.get(
                    "normalize_on_controls", False
                ),
                subsample_controls=self.config.control_subsampling.get(
                    "enabled", False
                ),
                control_gene=self.config.control_subsampling.get("control_gene", "NTC"),
                control_group_size=self.config.control_subsampling.get("group_size", 4),
                random_seed=self.config.control_subsampling.get("random_seed"),
                fit_on_aggregated_controls=self.config.fitted_embeddings.get(
                    "enabled", False
                ),
                use_pca_for_umap=self.config.fitted_embeddings.get(
                    "use_pca_for_umap", True
                ),
                leiden_resolutions=(
                    self.config.leiden_clustering.get("resolutions")
                    if self.config.leiden_clustering.get("enabled", False)
                    else None
                ),
            )

        logger.info("Comprehensive combination process complete.")
        return adata_guide, adata_gene
