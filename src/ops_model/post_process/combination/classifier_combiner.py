"""ClassifierCombiner: train an MLP on multi-reporter cell views and return
gene-level penultimate-layer embeddings.

See classifier_aggregator_plan.md (in ops_model/eval/) for full design details.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import anndata as ad
import torch

from .classifier_aggregator import ClassifierAggregator
from .config_handler import CombinationConfig

logger = logging.getLogger(__name__)


class ClassifierCombiner:
    """Config-driven combiner that trains a classifier and returns penultimate-layer
    embeddings as gene-level AnnData.

    Replaces mean-pooling aggregation: an MLP is trained to predict perturbation
    identity from multi-reporter concatenated views and the penultimate-layer
    representations become the gene-level embeddings.

    Only produces gene-level output (no guide-level AnnData).
    """

    def __init__(self, config: CombinationConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def combine(self) -> tuple[ad.AnnData, ad.AnnData]:
        """Run full pipeline: path resolution → view pre-computation → MLP training → transform.

        Returns
        -------
        guide_adata : ad.AnnData or None
            Guide-level AnnData ``(n_guides × last_hidden_dim)``, or ``None``
            if no per-construct identifier column (see ``uns["guide_col"]``,
            default ``"sgRNA"``) is found in the h5ad files.
        gene_adata : ad.AnnData
            Gene-level AnnData ``(n_perturbations × last_hidden_dim)``.
        """
        from ops_utils.data.feature_discovery import (
            build_signal_groups,
            find_cell_h5ad_path,
            get_channel_maps_path,
        )
        from ops_utils.data.feature_metadata import FeatureMetadata

        agg_cfg = self.config.classifier_aggregation

        # 1. Flatten experiments_channels → (exp, ch) pairs
        pairs: List[Tuple[str, str]] = [
            (exp, ch)
            for exp, channels in (self.config.experiments_channels or {}).items()
            for ch in channels
        ]
        if not pairs:
            raise ValueError(
                "No experiment/channel pairs found in config.experiments_channels."
            )

        # 2. Group by biological signal (reporter)
        maps_path = get_channel_maps_path()
        fm = FeatureMetadata(metadata_path=maps_path)
        signal_groups: Dict[str, List[Tuple[str, str]]] = build_signal_groups(pairs, fm)

        if not signal_groups:
            raise ValueError(
                "No experiment/channel pairs could be resolved to a biological signal."
            )

        reporters = list(signal_groups.keys())
        logger.info(f"Resolved {len(reporters)} reporters: {reporters}")

        # 3. Resolve h5ad paths using find_cell_h5ad_path (handles 3-assembly/
        #    subdirectory, reporter→channel fallback, and missing files)
        storage_roots = [Path(self.config.base_dir)]
        feature_dir = self.config.feature_dir

        h5ad_paths: List[Path] = []
        reporter_for_path: List[str] = []

        for signal, signal_pairs in signal_groups.items():
            for exp, ch in signal_pairs:
                path = find_cell_h5ad_path(
                    exp, ch, storage_roots, feature_dir, maps_path
                )
                if path is None:
                    logger.warning(
                        f"h5ad not found for {exp}/{ch} (reporter={signal!r}), skipping."
                    )
                    continue
                h5ad_paths.append(path)
                reporter_for_path.append(signal)
                logger.info(f"  {signal:<30} {exp}/{ch} → {path.name}")

        if not h5ad_paths:
            raise ValueError(
                "No h5ad files could be resolved. Check base_dir and feature_dir."
            )

        # 4. Discover perturbations by scanning obs only (no .X loaded)
        perturbations = self._discover_perturbations(h5ad_paths)
        logger.info(f"Discovered {len(perturbations)} perturbations across all files.")
        self._log_cells_per_perturbation(h5ad_paths, reporter_for_path, reporters)

        # 5. Instantiate ClassifierAggregator from config params
        aggregator = ClassifierAggregator(
            hidden_dims=tuple(agg_cfg.get("hidden_dims", [512, 512, 512])),
            dropout=agg_cfg.get("dropout", 0.4),
            cosine_classifier=agg_cfg.get("cosine_classifier", True),
            batch_size=agg_cfg.get("batch_size", 256),
            num_epochs=agg_cfg.get("num_epochs", 50),
            learning_rate=agg_cfg.get("learning_rate", 1e-3),
            weight_decay=agg_cfg.get("weight_decay", 1e-4),
            val_fraction=agg_cfg.get("val_fraction", 0.2),
            seed=agg_cfg.get("seed", 42),
        )

        device = torch.device(agg_cfg.get("device", "cpu"))

        # 6. Fit
        aggregator.fit(
            h5ad_paths=h5ad_paths,
            reporter_for_path=reporter_for_path,
            reporters=reporters,
            perturbations=perturbations,
            n_cells=agg_cfg["n_cells_per_view"],
            n_views=agg_cfg["n_views"],
            device=device,
            wandb_project=agg_cfg.get("wandb_project"),
            wandb_run_name=agg_cfg.get("wandb_run_name"),
        )

        # 7. Save weights
        weights_path = agg_cfg.get("weights_path")
        if weights_path:
            aggregator.save_weights(weights_path)

        # 8. Extract and return guide- and gene-level embeddings
        n_passes = agg_cfg.get("inference_n_passes", 100)
        return aggregator.transform(device, n_passes=n_passes)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _log_cells_per_perturbation(
        self,
        h5ad_paths: List[Path],
        reporter_for_path: List[str],
        reporters: List[str],
    ) -> None:
        """Log mean cells per perturbation for each reporter (obs-only scan)."""
        from collections import defaultdict

        reporter_paths: Dict[str, List[Path]] = defaultdict(list)
        for path, reporter in zip(h5ad_paths, reporter_for_path):
            reporter_paths[reporter].append(path)

        logger.info("Mean cells per perturbation per reporter:")
        for reporter in reporters:
            paths = reporter_paths.get(reporter, [])
            if not paths:
                continue

            pert_counts: Dict[str, int] = defaultdict(int)
            for path in paths:
                adata = ad.read_h5ad(path, backed="r")
                try:
                    col = (
                        "perturbation"
                        if "perturbation" in adata.obs.columns
                        else "label_str"
                    )
                    for pert, count in adata.obs[col].value_counts().items():
                        pert_counts[pert] += count
                finally:
                    adata.file.close()

            if pert_counts:
                mean_cells = sum(pert_counts.values()) / len(pert_counts)
                logger.info(
                    f"  {reporter:<45} {mean_cells:>8.0f} cells/perturbation"
                    f"  ({len(paths)} file(s), {len(pert_counts)} perturbations)"
                )

    def _discover_perturbations(self, h5ad_paths: List[Path]) -> List[str]:
        """Collect the union of perturbation labels across all h5ad files.

        Uses AnnData backed mode so only ``.obs`` is read — ``.X`` is never
        loaded into memory.

        Falls back to ``label_str`` if ``perturbation`` is absent (backwards
        compatibility with older h5ad files).
        """
        all_perts: set[str] = set()
        for path in h5ad_paths:
            adata = ad.read_h5ad(path, backed="r")
            try:
                col = (
                    "perturbation"
                    if "perturbation" in adata.obs.columns
                    else "label_str"
                )
                all_perts.update(adata.obs[col].unique())
            finally:
                adata.file.close()
        return sorted(all_perts)
