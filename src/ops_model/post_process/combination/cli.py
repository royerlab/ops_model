"""
CLI and programmatic entry point for the AnnData combination pipeline.

This module provides the main orchestration logic for the combination process,
tying together configuration, file validation, and the core combination logic.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import anndata as ad

from .config_handler import CombinationConfig, load_config
from .file_validator import FileValidator
from .combiners import ComprehensiveCombiner
from ..anndata_processing.anndata_validator import AnndataValidator, IssueLevel

# Initialize logger
logger = logging.getLogger(__name__)


def setup_logging(level: int = logging.INFO):
    """
    Configures the root logger for the application.

    Args:
        level: The logging level to set (e.g., logging.INFO, logging.DEBUG).
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )


def load_anndata_objects(file_paths: List[Path]) -> Dict[str, ad.AnnData]:
    """
    Loads a list of AnnData files from disk into a dictionary.

    Args:
        file_paths: A list of Path objects pointing to .h5ad files.

    Returns:
        A dictionary mapping stringified file paths to loaded AnnData objects.
    """
    adata_objects = {}
    logger.info(f"Loading {len(file_paths)} AnnData object(s)...")
    for path in file_paths:
        try:
            adata_objects[str(path)] = ad.read_h5ad(path)
            logger.debug(f"Successfully loaded {path}")
        except Exception as e:
            logger.error(f"Failed to load AnnData file: {path}. Error: {e}")
            # The "warn and continue" policy applies to finding files.
            # If a found file fails to load, it's a critical error.
            raise
    return adata_objects


def validate_and_save(adata: ad.AnnData, path: Path, level: str):
    """
    Validate an AnnData object against a schema and save it to a .h5ad file.

    This function enforces hard validation constraints. If any ERROR-level issues
    are found, it logs a detailed report and raises a ValueError, preventing
    the invalid object from being saved.

    Args:
        adata: The AnnData object to validate and save.
        path: The Path object where the file will be saved.
        level: The schema level to validate against (e.g., "multi_experiment").

    Raises:
        ValueError: If the AnnData object fails validation.
    """
    logger.info(f"Validating final AnnData object against '{level}' schema...")
    validator = AnndataValidator()
    issues = validator.validate(adata, level=level, strict=False)
    errors = [issue for issue in issues.errors if issue.level == IssueLevel.ERROR]

    if errors:
        error_summary = (
            f"{len(errors)} validation error(s) found in {level}-level AnnData"
        )
        logger.error(f"✗ Validation FAILED: {error_summary}")
        for issue in errors[:10]:  # Log up to 10 specific errors
            logger.error(
                f"  - {issue.component}{'['+issue.field+']' if issue.field else ''}: {issue.message}"
            )
        raise ValueError(error_summary)

    logger.info(f"✓ Validation passed. Saving file to {path}...")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        adata.write_h5ad(path)
        logger.info(f"✓ File saved successfully.")
    except Exception as e:
        logger.error(f"Failed to save final AnnData object: {e}")
        raise


def run_combination(config: CombinationConfig):
    """
    Programmatic entry point for running the full combination pipeline.
    """
    logger.info("--- Starting Combination Pipeline ---")

    # 1. Validate and collect input files
    file_validator = FileValidator(config)
    valid_files = file_validator.validate_and_collect_files()

    # The `comprehensive` method handles its own file loading.
    if config.concatenation_method == "comprehensive":
        combiner = ComprehensiveCombiner(config)
        adata_guide, adata_gene = combiner.combine()
    elif config.concatenation_method == "vertical":
        # Vertical concatenation - pool same biological signal across experiments
        from pathlib import Path
        from ops_model.features.anndata_utils import _process_vertical_group

        if not config.experiments or not config.channel:
            logger.error(
                "Vertical concatenation requires 'experiments' and 'channel' in config."
            )
            return

        # Convert experiments to (experiment, channel) pairs
        exp_channel_pairs = [(exp, config.channel) for exp in config.experiments]

        # Determine feature directory
        if config.feature_dir:
            feature_dir = config.feature_dir
        elif config.feature_type == "cellprofiler":
            feature_dir = "cell-profiler"
        else:
            feature_dir = f"{config.feature_type}_features"

        logger.info(
            f"Running vertical concatenation for {len(config.experiments)} experiments"
        )
        logger.info(f"Channel: {config.channel}")
        logger.info(f"Per-experiment mode: {config.aggregation_per_experiment}")
        logger.info(f"Per-well mode: {config.aggregation_per_well}")

        # Process to guide level
        adata_guide = _process_vertical_group(
            exp_channel_pairs=exp_channel_pairs,
            feature_type=config.feature_type,
            base_dir=Path(config.base_dir),
            feature_dir=feature_dir,
            target_level="guide",
            join="inner",
            verbose=True,
            subsample_controls=False,
            control_gene=config.control_subsampling.get("control_gene", "NTC"),
            control_group_size=config.control_subsampling.get("group_size", 4),
            random_seed=config.control_subsampling.get("random_seed"),
            normalize_on_pooling=config.normalization.get("normalize_on_pooling", True),
            normalize_on_controls=config.normalization.get(
                "normalize_on_controls", False
            ),
            per_experiment=config.aggregation_per_experiment,
            keep_shared_only=not config.aggregation_per_well,  # shared-only doesn't apply in per-well mode
            per_well=config.aggregation_per_well,
        )

        # Aggregate to gene level.
        # When per_well is active, group by [perturbation, well, experiment] so that
        # each (gene, well, experiment) triple becomes one observation.
        from ops_model.features.anndata_utils import aggregate_to_level

        if config.aggregation_per_well:
            gene_batch_cols = ["well", "experiment"]
        elif config.aggregation_per_experiment:
            gene_batch_cols = ["experiment"]
        else:
            gene_batch_cols = None

        logger.info("Aggregating to gene level...")
        adata_gene = aggregate_to_level(
            adata_guide,
            level="gene",
            method="mean",
            preserve_batch_info=False,
            batch_cols=gene_batch_cols,
            subsample_controls=config.control_subsampling.get("enabled", False),
            control_gene=config.control_subsampling.get("control_gene", "NTC"),
            control_group_size=config.control_subsampling.get("group_size", 4),
            random_seed=config.control_subsampling.get("random_seed"),
        )

        logger.info(f"Guide-level: {adata_guide.shape}")
        logger.info(f"Gene-level: {adata_gene.shape}")
    else:
        # For other methods, load all objects first
        if not valid_files:
            logger.error("No valid input files found. Aborting pipeline.")
            return

        anndata_objects = load_anndata_objects(valid_files)
        logger.error(
            f"Method '{config.concatenation_method}' not yet implemented in this CLI."
        )
        return

    # 4. Final validation and saving
    if config.output_path:
        output_path = Path(config.output_path)
        guide_path = output_path.parent / f"{output_path.stem}_guide.h5ad"
        gene_path = output_path.parent / f"{output_path.stem}_gene.h5ad"

        validate_and_save(adata_guide, guide_path, level="multi_experiment")
        validate_and_save(adata_gene, gene_path, level="multi_experiment")
    else:
        logger.warning("Output path is not set. Skipping save.")

    logger.info("--- Combination Pipeline Finished ---")


def main():
    """CLI entry point for the AnnData combination script."""
    parser = argparse.ArgumentParser(
        description="Combine AnnData objects from multiple experiments."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        help="Optional: Path to save the final output file. Overrides the path in the config file.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug-level logging for more detailed output.",
    )
    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(log_level)

    # Load configuration
    config = load_config(args.config, args.output_path)

    # Run the full combination pipeline
    run_combination(config)


if __name__ == "__main__":
    main()
