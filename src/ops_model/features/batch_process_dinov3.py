"""
Batch processing and validation script for DinoV3 and CellProfiler features.

This script orchestrates the processing of both DinoV3 embeddings and CellProfiler
features for multiple experiments, checking for CSV existence, processing them into
AnnData objects, and validating the output against specifications.

Usage:
    # Process from config list (recommended - reads experiments/channels from configs)
    python batch_process_dinov3.py --config_list configs/dinov3/dino_configs_all.txt

    # Process single config file
    python batch_process_dinov3.py --config_list configs/dinov3/ops0031_dino.yml

    # DinoV3 features (legacy)
    python batch_process_dinov3.py --feature_type dinov3 --experiments ops0089_20251119 ops0084_20250101
    python batch_process_dinov3.py --feature_type dinov3 --experiments ops0089_20251119 --channels Phase2D GFP

    # CellProfiler features (legacy)
    python batch_process_dinov3.py --feature_type cellprofiler --experiments ops0089_20251119 ops0084_20250101
"""

import argparse
from pathlib import Path
from typing import List, Optional, Dict, Tuple
import sys
import json
from datetime import datetime
import yaml

import numpy as np
import pandas as pd
import anndata as ad

from ops_model.features.evaluate_dinov3 import process_dinov3
from ops_model.features.evaluate_cp import process


# Base directory for OPS experiments
BASE_DIR = Path("/hpc/projects/intracellular_dashboard/ops")


class ExperimentValidator:
    """Validates DinoV3 and CellProfiler AnnData objects against specifications."""

    def __init__(self, feature_type: str = "dinov3", verbose: bool = True):
        self.feature_type = feature_type
        self.verbose = verbose
        self.errors = []
        self.warnings = []

    def log_error(self, message: str):
        """Log an error message."""
        self.errors.append(message)
        if self.verbose:
            print(f"  ✗ ERROR: {message}")

    def log_warning(self, message: str):
        """Log a warning message."""
        self.warnings.append(message)
        if self.verbose:
            print(f"  ⚠ WARNING: {message}")

    def log_success(self, message: str):
        """Log a success message."""
        if self.verbose:
            print(f"  ✓ {message}")

    def validate_cell_level(
        self, adata: ad.AnnData, expected_n_features: Optional[int] = None
    ) -> bool:
        """
        Validate cell-level AnnData object.

        Required checks:
        - Shape and dimensions
        - Metadata columns in .obs
        - PCA/UMAP presence (feature-type specific)
        - Data types
        - No NaN values

        Args:
            adata: AnnData object to validate
            expected_n_features: Expected number of features (if None, skip check)
                                 Default: 1024 for DinoV3, None for CellProfiler

        Returns:
            True if all checks pass, False otherwise
        """
        is_valid = True

        # Check 1: Shape
        if expected_n_features is not None:
            if adata.shape[1] != expected_n_features:
                self.log_error(
                    f"Expected {expected_n_features} features, got {adata.shape[1]}"
                )
                is_valid = False
            else:
                self.log_success(f"Correct feature dimension: {expected_n_features}")
        else:
            self.log_success(f"Feature dimension: {adata.shape[1]}")

        # Check 2: Required metadata columns
        required_obs = ["label_str", "label_int", "sgRNA", "well"]
        missing_cols = [col for col in required_obs if col not in adata.obs.columns]
        if missing_cols:
            self.log_error(f"Missing required .obs columns: {missing_cols}")
            is_valid = False
        else:
            self.log_success(f"All required metadata columns present")

        # Check 3: PCA/UMAP presence (feature-type specific)
        if "X_pca" in adata.obsm.keys():
            self.log_error(
                "Cell-level should NOT have X_pca (only at aggregation level)"
            )
            is_valid = False
        if "X_umap" in adata.obsm.keys():
            self.log_error(
                "Cell-level should NOT have X_umap (only at aggregation level)"
            )
            is_valid = False
        if "X_pca" not in adata.obsm.keys() and "X_umap" not in adata.obsm.keys():
            self.log_success("Correctly has no PCA/UMAP (only at aggregation level)")

        # Check 4: Data types
        if adata.X.dtype not in [np.float32, np.float64]:
            self.log_warning(f"Unexpected dtype for .X: {adata.X.dtype}")

        # Check 5: No NaN values
        if np.isnan(adata.X).any():
            self.log_error("Found NaN values in .X")
            is_valid = False
        else:
            self.log_success("No NaN values in data")

        # Check 6: Reasonable value ranges
        x_min = adata.X.min()
        x_max = adata.X.max()
        if x_min < -100 or x_max > 100:
            self.log_warning(f"Unusual value range: [{x_min:.2f}, {x_max:.2f}]")

        # Check 7: Label distribution
        n_genes = adata.obs["label_str"].nunique()
        n_guides = adata.obs["sgRNA"].nunique()
        self.log_success(f"{n_genes} unique genes, {n_guides} unique guides")

        # Check for NTC
        if "NTC" not in adata.obs["label_str"].values:
            self.log_warning("No NTC control cells found")

        return is_valid

    def validate_aggregated(
        self, adata: ad.AnnData, level: str, expected_n_features: Optional[int] = None
    ) -> bool:
        """
        Validate guide-level or gene-level aggregated AnnData.

        Required checks:
        - Shape and dimensions
        - PCA and UMAP present
        - Metadata columns
        - Aggregation correctness

        Args:
            adata: AnnData object to validate
            level: "guide" or "gene"
            expected_n_features: Expected number of features (if None, skip check)

        Returns:
            True if all checks pass, False otherwise
        """
        is_valid = True

        # Check 1: Shape
        if expected_n_features is not None:
            if adata.shape[1] != expected_n_features:
                self.log_error(
                    f"Expected {expected_n_features} features, got {adata.shape[1]}"
                )
                is_valid = False
        else:
            self.log_success(f"Feature dimension: {adata.shape[1]}")

        # Check 2: Required metadata
        if level == "guide":
            if "sgRNA" not in adata.obs.columns:
                self.log_error("Missing sgRNA column in guide-level data")
                is_valid = False
        elif level == "gene":
            if "label_str" not in adata.obs.columns:
                self.log_error("Missing label_str column in gene-level data")
                is_valid = False

        # Check 3: PCA present
        if "X_pca" not in adata.obsm.keys():
            self.log_error(f"{level}-level should have X_pca")
            is_valid = False
        else:
            pca_shape = adata.obsm["X_pca"].shape
            max_components = min(128, adata.shape[0] - 1)
            if pca_shape[1] > max_components:
                self.log_error(
                    f"Too many PCA components: {pca_shape[1]} (max: {max_components})"
                )
                is_valid = False
            else:
                self.log_success(f"PCA present: {pca_shape}")

            # Check explained variance
            if "pca" in adata.uns.keys() and "variance" in adata.uns["pca"].keys():
                self.log_success("PCA explained variance stored")
            else:
                self.log_warning("PCA explained variance not found in .uns")

        # Check 4: UMAP present
        if "X_umap" not in adata.obsm.keys():
            self.log_error(f"{level}-level should have X_umap")
            is_valid = False
        else:
            umap_shape = adata.obsm["X_umap"].shape
            if umap_shape[1] != 2:
                self.log_error(f"UMAP should be 2D, got {umap_shape[1]}D")
                is_valid = False
            else:
                self.log_success(f"UMAP present: {umap_shape}")

            # Check UMAP coordinates are finite
            if not np.isfinite(adata.obsm["X_umap"]).all():
                self.log_error("UMAP contains non-finite values")
                is_valid = False

        # Check 5: PHATE present
        if "X_phate" not in adata.obsm.keys():
            self.log_error(f"{level}-level should have X_phate")
            is_valid = False
        else:
            phate_shape = adata.obsm["X_phate"].shape
            if phate_shape[1] != 2:
                self.log_error(f"PHATE should be 2D, got {phate_shape[1]}D")
                is_valid = False
            else:
                self.log_success(f"PHATE present: {phate_shape}")

            # Check PHATE coordinates are finite
            if not np.isfinite(adata.obsm["X_phate"]).all():
                self.log_error("PHATE contains non-finite values")
                is_valid = False

        # Check 6: No duplicate entries
        if level == "guide":
            n_unique = adata.obs["sgRNA"].nunique()
            if n_unique != adata.shape[0]:
                self.log_error(
                    f"Duplicate guides: {adata.shape[0]} rows, {n_unique} unique"
                )
                is_valid = False
        elif level == "gene":
            n_unique = adata.obs["label_str"].nunique()
            if n_unique != adata.shape[0]:
                self.log_error(
                    f"Duplicate genes: {adata.shape[0]} rows, {n_unique} unique"
                )
                is_valid = False

        return is_valid

    def validate_experiment(
        self,
        experiment: str,
        channel: Optional[str],
        feature_dir: Path,
        config: Optional[dict] = None,
    ) -> Dict[str, bool]:
        """
        Validate all three AnnData files for an experiment.

        Args:
            experiment: Experiment name
            channel: Channel name (None for CellProfiler without reporter names)
            feature_dir: Path to feature directory
            config: Configuration dictionary (required for CellProfiler with reporter names)

        Returns:
            Dictionary with validation results for each file
        """
        print(f"\n{'=' * 60}")
        if channel:
            print(f"Validating: {experiment} / {channel}")
        else:
            print(f"Validating: {experiment}")
        print(f"{'=' * 60}\n")

        results = {}
        anndata_dir = feature_dir / "anndata_objects"

        # Determine filename suffix (channel vs reporter) and expected features
        filename_suffix = None
        expected_n_features = None

        if self.feature_type == "cellprofiler":
            # Check if using reporter names
            use_reporter_names = False
            if config and "processing" in config:
                use_reporter_names = config["processing"].get(
                    "use_reporter_names", False
                )

            if use_reporter_names and channel:
                # Import FeatureMetadata to get reporter name
                from ops_model.data.feature_metadata import FeatureMetadata

                meta = FeatureMetadata()

                # Extract short experiment name (e.g., "ops0089_20251119" -> "ops0089")
                exp_short = experiment.split("_")[0]

                # Get reporter name
                reporter = meta.get_biological_signal(exp_short, channel)
                filename_suffix = reporter
                print(f"  Using reporter name: {reporter} (channel: {channel})")
            else:
                # Legacy CellProfiler: no suffix
                filename_suffix = None
                print(f"  Using legacy naming (no suffix)")

            # CellProfiler has variable feature count
            expected_n_features = None
        else:
            # DinoV3: always use channel name
            filename_suffix = channel
            expected_n_features = 1024

        # Construct filenames based on suffix
        if filename_suffix:
            cell_filename = f"features_processed_{filename_suffix}.h5ad"
            guide_filename = f"guide_bulked_umap_{filename_suffix}.h5ad"
            gene_filename = f"gene_bulked_umap_{filename_suffix}.h5ad"
        else:
            # Legacy CellProfiler naming (no suffix)
            cell_filename = "features_processed.h5ad"
            guide_filename = "guide_bulked_umap.h5ad"
            gene_filename = "gene_bulked_umap.h5ad"

        # Validate cell-level
        print(f"Cell-level ({cell_filename}):")
        cell_path = anndata_dir / cell_filename
        if not cell_path.exists():
            self.log_error(f"File not found: {cell_path}")
            results["cell_level"] = False
        else:
            try:
                adata = ad.read_h5ad(cell_path)
                results["cell_level"] = self.validate_cell_level(
                    adata, expected_n_features
                )
            except Exception as e:
                self.log_error(f"Failed to load: {e}")
                results["cell_level"] = False

        # Validate guide-level
        print(f"\nGuide-level ({guide_filename}):")
        guide_path = anndata_dir / guide_filename
        if not guide_path.exists():
            self.log_error(f"File not found: {guide_path}")
            results["guide_level"] = False
        else:
            try:
                adata = ad.read_h5ad(guide_path)
                results["guide_level"] = self.validate_aggregated(
                    adata, "guide", expected_n_features
                )
            except Exception as e:
                self.log_error(f"Failed to load: {e}")
                results["guide_level"] = False

        # Validate gene-level
        print(f"\nGene-level ({gene_filename}):")
        gene_path = anndata_dir / gene_filename
        if not gene_path.exists():
            self.log_error(f"File not found: {gene_path}")
            results["gene_level"] = False
        else:
            try:
                adata = ad.read_h5ad(gene_path)
                results["gene_level"] = self.validate_aggregated(
                    adata, "gene", expected_n_features
                )
            except Exception as e:
                self.log_error(f"Failed to load: {e}")
                results["gene_level"] = False

        # Summary
        print(f"\n{'─' * 60}")
        all_valid = all(results.values())
        if all_valid:
            print("✓ All validation checks passed!")
        else:
            print("✗ Some validation checks failed")
            failed = [k for k, v in results.items() if not v]
            print(f"  Failed: {', '.join(failed)}")
        print(f"{'─' * 60}")

        return results


def check_csv_exists(
    experiment: str,
    feature_type: str,
    channel: Optional[str] = None,
    base_dir: Path = BASE_DIR,
    config: Optional[dict] = None,
) -> Tuple[bool, Optional[Path]]:
    """
    Check if feature CSV exists for given experiment.

    Args:
        experiment: Experiment name (e.g., "ops0089_20251119")
        feature_type: "dinov3" or "cellprofiler"
        channel: Channel name (e.g., "Phase2D", "GFP") - required for dinov3, ignored for cellprofiler
        base_dir: Base directory for experiments (used if config not provided)
        config: Optional config dict with output_dir specified

    Returns:
        Tuple of (exists, path)
    """
    # If config provided, extract output_dir from it
    if config and "output_dir" in config:
        output_dir = Path(config["output_dir"])

        if feature_type == "dinov3":
            if channel is None:
                raise ValueError("channel is required for dinov3 feature type")
            csv_path = output_dir / f"dinov3_features_{channel}.csv"
        elif feature_type == "cellprofiler":
            csv_path = output_dir / "cp_features.csv"
        else:
            raise ValueError(
                f"Invalid feature_type: {feature_type}. Must be 'dinov3' or 'cellprofiler'"
            )
    else:
        # Fallback to hardcoded paths (backwards compatibility)
        if feature_type == "dinov3":
            if channel is None:
                raise ValueError("channel is required for dinov3 feature type")
            csv_path = (
                base_dir
                / experiment
                / "3-assembly"
                / "dino_features"
                / f"dinov3_features_{channel}.csv"
            )
        elif feature_type == "cellprofiler":
            csv_path = (
                base_dir
                / experiment
                / "3-assembly"
                / "cell-profiler"
                / "cp_features.csv"
            )
        else:
            raise ValueError(
                f"Invalid feature_type: {feature_type}. Must be 'dinov3' or 'cellprofiler'"
            )

    return csv_path.exists(), csv_path if csv_path.exists() else None


def process_experiment(
    experiment: str,
    feature_type: str,
    channel: Optional[str] = None,
    force_reprocess: bool = False,
    validate_only: bool = False,
    config_path: Optional[str] = None,
) -> bool:
    """
    Process a single experiment's features.

    Args:
        experiment: Experiment name
        feature_type: "dinov3" or "cellprofiler"
        channel: Channel name (required for dinov3, ignored for cellprofiler)
        force_reprocess: If True, reprocess even if outputs exist
        validate_only: If True, only validate without processing
        config_path: Optional path to configuration YAML file

    Returns:
        True if successful, False otherwise
    """
    print(f"\n{'=' * 80}")
    if channel:
        print(f"Processing: {experiment} / {channel} ({feature_type})")
    else:
        print(f"Processing: {experiment} ({feature_type})")
    print(f"{'=' * 80}\n")

    # Load config if provided (need this first to get output_dir)
    config = None
    if config_path:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        print(f"✓ Loaded config: {config_path}")

    # Check CSV exists
    csv_exists, csv_path = check_csv_exists(
        experiment, feature_type, channel, config=config
    )
    if not csv_exists:
        if feature_type == "dinov3":
            print(f"✗ CSV not found: dinov3_features_{channel}.csv")
        else:
            print(f"✗ CSV not found: cp_features.csv")
        return False

    print(f"✓ Found CSV: {csv_path}")

    # Check if already processed
    feature_dir = csv_path.parent
    anndata_dir = feature_dir / "anndata_objects"

    # Determine output filenames based on feature type and config
    if feature_type == "dinov3":
        output_files = [
            anndata_dir / f"features_processed_{channel}.h5ad",
            anndata_dir / f"guide_bulked_umap_{channel}.h5ad",
            anndata_dir / f"gene_bulked_umap_{channel}.h5ad",
        ]
    else:  # cellprofiler
        # Check if using reporter names
        use_reporter_names = False
        if config and "processing" in config:
            use_reporter_names = config["processing"].get("use_reporter_names", False)

        if use_reporter_names and channel:
            from ops_model.data.feature_metadata import FeatureMetadata

            meta = FeatureMetadata()
            exp_short = experiment.split("_")[0]
            reporter = meta.get_biological_signal(exp_short, channel)
            output_files = [
                anndata_dir / f"features_processed_{reporter}.h5ad",
                anndata_dir / f"guide_bulked_umap_{reporter}.h5ad",
                anndata_dir / f"gene_bulked_umap_{reporter}.h5ad",
            ]
        else:
            # Legacy naming
            output_files = [
                anndata_dir / "features_processed.h5ad",
                anndata_dir / "guide_bulked_umap.h5ad",
                anndata_dir / "gene_bulked_umap.h5ad",
            ]

    already_processed = all(f.exists() for f in output_files)

    if already_processed and not force_reprocess:
        print(f"✓ Already processed (outputs exist)")
        if not validate_only:
            print("  Use --force to reprocess")
    elif not validate_only:
        # Process
        print(f"\nProcessing {feature_type} features...")
        try:
            if feature_type == "dinov3":
                adata = process_dinov3(str(csv_path), config_path=config_path)
            else:  # cellprofiler
                # Extract processing config and merge with cell-profiler flag

                adata = process(str(csv_path), config_path=config_path)
            print(f"✓ Processing complete")
        except Exception as e:
            print(f"✗ Processing failed: {e}")
            import traceback

            traceback.print_exc()
            return False

    # Validate outputs
    print(f"\nValidating outputs...")
    validator = ExperimentValidator(feature_type=feature_type, verbose=True)
    results = validator.validate_experiment(experiment, channel, feature_dir, config)

    return all(results.values())


def batch_process(
    experiments: List[str],
    feature_type: str,
    channels: Optional[List[str]] = None,
    config_path: Optional[str] = None,
    force_reprocess: bool = False,
    validate_only: bool = False,
    continue_on_error: bool = True,
    output_report: Optional[str] = None,
) -> Dict[str, Dict[str, bool]]:
    """
    Batch process multiple experiments.

    Args:
        experiments: List of experiment names
        feature_type: "dinov3" or "cellprofiler"
        channels: List of channel names to process (only for dinov3)
        force_reprocess: Reprocess even if outputs exist
        validate_only: Only validate, don't process
        continue_on_error: Continue processing other experiments if one fails
        output_report: Path to save JSON report (optional)

    Returns:
        Dictionary mapping (experiment, channel) -> success status
    """
    # Handle channel defaults
    if feature_type == "dinov3":
        if channels is None:
            channels = ["Phase2D"]
    else:  # cellprofiler
        if channels is not None and len(channels) > 0:
            print("⚠ WARNING: --channels is ignored for cellprofiler feature type")
        channels = [None]  # Single pass, no channel

    print(f"\n{'=' * 80}")
    print(f"BATCH PROCESSING {feature_type.upper()} FEATURES")
    print(f"{'=' * 80}")
    print(f"Experiments: {len(experiments)}")
    if feature_type == "dinov3":
        print(f"Channels: {channels}")
    print(f"Force reprocess: {force_reprocess}")
    print(f"Validate only: {validate_only}")
    print(f"{'=' * 80}\n")

    results = {}
    for experiment in experiments:
        for channel in channels:
            if channel:
                key = f"{experiment}_{channel}"
            else:
                key = experiment
            try:
                success = process_experiment(
                    experiment,
                    feature_type,
                    channel=channel,
                    config_path=config_path,
                    force_reprocess=force_reprocess,
                    validate_only=validate_only,
                )
                results[key] = success
            except Exception as e:
                if channel:
                    print(
                        f"\n✗ Unexpected error processing {experiment}/{channel}: {e}"
                    )
                else:
                    print(f"\n✗ Unexpected error processing {experiment}: {e}")
                results[key] = False
                if not continue_on_error:
                    raise

    # Summary
    print(f"\n{'=' * 80}")
    print("BATCH PROCESSING SUMMARY")
    print(f"{'=' * 80}\n")

    n_total = len(results)
    n_success = sum(1 for v in results.values() if v)
    n_failed = n_total - n_success

    print(f"Total: {n_total}")
    print(f"✓ Success: {n_success}")
    print(f"✗ Failed: {n_failed}")

    if n_failed > 0:
        print(f"\nFailed:")
        for key, success in results.items():
            if not success:
                print(f"  - {key}")

    # Save report
    if output_report:
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "feature_type": feature_type,
            "experiments": experiments,
            "channels": channels if feature_type == "dinov3" else None,
            "force_reprocess": force_reprocess,
            "validate_only": validate_only,
            "results": results,
            "summary": {"total": n_total, "success": n_success, "failed": n_failed},
        }

        output_path = Path(output_report)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report_data, f, indent=2)
        print(f"\n✓ Report saved to: {output_path}")

    print(f"{'=' * 80}\n")

    return results


def parse_config_file(config_path: str) -> Tuple[str, List[str], List[str], Dict]:
    """
    Parse a config file to extract experiment names, channels, and processing options.

    Args:
        config_path: Path to YAML config file

    Returns:
        Tuple of (feature_type, experiments, channels, processing_options)
        - feature_type: "dinov3" or "cellprofiler"
        - experiments: List of experiment names (e.g., ["ops0031_20250424"])
        - channels: List of channels (e.g., ["Phase2D", "GFP"])
        - processing_options: Dict with force_reprocess, validate_only, etc.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Extract feature type
    feature_type = config.get("model_type")
    if feature_type not in ["dinov3", "cellprofiler"]:
        raise ValueError(
            f"Invalid model_type in {config_path}: {feature_type}. Must be 'dinov3' or 'cellprofiler'"
        )

    # Extract experiments (keys from data_manager.experiments)
    if "data_manager" not in config or "experiments" not in config["data_manager"]:
        raise ValueError(f"Config missing data_manager.experiments: {config_path}")

    experiments = list(config["data_manager"]["experiments"].keys())

    # Extract channels
    channels = config["data_manager"].get("out_channels", [])

    # Extract batch processing options (with defaults)
    batch_config = config.get("batch_processing", {})
    processing_options = {
        "force_reprocess": batch_config.get("force_reprocess", False),
        "validate_only": batch_config.get("validate_only", False),
        "stop_on_error": batch_config.get("stop_on_error", False),
        "output_report": batch_config.get("output_report", None),
    }

    return feature_type, experiments, channels, processing_options


def process_from_config_list(config_list_path: str) -> Dict[str, bool]:
    """
    Process experiments from a list of config files.

    Each config file is processed separately with its own batch_process() call.
    All configs must have the same feature_type.

    Args:
        config_list_path: Path to text file with config paths (one per line)

    Returns:
        Dictionary mapping (experiment, channel) -> success status
    """
    config_list_path = Path(config_list_path)

    if not config_list_path.exists():
        raise FileNotFoundError(f"Config list file not found: {config_list_path}")

    # Read config paths (skip comments and blank lines)
    with open(config_list_path, "r") as f:
        config_paths = [
            line.strip()
            for line in f
            if line.strip() and not line.strip().startswith("#")
        ]

    if not config_paths:
        raise ValueError(f"No config files found in {config_list_path}")

    print(f"\n{'=' * 80}")
    print(f"PROCESSING FROM CONFIG LIST: {config_list_path}")
    print(f"{'=' * 80}")
    print(f"Found {len(config_paths)} config files")
    print(f"{'=' * 80}\n")

    # Parse all configs and validate they have the same feature type
    parsed_configs = []
    for config_path in config_paths:
        # Handle relative paths (relative to config_list directory or repo root)
        config_path_obj = Path(config_path)
        if not config_path_obj.is_absolute():
            # Try relative to config list location first
            relative_to_list = config_list_path.parent / config_path
            if relative_to_list.exists():
                config_path_obj = relative_to_list
            else:
                # Try relative to current directory
                if not config_path_obj.exists():
                    raise FileNotFoundError(f"Config file not found: {config_path}")

        try:
            feature_type, experiments, channels, options = parse_config_file(
                str(config_path_obj)
            )
            parsed_configs.append(
                (str(config_path_obj), feature_type, experiments, channels, options)
            )
        except Exception as e:
            print(f"✗ Error parsing {config_path}: {e}")
            raise

    # Validate all configs have same feature type
    feature_types = set(config[1] for config in parsed_configs)
    if len(feature_types) > 1:
        raise ValueError(
            f"All configs must have the same feature_type. Found: {feature_types}\n"
            f"Please separate configs by feature type into different list files."
        )

    feature_type = list(feature_types)[0]
    print(f"Feature type: {feature_type}")
    print(f"Processing {len(parsed_configs)} config(s)...\n")

    # Process each config separately
    all_results = {}
    for i, (config_path, feat_type, experiments, channels, options) in enumerate(
        parsed_configs, 1
    ):
        print(f"\n{'─' * 80}")
        print(f"CONFIG {i}/{len(parsed_configs)}: {Path(config_path).name}")
        print(f"{'─' * 80}")
        print(f"Experiments: {experiments}")
        print(f"Channels: {channels if channels else 'N/A'}")
        print(f"Options: {options}")
        print(f"{'─' * 80}\n")

        # Call batch_process for this config
        results = batch_process(
            experiments=experiments,
            feature_type=feat_type,
            channels=channels if channels else None,
            config_path=config_path,
            force_reprocess=options["force_reprocess"],
            validate_only=options["validate_only"],
            continue_on_error=not options["stop_on_error"],
            output_report=options["output_report"],
        )

        # Merge results
        all_results.update(results)

    # Overall summary
    print(f"\n{'=' * 80}")
    print("OVERALL SUMMARY")
    print(f"{'=' * 80}")
    n_total = len(all_results)
    n_success = sum(1 for v in all_results.values() if v)
    n_failed = n_total - n_success

    print(f"Total: {n_total}")
    print(f"✓ Success: {n_success}")
    print(f"✗ Failed: {n_failed}")

    if n_failed > 0:
        print(f"\nFailed:")
        for key, success in all_results.items():
            if not success:
                print(f"  - {key}")

    print(f"{'=' * 80}\n")

    return all_results


def _build_arg_parser():
    """Build argument parser."""
    parser = argparse.ArgumentParser(
        description="Batch process and validate DinoV3 or CellProfiler features for multiple experiments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process from config list (recommended - reads experiments/channels from configs)
  python batch_process_dinov3.py --config_list configs/dinov3/dino_configs_all.txt

  # Process single config file
  python batch_process_dinov3.py --config_list configs/dinov3/ops0031_dino.yml

  # DinoV3: Process two experiments with Phase2D channel
  python batch_process_dinov3.py --feature_type dinov3 --experiments ops0089_20251119 ops0084_20250101

  # DinoV3: Process multiple channels
  python batch_process_dinov3.py --feature_type dinov3 --experiments ops0089_20251119 --channels Phase2D GFP

  # CellProfiler: Process experiments (channels ignored)
  python batch_process_dinov3.py --feature_type cellprofiler --experiments ops0089_20251119 ops0084_20250101

  # Validate only (don't reprocess)
  python batch_process_dinov3.py --feature_type dinov3 --experiments ops0089_20251119 --validate_only

  # Force reprocessing
  python batch_process_dinov3.py --feature_type dinov3 --experiments ops0089_20251119 --force

  # Load experiments from file
  python batch_process_dinov3.py --feature_type cellprofiler --experiments_file experiments.txt

  # Save validation report
  python batch_process_dinov3.py --feature_type dinov3 --experiments ops0089_20251119 --output_report report.json
        """,
    )

    # Feature type selection
    parser.add_argument(
        "--feature_type",
        type=str,
        required=False,
        choices=["dinov3", "cellprofiler"],
        help="Type of features to process: 'dinov3' or 'cellprofiler' (not needed with --config_list)",
    )

    # Experiment selection
    exp_group = parser.add_mutually_exclusive_group(required=True)
    exp_group.add_argument(
        "--experiments", nargs="+", help="List of experiment names to process"
    )
    exp_group.add_argument(
        "--experiments_file",
        type=str,
        help="File containing experiment names (one per line)",
    )
    exp_group.add_argument(
        "--config_list",
        type=str,
        help="File containing config paths (one per line). Extracts experiments/channels from configs.",
    )

    # Channel selection
    parser.add_argument(
        "--channels",
        nargs="+",
        default=None,
        help="Channels to process (default: Phase2D for dinov3, ignored for cellprofiler)",
    )

    # Processing options
    parser.add_argument(
        "--force", action="store_true", help="Force reprocessing even if outputs exist"
    )
    parser.add_argument(
        "--validate_only",
        action="store_true",
        help="Only validate existing outputs, don't process",
    )
    parser.add_argument(
        "--stop_on_error",
        action="store_true",
        help="Stop processing if any experiment fails (default: continue)",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        help="Path to configuration YAML file for processing options",
    )

    # Output options
    parser.add_argument(
        "--output_report",
        type=str,
        help="Path to save JSON report of processing results",
    )

    return parser


def main():
    """Main entry point."""
    parser = _build_arg_parser()
    args = parser.parse_args()

    # Handle config list (new pathway)
    if args.config_list:
        try:
            results = process_from_config_list(args.config_list)
        except Exception as e:
            print(f"\n✗ Error processing config list: {e}")
            import traceback

            traceback.print_exc()
            sys.exit(1)

        # Exit code
        all_success = all(results.values())
        sys.exit(0 if all_success else 1)

    # Original pathway: load experiments from args
    # Validate that feature_type is provided when not using config_list
    if not args.feature_type:
        print("Error: --feature_type is required when not using --config_list")
        parser.print_help()
        sys.exit(1)

    if args.experiments:
        experiments = args.experiments
    else:
        experiments_file = Path(args.experiments_file)
        if not experiments_file.exists():
            print(f"Error: Experiments file not found: {experiments_file}")
            sys.exit(1)
        with open(experiments_file, "r") as f:
            experiments = [line.strip() for line in f if line.strip()]

    if not experiments:
        print("Error: No experiments specified")
        sys.exit(1)

    # Set default channels for dinov3 if not specified
    channels = args.channels
    if args.feature_type == "dinov3" and channels is None:
        channels = ["Phase2D"]

    # Process
    results = batch_process(
        experiments=experiments,
        feature_type=args.feature_type,
        channels=channels,
        config_path=args.config_path,
        force_reprocess=args.force,
        validate_only=args.validate_only,
        continue_on_error=not args.stop_on_error,
        output_report=args.output_report,
    )

    # Exit code
    all_success = all(results.values())
    sys.exit(0 if all_success else 1)


if __name__ == "__main__":
    main()
