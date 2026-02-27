"""
Batch processing and validation script for embedding and CellProfiler features.

This script orchestrates the processing of neural-network embeddings (DinoV3,
Cell-DINO, etc.) and CellProfiler features for multiple experiments, checking
for CSV existence, processing them into AnnData objects, and validating outputs.

Usage:
    # Process from config list (recommended - reads experiments/channels from configs)
    python batch_process_embeddings.py --config_list configs/dinov3/dino_configs_all.txt

    # Process single config file
    python batch_process_embeddings.py --config_list configs/dinov3/ops0031_dino.yml

    # Embedding features (legacy)
    python batch_process_embeddings.py --feature_type dinov3 --experiments ops0089_20251119 ops0084_20250101
    python batch_process_embeddings.py --feature_type cell_dino --experiments ops0089_20251119 --channels Phase2D GFP

    # CellProfiler features (legacy)
    python batch_process_embeddings.py --feature_type cellprofiler --experiments ops0089_20251119 ops0084_20250101
"""

import argparse
from pathlib import Path
from typing import List, Optional, Dict, Tuple
import sys
import json
from datetime import datetime
import yaml


from ops_model.features.evaluate_embeddings import process_embedding_csv
from ops_model.features.evaluate_cp import process


# Base directory for OPS experiments
BASE_DIR = Path("/hpc/projects/intracellular_dashboard/ops")


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
        feature_type: "cellprofiler" or any embedding type (e.g., "dinov3", "cell_dino")
        channel: Channel name (e.g., "Phase2D", "GFP") - required for embeddings, ignored for cellprofiler
        base_dir: Base directory for experiments (used if config not provided)
        config: Optional config dict with output_dir specified

    Returns:
        Tuple of (exists, path)
    """
    # If config provided, extract output_dir from it
    if config and "output_dir" in config:
        output_dir = Path(config["output_dir"])

        if feature_type == "cellprofiler":
            csv_path = output_dir / "cp_features.csv"
        else:
            if channel is None:
                raise ValueError(
                    f"channel is required for feature_type '{feature_type}'"
                )
            csv_path = output_dir / f"{feature_type}_features_{channel}.csv"
    else:
        raise ValueError("Config with output_dir is required to determine CSV path")

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
        feature_type: "cellprofiler" or any embedding type (e.g., "dinov3", "cell_dino")
        channel: Channel name (required for embeddings, ignored for cellprofiler)
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
        print(f"✗ CSV not found for {experiment}" + (f"/{channel}" if channel else ""))
        return False

    print(f"✓ Found CSV: {csv_path}")

    # Check if already processed
    feature_dir = csv_path.parent
    anndata_dir = feature_dir / "anndata_objects"

    # Determine output filenames: always use reporter name as suffix
    from ops_model.data.feature_metadata import FeatureMetadata

    meta = FeatureMetadata()
    exp_short = experiment.split("_")[0]
    reporter = meta.get_biological_signal(exp_short, channel)
    output_files = [
        anndata_dir / f"features_processed_{reporter}.h5ad",
        anndata_dir / f"guide_bulked_{reporter}.h5ad",
        anndata_dir / f"gene_bulked_{reporter}.h5ad",
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
            if feature_type == "cellprofiler":
                adata = process(str(csv_path), config_path=config_path)
            else:
                adata = process_embedding_csv(str(csv_path), config_path=config_path)
            print(f"✓ Processing complete")
        except Exception as e:
            print(f"✗ Processing failed: {e}")
            import traceback

            traceback.print_exc()
            return False

    return True


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
        feature_type: "cellprofiler" or any embedding type (e.g., "dinov3", "cell_dino")
        channels: List of channel names to process (ignored for cellprofiler)
        force_reprocess: Reprocess even if outputs exist
        validate_only: Only validate, don't process
        continue_on_error: Continue processing other experiments if one fails
        output_report: Path to save JSON report (optional)

    Returns:
        Dictionary mapping (experiment, channel) -> success status
    """
    # Handle channel defaults
    if feature_type == "cellprofiler":
        if channels is not None and len(channels) > 0:
            print("⚠ WARNING: --channels is ignored for cellprofiler feature type")
        channels = [None]  # Single pass, no channel
    else:
        if channels is None:
            channels = ["Phase2D"]

    print(f"\n{'=' * 80}")
    print(f"BATCH PROCESSING {feature_type.upper()} FEATURES")
    print(f"{'=' * 80}")
    print(f"Experiments: {len(experiments)}")
    if feature_type != "cellprofiler":
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
            "channels": channels if feature_type != "cellprofiler" else None,
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
        - feature_type: "cellprofiler" or any embedding type (e.g., "dinov3", "cell_dino")
        - experiments: List of experiment names (e.g., ["ops0031_20250424"])
        - channels: List of channels (e.g., ["Phase2D", "GFP"])
        - processing_options: Dict with force_reprocess, validate_only, etc.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Extract feature type
    feature_type = config.get("model_type")
    if not feature_type:
        raise ValueError(f"Config missing model_type: {config_path}")

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
        description="Batch process and validate embedding or CellProfiler features for multiple experiments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process from config list (recommended - reads experiments/channels from configs)
  python batch_process_embeddings.py --config_list configs/dinov3/dino_configs_all.txt

  # Process single config file
  python batch_process_embeddings.py --config_list configs/dinov3/ops0031_dino.yml

  # Embedding features: process two experiments with Phase2D channel
  python batch_process_embeddings.py --feature_type dinov3 --experiments ops0089_20251119 ops0084_20250101
  python batch_process_embeddings.py --feature_type cell_dino --experiments ops0089_20251119 --channels Phase2D GFP

  # CellProfiler: Process experiments (channels ignored)
  python batch_process_embeddings.py --feature_type cellprofiler --experiments ops0089_20251119 ops0084_20250101

  # Validate only (don't reprocess)
  python batch_process_embeddings.py --feature_type dinov3 --experiments ops0089_20251119 --validate_only

  # Force reprocessing
  python batch_process_embeddings.py --feature_type cell_dino --experiments ops0089_20251119 --force

  # Load experiments from file
  python batch_process_embeddings.py --feature_type cellprofiler --experiments_file experiments.txt

  # Save validation report
  python batch_process_embeddings.py --feature_type dinov3 --experiments ops0089_20251119 --output_report report.json
        """,
    )

    # Feature type selection
    parser.add_argument(
        "--feature_type",
        type=str,
        required=False,
        help="Type of features to process: 'cellprofiler' or any embedding type (e.g. 'dinov3', 'cell_dino'). Not needed with --config_list.",
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
        help="Channels to process (default: Phase2D for embedding types, ignored for cellprofiler)",
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

    # Set default channels for embedding types if not specified
    channels = args.channels
    if args.feature_type != "cellprofiler" and channels is None:
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
