"""
Batch processing and validation script for embedding and CellProfiler features.

This script orchestrates the processing of neural-network embeddings (DinoV3,
Cell-DINO, etc.) and CellProfiler features for multiple experiments, checking
for CSV existence, processing them into AnnData objects, and validating outputs.

Usage:
    # Process from config list — submits all configs as a single SLURM batch
    python batch_process_embeddings.py --config_list configs/dinov3/dino_configs_all.txt

    # Process a single config file
    python batch_process_embeddings.py --config configs/dinov3/ops0031_dino.yml

    # Config list, sequential (no SLURM)
    python batch_process_embeddings.py --config_list configs/dinov3/dino_configs_all.txt --no-slurm

    # Config list, force reprocess across all configs
    python batch_process_embeddings.py --config_list configs/dinov3/dino_configs_all.txt --force
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
    config: Optional[dict] = None,
) -> Tuple[bool, Optional[Path]]:
    """
    Check if feature CSV exists for given experiment.

    Args:
        experiment: Experiment name (e.g., "ops0089_20251119")
        feature_type: "cellprofiler" or any embedding type (e.g., "dinov3", "cell_dino")
        channel: Channel name (e.g., "Phase2D", "GFP") - required for embeddings, ignored for cellprofiler
        config: Config dict with output_dir specified

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
    from ops_utils.data.feature_metadata import FeatureMetadata

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


def batch_process_slurm(
    config_list_path: Optional[str] = None,
    config_path: Optional[str] = None,
    force_reprocess: bool = False,
    slurm_config: Optional[Dict] = None,
) -> Dict[str, bool]:
    """
    Submit per-channel batch processing as parallel SLURM jobs.

    Each (experiment, channel) pair becomes a separate SLURM job submitted
    via submitit using ops_utils.hpc.slurm_batch_utils.submit_parallel_jobs.

    Args:
        config_list_path: Path to text file with config paths (one per line).
            Extracts experiments/channels/feature_type from each config.
        config_path: Path to a single YAML config file.
        force_reprocess: Reprocess even if outputs exist
        slurm_config: SLURM parameters dict (partition, mem, cpus, time, etc.).
            Caller-supplied values override values read from config files.
    """
    from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs

    # Resolve the list of config paths to process
    if config_list_path:
        config_list = Path(config_list_path)
        with open(config_list) as f:
            config_paths = [
                l.strip() for l in f if l.strip() and not l.strip().startswith("#")
            ]
        # Resolve relative paths against the list file's directory
        resolved = []
        for cp in config_paths:
            cp_obj = Path(cp)
            if not cp_obj.is_absolute():
                relative = config_list.parent / cp
                cp_obj = relative if relative.exists() else cp_obj
            resolved.append(str(cp_obj))
        experiment_label = f"{len(resolved)}_configs"
    elif config_path:
        resolved = [config_path]
        experiment_label = Path(config_path).stem
    else:
        raise ValueError("Either config_list_path or config_path must be provided")

    # --- Parse configs and build flat job list ---
    parsed = []
    for cp in resolved:
        feat_type, exps, chans, opts = parse_config_file(cp)
        parsed.append((cp, feat_type, exps, chans, opts))

    jobs_to_submit = []
    for cp, feat_type, exps, chans, opts in parsed:
        force = force_reprocess or opts["force_reprocess"]
        channel_list = [None] if feat_type == "cellprofiler" else (chans or ["Phase2D"])
        for experiment in exps:
            for channel in channel_list:
                name = f"{experiment}/{channel}" if channel else experiment
                jobs_to_submit.append(
                    {
                        "name": name,
                        "func": process_experiment,
                        "kwargs": {
                            "experiment": experiment,
                            "feature_type": feat_type,
                            "channel": channel,
                            "force_reprocess": force,
                            "validate_only": False,
                            "config_path": cp,
                        },
                        "metadata": {
                            "experiment": experiment,
                            "channel": channel or "all",
                        },
                    }
                )

    if not jobs_to_submit:
        print("No jobs to submit!")
        return {}

    # SLURM defaults (CPU-only, moderate memory for anndata)
    defaults = {
        "timeout_min": 30,
        "slurm_partition": "cpu",
        "cpus_per_task": 4,
        "mem": "72G",
    }
    if slurm_config:
        defaults.update(slurm_config)

    # Convert mem string (e.g., "48G") to mem_gb int for submitit
    mem_str = defaults.pop("mem", "48G")
    if isinstance(mem_str, str):
        defaults["mem_gb"] = int(mem_str.rstrip("GgBb"))
    else:
        defaults["mem_gb"] = int(mem_str)
    defaults["slurm_mem"] = mem_str

    result = submit_parallel_jobs(
        jobs_to_submit=jobs_to_submit,
        experiment=experiment_label,
        slurm_params=defaults,
        log_dir=f"slurm_embeddings/{experiment_label}",
        manifest_prefix="embeddings_combine",
        wait_for_completion=True,
        verbose=True,
    )

    if result.get("all_completed"):
        print("\nAll jobs processed successfully!")
    elif result.get("failed"):
        print(f"\n{len(result['failed'])} job(s) failed. Check logs.")

    return result


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


def _build_arg_parser():
    """Build argument parser."""
    parser = argparse.ArgumentParser(
        description="Batch process and validate embedding or CellProfiler features for multiple experiments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process from config list (reads experiments/channels from each config)
  python batch_process_embeddings.py --config_list configs/dinov3/dino_configs_all.txt

  # Process a single config file
  python batch_process_embeddings.py --config configs/dinov3/ops0031_dino.yml

  # Sequential (no SLURM)
  python batch_process_embeddings.py --config_list configs/dinov3/dino_configs_all.txt --no-slurm

  # Force reprocessing
  python batch_process_embeddings.py --config_list configs/dinov3/dino_configs_all.txt --force
        """,
    )

    # Input: mutually exclusive — config list file or single config
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--config_list",
        type=str,
        help="Text file with one config path per line. Extracts experiments/channels/feature_type from each config.",
    )
    input_group.add_argument(
        "--config",
        type=str,
        help="Path to a single YAML config file.",
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
        "--no-slurm",
        action="store_true",
        dest="no_slurm",
        help="Run sequentially instead of submitting SLURM jobs (default: use SLURM)",
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

    if not args.no_slurm:
        result = batch_process_slurm(
            config_list_path=args.config_list,
            config_path=args.config,
            force_reprocess=args.force,
        )
        sys.exit(0 if result.get("all_completed") else 1)
    else:
        # --no-slurm: resolve config paths, parse each, call batch_process()
        if args.config_list:
            cl = Path(args.config_list)
            with open(cl) as f:
                config_paths = [
                    l.strip() for l in f if l.strip() and not l.strip().startswith("#")
                ]
            resolved = [
                str(Path(cp) if Path(cp).is_absolute() else (cl.parent / cp))
                for cp in config_paths
            ]
        else:
            resolved = [args.config]

        all_results = {}
        for cp in resolved:
            feat_type, exps, chans, opts = parse_config_file(cp)
            results = batch_process(
                experiments=exps,
                feature_type=feat_type,
                channels=chans or None,
                config_path=cp,
                force_reprocess=args.force or opts["force_reprocess"],
                validate_only=args.validate_only,
                continue_on_error=not opts["stop_on_error"],
                output_report=opts["output_report"],
            )
            all_results.update(results)
        sys.exit(0 if all(all_results.values()) else 1)


if __name__ == "__main__":
    main()
