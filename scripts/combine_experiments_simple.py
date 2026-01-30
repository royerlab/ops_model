#!/usr/bin/env python
"""
Combine data from multiple experiments using a config file.

Supports two concatenation methods:
- Vertical: Combine observations (cells/guides/genes) across experiments
- Horizontal: Combine features from different experiment/channel pairs

Usage:
    # Using config file
    python scripts/combine_experiments_simple.py --config configs/combine_experiments/phase2d_all.yml

    # Override output path
    python scripts/combine_experiments_simple.py --config configs/combine_experiments/phase2d_all.yml --output-path /custom/path.h5ad

    # Validate config without running
    python scripts/combine_experiments_simple.py --config configs/combine_experiments/phase2d_all.yml --validate-only
"""

import argparse
import sys
from pathlib import Path
import yaml

from ops_model.features.anndata_utils import (
    concatenate_anndata_objects,
    concatenate_features_by_channel,
    concatenate_experiments_comprehensive,
    compute_embeddings,
    create_aggregated_embeddings,
)


def load_config(config_path: str) -> dict:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Dictionary with configuration
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Set default concatenation method
    if "concatenation_method" not in config:
        config["concatenation_method"] = "vertical"

    # Apply defaults based on concatenation method
    method = config["concatenation_method"]

    defaults = {
        "base_dir": "/hpc/projects/intracellular_dashboard/ops",
        "embeddings": {"n_pca_components": 128, "n_umap_neighbors": 15},
        "normalization": {"normalize_on_pooling": True, "normalize_on_controls": False},
        "control_subsampling": {
            "enabled": False,
            "control_gene": "NTC",
            "group_size": 4,
            "random_seed": None,
        },
        "fitted_embeddings": {"enabled": False, "use_pca_for_umap": True},
        "leiden_clustering": {"enabled": False, "resolutions": [1.0]},
    }

    # Add method-specific defaults
    if method == "vertical":
        defaults["concatenation"] = {"batch_key": "experiment", "join": "inner"}
        if "aggregation_level" not in config:
            config["aggregation_level"] = "cell"
    elif method == "horizontal":
        if "aggregation_level" not in config:
            raise ValueError(
                "aggregation_level is required for horizontal concatenation"
            )

    # Merge defaults
    for key, value in defaults.items():
        if key not in config:
            config[key] = value
        elif isinstance(value, dict):
            for subkey, subvalue in value.items():
                if subkey not in config[key]:
                    config[key][subkey] = subvalue

    # Generate output path if not specified
    if "output_path" not in config or config["output_path"] is None:
        feature_type = config["feature_type"]
        aggregation_level = config.get("aggregation_level", "cell")

        if method == "vertical":
            channel = config.get("channel", "all")
            channel_str = channel if channel else "all"
            config["output_path"] = (
                f"{config['base_dir']}/combined_datasets/"
                f"combined_{feature_type}_{channel_str}_{aggregation_level}.h5ad"
            )
        elif method == "horizontal":
            config["output_path"] = (
                f"{config['base_dir']}/combined_datasets/"
                f"combined_{feature_type}_horizontal_{aggregation_level}.h5ad"
            )
        else:  # comprehensive
            config["output_path"] = (
                f"{config['base_dir']}/combined_datasets/"
                f"combined_{feature_type}_comprehensive.h5ad"
            )

    return config


def validate_config(config: dict) -> bool:
    """
    Validate configuration parameters.

    Args:
        config: Configuration dictionary

    Returns:
        True if valid, raises ValueError otherwise
    """
    # Validate concatenation method
    method = config.get("concatenation_method", "vertical")
    valid_methods = ["vertical", "horizontal", "comprehensive"]
    if method not in valid_methods:
        raise ValueError(
            f"concatenation_method must be one of {valid_methods}, got: {method}"
        )

    # Validate feature_type
    if "feature_type" not in config:
        raise ValueError("Missing required field: feature_type")

    valid_feature_types = ["dino", "dinov3", "cellprofiler"]
    if config["feature_type"] not in valid_feature_types:
        raise ValueError(
            f"feature_type must be one of {valid_feature_types}, "
            f"got: {config['feature_type']}"
        )

    # Method-specific validation
    if method == "vertical":
        # Vertical requires: experiments, channel (for dino)
        if "experiments" not in config:
            raise ValueError("Missing required field for vertical: experiments")

        if (
            not isinstance(config["experiments"], list)
            or len(config["experiments"]) == 0
        ):
            raise ValueError("experiments must be a non-empty list")

        if config["feature_type"] in ["dino", "dinov3"] and "channel" not in config:
            raise ValueError("channel is required for dino/dinov3 feature_type")

        # Validate concatenation settings
        valid_joins = ["inner", "outer"]
        join = config["concatenation"]["join"]
        if join not in valid_joins:
            raise ValueError(
                f"concatenation.join must be one of {valid_joins}, got: {join}"
            )

        # Validate aggregation level
        valid_levels = ["cell", "guide", "gene"]
        aggregation_level = config.get("aggregation_level", "cell")
        if aggregation_level not in valid_levels:
            raise ValueError(
                f"aggregation_level must be one of {valid_levels}, got: {aggregation_level}"
            )

    elif method == "horizontal":
        # Horizontal requires: experiments_channels, aggregation_level
        if "experiments_channels" not in config:
            raise ValueError(
                "Missing required field for horizontal: experiments_channels"
            )

        if (
            not isinstance(config["experiments_channels"], list)
            or len(config["experiments_channels"]) == 0
        ):
            raise ValueError("experiments_channels must be a non-empty list")

        # Validate each entry has experiment and channel
        for i, item in enumerate(config["experiments_channels"]):
            if not isinstance(item, dict):
                raise ValueError(
                    f"experiments_channels[{i}] must be a dict with 'experiment' and 'channel'"
                )
            if "experiment" not in item or "channel" not in item:
                raise ValueError(
                    f"experiments_channels[{i}] must have 'experiment' and 'channel' fields"
                )

        # Validate aggregation level (cell not supported for horizontal)
        if "aggregation_level" not in config:
            raise ValueError(
                "aggregation_level is required for horizontal concatenation"
            )

        aggregation_level = config["aggregation_level"]
        valid_levels = ["guide", "gene"]
        if aggregation_level not in valid_levels:
            raise ValueError(
                f"aggregation_level for horizontal must be one of {valid_levels}, got: {aggregation_level}. "
                "Cell-level not supported for horizontal concatenation."
            )

    elif method == "comprehensive":
        # Comprehensive requires: experiments_channels (same as horizontal)
        if "experiments_channels" not in config:
            raise ValueError(
                "Missing required field for comprehensive: experiments_channels"
            )

        if (
            not isinstance(config["experiments_channels"], list)
            or len(config["experiments_channels"]) == 0
        ):
            raise ValueError("experiments_channels must be a non-empty list")

        # Validate each entry has experiment and channel
        for i, item in enumerate(config["experiments_channels"]):
            if not isinstance(item, dict):
                raise ValueError(
                    f"experiments_channels[{i}] must be a dict with 'experiment' and 'channel'"
                )
            if "experiment" not in item or "channel" not in item:
                raise ValueError(
                    f"experiments_channels[{i}] must have 'experiment' and 'channel' fields"
                )

        # Comprehensive always produces guide + gene, aggregation_level not needed
        # But if provided, warn that it's ignored
        if "aggregation_level" in config:
            print(
                "Note: aggregation_level is ignored for comprehensive method (produces both guide and gene)"
            )

    print("✓ Config validation passed")
    return True


def validate_experiment_files(config: dict) -> tuple:
    """
    Check if experiment files exist.

    Args:
        config: Configuration dictionary

    Returns:
        Tuple of (all_exist: bool, file_paths: list)
    """
    base_dir = Path(config["base_dir"])
    feature_type = config["feature_type"]
    method = config.get("concatenation_method", "vertical")

    file_paths = []
    missing = []

    if method == "vertical":
        # Vertical: check cell-level or aggregated files
        channel = config.get("channel")
        aggregation_level = config.get("aggregation_level", "cell")

        for exp in config["experiments"]:
            exp_short = exp.split("_")[0]
            # Find experiment directory (may have date suffix)
            exp_dirs = list(base_dir.glob(f"{exp_short}*"))
            if not exp_dirs:
                print(f"✗ Experiment directory not found: {base_dir}/{exp_short}*")
                missing.append(f"{exp_short}*")
                continue
            exp_dir = exp_dirs[0]

            # Build file path based on aggregation level
            if aggregation_level == "cell":
                if feature_type == "cellprofiler":
                    file_path = (
                        exp_dir
                        / "3-assembly"
                        / "cell-profiler"
                        / "anndata_objects"
                        / "features_processed.h5ad"
                    )
                else:  # dino/dinov3
                    file_path = (
                        exp_dir
                        / "3-assembly"
                        / f"{feature_type}_features"
                        / "anndata_objects"
                        / f"features_processed_{channel}.h5ad"
                    )
            else:  # guide or gene
                file_prefix = f"{aggregation_level}_bulked_umap"
                if feature_type == "cellprofiler":
                    file_path = (
                        exp_dir
                        / "3-assembly"
                        / "cell-profiler"
                        / "anndata_objects"
                        / f"{file_prefix}.h5ad"
                    )
                else:  # dino/dinov3
                    file_path = (
                        exp_dir
                        / "3-assembly"
                        / f"{feature_type}_features"
                        / "anndata_objects"
                        / f"{file_prefix}_{channel}.h5ad"
                    )

            file_paths.append(str(file_path))

            # Check if exists
            if not file_path.exists():
                missing.append(str(file_path))
                print(f"✗ Missing: {file_path}")
            else:
                print(f"✓ Found: {exp}")

    elif method == "horizontal":
        # Horizontal: check aggregated files for each experiment/channel pair
        aggregation_level = config["aggregation_level"]
        file_prefix = f"{aggregation_level}_bulked_umap"

        for item in config["experiments_channels"]:
            exp = item["experiment"]
            channel = item["channel"]

            exp_short = exp.split("_")[0]
            # Find experiment directory
            exp_dirs = list(base_dir.glob(f"{exp_short}*"))
            if not exp_dirs:
                print(f"✗ Experiment directory not found: {base_dir}/{exp_short}*")
                missing.append(f"{exp_short}*")
                continue
            exp_dir = exp_dirs[0]

            # Build file path
            if feature_type == "cellprofiler":
                file_path = (
                    exp_dir
                    / "3-assembly"
                    / "cell-profiler"
                    / "anndata_objects"
                    / f"{file_prefix}_{channel}.h5ad"
                )
            else:  # dino/dinov3
                file_path = (
                    exp_dir
                    / "3-assembly"
                    / f"{feature_type}_features"
                    / "anndata_objects"
                    / f"{file_prefix}_{channel}.h5ad"
                )

            file_paths.append(str(file_path))

            # Check if exists
            if not file_path.exists():
                missing.append(str(file_path))
                print(f"✗ Missing: {file_path}")
            else:
                print(f"✓ Found: {exp}/{channel}")

    elif method == "comprehensive":
        # Comprehensive: check cell-level files for each experiment/channel pair
        for item in config["experiments_channels"]:
            exp = item["experiment"]
            channel = item["channel"]

            exp_short = exp.split("_")[0]
            # Find experiment directory
            exp_dirs = list(base_dir.glob(f"{exp_short}*"))
            if not exp_dirs:
                print(f"✗ Experiment directory not found: {base_dir}/{exp_short}*")
                missing.append(f"{exp_short}*")
                continue
            exp_dir = exp_dirs[0]

            # Build file path to cell-level data
            if feature_type == "cellprofiler":
                file_path = (
                    exp_dir
                    / "3-assembly"
                    / "cell-profiler"
                    / "anndata_objects"
                    / f"features_processed_{channel}.h5ad"
                )
            else:  # dino/dinov3
                file_path = (
                    exp_dir
                    / "3-assembly"
                    / f"{feature_type}_features"
                    / "anndata_objects"
                    / f"features_processed_{channel}.h5ad"
                )

            file_paths.append(str(file_path))

            # Check if exists
            if not file_path.exists():
                missing.append(str(file_path))
                print(f"✗ Missing: {file_path}")
            else:
                print(f"✓ Found: {exp}/{channel}")

    if missing:
        print(f"\n✗ Error: {len(missing)} file(s) not found")
        return False, file_paths
    else:
        print(f"\n✓ All {len(file_paths)} files found")
        return True, file_paths


def combine_experiments_vertical(config: dict, file_paths: list) -> None:
    """
    Combine experiments vertically (concatenate observations).

    Args:
        config: Configuration dictionary
        file_paths: List of file paths to combine
    """
    experiments = config["experiments"]
    feature_type = config["feature_type"]
    channel = config.get("channel", "all")
    output_path = config["output_path"]
    batch_key = config["concatenation"]["batch_key"]
    join = config["concatenation"]["join"]
    n_pca = config["embeddings"]["n_pca_components"]
    n_neighbors = config["embeddings"]["n_umap_neighbors"]

    print("\n" + "=" * 80)
    print("COMBINING EXPERIMENTS")
    print("=" * 80)
    print(f"Experiments: {len(experiments)}")
    print(f"Feature type: {feature_type}")
    print(f"Channel: {channel}")
    print(f"Join method: {join}")
    print("=" * 80 + "\n")

    # Concatenate
    print("Concatenating AnnData objects...")
    adata_combined = concatenate_anndata_objects(
        file_paths,
        batch_key=batch_key,
        join=join,
    )

    print(
        f"✓ Combined shape: {adata_combined.shape[0]} cells × {adata_combined.shape[1]} features"
    )

    # Compute embeddings on cell-level data
    print(
        f"\nComputing cell-level embeddings (PCA: {n_pca}, UMAP neighbors: {n_neighbors})..."
    )
    adata_combined = compute_embeddings(
        adata_combined,
        n_pca_components=n_pca,
        n_neighbors=n_neighbors,
    )

    # Save cell-level data
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving cell-level data to: {output_path}")
    adata_combined.write_h5ad(output_path)
    print(f"✓ Saved cell-level: {adata_combined.shape}")

    # Guide-level aggregation and embeddings
    print(f"\nCreating guide-level aggregation...")
    adata_guide = create_aggregated_embeddings(
        adata_combined,
        level="guide",
        n_pca_components=n_pca,
        n_neighbors=n_neighbors,
        preserve_batch_info=True,  # Keep experiment tracking
        subsample_controls=config["control_subsampling"]["enabled"],
        control_gene=config["control_subsampling"]["control_gene"],
        control_group_size=config["control_subsampling"]["group_size"],
        random_seed=config["control_subsampling"]["random_seed"],
    )
    guide_path = output_path.parent / output_path.name.replace(".h5ad", "_guide.h5ad")
    adata_guide.write_h5ad(guide_path)
    print(f"✓ Saved guide-level: {guide_path}")
    print(f"  Shape: {adata_guide.shape}")

    # Gene-level aggregation and embeddings
    print(f"\nCreating gene-level aggregation...")
    adata_gene = create_aggregated_embeddings(
        adata_combined,
        level="gene",
        n_pca_components=n_pca,
        n_neighbors=n_neighbors,
        preserve_batch_info=True,
        subsample_controls=config["control_subsampling"]["enabled"],
        control_gene=config["control_subsampling"]["control_gene"],
        control_group_size=config["control_subsampling"]["group_size"],
        random_seed=config["control_subsampling"]["random_seed"],
    )
    gene_path = output_path.parent / output_path.name.replace(".h5ad", "_gene.h5ad")
    adata_gene.write_h5ad(gene_path)
    print(f"✓ Saved gene-level: {gene_path}")
    print(f"  Shape: {adata_gene.shape}")

    print("\n" + "=" * 80)
    print("DONE!")
    print("=" * 80)
    print(f"Cell-level: {adata_combined.shape} → {output_path}")
    print(f"Guide-level: {adata_guide.shape} → {guide_path}")
    print(f"Gene-level: {adata_gene.shape} → {gene_path}")
    print(f"Experiments: {list(adata_combined.obs['experiment'].unique())}")
    print("=" * 80 + "\n")


def combine_experiments_horizontal(config: dict) -> None:
    """
    Combine experiments horizontally (concatenate features).

    Args:
        config: Configuration dictionary
    """
    # Convert config format to function format
    experiments_channels = [
        (item["experiment"], item["channel"]) for item in config["experiments_channels"]
    ]

    feature_type = config["feature_type"]
    aggregation_level = config["aggregation_level"]
    output_path = config["output_path"]
    n_pca = config["embeddings"]["n_pca_components"]
    n_neighbors = config["embeddings"]["n_umap_neighbors"]

    print("\n" + "=" * 80)
    print("HORIZONTAL FEATURE CONCATENATION")
    print("=" * 80)
    print(f"Sources: {len(experiments_channels)}")
    print(f"Feature type: {feature_type}")
    print(f"Aggregation level: {aggregation_level}")
    for exp, ch in experiments_channels:
        print(f"  - {exp}/{ch}")
    print("=" * 80 + "\n")

    # Concatenate features
    print("Concatenating features...")
    adata_combined = concatenate_features_by_channel(
        experiments_channels=experiments_channels,
        feature_type=feature_type,
        aggregation_level=aggregation_level,
        base_dir=config["base_dir"],
        recompute_embeddings=True,
        n_pca_components=n_pca,
        n_umap_neighbors=n_neighbors,
    )

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving to: {output_path}")
    adata_combined.write_h5ad(output_path)

    print("\n" + "=" * 80)
    print("DONE!")
    print("=" * 80)
    print(f"Combined data: {adata_combined.shape}")
    print(f"Sources combined: {len(experiments_channels)}")
    print(f"Output: {output_path}")
    print("=" * 80 + "\n")


def combine_experiments_comprehensive(config: dict) -> None:
    """
    Combine experiments using comprehensive strategy.

    Args:
        config: Configuration dictionary
    """
    # Convert config format to function format
    experiments_channels = [
        (item["experiment"], item["channel"]) for item in config["experiments_channels"]
    ]

    feature_type = config["feature_type"]
    output_path = config["output_path"]
    n_pca = config["embeddings"]["n_pca_components"]
    n_neighbors = config["embeddings"]["n_umap_neighbors"]

    print("\n" + "=" * 80)
    print("COMPREHENSIVE COMBINATION")
    print("=" * 80)
    print(f"Sources: {len(experiments_channels)}")
    print(f"Feature type: {feature_type}")
    for exp, ch in experiments_channels:
        print(f"  - {exp}/{ch}")
    print("=" * 80 + "\n")

    # Run comprehensive combination
    adata_guide, adata_gene = concatenate_experiments_comprehensive(
        experiments_channels=experiments_channels,
        feature_type=feature_type,
        base_dir=config["base_dir"],
        recompute_embeddings=True,
        n_pca_components=n_pca,
        n_umap_neighbors=n_neighbors,
        verbose=True,
        normalize_on_pooling=config["normalization"]["normalize_on_pooling"],
        normalize_on_controls=config["normalization"]["normalize_on_controls"],
        subsample_controls=config["control_subsampling"]["enabled"],
        control_gene=config["control_subsampling"]["control_gene"],
        control_group_size=config["control_subsampling"]["group_size"],
        random_seed=config["control_subsampling"]["random_seed"],
        fit_on_aggregated_controls=config["fitted_embeddings"]["enabled"],
        use_pca_for_umap=config["fitted_embeddings"]["use_pca_for_umap"],
        leiden_resolutions=(
            config["leiden_clustering"]["resolutions"]
            if config["leiden_clustering"]["enabled"]
            else None
        ),
    )

    # Save both levels
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    guide_path = output_path.parent / output_path.name.replace(".h5ad", "_guide.h5ad")
    gene_path = output_path.parent / output_path.name.replace(".h5ad", "_gene.h5ad")

    print(f"\nSaving outputs...")
    adata_guide.write_h5ad(guide_path)
    print(f"  Guide-level: {guide_path}")
    adata_gene.write_h5ad(gene_path)
    print(f"  Gene-level: {gene_path}")

    print("\n" + "=" * 80)
    print("DONE!")
    print("=" * 80)
    print(f"Guide-level: {adata_guide.shape} → {guide_path}")
    print(f"Gene-level: {adata_gene.shape} → {gene_path}")

    # Print biological groups summary
    if "comprehensive_metadata" in adata_gene.uns:
        meta = adata_gene.uns["comprehensive_metadata"]
        print(f"\nBiological groups: {meta['n_biological_signals']}")
        for bio_signal, info in meta["biological_groups"].items():
            print(
                f"  [{info['aggregation_type'].upper():10s}] {bio_signal}: {info['n_cells_total']} cells"
            )

    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Combine cell-level data from multiple experiments using a config file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using config file
  python scripts/combine_experiments_simple.py --config configs/combine_experiments/phase2d_all.yml

  # Override output path
  python scripts/combine_experiments_simple.py --config configs/combine_experiments/phase2d_all.yml --output-path /custom/path.h5ad

  # Validate config without running
  python scripts/combine_experiments_simple.py --config configs/combine_experiments/phase2d_all.yml --validate-only
        """,
    )

    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML config file"
    )
    parser.add_argument(
        "--output-path", type=str, default=None, help="Override output path from config"
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate config and check files without running",
    )

    args = parser.parse_args()

    try:
        # Load config
        print(f"Loading config: {args.config}")
        config = load_config(args.config)

        # Override output path if provided
        if args.output_path:
            config["output_path"] = args.output_path

        # Print config summary
        method = config.get("concatenation_method", "vertical")
        print("\nConfiguration:")
        print(f"  Method: {method}")
        if method == "vertical":
            print(f"  Experiments: {len(config['experiments'])}")
            print(f"  Channel: {config.get('channel', 'N/A')}")
        else:  # horizontal
            print(f"  Sources: {len(config['experiments_channels'])}")
        print(f"  Feature type: {config['feature_type']}")
        print(f"  Aggregation level: {config.get('aggregation_level', 'cell')}")
        print(f"  Output: {config['output_path']}")
        if "description" in config:
            print(f"  Description: {config['description']}")

        # Validate config
        print("\nValidating configuration...")
        validate_config(config)

        # Check experiment files
        method = config.get("concatenation_method", "vertical")
        if method == "vertical":
            print(f"\nChecking {len(config['experiments'])} experiment files...")
        else:
            print(f"\nChecking {len(config['experiments_channels'])} source files...")

        files_exist, file_paths = validate_experiment_files(config)

        if not files_exist:
            print("\n✗ Some files are missing. Please check the paths above.")
            sys.exit(1)

        # If validate-only, stop here
        if args.validate_only:
            print("\n✓ Validation complete. Use without --validate-only to run.")
            sys.exit(0)

        # Combine experiments based on method
        if method == "vertical":
            combine_experiments_vertical(config, file_paths)
        elif method == "horizontal":
            combine_experiments_horizontal(config)
        elif method == "comprehensive":
            combine_experiments_comprehensive(config)
        else:
            raise ValueError(f"Unknown concatenation method: {method}")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
