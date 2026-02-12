import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import yaml
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingConfig:
    """Configuration for embeddings at a specific level (cell/guide/gene)."""

    compute_embeddings: bool = True
    n_pca_components: int = 128
    n_neighbors: int = 15
    pca: bool = True
    umap: bool = True
    phate: bool = True

    def __post_init__(self):
        """Validate embedding configuration."""
        # Note: UMAP can be computed on PCA features (pca=True) or raw features (pca=False)
        # Both are valid - no forced validation needed
        pass

    def get_embeddings_list(self) -> List[str]:
        """Return list of enabled embedding types."""
        return [
            name
            for name, enabled in [
                ("PCA", self.pca),
                ("UMAP", self.umap),
                ("PHATE", self.phate),
            ]
            if enabled and self.compute_embeddings
        ]


@dataclass
class CombinationConfig:
    """Main configuration object for combining experiments."""

    # Core Fields
    concatenation_method: str
    feature_type: str
    base_dir: str
    feature_dir: str
    output_path: Optional[str] = None

    # Validator Fields - these will be populated from input anndata objects
    cell_type: Optional[str] = None
    embedding_type: Optional[str] = None

    # Experiment/Channel Fields
    experiments: Optional[List[str]] = None
    experiments_channels: Optional[Dict[str, List[str]]] = None
    channel: Optional[str] = None

    # Aggregation & Normalization
    aggregation_level: Optional[str] = "cell"
    normalization: Dict[str, Any] = field(default_factory=dict)
    control_subsampling: Dict[str, Any] = field(default_factory=dict)
    fitted_embeddings: Dict[str, Any] = field(default_factory=dict)
    leiden_clustering: Dict[str, Any] = field(default_factory=dict)

    # Embedding Configs
    embeddings: Dict[str, EmbeddingConfig] = field(default_factory=dict)

    # Raw config for reference
    raw_config: Dict[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        # Initialize embedding configs from raw dict if present
        if "embeddings" in self.raw_config and isinstance(
            self.raw_config["embeddings"], dict
        ):
            self.embeddings = {
                level: EmbeddingConfig(**params)
                for level, params in self.raw_config["embeddings"].items()
            }
        else:
            self.embeddings = {
                "cell_level": EmbeddingConfig(),
                "guide_level": EmbeddingConfig(),
                "gene_level": EmbeddingConfig(),
            }


def load_config(
    config_path: Union[str, Path], output_path_override: Optional[str] = None
) -> CombinationConfig:
    """
    Loads, validates, and processes the configuration file.

    Args:
        config_path: Path to the YAML configuration file.
        output_path_override: Optional path to override the output path in the config.

    Returns:
        A validated CombinationConfig object.
    """
    with open(config_path, "r") as f:
        raw_config = yaml.safe_load(f)

    # Store raw config for embedding initialization
    config_dict = raw_config.copy()

    # Auto-detection mode
    if "feature_extraction_configs" in config_dict:
        _auto_detect_experiments(config_dict)

    # Backwards compatibility for embedding settings
    if "embedding_settings" in config_dict:
        config_dict["embeddings"] = config_dict.pop("embedding_settings")
        for level in ["cell_level", "guide_level", "gene_level"]:
            if level not in config_dict["embeddings"]:
                config_dict["embeddings"][level] = {}

    # Auto-generate output path if not specified
    if not config_dict.get("output_path"):
        config_dict["output_path"] = _generate_output_path(config_dict)

    if output_path_override:
        config_dict["output_path"] = output_path_override

    # Create dataclass instance
    # Pass raw_config separately to handle nested dataclasses
    final_config_dict = {
        k: v for k, v in config_dict.items() if k in CombinationConfig.__annotations__
    }
    final_config_dict["raw_config"] = config_dict

    return CombinationConfig(**final_config_dict)


def _generate_output_path(config: Dict[str, Any]) -> str:
    """Auto-generates an output path if not specified."""
    base_dir = config.get("base_dir", ".")
    method = config.get("concatenation_method", "combined")
    feature_type = config.get("feature_type", "features")

    experiment_part = "multi-experiment"
    if config.get("experiments"):
        if len(config["experiments"]) == 1:
            experiment_part = config["experiments"][0]
        else:
            experiment_part = f"{config['experiments'][0]}_etc"

    filename = f"{method}_{feature_type}_{experiment_part}.h5ad"
    return str(Path(base_dir) / "combined_anndata" / filename)


def normalize_feature_type(feature_type: str) -> str:
    """Normalize feature type names to canonical form."""
    aliases = {
        "dino": "dinov3",
        "dinov3": "dinov3",
        "cellprofiler": "cellprofiler",
        "cell-profiler": "cellprofiler",
    }
    canonical = aliases.get(feature_type.lower().strip())
    if canonical is None:
        logger.warning(f"Unknown feature_type '{feature_type}' - using as-is")
        return feature_type
    return canonical


def _parse_extraction_config(config_path: Path) -> Optional[Tuple[str, str, List[str]]]:
    """Parse a feature extraction config to extract metadata."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    dm_config = config.get("data_manager", {})
    experiments = dm_config.get("experiments", {})
    if not experiments:
        raise ValueError(
            f"Config {config_path} has empty or missing 'data_manager.experiments'"
        )

    experiment_name = list(experiments.keys())[0]
    feature_type = normalize_feature_type(config.get("model_type", ""))
    out_channels = dm_config.get("out_channels", [])

    if isinstance(out_channels, str) and out_channels.lower() in ["random", "all"]:
        logger.warning(
            f"Skipping {config_path.name}: out_channels='{out_channels}' is not supported for auto-detection."
        )
        return None

    channels = [out_channels] if isinstance(out_channels, str) else out_channels
    return experiment_name, feature_type, channels


def _auto_detect_experiments(config_dict: Dict[str, Any]):
    """Auto-detects experiments and channels from feature extraction configs."""
    logger.info("AUTO-DETECTION MODE: Using 'feature_extraction_configs'")
    extraction_configs = config_dict["feature_extraction_configs"]

    detected_experiments = {}
    for config_path_str in extraction_configs:
        try:
            result = _parse_extraction_config(Path(config_path_str))
            if result:
                exp, _, channels = result
                if exp not in detected_experiments:
                    detected_experiments[exp] = []
                detected_experiments[exp].extend(channels)
        except (FileNotFoundError, ValueError) as e:
            logger.error(f"Error parsing {config_path_str}: {e}")
            raise

    config_dict["experiments_channels"] = detected_experiments
    logger.info(f"Auto-detected {len(detected_experiments)} experiments.")
