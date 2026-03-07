import logging
from pathlib import Path
from typing import List, Optional, Tuple

from .config_handler import CombinationConfig
from ops_model.data.feature_metadata import FeatureMetadata

# Initialize logger
logger = logging.getLogger(__name__)


class FilePathBuilder:
    """Constructs file paths for AnnData objects based on configuration."""

    def __init__(self, base_dir: str, feature_dir: str):
        self.base_dir = Path(base_dir)
        self.feature_dir = feature_dir
        self.meta = FeatureMetadata()

    def get_anndata_path(
        self,
        experiment: str,
        feature_type: str,
        aggregation_level: str,
        channel: Optional[str] = None,
        reporter: Optional[str] = None,
    ) -> Path:
        """
        Constructs the path to a specific AnnData file.
        """
        exp_short = experiment.split("_")[0]
        try:
            exp_dir = self._find_experiment_dir(exp_short)
        except FileNotFoundError:
            raise

        anndata_dir = exp_dir / "3-assembly" / self.feature_dir / "anndata_objects"

        # If reporter is not provided, derive it from the channel
        if reporter is None and channel:
            reporter = self.meta.get_biological_signal(exp_short, channel)

        # Determine filename based on aggregation and feature type
        if aggregation_level == "cell":
            filename = (
                f"features_processed_{reporter}.h5ad"
                if reporter
                else "features_processed.h5ad"
            )
        elif aggregation_level in ["guide", "gene"]:
            filename = (
                f"{aggregation_level}_bulked_{reporter}.h5ad"
                if reporter
                else f"{aggregation_level}_bulked.h5ad"
            )
        else:
            raise ValueError(f"Unknown aggregation level: {aggregation_level}")

        return anndata_dir / filename

    def _find_experiment_dir(self, exp_short_name: str) -> Path:
        """
        Finds the full experiment directory, which may have a date suffix.

        Args:
            exp_short_name: The short name of the experiment (e.g., "ops0089").

        Returns:
            The full Path to the experiment directory.

        Raises:
            FileNotFoundError: If no matching directory is found.
        """
        exp_dirs = list(self.base_dir.glob(f"{exp_short_name}*"))
        if not exp_dirs:
            raise FileNotFoundError(
                f"Experiment directory not found for pattern: {exp_short_name}* in {self.base_dir}"
            )
        # Assuming the first match is the correct one
        return exp_dirs[0]


class FileValidator:
    """Validates the existence of input AnnData files based on the configuration."""

    def __init__(self, config: CombinationConfig):
        self.config = config
        self.builder = FilePathBuilder(config.base_dir, config.feature_dir)

    def validate_and_collect_files(self) -> List[Path]:
        """
        Validates that required input files exist and returns a list of valid paths.

        Implements the "warn and continue" strategy. If a file is not found, a
        warning is logged, and the file is omitted from the returned list.

        Returns:
            A list of Path objects for all found AnnData files.
        """
        logger.info("Starting input file validation...")
        valid_paths = []
        method = self.config.concatenation_method

        if method == "vertical":
            paths_to_check = self._get_paths_for_vertical()
        elif method in ["horizontal", "comprehensive"]:
            paths_to_check = self._get_paths_for_horizontal_or_comprehensive()
        else:
            raise ValueError(f"Unknown concatenation method for validation: {method}")

        for path in paths_to_check:
            if path.exists():
                valid_paths.append(path)
                logger.info(f"✓ Found file: {path}")
            else:
                logger.warning(f"✗ WARNING: File not found, skipping: {path}")

        if not valid_paths:
            logger.error("No valid input files were found. Cannot proceed.")

        logger.info(f"Validation complete. Found {len(valid_paths)} valid input files.")
        return valid_paths

    def _get_paths_for_vertical(self) -> List[Path]:
        """Generate expected file paths for vertical combination."""
        paths = []
        for exp in self.config.experiments or []:
            try:
                path = self.builder.get_anndata_path(
                    experiment=exp,
                    feature_type=self.config.feature_type,
                    aggregation_level=self.config.aggregation_level or "cell",
                    channel=self.config.channel,
                )
                paths.append(path)
            except FileNotFoundError as e:
                logger.warning(e)
        return paths

    def _get_paths_for_horizontal_or_comprehensive(self) -> List[Path]:
        """Generate expected file paths for horizontal/comprehensive combination."""
        paths = []
        for exp, channels in (self.config.experiments_channels or {}).items():
            for channel in channels:
                try:
                    path = self.builder.get_anndata_path(
                        experiment=exp,
                        feature_type=self.config.feature_type,
                        aggregation_level="cell",  # Horizontal is always cell level first
                        channel=channel,
                    )
                    paths.append(path)
                except FileNotFoundError as e:
                    logger.warning(e)
        return paths
