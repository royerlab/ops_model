"""
Core CellProfiler feature extraction functions.

This module contains the fundamental feature extraction logic for single object,
colocalization, and neighbor measurements, independent of distributed execution.
"""

import numpy as np
import pandas as pd
import torch
import warnings
from torch.utils.data import Subset

from cp_measure.bulk import get_core_measurements, get_correlation_measurements
from ops_model.data import data_loader


# Cache for feature names to avoid regenerating dummy features
_FEATURE_NAMES_CACHE = {}


def _generate_feature_names_with_nan(
    measurements, prefix, measurement_type="single_object"
):
    """
    Generate feature names by calling measurement functions with dummy data.
    Returns a dictionary with feature names mapped to NaN values.
    Uses caching to avoid regenerating feature names repeatedly.

    Args:
        measurements: Dictionary of measurement functions
        prefix: Prefix for feature names
        measurement_type: Type of measurement ('single_object', 'colocalization', or 'neighbor')

    Returns:
        Dictionary with feature names as keys and np.nan as values
    """
    if not measurements:
        return {}

    # Create cache key from measurement function names and prefix
    cache_key = (tuple(sorted(measurements.keys())), prefix, measurement_type)

    # Check cache first
    if cache_key in _FEATURE_NAMES_CACHE:
        return _FEATURE_NAMES_CACHE[cache_key].copy()

    # Generate feature names with dummy data
    results = {}

    # Create dummy data based on measurement type
    dummy_mask = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=np.int32,
    )

    np.random.seed(42)  # For reproducibility
    dummy_img = np.random.rand(5, 5).astype(np.float32) * 2 - 1  # Range [-1, 1]

    for name, fn in measurements.items():
        try:
            # Call measurement function with dummy data, suppressing warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                if measurement_type == "single_object":
                    dummy_features = fn(dummy_mask, dummy_img)
                elif measurement_type == "colocalization":
                    dummy_img2 = np.random.rand(5, 5).astype(np.float32) * 2 - 1
                    dummy_features = fn(dummy_img, dummy_img2, dummy_mask)
                elif measurement_type == "neighbor":
                    dummy_mask2 = np.array(
                        [
                            [0, 0, 0, 0, 0],
                            [0, 2, 2, 2, 0],
                            [0, 2, 2, 2, 0],
                            [0, 2, 2, 2, 0],
                            [0, 0, 0, 0, 0],
                        ],
                        dtype=np.uint16,
                    )
                    dummy_mask_uint = dummy_mask.astype(np.uint16)
                    dummy_features = fn(dummy_mask_uint, dummy_mask2)
                else:
                    continue

            # Map feature names to NaN
            features_prefixed = {
                f"{prefix}_{feat_name}": np.nan for feat_name in dummy_features.keys()
            }
            results.update(features_prefixed)
        except Exception:
            # Silently skip measurements that can't generate feature names
            # This is expected for some edge cases with small dummy masks
            continue

    # Cache the result for future use
    _FEATURE_NAMES_CACHE[cache_key] = results.copy()

    return results


def single_object_features(
    img: np.ndarray,
    mask: np.ndarray,
    measurements: dict = None,
    prefix: str = "",
):
    """
    Extract single object features from an image and mask.

    cp_measure requires:
    img: H, W (either int of float between -1 and 1)
    mask: H, W (int)

    Args:
        img: Image array (H, W)
        mask: Mask array (H, W)
        measurements: Dictionary of measurement functions
        prefix: Prefix for feature names

    Returns:
        Dictionary of features with prefixed names
    """

    img = np.clip(np.squeeze(img), -1, 1)
    mask = np.squeeze(mask)

    results = {}

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for name, fn in measurements.items():
                if name == "texture":
                    img = (img - np.min(img)) / (
                        np.max(img) - np.min(img) + 1e-8
                    ) * 2 - 1
                features = fn(mask, img)
                features_prefixed = {
                    f"{prefix}_{feat_name}": v for feat_name, v in features.items()
                }
                results.update(features_prefixed)
    except (ValueError, IndexError):
        # Mask is empty or invalid - generate NaN features
        return _generate_feature_names_with_nan(measurements, prefix, "single_object")

    return results


def colocalization_features(
    img1: np.ndarray,
    img2: np.ndarray,
    mask: np.ndarray,
    measurements: dict = None,
    prefix: str = "",
):
    """
    Extract colocalization features from two images and a mask.

    cp_measure requires:
    img: H, W (either int of float between -1 and 1)
    mask: H, W (int)

    Args:
        img1: First image array (H, W)
        img2: Second image array (H, W)
        mask: Mask array (H, W)
        measurements: Dictionary of measurement functions
        prefix: Prefix for feature names

    Returns:
        Dictionary of features with prefixed names
    """

    img1 = np.squeeze(img1)
    img2 = np.squeeze(img2)
    mask = np.squeeze(mask)

    results = {}

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for name, fn in measurements.items():
                features = fn(img1, img2, mask)
                features_prefixed = {
                    f"{prefix}_{feat_name}": v for feat_name, v in features.items()
                }
                results.update(features_prefixed)
    except (ValueError, IndexError):
        # Mask is empty or invalid - generate NaN features
        return _generate_feature_names_with_nan(measurements, prefix, "colocalization")

    return results


def neighbor_features(
    mask1: np.ndarray,
    mask2: np.ndarray,
    measurements: dict = None,
    prefix: str = "",
):
    """
    Extract neighbor features from two masks.

    Args:
        mask1: First mask array (H, W)
        mask2: Second mask array (H, W)
        measurements: Dictionary of measurement functions
        prefix: Prefix for feature names

    Returns:
        Dictionary of features with prefixed names
    """
    mask1 = np.squeeze(mask1).astype(np.uint16)
    mask2 = np.squeeze(mask2).astype(np.uint16)

    results = {}

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for name, fn in measurements.items():
                features = fn(mask1, mask2)
                features_prefixed = {
                    f"{prefix}_{feat_name}": v for feat_name, v in features.items()
                }
                results.update(features_prefixed)
    except (ValueError, IndexError):
        # Mask is empty or invalid - generate NaN features
        return _generate_feature_names_with_nan(measurements, prefix, "neighbor")

    return results


def create_subset(
    experiment_dict,
    bounds: list[int],
):
    """
    Create a dataset subset for the specified index range.

    Args:
        experiment_dict: Dictionary mapping experiment names to FOV lists
        bounds: [start, end] index range

    Returns:
        Tuple of (Subset dataset, label lookup table)
    """

    data_manager = data_loader.OpsDataManager(
        experiments=experiment_dict,
        batch_size=1,
        data_split=(1, 0, 0),
        out_channels=["Phase2D", "mCherry"],
        initial_yx_patch_size=(256, 256),
        verbose=False,
    )

    data_manager.construct_dataloaders(num_workers=0, dataset_type="cell_profile")
    dataset = data_manager.train_loader.dataset

    return (
        Subset(dataset, list(range(bounds[0], bounds[1]))),
        data_manager.train_loader.dataset.int_label_lut,
    )


def extract_cp_features(
    experiment_dict: dict,
    bounds: list[int],
):
    """
    Extract CellProfiler features for a subset of the dataset.

    Args:
        experiment_dict: Dictionary mapping experiment names to FOV lists
        bounds: [start, end] index range to process

    Returns:
        pd.DataFrame: DataFrame with extracted features and metadata
    """

    object_measurements = get_core_measurements()
    colocalization_measurements = get_correlation_measurements()

    subset, int_label_lut = create_subset(experiment_dict, bounds)

    # Pre-define channel and mask metadata
    channels = [("Phase2D", 0), ("mCherry", 1)]
    mask_configs = [
        ("cell", "cell_mask"),
        ("nucleus", "nuc_mask"),
        ("cytoplasm", "cyto_mask"),
    ]

    # Pre-compute channel pairs for colocalization
    channel_pairs = [
        (channels[i], channels[j])
        for i in range(len(channels))
        for j in range(i + 1, len(channels))
    ]

    results_list = []
    for batch in subset:

        if isinstance(batch["data"], torch.Tensor):
            data_np = batch["data"].detach().cpu().numpy()
        else:
            data_np = batch["data"]

        masks_np = {}
        for mask_key in ["cell_mask", "nuc_mask", "cyto_mask"]:
            if isinstance(batch[mask_key], torch.Tensor):
                masks_np[mask_key] = batch[mask_key].detach().cpu().numpy()
            else:
                masks_np[mask_key] = batch[mask_key]

        cell_features = {}

        # Process each mask type
        for mk_name, mk_key in mask_configs:
            mask = masks_np[mk_key][0].astype(np.int32)

            # Single object features for each channel
            for ch_name, ch_idx in channels:
                img = data_np[ch_idx]

                features = single_object_features(
                    img,
                    mask,
                    measurements=object_measurements,
                    prefix=f"single_object_{ch_name}_{mk_name}",
                )
                cell_features.update(features)

            # Colocalization features for channel pairs
            for (ch_name_1, ch_idx_1), (ch_name_2, ch_idx_2) in channel_pairs:
                img1 = data_np[ch_idx_1]
                img2 = data_np[ch_idx_2]

                coloc_features = colocalization_features(
                    img1,
                    img2,
                    mask,
                    measurements=colocalization_measurements,
                    prefix=f"coloc_{ch_name_1}_{ch_name_2}_{mk_name}",
                )
                cell_features.update(coloc_features)

        cell_features["label_int"] = int(batch["gene_label"])
        cell_features["label_str"] = int_label_lut[batch["gene_label"]]
        cell_features["sgRNA"] = batch["crop_info"]["sgRNA"]
        cell_features["experiment"] = batch["crop_info"]["store_key"]
        cell_features["x_position"] = batch["crop_info"]["x_pheno"]
        cell_features["y_position"] = batch["crop_info"]["y_pheno"]
        cell_features["well"] = (
            batch["crop_info"]["well"] + "_" + batch["crop_info"]["store_key"]
        )
        results_list.append(cell_features)

    return pd.DataFrame(results_list)
