"""
Core CellProfiler feature extraction functions.

This module contains the fundamental feature extraction logic for single object,
colocalization, and neighbor measurements, independent of distributed execution.
"""

import ast
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch
import warnings
import zarr
from torch.utils.data import Subset

from cp_measure.bulk import get_core_measurements, get_correlation_measurements
from ops_model.data import data_loader
from ops_model.data.data_loader import CellProfileDataset
from ops_model.data.paths import OpsPaths


# Cache for feature names to avoid regenerating dummy features
_FEATURE_NAMES_CACHE = {}

# Shared state for Pool workers (set via initializer, inherited via fork)
_worker_state = {}


def safe_item_conversion(v, feature_name, crop_info=None):
    """
    Safely convert a feature value to a scalar.

    If the value is an array with multiple elements (indicating multiple
    objects were detected), log a warning and return NaN.

    Args:
        v: Feature value (scalar or array)
        feature_name: Name of the feature for logging
        crop_info: Optional crop info dict for logging

    Returns:
        Scalar value or np.nan
    """
    arr = np.asarray(v)
    if arr.size == 1:
        return arr.item()
    else:
        # Multiple objects detected in mask
        print(f"WARNING: Feature '{feature_name}' has {arr.size} values (expected 1)")
        if crop_info:
            print(f"  Crop info: {crop_info}")
        return np.nan


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

            # Map feature names to NaN (as arrays for consistency)
            features_prefixed = {
                f"{prefix}_{feat_name}": np.array([np.nan])
                for feat_name in dummy_features.keys()
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

    # img = np.clip(np.squeeze(img), -1, 1) TODO: figure out why this was here, does not makes sense
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
    out_channels: list[str] = None,
):
    """
    Create a dataset subset for the specified index range.

    Args:
        experiment_dict: Dictionary mapping experiment names to FOV lists
        bounds: [start, end] index range
        out_channels: List of channel names (default: ["Phase2D", "mCherry"])

    Returns:
        Tuple of (Subset dataset, label lookup table)
    """
    if out_channels is None:
        out_channels = ["Phase2D", "mCherry"]

    data_manager = data_loader.OpsDataManager(
        experiments=experiment_dict,
        batch_size=1,
        data_split=(1, 0, 0),
        out_channels=out_channels,
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
    out_channels: list[str] = None,
):
    """
    Extract CellProfiler features for a subset of the dataset.

    Args:
        experiment_dict: Dictionary mapping experiment names to FOV lists
        bounds: [start, end] index range to process
        out_channels: List of channel names (default: ["Phase2D", "mCherry"])

    Returns:
        pd.DataFrame: DataFrame with extracted features and metadata
    """
    if out_channels is None:
        out_channels = ["Phase2D", "mCherry"]

    object_measurements = get_core_measurements()
    shape_features_fcn = object_measurements.pop("sizeshape")
    colocalization_measurements = get_correlation_measurements()

    subset, int_label_lut = create_subset(experiment_dict, bounds, out_channels)

    # Pre-define channel and mask metadata
    # Create indexed channel list: [(channel_name, index), ...]
    channels = [(ch, i) for i, ch in enumerate(out_channels)]
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

            shape_features = shape_features_fcn(mask, np.zeros_like(mask))
            shape_features_prefixed = {
                f"{mk_name}_{feat_name}": v for feat_name, v in shape_features.items()
            }
            cell_features.update(
                {
                    k: safe_item_conversion(v, k, batch.get("crop_info"))
                    for k, v in shape_features_prefixed.items()
                }
            )

            # Single object features for each channel
            for ch_name, ch_idx in channels:
                img = data_np[ch_idx]

                features = single_object_features(
                    img,
                    mask,
                    measurements=object_measurements,
                    prefix=f"single_object_{ch_name}_{mk_name}",
                )
                cell_features.update(
                    {
                        k: safe_item_conversion(v, k, batch.get("crop_info"))
                        for k, v in features.items()
                    }
                )

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
                cell_features.update(
                    {
                        k: safe_item_conversion(v, k, batch.get("crop_info"))
                        for k, v in coloc_features.items()
                    }
                )

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


# ---------------------------------------------------------------------------
# Parallel extraction via multiprocessing.Pool with shared initializer
# ---------------------------------------------------------------------------


def _read_well_labels(args):
    """Read and filter labels CSV for a single well. Used with ThreadPoolExecutor."""
    exp_name, well = args
    labels_tmp = pd.read_csv(OpsPaths(exp_name, well=well).links["training"])
    labels_tmp = labels_tmp.dropna(subset=["segmentation_id"])
    from ops_model.data.qc.qc_labels import filter_small_bboxes

    labels_tmp, _ = filter_small_bboxes(labels_tmp, threshold=5)
    labels_tmp["store_key"] = exp_name
    labels_tmp["well"] = well
    return labels_tmp


def load_labels_parallel(experiment_dict: dict) -> pd.DataFrame:
    """Read label CSVs for all wells in parallel via ThreadPoolExecutor."""
    well_args = [
        (exp_name, well)
        for exp_name, wells in experiment_dict.items()
        for well in wells
    ]
    with ThreadPoolExecutor(max_workers=len(well_args)) as pool:
        dfs = list(pool.map(_read_well_labels, well_args))

    labels_df = pd.concat(dfs, ignore_index=True)
    if "Gene name" in labels_df.columns:
        labels_df["gene_name"] = labels_df["Gene name"].fillna("NTC")
    elif "gene_name" in labels_df.columns:
        labels_df["gene_name"] = labels_df["gene_name"].fillna("NTC")
    else:
        raise ValueError("No gene name column found in labels file")
    labels_df["total_index"] = np.arange(len(labels_df))
    return labels_df


def _init_worker(labels_df, label_int_lut, int_label_lut, experiment_dict, out_channels):
    """Pool initializer: store shared data and open per-process zarr handles."""
    stores = {}
    for exp_name in experiment_dict:
        stores[exp_name] = zarr.open_group(
            str(OpsPaths(exp_name).stores["phenotyping_v3"]), mode="r"
        )

    _worker_state["dataset"] = CellProfileDataset(
        stores=stores,
        labels_df=labels_df,
        initial_yx_patch_size=(256, 256),
        final_yx_patch_size=(128, 128),
        out_channels=out_channels,
        label_int_lut=label_int_lut,
        int_label_lut=int_label_lut,
    )
    _worker_state["int_label_lut"] = int_label_lut
    _worker_state["out_channels"] = out_channels


def _process_cell_range(bounds):
    """Worker function: extract features for a range of cells using shared state."""
    dataset = _worker_state["dataset"]
    int_label_lut = _worker_state["int_label_lut"]
    out_channels = _worker_state["out_channels"]

    subset = Subset(dataset, list(range(bounds[0], bounds[1])))

    object_measurements = get_core_measurements()
    shape_features_fcn = object_measurements.pop("sizeshape")
    colocalization_measurements = get_correlation_measurements()

    channels = [(ch, i) for i, ch in enumerate(out_channels)]
    mask_configs = [
        ("cell", "cell_mask"),
        ("nucleus", "nuc_mask"),
        ("cytoplasm", "cyto_mask"),
    ]
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
        for mk_name, mk_key in mask_configs:
            mask = masks_np[mk_key][0].astype(np.int32)
            shape_features = shape_features_fcn(mask, np.zeros_like(mask))
            shape_features_prefixed = {
                f"{mk_name}_{feat_name}": v for feat_name, v in shape_features.items()
            }
            cell_features.update(
                {
                    k: safe_item_conversion(v, k, batch.get("crop_info"))
                    for k, v in shape_features_prefixed.items()
                }
            )
            for ch_name, ch_idx in channels:
                img = data_np[ch_idx]
                features = single_object_features(
                    img, mask, measurements=object_measurements,
                    prefix=f"single_object_{ch_name}_{mk_name}",
                )
                cell_features.update(
                    {
                        k: safe_item_conversion(v, k, batch.get("crop_info"))
                        for k, v in features.items()
                    }
                )
            for (ch_name_1, ch_idx_1), (ch_name_2, ch_idx_2) in channel_pairs:
                coloc_features = colocalization_features(
                    data_np[ch_idx_1], data_np[ch_idx_2], mask,
                    measurements=colocalization_measurements,
                    prefix=f"coloc_{ch_name_1}_{ch_name_2}_{mk_name}",
                )
                cell_features.update(
                    {
                        k: safe_item_conversion(v, k, batch.get("crop_info"))
                        for k, v in coloc_features.items()
                    }
                )

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


def extract_cp_features_parallel(
    experiment_dict: dict,
    bounds: list[int],
    out_channels: list[str] = None,
    num_workers: int = None,
):
    """
    Extract CellProfiler features using multiprocessing.Pool with shared state.

    Oversubscribes workers relative to CPUs: each worker is ~60% CPU / ~40% NFS
    I/O wait. With 1.5x workers per CPU, the OS scheduler fills idle time from
    I/O-blocked processes with compute from other processes — effective pipelining
    without pickle overhead.

    Args:
        experiment_dict: Dictionary mapping experiment names to FOV lists
        bounds: [start, end] index range to process
        out_channels: List of channel names
        num_workers: Number of worker processes (default: CPUs × 1.5)

    Returns:
        pd.DataFrame with extracted features
    """
    if out_channels is None:
        out_channels = ["Phase2D", "mCherry"]
    if num_workers is None:
        cpus = len(os.sched_getaffinity(0))
        # Oversubscribe 1.5x: workers are ~60% CPU / ~40% NFS I/O wait.
        # When one process blocks on I/O, another runs on that core.
        num_workers = max(1, int(cpus * 1.5))

    # Load labels ONCE in parallel (ThreadPool for NFS I/O)
    labels_df = load_labels_parallel(experiment_dict)

    # Build LUTs once
    gene_labels = sorted(labels_df["gene_name"].unique())
    label_int_lut = {gene: i for i, gene in enumerate(gene_labels)}
    int_label_lut = {i: gene for i, gene in enumerate(gene_labels)}

    # Split bounds into sub-ranges
    total = bounds[1] - bounds[0]
    chunk = math.ceil(total / num_workers)
    sub_bounds = [
        [bounds[0] + i * chunk, min(bounds[0] + (i + 1) * chunk, bounds[1])]
        for i in range(num_workers)
        if bounds[0] + i * chunk < bounds[1]
    ]

    # Process in parallel — workers inherit labels_df via fork (copy-on-write)
    # Each worker opens its own zarr handles in the initializer
    with Pool(
        processes=len(sub_bounds),
        initializer=_init_worker,
        initargs=(labels_df, label_int_lut, int_label_lut, experiment_dict, out_channels),
    ) as pool:
        dfs = pool.map(_process_cell_range, sub_bounds)

    return pd.concat(dfs, ignore_index=True)


# ---------------------------------------------------------------------------
# Bulk-read architecture: read all needed zarr chunks into RAM, then compute
# ---------------------------------------------------------------------------


def _bulk_read_chunks(zarr_arr, chunk_coords, channel_indices=None, num_threads=16):
    """Read specific chunks from a zarr array in parallel.

    Returns dict mapping (chunk_y, chunk_x) -> numpy array.
    For image arrays with channel_indices, values are (C, H, W).
    For mask arrays, values are (H, W).
    """
    chunk_h = zarr_arr.chunks[-2]
    chunk_w = zarr_arr.chunks[-1]
    max_y = zarr_arr.shape[-2]
    max_x = zarr_arr.shape[-1]

    def _read_one(coord):
        cy, cx = coord
        y0 = cy * chunk_h
        x0 = cx * chunk_w
        y1 = min(y0 + chunk_h, max_y)
        x1 = min(x0 + chunk_w, max_x)
        if channel_indices is not None:
            # Read each channel separately to avoid zarr v3 fancy indexing issues
            slices = [
                np.asarray(zarr_arr[0, ch, 0, y0:y1, x0:x1])
                for ch in channel_indices
            ]
            data = np.stack(slices, axis=0)  # (C, H, W)
        else:
            data = np.asarray(zarr_arr[0, 0, 0, y0:y1, x0:x1])
        return coord, data

    cache = {}
    coords = list(chunk_coords)
    with ThreadPoolExecutor(num_threads) as pool:
        for coord, data in pool.map(_read_one, coords):
            cache[coord] = data
    return cache


def _crop_from_cache(cache, chunk_size, y0, x0, y1, x1):
    """Crop a rectangular region from a chunk cache. Handles multi-chunk spans."""
    cy0, cx0 = y0 // chunk_size, x0 // chunk_size
    cy1, cx1 = (y1 - 1) // chunk_size, (x1 - 1) // chunk_size

    # Fast path: single chunk (most common — 256×256 crop on 512×512 chunks)
    if cy0 == cy1 and cx0 == cx1:
        chunk = cache[(cy0, cx0)]
        sy = y0 - cy0 * chunk_size
        sx = x0 - cx0 * chunk_size
        return chunk[..., sy:sy + (y1 - y0), sx:sx + (x1 - x0)].copy()

    # Multi-chunk path
    h, w = y1 - y0, x1 - x0
    sample = cache[(cy0, cx0)]
    if sample.ndim == 3:
        result = np.empty((sample.shape[0], h, w), dtype=sample.dtype)
    else:
        result = np.empty((h, w), dtype=sample.dtype)

    for cy in range(cy0, cy1 + 1):
        for cx in range(cx0, cx1 + 1):
            chunk = cache[(cy, cx)]
            src_y0 = max(0, y0 - cy * chunk_size)
            src_x0 = max(0, x0 - cx * chunk_size)
            src_y1 = min(chunk.shape[-2], y1 - cy * chunk_size)
            src_x1 = min(chunk.shape[-1], x1 - cx * chunk_size)
            dst_y0 = cy * chunk_size + src_y0 - y0
            dst_x0 = cx * chunk_size + src_x0 - x0
            dst_y1 = dst_y0 + (src_y1 - src_y0)
            dst_x1 = dst_x0 + (src_x1 - src_x0)
            result[..., dst_y0:dst_y1, dst_x0:dst_x1] = chunk[..., src_y0:src_y1, src_x0:src_x1]

    return result


def _init_cached_worker(out_channels):
    """Pool initializer for bulk-read workers. Caches are inherited via fork COW."""
    _worker_state["object_measurements"] = get_core_measurements()
    _worker_state["shape_features_fcn"] = _worker_state["object_measurements"].pop("sizeshape")
    _worker_state["colocalization_measurements"] = get_correlation_measurements()
    _worker_state["out_channels"] = out_channels
    channels = [(ch, i) for i, ch in enumerate(out_channels)]
    _worker_state["channels"] = channels
    _worker_state["mask_configs"] = [
        ("cell", "cell_mask"),
        ("nucleus", "nuc_mask"),
        ("cytoplasm", "cyto_mask"),
    ]
    _worker_state["channel_pairs"] = [
        (channels[i], channels[j])
        for i in range(len(channels))
        for j in range(i + 1, len(channels))
    ]


def _process_single_cell_cached(idx):
    """Worker: extract features for one cell from cached chunks. Zero I/O."""
    img_cache = _worker_state["img_cache"]
    cell_seg_cache = _worker_state["cell_seg_cache"]
    nuc_seg_cache = _worker_state["nuc_seg_cache"]
    labels_df = _worker_state["labels_df"]
    label_int_lut = _worker_state["label_int_lut"]
    int_label_lut = _worker_state["int_label_lut"]
    chunk_size = _worker_state["chunk_size"]
    cell_masks_flag = _worker_state["cell_masks"]

    obj_meas = _worker_state["object_measurements"]
    shape_fn = _worker_state["shape_features_fcn"]
    coloc_meas = _worker_state["colocalization_measurements"]
    channels = _worker_state["channels"]
    mask_configs = _worker_state["mask_configs"]
    channel_pairs = _worker_state["channel_pairs"]

    ci = labels_df.iloc[idx]
    bbox = ast.literal_eval(ci.bbox)
    y0, x0, y1, x1 = bbox

    # Crop from in-memory cache — pure RAM access, no NFS
    data = _crop_from_cache(img_cache, chunk_size, y0, x0, y1, x1)
    cell_mask_raw = _crop_from_cache(cell_seg_cache, chunk_size, y0, x0, y1, x1)
    nuc_mask_raw = _crop_from_cache(nuc_seg_cache, chunk_size, y0, x0, y1, x1)

    # Replicate CellProfileDataset.__getitem__ mask logic
    cell_mask = np.expand_dims(cell_mask_raw, axis=0)
    nuc_mask = np.expand_dims(nuc_mask_raw, axis=0)
    sc_mask = cell_mask == ci.segmentation_id

    if cell_masks_flag:
        data = data * sc_mask
        nuc_mask = (nuc_mask * sc_mask) > 0

    cyto_mask = sc_mask & (nuc_mask == 0)

    masks_np = {
        "cell_mask": sc_mask,
        "nuc_mask": nuc_mask,
        "cyto_mask": cyto_mask,
    }
    crop_info = ci.to_dict()

    cell_features = {}
    for mk_name, mk_key in mask_configs:
        mask = masks_np[mk_key][0].astype(np.int32)
        shape_features = shape_fn(mask, np.zeros_like(mask))
        cell_features.update(
            {
                f"{mk_name}_{k}": safe_item_conversion(v, f"{mk_name}_{k}", crop_info)
                for k, v in shape_features.items()
            }
        )
        for ch_name, ch_idx in channels:
            features = single_object_features(
                data[ch_idx], mask, measurements=obj_meas,
                prefix=f"single_object_{ch_name}_{mk_name}",
            )
            cell_features.update(
                {
                    k: safe_item_conversion(v, k, crop_info)
                    for k, v in features.items()
                }
            )
        for (ch1_name, ch1_idx), (ch2_name, ch2_idx) in channel_pairs:
            coloc = colocalization_features(
                data[ch1_idx], data[ch2_idx], mask,
                measurements=coloc_meas,
                prefix=f"coloc_{ch1_name}_{ch2_name}_{mk_name}",
            )
            cell_features.update(
                {
                    k: safe_item_conversion(v, k, crop_info)
                    for k, v in coloc.items()
                }
            )

    cell_features["label_int"] = int(label_int_lut[ci.gene_name])
    cell_features["label_str"] = int_label_lut[label_int_lut[ci.gene_name]]
    cell_features["sgRNA"] = crop_info["sgRNA"]
    cell_features["experiment"] = crop_info["store_key"]
    cell_features["x_position"] = crop_info["x_pheno"]
    cell_features["y_position"] = crop_info["y_pheno"]
    cell_features["well"] = crop_info["well"] + "_" + crop_info["store_key"]
    return cell_features


def extract_cp_features_bulk_read(
    experiment_dict: dict,
    bounds: list[int],
    out_channels: list[str] = None,
    num_workers: int = None,
):
    """
    Bulk-read architecture: read all needed zarr chunks into RAM, then compute.

    Phase 1 — Bulk Read (~15-30s):
      Parse cell bounding boxes, identify unique 512×512 zarr chunks,
      read them all in parallel with 16 threads. ~30 GB into RAM.

    Phase 2 — Compute (zero I/O):
      Fork workers. They inherit the chunk caches via COW (zero copy).
      Workers crop from RAM and extract features at ~100% CPU.

    Requires enough RAM for the chunk caches (~30 GB per 55k-cell task).
    Request slurm_mem_gb >= 200 to be safe.
    """
    if out_channels is None:
        out_channels = ["Phase2D", "mCherry"]
    if num_workers is None:
        num_workers = max(1, len(os.sched_getaffinity(0)) - 2)

    t0 = time.perf_counter()

    # Load labels
    labels_df = load_labels_parallel(experiment_dict)
    gene_labels = sorted(labels_df["gene_name"].unique())
    label_int_lut = {gene: i for i, gene in enumerate(gene_labels)}
    int_label_lut = {i: gene for i, gene in enumerate(gene_labels)}

    # Open zarr stores (metadata only)
    stores = {}
    for exp_name in experiment_dict:
        stores[exp_name] = zarr.open_group(
            str(OpsPaths(exp_name).stores["phenotyping_v3"]), mode="r"
        )

    # Resolve channel indices from zarr metadata
    subset_df = labels_df.iloc[bounds[0]:bounds[1]]
    first_row = subset_df.iloc[0]
    well = first_row.well
    store_key = first_row.store_key
    attrs = stores[store_key][well].attrs.asdict()
    all_channel_names = [a["label"] for a in attrs["ome"]["omero"]["channels"]]
    channel_indices = [all_channel_names.index(c) for c in out_channels]

    # Get zarr arrays
    img_arr = stores[store_key][well]["0"]
    mask_label = getattr(first_row, "mask_label", "cell_seg")
    cell_seg_arr = stores[store_key][well]["labels"][mask_label]["0"]
    nuc_seg_arr = stores[store_key][well]["labels"]["nuclear_seg"]["0"]
    chunk_size = img_arr.chunks[-1]  # 512

    # Phase 1: Identify unique chunks from bounding boxes
    chunk_coords = set()
    for bbox_str in subset_df["bbox"]:
        y0, x0, y1, x1 = ast.literal_eval(bbox_str)
        for cy in range(y0 // chunk_size, y1 // chunk_size + 1):
            for cx in range(x0 // chunk_size, x1 // chunk_size + 1):
                chunk_coords.add((cy, cx))

    n_chunks = len(chunk_coords)
    est_gb = n_chunks * chunk_size * chunk_size * 4 * (len(channel_indices) + 2) / 1e9
    print(f"  Bulk read: {n_chunks} unique chunks, ~{est_gb:.1f} GB "
          f"({len(channel_indices)} img ch + 2 masks)")

    # Parallel bulk read
    t_read = time.perf_counter()
    img_cache = _bulk_read_chunks(img_arr, chunk_coords, channel_indices, num_threads=16)
    cell_seg_cache = _bulk_read_chunks(cell_seg_arr, chunk_coords, num_threads=16)
    nuc_seg_cache = _bulk_read_chunks(nuc_seg_arr, chunk_coords, num_threads=16)
    read_time = time.perf_counter() - t_read
    print(f"  Bulk read completed in {read_time:.1f}s")

    # Set caches in module globals BEFORE forking (workers inherit via COW)
    _worker_state["img_cache"] = img_cache
    _worker_state["cell_seg_cache"] = cell_seg_cache
    _worker_state["nuc_seg_cache"] = nuc_seg_cache
    _worker_state["labels_df"] = labels_df
    _worker_state["label_int_lut"] = label_int_lut
    _worker_state["int_label_lut"] = int_label_lut
    _worker_state["chunk_size"] = chunk_size
    _worker_state["cell_masks"] = True  # match CellProfileDataset default

    # Phase 2: Compute with dynamic load balancing (zero I/O)
    # imap_unordered with chunksize=64: workers grab batches of 64 cells.
    # Fast workers automatically get more work — no tail idle time.
    indices = list(range(bounds[0], bounds[1]))

    t_compute = time.perf_counter()
    with Pool(
        processes=num_workers,
        initializer=_init_cached_worker,
        initargs=(out_channels,),
    ) as pool:
        results = list(pool.imap_unordered(
            _process_single_cell_cached, indices, chunksize=64,
        ))
    compute_time = time.perf_counter() - t_compute

    total_time = time.perf_counter() - t0
    print(f"  Compute completed in {compute_time:.1f}s")
    print(f"  Total: {total_time:.1f}s (read {read_time:.1f}s + compute {compute_time:.1f}s)")

    return pd.DataFrame(results)
