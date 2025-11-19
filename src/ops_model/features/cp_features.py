# %%
from joblib import Parallel, delayed

from tqdm import tqdm
import numpy as np
import pandas as pd
from cp_measure.bulk import (
    get_core_measurements,
    get_correlation_measurements,
    get_multimask_measurements,
)
import torch
import time
from collections import defaultdict
from multiprocessing import Pool, Manager
from functools import partial
import warnings

from ops_model.data import data_loader
from ops_model.data.paths import OpsPaths

torch.multiprocessing.set_sharing_strategy("file_system")


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
    cp_measure requires:
    img: H, W (either int of float between -1 and 1)
    mask: H, W (int)
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
                # Fixed: use feat_name instead of name to avoid shadowing the outer loop variable
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
    cp_measure requires:
    img: H, W (either int of float between -1 and 1)
    mask: H, W (int)
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
                # Fixed: use feat_name instead of name to avoid shadowing the outer loop variable
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
    mask1 = np.squeeze(mask1).astype(np.uint16)
    mask2 = np.squeeze(mask2).astype(np.uint16)

    results = {}

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for name, fn in measurements.items():
                features = fn(mask1, mask2)
                # Fixed: use feat_name instead of name to avoid shadowing the outer loop variable
                features_prefixed = {
                    f"{prefix}_{feat_name}": v for feat_name, v in features.items()
                }
                results.update(features_prefixed)
    except (ValueError, IndexError):
        # Mask is empty or invalid - generate NaN features
        return _generate_feature_names_with_nan(measurements, prefix, "neighbor")

    return results


def batched_single_object_features(
    # imgs: np.ndarray | torch.Tensor,
    # masks: np.ndarray | torch.Tensor,
    batch: dict,
    measurements: dict = None,
):
    """
    Process a batch of images and masks to extract features.

    Args:
        imgs: (B, C, H, W) batch of images
        masks: (B, H, W) or (B, C, H, W) batch of masks
        measurements: Dictionary of measurement functions from cp_measure
        prefix: Prefix for feature names
        original_sizes: List of (h, w) tuples indicating original size of each image before padding

    Returns:
        List of dictionaries, one per image in the batch (each dict contains all mask/channel combos)
    """
    # Pre-convert entire batch to numpy once (avoid redundant conversions in loops)
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

    # Pre-define channel and mask metadata
    channels = [("Phase2D", 0), ("mCherry", 1)]
    mask_configs = [
        ("cell", "cell_mask"),
        ("nucleus", "nuc_mask"),
        ("cytoplasm", "cyto_mask"),
    ]

    # Pre-compute channel pairs for colocalization (eliminates exclude list logic)
    channel_pairs = [
        (channels[i], channels[j])
        for i in range(len(channels))
        for j in range(i + 1, len(channels))
    ]

    batch_size = data_np.shape[0]
    results_list = []

    # Process each cell
    for i in range(batch_size):
        cell_features = {}

        # Process each mask type
        for mk_name, mk_key in mask_configs:
            mask = masks_np[mk_key][i]

            # Single object features for each channel
            for ch_name, ch_idx in channels:
                img = data_np[i, ch_idx]

                features = single_object_features(
                    img,
                    mask,
                    measurements=measurements["object"],
                    prefix=f"single_object_{ch_name}_{mk_name}",
                )
                cell_features.update(features)

            # Colocalization features for channel pairs
            for (ch_name_1, ch_idx_1), (ch_name_2, ch_idx_2) in channel_pairs:
                img1 = data_np[i, ch_idx_1]
                img2 = data_np[i, ch_idx_2]

                coloc_features = colocalization_features(
                    img1,
                    img2,
                    mask,
                    measurements=measurements["colocalization"],
                    prefix=f"coloc_{ch_name_1}_{ch_name_2}_{mk_name}",
                )
                cell_features.update(coloc_features)

        # Neighborhood features (uncomment and add mask_pairs definition if needed)
        # Pre-compute mask pairs outside the loop: mask_pairs = [(mask_configs[i], mask_configs[j]) for i in range(len(mask_configs)) for j in range(i+1, len(mask_configs))]
        # for (mk_name_1, mk_key_1), (mk_name_2, mk_key_2) in mask_pairs:
        #     mask1 = masks_np[mk_key_1][i]
        #     mask2 = masks_np[mk_key_2][i]
        #     neigh_features = neighbor_features(
        #         mask1, mask2,
        #         measurements=measurements['neighborhood'],
        #         prefix=f"neigh_{mk_name_1}_{mk_name_2}"
        #     )
        #     cell_features.update(neigh_features)

        # After processing all masks and channels, append the complete feature dict for this cell
        results_list.append(cell_features)

    return results_list


def save_cp_features(experiment_dict, features, override: bool = False):

    save_path = OpsPaths(list(experiment_dict.keys())[0]).cell_profiler_out
    if save_path.exists() and not override:
        print(f"file exists at {save_path}, use override=True to overwrite")

    if not save_path.exists():
        save_path.parent.mkdir(parents=True, exist_ok=True)

    features.to_csv(save_path, index=False)

    return


def _process_batch_parallel(batch, measurements, int_label_lut):
    """
    Process a single batch for parallel execution.
    This function must be at module level to be picklable.
    """
    # Extract batch metadata
    gene_labels = batch["gene_label"]  # (B,)
    sgRNA_labels = [a["sgRNA"] for a in batch["crop_info"]]
    experiment_label = [a["store_key"] for a in batch["crop_info"]]
    well_label = [a["well"] for a in batch["crop_info"]]

    # Process entire batch - extracts features for all mask/channel combinations per cell
    features_list = batched_single_object_features(batch, measurements=measurements)

    # Add gene labels to each cell's feature dictionary
    batch_results = []
    batch_size_actual = batch["data"].shape[0]
    for i in range(batch_size_actual):
        # features_list[i] already contains all features for cell i (all mask/channel combos)
        cell_features = features_list[i].copy()
        cell_features["label_int"] = gene_labels[i].item()
        cell_features["label_str"] = int_label_lut[gene_labels[i].item()]
        cell_features["sgRNA"] = sgRNA_labels[i]
        cell_features["experiment"] = experiment_label[i]
        cell_features["well"] = well_label[i] + "_" + experiment_label[i]

        batch_results.append(cell_features)

    df = pd.DataFrame(batch_results)

    return df


def cp_features(
    experiment_dict: dict = None,
    verbose: bool = False,
    num_workers: int = 4,
    num_processing_workers: int = None,
    batch_size: int = 32,
    profile: bool = False,
    max_batches: int = 20000,
    max_queue_size: int = 200,
):
    """
    Extract CellProfiler features from a batch of images.

    Args:
        experiment_dict: Dictionary of experiments and wells to process
        verbose: Print progress information
        num_workers: Number of worker processes for data loading (I/O bound)
        num_processing_workers: Number of worker processes for feature extraction (CPU bound).
                                If None, uses all available CPUs.
        batch_size: Number of cells to process in each batch
        profile: If True, print detailed timing information
        max_batches: Maximum number of batches to process
        max_queue_size: Maximum number of batches to queue for processing (default 200)

    CAUTION: should only be run for a single experiment at a time
    """

    data_manager = data_loader.OpsDataManager(
        experiments=experiment_dict,
        batch_size=batch_size,
        data_split=(1, 0, 0),
        out_channels=["Phase2D", "mCherry"],
        initial_yx_patch_size=(256, 256),
        verbose=verbose,
    )

    data_manager.construct_dataloaders(
        num_workers=num_workers, dataset_type="cell_profile"
    )
    train_loader = data_manager.train_loader

    object_measurements = get_core_measurements()
    colocalization_measurements = get_correlation_measurements()
    neighborhood_measurements = get_multimask_measurements()
    measurements = {
        "object": object_measurements,
        "colocalization": colocalization_measurements,
        "neighborhood": neighborhood_measurements,
    }
    # Time the entire data loading + processing loop
    total_start = time.time()

    # Create iterator with limited batches
    def batch_generator():
        iterator = train_loader
        for batch_idx, batch in enumerate(iterator):
            if batch_idx >= max_batches:
                break
            yield batch

    if verbose:
        print(f"Processing up to {max_batches} batches in parallel...")

    # Use multiprocessing Pool to process batches in parallel
    # imap will pull batches from the generator as workers become available
    import multiprocessing

    # Determine number of processing workers
    if num_processing_workers is None:
        n_cpus = multiprocessing.cpu_count()
    else:
        n_cpus = num_processing_workers

    if verbose:
        print(f"DataLoader workers: {num_workers}")
        print(f"Feature extraction workers: {n_cpus}")

    process_start = time.time()

    # Manual queue management to limit in-flight batches
    from collections import deque

    with Pool(processes=n_cpus) as pool:
        # Create partial function with required parameters
        process_func = partial(
            _process_batch_parallel,
            measurements=measurements,
            int_label_lut=data_manager.int_label_lut,
        )

        # Initialize
        batch_gen = batch_generator()
        pending_results = deque()
        all_features = []
        batches_submitted = 0

        if verbose:
            pbar = tqdm(total=max_batches, desc="Processing batches")

        # Pre-fill with small initial buffer to minimize startup hang
        # Queue will naturally fill to max_queue_size during processing
        initial_buffer = min(10, max_queue_size, max_batches)
        for _ in range(initial_buffer):
            try:
                batch = next(batch_gen)
                async_result = pool.apply_async(process_func, (batch,))
                pending_results.append(async_result)
                batches_submitted += 1
            except StopIteration:
                break

        # Process results and maintain bounded queue
        while pending_results:
            # Fill queue up to max_queue_size before checking results
            while len(pending_results) < max_queue_size:
                try:
                    batch = next(batch_gen)
                    new_async_result = pool.apply_async(process_func, (batch,))
                    pending_results.append(new_async_result)
                    batches_submitted += 1
                except StopIteration:
                    break

            # Check for ready results (non-blocking)
            for _ in range(len(pending_results)):
                async_result = pending_results[0]
                if async_result.ready():
                    # Result is ready, collect it
                    async_result = pending_results.popleft()
                    df_batch = async_result.get()
                    all_features.append(df_batch)

                    if verbose:
                        pbar.update(1)

                    break  # Start over to fill queue and check from the beginning
                else:
                    # Not ready, rotate to check next one
                    pending_results.rotate(-1)
            else:
                # No results ready, wait a bit
                import time as time_module

                time_module.sleep(0.01)

        if verbose:
            pbar.close()

    process_end = time.time()
    total_end = time.time()

    all_features_df = pd.concat(all_features, ignore_index=True)
    if verbose:
        print(f"\n{len(all_features_df)} cells measured")

    # Print profiling results
    if profile:
        print("\n" + "=" * 60)
        print("PROFILING RESULTS (PARALLEL PROCESSING)")
        print("=" * 60)
        print(f"Total time: {total_end - total_start:.2f}s")
        print(f"Total batch processing time: {process_end - process_start:.2f}s")
        print(f"Total cells processed: {len(all_features_df)}")
        print(
            f"Cells per second: {len(all_features_df) / (total_end - total_start):.1f}"
        )
        print(f"\nBatch size: {batch_size}")
        print(f"Number of batches: {len(all_features)}")
        print(f"Number of CPUs used: {n_cpus}")
        print(
            f"\nAverage time per batch (wall time): {(process_end - process_start) / len(all_features):.2f}s"
        )
        print(f"\nNumber of features extracted: {all_features_df.shape[1]}")
        print("=" * 60)

    # TODO: need to handle nans in a smart way
    all_features_df = all_features_df.fillna(0)

    save_cp_features(experiment_dict, all_features_df, override=True)

    return all_features_df


def _build_arg_parser():
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract CellProfiler features from OPS experiments."
    )

    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        help="Experiment name (e.g., ops0033_20250429)",
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of worker processes for data loading (I/O bound)",
    )

    parser.add_argument(
        "--num_processing_workers",
        type=int,
        default=None,
        help="Number of worker processes for feature extraction (CPU bound). If None, uses all available CPUs.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Number of cells to process in each batch",
    )

    parser.add_argument(
        "--profile",
        action="store_true",
        help="If set, print detailed timing information",
    )

    parser.add_argument(
        "--max_batches",
        type=int,
        default=20000,
        help="Maximum number of batches to process",
    )

    parser.add_argument(
        "--max_queue_size",
        type=int,
        default=20,
        help="Maximum number of batches to queue for processing (limits memory usage)",
    )

    return parser


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    experiment_dict = {args.experiment: ["A/1/0", "A/2/0", "A/3/0"]}

    cp_features(
        experiment_dict=experiment_dict,
        verbose=True,
        num_workers=args.num_workers,
        num_processing_workers=args.num_processing_workers,
        batch_size=args.batch_size,
        profile=args.profile,
        max_batches=args.max_batches,
        max_queue_size=args.max_queue_size,
    )


if __name__ == "__main__":
    main()
