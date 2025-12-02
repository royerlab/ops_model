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


def write_index_ranges_to_yaml(
    dataset_size: int, output_path: str, chunk_size: int = None, num_chunks: int = None
) -> dict:
    """
    Create a YAML file with dictionary keys corresponding to index ranges.

    Args:
        dataset_size: Total size of the dataset
        output_path: Path to write the YAML file
        chunk_size: Size of each chunk (mutually exclusive with num_chunks)
        num_chunks: Number of chunks to split into (mutually exclusive with chunk_size)

    Returns:
        Dictionary mapping job IDs to [start, end] index ranges

    Example:
        >>> write_index_ranges_to_yaml(1000, 'ranges.yaml', chunk_size=100)
        {0: [0, 100], 1: [100, 200], ..., 9: [900, 1000]}
    """
    if chunk_size is None and num_chunks is None:
        raise ValueError("Must provide either chunk_size or num_chunks")

    if chunk_size is not None and num_chunks is not None:
        raise ValueError("Cannot provide both chunk_size and num_chunks")

    # Calculate chunks
    if chunk_size is not None:
        num_chunks = (dataset_size + chunk_size - 1) // chunk_size  # Ceiling division
    else:
        chunk_size = dataset_size // num_chunks

    # Create index ranges
    index_ranges = {}
    for job_id in range(num_chunks):
        start_idx = job_id * chunk_size
        end_idx = min(start_idx + chunk_size, dataset_size)
        index_ranges[job_id] = [start_idx, end_idx]

    # Write to YAML file
    with open(output_path, "w") as f:
        yaml.dump(index_ranges, f, default_flow_style=False, sort_keys=False)

    print(f"Written {num_chunks} index ranges to {output_path}")
    print(f"Total indices: {dataset_size}, Chunk size: ~{chunk_size}")

    return index_ranges


def create_subset(
    experiment_dict,
    bounds: list[int],
):
    """
    A helper function to subset the total dataset given specific bounds
    Create a dataset and subset it given the bounds
    """

    from torch.utils.data import Subset

    data_manager = data_loader.OpsDataManager(
        experiments=experiment_dict,
        batch_size=1,
        data_split=(1, 0, 0),
        out_channels=["Phase2D", "mCherry"],
        initial_yx_patch_size=(256, 256),
        verbose=False,
    )

    data_manager.construct_dataloaders(num_workers=1, dataset_type="cell_profile")
    dataset = data_manager.train_loader.dataset

    return (
        Subset(dataset, list(range(bounds[0], bounds[1]))),
        data_manager.train_loader.dataset.int_label_lut,
    )


def extract_cp_features(
    experiment_dict: dict,
    bounds: list[int],
):
    """ """
    import torch
    from cp_measure.bulk import get_core_measurements, get_correlation_measurements
    from ops_model.features.cp_features import (
        single_object_features,
        colocalization_features,
    )

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

    # Pre-compute channel pairs for colocalization (eliminates exclude list logic)
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


def cp_features_worker_fcn(
    experiment_dict: dict,
    bounds: list[int],
    job_id: int,
    output_dir: str,
):
    """
    Worker function for distributed CP feature extraction using submitit.

    This function:
    1. Extracts CP features for the specified index range
    2. Saves results to a CSV file named with the job_id

    Args:
        experiment_dict: Dictionary mapping experiment names to FOV lists
        bounds: [start, end] index range to process
        job_id: Job ID for naming the output file
        output_dir: Directory to save the output CSV file

    Returns:
        str: Path to the saved CSV file

    Example:
        >>> cp_features_worker_fcn(
        ...     experiment_dict={"ops0031_20250424": ["A/1/0", "A/2/0"]},
        ...     bounds=[0, 100],
        ...     job_id=0,
        ...     output_dir='./results/'
        ... )
        './results/cp_features_job_0.csv'
    """
    import os

    print(f"Job {job_id}: Processing indices {bounds[0]} to {bounds[1]}")

    # Extract features for this subset
    results_df = extract_cp_features(
        experiment_dict=experiment_dict,
        bounds=bounds,
    )

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save to CSV
    output_path = os.path.join(output_dir, f"cp_features_job_{job_id}.csv")
    results_df.to_csv(output_path, index=False)

    print(f"Job {job_id}: Saved {len(results_df)} rows to {output_path}")

    return output_path


def concatenate_results(output_dir: str, final_output_path: str):
    """
    Concatenate all CSV files from individual jobs into one final file.

    Call this after all array jobs have completed.

    Args:
        output_dir: Directory containing individual job CSV files
        final_output_path: Path to save the concatenated results

    Returns:
        pd.DataFrame: Concatenated results

    Example:
        >>> concatenate_results(
        ...     output_dir='./results/',
        ...     final_output_path='./results/cp_features_all.csv'
        ... )
    """
    import glob
    import os

    # Find all job CSV files
    pattern = os.path.join(output_dir, "cp_features_job_*.csv")
    csv_files = sorted(glob.glob(pattern))

    if not csv_files:
        raise ValueError(f"No CSV files found matching {pattern}")

    print(f"Found {len(csv_files)} CSV files to concatenate")

    # Load and concatenate all dataframes
    dfs = []
    for csv_file in tqdm(csv_files, desc="Loading CSV files"):
        df = pd.read_csv(csv_file)
        dfs.append(df)

    final_df = pd.concat(dfs, ignore_index=True)

    # Save concatenated results
    final_df.to_csv(final_output_path, index=False)

    print(f"Saved {len(final_df)} total rows to {final_output_path}")

    return final_df


def cp_features_main(
    experiment: str = None,
    chunk_size: int = 100,
    wait_for_completion: bool = False,
):
    import submitit

    # Step 1: Create index ranges for your dataset
    experiment_dict = {experiment: ["A/1/0", "A/2/0", "A/3/0"]}

    output_csv = OpsPaths(experiment).cell_profiler_out

    # Get total dataset size
    data_manager = data_loader.OpsDataManager(
        experiments=experiment_dict,
        batch_size=1,
        data_split=(1, 0, 0),
        out_channels=["Phase2D", "mCherry"],
        initial_yx_patch_size=(256, 256),
        verbose=False,
    )
    data_manager.construct_dataloaders(num_workers=1, dataset_type="cell_profile")
    total_size = len(data_manager.train_loader.dataset)

    # Create YAML with index ranges (e.g., 100 samples per job)
    write_index_ranges_to_yaml(
        dataset_size=total_size,
        output_path=output_csv.parent / "index_ranges.yaml",
        chunk_size=chunk_size,  # Adjust based on your needs
    )

    # Step 2: Submit as a single SLURM array job with submitit
    executor = submitit.AutoExecutor(folder=output_csv.parent / "submitit_logs")
    executor.update_parameters(
        timeout_min=60,
        slurm_partition="cpu",  # Change to your partition name
        slurm_array_parallelism=100,  # Max 100 concurrent jobs
        cpus_per_task=2,
        mem_gb=16,
    )

    # Load index ranges
    with open(output_csv.parent / "index_ranges.yaml", "r") as f:
        index_ranges = yaml.safe_load(f)

    # Prepare arguments for array job
    output_dir = output_csv.parent / "cp_feature_chunks"
    num_jobs = len(index_ranges)

    # Submit as single array job using map_array
    array_jobs = executor.map_array(
        cp_features_worker_fcn,
        [experiment_dict] * num_jobs,  # Same experiment_dict for all jobs
        list(index_ranges.values()),  # Different bounds for each job
        list(index_ranges.keys()),  # Job IDs: 0, 1, 2, ...
        [output_dir] * num_jobs,  # Same output directory for all jobs
    )

    # Get the array job ID (all tasks share the same base job_id)
    array_job_id = array_jobs[0].job_id.split("_")[
        0
    ]  # Remove array task suffix if present
    print(
        f"Submitted array job {array_job_id} with {len(array_jobs)} tasks (max 100 concurrent)"
    )

    # Step 3: Submit concatenation job with dependency on array job completion
    concat_executor = submitit.AutoExecutor(folder=output_csv.parent / "submitit_logs")
    concat_executor.update_parameters(
        timeout_min=30,
        slurm_partition="cpu",
        cpus_per_task=1,
        mem_gb=32,
        slurm_additional_parameters={"dependency": f"afterok:{array_job_id}"},
    )

    concat_job = concat_executor.submit(
        concatenate_results,
        output_dir=str(output_csv.parent / "cp_feature_chunks"),
        final_output_path=str(output_csv),
    )

    print(
        f"Submitted concatenation job {concat_job.job_id} (depends on {array_job_id})"
    )

    if wait_for_completion:
        # Wait for concatenation job to complete
        print("Waiting for all jobs to complete...")
        final_df = concat_job.result()
        print(f"All jobs completed. Final dataframe shape: {final_df.shape}")
        return final_df
    else:
        print(f"Jobs submitted. Check status with 'squeue -u $USER'")
        return array_jobs, concat_job


if __name__ == "__main__":
    cp_features_main(
        experiment="ops0031_20250424",
        chunk_size=100,
    )
