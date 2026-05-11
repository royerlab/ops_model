"""
SLURM-based distributed CellProfiler feature extraction and processing pipeline.

This module orchestrates a complete 3-stage SLURM pipeline:
1. Array jobs: Extract raw CP features in parallel chunks
2. Concatenation job: Merge chunks into single CSV
3. AnnData conversion job: Process CSV into AnnData objects with PCA/UMAP

Pipeline Flow:
--------------
cp_features_main(config_path) submits 3 dependent jobs:

    Array Job (max 100 concurrent)
    ├─ Job 0: extract features [0:100] → cp_features_job_0.csv
    ├─ Job 1: extract features [100:200] → cp_features_job_1.csv
    └─ Job N: extract features [N*100:(N+1)*100] → cp_features_job_N.csv
            ↓ (dependency: afterany)
    Concatenation Job (1 CPU, 128GB)
    └─ Merge all chunks → cp_features.csv
            ↓ (dependency: afterok)
    AnnData Conversion Job (8 CPUs, 128GB)
    └─ Process CSV → 3 .h5ad files (cell, guide, gene level)

Configuration:
--------------
YAML config should contain:
- data_manager: Experiment specs for extraction (experiments, channels, etc.)
- chunk_size: Number of cells per array job
- output_dir: Where to save results
- wait_for_completion: Whether to block until all jobs finish
- processing: (optional) Parameters for AnnData conversion
    - normalize_features: bool (default True)
    - cell-profiler: bool (default False)
    - max_nan_features_per_cell: int (default 0)
    - use_reporter_names: bool (default False)

Core extraction functions are in cp_extraction.py.
AnnData processing functions are in evaluate_cp.py.
"""

import os
import yaml
import argparse
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import submitit

from ops_model.data import data_loader
from ops_model.data.labels import load_immunostaining_labels, SOURCE_FILENAME_TEMPLATES
from ops_model.features.cp_extraction import (
    extract_cp_features,
    extract_cp_features_parallel,
    extract_cp_features_bulk_read,
)


# Environment variables to forward from the launcher to SLURM worker jobs
_OPS_ENV_VARS = ["OPS_OUTPUT_BASE_DIR", "OPS_FAST_OUTPUT_BASE_DIR", "OPS_CONFIGS_DIR"]


def _build_slurm_setup(num_threads: int = 2) -> list[str]:
    """Build slurm_setup commands including thread config and OPS env var forwarding."""
    setup = [
        f"export OMP_NUM_THREADS={num_threads}",
        f"export MKL_NUM_THREADS={num_threads}",
        f"export OPENBLAS_NUM_THREADS={num_threads}",
        f"export NUMEXPR_NUM_THREADS={num_threads}",
        "export BLOSC_NTHREADS=1",
        "export BLOSC2_NTHREADS=1",
        "export ZARR__THREADING__MAX_WORKERS=1",
        "export ZARR__ASYNC__CONCURRENCY=1",
        "export SLURM_CPU_BIND=none",
        'export CUPY_CACHE_DIR=/tmp/cupy_cache_${SLURM_JOB_ID:-$$}',
        'export CUDA_CACHE_PATH=/tmp/cuda_cache_${SLURM_JOB_ID:-$$}',
    ]
    for var in _OPS_ENV_VARS:
        val = os.environ.get(var)
        if val is not None:
            setup.append(f"export {var}={val}")
    return setup


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


def cp_features_worker_fcn(
    experiment_dict: dict,
    bounds: list[int],
    job_id: int,
    output_dir: str,
    out_channels: list[str] = None,
    csv_source: str = None,
    filename_template: str = None,
    base_path: str = None,
    indices: list[int] = None,
    guide_col: str = "sgRNA",
):
    """
    Worker function for distributed CP feature extraction using submitit.

    Uses multiprocessing.Pool with shared initializer:
    - Labels CSV read once in parallel (ThreadPool for NFS I/O)
    - Worker processes inherit labels via fork (copy-on-write, zero copy)
    - Each worker opens its own zarr handles (~0.1s)

    Args:
        experiment_dict: Dictionary mapping experiment names to FOV lists
        bounds: [start, end] index range to process
        job_id: Job ID for naming the output file
        output_dir: Directory to save the output parquet file
        out_channels: List of channel names (default: ["Phase2D", "mCherry"])

    Returns:
        str: Path to the saved parquet file
    """
    import time

    t0 = time.perf_counter()
    resolved_indices = indices if indices is not None else list(range(bounds[0], bounds[1]))
    total_cells = len(resolved_indices)
    # 30 workers — CuPy pool capped at 1GB/worker in _init_worker
    num_workers = max(1, len(os.sched_getaffinity(0)) - 2)

    print(
        f"Job {job_id}: Processing {total_cells} cells with {num_workers} workers"
    )

    labels_df = None
    if csv_source in ("cell_painting", "four_i", "immunostaining"):
        labels_df = load_immunostaining_labels(
            experiments=experiment_dict,
            filename_template=filename_template,
            base_path=base_path,
        )

    results_df = extract_cp_features_parallel(
        experiment_dict=experiment_dict,
        indices=resolved_indices,
        out_channels=out_channels,
        num_workers=num_workers,
        labels_df=labels_df,
        guide_col=guide_col,
    )

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save to parquet (5-10x faster than CSV for numerical data)
    output_path = os.path.join(output_dir, f"cp_features_job_{job_id}.parquet")
    results_df.to_parquet(output_path, index=False)

    elapsed = time.perf_counter() - t0
    throughput = len(results_df) / elapsed if elapsed > 0 else 0
    print(
        f"Job {job_id}: Saved {len(results_df)} rows to {output_path} "
        f"in {elapsed:.1f}s ({throughput:.1f} cells/s)"
    )

    return output_path


def concatenate_results(output_dir: str, final_output_path: str):
    """
    Concatenate all chunk files from individual jobs into one final CSV.

    Reads parquet chunk files, concatenates, and writes the final output as CSV
    for backward compatibility with the AnnData conversion step.

    Args:
        output_dir: Directory containing individual job parquet files
        final_output_path: Path to save the concatenated CSV results

    Returns:
        pd.DataFrame: Concatenated results
    """
    import glob
    import os

    # Find chunk files (parquet first, fall back to CSV for backward compat)
    parquet_pattern = os.path.join(output_dir, "cp_features_job_*.parquet")
    chunk_files = sorted(glob.glob(parquet_pattern))

    if not chunk_files:
        # Fall back to CSV for backward compatibility
        csv_pattern = os.path.join(output_dir, "cp_features_job_*.csv")
        chunk_files = sorted(glob.glob(csv_pattern))
        read_fn = pd.read_csv
        fmt = "CSV"
    else:
        read_fn = pd.read_parquet
        fmt = "parquet"

    if not chunk_files:
        raise ValueError(f"No chunk files found in {output_dir}")

    print(f"Found {len(chunk_files)} {fmt} files to concatenate")

    dfs = []
    for f in tqdm(chunk_files, desc=f"Loading {fmt} files"):
        dfs.append(read_fn(f))

    final_df = pd.concat(dfs, ignore_index=True)

    # Save as CSV for backward compatibility with evaluate_cp.process()
    final_df.to_csv(final_output_path, index=False)

    print(f"Saved {len(final_df)} total rows to {final_output_path}")

    return final_df


def anndata_conversion_worker(csv_path: str, config_path: str = None):
    """
    Worker function to convert concatenated CSV to AnnData objects.

    This function:
    1. Loads the concatenated CellProfiler features CSV
    2. Processes features (normalization, filtering, etc.)
    3. Creates cell-level, guide-level, and gene-level AnnData objects
    4. Saves 3 .h5ad files to {csv_parent}/anndata_objects/

    Args:
        csv_path: Path to the concatenated cp_features.csv file
        config_path: Optional path to YAML config with processing parameters

    Returns:
        str: Path to the anndata_objects directory

    Example:
        >>> anndata_conversion_worker(
        ...     csv_path='/path/to/cp_features.csv',
        ...     config_path='/path/to/config.yml'
        ... )
        '/path/to/anndata_objects'
    """
    from ops_model.features.evaluate_cp import process

    print(f"Starting AnnData conversion for {csv_path}")
    print(f"Using config: {config_path if config_path else 'default settings'}")

    # Process the CSV (creates 3 .h5ad files)
    process(save_path=csv_path, config_path=config_path)

    # Return the output directory
    output_dir = Path(csv_path).parent / "anndata_objects"
    print(f"AnnData conversion complete. Files saved to {output_dir}")

    return str(output_dir)


def cp_features_main(
    config_path: str,
    wait_for_completion: bool | None = None,
):
    """
    Main function to orchestrate distributed CellProfiler feature extraction and processing.

    Submits 3 dependent SLURM jobs:
    1. Array jobs: Extract raw CP features in parallel chunks
    2. Concatenation job: Merge all chunks into single CSV
    3. AnnData conversion job: Process CSV into 3 AnnData objects with PCA/UMAP

    Args:
        config_path: Path to YAML configuration file

    Returns:
        If wait_for_completion=True: DataFrame with concatenated results
        If wait_for_completion=False: Tuple of (array_jobs, concat_job, anndata_job)
    """

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if wait_for_completion is not None:
        config["wait_for_completion"] = wait_for_completion

    output_csv = Path(config["output_dir"]) / f"cp_features.csv"
    output_dir = output_csv.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_source = config.get("csv_source", "standard")
    filename_template = config.get("filename_template") or SOURCE_FILENAME_TEMPLATES.get(csv_source)
    base_path = config.get("base_path")
    labels_df = None
    if csv_source in ("cell_painting", "four_i", "immunostaining"):
        labels_df = load_immunostaining_labels(
            experiments=config["data_manager"]["experiments"],
            filename_template=filename_template,
            base_path=base_path,
        )

    # Get total dataset size
    data_manager = data_loader.OpsDataManager(
        experiments=config["data_manager"]["experiments"],
        batch_size=config["data_manager"]["batch_size"],
        data_split=tuple(config["data_manager"]["data_split"]),
        out_channels=config["data_manager"]["out_channels"],
        initial_yx_patch_size=tuple(config["data_manager"]["initial_yx_patch_size"]),
        link_csv_dir=config["data_manager"].get("link_csv_dir"),
        verbose=False,
        guide_col=config.get("guide_col", "sgRNA"),
    )
    data_manager.construct_dataloaders(labels_df=labels_df, num_workers=0, dataset_type="cell_profile")
    total_size = len(data_manager.train_loader.dataset)

    # Create YAML with index ranges (e.g., 100 samples per job)
    write_index_ranges_to_yaml(
        dataset_size=total_size,
        output_path=output_dir / "index_ranges.yaml",
        chunk_size=config["chunk_size"],
    )

    # Step 2: Submit as a single SLURM array job with submitit
    cpus = config.get("slurm_cpus_per_task", 32)
    mem = config.get("slurm_mem_gb", 64)
    executor = submitit.AutoExecutor(folder=output_dir / "submitit_logs")
    executor.update_parameters(
        timeout_min=config.get("slurm_timeout_min", 120),
        slurm_partition=config.get("slurm_partition", "cpu"),
        slurm_array_parallelism=100,
        cpus_per_task=cpus,
        mem_gb=mem,
        slurm_additional_parameters={"gres": "gpu:1"},
        # num_threads=1: parallelism comes from ProcessPoolExecutor, not numpy threading.
        # Setting this to cpus would cause 30 workers × 32 OMP threads = 960 threads.
        slurm_setup=_build_slurm_setup(num_threads=1),
    )

    # Load index ranges
    with open(output_dir / "index_ranges.yaml", "r") as f:
        index_ranges = yaml.safe_load(f)

    # Prepare arguments for array job
    output_dir_chunks = output_dir / "cp_feature_chunks"
    num_jobs = len(index_ranges)

    # Extract channel names from config
    out_channels = config["data_manager"]["out_channels"]
    guide_col = config.get("guide_col", "sgRNA")

    # Submit as single array job using map_array
    array_jobs = executor.map_array(
        cp_features_worker_fcn,
        [config["data_manager"]["experiments"]] * num_jobs,
        list(index_ranges.values()),  # Different bounds for each job
        list(index_ranges.keys()),  # Job IDs: 0, 1, 2, ...
        [output_dir_chunks] * num_jobs,
        [out_channels] * num_jobs,
        [csv_source] * num_jobs,
        [filename_template] * num_jobs,
        [base_path] * num_jobs,
        [None] * num_jobs,  # indices (default)
        [guide_col] * num_jobs,
    )

    # Get the array job ID (all tasks share the same base job_id)
    array_job_id = array_jobs[0].job_id.split("_")[
        0
    ]  # Remove array task suffix if present
    print(
        f"Submitted array job {array_job_id} with {len(array_jobs)} tasks (max 100 concurrent)"
    )

    # Step 3: Submit concatenation job with dependency on array job completion
    concat_executor = submitit.AutoExecutor(folder=output_dir / "submitit_logs")
    concat_executor.update_parameters(
        timeout_min=120,
        slurm_partition=config.get("slurm_partition", "cpu"),
        cpus_per_task=1,
        mem_gb=128,
        slurm_additional_parameters={"dependency": f"afterany:{array_job_id}"},
        slurm_setup=_build_slurm_setup(num_threads=2),
    )

    concat_job = concat_executor.submit(
        concatenate_results,
        output_dir=str(output_dir_chunks),
        final_output_path=str(output_csv),
    )

    print(
        f"Submitted concatenation job {concat_job.job_id} (depends on {array_job_id})"
    )

    # Step 4: Submit AnnData conversion job with dependency on concatenation success
    anndata_executor = submitit.AutoExecutor(folder=output_dir / "submitit_logs")
    anndata_executor.update_parameters(
        timeout_min=120,  # 2 hours for processing
        slurm_partition=config.get("slurm_partition", "cpu"),
        cpus_per_task=8,
        mem_gb=128,
        slurm_additional_parameters={"dependency": f"afterok:{concat_job.job_id}"},
        slurm_setup=_build_slurm_setup(num_threads=8),
    )

    # Pass the config_path if processing section exists in config
    config_path_for_processing = config_path if "processing" in config else None

    anndata_job = anndata_executor.submit(
        anndata_conversion_worker,
        csv_path=str(output_csv),
        config_path=config_path_for_processing,
    )

    print(
        f"Submitted AnnData conversion job {anndata_job.job_id} (depends on {concat_job.job_id})"
    )

    if config["wait_for_completion"]:
        # Wait for all jobs to complete
        print("Waiting for all jobs to complete...")
        final_df = concat_job.result()
        print(f"Concatenation completed. Final dataframe shape: {final_df.shape}")

        anndata_dir = anndata_job.result()
        print(f"AnnData conversion completed. Files saved to {anndata_dir}")
        print(
            f"\nPipeline complete! Generated files:\n"
            f"  - {output_csv}\n"
            f"  - {anndata_dir}/features_processed.h5ad\n"
            f"  - {anndata_dir}/guide_bulked.h5ad\n"
            f"  - {anndata_dir}/gene_bulked.h5ad"
        )
        return final_df
    else:
        print(f"\nAll jobs submitted. Check status with 'squeue -u $USER'")
        print(
            f"Job chain: {array_job_id} -> {concat_job.job_id} -> {anndata_job.job_id}"
        )
        return array_jobs, concat_job, anndata_job


def cp_features_bulk_main(config_paths: list[str]):
    """Submit CellProfiler feature extraction pipelines for multiple configs.

    Iterates configs sequentially, submitting each pipeline with
    wait_for_completion=False (fire-and-forget). SLURM handles parallelism.

    Args:
        config_paths: List of absolute paths to YAML configuration files.
    """
    submitted = []
    failed = []

    print(f"\n{'='*50}")
    print(f"BULK CELLPROFILER SUBMISSION")
    print(f"{'='*50}")
    print(f"Total configs: {len(config_paths)}\n")

    for i, config_path in enumerate(config_paths):
        print(f"[{i + 1}/{len(config_paths)}] Submitting: {config_path}")
        try:
            result = cp_features_main(config_path, wait_for_completion=False)
            submitted.append((config_path, result))
            print(f"  ✓ Jobs submitted")
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            failed.append(config_path)
        print()

    print(f"{'='*50}")
    print(f"SUMMARY")
    print(f"{'='*50}")
    print(f"Submitted: {len(submitted)}/{len(config_paths)}")
    if failed:
        print(f"Failed ({len(failed)}):")
        for f in failed:
            print(f"  - {f}")
    print(f"\nCheck job status with: squeue -u $USER")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract CellProfiler features using distributed SLURM jobs"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--config_path",
        type=str,
        help="Path to a single YAML config file",
    )
    group.add_argument(
        "--config_list",
        type=str,
        help="Path to .txt file with one absolute config path per line",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.config_list:
        with open(args.config_list) as f:
            config_paths = [
                line.strip()
                for line in f
                if line.strip() and not line.strip().startswith("#")
            ]
        cp_features_bulk_main(config_paths)
    else:
        cp_features_main(args.config_path)
