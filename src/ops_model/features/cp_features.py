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

import yaml
import argparse
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import submitit

from ops_model.data import data_loader
from ops_model.features.cp_extraction import extract_cp_features


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
        out_channels: List of channel names (default: ["Phase2D", "mCherry"])

    Returns:
        str: Path to the saved CSV file

    Example:
        >>> cp_features_worker_fcn(
        ...     experiment_dict={"ops0031_20250424": ["A/1/0", "A/2/0"]},
        ...     bounds=[0, 100],
        ...     job_id=0,
        ...     output_dir='./results/',
        ...     out_channels=["Phase2D", "mCherry"]
        ... )
        './results/cp_features_job_0.csv'
    """
    import os

    print(f"Job {job_id}: Processing indices {bounds[0]} to {bounds[1]}")

    # Extract features for this subset
    results_df = extract_cp_features(
        experiment_dict=experiment_dict,
        bounds=bounds,
        out_channels=out_channels,
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

    output_csv = Path(config["output_dir"]) / f"cp_features.csv"
    output_dir = output_csv.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get total dataset size
    data_manager = data_loader.OpsDataManager(
        experiments=config["data_manager"]["experiments"],
        batch_size=config["data_manager"]["batch_size"],
        data_split=tuple(config["data_manager"]["data_split"]),
        out_channels=config["data_manager"]["out_channels"],
        initial_yx_patch_size=tuple(config["data_manager"]["initial_yx_patch_size"]),
        verbose=False,
    )
    data_manager.construct_dataloaders(num_workers=0, dataset_type="cell_profile")
    total_size = len(data_manager.train_loader.dataset)

    # Create YAML with index ranges (e.g., 100 samples per job)
    write_index_ranges_to_yaml(
        dataset_size=total_size,
        output_path=output_dir / "index_ranges.yaml",
        chunk_size=config["chunk_size"],
    )

    # Step 2: Submit as a single SLURM array job with submitit
    executor = submitit.AutoExecutor(folder=output_dir / "submitit_logs")
    executor.update_parameters(
        timeout_min=30,
        slurm_partition="cpu",  # Change to your partition name
        slurm_array_parallelism=100,  # Max 100 concurrent jobs
        cpus_per_task=2,
        mem_gb=8,
        slurm_setup=[
            "export OMP_NUM_THREADS=2",
            "export MKL_NUM_THREADS=2",
            "export OPENBLAS_NUM_THREADS=2",
            "export NUMEXPR_NUM_THREADS=2",
            "export SLURM_CPU_BIND=none",
        ],
    )

    # Load index ranges
    with open(output_dir / "index_ranges.yaml", "r") as f:
        index_ranges = yaml.safe_load(f)

    # Prepare arguments for array job
    output_dir_chunks = output_dir / "cp_feature_chunks"
    num_jobs = len(index_ranges)

    # Extract channel names from config
    out_channels = config["data_manager"]["out_channels"]

    # Submit as single array job using map_array
    array_jobs = executor.map_array(
        cp_features_worker_fcn,
        [config["data_manager"]["experiments"]]
        * num_jobs,  # Same experiment_dict for all jobs
        list(index_ranges.values()),  # Different bounds for each job
        list(index_ranges.keys()),  # Job IDs: 0, 1, 2, ...
        [output_dir_chunks] * num_jobs,  # Same output directory for all jobs
        [out_channels] * num_jobs,  # Same channels for all jobs
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
        slurm_partition="cpu",
        cpus_per_task=1,
        mem_gb=128,
        slurm_additional_parameters={"dependency": f"afterany:{array_job_id}"},
        slurm_setup=[
            "export OMP_NUM_THREADS=2",
            "export MKL_NUM_THREADS=2",
            "export OPENBLAS_NUM_THREADS=2",
            "export NUMEXPR_NUM_THREADS=2",
            "export SLURM_CPU_BIND=none",
        ],
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
        slurm_partition="cpu",
        cpus_per_task=8,
        mem_gb=128,
        slurm_additional_parameters={"dependency": f"afterok:{concat_job.job_id}"},
        slurm_setup=[
            "export OMP_NUM_THREADS=8",
            "export MKL_NUM_THREADS=8",
            "export OPENBLAS_NUM_THREADS=8",
            "export NUMEXPR_NUM_THREADS=8",
            "export SLURM_CPU_BIND=none",
        ],
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
            f"  - {anndata_dir}/guide_bulked_umap.h5ad\n"
            f"  - {anndata_dir}/gene_bulked_umap.h5ad"
        )
        return final_df
    else:
        print(f"\nAll jobs submitted. Check status with 'squeue -u $USER'")
        print(
            f"Job chain: {array_job_id} -> {concat_job.job_id} -> {anndata_job.job_id}"
        )
        return array_jobs, concat_job, anndata_job


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract CellProfiler features using distributed SLURM jobs"
    )
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to YAML config file specifying parameters",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config_path = args.config_path
    cp_features_main(config_path)
