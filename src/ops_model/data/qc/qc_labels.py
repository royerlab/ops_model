import os
from ast import literal_eval
from pathlib import Path

import pandas as pd
import numpy as np
import zarr
from iohub import open_ome_zarr


def csv_to_parquet(csv_path, parquet_path=None):
    """
    Convert a CSV file to Parquet format.

    Parameters
    ----------
    csv_path : str or Path
        Path to the input CSV file.
    parquet_path : str or Path, optional
        Path for the output Parquet file. If None, uses the same location
        with a `.parquet` extension.

    Returns
    -------
    parquet_path : Path
        Path to the written Parquet file.
    """
    csv_path = Path(csv_path)
    if parquet_path is None:
        parquet_path = csv_path.with_suffix(".parquet")
    else:
        parquet_path = Path(parquet_path)

    df = pd.read_csv(csv_path, low_memory=False)
    df.to_parquet(parquet_path, index=False)

    csv_size = os.path.getsize(csv_path) / (1024 * 1024)
    pq_size = os.path.getsize(parquet_path) / (1024 * 1024)
    print(f"CSV:     {csv_path} ({csv_size:.1f} MB)")
    print(f"Parquet: {parquet_path} ({pq_size:.1f} MB)")
    print(f"Compression ratio: {csv_size / pq_size:.1f}x")

    return parquet_path


def filter_small_bboxes(
    df: pd.DataFrame,
    threshold: int = 5,
) -> pd.DataFrame:
    """
    Filter out crops with bounding boxes smaller than threshold.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'bbox' column
    threshold : int
        Minimum size in pixels for both Y and X dimensions

    Returns
    -------
    filtered_df : pd.DataFrame
        Filtered DataFrame
    num_cells_removed : int
        Number of cells removed
    """
    def bbox_y_length(s):
        t = literal_eval(s)
        return (t[2] - t[0]) > threshold

    def bbox_x_length(s):
        t = literal_eval(s)
        return (t[3] - t[1]) > threshold

    y_pass = df["bbox"].apply(bbox_y_length)
    x_pass = df["bbox"].apply(bbox_x_length)
    length_pass = y_pass & x_pass
    filtered_df = df[length_pass]
    num_cells_removed = len(df) - len(filtered_df)

    return filtered_df, num_cells_removed


def filter_invalid_crops(
    df: pd.DataFrame,
    stores: dict,
    nan_threshold: float = 0.9,
    zero_threshold: float = 0.95,
    check_sample_size: int = None,
) -> tuple[pd.DataFrame, int]:
    """
    Filter out crops that are mostly NaN or have invalid data.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with crop information (must have: store_key, well, bbox columns)
    stores : dict
        Dictionary of opened zarr stores {store_key: zarr_group}
    nan_threshold : float
        Remove crops where fraction of NaN pixels exceeds this (0.0-1.0)
    zero_threshold : float
        Remove crops where fraction of zero pixels exceeds this (0.0-1.0)
    check_sample_size : int, optional
        If provided, only check this many random samples (for speed)

    Returns
    -------
    filtered_df : pd.DataFrame
        Filtered DataFrame
    num_cells_removed : int
        Number of cells removed due to invalid data
    """
    if check_sample_size and check_sample_size < len(df):
        # Sample for faster QC
        check_indices = np.random.choice(len(df), size=check_sample_size, replace=False)
        check_df = df.iloc[check_indices].copy()
    else:
        check_df = df.copy()

    invalid_indices = []

    print(f"Checking {len(check_df)} crops for invalid data...")
    for i, (idx, row) in enumerate(check_df.iterrows()):
        if i % 1000 == 0:
            print(f"  Progress: {i}/{len(check_df)} crops checked, {len(invalid_indices)} invalid found")

        try:
            store = stores[row.store_key]
            well = row.well
            fov = store[well]["0"]
            bbox = literal_eval(row.bbox)

            # Load crop data for all channels
            # Format: [T, C, Z, Y, X]
            crop_data = np.asarray(
                fov[
                    0:1,  # Time
                    :,    # All channels
                    0:1,  # Z-slice
                    slice(bbox[0], bbox[2]),  # Y
                    slice(bbox[1], bbox[3]),  # X
                ]
            )

            # Check for invalid data
            total_pixels = crop_data.size
            nan_count = np.isnan(crop_data).sum()
            zero_count = (crop_data == 0).sum()
            inf_count = np.isinf(crop_data).sum()

            nan_fraction = nan_count / total_pixels
            zero_fraction = zero_count / total_pixels

            if nan_fraction > nan_threshold:
                invalid_indices.append(idx)
            elif zero_fraction > zero_threshold:
                invalid_indices.append(idx)
            elif inf_count > 0:
                invalid_indices.append(idx)

        except Exception as e:
            # If we can't load the crop, mark as invalid
            print(f"  Warning: Could not load crop at index {idx}: {e}")
            invalid_indices.append(idx)

    # Apply filtering
    filtered_df = df.drop(index=invalid_indices)
    num_cells_removed = len(invalid_indices)

    return filtered_df, num_cells_removed
