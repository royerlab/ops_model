"""
Load labels from cell_painting_linked CSVs with channel-dependent bbox selection.

For live cell channels (Phase2D, etc.): uses bbox + segmentation_id (pheno coords)
For cell painting channels (CP1_*, CP2_*): uses cp_bbox + cp_cell_seg_id (CP coords)
"""

import numpy as np
import pandas as pd
from pathlib import Path


def load_cell_painting_labels(
    experiments: dict,
    out_channels: list[str],
    cell_painting_channels: list[str] | None = None,
) -> pd.DataFrame:
    """
    Load labels from cell_painting_linked CSVs with channel-dependent bbox selection.

    Args:
        experiments: Dict of {experiment_name: [well_list]}
        out_channels: List of channels being processed (typically single channel per job)
        cell_painting_channels: List of channel names that should use cp_bbox.
            If None, auto-detects channels starting with "CP1_" or "CP2_".

    Returns:
        labels_df ready to pass to OpsDataManager.construct_dataloaders()
    """
    from ops_model.data.qc.qc_labels import filter_small_bboxes

    # Auto-detect cell painting channels if not specified
    if cell_painting_channels is None:
        cell_painting_channels = []

    current_channel = out_channels[0]
    use_cp_bbox = (
        current_channel in cell_painting_channels
        or current_channel.startswith("CP1_")
        or current_channel.startswith("CP2_")
    )

    print(f"Cell painting loader: channel={current_channel}, use_cp_bbox={use_cp_bbox}")

    labels = []
    for exp_name, wells in experiments.items():
        # Cell painting CSVs are on fast_ops
        base = Path(f"/hpc/projects/intracellular_dashboard/fast_ops/{exp_name}/3-assembly")

        for w in wells:
            well_safe = w.replace("/", "_")
            csv_path = base / f"cell_painting_linked_{well_safe}.csv"

            if not csv_path.exists():
                print(f"WARNING: {csv_path} not found, skipping")
                continue

            df = pd.read_csv(csv_path, low_memory=False)
            print(f"  Loaded {csv_path.name}: {len(df)} cells")

            # Add missing columns expected by data_loader
            df["well"] = w
            df["store_key"] = exp_name

            # Swap bbox columns based on channel type
            if use_cp_bbox:
                # Cell painting channels: use CP bbox and segmentation
                df["bbox"] = df["cp_bbox"]
                df["segmentation_id"] = df["cp_cell_seg_id"]
                # Store CP centroids as x_pheno/y_pheno for feature output compatibility
                df["x_pheno"] = df["x_cp1"]
                df["y_pheno"] = df["y_cp1"]
            else:
                # Live cell channels: keep original bbox/segmentation_id
                # Use pheno centroids
                df["x_pheno"] = df["x_pheno_centroid"]
                df["y_pheno"] = df["y_pheno_centroid"]

            # Drop rows missing required fields
            df = df.dropna(subset=["bbox", "segmentation_id"])

            # Filter small bboxes
            df, num_rem = filter_small_bboxes(df, threshold=5)
            if num_rem > 0:
                print(f"  Removed {num_rem} cells with small bboxes")

            labels.append(df)

    labels_df = pd.concat(labels, ignore_index=True)

    # Ensure gene_name column exists
    if "gene_name" in labels_df.columns:
        labels_df["gene_name"] = labels_df["gene_name"].fillna("NTC")
    elif "Gene name" in labels_df.columns:
        labels_df["gene_name"] = labels_df["Gene name"].fillna("NTC")
    else:
        raise ValueError("No gene name column found in cell painting CSV")

    labels_df["total_index"] = np.arange(len(labels_df))

    print(f"Total cells for {current_channel}: {len(labels_df)}")
    return labels_df
