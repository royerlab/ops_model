"""
Load labels from immunostaining-linked CSVs (cell painting or 4i).

Column names are auto-detected from the CSV: CP naming (cp_bbox, cp_cell_seg_id, ...)
is tried first, with 4i naming (4i_bbox, 4i_segmentation_id, ...) as fallback.
"""

import numpy as np
import pandas as pd
from pathlib import Path

_DEFAULT_BASE_PATH = "/hpc/projects/intracellular_dashboard/fast_ops"

# Backward-compatible filename templates for legacy csv_source values
SOURCE_FILENAME_TEMPLATES = {
    "cell_painting": "cell_painting_linked_{well}.csv",
    "four_i": "four_i_linked_{well}.csv",
}

_BBOX_CANDIDATES = ["cp_bbox", "4i_bbox"]
_SEG_ID_CANDIDATES = ["cp_cell_seg_id", "4i_segmentation_id"]
_X_CANDIDATES = ["x_cp1", "x_4i", "4i_x"]
_Y_CANDIDATES = ["y_cp1", "y_4i", "4i_y"]
_MASK_LABEL_BY_BBOX = {
    "cp_bbox": "cp_cell_seg",
    "4i_bbox": "4i_cell_seg",
}


def _resolve(candidates: list[str], columns: set[str], name: str) -> str:
    for col in candidates:
        if col in columns:
            return col
    raise ValueError(f"No column found for {name}: tried {candidates}")


def load_immunostaining_labels(
    experiments: dict,
    filename_template: str,
    base_path: str | None = None,
) -> pd.DataFrame:
    """
    Load labels from immunostaining-linked CSVs.

    Args:
        experiments: Dict of {experiment_name: [well_list]}
        filename_template: Filename pattern with {well} placeholder,
            e.g. "cell_painting_linked_{well}.csv" or "four_i_linked_{well}.csv"
        base_path: Base directory containing per-experiment subdirectories.
            Defaults to /hpc/projects/intracellular_dashboard/fast_ops.

    Returns:
        labels_df ready to pass to OpsDataManager.construct_dataloaders()
    """
    from ops_model.data.qc.qc_labels import filter_small_bboxes

    if base_path is None:
        base_path = _DEFAULT_BASE_PATH

    labels = []
    resolved = None

    for exp_name, wells in experiments.items():
        base = Path(base_path) / exp_name / "3-assembly"

        for w in wells:
            well_safe = w.replace("/", "_")
            well_parts = w.split("/")
            well_prefix = well_parts[0] + well_parts[1] if len(well_parts) >= 2 else well_safe
            csv_path = base / filename_template.format(well=well_safe, well_prefix=well_prefix)

            if not csv_path.exists():
                print(f"WARNING: {csv_path} not found, skipping")
                continue

            df = pd.read_csv(csv_path, low_memory=False)
            print(f"  Loaded {csv_path.name}: {len(df)} cells")

            if resolved is None:
                cols = set(df.columns)
                bbox_col = _resolve(_BBOX_CANDIDATES, cols, "bbox")
                seg_id_col = _resolve(_SEG_ID_CANDIDATES, cols, "segmentation_id")
                x_col = _resolve(_X_CANDIDATES, cols, "x_pheno")
                y_col = _resolve(_Y_CANDIDATES, cols, "y_pheno")
                mask_label = _MASK_LABEL_BY_BBOX[bbox_col]
                resolved = {
                    "bbox": bbox_col,
                    "seg_id": seg_id_col,
                    "mask": mask_label,
                    "x": x_col,
                    "y": y_col,
                }
                print(
                    f"Resolved columns: bbox={bbox_col}, seg_id={seg_id_col}, "
                    f"mask={mask_label}, x={x_col}, y={y_col}"
                )

            df["well"] = w
            df["store_key"] = exp_name
            df["bbox"] = df[resolved["bbox"]]
            df["segmentation_id"] = df[resolved["seg_id"]]
            df["mask_label"] = resolved["mask"]
            df["x_pheno"] = df[resolved["x"]]
            df["y_pheno"] = df[resolved["y"]]

            df = df.dropna(subset=["bbox", "segmentation_id"])

            df, num_rem = filter_small_bboxes(df, threshold=5)
            if num_rem > 0:
                print(f"  Removed {num_rem} cells with small bboxes")

            labels.append(df)

    labels_df = pd.concat(labels, ignore_index=True)

    if "gene_name" in labels_df.columns:
        labels_df["gene_name"] = labels_df["gene_name"].fillna("NTC")
    elif "Gene name" in labels_df.columns:
        labels_df["gene_name"] = labels_df["Gene name"].fillna("NTC")
    else:
        raise ValueError("No gene name column found in CSV")

    labels_df["total_index"] = np.arange(len(labels_df))
    print(f"Total cells: {len(labels_df)}")
    return labels_df
