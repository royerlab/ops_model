"""Tests for the 4i / cell-painting label-loading pathway.

`load_immunostaining_labels` is a parallel data-loading entry point used by
cp_features, cell_dino, and dinov3 whenever ``config["csv_source"]`` is one of
``"cell_painting"``, ``"four_i"``, or ``"immunostaining"``. It builds a
labels_df from per-well immunostaining-linked CSVs and resolves both the
cell-painting (``cp_*``) and 4i (``4i_*``) column-naming conventions
automatically.

These tests exist to lock down that pathway's contract after the configurable
``guide_col`` refactor, since the path doesn't go through ``OpsDataManager.get_labels()``
or the alias mechanism that PR5 deleted.
"""

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import pandas as pd
import pytest

from ops_model.data import data_loader
from ops_model.data.labels import (
    SOURCE_FILENAME_TEMPLATES,
    load_immunostaining_labels,
)


# ---------------------------------------------------------------------------
# Helpers — build synthetic immunostaining-linked CSVs
# ---------------------------------------------------------------------------


def _bbox(y0: int, x0: int, y1: int, x1: int) -> str:
    # Stored as a string literal that ast.literal_eval can parse; the filter
    # checks (y1-y0) and (x1-x0) > 5 by default.
    return f"({y0}, {x0}, {y1}, {x1})"


def _write_cell_painting_csv(
    csv_path,
    n_rows: int = 4,
    *,
    include_sgrna: bool = True,
    include_gene_name: bool = True,
) -> pd.DataFrame:
    """Write a minimal cell-painting linked CSV at csv_path."""
    rows = []
    for i in range(n_rows):
        row = {
            "cp_bbox": _bbox(0, 0, 100, 100),
            "cp_cell_seg_id": i + 1,
            "x_cp1": float(100 + i),
            "y_cp1": float(200 + i),
        }
        if include_sgrna:
            row["sgRNA"] = f"sg_{i:03d}"
        if include_gene_name:
            row["gene_name"] = "GENE_A" if i < 2 else "NTC"
        rows.append(row)
    df = pd.DataFrame(rows)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    return df


def _write_four_i_csv(csv_path, n_rows: int = 4) -> pd.DataFrame:
    """Write a minimal 4i linked CSV at csv_path."""
    rows = []
    for i in range(n_rows):
        rows.append(
            {
                "4i_bbox": _bbox(0, 0, 100, 100),
                "4i_segmentation_id": i + 1,
                "4i_x": float(300 + i),
                "4i_y": float(400 + i),
                "sgRNA": f"4i_sg_{i:03d}",
                "gene_name": "GENE_B" if i < 2 else "NTC",
            }
        )
    df = pd.DataFrame(rows)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    return df


def _exp_layout(tmp_path, exp_name: str, well_files: dict):
    """Lay out the directory structure load_immunostaining_labels expects.

    well_files: mapping of well-string (e.g. "A/1/0") -> writer callable
                that takes a csv_path and writes a fixture CSV there.
    """
    base = tmp_path / exp_name / "3-assembly"
    base.mkdir(parents=True, exist_ok=True)
    for well, writer in well_files.items():
        well_safe = well.replace("/", "_")
        # caller decides filename via the filename_template
        writer(base, well_safe)


# ---------------------------------------------------------------------------
# Cell-painting path
# ---------------------------------------------------------------------------


def test_load_cell_painting_resolves_cp_columns(tmp_path):
    exp_name = "ops_test_cp"
    well = "A/1/0"
    well_safe = well.replace("/", "_")
    csv_path = (
        tmp_path
        / exp_name
        / "3-assembly"
        / SOURCE_FILENAME_TEMPLATES["cell_painting"].format(well=well_safe)
    )
    _write_cell_painting_csv(csv_path, n_rows=4)

    labels = load_immunostaining_labels(
        experiments={exp_name: [well]},
        filename_template=SOURCE_FILENAME_TEMPLATES["cell_painting"],
        base_path=str(tmp_path),
    )

    # The canonical columns the dataset code consumes downstream.
    for col in [
        "well",
        "store_key",
        "bbox",
        "segmentation_id",
        "mask_label",
        "x_pheno",
        "y_pheno",
        "gene_name",
        "total_index",
    ]:
        assert col in labels.columns, f"missing canonical column {col!r}"

    # Resolver should have promoted the cp_* source columns into canonical ones.
    assert labels["mask_label"].iloc[0] == "cp_cell_seg"
    assert (labels["x_pheno"].astype(float) == labels["x_cp1"].astype(float)).all()
    assert (labels["y_pheno"].astype(float) == labels["y_cp1"].astype(float)).all()
    assert (labels["bbox"] == labels["cp_bbox"]).all()
    assert (labels["segmentation_id"] == labels["cp_cell_seg_id"]).all()
    assert (labels["store_key"] == exp_name).all()
    assert (labels["well"] == well).all()
    assert len(labels) == 4


def test_load_cell_painting_preserves_sgrna_column(tmp_path):
    """sgRNA arrives unchanged so the default guide_col path works downstream."""
    exp_name = "ops_test_cp"
    well = "A/1/0"
    well_safe = well.replace("/", "_")
    csv_path = (
        tmp_path
        / exp_name
        / "3-assembly"
        / SOURCE_FILENAME_TEMPLATES["cell_painting"].format(well=well_safe)
    )
    _write_cell_painting_csv(csv_path, n_rows=3)
    labels = load_immunostaining_labels(
        experiments={exp_name: [well]},
        filename_template=SOURCE_FILENAME_TEMPLATES["cell_painting"],
        base_path=str(tmp_path),
    )
    assert "sgRNA" in labels.columns
    assert list(labels["sgRNA"]) == ["sg_000", "sg_001", "sg_002"]


# ---------------------------------------------------------------------------
# 4i path
# ---------------------------------------------------------------------------


def test_load_four_i_resolves_4i_columns(tmp_path):
    exp_name = "ops_test_4i"
    well = "B/2/0"
    well_safe = well.replace("/", "_")
    csv_path = (
        tmp_path
        / exp_name
        / "3-assembly"
        / SOURCE_FILENAME_TEMPLATES["four_i"].format(well=well_safe)
    )
    _write_four_i_csv(csv_path, n_rows=4)

    labels = load_immunostaining_labels(
        experiments={exp_name: [well]},
        filename_template=SOURCE_FILENAME_TEMPLATES["four_i"],
        base_path=str(tmp_path),
    )

    # 4i resolver picks the 4i_* candidates.
    assert labels["mask_label"].iloc[0] == "4i_cell_seg"
    assert (labels["bbox"] == labels["4i_bbox"]).all()
    assert (labels["segmentation_id"] == labels["4i_segmentation_id"]).all()
    assert (labels["x_pheno"].astype(float) == labels["4i_x"].astype(float)).all()
    assert (labels["y_pheno"].astype(float) == labels["4i_y"].astype(float)).all()


# ---------------------------------------------------------------------------
# Resolver behaviour
# ---------------------------------------------------------------------------


def test_resolver_prefers_cp_when_both_naming_present(tmp_path):
    """When a CSV happens to have both cp_* and 4i_* columns, cp_* wins
    (matches the candidate order in _BBOX_CANDIDATES)."""
    exp_name = "ops_test_mixed"
    well = "A/1/0"
    well_safe = well.replace("/", "_")
    csv_path = (
        tmp_path
        / exp_name
        / "3-assembly"
        / SOURCE_FILENAME_TEMPLATES["cell_painting"].format(well=well_safe)
    )
    df = pd.DataFrame(
        [
            {
                "cp_bbox": _bbox(0, 0, 100, 100),
                "4i_bbox": _bbox(10, 10, 90, 90),
                "cp_cell_seg_id": 1,
                "4i_segmentation_id": 999,
                "x_cp1": 1.0,
                "y_cp1": 2.0,
                "4i_x": 3.0,
                "4i_y": 4.0,
                "sgRNA": "sg_x",
                "gene_name": "GENE_X",
            }
        ]
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)

    labels = load_immunostaining_labels(
        experiments={exp_name: [well]},
        filename_template=SOURCE_FILENAME_TEMPLATES["cell_painting"],
        base_path=str(tmp_path),
    )
    # Resolver picked cp over 4i.
    assert labels["mask_label"].iloc[0] == "cp_cell_seg"
    assert labels["segmentation_id"].iloc[0] == 1  # cp_cell_seg_id, not 999
    assert labels["x_pheno"].iloc[0] == 1.0
    assert labels["y_pheno"].iloc[0] == 2.0


def test_missing_bbox_column_raises(tmp_path):
    """Neither cp_bbox nor 4i_bbox present → clear ValueError."""
    exp_name = "ops_test_nobox"
    well = "A/1/0"
    well_safe = well.replace("/", "_")
    csv_path = (
        tmp_path
        / exp_name
        / "3-assembly"
        / SOURCE_FILENAME_TEMPLATES["cell_painting"].format(well=well_safe)
    )
    # Missing the bbox column entirely.
    df = pd.DataFrame(
        [
            {
                "cp_cell_seg_id": 1,
                "x_cp1": 1.0,
                "y_cp1": 2.0,
                "sgRNA": "sg_x",
                "gene_name": "GENE_X",
            }
        ]
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match=r"No column found for bbox"):
        load_immunostaining_labels(
            experiments={exp_name: [well]},
            filename_template=SOURCE_FILENAME_TEMPLATES["cell_painting"],
            base_path=str(tmp_path),
        )


# ---------------------------------------------------------------------------
# gene_name handling
# ---------------------------------------------------------------------------


def test_gene_name_promoted_from_capitalised(tmp_path):
    """If only "Gene name" is present, it is promoted to gene_name with NaN
    filled by 'NTC'."""
    exp_name = "ops_test_capgn"
    well = "A/1/0"
    well_safe = well.replace("/", "_")
    csv_path = (
        tmp_path
        / exp_name
        / "3-assembly"
        / SOURCE_FILENAME_TEMPLATES["cell_painting"].format(well=well_safe)
    )
    df = pd.DataFrame(
        [
            {
                "cp_bbox": _bbox(0, 0, 100, 100),
                "cp_cell_seg_id": 1,
                "x_cp1": 1.0,
                "y_cp1": 2.0,
                "sgRNA": "sg_a",
                "Gene name": "EGFR",
            },
            {
                "cp_bbox": _bbox(0, 0, 100, 100),
                "cp_cell_seg_id": 2,
                "x_cp1": 3.0,
                "y_cp1": 4.0,
                "sgRNA": "sg_b",
                "Gene name": None,  # NaN should become NTC
            },
        ]
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)

    labels = load_immunostaining_labels(
        experiments={exp_name: [well]},
        filename_template=SOURCE_FILENAME_TEMPLATES["cell_painting"],
        base_path=str(tmp_path),
    )
    assert list(labels["gene_name"]) == ["EGFR", "NTC"]


def test_no_gene_column_raises(tmp_path):
    """Neither 'gene_name' nor 'Gene name' present → ValueError."""
    exp_name = "ops_test_nogene"
    well = "A/1/0"
    well_safe = well.replace("/", "_")
    csv_path = (
        tmp_path
        / exp_name
        / "3-assembly"
        / SOURCE_FILENAME_TEMPLATES["cell_painting"].format(well=well_safe)
    )
    _write_cell_painting_csv(csv_path, n_rows=2, include_gene_name=False)

    with pytest.raises(ValueError, match=r"No gene name column found"):
        load_immunostaining_labels(
            experiments={exp_name: [well]},
            filename_template=SOURCE_FILENAME_TEMPLATES["cell_painting"],
            base_path=str(tmp_path),
        )


# ---------------------------------------------------------------------------
# Multi-well aggregation
# ---------------------------------------------------------------------------


def test_multiple_wells_concatenate(tmp_path):
    exp_name = "ops_test_multi"
    wells = ["A/1/0", "A/2/0", "B/3/0"]
    for well in wells:
        well_safe = well.replace("/", "_")
        csv_path = (
            tmp_path
            / exp_name
            / "3-assembly"
            / SOURCE_FILENAME_TEMPLATES["cell_painting"].format(well=well_safe)
        )
        _write_cell_painting_csv(csv_path, n_rows=3)

    labels = load_immunostaining_labels(
        experiments={exp_name: wells},
        filename_template=SOURCE_FILENAME_TEMPLATES["cell_painting"],
        base_path=str(tmp_path),
    )
    assert len(labels) == 9
    assert sorted(labels["well"].unique()) == wells
    # total_index is monotonic across all wells.
    assert list(labels["total_index"]) == list(range(9))


def test_missing_well_file_is_skipped_with_warning(tmp_path, capsys):
    """If one well's CSV doesn't exist, it is skipped (warning printed) and the
    remaining wells still produce a labels_df."""
    exp_name = "ops_test_skip"
    # Write only one of the two wells.
    well_present = "A/1/0"
    well_missing = "A/2/0"
    csv_path = (
        tmp_path
        / exp_name
        / "3-assembly"
        / SOURCE_FILENAME_TEMPLATES["cell_painting"].format(
            well=well_present.replace("/", "_")
        )
    )
    _write_cell_painting_csv(csv_path, n_rows=2)

    labels = load_immunostaining_labels(
        experiments={exp_name: [well_present, well_missing]},
        filename_template=SOURCE_FILENAME_TEMPLATES["cell_painting"],
        base_path=str(tmp_path),
    )
    assert len(labels) == 2
    assert (labels["well"] == well_present).all()
    captured = capsys.readouterr()
    assert "not found" in captured.out


# ---------------------------------------------------------------------------
# Integration with OpsDataManager / BaseDataset (guide_col plumbing)
# ---------------------------------------------------------------------------


def test_labels_df_works_with_ops_data_manager_default_guide_col(tmp_path):
    """labels_df from load_immunostaining_labels can be passed to BaseDataset,
    and the dataset's default guide_col remains 'sgRNA' — matching the
    existing immunostaining schema."""
    exp_name = "ops_test_dm"
    well = "A/1/0"
    well_safe = well.replace("/", "_")
    csv_path = (
        tmp_path
        / exp_name
        / "3-assembly"
        / SOURCE_FILENAME_TEMPLATES["cell_painting"].format(well=well_safe)
    )
    _write_cell_painting_csv(csv_path, n_rows=3)

    labels = load_immunostaining_labels(
        experiments={exp_name: [well]},
        filename_template=SOURCE_FILENAME_TEMPLATES["cell_painting"],
        base_path=str(tmp_path),
    )

    # Construct a BaseDataset with the labels_df. Default guide_col='sgRNA'
    # matches the column that load_immunostaining_labels preserved from the CSV.
    ds = data_loader.BaseDataset(stores={}, labels_df=labels)
    assert ds.guide_col == "sgRNA"
    assert ds.guide_col in ds.labels_df.columns

    # And the OpsDataManager flow accepts an explicit guide_col override too.
    dm = data_loader.OpsDataManager(
        experiments={exp_name: [well]},
        guide_col="sgRNA",
    )
    assert dm.guide_col == "sgRNA"


def test_labels_df_with_alternate_guide_col(tmp_path):
    """A non-default guide_col (e.g. a hypothetical 'antibody_id' for an
    immunostaining-with-antibody-panel experiment) is preserved on the
    dataset, and the column survives into crop_info via labels_df.iloc[i]."""
    exp_name = "ops_test_antib"
    well = "A/1/0"
    well_safe = well.replace("/", "_")
    csv_path = (
        tmp_path
        / exp_name
        / "3-assembly"
        / SOURCE_FILENAME_TEMPLATES["cell_painting"].format(well=well_safe)
    )
    df = pd.DataFrame(
        [
            {
                "cp_bbox": _bbox(0, 0, 100, 100),
                "cp_cell_seg_id": 1,
                "x_cp1": 1.0,
                "y_cp1": 2.0,
                "antibody_id": "ab_GAPDH",
                "gene_name": "GAPDH",
            },
            {
                "cp_bbox": _bbox(0, 0, 100, 100),
                "cp_cell_seg_id": 2,
                "x_cp1": 3.0,
                "y_cp1": 4.0,
                "antibody_id": "ab_TUBB",
                "gene_name": "TUBB",
            },
        ]
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)

    labels = load_immunostaining_labels(
        experiments={exp_name: [well]},
        filename_template=SOURCE_FILENAME_TEMPLATES["cell_painting"],
        base_path=str(tmp_path),
    )
    assert "antibody_id" in labels.columns

    ds = data_loader.BaseDataset(
        stores={}, labels_df=labels, guide_col="antibody_id"
    )
    assert ds.guide_col == "antibody_id"
    # crop_info is the row.to_dict() — confirm the configured column round-trips.
    crop_info = ds.labels_df.iloc[0].to_dict()
    assert crop_info["antibody_id"] == "ab_GAPDH"
