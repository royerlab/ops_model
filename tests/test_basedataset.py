import pytest
import torch
import zarr
import numpy as np
from ops_model.data.base_dataset import BaseDataset
from ops_model.data import data_loader

import warnings

warnings.filterwarnings("ignore", category=zarr.errors.ZarrUserWarning)


@pytest.fixture(scope="module")
def dataset_args():
    experiment_dict = {"ops0031_20250424": ["A/1/0", "A/2/0", "A/3/0"]}
    dm = data_loader.OpsDataManager(
        experiments=experiment_dict,
        batch_size=2,
        data_split=(1, 0, 0),
        out_channels=["Phase2D", "mCherry"],
        initial_yx_patch_size=(256, 256),
        verbose=False,
    )

    labels_df = dm.get_labels()
    stores = dm.combine_stores()

    return stores, labels_df


def test_base_dataset_defaults(dataset_args):
    stores, labels_df = dataset_args
    # leave all optional args as defaults
    dataset = BaseDataset(
        stores=stores,
        labels_df=labels_df,
        out_channels=["Phase2D"],
    )
    basic_item = dataset[0]
    expexted_keys = {
        "data": torch.Tensor,
        "mask": torch.Tensor,
        "gene_label": int,
        "marker_label": list,
        "total_index": int,
        "crop_info": dict,
    }
    batch_keys = list(basic_item.keys())
    for k, v in expexted_keys.items():
        assert k in batch_keys
        print(f"Testing key: {k}")
        assert isinstance(basic_item[k], v)

    return


def test_base_dataset_out_channels(dataset_args):
    stores, labels_df = dataset_args
    # test with multiple out channels
    dataset_a = BaseDataset(
        stores=stores,
        labels_df=labels_df,
        out_channels=["Phase2D", "mCherry"],
    )
    basic_item = dataset_a[0]
    expected_marker_labels = ["Phase2D", "mCherry"]
    assert basic_item["marker_label"] == expected_marker_labels
    shape_a = basic_item["data"].shape
    assert shape_a[0] == 2  # 2 out channels are expected

    dataset_b = BaseDataset(
        stores=stores,
        labels_df=labels_df,
        out_channels=["Phase2D"],
    )
    basic_item = dataset_b[0]
    expected_marker_labels = ["Phase2D"]
    assert basic_item["marker_label"] == expected_marker_labels
    shape_b = basic_item["data"].shape
    assert shape_b[0] == 1  # 1 out channel is expected

    return


def test_base_dataset_mask_cell(dataset_args):
    stores, labels_df = dataset_args
    # test with mask_cell True
    dataset_no_mask = BaseDataset(
        stores=stores,
        labels_df=labels_df,
        out_channels=["Phase2D"],
        mask_cell=False,
        use_original_crop_size=True,
    )
    basic_item_no_mask = dataset_no_mask[0]

    a, b = np.where(basic_item_no_mask["data"][0] == 0)
    assert len(a) == 0  # there should be no zeros in data when mask_cell is False

    dataset_mask = BaseDataset(
        stores=stores,
        labels_df=labels_df,
        out_channels=["Phase2D"],
        mask_cell=True,
        use_original_crop_size=True,
    )
    basic_item_mask = dataset_mask[0]

    assert not np.array_equal(basic_item_no_mask["data"], basic_item_mask["data"])

    return


def test_base_dataset_original_shape(dataset_args):
    stores, labels_df = dataset_args
    # test with multiple out channels
    dataset_a = BaseDataset(
        stores=stores,
        labels_df=labels_df,
        out_channels=["Phase2D", "mCherry"],
        use_original_crop_size=False,
    )
    basic_item_a = dataset_a[0]

    dataset_b = BaseDataset(
        stores=stores,
        labels_df=labels_df,
        out_channels=["Phase2D", "mCherry"],
        use_original_crop_size=True,
    )
    basic_item_b = dataset_b[0]

    assert basic_item_a["data"].shape != basic_item_b["data"].shape

    return
