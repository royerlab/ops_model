import pytest
import torch
import zarr
from ops_model.data import data_loader

import warnings

warnings.filterwarnings("ignore", category=zarr.errors.ZarrUserWarning)


@pytest.fixture(scope="module")
def feature_data_manager():
    """Create data manager for testing (reused across all tests in module)."""
    experiment_dict = {"ops0031_20250424": ["A/1/0", "A/2/0", "A/3/0"]}
    dm = data_loader.OpsDataManager(
        experiments=experiment_dict,
        batch_size=2,
        data_split=(1, 0, 0),
        out_channels=["Phase2D", "mCherry"],
        initial_yx_patch_size=(256, 256),
        verbose=False,
    )
    dm.construct_dataloaders(num_workers=1, dataset_type="cell_profile")
    return dm


@pytest.fixture(scope="module")
def basic_data_manager():
    """Create data manager for testing (reused across all tests in module)."""
    experiment_dict = {"ops0031_20250424": ["A/1/0", "A/2/0", "A/3/0"]}
    dm = data_loader.OpsDataManager(
        experiments=experiment_dict,
        batch_size=2,
        data_split=(1, 0, 0),
        out_channels=["Phase2D", "mCherry"],
        initial_yx_patch_size=(256, 256),
        verbose=False,
    )
    dm.construct_dataloaders(num_workers=1, dataset_type="basic")
    return dm


@pytest.fixture(scope="module")
def feature_batch(feature_data_manager):
    """Get a single batch for testing (reused across all tests in module)."""
    train_loader = feature_data_manager.train_loader
    return next(iter(train_loader))


@pytest.fixture(scope="module")
def basic_batch(basic_data_manager):
    """Get a single batch for testing (reused across all tests in module)."""
    train_loader = basic_data_manager.train_loader
    return next(iter(train_loader))


def test_batch_keys_cellprofiler(feature_batch):
    expected_keys = {
        "data": torch.Tensor,
        "cell_mask": torch.Tensor,
        "nuc_mask": torch.Tensor,
        "cyto_mask": torch.Tensor,
        "gene_label": torch.Tensor,
        "marker_label": list,
        "total_index": torch.Tensor,
        "original_sizes": list,
        "crop_info": list,
    }

    batch_keys = list(feature_batch.keys())
    for k, v in expected_keys.items():
        assert k in batch_keys

        assert isinstance(feature_batch[k], v)
    return


def test_batch_keys_basic(basic_batch):
    expexted_keys = {
        "data": torch.Tensor,
        "mask": torch.Tensor,
        "gene_label": torch.Tensor,
        "marker_label": list,
        "total_index": torch.Tensor,
        "crop_info": list,
    }
    batch_keys = list(basic_batch.keys())
    for k, v in expexted_keys.items():
        assert k in batch_keys

        assert isinstance(basic_batch[k], v)
    return


# Test that the data returned is normalized
def test_data_normalization(basic_batch):
    data = basic_batch["data"]
    # compute mean over all but batch and channel dimensions
    mean = torch.mean(data, dim=(0, 2, 3))

    # assert that mean is approximately 0
    assert torch.allclose(mean, torch.zeros_like(mean), atol=1)

    return


# test that requesting different out channels works
def test_out_channels(basic_data_manager, basic_batch):

    shape = basic_batch["data"].shape
    assert shape[1] == 2  # 2 out channels requested

    basic_data_manager.out_channels = ["Phase2D"]
    basic_data_manager.construct_dataloaders(num_workers=1, dataset_type="basic")
    batch = next(iter(basic_data_manager.train_loader))
    shape_1 = batch["data"].shape
    assert shape_1[1] == 1  # 1 out channel requested

    basic_data_manager.out_channels = ["mCherry"]
    basic_data_manager.construct_dataloaders(num_workers=1, dataset_type="basic")
    batch = next(iter(basic_data_manager.train_loader))
    shape_2 = batch["data"].shape
    assert shape_2[1] == 1  # 1 out channel requested

    return


# Test that turning masking on/off works
def test_cell_masking(feature_data_manager, feature_batch):

    data = feature_batch["data"]
    cell_mask = feature_batch["cell_mask"]
    # assert that where cell_mask is 0, data is also 0
    masked_data = data * (cell_mask == 0)
    assert torch.sum(masked_data) == 0

    experiment_dict = {"ops0031_20250424": ["A/1/0", "A/2/0", "A/3/0"]}
    dm = data_loader.OpsDataManager(
        experiments=experiment_dict,
        batch_size=2,
        data_split=(1, 0, 0),
        out_channels=["Phase2D", "mCherry"],
        initial_yx_patch_size=(256, 256),
        verbose=False,
    )
    dm.construct_dataloaders(
        num_workers=1, dataset_type="cell_profile", cp_kwargs={"cell_masks": False}
    )
    batch = next(iter(dm.train_loader))
    data_no_mask = batch["data"]
    # assert that data_no_mask is not equal to data everywhere
    assert not torch.equal(data, data_no_mask)

    return


# Test that different patch sizes work
def test_patch_size(basic_data_manager, basic_batch):

    shape = basic_batch["data"].shape
    assert shape[2] == 128  # initial patch size
    assert shape[3] == 128

    basic_data_manager.final_yx_patch_size = (256, 256)
    basic_data_manager.construct_dataloaders(num_workers=1, dataset_type="basic")
    batch = next(iter(basic_data_manager.train_loader))
    shape_1 = batch["data"].shape
    assert shape_1[2] == 256  # changed patch size
    assert shape_1[3] == 256

    basic_data_manager.final_yx_patch_size = (64, 64)
    basic_data_manager.construct_dataloaders(num_workers=1, dataset_type="basic")
    batch = next(iter(basic_data_manager.train_loader))
    shape_2 = batch["data"].shape
    assert shape_2[2] == 64  # changed patch size again
    assert shape_2[3] == 64

    return


def test_balanced_sampling(basic_data_manager):

    gene_names = ["NTC", "KIF23"]
    all_labels_df = basic_data_manager.get_labels()
    gene_cells = all_labels_df[all_labels_df["gene_name"].isin(gene_names)]
    basic_data_manager.construct_dataloaders(
        labels_df=gene_cells,
        num_workers=1,
        dataset_type="basic",
        balanced_sampling=True,
    )
    batch = next(iter(basic_data_manager.train_loader))
    labels = batch["gene_label"]
    unique, counts = torch.unique(labels, return_counts=True)
    assert counts[0] == counts[1]  # balanced sampling should give equal counts
    assert len(unique) == 2  # both classes should be present

    return


def test_single_channel_data():
    experiment_dict = {"ops0031_20250424": ["A/1/0", "A/2/0", "A/3/0"]}
    dm = data_loader.OpsDataManager(
        experiments=experiment_dict,
        batch_size=2,
        data_split=(1, 0, 0),
        out_channels=["Phase2D"],
        initial_yx_patch_size=(256, 256),
        verbose=False,
    )
    dm.construct_dataloaders(num_workers=1, dataset_type="basic")
    train_loader = dm.train_loader
    batch = next(iter(train_loader))
    data = batch["data"]
    shape = data.shape
    assert shape[1] == 1  # single channel data should have channel dimension of

    return


def test_contrastive_dataset():
    experiment_dict = {"ops0031_20250424": ["A/1/0", "A/2/0", "A/3/0"]}
    dm = data_loader.OpsDataManager(
        experiments=experiment_dict,
        batch_size=2,
        data_split=(1, 0, 0),
        out_channels=["Phase2D"],
        initial_yx_patch_size=(256, 256),
        final_yx_patch_size=(128, 128),
        verbose=False,
    )
    dm.construct_dataloaders(
        num_workers=1,
        dataset_type="contrastive",
        contrastive_kwargs={"positive_source": "self"},
    )
    train_loader = dm.train_loader
    batch = next(iter(train_loader))

    expected_keys = {
        "anchor": torch.Tensor,
        "positive": torch.Tensor,
        "gene_label": list,
        "marker_label": list,
        "crop_info": list,
    }

    batch_keys = list(batch.keys())
    for k, v in expected_keys.items():
        assert k in batch_keys

        assert isinstance(batch[k], v)

    return


def test_contrastive_dataset_negative():
    experiment_dict = {"ops0031_20250424": ["A/1/0", "A/2/0", "A/3/0"]}
    dm = data_loader.OpsDataManager(
        experiments=experiment_dict,
        batch_size=2,
        data_split=(1, 0, 0),
        out_channels=["Phase2D"],
        initial_yx_patch_size=(256, 256),
        final_yx_patch_size=(128, 128),
        verbose=False,
    )
    dm.construct_dataloaders(
        num_workers=1,
        dataset_type="contrastive",
        contrastive_kwargs={
            "positive_source": "self",
            "use_negative": True,
        },
    )
    train_loader = dm.train_loader
    batch = next(iter(train_loader))

    expected_keys = {
        "negative": torch.Tensor,
    }

    batch_keys = list(batch.keys())
    for k, v in expected_keys.items():
        assert k in batch_keys

        assert isinstance(batch[k], v)

    return
