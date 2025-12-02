import pytest
import torch
from ops_model.data import data_loader


@pytest.fixture(scope="module")
def feature_data_manager():
    """Create data manager for testing (reused across all tests in module)."""
    experiment_dict = {"ops0033_20250429": ["A/1/0", "A/2/0", "A/3/0"]}
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
    experiment_dict = {"ops0033_20250429": ["A/1/0", "A/2/0", "A/3/0"]}
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
    expected_keys = [
        "data",
        "cell_mask",
        "nuc_mask",
        "cyto_mask",
        "gene_label",
        "marker_label",
        "total_index",
        "original_sizes",
        "crop_info",
    ]

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


# Test that the data returned is normalized
def test_data_normalization(basic_batch):
    data = basic_batch["data"]
    # compute mean over all but batch and channel dimensions
    mean = torch.mean(data, dim=(0, 2, 3))

    # assert that mean is approximately 0
    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-1)

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

    feature_data_manager.train_loader.dataset.use_cell_mask = False
    batch = next(iter(feature_data_manager.train_loader))
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
