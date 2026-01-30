import pytest
import numpy as np
from cp_measure.bulk import get_correlation_measurements

from ops_model.data import data_loader
from ops_model.data.paths import OpsPaths
from ops_model.features.cp_extraction import colocalization_features


@pytest.fixture(scope="module")
def data_manager():
    """Create data manager for testing (reused across all tests in module)."""
    experiment_dict = {"ops0033_20250429": ["A/1/0", "A/2/0", "A/3/0"]}
    dm = data_loader.OpsDataManager(
        experiments=experiment_dict,
        batch_size=1,
        data_split=(1, 0, 0),
        out_channels=["Phase2D", "mCherry"],
        initial_yx_patch_size=(256, 256),
        verbose=False,
    )
    dm.construct_dataloaders(num_workers=1, dataset_type="cell_profile")
    return dm


@pytest.fixture(scope="module")
def batch(data_manager):
    """Get a single batch for testing (reused across all tests in module)."""
    train_loader = data_manager.train_loader
    return next(iter(train_loader))


@pytest.fixture(scope="module")
def colocalization_measurements():
    """Get colocalization measurement functions."""
    return get_correlation_measurements()


def test_costes_fcn(batch, colocalization_measurements):
    """Test Costes colocalization function."""
    costes_fcn = colocalization_measurements["costes"]

    out = costes_fcn(
        batch["data"][0, 0].detach().cpu().numpy(),
        batch["data"][0, 1].detach().cpu().numpy(),
        masks=batch["cell_mask"][0, 0].detach().cpu().numpy(),
    )

    # Add assertions
    assert out is not None
    assert isinstance(out, dict)
    for k, v in out.items():
        assert v is not None
        assert isinstance(v, list)
        assert len(v) == 1
        assert isinstance(v[0], (float, np.floating))

    return


def test_pearson_fcn(batch, colocalization_measurements):
    """Test Pearson colocalization function."""
    pearson_fcn = colocalization_measurements["pearson"]

    out = pearson_fcn(
        batch["data"][0, 0].detach().cpu().numpy(),
        batch["data"][0, 1].detach().cpu().numpy(),
        masks=batch["cell_mask"][0, 0].detach().cpu().numpy(),
    )

    # Add assertions
    assert out is not None
    assert isinstance(out, dict)
    for k, v in out.items():
        assert v is not None
        assert isinstance(v, list)
        assert len(v) == 1
        assert isinstance(v[0], (float, np.floating))

    return


def test_manders_fold_fcn(batch, colocalization_measurements):
    """Test Manders' colocalization function."""
    manders_fcn = colocalization_measurements["manders_fold"]

    out = manders_fcn(
        batch["data"][0, 0].detach().cpu().numpy(),
        batch["data"][0, 1].detach().cpu().numpy(),
        masks=batch["cell_mask"][0, 0].detach().cpu().numpy(),
    )

    # Add assertions
    assert out is not None
    assert isinstance(out, dict)
    for k, v in out.items():
        assert v is not None
        assert isinstance(v, list)
        assert len(v) == 1
        assert isinstance(v[0], (float, np.floating))

    return


def test_rwc_fcn(batch, colocalization_measurements):
    """Test RWC colocalization function."""
    rwc_fcn = colocalization_measurements["rwc"]

    out = rwc_fcn(
        batch["data"][0, 0].detach().cpu().numpy(),
        batch["data"][0, 1].detach().cpu().numpy(),
        masks=batch["cell_mask"][0, 0].detach().cpu().numpy(),
    )

    # Add assertions
    assert out is not None
    assert isinstance(out, dict)
    for k, v in out.items():
        assert v is not None
        assert isinstance(v, list)
        assert len(v) == 1
        assert isinstance(v[0], (float, np.floating))

    return


def test_colocalization_features(batch, colocalization_measurements):
    img1 = batch["data"][0, 0].detach().cpu().numpy()
    img2 = batch["data"][0, 1].detach().cpu().numpy()
    masks = batch["cell_mask"][0, 0].detach().cpu().numpy()
    out = colocalization_features(img1, img2, masks, colocalization_measurements)

    # Add assertions
    assert out is not None
    assert isinstance(out, dict)
    assert len(out) == 8
    for k, v in out.items():
        assert v is not None
        assert isinstance(v, list)
        assert len(v) == 1
        assert isinstance(v[0], (float, np.floating))

    return


def test_empty_mask(batch, colocalization_measurements):
    img1 = np.zeros_like(batch["data"][0, 0].detach().cpu().numpy())
    mask1 = np.zeros_like(batch["cell_mask"][0, 0].detach().cpu().numpy())
    out = colocalization_features(
        img1,
        img1,
        mask1,
        measurements=colocalization_measurements,
        prefix="empty_",
    )

    assert isinstance(out, dict)
    print(len(out))

    # assert that all are nan (returned as arrays for consistency)
    for k, v in out.items():
        assert v is not None
        assert isinstance(v, np.ndarray)
        assert len(v) == 1
        assert np.isnan(v[0])

    return
