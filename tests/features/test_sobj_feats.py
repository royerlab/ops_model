import pytest
import numpy as np
from cp_measure.bulk import get_core_measurements

from ops_model.data import data_loader
from ops_model.data.paths import OpsPaths
from ops_model.features.cp_features import single_object_features

core_fcns = get_core_measurements()


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
def core_measurements():
    """Get core measurement functions."""
    return get_core_measurements()


def test_radial_distribution(batch, core_measurements):
    fcn = core_measurements["radial_distribution"]
    out = fcn(
        batch["cell_mask"][0, 0].detach().cpu().numpy(),
        batch["data"][0, 0].detach().cpu().numpy(),
    )
    assert isinstance(out, dict)
    assert len(out) == 12
    for k, v in out.items():
        assert out[k] is not None
        assert isinstance(v, (np.ndarray))
        assert len(v) == 1
        assert isinstance(v[0], (float, np.floating))

    return


def test_radial_zernikes(batch, core_measurements):
    fcn = core_measurements["radial_zernikes"]
    out = fcn(
        batch["cell_mask"][0, 0].detach().cpu().numpy(),
        batch["data"][0, 0].detach().cpu().numpy(),
    )
    assert isinstance(out, dict)
    assert len(out) == 60
    assert any(v[0] != 0 for v in out.values())
    for k, v in out.items():
        assert out[k] is not None
        assert isinstance(v, (np.ndarray))
        assert len(v) == 1
        assert isinstance(v[0], (float, np.floating))

    return


def test_intensity(batch, core_measurements):
    fcn = core_measurements["intensity"]
    out = fcn(
        batch["cell_mask"][0, 0].detach().cpu().numpy(),
        batch["data"][0, 0].detach().cpu().numpy(),
    )
    assert isinstance(out, dict)
    assert len(out) == 21
    assert any(v[0] != 0 for v in out.values())
    for k, v in out.items():
        assert out[k] is not None
        assert isinstance(v, (np.ndarray))
        assert len(v) == 1
        assert isinstance(v[0], (float, np.floating))

    return


def test_sizeshape(batch, core_measurements):
    fcn = core_measurements["sizeshape"]
    out = fcn(
        batch["cell_mask"][0, 0].detach().cpu().numpy(),
        batch["data"][0, 0].detach().cpu().numpy(),
    )
    assert isinstance(out, dict)
    assert len(out) == 78
    assert any(v[0] != 0 for v in out.values())
    for k, v in out.items():
        assert out[k] is not None
        assert isinstance(v, (np.ndarray))
        assert len(v) == 1
        assert isinstance(v[0], (float, np.floating, np.number))

    return


def test_zernike(batch, core_measurements):
    fcn = core_measurements["zernike"]
    out = fcn(
        batch["cell_mask"][0, 0].detach().cpu().numpy(),
        batch["data"][0, 0].detach().cpu().numpy(),
    )
    assert isinstance(out, dict)
    assert len(out) == 30
    assert any(v[0] != 0 for v in out.values())
    for k, v in out.items():
        assert out[k] is not None
        assert isinstance(v, (np.ndarray))
        assert len(v) == 1
        assert isinstance(v[0], (float, np.floating, np.number))

    return


def test_ferret(batch, core_measurements):
    fcn = core_measurements["ferret"]
    out = fcn(
        batch["cell_mask"][0, 0].detach().cpu().numpy(),
        batch["data"][0, 0].detach().cpu().numpy(),
    )
    assert isinstance(out, dict)
    assert len(out) == 2
    for k, v in out.items():
        assert out[k] is not None
        assert isinstance(v, (np.ndarray))
        assert len(v) == 1
        assert isinstance(v[0], (float, np.floating))

    return


def test_texture(batch, core_measurements):
    fcn = core_measurements["texture"]
    out = fcn(
        batch["cell_mask"][0, 0].detach().cpu().numpy(),
        np.clip(batch["data"][0, 0].detach().cpu().numpy(), -1, 1),
    )
    assert isinstance(out, dict)
    assert len(out) == 52
    for k, v in out.items():
        assert out[k] is not None
        assert isinstance(v, (np.ndarray))
        assert len(v) == 1
        assert isinstance(v[0], (float, np.floating))

    return


def test_granularity(batch, core_measurements):
    fcn = core_measurements["granularity"]
    out = fcn(
        batch["cell_mask"][0, 0].detach().cpu().numpy(),
        np.clip(batch["data"][0, 0].detach().cpu().numpy(), -1, 1),
    )
    assert isinstance(out, dict)
    assert len(out) == 16
    for k, v in out.items():
        assert out[k] is not None
        assert isinstance(v, (np.ndarray))
        assert len(v) == 1
        assert isinstance(v[0], (float, np.floating))

    return


def test_single_object_features(batch, core_measurements):
    img1 = batch["data"][0, 0].detach().cpu().numpy()
    mask1 = batch["cell_mask"][0, 0].detach().cpu().numpy()
    out = single_object_features(
        img=img1,
        mask=mask1,
        measurements=core_measurements,
    )

    assert isinstance(out, dict)
    assert len(out) > 0

    # assert that none are nan
    for k, v in out.items():
        assert v is not None
        assert not (isinstance(v, float) and np.isnan(v))

    return


def test_empty_mask(batch, core_measurements):
    img1 = np.zeros_like(batch["data"][0, 0].detach().cpu().numpy())
    mask1 = np.zeros_like(batch["cell_mask"][0, 0].detach().cpu().numpy())
    out = single_object_features(
        img=img1,
        mask=mask1,
        measurements=core_measurements,
        prefix="empty_",
    )

    assert isinstance(out, dict)
    print(len(out))

    # assert that all are nan
    for k, v in out.items():
        assert v is not None
        assert isinstance(v, float) and np.isnan(v)

    return
