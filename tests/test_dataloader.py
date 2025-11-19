import pytest
import torch
from ops_model.data import data_loader


@pytest.fixture(scope="module")
def data_manager():
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
def batch(data_manager):
    """Get a single batch for testing (reused across all tests in module)."""
    train_loader = data_manager.train_loader
    return next(iter(train_loader))


def test_batch_keys_cellprofiler(batch):
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

    batch_keys = list(batch.keys())
    for k, v in expected_keys.items():
        assert k in batch_keys

        assert isinstance(batch[k], v)

    return


# def test_batch_keys_basic(batch):

#     return


# def test_data_loader_consistancy(data_manager):
#     dm, batch = data_manager

#     new_data_manager, _ = create_data_manager()

#     batch_labels = batch["gene_label"].detach().cpu().numpy()
#     total_indxs = batch["total_index"].detach().cpu().numpy()

#     gene_names = dm.labels_df.iloc[total_indxs].gene_name.to_list()
#     mapped_labels = np.asarray([new_data_manager.label_int_lut[a] for a in gene_names])

#     assert np.all(batch_labels == mapped_labels)

#     return
