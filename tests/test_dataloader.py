import numpy as np
import pytest
import torch
from iohub import open_ome_zarr
from ops_model.data import data_loader


def create_data_manager():
    experiment = "ops0033_20250429"

    store = open_ome_zarr(
        "/hpc/projects/intracellular_dashboard/ops/ops0033_20250429/3-assembly/phenotyping_og.zarr"
    )
    num_channels = len(store.channel_names)

    data_manager = data_loader.OpsDataManager(
        experiments={experiment: ["A/1/0", "A/2/0", "A/3/0"]},
        batch_size=2,
        out_channels=["Phase"],
    )
    data_manager.construct_dataloaders(num_workers=1, dataset_type="basic")

    l = data_manager

    batch = next(iter(l.train_loader))

    return l, batch


@pytest.fixture(scope="module")
def data_manager():
    return create_data_manager()


def test_ops_dataloader(data_manager):

    # for consistancy:
    patch_size = 128

    _, batch = data_manager

    # Batch is a dictionary containing [data, mask, ...] tensors
    assert isinstance(batch, dict)
    assert "data" in batch
    # assert "label" in batch
    assert "mask" in batch

    # Check that data and mask are tensors
    assert isinstance(batch["data"], torch.Tensor)
    assert isinstance(batch["mask"], torch.Tensor)

    # data should have 3 channels and be (128, 128)
    assert batch["data"].shape[2:] == (patch_size, patch_size)

    # mask should have no channel dimension and be (1, 128, 128)
    # Need the channel dimension for MONAI transforms
    assert batch["mask"].shape[1:] == (1, patch_size, patch_size)

    return


def test_data_loader_consistancy(data_manager):
    dm, batch = data_manager

    new_data_manager, _ = create_data_manager()

    batch_labels = batch["gene_label"].detach().cpu().numpy()
    total_indxs = batch["total_index"].detach().cpu().numpy()

    gene_names = dm.labels_df.iloc[total_indxs].gene_name.to_list()
    mapped_labels = np.asarray([new_data_manager.label_int_lut[a] for a in gene_names])

    assert np.all(batch_labels == mapped_labels)

    return


# def test_triplet_data_loader():
#     experiment = "ops0033_20250429"

#     dataset = OpsDataset(experiment)
#     store = open_ome_zarr(dataset.store_paths["pheno_assembled"])
#     num_channels = len(store.channel_names)

#     data_manager = data_loader.OpsDataManager(
#         experiments={experiment: ["A/1/0", "A/2/0", "A/3/0"]},
#         batch_size=2,
#         out_channels=["Phase"],
#         initial_yx_patch_size=(1, 256, 256),
#         final_yx_patch_size=(1, 256, 256),
#     )
#     data_manager.construct_dataloaders(num_workers=1, dataset_type="triplet")

#     l = data_manager
#     batch = next(iter(l.train_loader))

#     return
