import ast
import random
from typing import Callable, List, Literal, Optional

import zarr
import numpy as np
import pandas as pd
import torch
from iohub import open_ome_zarr
from monai.transforms import (
    CenterSpatialCropd,
    Compose,
    RandAffined,
    RandFlipd,
    RandRotate90d,
    SpatialPadd,
    ToTensord,
)

from torch.utils.data import Dataset
from viscy.transforms import (
    RandAdjustContrastd,
    RandAffined,
    RandGaussianNoised,
    RandScaleIntensityd,
)

from .paths import OpsPaths


def collate_variable_size_cells(batch):
    """
    Custom collate function for batches with variable-sized cell images.
    Pads all images and masks to the maximum size in the batch.

    Args:
        batch: List of dictionaries from dataset __getitem__

    Returns:
        Dictionary with batched tensors, all padded to max size in batch
    """
    # Find maximum height and width in this batch
    max_h = max(item["data"].shape[-2] for item in batch)
    max_w = max(item["data"].shape[-1] for item in batch)

    # Initialize lists for batched data
    data_list = []
    mask_list = []
    nuc_mask_list = []
    cyto_mask_list = []
    gene_labels = []
    marker_labels = []
    total_indices = []
    original_sizes = []  # Store original sizes for downstream use
    crop_infos = []  # Store crop info dicts

    for item in batch:
        data = item["data"]
        mask = item["cell_mask"]
        nuc_mask = item["nuc_mask"]
        cyto_mask = item["cyto_mask"]

        # Get original size
        _, h, w = data.shape if data.ndim == 3 else (1, *data.shape)
        original_sizes.append((h, w))

        # Pad data and mask to max size
        pad_h = max_h - h
        pad_w = max_w - w

        # Pad: (left, right, top, bottom) for 2D, or (c, h, w) dimensions
        if data.ndim == 3:  # (C, H, W)
            data_padded = torch.nn.functional.pad(
                torch.from_numpy(data) if isinstance(data, np.ndarray) else data,
                (0, pad_w, 0, pad_h),
                mode="constant",
                value=0,
            )
        else:  # (H, W)
            data_padded = torch.nn.functional.pad(
                torch.from_numpy(data) if isinstance(data, np.ndarray) else data,
                (0, pad_w, 0, pad_h),
                mode="constant",
                value=0,
            )

        # Convert mask to tensor and ensure it's int32 for instance segmentation
        mask_tensor = (
            torch.from_numpy(mask.astype(np.int32))
            if isinstance(mask, np.ndarray)
            else mask.to(torch.int32)
        )

        if mask.ndim == 3:  # (C, H, W)
            mask_padded = torch.nn.functional.pad(
                mask_tensor, (0, pad_w, 0, pad_h), mode="constant", value=0
            )
        else:  # (H, W)
            mask_padded = torch.nn.functional.pad(
                mask_tensor, (0, pad_w, 0, pad_h), mode="constant", value=0
            )

        # Pad nuc_mask
        nuc_mask_tensor = (
            torch.from_numpy(nuc_mask.astype(np.int32))
            if isinstance(nuc_mask, np.ndarray)
            else nuc_mask.to(torch.int32)
        )

        if nuc_mask.ndim == 3:  # (C, H, W)
            nuc_mask_padded = torch.nn.functional.pad(
                nuc_mask_tensor, (0, pad_w, 0, pad_h), mode="constant", value=0
            )
        else:  # (H, W)
            nuc_mask_padded = torch.nn.functional.pad(
                nuc_mask_tensor, (0, pad_w, 0, pad_h), mode="constant", value=0
            )
        # Pad cyto_mask
        cyto_mask_tensor = (
            torch.from_numpy(cyto_mask.astype(np.int32))
            if isinstance(cyto_mask, np.ndarray)
            else cyto_mask.to(torch.int32)
        )

        if cyto_mask.ndim == 3:  # (C, H, W)
            cyto_mask_padded = torch.nn.functional.pad(
                cyto_mask_tensor, (0, pad_w, 0, pad_h), mode="constant", value=0
            )
        else:  # (H, W)
            cyto_mask_padded = torch.nn.functional.pad(
                cyto_mask_tensor, (0, pad_w, 0, pad_h), mode="constant", value=0
            )
        data_list.append(data_padded)
        mask_list.append(mask_padded)
        nuc_mask_list.append(nuc_mask_padded)
        cyto_mask_list.append(cyto_mask_padded)

        gene_labels.append(item["gene_label"])
        marker_labels.append(item["marker_label"])
        total_indices.append(item["total_index"])
        crop_infos.append(item["crop_info"])

    # Stack into batch tensors
    batched = {
        "data": torch.stack(data_list),
        "cell_mask": torch.stack(mask_list),
        "nuc_mask": torch.stack(nuc_mask_list),
        "cyto_mask": torch.stack(cyto_mask_list),
        "gene_label": torch.tensor(gene_labels),
        "marker_label": marker_labels,  # Keep as list since these are strings
        "total_index": torch.tensor(total_indices),
        "original_sizes": original_sizes,  # Keep as list of tuples
        "crop_info": crop_infos,  # Keep as list of dicts
    }

    return batched


class BaseDataset(Dataset):

    def __init__(
        self,
        stores: dict,
        labels_df: pd.DataFrame,
        initial_yx_patch_size: tuple = (128, 128),
        final_yx_patch_size: tuple = (128, 128),
        out_channels: List[str] | Literal["random"] | Literal["all"] = "random",
        label_int_lut: dict = None,  # string --> int
        int_label_lut: dict = None,  # int --> string
        cell_masks: bool = True,
    ):
        self.stores = stores
        self.labels_df = labels_df
        self.initial_yx_patch_size = initial_yx_patch_size
        self.final_yx_patch_size = final_yx_patch_size
        self.out_channels = out_channels
        self.label_int_lut = label_int_lut
        self.int_label_lut = int_label_lut
        self.cell_masks = cell_masks

        self.transform = Compose(
            [
                SpatialPadd(
                    keys=["data", "mask"],
                    spatial_size=self.initial_yx_patch_size,
                ),
                CenterSpatialCropd(
                    keys=["data", "mask"], roi_size=(self.final_yx_patch_size)
                ),
                RandFlipd(
                    # Vertical Flip
                    keys=["data", "mask"],
                    prob=0.5,
                    spatial_axis=-2,
                ),
                RandFlipd(
                    # Horizontal Flip
                    keys=["data", "mask"],
                    prob=0.5,
                    spatial_axis=-1,
                ),
                RandRotate90d(
                    keys=["data", "mask"],
                    prob=0.5,
                    max_k=3,
                ),
                ToTensord(
                    keys=["data", "mask"],
                ),
            ]
        )
        return

    def _normalize_data(self, ci, channel_names, data, masks):

        # Temporary Fix
        # normalize per crop and squash all values between -1 and 1
        data_shift = data - np.mean(data)
        lo, hi = np.percentile(data_shift, [1, 99.5])
        scale = max(abs(lo), abs(hi))  # symmetric mapping
        data_norm = np.clip(data_shift, -scale, scale) / scale
        if self.cell_masks:
            data_norm = data_norm * masks

        # fov_attrs = self.stores[ci.store_key][
        #     ci.tile_pheno
        # ].zattrs.asdict()  # can create dict for all tiles at beginning

        # # TODO: need a real measure of dataset background
        # bg = [np.percentile(data, 1)]

        # iqrs = [
        #     fov_attrs["normalization"][i]["fov_statistics"]["iqr"]
        #     for i in channel_names
        # ]
        # means = [
        #     fov_attrs["normalization"][i]["fov_statistics"]["mean"]
        #     for i in channel_names
        # ]

        # data_bg_sub = np.clip(data - np.expand_dims(bg, (1, 2)), a_min=0, a_max=None)

        # if self.cell_masks:
        #     data_bg_sub = data_bg_sub * masks

        # data_iqr = (data_bg_sub - np.expand_dims(means, (1, 2))) / (
        #     np.expand_dims(iqrs, (1, 2)) + 1e-6
        # )

        # # TODO: Need to fix to work with multiple channels
        # lo, hi = np.percentile(data_iqr, [1, 99.5])
        # scale = max(abs(lo), abs(hi))   # symmetric mapping
        # data_norm = np.clip(data_iqr, -scale, scale) / scale

        return data_norm

    def _get_channels(self, ci, well):

        attrs = self.stores[ci.store_key][well].attrs.asdict()
        all_channel_names = [a["label"] for a in attrs["omero"]["channels"]]

        if self.out_channels == "random":
            channel_names = [random.choice(all_channel_names)]
        if self.out_channels == "all":
            channel_names = all_channel_names
        else:
            channel_names = self.out_channels
        channel_index = [all_channel_names.index(c) for c in channel_names]

        return channel_names, channel_index

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, index):
        ci = self.labels_df.iloc[index]  # crop info

        well = ci.well
        fov = self.stores[ci.store_key][well][0]
        mask_fov = self.stores[ci.store_key][well]["seg"][0]
        bbox = ast.literal_eval(ci.bbox)
        gene_label = self.label_int_lut[ci.gene_name]
        total_index = ci.total_index

        channel_names, channel_index = self._get_channels(
            ci, well
        )  # probably doesn't have to be done per dataset

        data = np.asarray(
            fov[0, channel_index, 0, slice(bbox[0], bbox[2]), slice(bbox[1], bbox[3])]
        ).copy()

        mask = np.asarray(
            mask_fov[0, :, 0, slice(bbox[0], bbox[2]), slice(bbox[1], bbox[3])]
        ).copy()
        sc_mask = mask == ci.segmentation_id

        data_norm = self._normalize_data(ci, channel_names, data, sc_mask)

        batch = {
            "data": data_norm.astype(np.float32),
            "mask": sc_mask,
            "gene_label": gene_label,
            "marker_label": channel_names,
            "total_index": total_index,
        }

        batch = self.transform(batch)

        return batch


class TripletDataset(BaseDataset):
    pass


class RandomCropDataset(BaseDataset):
    pass


class CellProfileDataset(BaseDataset):
    def __init__(self, stores: dict, labels_df: pd.DataFrame, **kwargs):
        super().__init__(stores, labels_df, **kwargs)

    def __getitem__(self, index):

        ci = self.labels_df.iloc[index]  # crop info

        well = ci.well
        fov = self.stores[ci.store_key][well][0]
        cell_mask_fov = self.stores[ci.store_key][well]["seg"][0]
        nuc_mask_fov = self.stores[ci.store_key][well]["nuclear_seg"][0]
        bbox = ast.literal_eval(ci.bbox)
        gene_label = self.label_int_lut[ci.gene_name]
        total_index = ci.total_index

        channel_names, channel_index = self._get_channels(ci, well)

        data = np.asarray(
            fov[0, channel_index, 0, slice(bbox[0], bbox[2]), slice(bbox[1], bbox[3])]
        ).copy()

        cell_mask = np.asarray(
            cell_mask_fov[0, :, 0, slice(bbox[0], bbox[2]), slice(bbox[1], bbox[3])]
        ).copy()
        nuc_mask = np.asarray(
            nuc_mask_fov[0, :, 0, slice(bbox[0], bbox[2]), slice(bbox[1], bbox[3])]
        ).copy()
        sc_mask = cell_mask == ci.segmentation_id

        # ensure that all output masks as binary
        if self.cell_masks:
            data = data * sc_mask
            nuc_mask = (nuc_mask * sc_mask) > 0

        cyto_mask = sc_mask & (nuc_mask == 0)

        batch = {
            "data": data,
            "cell_mask": sc_mask,
            "gene_label": gene_label,
            "cyto_mask": cyto_mask,
            "nuc_mask": nuc_mask,
            "marker_label": channel_names,
            "total_index": total_index,
            "crop_info": ci.to_dict(),
        }

        return batch


class OpsDataManager:
    def __init__(
        self,
        experiments: dict,
        data_split: tuple = (0.9, 0.05, 0.05),
        shuffle_seed: int = 1,
        initial_yx_patch_size: tuple = (200, 200),
        final_yx_patch_size: tuple = (128, 128),
        batch_size: int = 32,
        out_channels: List[str] | Literal["random"] = "random",
        verbose: bool = False,
    ):
        self.experiments = experiments
        self.data_split = data_split
        self.shuffle_seed = shuffle_seed
        self.num_workers = 1
        self.initial_yx_patch_size = initial_yx_patch_size
        self.final_yx_patch_size = final_yx_patch_size
        self.batch_size = batch_size
        self.out_channels = out_channels
        self.verbose = verbose
        self.collate_fcn = None

        self.store_dict = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

    def split_data(self, labels_df):
        """Splits data into train, val, and test sets"""
        np.random.seed(self.shuffle_seed)
        ind = np.random.choice(len(labels_df), size=len(labels_df), replace=False)
        split_ind = list(
            np.cumsum([int(len(labels_df) * i) for i in self.data_split[:-1]])
        )
        train_ind = ind[0 : split_ind[0]]
        val_ind = ind[split_ind[0] : split_ind[1]]
        test_ind = ind[split_ind[1] : len(labels_df)]

        return train_ind, val_ind, test_ind

    def get_labels(self):
        """ """

        labels = []
        for exp_name, wells in self.experiments.items():

            for w in wells:
                if self.verbose:
                    print("reading labels for", exp_name, f"links_{w[0]}{w[2]}")
                labels_tmp = pd.read_csv(OpsPaths(exp_name, well=f"{w[0]}{w[2]}").links)
                labels_tmp["store_key"] = exp_name
                labels_tmp["well"] = w
                if self.verbose:
                    print(f"{exp_name} {w[0]}{w[2]}: {len(labels_tmp)} cells")
                labels.append(labels_tmp)
        labels_df = pd.concat(labels, ignore_index=True)
        labels_df["gene_name"] = labels_df["Gene name"].fillna("NTC")
        labels_df["total_index"] = np.arange(len(labels_df))  # add index column
        if self.verbose:
            print(f"Total cells: {len(labels_df)}")

        return labels_df

    def gene_label_converter(self):
        """ """
        gene_labels = sorted(self.labels_df["gene_name"].unique())
        label_int_lut = {gene: i for i, gene in enumerate(gene_labels)}
        int_label_lut = {i: gene for i, gene in enumerate(gene_labels)}

        return label_int_lut, int_label_lut

    def combine_stores(self):
        """ """
        stores = {}
        for exp_name, wells in self.experiments.items():
            stores[f"{exp_name}"] = zarr.open_group(
                OpsPaths(exp_name).phenotyping, mode="r"
            )

        return stores

    def construct_dataloaders(
        self,
        num_workers: int = 1,
        shuffle: bool = True,
        dataset_type: Literal["basic", "triplet"] = "basic",
        triplet_kwargs: dict = None,
        basic_kwargs: dict = None,
        cp_kwargs: dict = None,
    ):
        """
        Returns train, val and test dataloaders
        """
        self.store_dict = self.combine_stores()
        labels_df = self.get_labels()
        self.labels_df = labels_df
        self.label_int_lut, self.int_label_lut = self.gene_label_converter()

        train_ind, val_ind, test_ind = self.split_data(labels_df)

        common_kwargs = {
            "initial_yx_patch_size": self.initial_yx_patch_size,
            "final_yx_patch_size": self.final_yx_patch_size,
            "out_channels": self.out_channels,
            "label_int_lut": self.label_int_lut,
            "int_label_lut": self.int_label_lut,
        }

        if dataset_type == "basic":
            DS = BaseDataset
            dataset_kwargs = {**common_kwargs, **(basic_kwargs if basic_kwargs else {})}
            self.collate_fcn = None  # Use default collate

        elif dataset_type == "cell_profile":
            DS = CellProfileDataset
            dataset_kwargs = {**common_kwargs, **(cp_kwargs if cp_kwargs else {})}
            self.collate_fcn = (
                collate_variable_size_cells  # Use custom collate for variable sizes
            )

        # elif dataset_type == "triplet":
        #     DS = TripletOpsDataset
        #     dataset_kwargs = {
        #         **common_kwargs,
        #         **(triplet_kwargs if triplet_kwargs else {}),
        #     }

        if len(train_ind) > 0:
            train_dataset = DS(
                stores=self.store_dict,
                labels_df=labels_df.iloc[train_ind],
                **dataset_kwargs,
            )
            train_loader = torch.utils.data.DataLoader(
                dataset=train_dataset,
                batch_size=self.batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                collate_fn=self.collate_fcn,
            )

            self.train_loader = train_loader

        if len(val_ind) > 0:
            val_dataset = DS(
                stores=self.store_dict,
                labels_df=labels_df.iloc[val_ind],
                **dataset_kwargs,
            )
            val_loader = torch.utils.data.DataLoader(
                dataset=val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=self.collate_fcn,
            )

            self.val_loader = val_loader

        if len(test_ind) > 0:
            test_dataset = DS(
                stores=self.store_dict,
                labels_df=labels_df.iloc[test_ind],
                **dataset_kwargs,
            )
            test_loader = torch.utils.data.DataLoader(
                dataset=test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=self.collate_fcn,
            )

            self.test_loader = test_loader

        return
