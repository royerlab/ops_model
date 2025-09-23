import ast
import random
from typing import Callable, List, Literal, Optional

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
    EnsureTyped,
    EnsureChannelFirstd,
)

from torch.utils.data import Dataset
from viscy.transforms import (
    RandAdjustContrastd,
    RandAffined,
    RandGaussianNoised,
    RandScaleIntensityd,
)

from .paths import OpsPaths


class BaseDataset(Dataset):

    def __init__(
        self,
        stores: dict,
        labels_df: pd.DataFrame,
        initial_yx_patch_size: tuple = (128, 128),
        final_yx_patch_size: tuple = (128, 128),
        out_channels: List[str] | Literal["random"] = "random",
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

    def _normalize_data(self, ci, channel_names, data):
        fov_attrs = self.stores[ci.store_key][
            ci.tile_pheno
        ].zattrs.asdict()  # can create dict for all tiles at beginning

        means = [
            fov_attrs["normalization"][i]["fov_statistics"]["mean"]
            for i in channel_names
        ]

        stds = [
            fov_attrs["normalization"][i]["fov_statistics"]["std"]
            for i in channel_names
        ]

        mean_divisor = np.expand_dims(np.asarray(means), (1, 2))
        std_subtractor = np.expand_dims(np.asarray(stds), (1, 2))

        data_norm = (data - mean_divisor) / std_subtractor

        return data_norm

    def _get_channels(self, ci):

        all_channel_names = self.stores[ci.store_key][
            ci.tile_pheno
        ].channel_names  # can cache

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
        fov = self.stores[ci.store_key][ci.tile_pheno][0]
        mask_fov = self.stores[ci.store_key][ci.tile_pheno]["seg"]
        bbox = ast.literal_eval(ci.bbox)
        gene_label = self.label_int_lut[ci.gene_name]
        total_index = ci.total_index

        channel_names, channel_index = self._get_channels(ci)

        data = np.asarray(
            fov[0, channel_index, 0, slice(bbox[0], bbox[2]), slice(bbox[1], bbox[3])]
        ).copy()

        mask = np.asarray(
            mask_fov[0, :, 0, slice(bbox[0], bbox[2]), slice(bbox[1], bbox[3])]
        ).copy()
        sc_mask = mask == ci.segmentation_id

        data_norm = self._normalize_data(ci, channel_names, data)

        if self.cell_masks:
            data_norm = data_norm * sc_mask

        print(data_norm.shape, sc_mask.shape)
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
        fov = self.stores[ci.store_key][ci.tile_pheno][0]
        mask_fov = self.stores[ci.store_key][ci.tile_pheno]["seg"]
        bbox = ast.literal_eval(ci.bbox)
        gene_label = self.label_int_lut[ci.gene_name]
        total_index = ci.total_index

        channel_names, channel_index = self._get_channels(ci)

        data = np.asarray(
            fov[0, channel_index, 0, slice(bbox[0], bbox[2]), slice(bbox[1], bbox[3])]
        ).copy()

        mask = np.asarray(
            mask_fov[0, :, 0, slice(bbox[0], bbox[2]), slice(bbox[1], bbox[3])]
        ).copy()
        sc_mask = mask == ci.segmentation_id

        if self.cell_masks:
            data = data * sc_mask

        batch = {
            "data": data,
            "mask": sc_mask,
            "gene_label": gene_label,
            "marker_label": channel_names,
            "total_index": total_index,
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
                if self.verbose:
                    print(f"{exp_name} {w[0]}{w[2]}: {len(labels_tmp)} cells")
                labels.append(labels_tmp)
        labels_df = pd.concat(labels, ignore_index=True)
        labels_df["gene_name"] = labels_df["gene_name"].fillna("NTC")
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
            stores[f"{exp_name}"] = open_ome_zarr(
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

        elif dataset_type == "cell_profile":
            DS = CellProfileDataset
            dataset_kwargs = {**common_kwargs, **(cp_kwargs if cp_kwargs else {})}
            self.batch_size = 1  # cell profile only supports batch size of 1 for now

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
            )

            self.test_loader = test_loader

        return
