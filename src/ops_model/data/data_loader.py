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
)
from ops_analysis.data.experiment import OpsDataset
from torch.utils.data import Dataset
from viscy.transforms import (
    RandAdjustContrastd,
    RandAffined,
    RandGaussianNoised,
    RandScaleIntensityd,
)


class scDataSet(Dataset):

    def __init__(
        self,
        stores: dict,
        labels_df: pd.DataFrame,
        transform: Optional[Callable] = None,
        initial_yx_patch_size: tuple = (128, 128),
        final_yx_patch_size: tuple = (128, 128),
        out_channels: List[str] | Literal["random"] = "random",
        one_hot_lut: dict = None,
        cell_masks: bool = True,
    ):

        self.stores = stores
        self.labels_df = labels_df
        self.initial_yx_patch_size = initial_yx_patch_size
        self.final_yx_patch_size = final_yx_patch_size
        self.out_channels = out_channels
        self.one_hot_lut = one_hot_lut
        self.int_label_lut = None
        if self.one_hot_lut is not None:
            self.int_label_lut = {v: k for k, v in one_hot_lut.items()}
        self.cell_masks = cell_masks

        if transform is None:
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
                        spatial_axis=0,
                    ),
                    RandFlipd(
                        # Horizontal Flip
                        keys=["data", "mask"],
                        prob=0.5,
                        spatial_axis=1,
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
        else:
            self.transform = transform

        return

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, index):
        ci = self.labels_df.iloc[index]  # crop info
        fov = self.stores[ci.store_key][ci.tile_pheno][0]
        mask_fov = self.stores[ci.store_key][ci.tile_pheno]["seg"]

        fov_attrs = self.stores[ci.store_key][ci.tile_pheno].zattrs.asdict()

        all_channel_names = self.stores[ci.store_key][ci.tile_pheno].channel_names
        if self.out_channels == "random":
            channel_names = [random.choice(all_channel_names)]
        if self.out_channels == "all":
            channel_names = all_channel_names
        else:
            channel_names = self.out_channels
        channel_index = [all_channel_names.index(c) for c in channel_names]

        means = [
            fov_attrs["normalization"][i]["fov_statistics"]["mean"]
            for i in channel_names
        ]
        mean_divisor = np.expand_dims(np.asarray(means), (1, 2))
        stds = [
            fov_attrs["normalization"][i]["fov_statistics"]["std"]
            for i in channel_names
        ]
        std_subtractor = np.expand_dims(np.asarray(stds), (1, 2))

        bbox = ast.literal_eval(ci.bbox)

        data = np.asarray(
            fov[0, channel_index, 0, slice(bbox[0], bbox[2]), slice(bbox[1], bbox[3])]
        ).copy()

        mask = np.asarray(
            mask_fov[0, :, 0, slice(bbox[0], bbox[2]), slice(bbox[1], bbox[3])]
        ).copy()
        sc_mask = mask == ci.segmentation_id

        data_norm = (data - mean_divisor) / std_subtractor

        if self.cell_masks:
            data_norm = data_norm * sc_mask

        gene_label = self.one_hot_lut[ci.gene_name]

        batch = {
            "data": data_norm.astype(np.float32),
            "gene_label": gene_label,
            "marker_label": channel_names,
            "mask": sc_mask,
            "total_index": ci.total_index,
        }

        if self.transform is not None:
            batch = self.transform(batch)

        return batch


class TripletOpsDataset(scDataSet):
    def __init__(
        self,
        *args,
        transform=None,
        cell_masks=True,
        positive_source="perturbation",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.cell_masks = cell_masks
        self.positive_source = positive_source

        if transform is None:
            self.transform = Compose(
                [
                    SpatialPadd(
                        keys=[
                            "data",
                        ],
                        spatial_size=self.initial_yx_patch_size,
                    ),
                    RandFlipd(
                        keys=["data"],
                        prob=0.5,
                        spatial_axis=0,
                    ),
                    RandFlipd(
                        keys=["data"],
                        prob=0.5,
                        spatial_axis=1,
                    ),
                    RandAffined(
                        keys=["data"],
                        prob=0.8,
                        rotate_range=(3.14, 0),
                        scale_range=(0.2, 0.2),
                        shear_range=(0, 0),
                        padding_mode="zeros",
                    ),
                    RandAdjustContrastd(
                        keys=["data"],
                        prob=0.5,
                        gamma=[0.8, 1.2],
                    ),
                    RandScaleIntensityd(keys=["data"], prob=0.5, factors=0.5),
                    RandGaussianNoised(keys=["data"], prob=0.5, mean=0.0, std=0.05),
                    # RandGaussianSmoothd(
                    #     keys=["data"],
                    #     prob=1,
                    #     sigma_x=(0.04, 0.1),
                    #     sigma_y=(0.04, 0.1),
                    #     sigma_z=(0, 0),
                    # ),
                    CenterSpatialCropd(
                        keys=["data"],
                        roi_size=(self.final_yx_patch_size),
                    ),
                    ToTensord(
                        keys=["data"],
                    ),
                ]
            )
        else:
            self.transform = transform

    def get_crop(self, index):

        info = self.labels_df.iloc[index]

        fov = self.stores[info.store_key][info.tile_pheno][0]
        mask = self.stores[info.store_key][info.tile_pheno]["seg"]

        bbox = list(ast.literal_eval(info.bbox))

        fov_attrs = self.stores[info.store_key][info.tile_pheno].zattrs.asdict()
        all_channel_names = self.stores[info.store_key][info.tile_pheno].channel_names
        if self.out_channels == "random":
            channel_names = [random.choice(all_channel_names)]
        if self.out_channels == "all":
            channel_names = all_channel_names
        else:
            channel_names = self.out_channels
        channel_index = [all_channel_names.index(c) for c in channel_names]

        data = np.asarray(
            fov[0, channel_index, 0, slice(bbox[0], bbox[2]), slice(bbox[1], bbox[3])]
        ).copy()
        mask = np.asarray(
            mask[0, :, 0, slice(bbox[0], bbox[2]), slice(bbox[1], bbox[3])]
        ).copy()
        sc_mask = mask == info.segmentation_id

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
        if self.cell_masks:
            data_norm = data_norm * sc_mask

        return np.expand_dims(data_norm, axis=0).astype(np.float32)

    def __getitem__(self, index):
        anchor_i = self.labels_df.iloc[index]  # crop info
        anchor_crop = self.get_crop(index)

        if self.positive_source == "perturbation":
            positive_rows = np.flatnonzero(
                self.labels_df["gene_name"].values == anchor_i["gene_name"]
            )
            positive_indx = np.random.choice(positive_rows)
            positive_crop = self.get_crop(positive_indx)
        elif self.positive_source == "anchor":
            positive_crop = anchor_crop

        negative_rows = np.flatnonzero(
            self.labels_df["gene_name"].values != anchor_i["gene_name"]
        )
        negative_indx = np.random.choice(negative_rows)
        negative_crop = self.get_crop(negative_indx)

        gene_label = self.one_hot_lut[anchor_i.gene_name]
        batch = {
            "anchor": anchor_crop,
            "positive": positive_crop,
            "negative": negative_crop,
            "total_index": anchor_i.total_index,
            "gene_label": gene_label,
        }

        if self.transform is not None:
            for k, v in batch.items():
                if k == "total_index" or k == "gene_label":
                    continue
                mini_batch = {"data": v}
                mini_batch_trans = self.transform(mini_batch)
                batch[k] = mini_batch_trans["data"]

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
    ):
        self.experiments = experiments
        self.data_split = data_split
        self.shuffle_seed = shuffle_seed
        self.num_workers = 1
        self.initial_yx_patch_size = initial_yx_patch_size
        self.final_yx_patch_size = final_yx_patch_size
        self.batch_size = batch_size
        self.out_channels = out_channels

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
            dataset = OpsDataset(exp_name)
            for w in wells:
                labels_tmp = pd.read_csv(dataset.append_well("linked_results", w))
                labels_tmp["store_key"] = exp_name
                labels.append(self.remove_padding(labels_tmp, exp_name))
        labels_df = pd.concat(labels, ignore_index=True)
        labels_df["gene_name"] = labels_df["gene_name"].fillna("NTC")
        labels_df["total_index"] = np.arange(len(labels_df))  # add index column

        return labels_df

    def gene_label_converter(self):
        """ """
        gene_labels = sorted(self.labels_df["gene_name"].unique())
        # one_hot_array = torch.nn.functional.one_hot(torch.arange(len(gene_labels)))
        # one_hot_lut = {gene: one_hot_array[i] for i, gene in enumerate(gene_labels)}
        label_int_lut = {gene: i for i, gene in enumerate(gene_labels)}

        return label_int_lut

    def remove_padding(self, labels_df, exp_name):
        """
        In the OPS dataset the fluorescent channels are not perfectly aligned with the phase channel,
        correcting for this alignment produces a pad of zeros around the edges of each image. We want only cells
        that have the full crop within the fluoescent channel, without any padding appearing in the crop.
        """
        y_half, x_half = (d // 2 for d in self.initial_yx_patch_size[-2:])

        rand_tile = labels_df["tile_pheno"].sample(n=1).to_list()[0]

        array = self.store_dict[exp_name][rand_tile][0][0, :, 0, :, :]
        mask_2d = np.all(array > 0, axis=0)
        ys, xs = np.nonzero(mask_2d)
        y_min, y_max = ys.min(), ys.max()
        x_min, x_max = xs.min(), xs.max()

        y_range = (y_min + y_half, y_max - y_half)
        x_range = (x_min + x_half, x_max - x_half)
        labels_df_filtered = labels_df[
            labels_df["x_local_pheno"].between(*x_range, inclusive="neither")
            & labels_df["y_local_pheno"].between(*y_range, inclusive="neither")
        ]

        # temporary!!!
        # filter out cells that are too close to the edge of the well, roughly further
        # than 05026 from the center
        x_center = (
            labels_df_filtered["x_global_pheno"].max()
            - labels_df_filtered["x_global_pheno"].min()
        ) / 2 + labels_df_filtered["x_global_pheno"].min()
        y_center = (
            labels_df_filtered["y_global_pheno"].max()
            - labels_df_filtered["y_global_pheno"].min()
        ) / 2 + labels_df_filtered["y_global_pheno"].min()
        distances = np.sqrt(
            (labels_df_filtered["x_global_pheno"] - x_center) ** 2
            + (labels_df_filtered["y_global_pheno"] - y_center) ** 2
        )

        r = 46_000
        labels_df_filtered = labels_df_filtered[distances <= r]

        return labels_df_filtered

    def combine_stores(self):
        """ """
        stores = {}
        for exp_name, wells in self.experiments.items():
            dataset = OpsDataset(exp_name)
            stores[f"{exp_name}"] = open_ome_zarr(
                dataset.store_paths["pheno_assembled"], mode="r"
            )

        return stores

    def construct_dataloaders(
        self,
        num_workers: int = 1,
        transform=None,
        shuffle: bool = True,
        dataset_type: Literal["basic", "triplet"] = "basic",
        triplet_kwargs: dict = None,
        basic_kwargs: dict = None,
    ):
        """
        Returns train, val and test dataloaders
        """
        self.store_dict = self.combine_stores()
        labels_df = self.get_labels()
        self.labels_df = labels_df
        self.label_int_lut = self.gene_label_converter()

        train_ind, val_ind, test_ind = self.split_data(labels_df)

        common_kwargs = {
            "initial_yx_patch_size": self.initial_yx_patch_size,
            "final_yx_patch_size": self.final_yx_patch_size,
            "out_channels": self.out_channels,
            "one_hot_lut": self.label_int_lut,
        }

        if dataset_type == "basic":
            DS = scDataSet
            dataset_kwargs = {**common_kwargs, **(basic_kwargs if basic_kwargs else {})}
        elif dataset_type == "triplet":
            DS = TripletOpsDataset
            dataset_kwargs = {
                **common_kwargs,
                **(triplet_kwargs if triplet_kwargs else {}),
            }

        if len(train_ind) > 0:
            train_dataset = DS(
                stores=self.store_dict,
                labels_df=labels_df.iloc[train_ind],
                transform=transform,
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
                transform=transform,
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
                transform=transform,
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
