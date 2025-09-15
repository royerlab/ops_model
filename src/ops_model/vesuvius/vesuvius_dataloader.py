import random
from pathlib import Path
from typing import List, Literal

import numpy as np
import pandas as pd
import torch
import zarr
from monai.transforms import (
    RandAffined,
    RandFlipd,
    RandRotate90d,
    ToTensord,
)
from torch.utils.data import Dataset
from torchvision import transforms
from viscy.transforms import (
    RandAdjustContrastd,
    RandAffined,
    RandGaussianNoised,
    RandGaussianSmoothd,
)


class VesuviusDataSet(Dataset):

    def __init__(
        self,
        store_path: Path,
        labels_df: pd.DataFrame,
        dataset_type: str,
        positive_source: str,
        in_channels: list | str,
        transform: list,
    ):
        self.store = zarr.open(store_path, mode="r")
        self.means = np.load(
            "/hpc/mydata/alexander.hillsley/ops/ops_analysis/ops_analysis/model/vesuvius/dataset_means.npy"
        )
        self.stds = np.load(
            "/hpc/mydata/alexander.hillsley/ops/ops_analysis/ops_analysis/model/vesuvius/dataset_stds.npy"
        )
        self.labels_df = labels_df
        self.dataset_type = dataset_type
        self.positive_source = positive_source
        self.num_genes = len(self.labels_df["gene_name"].unique())

        self.channels = [
            "1-DAPI",
            "2-Tubulin",
            "3-gamma-H2AX",
            "4-Actin",
        ]
        if isinstance(in_channels, list):
            if in_channels[0] == "all":
                self.in_channels = [0, 1, 2, 3]
            else:
                self.in_channels = []
                for c in in_channels:
                    self.in_channels.append(self.channels.index(c))

        self.transform = transforms.Compose(transform)

        return

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, index):
        crop_info = self.labels_df.iloc[index]
        img_crop = self.store[crop_info.path][crop_info.crop][self.in_channels].astype(
            float
        )
        int_label = int(crop_info["path"].rsplit("/")[0])

        batch = {
            "data": img_crop,
            "gene_label": int_label,
            "total_index": crop_info["count"],
        }

        if self.dataset_type == "triplet":
            temp_anchor = batch.pop("data")
            batch["anchor"] = temp_anchor
            if self.positive_source == "gene":
                num_pos_ind = self.store[crop_info["path"]].shape[0]
                pos_ind = random.randint(0, num_pos_ind - 1)
                positive_sample = self.store[crop_info["path"]][pos_ind]
            if self.positive_source == "self":
                positive_sample = img_crop

            negative_gene_path = (
                self.labels_df[self.labels_df["path"] != crop_info["path"]]["path"]
                .sample(n=1)
                .iloc[0]
            )
            num_neg_ind = self.store[f"{negative_gene_path}"].shape[0]
            neg_ind = random.randint(0, num_neg_ind - 1)
            negative_sample = self.store[f"{negative_gene_path}"][neg_ind]

            batch["positive"] = positive_sample
            batch["negative"] = negative_sample

        if self.transform is not None:
            for k, v in batch.items():
                if k == "total_index" or k == "gene_label":
                    continue
                v_norm = (
                    v - np.expand_dims(self.means[self.in_channels], axis=(1, 2))
                ) / np.expand_dims(self.stds[self.in_channels], axis=(1, 2))
                mini_batch = {"data": v_norm}
                mini_batch_trans = self.transform(mini_batch)
                if self.dataset_type == "triplet":
                    batch[k] = torch.unsqueeze(mini_batch_trans["data"], dim=1)
                else:
                    batch[k] = mini_batch_trans["data"]

        return batch


class VesuviusDataManager:
    def __init__(
        self,
        data_split: tuple = (0.9, 0.05, 0.05),
        shuffle_seed: int = 1,
        final_yx_patch_size: tuple = (100, 100),
        batch_size: int = 32,
        in_channels: List[str] | Literal["random"] = "random",
        **kwargs,
    ):
        self.store_path = (
            "/hpc/projects/intracellular_dashboard/ops/analysis/vesuvius.zarr"
        )
        self.label_path = "/hpc/mydata/alexander.hillsley/ops/ops_analysis/ops_analysis/model/vesuvius/vesuvius_labels.csv"
        self.labels_df = pd.read_csv(self.label_path)
        self.data_split = data_split
        self.shuffle_seed = shuffle_seed
        self.num_workers = 1
        self.final_yx_patch_size = final_yx_patch_size
        self.batch_size = batch_size
        self.in_channels = in_channels

        self.store_dict = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.gene_label_converter()

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

    def gene_label_converter(self):
        """ """
        unique_genes = self.labels_df.drop_duplicates(subset="path", keep="first")
        self.int_label_lut = {
            a.rsplit("/")[0]: g
            for (a, g) in zip(unique_genes["path"], unique_genes["gene_name"])
        }
        self.label_int_lut = {v: k for k, v in self.int_label_lut.items()}

        return

    def construct_dataloaders(
        self,
        num_workers: int = 1,
        transform=[
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
            RandAdjustContrastd(
                keys=["data"],
                prob=0.5,
                gamma=[0.8, 1.2],
            ),
            RandRotate90d(
                keys=["data"],
                prob=0.5,
                max_k=3,
            ),
            RandAffined(
                keys=["data"],
                prob=0.8,
                rotate_range=(3.14, 0),
                scale_range=(0.2, 0.2),
                shear_range=(0, 0),
                padding_mode="reflection",
            ),
            RandGaussianSmoothd(
                keys=["data"],
                prob=1,
                sigma_x=(0.04, 0.1),
                sigma_y=(0.04, 0.1),
                sigma_z=(0, 0),
            ),
            RandGaussianNoised(keys=["data"], prob=0.5, mean=0.0, std=0.1),
            ToTensord(
                keys=["data"],
            ),
        ],
        shuffle: bool = True,
        dataset_kwargs: dict = None,
    ):
        """
        Returns train, val and test dataloaders
        """
        train_ind, val_ind, test_ind = self.split_data(self.labels_df)

        common_kwargs = {
            "in_channels": self.in_channels,
            # "one_hot_lut": self.label_int_lut,
        }
        dataset_kwargs = {**common_kwargs, **(dataset_kwargs if dataset_kwargs else {})}

        if len(train_ind) > 0:
            train_dataset = VesuviusDataSet(
                store_path=self.store_path,
                labels_df=self.labels_df.iloc[train_ind],
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
            val_dataset = VesuviusDataSet(
                store_path=self.store_path,
                labels_df=self.labels_df.iloc[val_ind],
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
            test_dataset = VesuviusDataSet(
                store_path=self.store_path,
                labels_df=self.labels_df.iloc[test_ind],
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
