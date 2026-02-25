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
import random
from viscy.transforms import (
    RandAdjustContrastd,
    RandAffined,
    RandGaussianNoised,
    RandScaleIntensityd,
)

from .paths import OpsPaths
from .qc.qc_labels import filter_small_bboxes
from .collate_utils import (
    collate_basic_dataset,
    collate_variable_size_cells,
    create_contrastive_collate_fcn,
)

import warnings

# Filter zarr-related warnings (ZarrUserWarning was removed in zarr v2.18+)
warnings.filterwarnings("ignore", module="zarr")


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
        transform: Optional[Callable] = None,
    ):
        self.stores = stores
        self.labels_df = labels_df
        self.initial_yx_patch_size = initial_yx_patch_size
        self.final_yx_patch_size = final_yx_patch_size
        self.out_channels = out_channels
        self.label_int_lut = label_int_lut
        self.int_label_lut = int_label_lut
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
                        prob=0.75,
                        max_k=3,
                        spatial_axes=(-2, -1),
                    ),
                    RandAdjustContrastd(
                        keys=["data"],
                        prob=0.5,
                        gamma=[0.8, 1.2],
                    ),
                    RandScaleIntensityd(keys=["data"], prob=0.5, factors=0.5),
                    RandGaussianNoised(keys=["data"], prob=0.5, mean=0.0, std=0.05),
                    ToTensord(
                        keys=["data", "mask"],
                    ),
                ]
            )
        else:
            self.transform = transform
        return

    def _normalize_data(self, channel_names, data):
        img_list = []
        for ch in channel_names:
            img = data[channel_names.index(ch)]
            if ch == "Phase2D" or ch == "Focus3D":
                # Use nanstd to ignore NaN values during std calculation
                std = np.nanstd(img)
                img_norm = img / std if std > 1e-8 else np.zeros_like(img)
                img_list.append(img_norm)
            else:
                # apply log normalization, clip negative values to avoid NaN
                img_clipped = np.clip(img, 0, None)
                log_img = np.log1p(img_clipped)
                # Use nanstd and nanmean to ignore NaN values
                std = np.nanstd(log_img)
                if std > 1e-8:
                    img_norm = (log_img - np.nanmean(log_img)) / std
                else:
                    img_norm = np.zeros_like(log_img)
                img_list.append(img_norm)

        data_norm = np.stack(img_list, axis=0)

        return data_norm

    def _clip_bbox_to_fov(self, bbox, fov_shape):
        """
        Adjust bbox to stay within FOV boundaries while preserving size.
        If bbox extends outside FOV, shift it inward. If it's still too large,
        then clip it.

        Parameters
        ----------
        bbox : tuple
            (ymin, xmin, ymax, xmax)
        fov_shape : tuple
            FOV shape (T, C, Z, Y, X)

        Returns
        -------
        tuple
            Adjusted (ymin, xmin, ymax, xmax)
        """
        ymin, xmin, ymax, xmax = bbox
        fov_height = fov_shape[-2]
        fov_width = fov_shape[-1]

        bbox_height = ymax - ymin
        bbox_width = xmax - xmin

        # Shift bbox if it extends beyond boundaries
        if ymin < 0:
            # Shift down
            ymax = ymax - ymin  # ymax + |ymin|
            ymin = 0
        if xmin < 0:
            # Shift right
            xmax = xmax - xmin  # xmax + |xmin|
            xmin = 0

        if ymax > fov_height:
            # Shift up
            ymin = ymin - (ymax - fov_height)
            ymax = fov_height
        if xmax > fov_width:
            # Shift left
            xmin = xmin - (xmax - fov_width)
            xmax = fov_width

        # Final clip in case bbox is larger than FOV
        ymin_clipped = max(0, ymin)
        xmin_clipped = max(0, xmin)
        ymax_clipped = min(fov_height, ymax)
        xmax_clipped = min(fov_width, xmax)

        return (ymin_clipped, xmin_clipped, ymax_clipped, xmax_clipped)

    def _pad_bbox(self, bbox, final_shape):
        """
        bbox: (ymin, xmin, ymax, xmax)
        final_shape: (height, width)

        If bbox is smaller than final_shape, pad it equally on all sides to reach final_shape.
        """
        if len(final_shape) > 2:
            final_shape = final_shape[-2:]

        ymin, xmin, ymax, xmax = bbox
        target_height, target_width = final_shape

        # Calculate current bbox dimensions
        current_height = ymax - ymin
        current_width = xmax - xmin

        # Calculate padding needed
        height_padding = max(0, target_height - current_height)
        width_padding = max(0, target_width - current_width)

        # Distribute padding equally on both sides
        pad_top = height_padding / 2
        pad_bottom = height_padding / 2
        pad_left = width_padding / 2
        pad_right = width_padding / 2

        # Apply padding
        new_ymin = int(ymin - pad_top)
        new_ymax = int(ymax + pad_bottom)
        new_xmin = int(xmin - pad_left)
        new_xmax = int(xmax + pad_right)

        return (new_ymin, new_xmin, new_ymax, new_xmax)

    def _get_channels(self, ci, well):

        attrs = self.stores[ci.store_key][well].attrs.asdict()
        all_channel_names = [a["label"] for a in attrs["ome"]["omero"]["channels"]]

        channel_names = [ci.channel]
        channel_index = [all_channel_names.index(c) for c in channel_names]

        return channel_names, channel_index

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, index):
        ci = self.labels_df.iloc[index]  # crop info

        well = ci.well
        fov = self.stores[ci.store_key][well]["0"]
        gene_label = self.label_int_lut[ci.gene_name]
        total_index = ci.total_index
        bbox = ast.literal_eval(ci.bbox)
        if not self.cell_masks:
            bbox = self._pad_bbox(bbox, self.initial_yx_patch_size)

        # Clip bbox to FOV boundaries to prevent NaN from out-of-bounds access
        bbox = self._clip_bbox_to_fov(bbox, fov.shape)

        channel_names, channel_index = self._get_channels(ci, well)

        data = np.asarray(
            fov[
                0:1,
                channel_index,
                0:1,
                slice(bbox[0], bbox[2]),
                slice(bbox[1], bbox[3]),
            ]
        ).copy()
        data = np.squeeze(data)
        if len(data.shape) == 2:
            data = np.expand_dims(data, axis=0)

        if self.cell_masks:
            mask_fov = self.stores[ci.store_key][well]["labels"]["cell_seg"]["0"]
            mask = np.asarray(
                mask_fov[
                    0:1, :, 0:1, slice(bbox[0], bbox[2]), slice(bbox[1], bbox[3])
                ]
            ).copy()
            mask = np.squeeze(mask)
            mask = np.expand_dims(mask, axis=0)
            sc_mask = mask == ci.segmentation_id
            data_norm = self._normalize_data(channel_names, data)
            data_norm = data_norm * sc_mask
        else:
            sc_mask = np.ones_like(data[:1], dtype=bool)
            data_norm = self._normalize_data(channel_names, data)

        if len(self.final_yx_patch_size) == 3:
            data_norm = np.expand_dims(data_norm, axis=0)
            sc_mask = np.expand_dims(sc_mask, axis=0)

        batch = {
            "data": data_norm.astype(np.float32),
            "mask": sc_mask,
            "gene_label": gene_label,
            "marker_label": channel_names,
            "total_index": total_index,
            "crop_info": ci.to_dict(),
        }

        batch = self.transform(batch)

        return batch


class ContrastiveDataset(BaseDataset):
    def __init__(
        self,
        transform=None,
        transforms: list = None,
        positive_source: str | Literal["self"] | Literal["perturbation"] = "self",
        use_negative: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.positive_source = positive_source
        self.use_negative = use_negative

        # Priority: explicit transform object > transforms config list > default
        if transform is not None:
            # Explicit Compose object passed (for programmatic use)
            self.transform = transform
        elif transforms is not None:
            # List of transform configs from YAML (for config-based use)
            self.transform = self._build_transform_from_config(transforms)
        else:
            # Use default transforms
            self.transform = Compose(
                [
                    SpatialPadd(
                        keys=["data"],
                        spatial_size=self.initial_yx_patch_size,
                    ),
                    RandFlipd(
                        # horizontal flip
                        keys=["data"],
                        prob=0.5,
                        spatial_axis=-1,
                    ),
                    RandFlipd(
                        # vertical flip
                        keys=["data"],
                        prob=0.5,
                        spatial_axis=-2,
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
                    CenterSpatialCropd(
                        keys=["data"],
                        roi_size=(self.final_yx_patch_size),
                    ),
                    ToTensord(
                        keys=["data"],
                    ),
                ]
            )
        return

    @staticmethod
    def _build_transform_from_config(transform_config: list):
        """Build MONAI Compose transform from YAML config.

        Parameters
        ----------
        transform_config : list
            List of dicts with 'class_path' and 'init_args'

        Returns
        -------
        Compose
            MONAI Compose transform
        """
        from lightning.pytorch.cli import instantiate_class

        transforms = []
        for t_config in transform_config:
            # Use Lightning's instantiate_class to handle class_path/init_args
            transform_obj = instantiate_class(tuple(), t_config)
            transforms.append(transform_obj)

        return Compose(transforms)

    def get_crop(self, index):
        ci = self.labels_df.iloc[index]  # crop info

        well = ci.well
        fov = self.stores[ci.store_key][well]["0"]
        mask_fov = self.stores[ci.store_key][well]["labels"]["cell_seg"]["0"]
        gene_label = self.label_int_lut[ci.gene_name]
        total_index = ci.total_index
        bbox = ast.literal_eval(ci.bbox)
        if not self.cell_masks:
            bbox = self._pad_bbox(bbox, self.initial_yx_patch_size)

        # Clip bbox to FOV boundaries to prevent NaN from out-of-bounds access
        bbox = self._clip_bbox_to_fov(bbox, fov.shape)

        channel_names, channel_index = self._get_channels(ci, well)

        data = np.asarray(
            fov[
                0:1,
                channel_index,
                0:1,
                slice(bbox[0], bbox[2]),
                slice(bbox[1], bbox[3]),
            ]
        ).copy()
        data = np.squeeze(data)
        if len(data.shape) == 2:
            data = np.expand_dims(data, axis=0)

        mask = np.asarray(
            mask_fov[0:1, :, 0:1, slice(bbox[0], bbox[2]), slice(bbox[1], bbox[3])]
        ).copy()
        mask = np.squeeze(mask)
        mask = np.expand_dims(mask, axis=0)
        sc_mask = mask == ci.segmentation_id

        data_norm = self._normalize_data(channel_names, data)

        if self.cell_masks:
            data_norm = data_norm * sc_mask

        if len(self.final_yx_patch_size) == 3:
            data_norm = np.expand_dims(data_norm, axis=0)
            sc_mask = np.expand_dims(sc_mask, axis=0)

        return data_norm, sc_mask, gene_label, channel_names, total_index, ci

    def __getitem__(self, index):
        anchor_i = self.labels_df.iloc[index]
        anchor_data, _, anchor_gene_label, channel_names, _, _ = self.get_crop(index)

        if self.positive_source == "self":
            positive_data = anchor_data.copy()
            positive_gene_label = anchor_gene_label
            positive_i = anchor_i

        if self.positive_source == "perturbation":
            positive_rows = np.flatnonzero(
                self.labels_df["gene_name"].values == anchor_i["gene_name"]
            )
            positive_indx = np.random.choice(positive_rows)
            positive_data, _, positive_gene_label, _, _, positive_i = self.get_crop(
                positive_indx
            )

        batch = {
            "anchor": anchor_data.astype(np.float32),
            "positive": positive_data.astype(np.float32),
            "gene_label": {
                "anchor": anchor_gene_label,
                "positive": positive_gene_label,
            },
            "marker_label": channel_names,
            "crop_info": {
                "anchor": anchor_i.to_dict(),
                "positive": positive_i.to_dict(),
            },
        }

        if self.use_negative:
            negative_rows = np.flatnonzero(
                self.labels_df["gene_name"].values != anchor_i["gene_name"]
            )
            negative_indx = np.random.choice(negative_rows)

            negative_data, _, negative_gene_label, _, _, negative_i = self.get_crop(
                negative_indx
            )

            batch["negative"] = negative_data.astype(np.float32)
            batch["gene_label"]["negative"] = negative_gene_label
            batch["crop_info"]["negative"] = negative_i.to_dict()

        if self.transform is not None:
            for k, v in batch.items():
                if k in ["gene_label", "marker_label", "crop_info"]:
                    continue

                # Force fresh randomization for MONAI transforms
                # Without this, validation samples get identical augmentations each epoch
                for t in self.transform.transforms:
                    if hasattr(t, "set_random_state"):
                        t.set_random_state(seed=None)

                mini_batch = {"data": v}
                mini_batch_trans = self.transform(mini_batch)
                batch[k] = mini_batch_trans["data"]

        return batch


class RandomCropDataset(BaseDataset):
    pass


class CellProfileDataset(BaseDataset):
    def __init__(self, stores: dict, labels_df: pd.DataFrame, **kwargs):
        super().__init__(stores, labels_df, **kwargs)

    def __getitem__(self, index):

        ci = self.labels_df.iloc[index]  # crop info

        well = ci.well
        fov = self.stores[ci.store_key][well]["0"]
        cell_mask_fov = self.stores[ci.store_key][well]["labels"]["cell_seg"]["0"]
        nuc_mask_fov = self.stores[ci.store_key][well]["labels"]["nuclear_seg"]["0"]
        bbox = ast.literal_eval(ci.bbox)
        gene_label = self.label_int_lut[ci.gene_name]
        total_index = ci.total_index

        channel_names, channel_index = self._get_channels(ci, well)

        data = np.asarray(
            fov[
                0:1,
                channel_index,
                0:1,
                slice(bbox[0], bbox[2]),
                slice(bbox[1], bbox[3]),
            ]
        ).copy()
        data = np.squeeze(data)
        if len(data.shape) == 2:
            data = np.expand_dims(data, axis=0)

        cell_mask = np.asarray(
            cell_mask_fov[0:1, :, 0:1, slice(bbox[0], bbox[2]), slice(bbox[1], bbox[3])]
        ).copy()
        cell_mask = np.squeeze(cell_mask)
        cell_mask = np.expand_dims(cell_mask, axis=0)
        nuc_mask = np.asarray(
            nuc_mask_fov[0:1, :, 0:1, slice(bbox[0], bbox[2]), slice(bbox[1], bbox[3])]
        ).copy()
        nuc_mask = np.squeeze(nuc_mask)
        nuc_mask = np.expand_dims(nuc_mask, axis=0)
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
                labels_path = OpsPaths(exp_name, well=w).links["training"]
                if labels_path.endswith(".parquet"):
                    labels_tmp = pd.read_parquet(labels_path)
                else:
                    labels_tmp = pd.read_csv(labels_path, low_memory=False)

                # remove rows with NaN segmentation_id
                labels_tmp = labels_tmp.dropna(subset=["segmentation_id"])

                # remove cells with suspiciously small bounding boxes
                labels_tmp, num_rem = filter_small_bboxes(labels_tmp, threshold=5)
                if self.verbose:
                    print(
                        f"Removed {num_rem} cells with small bounding boxes from {exp_name} {w[0]}{w[2]}"
                    )
                labels_tmp["store_key"] = exp_name
                labels_tmp["well"] = w
                if self.verbose:
                    print(f"{exp_name} {w[0]}{w[2]}: {len(labels_tmp)} cells")
                labels.append(labels_tmp)
        labels_df = pd.concat(labels, ignore_index=True)
        if "Gene name" in labels_df.columns:
            labels_df["gene_name"] = labels_df["Gene name"].fillna("NTC")
        elif "gene_name" in labels_df.columns:
            labels_df["gene_name"] = labels_df["gene_name"].fillna("NTC")
        else:
            raise ValueError("No gene name column found in labels file")
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
                OpsPaths(exp_name).stores["phenotyping_v3"], mode="r"
            )

        return stores

    def balanced_sample_weights(self, df: pd.DataFrame, balance_col: str = "gene_name"):
        """
        Needs to be run on the individual train/val/test dataframes, becuase the
        entries get shuffled during splitting.
        """
        class_counts = df[balance_col].value_counts().to_dict()
        class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
        sample_weights = np.array(
            [class_weights[gene_name] for gene_name in df[balance_col]]
        )

        return sample_weights

    def create_sampler(
        self, df: pd.DataFrame, balanced: bool, balance_col: str = "gene_name"
    ):
        if not balanced:
            return None

        sample_weights = self.balanced_sample_weights(df, balance_col=balance_col)

        sampler = torch.utils.data.WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),  # total samples per epoch
            replacement=True,
        )
        return sampler

    @staticmethod
    def worker_init_fn(worker_id):
        """Initialize each dataloader worker with a unique random seed.

        This ensures that with seed_everything(), each worker produces
        different random augmentations instead of identical transforms.
        """
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

        # Also set MONAI's internal random state
        # MONAI transforms use their own RandomState instance
        import monai

        if hasattr(monai.transforms, "set_determinism"):
            # Don't set determinism - we want randomness
            pass

    def construct_dataloaders(
        self,
        labels_df: pd.DataFrame = None,
        num_workers: int = 1,
        pin_memory: bool = True,
        prefetch_factor: int = 2,
        dataset_type: Literal["basic", "contrastive", "cell_profile"] = "basic",
        balanced_sampling: bool = False,
        balanced_sampling_val: bool = False,
        balance_col: str = "gene_name",
        contrastive_kwargs: dict = None,
        basic_kwargs: dict = None,
        cp_kwargs: dict = None,
        train_loader_kwargs: dict = None,
    ):
        """
        Returns train, val and test dataloaders
        """

        if labels_df is None:
            labels_df = self.get_labels()
        self.labels_df = labels_df
        self.store_dict = self.combine_stores()
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
            self.collate_fcn = collate_basic_dataset

        elif dataset_type == "cell_profile":
            DS = CellProfileDataset
            dataset_kwargs = {**common_kwargs, **(cp_kwargs if cp_kwargs else {})}
            self.collate_fcn = (
                collate_variable_size_cells  # Use custom collate for variable sizes
            )
        elif dataset_type == "contrastive":
            DS = ContrastiveDataset
            dataset_kwargs = {
                **common_kwargs,
                **(contrastive_kwargs if contrastive_kwargs else {}),
            }
            self.collate_fcn = create_contrastive_collate_fcn(
                use_negative=contrastive_kwargs.get("use_negative", False)
            )
        else:
            raise ValueError(f"Unknown dataset_type: {dataset_type}")

        if len(train_ind) > 0:
            train_df = labels_df.iloc[train_ind]
            train_sampler = self.create_sampler(
                train_df, balanced_sampling, balance_col=balance_col
            )
            train_dataset = DS(
                stores=self.store_dict,
                labels_df=train_df,
                **dataset_kwargs,
            )
            train_loader = torch.utils.data.DataLoader(
                dataset=train_dataset,
                batch_size=self.batch_size,
                num_workers=num_workers,
                pin_memory=pin_memory,
                prefetch_factor=prefetch_factor if num_workers > 0 else None,
                worker_init_fn=self.worker_init_fn if num_workers > 0 else None,
                collate_fn=self.collate_fcn,
                sampler=train_sampler,
                **(train_loader_kwargs if train_loader_kwargs else {}),
            )

            self.train_loader = train_loader

        if len(val_ind) > 0:
            val_df = labels_df.iloc[val_ind]
            val_sampler = self.create_sampler(
                val_df, balanced_sampling_val, balance_col=balance_col
            )
            val_dataset = DS(
                stores=self.store_dict,
                labels_df=val_df,
                **dataset_kwargs,
            )
            val_loader = torch.utils.data.DataLoader(
                dataset=val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
                prefetch_factor=prefetch_factor if num_workers > 0 else None,
                worker_init_fn=self.worker_init_fn if num_workers > 0 else None,
                collate_fn=self.collate_fcn,
                sampler=val_sampler,
            )

            self.val_loader = val_loader

        if len(test_ind) > 0:
            test_df = labels_df.iloc[test_ind]
            test_sampler = self.create_sampler(
                test_df, balanced_sampling, balance_col=balance_col
            )
            test_dataset = DS(
                stores=self.store_dict,
                labels_df=test_df,
                **dataset_kwargs,
            )
            test_loader = torch.utils.data.DataLoader(
                dataset=test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
                prefetch_factor=prefetch_factor if num_workers > 0 else None,
                worker_init_fn=self.worker_init_fn if num_workers > 0 else None,
                collate_fn=self.collate_fcn,
                sampler=test_sampler,
            )

            self.test_loader = test_loader

        return
