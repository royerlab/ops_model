import torch
import numpy as np


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


def collate_basic_dataset(batch):
    """
    Custom collate function for BasicDataset batches.
    All images should already be the same size after transforms.

    Args:
        batch: List of dictionaries from dataset __getitem__

    Returns:
        Dictionary with batched tensors
    """
    # Initialize lists for batched data
    data_list = []
    mask_list = []
    gene_labels = []
    marker_labels = []
    total_indices = []
    crop_infos = []

    for item in batch:
        data_list.append(item["data"])
        mask_list.append(item["mask"])
        gene_labels.append(item["gene_label"])
        marker_labels.append(item["marker_label"])
        total_indices.append(item["total_index"])
        crop_infos.append(item["crop_info"])

    # Stack into batch tensors
    batched = {
        "data": torch.stack(data_list),
        "mask": torch.stack(mask_list),
        "gene_label": torch.tensor(gene_labels),
        "marker_label": marker_labels,  # Keep as list since these are strings
        "total_index": torch.tensor(total_indices),
        "crop_info": crop_infos,  # Keep as list of dicts
    }

    return batched


def create_contrastive_collate_fcn(use_negative: bool):
    def collate_contrastive_dataset(batch):
        """
        Custom collate function for ContrastiveDataset batches.
        All images should already be the same size after transforms.

        Args:
            batch: List of dictionaries from dataset __getitem__
        """

        anchor_list = []
        positive_list = []
        negative_list = []
        gene_labels = []
        marker_labels = []
        crop_infos = []

        for item in batch:
            anchor_list.append(item["anchor"])
            positive_list.append(item["positive"])
            if use_negative:
                negative_list.append(item["negative"])
            gene_labels.append(item["gene_label"])
            marker_labels.append(item["marker_label"])
            crop_infos.append(item["crop_info"])

        # Stack into batch tensors
        batched = {
            "anchor": torch.stack(anchor_list),
            "positive": torch.stack(positive_list),
            "negative": torch.stack(negative_list) if use_negative else None,
            "gene_label": gene_labels,  # Keep as list since these are dicts
            "marker_label": marker_labels,  # Keep as list since these are strings
            "crop_info": crop_infos,  # Keep as list of dicts
        }
        return batched

    return collate_contrastive_dataset
