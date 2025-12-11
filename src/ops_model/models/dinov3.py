from tqdm import tqdm

import torch
import numpy as np
import pandas as pd
import os
from pathlib import Path
from torchvision.transforms import v2
from monai.transforms import (
    CenterSpatialCropd,
    Compose,
    SpatialPadd,
    ToTensord,
)

from ops_model.data import data_loader


class DinoV3Model:
    """
    TODO: model.forward has the option to provide masks??
    """

    def __init__(self):
        super().__init__()
        self.load_model()
        self.model.cuda()

    def load_model(self):
        repo_dir = "/hpc/projects/icd.ops/models/model_checkpoints/dinov3/dinov3"
        checkpoint = "/hpc/projects/icd.ops/models/model_checkpoints/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"

        self.model = torch.hub.load(
            repo_dir, "dinov3_vitl16", source="local", weights=checkpoint
        )
        return

    def extract_features(self, batch: dict) -> torch.Tensor:

        x = batch["data"]
        inputs = self.preprocess(x)

        with torch.inference_mode():
            with torch.autocast(device_type="cuda", dtype=torch.float32):
                output = self.model(inputs.cuda())
        return output

    def preprocess(self, x: np.ndarray, resize_size: int = 256) -> torch.Tensor:
        """Important: images should have a mean of 1 and std of 1 before input to dinov3"""

        to_tensor = v2.ToImage()
        resize = v2.Resize((resize_size, resize_size), antialias=True)
        to_float = v2.ToDtype(torch.float32, scale=True)
        # goal is to have output images with zero mean and unit variance
        normalize = v2.Normalize(
            mean=(0, 0, 0),  # mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )
        return v2.Compose([to_tensor, resize, to_float, normalize])(x.cuda())

    def _reshape_to_spatial(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Reshape (B, N, D) patches to (B, D, H', W') spatial map.
        * Only need if running model.get_intermediate_layers, just calling
        model.forward will return a single embedding per image

        """
        batch_size, n_patches, embed_dim = patches.shape
        h_patches = w_patches = int(np.sqrt(n_patches))

        if h_patches * w_patches != n_patches:
            raise ValueError(
                f"Number of patches ({n_patches}) is not a perfect square. "
                f"Expected {h_patches}*{h_patches} = {h_patches * h_patches}."
            )

        # (B, N, D) -> (B, H', W', D) -> (B, D, H', W')
        spatial = patches.reshape(batch_size, h_patches, w_patches, embed_dim)
        return spatial.permute(0, 3, 1, 2)


def extract_dinov3_features(
    experiment_dict: dict,
    batch_size: int = 256,
    output_dir: str = "./dinov3_features",
    num_workers: int = 8,
):

    print(f"Extracting DINOv3 features for {list(experiment_dict.keys())}")
    dm = data_loader.OpsDataManager(
        experiments=experiment_dict,
        batch_size=batch_size,
        data_split=(0, 0, 1),
        out_channels=["Phase2D"],
        initial_yx_patch_size=(256, 256),
        verbose=False,
    )
    dm.construct_dataloaders(
        num_workers=num_workers,
        dataset_type="basic",
        basic_kwargs={
            "cell_masks": True,
            "transform": Compose(
                [
                    SpatialPadd(
                        keys=["data", "mask"],
                        spatial_size=dm.initial_yx_patch_size,
                    ),
                    CenterSpatialCropd(
                        keys=["data", "mask"], roi_size=(dm.final_yx_patch_size)
                    ),
                    ToTensord(
                        keys=["data", "mask"],
                    ),
                ]
            ),
        },
    )
    test_loader = dm.test_loader
    print(f"Created dataset with {len(test_loader.dataset)} crops")

    dino = DinoV3Model()
    print("DINOv3 model loaded")

    # Setup output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    chunk_subdir = output_dir / "chunks"
    chunk_subdir.mkdir(parents=True, exist_ok=True)

    save_every = 100  # Save every N batches
    all_features = []
    chunk_idx = 0

    # Extract features from all batches
    for batch_idx, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
        # Extract features using the DINOv3 model
        features = dino.extract_features(batch)

        # Convert to numpy and create dataframe with metadata
        features_np = features.cpu().numpy()

        features_db = pd.DataFrame(features_np)
        features_db["label_int"] = batch["gene_label"].numpy()
        features_db["label_str"] = [
            dm.int_label_lut[label] for label in batch["gene_label"].numpy()
        ]
        features_db["sgRNA"] = [a["sgRNA"] for a in batch["crop_info"]]
        features_db["experiment"] = [a["store_key"] for a in batch["crop_info"]]
        features_db["x_position"] = [a["x_pheno"] for a in batch["crop_info"]]
        features_db["y_position"] = [a["y_pheno"] for a in batch["crop_info"]]

        features_db["well"] = [
            a["well"] + "_" + a["store_key"] for a in batch["crop_info"]
        ]
        all_features.append(features_db)

        if batch_idx % save_every == 0 and batch_idx > 0:
            df_chunk = pd.concat(all_features, ignore_index=True)
            csv_path = chunk_subdir / f"features_chunk_{chunk_idx}.csv"
            df_chunk.to_csv(csv_path, index=False)
            print(
                f"Saved chunk {chunk_idx} with {len(all_features)} batches to {csv_path}"
            )

            # Reset for next chunk
            all_features = []
            chunk_idx += 1

    # Save any remaining features
    if all_features:
        df_chunk = pd.concat(all_features, ignore_index=True)
        csv_path = chunk_subdir / f"features_chunk_{chunk_idx}.csv"
        df_chunk.to_csv(csv_path, index=False)
        print(
            f"Saved final chunk {chunk_idx} with {len(all_features)} batches to {csv_path}"
        )
        chunk_idx += 1

    # Load and concatenate all CSV files
    print(f"\nLoading and concatenating {chunk_idx} chunks...")
    csv_files = sorted(chunk_subdir.glob("features_chunk_*.csv"))

    if not csv_files:
        print("No feature files found!")
        return None

    df_list = [pd.read_csv(csv_file) for csv_file in csv_files]
    final_df = pd.concat(df_list, ignore_index=True)

    # Save the final concatenated dataframe
    final_path = output_dir / "dinov3_features_all.csv"
    final_df.to_csv(final_path, index=False)
    print(f"Saved final concatenated features to {final_path}")
    print(f"Final dataframe shape: {final_df.shape}")

    return final_df


if __name__ == "__main__":
    experiment_dict = {"ops0031_20250424": ["A/1/0", "A/2/0", "A/3/0"]}
    extract_dinov3_features(
        experiment_dict=experiment_dict,
        batch_size=256,
        output_dir="/hpc/projects/icd.ops/ops0031_20250424/3-assembly/dino_features",
        num_workers=10,
    )
