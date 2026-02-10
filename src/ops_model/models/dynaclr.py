from pathlib import Path

import numpy as np
import pandas as pd
import anndata as ad
import zarr
from lightning.pytorch.callbacks import BasePredictionWriter
from pytorch_metric_learning.losses import NTXentLoss
from viscy.representation.contrastive import ContrastiveEncoder
from viscy.representation.engine import ContrastiveModule


class DynaClrPredictionWriter(BasePredictionWriter):
    def __init__(self, output_dir: str, zarr_suffix: str, write_interval):
        super().__init__(write_interval)
        self.output_dir = Path(output_dir)
        self.zarr_suffix = zarr_suffix
        self.high_count = 0
        self.low_count = 0
        self.metadata = {}

    def setup(self, trainer, pl_module, stage):
        if stage == "predict":
            self.emb_store = zarr.open_group(
                self.output_dir / f"features_{self.zarr_suffix}.zarr", mode="w"
            )
            self.emb_store.create_group(self.high_count)

        return

    def write_on_batch_end(
        self,
        trainer,
        pl_module,
        prediction,
        batch_indices,  # index of the sample within the epoch
        batch,
        batch_idx,
        dataloader_idx,
    ):

        out_dict = prediction
        features = out_dict["features"]
        projections = out_dict["projections"]

        self.emb_store[self.high_count].create_dataset(
            self.low_count,
            data=features.detach().cpu().numpy(),
            shape=features.shape,
            chunks=(1,) + features.shape[1:],
        )

        for i in range(features.shape[0]):
            total_index = batch["total_index"][i].detach().item()
            self.metadata[total_index] = {
                "position": f"{self.high_count}/{self.low_count}",
                "batch_index": batch_indices[i],
                "index": i,
                "gene_label": batch["gene_label"][i].detach().item(),
                "total_index": batch["total_index"][i].detach().item(),
            }

        self.emb_store.attrs.put(self.metadata)

        self.low_count += 1
        if self.low_count % 10 == 0:
            self.high_count += 1
            self.emb_store.create_group(self.high_count)
            self.low_count = 0

        return


class DynaClrAnnDataWriter(BasePredictionWriter):
    """
    Writes DynaCLR predictions directly to AnnData Zarr format.

    Accumulates predictions in memory and writes once at epoch end.
    Uses write_zarr() for efficient storage of large datasets.

    Parameters
    ----------
    output_dir : str
        Directory to save the AnnData file
    run_name : str
        Name for the output file
    labels_df : pd.DataFrame
        DataFrame containing cell metadata, indexed by total_index
    save_features : bool, default=True
        Save backbone features (embedding dimension)
    save_projections : bool, default=False
        Save projection head outputs
    """

    def __init__(
        self,
        output_dir: str,
        run_name: str,
        labels_df: pd.DataFrame,
        save_features: bool = True,
        save_projections: bool = False,
    ):
        super().__init__(write_interval="batch_and_epoch")
        self.output_dir = Path(output_dir)
        self.run_name = run_name
        self.labels_df = labels_df
        self.save_features = save_features
        self.save_projections = save_projections

        # Accumulate predictions in memory
        self.predictions_list = []

    def write_on_batch_end(
        self,
        trainer,
        pl_module,
        prediction,
        batch_indices,
        batch,
        batch_idx,
        dataloader_idx,
    ):
        """Accumulate predictions in memory."""
        batch_data = {
            "features": prediction["features"].detach().cpu().numpy(),
            "total_index": prediction["total_index"].cpu().numpy(),
            "gene_label": prediction["gene_label"].cpu().numpy(),
        }
        if self.save_projections:
            batch_data["projections"] = prediction["projections"].detach().cpu().numpy()

        self.predictions_list.append(batch_data)

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        """Concatenate all predictions and write to AnnData Zarr format."""

        # Concatenate all batches
        features = np.concatenate([p["features"] for p in self.predictions_list], axis=0)
        total_indices = np.concatenate(
            [p["total_index"] for p in self.predictions_list], axis=0
        )
        gene_labels = np.concatenate(
            [p["gene_label"] for p in self.predictions_list], axis=0
        )

        # Create AnnData object with features in .X (matching evaluate.py line 83)
        adata = ad.AnnData(features)

        # Add cell metadata to .obs (matching evaluate.py lines 84-99)
        adata.obs["gene_str"] = [
            self.labels_df.loc[idx, "gene_name"] for idx in total_indices
        ]
        adata.obs["gene_int"] = gene_labels
        adata.obs["experiment"] = [
            self.labels_df.loc[idx, "store_key"] for idx in total_indices
        ]
        adata.obs["tile"] = [
            self.labels_df.loc[idx, "tile_pheno"] for idx in total_indices
        ]
        adata.obs["x_local"] = [
            self.labels_df.loc[idx, "x_local_pheno"] for idx in total_indices
        ]
        adata.obs["y_local"] = [
            self.labels_df.loc[idx, "y_local_pheno"] for idx in total_indices
        ]

        # Add bbox to .obsm (matching evaluate.py line 100)
        adata.obsm["bbox"] = np.asarray(
            [self.labels_df.loc[idx, "bbox"] for idx in total_indices]
        )

        # Set feature names
        adata.var_names = [f"feature_{i}" for i in range(features.shape[1])]

        # Add projections to obsm if requested
        if self.save_projections:
            projections = np.concatenate(
                [p["projections"] for p in self.predictions_list], axis=0
            )
            adata.obsm["X_projection"] = projections

        # Save to disk using write_zarr (matching evaluate.py line 148)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / f"dynaclr_embeddings_{self.run_name}.zarr"
        adata.write_zarr(output_path)

        print(f"\n{'='*60}")
        print(f"Saved embeddings to: {output_path}")
        print(f"Shape: {adata.shape} (cells × features)")
        print(f"AnnData structure:")
        print(f"  - adata.X: embeddings ({features.shape})")
        print(f"  - adata.obs: metadata columns = {list(adata.obs.columns)}")
        obsm_keys = ["bbox"]
        if self.save_projections:
            obsm_keys.append("X_projection")
        print(f"  - adata.obsm: {', '.join(obsm_keys)}")
        print(f"{'='*60}\n")


class LitDynaClr(ContrastiveModule):
    def __init__(
        self,
        encoder=None,
        # loss_function=nn.TripletMarginLoss(margin=0.5),
        loss_function=NTXentLoss(temperature=0.07),
        lr=1e-3,
        schedule="Constant",
        log_batches_per_epoch=8,
        log_samples_per_batch=1,
        log_embeddings=False,
        example_input_array_shape=(1, 1, 1, 256, 256),
        **encoder_kwargs,
    ):
        if encoder is None:
            encoder = ContrastiveEncoder(**encoder_kwargs)

        super().__init__(
            encoder=encoder,
            loss_function=loss_function,
            lr=lr,
            schedule=schedule,
            log_batches_per_epoch=log_batches_per_epoch,
            log_samples_per_batch=log_samples_per_batch,
            log_embeddings=log_embeddings,
            example_input_array_shape=example_input_array_shape,
        )

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """Prediction step for extracting embeddings.

        Handles both contrastive dataset (has "anchor") and basic dataset (has "data").
        """
        # Handle both contrastive dataset (has "anchor") and basic dataset (has "data")
        if "anchor" in batch:
            input_data = batch["anchor"]
        else:
            input_data = batch["data"]

        features, projections = self.model(input_data)
        return {
            "features": features,
            "projections": projections,
            "total_index": batch["total_index"],
            "gene_label": batch["gene_label"],
        }
