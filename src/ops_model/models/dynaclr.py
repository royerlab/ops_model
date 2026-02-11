from pathlib import Path

import numpy as np
import pandas as pd
import anndata as ad
import zarr
from lightning.pytorch.callbacks import BasePredictionWriter
from pytorch_metric_learning.losses import NTXentLoss
from viscy.representation.contrastive import ContrastiveEncoder
from viscy.representation.engine import ContrastiveModule

# FIXME: celltype should be added to the csv or metadata file instead of hardcoding it here


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
    Streams DynaCLR predictions batch-by-batch to an AnnData-compatible zarr store.

    Instead of accumulating all predictions in memory, this writer pre-allocates
    zarr arrays on disk and writes each batch directly in ``write_on_batch_end``.
    The resulting zarr store follows the AnnData zarr layout so it can be loaded
    with ``ad.read_zarr()``.

    Parameters
    ----------
    output_dir : str
        Directory to save the AnnData file.
    run_name : str
        Name for the output file.
    labels_df : pd.DataFrame
        DataFrame containing cell metadata, indexed by total_index.
    save_features : bool, default=True
        Save backbone features (embedding dimension).
    save_projections : bool, default=False
        Save projection head outputs.
    cell_type : str, optional
        Cell type metadata stored in ``adata.uns``.
    embedding_type : str, optional
        Embedding type metadata stored in ``adata.uns``.
    """

    def __init__(
        self,
        output_dir: str,
        run_name: str,
        labels_df: pd.DataFrame,
        save_features: bool = True,
        save_projections: bool = False,
        cell_type: str = None,
        embedding_type: str = None,
    ):
        super().__init__(write_interval="batch_and_epoch")
        self.output_dir = Path(output_dir)
        self.run_name = run_name
        self.labels_df = labels_df
        self.save_features = save_features
        self.save_projections = save_projections
        self.cell_type = cell_type
        self.embedding_type = embedding_type

        self.row_idx = 0
        self._zarr_initialized = False

    def setup(self, trainer, pl_module, stage):
        if stage != "predict":
            return

        n_obs = len(self.labels_df)
        self.n_obs = n_obs
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_path = self.output_dir / f"dynaclr_embeddings_{self.run_name}.zarr"

        self.root = zarr.open_group(str(self.output_path), mode="w")
        self.root.attrs["encoding-type"] = "anndata"
        self.root.attrs["encoding-version"] = "0.1.0"

        # --- obs (dataframe) ---
        obs_group = self.root.create_group("obs")
        obs_group.attrs["encoding-type"] = "dataframe"
        obs_group.attrs["encoding-version"] = "0.2.0"
        obs_group.attrs["_index"] = "_index"
        obs_group.attrs["column-order"] = [
            "perturbation",
            "gene_int",
            "experiment",
            "tile",
            "x_local",
            "y_local",
        ]

        obs_group.create_array(
            "_index",
            shape=(n_obs,),
            chunks=(min(n_obs, 10000),),
            dtype="<U32",
            fill_value="",
        )
        obs_group["_index"].attrs["encoding-type"] = "string-array"
        obs_group["_index"].attrs["encoding-version"] = "0.2.0"

        for col in ["perturbation", "experiment", "tile"]:
            obs_group.create_array(
                col,
                shape=(n_obs,),
                chunks=(min(n_obs, 10000),),
                dtype="<U128",
                fill_value="",
            )
            obs_group[col].attrs["encoding-type"] = "string-array"
            obs_group[col].attrs["encoding-version"] = "0.2.0"

        obs_group.create_array(
            "gene_int",
            shape=(n_obs,),
            chunks=(min(n_obs, 10000),),
            dtype="int64",
            fill_value=0,
        )
        obs_group["gene_int"].attrs["encoding-type"] = "array"
        obs_group["gene_int"].attrs["encoding-version"] = "0.2.0"

        for col in ["x_local", "y_local"]:
            obs_group.create_array(
                col,
                shape=(n_obs,),
                chunks=(min(n_obs, 10000),),
                dtype="float64",
                fill_value=0.0,
            )
            obs_group[col].attrs["encoding-type"] = "array"
            obs_group[col].attrs["encoding-version"] = "0.2.0"

        # --- obsm ---
        obsm_group = self.root.create_group("obsm")
        obsm_group.attrs["encoding-type"] = "dict"
        obsm_group.attrs["encoding-version"] = "0.1.0"

        bbox_sample = self.labels_df.iloc[0]["bbox"]
        if isinstance(bbox_sample, str):
            bbox_dim = len(bbox_sample.strip("()").split(","))
        elif hasattr(bbox_sample, "__len__"):
            bbox_dim = len(bbox_sample)
        else:
            bbox_dim = 4
        obsm_group.create_array(
            "bbox",
            shape=(n_obs, bbox_dim),
            chunks=(min(n_obs, 10000), bbox_dim),
            dtype="float64",
            fill_value=0.0,
        )
        obsm_group["bbox"].attrs["encoding-type"] = "array"
        obsm_group["bbox"].attrs["encoding-version"] = "0.2.0"

        # --- uns ---
        uns_group = self.root.create_group("uns")
        uns_group.attrs["encoding-type"] = "dict"
        uns_group.attrs["encoding-version"] = "0.1.0"
        if self.cell_type is not None:
            uns_group.create_array("cell_type", data=np.array(self.cell_type))
            uns_group["cell_type"].attrs["encoding-type"] = "string"
            uns_group["cell_type"].attrs["encoding-version"] = "0.2.0"
        if self.embedding_type is not None:
            uns_group.create_array("embedding_type", data=np.array(self.embedding_type))
            uns_group["embedding_type"].attrs["encoding-type"] = "string"
            uns_group["embedding_type"].attrs["encoding-version"] = "0.2.0"

        # Create empty groups that AnnData expects
        for key in ["var", "varm", "obsp", "varp", "layers"]:
            g = self.root.create_group(key)
            if key == "var":
                g.attrs["encoding-type"] = "dataframe"
                g.attrs["encoding-version"] = "0.2.0"
                g.attrs["_index"] = "_index"
                g.attrs["column-order"] = []
            else:
                g.attrs["encoding-type"] = "dict"
                g.attrs["encoding-version"] = "0.1.0"

    def _init_X_and_var(self, embedding_dim):
        """Create X array and var_names once embedding_dim is known from the first batch."""
        batch_chunk = min(self.n_obs, 10000)
        self.root.create_array(
            "X",
            shape=(self.n_obs, embedding_dim),
            chunks=(batch_chunk, embedding_dim),
            dtype="float32",
            fill_value=0.0,
        )
        self.root["X"].attrs["encoding-type"] = "array"
        self.root["X"].attrs["encoding-version"] = "0.2.0"

        var_names = np.array([f"feature_{i}" for i in range(embedding_dim)])
        self.root["var"].create_array("_index", data=var_names)
        self.root["var"]["_index"].attrs["encoding-type"] = "string-array"
        self.root["var"]["_index"].attrs["encoding-version"] = "0.2.0"

        self._zarr_initialized = True

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
        features = prediction["features"].detach().cpu().numpy()
        total_indices = prediction["total_index"].cpu().numpy()
        gene_labels = prediction["gene_label"].cpu().numpy()
        bs = features.shape[0]
        start = self.row_idx
        end = start + bs

        if not self._zarr_initialized:
            embedding_dim = features.shape[1]
            self._init_X_and_var(embedding_dim)

            if self.save_projections:
                proj_dim = prediction["projections"].shape[1]
                self.root["obsm"].create_array(
                    "X_projection",
                    shape=(self.n_obs, proj_dim),
                    chunks=(min(self.n_obs, 10000), proj_dim),
                    dtype="float32",
                    fill_value=0.0,
                )
                self.root["obsm"]["X_projection"].attrs["encoding-type"] = "array"
                self.root["obsm"]["X_projection"].attrs["encoding-version"] = "0.2.0"

        self.root["X"][start:end] = features

        obs = self.root["obs"]
        obs["_index"][start:end] = np.array([str(idx) for idx in total_indices])
        obs["perturbation"][start:end] = np.array(
            [self.labels_df.loc[idx, "gene_name"] for idx in total_indices]
        )
        obs["gene_int"][start:end] = gene_labels
        obs["experiment"][start:end] = np.array(
            [self.labels_df.loc[idx, "store_key"] for idx in total_indices]
        )
        obs["tile"][start:end] = np.array(
            [self.labels_df.loc[idx, "tile_pheno"] for idx in total_indices]
        )
        obs["x_local"][start:end] = np.array(
            [self.labels_df.loc[idx, "x_local_pheno"] for idx in total_indices],
            dtype="float64",
        )
        obs["y_local"][start:end] = np.array(
            [self.labels_df.loc[idx, "y_local_pheno"] for idx in total_indices],
            dtype="float64",
        )

        bbox_raw = [self.labels_df.loc[idx, "bbox"] for idx in total_indices]
        if isinstance(bbox_raw[0], str):
            bbox_arr = np.array(
                [[float(x) for x in b.strip("()").split(",")] for b in bbox_raw]
            )
        else:
            bbox_arr = np.asarray(bbox_raw, dtype="float64")
        self.root["obsm"]["bbox"][start:end] = bbox_arr

        if self.save_projections:
            projections = prediction["projections"].detach().cpu().numpy()
            self.root["obsm"]["X_projection"][start:end] = projections

        self.row_idx = end

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        if self.row_idx < self.n_obs:
            if "X" in self.root:
                self.root["X"].resize((self.row_idx, self.root["X"].shape[1]))
            for col in self.root["obs"]:
                self.root["obs"][col].resize((self.row_idx,))
            for key in self.root["obsm"]:
                self.root["obsm"][key].resize(
                    (self.row_idx, self.root["obsm"][key].shape[1])
                )

        zarr.consolidate_metadata(self.root.store)

        n_features = self.root["X"].shape[1] if "X" in self.root else 0
        obs_cols = list(self.root["obs"].attrs.get("column-order", []))
        obsm_keys = list(self.root["obsm"].array_keys())

        print(f"\n{'=' * 60}")
        print(f"Saved embeddings to: {self.output_path}")
        print(f"Shape: ({self.row_idx}, {n_features}) (cells x features)")
        print(f"AnnData structure:")
        print(f"  - adata.X: embeddings ({self.row_idx}, {n_features})")
        print(f"  - adata.obs: metadata columns = {obs_cols}")
        print(f"  - adata.obsm: {', '.join(obsm_keys)}")
        print(f"{'=' * 60}\n")


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
