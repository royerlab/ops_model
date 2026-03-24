import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import anndata as ad
import torch
import yaml
import zarr
import lightning as L
from lightning import seed_everything
from lightning.pytorch.callbacks import BasePredictionWriter
from pytorch_metric_learning.losses import NTXentLoss
from viscy.representation.contrastive import ContrastiveEncoder
from viscy.representation.engine import ContrastiveModule

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

        # Define all obs columns with their zarr dtype and encoding type.
        # String columns use "string-array", numeric columns use "array".
        self._obs_string_cols = [
            "perturbation",
            "experiment",
            "tile",
            "barcode",
            "sgRNA",
            "reporter",
            "channel",
            "well",
            "subpool",
            "dep_map_gene_name",
        ]
        self._obs_int_cols = ["gene_int", "NCBI_ID"]
        self._obs_float_cols = [
            "x_local",
            "y_local",
            "x_pheno",
            "y_pheno",
            "gene_effect",
            "segmentation_id",
        ]
        obs_group.attrs["column-order"] = (
            self._obs_string_cols + self._obs_int_cols + self._obs_float_cols
        )

        chunk_size = min(n_obs, 10000)

        obs_group.create_array(
            "_index",
            shape=(n_obs,),
            chunks=(chunk_size,),
            dtype="<U32",
            fill_value="",
        )
        obs_group["_index"].attrs["encoding-type"] = "string-array"
        obs_group["_index"].attrs["encoding-version"] = "0.2.0"

        for col in self._obs_string_cols:
            obs_group.create_array(
                col,
                shape=(n_obs,),
                chunks=(chunk_size,),
                dtype="<U128",
                fill_value="",
            )
            obs_group[col].attrs["encoding-type"] = "string-array"
            obs_group[col].attrs["encoding-version"] = "0.2.0"

        for col in self._obs_int_cols:
            obs_group.create_array(
                col,
                shape=(n_obs,),
                chunks=(chunk_size,),
                dtype="int64",
                fill_value=0,
            )
            obs_group[col].attrs["encoding-type"] = "array"
            obs_group[col].attrs["encoding-version"] = "0.2.0"

        for col in self._obs_float_cols:
            obs_group.create_array(
                col,
                shape=(n_obs,),
                chunks=(chunk_size,),
                dtype="float64",
                fill_value=np.nan,
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
        obs["gene_int"][start:end] = gene_labels

        # Mapping from zarr obs column name -> labels_df column name.
        # Columns not listed here use the same name in both.
        _col_map = {
            "perturbation": "gene_name",
            "experiment": "store_key",
            "tile": "tile_pheno",
            "x_local": "x_local_pheno",
            "y_local": "y_local_pheno",
        }

        rows = self.labels_df.loc[total_indices]

        for col in self._obs_string_cols:
            src_col = _col_map.get(col, col)
            if src_col in rows.columns:
                vals = rows[src_col].fillna("").values.astype(str)
            else:
                vals = np.full(bs, "", dtype="<U128")
            obs[col][start:end] = vals

        for col in self._obs_int_cols:
            if col == "gene_int":
                continue  # already written above from prediction
            src_col = _col_map.get(col, col)
            if src_col in rows.columns:
                obs[col][start:end] = rows[src_col].fillna(0).values.astype(np.int64)
            else:
                obs[col][start:end] = np.zeros(bs, dtype=np.int64)

        for col in self._obs_float_cols:
            src_col = _col_map.get(col, col)
            if src_col in rows.columns:
                obs[col][start:end] = rows[src_col].fillna(np.nan).values.astype(np.float64)
            else:
                obs[col][start:end] = np.full(bs, np.nan, dtype=np.float64)

        bbox_raw = rows["bbox"].values
        if isinstance(bbox_raw[0], str):
            bbox_arr = np.array(
                [[float(x) for x in b.strip("()").split(",")] for b in bbox_raw]
            )
        else:
            bbox_arr = np.asarray(bbox_raw.tolist(), dtype="float64")
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
        loss_function=NTXentLoss(temperature=0.07),
        lr=1e-3,
        schedule="WarmupCosine",
        warmup_epochs=10,
        log_batches_per_epoch=8,
        log_samples_per_batch=1,
        log_embeddings=False,
        example_input_array_shape=(1, 1, 1, 256, 256),
        **encoder_kwargs,
    ):
        if encoder is None:
            encoder = ContrastiveEncoder(**encoder_kwargs)

        self.warmup_epochs = warmup_epochs

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

    def _log_step_samples(self, batch_idx, samples, stage):
        if self.trainer.is_global_zero:
            super()._log_step_samples(batch_idx, samples, stage)

    def on_train_epoch_end(self):
        if self.trainer.is_global_zero:
            super().on_train_epoch_end()
        else:
            self.training_step_outputs.clear()

    def on_validation_epoch_end(self):
        if self.trainer.is_global_zero:
            super().on_validation_epoch_end()
        else:
            self.validation_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        if self.schedule == "WarmupCosine":
            from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

            warmup = LinearLR(
                optimizer, start_factor=0.01, total_iters=self.warmup_epochs
            )
            cosine = CosineAnnealingLR(
                optimizer,
                T_max=self.trainer.max_epochs - self.warmup_epochs,
            )
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup, cosine],
                milestones=[self.warmup_epochs],
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
            }
        return optimizer

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


def extract_dynaclr_features(config: dict):
    """
    Run DynaCLR inference and write embeddings to an AnnData zarr store.

    Mirrors the DINOv3 ``extract_dinov3_features`` interface so both models
    can be driven from the same config-based SLURM orchestration pattern.

    Parameters
    ----------
    config : dict
        Parsed YAML config dict.  The ``data_manager`` block may contain
        either ``experiments`` (dict mapping experiment name → wells, loaded
        via ``OpsDataManager.get_labels()``) or ``labels_df_path`` (path to a
        pre-filtered parquet/CSV).

    Returns
    -------
    Path
        Path to the generated AnnData zarr file.
    """
    from ops_model.data import data_loader
    from monai.transforms import CenterSpatialCropd, Compose, SpatialPadd, ToTensord

    seed_everything(42)

    dm_cfg = config["data_manager"]

    # ------------------------------------------------------------------
    # Build labels_df — support both DINOv3-style experiments dict and
    # the legacy labels_df_path approach.
    # ------------------------------------------------------------------
    if "labels_df_path" in dm_cfg:
        labels_path = dm_cfg["labels_df_path"]
        if labels_path.endswith(".parquet"):
            labels_df = pd.read_parquet(labels_path)
        else:
            labels_df = pd.read_csv(labels_path, low_memory=False)
        labels_df.set_index("total_index", inplace=True, drop=False)
        experiment_dict = {exp: [] for exp in labels_df["store_key"].unique().tolist()}
    else:
        experiment_dict = dm_cfg["experiments"]
        data_manager_tmp = data_loader.OpsDataManager(
            experiments=experiment_dict,
            batch_size=dm_cfg["batch_size"],
            data_split=tuple(dm_cfg.get("data_split", [0.0, 0.0, 1.0])),
            out_channels=dm_cfg.get("out_channels", None),
            initial_yx_patch_size=tuple(dm_cfg["initial_yx_patch_size"]),
            final_yx_patch_size=tuple(dm_cfg["final_yx_patch_size"]),
        )
        labels_df = data_manager_tmp.get_labels()

        # BasicDataset requires a 'channel' column. If out_channels is specified,
        # expand each cell row into one row per channel so all channels are embedded.
        out_channels = dm_cfg.get("out_channels")
        if out_channels and "channel" not in labels_df.columns:
            labels_df = pd.concat(
                [labels_df.assign(channel=ch) for ch in out_channels],
                ignore_index=True,
            )
            labels_df["total_index"] = np.arange(len(labels_df))

        labels_df.set_index("total_index", inplace=True, drop=False)

    # ------------------------------------------------------------------
    # Data manager + dataloader
    # ------------------------------------------------------------------
    data_manager = data_loader.OpsDataManager(
        experiments=experiment_dict,
        batch_size=dm_cfg["batch_size"],
        data_split=tuple(dm_cfg.get("data_split", [0.0, 0.0, 1.0])),
        out_channels=dm_cfg.get("out_channels", None),
        initial_yx_patch_size=tuple(dm_cfg["initial_yx_patch_size"]),
        final_yx_patch_size=tuple(dm_cfg["final_yx_patch_size"]),
    )

    basic_kwargs = {
        "cell_masks": dm_cfg.get("cell_masks", False),
        "transform": Compose(
            [
                SpatialPadd(
                    keys=["data", "mask"],
                    spatial_size=data_manager.initial_yx_patch_size,
                ),
                CenterSpatialCropd(
                    keys=["data", "mask"],
                    roi_size=data_manager.final_yx_patch_size,
                ),
                ToTensord(keys=["data", "mask"]),
            ]
        ),
    }

    data_manager.construct_dataloaders(
        labels_df=labels_df,
        num_workers=dm_cfg["num_workers"],
        pin_memory=dm_cfg.get("pin_memory", True),
        prefetch_factor=dm_cfg.get("prefetch_factor", 2),
        dataset_type="basic",
        basic_kwargs=basic_kwargs,
    )
    test_loader = data_manager.test_loader
    print(f"Created dataset with {len(test_loader.dataset)} crops")

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    model_config = config["model"].copy()
    encoder_config = model_config.pop("encoder")
    temperature = model_config.pop("temperature")

    lit_model = LitDynaClr.load_from_checkpoint(
        config["ckpt_path"],
        loss_function=NTXentLoss(temperature=temperature),
        **model_config,
        **encoder_config,
    )
    lit_model.eval()
    lit_model.freeze()
    print("DynaCLR model loaded")

    # ------------------------------------------------------------------
    # Prediction writer + trainer
    # ------------------------------------------------------------------
    prediction_cfg = config.get("prediction", {})
    embedding_type = f"{config['model_type']}_{config['run_name']}"

    writer = DynaClrAnnDataWriter(
        output_dir=prediction_cfg["output_dir"],
        run_name=config["run_name"],
        labels_df=labels_df,
        save_features=prediction_cfg.get("save_features", True),
        save_projections=prediction_cfg.get("save_projections", False),
        cell_type=config.get("cell_type"),
        embedding_type=embedding_type,
    )

    trainer_cfg = config.get("trainer", {})
    trainer = L.Trainer(
        accelerator=trainer_cfg.get("accelerator", "gpu"),
        devices=trainer_cfg.get("devices", 1),
        precision=trainer_cfg.get("precision", "32-true"),
        callbacks=[writer],
        logger=False,
    )

    print(f"Running prediction on {len(test_loader.dataset)} samples...")
    trainer.predict(lit_model, dataloaders=test_loader)

    output_path = (
        Path(prediction_cfg["output_dir"])
        / f"dynaclr_embeddings_{config['run_name']}.zarr"
    )
    print(f"\nPrediction complete! Output saved to:\n{output_path}")
    return output_path


def dynaclr_main(config_path: str):
    """
    Main orchestrator for DynaCLR feature extraction.

    Submits a SLURM job via submitit that calls ``extract_dynaclr_features``.
    Mirrors the ``dinov3_main`` pattern so both models share the same
    config-driven submission workflow.

    Parameters
    ----------
    config_path : str
        Path to YAML configuration file.

    Returns
    -------
    submitit.Job
        The submitted SLURM job object.
    """
    import submitit

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    experiments = list(
        config["data_manager"].get(
            "experiments", {"unknown": []}
        ).keys()
    )
    output_dir = Path(config["prediction"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    slurm_cfg = config.get("slurm", {})
    partition = slurm_cfg.get("partition", "gpu")
    gres = slurm_cfg.get("gres", "gpu:1")
    cpus_per_task = slurm_cfg.get("cpus_per_task", 20)
    mem = slurm_cfg.get("mem", "64G")
    time_limit = slurm_cfg.get("time", "6:00:00")
    constraint = slurm_cfg.get("constraint", "h100|h200|a100|a40")

    mem_gb = int(mem.rstrip("G")) if isinstance(mem, str) else int(mem)

    if isinstance(time_limit, int):
        timeout_min = time_limit
    elif isinstance(time_limit, str):
        parts = time_limit.split(":")
        if len(parts) == 3:
            timeout_min = int(parts[0]) * 60 + int(parts[1])
        elif len(parts) == 2:
            timeout_min = int(parts[0])
        else:
            timeout_min = 360
    else:
        timeout_min = 360

    exp_tag = experiments[0].split("_")[0] if experiments else "dynaclr"
    log_dir = Path("slurm_logs") / "slurm_dynaclr_inference" / (experiments[0] if experiments else "run")
    log_dir.mkdir(parents=True, exist_ok=True)

    executor = submitit.AutoExecutor(folder=log_dir)
    executor.update_parameters(
        timeout_min=timeout_min,
        slurm_partition=partition,
        slurm_gres=gres,
        cpus_per_task=cpus_per_task,
        mem_gb=mem_gb,
        slurm_constraint=constraint,
        slurm_job_name=f"dynaclr_{exp_tag}",
    )

    print(f"Submitting DynaCLR inference job for experiment(s): {experiments}")
    print(f"Output directory: {output_dir}")

    job = executor.submit(extract_dynaclr_features, config=config)
    print(f"Submitted job {job.job_id}")
    print("Check status with 'squeue -u $USER'")

    return job


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Extract DynaCLR features from OPS dataset based on config"
    )
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to the YAML config file",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    dynaclr_main(config_path=args.config_path)
