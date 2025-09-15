from pathlib import Path

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
        """Prediction step for extracting embeddings."""
        features, projections = self.model(batch["anchor"])
        return {
            "features": features,
            "projections": projections,
            "total_index": batch["total_index"],
            "gene_label": batch["gene_label"],
        }
