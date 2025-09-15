from pathlib import Path

import lightning as L
import timm
import torch
import zarr
from byol_pytorch import BYOL
from byol_pytorch.byol_pytorch import RandomApply
from lightning.pytorch.callbacks import BasePredictionWriter
from torchvision import transforms as T


class BYOLPredictionWriter(BasePredictionWriter):
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
                self.output_dir / f"emb_{self.zarr_suffix}.zarr", mode="w"
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

        projection, embedding = prediction  # probably will need to fix model.forward

        self.emb_store[self.high_count].create_dataset(
            self.low_count,
            data=embedding.detach().cpu().numpy(),
            shape=embedding.shape,
            chunks=(1,) + embedding.shape[1:],
        )
        for i in range(embedding.shape[0]):
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


class LitBYOL(L.LightningModule):
    def __init__(
        self,
        image_size,
        projection_size=256,
        projection_hidden_size=4096,
        lr=0.001,
        in_channels=1,
    ):
        super().__init__()
        self.augmentation = torch.nn.Sequential(
            T.RandomResizedCrop(
                size=(100, 100), scale=(0.08, 1.0), ratio=(3 / 4, 4 / 3)
            ),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            RandomApply(T.GaussianBlur(kernel_size=7, sigma=(0.1, 2.0)), p=0.2),
        )
        self.in_chans = in_channels
        self.save_hyperparameters()
        self.model = BYOL(
            net=timm.models.resnet.resnet50(
                pretrained=False,
                in_chans=self.in_chans,  # kwargs passed directly to ResNet class
            ),
            in_chans=self.in_chans,
            image_size=image_size,
            hidden_layer=-2,
            augment_fn=self.augmentation,
            projection_size=projection_size,
            projection_hidden_size=projection_hidden_size,
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def forward(self, batch):
        x = batch["data"]
        return self.model(x, return_embedding=True)

    def training_step(self, batch, batch_idx):
        x = batch["data"]
        loss = self.model(x)
        self.log("train_loss", loss, batch_size=x.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch["data"]
        loss = self.model(x)
        self.log("val_loss", loss, batch_size=x.size(0))
        return loss

    def test_step(self, batch, batch_idx):
        x = batch["data"]
        loss = self.model(x)
        self.log("test_loss", loss, batch_size=x.size(0))
        return loss

    def configure_optimizers(self):
        return self.optimizer
