import glob
from pathlib import Path

import anndata as ad
import lightning as L
import matplotlib.pyplot as plt
import torch
from lightning.pytorch.loggers import TensorBoardLogger
from tensorboard.backend.event_processing import event_accumulator
from torch import nn
from torch.utils.data import Dataset


class MlpDataset(Dataset):
    def __init__(
        self,
        emb_path,
    ):

        adata = ad.io.read_zarr(emb_path)

        self.embeddings = adata.X
        self.labels = adata.obs["gene_int"]
        self.num_classes = self.labels.nunique()

    def __len__(self):

        return len(self.embeddings)

    def __getitem__(self, index):

        # TODO: add type conversions here, will also need to convert to tensors

        emb = self.embeddings[index, :]
        label = self.labels.iloc[index]

        batch = {"emb": torch.tensor(emb), "label": torch.tensor(label)}

        return batch


class LitMLP(L.LightningModule):
    def __init__(self, num_classes, len_embeddings, lr=1e-4):
        super().__init__()
        self.num_classes = num_classes
        self.len_embeddings = len_embeddings

        self.mlp = nn.Sequential(
            nn.Linear(self.len_embeddings, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, self.num_classes),
        )

        self.optimizer = torch.optim.Adam(self.mlp.parameters(), lr=lr)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.mlp(x)

    def training_step(self, batch, batch_idx):
        x = batch["emb"]
        y = batch["label"]
        logits = self.forward(x)
        train_loss = self.loss(logits, y)
        self.log("train_loss", train_loss, batch_size=x.size(0))
        return

    def validation_step(self, batch, batch_idx):
        x = batch["emb"]
        y = batch["label"]
        logits = self.forward(x)
        val_loss = self.loss(logits, y)
        self.log("train_loss", val_loss, batch_size=x.size(0))
        return

    def test_step(self, batch, batch_idx):
        x = batch["emb"]
        y = batch["label"]
        logits = self.forward(x)
        test_loss = self.loss(logits, y)
        self.log("train_loss", test_loss, batch_size=x.size(0))
        return

    def configure_optimizers(self):
        return self.optimizer


def eval_mlp(emb_path, num_workers):
    emb_path = Path(emb_path)

    # Initialization
    emb_dataset = MlpDataset(emb_path)

    data_loader = torch.utils.data.DataLoader(
        dataset=emb_dataset,
        batch_size=128,
        shuffle=True,
        num_workers=num_workers,
    )

    lit_model = LitMLP(
        num_classes=emb_dataset.num_classes,
        len_embeddings=emb_dataset.embeddings.shape[1],
        lr=1e-5,
    )
    logger = TensorBoardLogger(
        save_dir=emb_path.parent,
        name=f"{emb_path.parent.name}_eval_mlp",
    )

    # Train
    trainer = L.Trainer(max_epochs=100, accelerator="gpu", devices=1, logger=logger)
    trainer.fit(lit_model, data_loader, data_loader)

    # Plot train loss
    logdir_path = glob.glob(
        f"{emb_path.parent}/{emb_path.parent.name}_eval_mlp/*/events.out*"
    )
    ea = event_accumulator.EventAccumulator(logdir_path[0])
    ea.Reload()
    loss_events = ea.Scalars("train_loss")
    step = [e.step for e in loss_events]
    train_loss = [e.value for e in loss_events]
    plt.plot(step, train_loss)
    plt.xlabel("Step")
    plt.ylabel("Train Loss")
    plt.savefig(emb_path.parent / "mlp_training_loss.png")

    return
