import glob
from pathlib import Path

import anndata as ad
import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import torch
from lightning.pytorch.loggers import TensorBoardLogger
from tensorboard.backend.event_processing import event_accumulator
from torch import nn
from torch.utils.data import Dataset
from torchmetrics import Accuracy


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


class AggregatedClassifierDataset(Dataset):
    """
    Dataset for pre-aggregated (experiment x gene) embeddings.

    Parameters
    ----------
    embeddings : np.ndarray
        (n_samples, n_features) array of mean-pooled embeddings.
    labels : np.ndarray
        Integer class labels per sample.
    """

    def __init__(self, embeddings, labels):
        self.embeddings = embeddings.astype(np.float32)
        self.labels = labels.astype(np.int64)

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, index):
        return {
            "emb": torch.from_numpy(self.embeddings[index]),
            "label": torch.tensor(self.labels[index]),
        }


class LitBatchCorrectionClassifier(L.LightningModule):
    """
    Gene classifier with BatchNorm for implicit batch correction.

    The penultimate layer (128-dim) provides batch-corrected representations
    because BatchNorm normalizes activations across experiments.

    Parameters
    ----------
    num_classes : int
        Number of gene classes to predict.
    input_dim : int
        Dimensionality of input embeddings.
    hidden_dims : tuple of int
        Hidden layer sizes.
    dropout : float
        Dropout rate.
    lr : float
        Learning rate for AdamW.
    weight_decay : float
        Weight decay for AdamW.
    """

    def __init__(
        self,
        num_classes,
        input_dim=768,
        hidden_dims=(256, 128),
        dropout=0.3,
        lr=1e-3,
        weight_decay=1e-2,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden_dims[1], num_classes)

        self.loss = nn.CrossEntropyLoss()
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        features = self.feature_extractor(x)
        return self.classifier(features)

    def extract_features(self, x):
        """Return penultimate-layer (batch-corrected) representations."""
        self.eval()
        with torch.no_grad():
            return self.feature_extractor(x)

    def training_step(self, batch, batch_idx):
        x, y = batch["emb"], batch["label"]
        logits = self.forward(x)
        loss = self.loss(logits, y)
        self.train_acc(logits, y)
        self.log("loss/train", loss, batch_size=x.size(0), prog_bar=True)
        self.log("acc/train", self.train_acc, batch_size=x.size(0), prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["emb"], batch["label"]
        logits = self.forward(x)
        loss = self.loss(logits, y)
        self.val_acc(logits, y)
        self.log("loss/val", loss, batch_size=x.size(0), prog_bar=True)
        self.log("acc/val", self.val_acc, batch_size=x.size(0), prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )


def extract_corrected_embeddings(model, adata_agg):
    """
    Extract batch-corrected embeddings from the classifier's penultimate layer.

    Passes ALL (experiment x gene) samples through the feature extractor in
    eval mode, then re-aggregates to gene level by averaging across experiments.

    Parameters
    ----------
    model : LitBatchCorrectionClassifier
        Trained classifier.
    adata_agg : ad.AnnData
        (experiment x gene)-level AnnData with 'experiment' and 'perturbation'
        in obs.

    Returns
    -------
    adata_corrected_expgene : ad.AnnData
        (experiment x gene)-level corrected embeddings.
    adata_corrected_gene : ad.AnnData
        Gene-level corrected embeddings (mean across experiments).
    """
    model.eval()
    device = next(model.parameters()).device

    X = torch.from_numpy(np.asarray(adata_agg.X).astype(np.float32)).to(device)
    with torch.no_grad():
        features = model.feature_extractor(X).cpu().numpy()

    adata_corrected_expgene = ad.AnnData(X=features)
    adata_corrected_expgene.obs = adata_agg.obs.copy()
    adata_corrected_expgene.obs_names = adata_agg.obs_names.copy()

    perturbations = adata_corrected_expgene.obs["perturbation"].values
    unique_genes = np.unique(perturbations)
    gene_features = []
    gene_names = []
    for gene in unique_genes:
        mask = perturbations == gene
        gene_features.append(features[mask].mean(axis=0))
        gene_names.append(gene)

    adata_corrected_gene = ad.AnnData(X=np.array(gene_features))
    adata_corrected_gene.obs["perturbation"] = gene_names
    adata_corrected_gene.obs_names = gene_names

    return adata_corrected_expgene, adata_corrected_gene


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
