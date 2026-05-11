"""Classifier-based aggregator for OPS embeddings.

Replaces mean-pooling aggregation with a learned MLP that predicts perturbation
identity from multi-reporter embeddings. The penultimate-layer representations
become the new gene-level aggregated embeddings.

See classifier_aggregator_plan.md (in ops_model/eval/) for full design details.
"""

from __future__ import annotations

import math
import time
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.random import SeedSequence, default_rng
from torch.utils.data import DataLoader, Dataset

from ops_model.features.anndata_utils import DEFAULT_GUIDE_COL, _guide_col
from ops_model.post_process.anndata_processing.anndata_validator import AnndataValidator


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class CosineClassifier(nn.Module):
    """L2-normalised linear head with learnable temperature scale."""

    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        init_scale: float = 20.0,
        learn_scale: bool = True,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_classes, in_dim))
        nn.init.normal_(self.weight, std=0.01)
        if learn_scale:
            self.log_scale = nn.Parameter(torch.tensor(math.log(init_scale)))
        else:
            self.register_buffer("log_scale", torch.tensor(math.log(init_scale)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.normalize(x, dim=1)
        w = F.normalize(self.weight, dim=1)
        return torch.exp(self.log_scale) * (x @ w.t())


class MLP(nn.Module):
    """MLP classifier with a split backbone / head for penultimate-layer extraction.

    Parameters
    ----------
    input_dim:
        Dimensionality of the concatenated multi-reporter input.
    num_classes:
        Number of perturbation classes.
    hidden_dims:
        Width of each hidden layer.
    dropout:
        Dropout rate applied after each hidden layer.
    batch_norm:
        If ``True`` (default), include ``BatchNorm1d`` after each linear layer.
    cosine_classifier:
        If ``True`` (default), use a cosine-similarity head instead of a plain
        linear layer.

    Attributes
    ----------
    backbone : nn.Sequential
        All layers up to and including the last hidden block (linear + BN + ReLU
        + dropout). Running ``backbone(x)`` returns the penultimate representation.
    head : nn.Module
        The final classifier layer (``CosineClassifier`` or ``nn.Linear``).
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: tuple[int, ...] = (512, 512, 512),
        dropout: float = 0.4,
        batch_norm: bool = True,
        cosine_classifier: bool = True,
    ):
        super().__init__()

        backbone_layers: list[nn.Module] = []
        prev_dim = input_dim
        for h in hidden_dims:
            backbone_layers.append(nn.Linear(prev_dim, h))
            if batch_norm:
                backbone_layers.append(nn.BatchNorm1d(h))
            backbone_layers.append(nn.ReLU())
            backbone_layers.append(nn.Dropout(dropout))
            prev_dim = h

        self.backbone = nn.Sequential(*backbone_layers)
        self.head: nn.Module = (
            CosineClassifier(prev_dim, num_classes)
            if cosine_classifier
            else nn.Linear(prev_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------


def _topk_accuracy(logits: torch.Tensor, labels: torch.Tensor, k: int = 5) -> int:
    """Number of samples whose true label is in the top-k predictions."""
    k = min(k, logits.size(1))
    topk_preds = logits.topk(k, dim=1).indices
    return topk_preds.eq(labels.unsqueeze(1)).any(dim=1).sum().item()  # type: ignore[return-value]


@torch.no_grad()
def _evaluate(
    model: MLP,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float]:
    """Return ``(avg_loss, top1_acc, top5_acc)`` on the given loader."""
    model.eval()
    total_loss = 0.0
    correct_top1 = 0
    correct_top5 = 0
    total = 0
    for embeddings, labels in loader:
        embeddings, labels = embeddings.to(device), labels.to(device)
        logits = model(embeddings)
        loss = criterion(logits, labels)
        total_loss += loss.item() * len(labels)
        correct_top1 += (logits.argmax(dim=1) == labels).sum().item()
        correct_top5 += _topk_accuracy(logits, labels, k=5)
        total += len(labels)
    return total_loss / total, correct_top1 / total, correct_top5 / total


def _train_loop(
    model: MLP,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int,
    learning_rate: float,
    weight_decay: float,
    device: torch.device,
    wandb_run=None,
) -> None:
    """Standard AdamW train/eval loop with per-epoch console logging.

    Parameters
    ----------
    wandb_run:
        An active ``wandb.Run`` object (returned by ``wandb.init()``).  When
        provided, per-epoch metrics are logged to W&B in addition to stdout.
        Pass ``None`` (default) to disable W&B logging.
    """
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    criterion = nn.CrossEntropyLoss()

    header = (
        f"{'Epoch':>5} | {'Train Loss':>10} | {'Train Top1':>10} "
        f"| {'Train Top5':>10} | {'Val Loss':>10} | {'Val Top1':>10} "
        f"| {'Val Top5':>10} | {'Time':>6}"
    )
    print(header)
    print("-" * len(header))

    for epoch in range(1, num_epochs + 1):
        t_start = time.time()
        model.train()
        running_loss = 0.0
        running_top1 = 0
        running_top5 = 0
        running_total = 0

        for embeddings, labels in train_loader:
            embeddings, labels = embeddings.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(embeddings)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * len(labels)
            running_top1 += (logits.argmax(dim=1) == labels).sum().item()
            running_top5 += _topk_accuracy(logits, labels, k=5)
            running_total += len(labels)

        train_loss = running_loss / running_total
        train_acc = running_top1 / running_total
        train_acc5 = running_top5 / running_total

        val_loss, val_acc, val_acc5 = _evaluate(model, val_loader, criterion, device)

        elapsed = time.time() - t_start
        print(
            f"{epoch:5d} | {train_loss:10.4f} | {train_acc:10.4%} "
            f"| {train_acc5:10.4%} | {val_loss:10.4f} | {val_acc:10.4%} "
            f"| {val_acc5:10.4%} | {elapsed:5.1f}s"
        )

        if wandb_run is not None:
            wandb_run.log(
                {
                    "train/loss": train_loss,
                    "train/top1_acc": train_acc,
                    "train/top5_acc": train_acc5,
                    "val/loss": val_loss,
                    "val/top1_acc": val_acc,
                    "val/top5_acc": val_acc5,
                },
                step=epoch,
            )

    print("  Training complete.")


# ---------------------------------------------------------------------------
# View pre-computation
# ---------------------------------------------------------------------------


def _peek_embedding_dim(path: Path) -> int:
    """Return the embedding dimensionality of an h5ad file without loading it.

    Tries a direct HDF5 read first (fast); falls back to AnnData backed mode for
    sparse matrices.
    """
    with h5py.File(path, "r") as f:
        x_node = f.get("X")
        if x_node is None:
            raise ValueError(f"No 'X' dataset found in {path}")
        if isinstance(x_node, h5py.Dataset):
            return int(x_node.shape[1])
        # Sparse encoding: shape is stored as an attribute on the group
        shape = x_node.attrs.get("shape")
        if shape is not None:
            return int(shape[1])
    # Fallback: open with AnnData backed mode (reads only metadata)
    adata = ad.read_h5ad(path, backed="r")
    dim = adata.n_vars
    adata.file.close()
    return dim


def precompute_views(
    h5ad_paths: list[Path],
    reporter_for_path: list[str],
    reporters: list[str],
    perturbations: list[str],
    n_cells: int,
    n_views: int,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Pre-compute averaged cell views for every (perturbation, reporter) pair.

    Reads each h5ad file exactly once and discards raw cell embeddings immediately
    after processing, keeping peak memory bounded to roughly one file at a time
    plus the growing views array.

    For each reporter, ``n_views`` is split evenly across the files that contribute
    to it.  Each file computes its budget of views by sampling ``n_cells`` cells
    uniformly at random and averaging them, independently for each view.

    Parameters
    ----------
    h5ad_paths:
        One path per (experiment, channel) pair, in processing order.
    reporter_for_path:
        Parallel list mapping each path to its biological signal / reporter name.
    reporters:
        Ordered list of unique reporter names. Defines axis-1 of the output.
    perturbations:
        Ordered list of unique perturbation names. Defines axis-0 of the output.
    n_cells:
        Number of cells averaged per view.
    n_views:
        Views to pre-compute per reporter per perturbation.  Each reporter
        gets this many views independently; if multiple files contribute to
        one reporter their budgets are summed to reach this total.
    seed:
        Base random seed for reproducibility.

    Returns
    -------
    views : np.ndarray, shape (n_perturbations, n_reporters, n_views, embedding_dim)
        Pre-computed averaged embeddings.  Unwritten slots (perturbation absent
        from a file) are zero and excluded by ``n_valid``.
    n_valid : np.ndarray, shape (n_perturbations, n_reporters), dtype int32
        Number of valid (written) views per (perturbation, reporter) slot.
        Always in ``[0, n_views]``.
    """
    n_p = len(perturbations)
    n_r = len(reporters)
    pert_to_idx = {p: i for i, p in enumerate(perturbations)}
    rep_to_idx = {r: i for i, r in enumerate(reporters)}

    # --- Per-reporter file lists and view budgets ----------------------------
    # reporter_files[r_idx] = list of (file_idx, budget) in processing order
    reporter_file_lists: list[list[int]] = [[] for _ in range(n_r)]
    for file_idx, reporter in enumerate(reporter_for_path):
        r_idx = rep_to_idx[reporter]
        reporter_file_lists[r_idx].append(file_idx)

    # Split n_views evenly across files per reporter; earlier files get +1 if
    # n_views is not divisible by n_files.
    file_view_budget: list[int] = [0] * len(h5ad_paths)
    for r_idx, file_indices in enumerate(reporter_file_lists):
        if not file_indices:
            continue
        splits = np.array_split(range(n_views), len(file_indices))
        for file_idx, split in zip(file_indices, splits):
            file_view_budget[file_idx] = len(split)

    # --- Discover embedding dim from first file ------------------------------
    embedding_dim = _peek_embedding_dim(h5ad_paths[0])
    print(
        f"\n  precompute_views: {n_p} perturbations × {n_r} reporters × "
        f"{n_views} views, embedding_dim={embedding_dim}"
    )
    print(
        f"  Allocating views array: "
        f"{n_p * n_r * n_views * embedding_dim * 4 / 1e9:.2f} GB"
    )

    views = np.zeros((n_p, n_r, n_views, embedding_dim), dtype=np.float32)
    n_valid = np.zeros((n_p, n_r), dtype=np.int32)

    # --- Independent RNG per file via SeedSequence --------------------------
    ss = SeedSequence(seed)
    file_rngs = [default_rng(child) for child in ss.spawn(len(h5ad_paths))]

    # --- Process files -------------------------------------------------------
    for file_idx, path in enumerate(h5ad_paths):
        reporter = reporter_for_path[file_idx]
        r_idx = rep_to_idx[reporter]
        budget = file_view_budget[file_idx]
        rng = file_rngs[file_idx]

        print(
            f"  [{file_idx + 1}/{len(h5ad_paths)}] {path.name}"
            f"  reporter={reporter!r}  budget={budget} views"
        )

        adata = ad.read_h5ad(path)
        X = adata.X
        if hasattr(X, "toarray"):
            X = X.toarray()
        X = np.asarray(X, dtype=np.float32)
        obs_perturbations: np.ndarray = adata.obs["perturbation"].to_numpy()
        del adata  # free raw cells immediately

        # Group row indices by perturbation
        cells_by_pert: dict[int, np.ndarray] = {}
        for row_idx, pert in enumerate(obs_perturbations):
            p_idx = pert_to_idx.get(pert)
            if p_idx is None:
                continue  # perturbation not in our master list — skip
            if p_idx not in cells_by_pert:
                cells_by_pert[p_idx] = []
            cells_by_pert[p_idx].append(row_idx)

        n_written_this_file = 0
        for p_idx, row_list in cells_by_pert.items():
            already = int(n_valid[p_idx, r_idx])
            n_to_write = min(budget, n_views - already)
            if n_to_write <= 0:
                continue  # view budget for this (perturbation, reporter) is full

            cell_indices = np.array(row_list, dtype=np.int32)
            n_available = len(cell_indices)
            replace = n_available < n_cells
            if replace:
                warnings.warn(
                    f"precompute_views: perturbation {perturbations[p_idx]!r}, "
                    f"reporter {reporter!r} has only {n_available} cells "
                    f"(need {n_cells}). Sampling with replacement.",
                    stacklevel=2,
                )

            for k in range(n_to_write):
                sampled = rng.choice(cell_indices, size=n_cells, replace=replace)
                views[p_idx, r_idx, already + k] = X[sampled].mean(axis=0)

            n_valid[p_idx, r_idx] += n_to_write
            n_written_this_file += n_to_write

        n_perts_this_file = len(cells_by_pert)
        del X, obs_perturbations, cells_by_pert
        print(
            f"    wrote {n_written_this_file:,} views across {n_perts_this_file} perturbations"
        )

    # --- Summary -------------------------------------------------------------
    mean_valid = float(n_valid.mean())
    min_valid = int(n_valid.min())
    missing = int((n_valid == 0).sum())
    print(
        f"\n  precompute_views complete."
        f"  mean valid views/slot={mean_valid:.1f}"
        f"  min={min_valid}"
        f"  empty slots={(missing)}"
    )

    return views, n_valid


def precompute_inference_views(
    h5ad_paths: list[Path],
    reporter_for_path: list[str],
    reporters: list[str],
    perturbations: list[str],
    n_cells: int,
    seed: int = 0,
    obs_col: str = "perturbation",
) -> tuple[list[np.ndarray], np.ndarray]:
    """Pre-compute non-overlapping averaged cell views for inference.

    Unlike :func:`precompute_views`, this function uses **all available cells**
    with no per-slot view-count cap.  For each reporter, all contributing files
    are pooled into a single cell matrix before views are created, so cells from
    different files for the same ``(perturbation, reporter)`` slot are combined
    into a shared pool.  The pool is shuffled once per slot and sliced into
    consecutive non-overlapping chunks of ``n_cells``.

    If a slot has fewer than ``n_cells`` cells across all files, all available
    cells are averaged into one view.

    Each reporter receives its own views array sized to its own maximum view
    count, avoiding the large wasted padding that would result from a global
    maximum when reporters have very different cell counts.

    Parameters
    ----------
    h5ad_paths, reporter_for_path, reporters, perturbations:
        Same semantics as :func:`precompute_views`.
    n_cells:
        Number of cells averaged per view.
    seed:
        Base random seed for cell-shuffle reproducibility.
    obs_col:
        The obs column used to identify entities (perturbations or guides).
        Defaults to ``"perturbation"`` with a ``"label_str"`` fallback for
        backward compatibility.  Pass ``"sgRNA"`` for guide-level views.

    Returns
    -------
    views : list[np.ndarray]
        One array per reporter, each shape
        ``(n_perturbations, max_views_r, embedding_dim)`` where
        ``max_views_r`` is the maximum view count for that reporter.
        Unwritten slots are zero; use ``n_valid`` to determine the valid prefix.
    n_valid : np.ndarray, shape (n_perturbations, n_reporters), dtype int32
        Number of valid views per slot.
    cell_counts : np.ndarray, shape (n_perturbations, n_reporters), dtype int64
        Total cell count per (perturbation, reporter) slot, from Phase 1.
    """
    n_p = len(perturbations)
    n_r = len(reporters)
    pert_to_idx = {p: i for i, p in enumerate(perturbations)}
    rep_to_idx = {r: i for i, r in enumerate(reporters)}

    # --- Phase 1: backed-mode scan for global cell counts per slot -----------
    print(
        "\n  precompute_inference_views — Phase 1: counting cells per "
        "(perturbation, reporter)..."
    )
    cell_counts = np.zeros((n_p, n_r), dtype=np.int64)
    for file_idx, path in enumerate(h5ad_paths):
        reporter = reporter_for_path[file_idx]
        r_idx = rep_to_idx[reporter]
        adata = ad.read_h5ad(path, backed="r")
        try:
            col = (
                ("perturbation" if "perturbation" in adata.obs.columns else "label_str")
                if obs_col == "perturbation"
                else obs_col
            )
            for pert, count in adata.obs[col].value_counts().items():
                p_idx = pert_to_idx.get(pert)
                if p_idx is not None:
                    cell_counts[p_idx, r_idx] += count
        finally:
            adata.file.close()

    # --- Compute view count per slot from global cell counts ----------------
    # Slots with 0 cells → 0 views
    # Slots with 1..n_cells-1 cells → 1 view (all cells; edge case)
    # Slots with ≥ n_cells cells → floor(count / n_cells) views
    n_views_per_slot = np.zeros((n_p, n_r), dtype=np.int32)
    mask_some = cell_counts > 0
    mask_full = cell_counts >= n_cells
    n_views_per_slot[mask_some & ~mask_full] = 1
    n_views_per_slot[mask_full] = (cell_counts[mask_full] // n_cells).astype(np.int32)

    # --- Report entities whose global cell count is below n_cells ----------
    for r_idx, reporter in enumerate(reporters):
        n_total = int(mask_some[:, r_idx].sum())
        n_few = int((mask_some[:, r_idx] & ~mask_full[:, r_idx]).sum())
        if n_few > 0:
            print(
                f"  {reporter}: {n_few} / {n_total} guides had < {n_cells} cells,"
                f" using all available as 1 view"
            )

    # Per-reporter max view count — avoids padding all reporters to a global max
    max_views_per_reporter = [
        (
            int(n_views_per_slot[:, r_idx].max())
            if n_views_per_slot[:, r_idx].max() > 0
            else 1
        )
        for r_idx in range(n_r)
    ]

    # --- Discover embedding dim and allocate per-reporter views arrays ------
    embedding_dim = _peek_embedding_dim(h5ad_paths[0])
    total_gb = sum(n_p * mv * embedding_dim * 4 for mv in max_views_per_reporter) / 1e9
    print(
        f"  precompute_inference_views: {n_p} perturbations × {n_r} reporters, "
        f"embedding_dim={embedding_dim}, total allocation={total_gb:.2f} GB"
    )
    for r_idx, reporter in enumerate(reporters):
        mv = max_views_per_reporter[r_idx]
        gb = n_p * mv * embedding_dim * 4 / 1e9
        print(f"    {reporter}: up to {mv} views  ({gb:.2f} GB)")

    views: list[np.ndarray] = [
        np.zeros((n_p, max_views_per_reporter[r_idx], embedding_dim), dtype=np.float32)
        for r_idx in range(n_r)
    ]
    n_written = np.zeros((n_p, n_r), dtype=np.int32)

    # --- Build reporter → file paths mapping --------------------------------
    reporter_to_paths: dict[str, list[Path]] = {r: [] for r in reporters}
    for path, reporter in zip(h5ad_paths, reporter_for_path):
        reporter_to_paths[reporter].append(path)

    # --- Independent RNG per (perturbation, reporter) slot ------------------
    ss = SeedSequence(seed)
    slot_rngs = default_rng(ss).integers(0, 2**31, size=(n_p, n_r), dtype=np.int64)

    # --- Phase 2: pool all files per reporter, then create views ------------
    print("  Phase 2: building non-overlapping inference views...")
    for r_idx, reporter in enumerate(reporters):
        paths = reporter_to_paths[reporter]
        if not paths:
            continue

        print(f"  reporter={reporter!r}  ({len(paths)} file(s))")

        # Pool cells from all files for this reporter
        adata_list = [ad.read_h5ad(p) for p in paths]
        adata_pooled = ad.concat(adata_list, join="inner", merge="same")
        del adata_list

        X = adata_pooled.X
        if hasattr(X, "toarray"):
            X = X.toarray()
        X = np.asarray(X, dtype=np.float32)
        col = (
            (
                "perturbation"
                if "perturbation" in adata_pooled.obs.columns
                else "label_str"
            )
            if obs_col == "perturbation"
            else obs_col
        )
        obs_labels = adata_pooled.obs[col].to_numpy()
        del adata_pooled

        # Group row indices by entity (perturbation or guide)
        cells_by_pert: dict[int, list[int]] = {}
        for row_idx, label in enumerate(obs_labels):
            p_idx = pert_to_idx.get(label)
            if p_idx is None:
                continue
            if p_idx not in cells_by_pert:
                cells_by_pert[p_idx] = []
            cells_by_pert[p_idx].append(row_idx)

        n_written_this_reporter = 0
        for p_idx, row_list in cells_by_pert.items():
            n_views = int(n_views_per_slot[p_idx, r_idx])
            if n_views == 0:
                continue

            cell_indices = np.array(row_list, dtype=np.int32)
            rng = default_rng(int(slot_rngs[p_idx, r_idx]))
            rng.shuffle(cell_indices)

            if len(cell_indices) < n_cells:
                # Fewer cells than n_cells globally — use all as 1 view
                views[r_idx][p_idx, 0] = X[cell_indices].mean(axis=0)
                n_written[p_idx, r_idx] = 1
                n_written_this_reporter += 1
            else:
                for k in range(n_views):
                    chunk = cell_indices[k * n_cells : (k + 1) * n_cells]
                    views[r_idx][p_idx, k] = X[chunk].mean(axis=0)
                n_written[p_idx, r_idx] = n_views
                n_written_this_reporter += n_views

        del X, obs_labels, cells_by_pert
        print(f"    wrote {n_written_this_reporter:,} views")

    n_valid = n_written
    mean_valid = float(n_valid.mean())
    min_valid = int(n_valid.min())
    missing = int((n_valid == 0).sum())
    print(
        f"\n  precompute_inference_views complete."
        f"  mean valid views/slot={mean_valid:.1f}"
        f"  min={min_valid}"
        f"  empty slots={missing}"
    )
    return views, n_valid, cell_counts


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class ClassifierAggregatorDataset(Dataset):
    """PyTorch Dataset over pre-computed per-(perturbation, reporter) views.

    Each item is a concatenation of one embedding vector per reporter,
    paired with an integer perturbation label.

    Parameters
    ----------
    views:
        Shape ``(n_perturbations, n_reporters, n_views, embedding_dim)``.
        Output of :func:`precompute_views`.
    n_valid:
        Shape ``(n_perturbations, n_reporters)``, dtype int32.
        Number of valid (written) views per slot; sampling is restricted to
        ``views[i, r, :n_valid[i, r], :]``.
    labels:
        Integer-encoded perturbation labels, shape ``(n_perturbations,)``.
    inference:
        If ``False`` (default, training mode), each ``__getitem__`` call
        independently samples one view index per reporter from the valid
        views, giving combinatorial augmentation across reporters.
        If ``True`` (inference mode), the valid views for each reporter are
        averaged deterministically before concatenation.
    """

    def __init__(
        self,
        views: np.ndarray,
        n_valid: np.ndarray,
        labels: np.ndarray,
        inference: bool = False,
    ):
        n_perturbations, n_reporters, n_views, embedding_dim = views.shape
        assert n_valid.shape == (n_perturbations, n_reporters)
        assert labels.shape == (n_perturbations,)

        self.views = views
        self.n_valid = n_valid
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.inference = inference
        self.n_reporters = n_reporters
        self.embedding_dim = embedding_dim

        empty = int((n_valid == 0).sum())
        if empty:
            warnings.warn(
                f"ClassifierAggregatorDataset: {empty} (perturbation, reporter) "
                f"slots have no valid views and will produce zero embeddings.",
                stacklevel=2,
            )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        parts: list[np.ndarray] = []

        for r in range(self.n_reporters):
            valid = int(self.n_valid[i, r])
            if valid == 0:
                parts.append(np.zeros(self.embedding_dim, dtype=np.float32))
            elif self.inference:
                parts.append(self.views[i, r, :valid].mean(axis=0))
            else:
                j = np.random.randint(0, valid)
                parts.append(self.views[i, r, j])

        embedding = torch.tensor(np.concatenate(parts), dtype=torch.float32)
        return embedding, self.labels[i]


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------


class ClassifierAggregator:
    """Train an MLP classifier on pre-computed multi-reporter views and extract
    penultimate-layer representations as gene-level aggregated embeddings.

    Parameters
    ----------
    hidden_dims:
        Width of each MLP hidden layer.
    dropout:
        Dropout rate applied after each hidden layer.
    cosine_classifier:
        Use a cosine-similarity head instead of a plain linear layer.
    batch_size:
        DataLoader batch size for both training and inference.
    num_epochs:
        Number of training epochs.
    learning_rate:
        AdamW learning rate.
    weight_decay:
        AdamW weight decay.
    val_fraction:
        Fraction of perturbations held out for validation (group split).
    seed:
        Random seed for train/val split and DataLoader worker init.

    Notes
    -----
    The MLP is instantiated during :meth:`fit` once the embedding
    dimensionality is known from the pre-computed views.
    """

    def __init__(
        self,
        hidden_dims: tuple[int, ...] = (512, 512, 512),
        dropout: float = 0.4,
        cosine_classifier: bool = True,
        batch_size: int = 256,
        num_epochs: int = 50,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        val_fraction: float = 0.2,
        seed: int = 42,
    ):
        self.hidden_dims = tuple(hidden_dims)
        self.dropout = dropout
        self.cosine_classifier = cosine_classifier
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.val_fraction = val_fraction
        self.seed = seed

        # Set after fit()
        self.model: MLP | None = None
        self.views: np.ndarray | None = None
        self.n_valid: np.ndarray | None = None
        self.perturbations: list[str] | None = None
        self.reporters: list[str] | None = None
        self._labels: np.ndarray | None = None
        self._cell_type: str = "cell"
        self._embedding_type: str = ""
        self._guide_col: str = DEFAULT_GUIDE_COL
        self._h5ad_paths: list[Path] | None = None
        self._reporter_for_path: list[str] | None = None
        self._n_cells_per_view: int | None = None
        self._n_experiments: int = 0

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        h5ad_paths: list[Path],
        reporter_for_path: list[str],
        reporters: list[str],
        perturbations: list[str],
        n_cells: int,
        n_views: int,
        device: torch.device,
        wandb_project: str | None = None,
        wandb_run_name: str | None = None,
    ) -> None:
        """Pre-compute views, train the MLP, and store state for transform.

        Parameters
        ----------
        h5ad_paths:
            One path per (experiment, channel) pair.
        reporter_for_path:
            Parallel list mapping each path to its reporter name.
        reporters:
            Ordered list of unique reporter names.
        perturbations:
            Ordered list of unique perturbation names (classification targets).
        n_cells:
            Cells averaged per view during pre-computation.
        n_views:
            Total views pre-computed per (perturbation, reporter).
        device:
            Torch device for MLP training.
        wandb_project:
            W&B project name.  When provided, a run is initialised with
            ``wandb.init()`` before training and finished afterwards.  Pass
            ``None`` (default) to disable W&B logging.
        wandb_run_name:
            Optional display name for the W&B run.  Ignored when
            ``wandb_project`` is ``None``.
        """
        # --- Pre-compute views ---
        views, n_valid = precompute_views(
            h5ad_paths=h5ad_paths,
            reporter_for_path=reporter_for_path,
            reporters=reporters,
            perturbations=perturbations,
            n_cells=n_cells,
            n_views=n_views,
            seed=self.seed,
        )

        n_perturbations, n_reporters, _, embedding_dim = views.shape
        input_dim = embedding_dim * n_reporters
        num_classes = n_perturbations
        labels = np.arange(n_perturbations, dtype=np.int64)

        # --- Train / val split (by view index, not perturbation) ---
        # All perturbations appear in both sets; val uses held-out views so the
        # model cannot be evaluated on classes it was never trained on.
        n_train_views = max(1, n_views - max(1, int(n_views * self.val_fraction)))
        n_val_views = n_views - n_train_views

        train_views = views[:, :, :n_train_views, :]
        val_views = views[:, :, n_train_views:, :]
        train_n_valid = np.minimum(n_valid, n_train_views)
        val_n_valid = np.maximum(n_valid - n_train_views, 0).astype(np.int32)

        print(
            f"\n  ClassifierAggregator.fit: {n_perturbations} perturbations, "
            f"{n_reporters} reporters, input_dim={input_dim}, "
            f"num_classes={num_classes}"
        )
        print(
            f"  Split: {n_train_views} train / {n_val_views} val views per (perturbation, reporter)"
        )

        train_ds = ClassifierAggregatorDataset(
            train_views, train_n_valid, labels, inference=False
        )
        val_ds = ClassifierAggregatorDataset(
            val_views, val_n_valid, labels, inference=True
        )

        train_loader = DataLoader(
            train_ds, batch_size=self.batch_size, shuffle=True, num_workers=0
        )
        val_loader = DataLoader(
            val_ds, batch_size=self.batch_size, shuffle=False, num_workers=0
        )

        # --- Instantiate and train MLP ---
        model = MLP(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
            cosine_classifier=self.cosine_classifier,
        ).to(device)

        total_params = sum(p.numel() for p in model.parameters())
        print(f"  MLP: {total_params:,} params  hidden_dims={self.hidden_dims}")
        print(model)

        # --- W&B setup ---
        wandb_run = None
        if wandb_project is not None:
            import wandb

            wandb_run = wandb.init(
                project=wandb_project,
                name=wandb_run_name,
                config={
                    "hidden_dims": self.hidden_dims,
                    "dropout": self.dropout,
                    "cosine_classifier": self.cosine_classifier,
                    "num_epochs": self.num_epochs,
                    "learning_rate": self.learning_rate,
                    "weight_decay": self.weight_decay,
                    "batch_size": self.batch_size,
                    "val_fraction": self.val_fraction,
                    "seed": self.seed,
                    "n_cells_per_view": n_cells,
                    "n_views": n_views,
                    "n_perturbations": n_perturbations,
                    "n_reporters": n_reporters,
                    "input_dim": input_dim,
                    "num_classes": num_classes,
                    "reporters": reporters,
                },
            )

        try:
            _train_loop(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                num_epochs=self.num_epochs,
                learning_rate=self.learning_rate,
                weight_decay=self.weight_decay,
                device=device,
                wandb_run=wandb_run,
            )
        finally:
            if wandb_run is not None:
                wandb_run.finish()

        # --- Infer uns metadata from first h5ad ---
        _first = ad.read_h5ad(h5ad_paths[0], backed="r")
        self._cell_type: str = _first.uns.get("cell_type", "cell")
        self._embedding_type: str = _first.uns.get("embedding_type", "")
        self._guide_col: str = _guide_col(_first)
        _first.file.close()

        # --- Store state ---
        self.model = model.cpu()
        self.views = views
        self.n_valid = n_valid
        self.perturbations = list(perturbations)
        self.reporters = list(reporters)
        self._labels = labels
        self._h5ad_paths = list(h5ad_paths)
        self._reporter_for_path = list(reporter_for_path)
        self._n_cells_per_view = n_cells
        self._n_experiments = max(Counter(reporter_for_path).values())

    # ------------------------------------------------------------------
    # Transform
    # ------------------------------------------------------------------

    def _tta_loop(
        self,
        inference_views: list[np.ndarray],
        inference_n_valid: np.ndarray,
        n_passes: int,
        device: torch.device,
    ) -> np.ndarray:
        """Run TTA inference and return averaged backbone embeddings.

        Parameters
        ----------
        inference_views:
            List of ``n_reporters`` arrays, each shape
            ``(n_entities, max_views_r, embedding_dim)``.  Output of
            :func:`precompute_inference_views`.
        inference_n_valid:
            Shape ``(n_entities, n_reporters)``, dtype int32.
        n_passes:
            Number of random-pairing passes to accumulate.
        device:
            Torch device for forward passes.

        Returns
        -------
        np.ndarray, shape ``(n_entities, last_hidden_dim)``, dtype float32
        """
        assert self.model is not None
        n_e, n_r = inference_n_valid.shape
        embedding_dim = inference_views[0].shape[-1]
        backbone_out_dim = self.hidden_dims[-1]

        rep_sum = np.zeros((n_e, backbone_out_dim), dtype=np.float64)
        rng = np.random.default_rng(self.seed)

        for _ in range(n_passes):
            inputs = np.zeros((n_e, n_r * embedding_dim), dtype=np.float32)
            for r in range(n_r):
                valid_counts = inference_n_valid[:, r]
                has_valid = valid_counts > 0
                j_r = rng.integers(0, np.maximum(valid_counts, 1), size=n_e)
                gathered = inference_views[r][np.arange(n_e), j_r, :].copy()
                gathered[~has_valid] = 0.0
                inputs[:, r * embedding_dim : (r + 1) * embedding_dim] = gathered

            for start in range(0, n_e, self.batch_size):
                batch = torch.tensor(inputs[start : start + self.batch_size]).to(device)
                rep_sum[start : start + self.batch_size] += (
                    self.model.backbone(batch).cpu().numpy()
                )

        return (rep_sum / n_passes).astype(np.float32)

    def _discover_guides(self) -> tuple[list[str], dict[str, str]]:
        """Backed-mode scan to find all guide IDs and their gene mappings.

        Reads ``obs[self._guide_col]`` (the per-construct identifier column —
        ``"sgRNA"`` for CRISPR, ``"minibinder_perturbation"`` for minibinder,
        resolved from the first h5ad's ``uns["guide_col"]``) and
        ``obs["perturbation"]`` (or ``"label_str"``) from every h5ad file. Each
        guide always maps to the same gene, so the relationship is collected
        into a ``guide → gene`` dict.

        Returns
        -------
        guides : list[str]
            Sorted list of unique guide IDs.
        guide_to_gene : dict[str, str]
            Maps each guide ID to its gene / perturbation label.
        """
        assert self._h5ad_paths is not None
        guide_to_gene: dict[str, str] = {}
        for path in self._h5ad_paths:
            adata = ad.read_h5ad(path, backed="r")
            try:
                gene_col = (
                    "perturbation"
                    if "perturbation" in adata.obs.columns
                    else "label_str"
                )
                mapping = adata.obs.groupby(self._guide_col, observed=True)[
                    gene_col
                ].first()
                guide_to_gene.update(mapping.to_dict())
            finally:
                adata.file.close()
        guides = sorted(guide_to_gene.keys())
        return guides, guide_to_gene

    def _make_adata(
        self,
        X: np.ndarray,
        entities: list[str],
    ) -> ad.AnnData:
        """Build a gene- or guide-level AnnData with standard uns fields."""
        assert self.reporters is not None
        obs = pd.DataFrame({"perturbation": entities}, index=entities)
        adata = ad.AnnData(X=X, obs=obs)
        adata.uns["aggregation_method"] = "classifier"
        adata.uns["reporters"] = self.reporters
        adata.uns["cell_type"] = self._cell_type
        adata.uns["embedding_type"] = self._embedding_type
        adata.uns["guide_col"] = self._guide_col
        return adata

    @torch.no_grad()
    def transform(
        self,
        device: torch.device,
        n_passes: int = 100,
    ) -> tuple[ad.AnnData, ad.AnnData]:
        """Extract penultimate-layer representations using TTA-style random pairing.

        For each of ``n_passes`` forward passes, each reporter independently
        samples one view at random from all available inference views; backbone
        outputs are averaged across passes.  Inference views are computed from
        **all available cells** using non-overlapping chunks of
        ``n_cells_per_view`` — no training view-count cap is applied.

        Requires a per-construct identifier column (named via
        ``adata.uns["guide_col"]``, default ``"sgRNA"``) in the h5ad obs.
        Inference is run at the **guide level** (one embedding per guide) and
        gene-level embeddings are derived by averaging guide embeddings with
        equal weight per guide.

        Parameters
        ----------
        device:
            Torch device for the forward pass.
        n_passes:
            Number of random-pairing passes to average.

        Returns
        -------
        guide_adata : ad.AnnData
            Guide-level AnnData ``(n_guides, last_hidden_dim)``.
        gene_adata : ad.AnnData
            Gene-level AnnData ``(n_perturbations, last_hidden_dim)``.

        Raises
        ------
        ValueError
            If the configured guide column is not present in the h5ad obs.
        """
        if self.model is None:
            raise RuntimeError("Call fit() before transform().")
        assert self.perturbations is not None
        assert self.reporters is not None
        assert self._h5ad_paths is not None
        assert self._reporter_for_path is not None
        assert self._n_cells_per_view is not None

        self.model.eval()
        self.model.to(device)

        # --- Require guide-level obs column ---
        _first = ad.read_h5ad(self._h5ad_paths[0], backed="r")
        has_guide_col = self._guide_col in _first.obs.columns
        _first.file.close()

        if not has_guide_col:
            raise ValueError(
                f"No {self._guide_col!r} column found in obs. Guide-level data "
                "is required for transform(). Ensure h5ad files have an "
                f"{self._guide_col!r} obs column."
            )

        # --- Guide-level inference ---
        guides, guide_to_gene = self._discover_guides()
        n_guides = len(guides)
        print(
            f"\n  transform (guide-level): {n_passes} passes, "
            f"{n_guides} guides, {len(self.reporters)} reporters"
        )

        g_views, g_n_valid, g_cell_counts = precompute_inference_views(
            h5ad_paths=self._h5ad_paths,
            reporter_for_path=self._reporter_for_path,
            reporters=self.reporters,
            perturbations=guides,
            n_cells=self._n_cells_per_view,
            seed=self.seed,
            obs_col=self._guide_col,
        )

        X_guides = self._tta_loop(g_views, g_n_valid, n_passes, device)

        guide_adata = self._make_adata(X_guides, guides)
        guide_adata.obs["perturbation"] = [guide_to_gene[g] for g in guides]
        guide_adata.obs[self._guide_col] = guides
        _counts = g_cell_counts.astype(np.float64)
        _counts[_counts == 0] = np.nan
        guide_adata.obs["n_cells"] = np.nanmin(_counts, axis=1).astype(np.int64)
        AnndataValidator().validate(guide_adata, level="guide", strict=True)

        # --- Gene-level: equal-weight mean of guide embeddings per gene ---
        gene_labels_s = pd.Series(guide_to_gene)[guides]
        guide_df = pd.DataFrame(X_guides, index=guides)
        gene_df = guide_df.groupby(gene_labels_s.values).mean()
        gene_names = gene_df.index.tolist()
        gene_adata = self._make_adata(gene_df.values.astype(np.float32), gene_names)

        gene_to_guides_map: dict[str, list[str]] = defaultdict(list)
        for guide, gene in guide_to_gene.items():
            gene_to_guides_map[gene].append(guide)

        guide_n_cells = pd.Series(
            np.nanmin(_counts, axis=1).astype(np.int64), index=guides
        )
        gene_n_cells = guide_n_cells.groupby(gene_labels_s.values).min()
        gene_adata.obs["n_cells"] = gene_n_cells[gene_names].values.astype(np.int64)
        gene_adata.obs["guides"] = [
            "|".join(sorted(gene_to_guides_map[g])) for g in gene_names
        ]
        gene_adata.obs["n_experiments"] = self._n_experiments
        AnndataValidator().validate(gene_adata, level="gene", strict=True)

        self.model.cpu()
        return guide_adata, gene_adata

    # ------------------------------------------------------------------
    # Weight persistence
    # ------------------------------------------------------------------

    def save_weights(self, path: Path | str) -> None:
        """Save MLP state dict to disk."""
        if self.model is None:
            raise RuntimeError("No model to save — call fit() first.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"  Saved classifier weights to {path}")

    def load_weights(
        self, path: Path | str, device: torch.device | None = None
    ) -> None:
        """Load MLP state dict from disk.

        The model must already be instantiated (i.e. ``fit()`` must have
        been called first to set ``self.model``).
        """
        if self.model is None:
            raise RuntimeError("Model not instantiated — call fit() first.")
        state = torch.load(path, map_location=device or "cpu", weights_only=True)
        self.model.load_state_dict(state)
        print(f"  Loaded classifier weights from {path}")
