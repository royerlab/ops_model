#!/usr/bin/env python
r"""Evaluate a trained set classifier vs. number of cells per set (with resampling error bars).

Loads a checkpoint (local path or W&B artifact) and validation data the same way as
``analysis-mixed-channel-attn.ipynb``: mixed-channel mode uses :class:`MixedChannelDataset`
from a ``dump_val_dir`` export. Only ``mixed_channels_mode=true`` is supported.

For each ``n_cells`` in ``n_cells_list``, runs ``n_repetitions`` full validation passes
with different RNG seeds (stochastic subsampling in the dataset). Plots mean **top-1**
and **top-5** accuracy (top-``k`` uses ``k = min(5, n_classes)``) with SEM error bars.

Usage
-----
.. code-block:: bash

    cd projects/katamari
    python katamari/evals/image_verifier/eval_set_classifier.py \\
        checkpoint_path=/path/to/best_set_classifier.pt \\
        val_dump_dir=/path/to/val_ops \\
        output_plot_path=/path/to/plot.png \\
        n_cells_list='[100,250,500]' \\
        n_repetitions=20 \\
        device=cuda
"""

from __future__ import annotations

import csv
import json
import os
import random
from pathlib import Path
from typing import cast

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .train import (
    MixedChannelClassifier,
    MixedChannelDataset,
    _load_label_map,
    load_val_dataset,
)

_CONFIG_DIR = str(
    Path(os.environ.get("CONFIG_PATH", Path(__file__).resolve().parent / "configs"))
)


def _load_checkpoint(checkpoint_path: str | None) -> dict:
    if not checkpoint_path:
        raise ValueError("checkpoint_path is required")
    path = Path(checkpoint_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def _per_class_stats(
    logits: torch.Tensor,
    labels: torch.Tensor,
    n_classes: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (top1_hits, top5_hits, ones) bool/long tensors on CPU for scatter accumulation."""
    k = min(5, n_classes)
    preds1 = logits.argmax(dim=-1)
    top1_hits = preds1 == labels
    if k >= 1:
        topk_idx = logits.topk(k, dim=-1).indices
        top5_hits = (topk_idx == labels.unsqueeze(1)).any(dim=1)
    else:
        top5_hits = torch.zeros_like(labels, dtype=torch.bool)
    return top1_hits, top5_hits, torch.ones_like(labels, dtype=torch.long)


@torch.no_grad()
def _evaluate_mixed_with_topk(
    model: MixedChannelClassifier,
    loader: DataLoader,
    device: torch.device,
    *,
    n_classes: int,
    n_track_genes: int,
    gene_indices_in_dataset_order: torch.Tensor,
    desc: str | None,
    show_batches: bool,
) -> tuple[float, float, float, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (mean_loss, top1_accuracy, top5_accuracy, per_gene_top1, per_gene_top5, per_gene_n).

    Per-gene tensors are sized ``n_track_genes`` and indexed by checkpoint gene index
    (regardless of whether the model outputs label or gene logits).
    """
    model.eval()
    total_loss = 0.0
    correct1 = 0
    correct5 = 0
    total = 0
    pc_top1 = torch.zeros(n_track_genes, dtype=torch.long)
    pc_top5 = torch.zeros(n_track_genes, dtype=torch.long)
    pc_n = torch.zeros(n_track_genes, dtype=torch.long)
    it = (
        tqdm(loader, desc=desc or "val", leave=False, unit="batch")
        if show_batches
        else loader
    )
    processed = 0
    for embs, ch_ids, masks, labels in it:
        embs = embs.to(device)
        ch_ids = ch_ids.to(device)
        masks = masks.to(device)
        labels = labels.to(device)
        logits = model(embs, ch_ids, masks)
        loss = F.cross_entropy(logits, labels)
        total_loss += loss.item() * labels.size(0)
        top1_hits, top5_hits, ones = _per_class_stats(logits, labels, n_classes)
        correct1 += top1_hits.sum().item()
        correct5 += top5_hits.sum().item()
        total += labels.size(0)
        bs = labels.size(0)
        # shuffle=False on the loader guarantees batches arrive in dataset-index order.
        gene_ids = gene_indices_in_dataset_order[processed : processed + bs]
        processed += bs
        pc_n.scatter_add_(0, gene_ids, ones.cpu())
        pc_top1.scatter_add_(0, gene_ids, top1_hits.long().cpu())
        pc_top5.scatter_add_(0, gene_ids, top5_hits.long().cpu())

    if total == 0:
        raise ValueError(
            "Mixed-channel eval saw zero samples (no batches). "
            "Check for an empty val dataset or val DataLoader configuration."
        )
    return (
        total_loss / total,
        correct1 / total,
        correct5 / total,
        pc_top1,
        pc_top5,
        pc_n,
    )


def _build_channel_alignment(
    val_channel_to_idx: dict[str, int],
    ckpt_channel_to_idx: dict[str, int],
) -> tuple[torch.Tensor, frozenset[int]]:
    """Val-export channel indices → checkpoint indices; return remap tensor and dropped val ids."""
    ckpt_names = set(ckpt_channel_to_idx.keys())
    val_names = set(val_channel_to_idx.keys())
    val_only = val_names - ckpt_names
    shared = ckpt_names & val_names
    if not shared:
        raise ValueError(
            "No channel names in common between val export and checkpoint."
        )

    dropped_val_channel_ids = frozenset(val_channel_to_idx[n] for n in val_only)

    ch_remap: dict[int, int] = {}
    for name in shared:
        ch_remap[val_channel_to_idx[name]] = ckpt_channel_to_idx[name]

    _max_val_ch = max(val_channel_to_idx.values())
    ch_remap_tensor = torch.zeros(_max_val_ch + 1, dtype=torch.long)
    for old, new in ch_remap.items():
        ch_remap_tensor[old] = new

    return ch_remap_tensor, dropped_val_channel_ids


def _filter_mixed_val_channels(
    ds: MixedChannelDataset,
    ch_remap_tensor: torch.Tensor,
    dropped_val_channel_ids: frozenset[int],
) -> None:
    """In-place: drop cells on channels absent from ckpt and remap channel ids to ckpt space.

    Leaves val's gene_to_idx and gene indices unchanged.
    """
    dropped_t = (
        torch.tensor(sorted(dropped_val_channel_ids), dtype=torch.long)
        if dropped_val_channel_ids
        else None
    )

    for g_idx in list(ds._gene_pools.keys()):
        pool = ds._gene_pools[g_idx]
        ch_ids = ds._gene_ch_ids[g_idx]
        if len(pool) == 0:
            continue
        if dropped_t is not None and len(dropped_t) > 0:
            keep = ~torch.isin(ch_ids, dropped_t)
            pool = pool[keep]
            ch_ids = ch_ids[keep]
        if len(pool) == 0:
            ds._gene_pools[g_idx] = torch.zeros(0, ds.emb_dim)
            ds._gene_ch_ids[g_idx] = torch.zeros(0, dtype=torch.long)
            if ds._gene_dump_meta is not None and g_idx in ds._gene_dump_meta:
                del ds._gene_dump_meta[g_idx]
            continue
        ch_ids = ch_remap_tensor[ch_ids.long()]
        ds._gene_pools[g_idx] = pool
        ds._gene_ch_ids[g_idx] = ch_ids
        if ds._gene_dump_meta is not None and g_idx in ds._gene_dump_meta:
            del ds._gene_dump_meta[g_idx]


def _filter_and_remap_mixed_val(
    ds: MixedChannelDataset,
    ch_remap_tensor: torch.Tensor,
    dropped_val_channel_ids: frozenset[int],
    ckpt_gene_to_idx: dict[str, int],
) -> None:
    """In-place: drop cells on channels absent from ckpt, remap channel ids, fix labels, drop empty genes."""
    _filter_mixed_val_channels(ds, ch_remap_tensor, dropped_val_channel_ids)

    # Labels must match checkpoint class indices
    val_gene_to_idx = ds.gene_to_idx
    name_to_ckpt_idx = {n: ckpt_gene_to_idx[n] for n in ckpt_gene_to_idx}
    ds.gene_to_idx = dict(ckpt_gene_to_idx)

    kept_genes: list[str] = []
    remap_pool: dict[int, torch.Tensor] = {}
    remap_ch: dict[int, torch.Tensor] = {}
    for name in ds.perturbation_list:
        vgi = val_gene_to_idx[name]
        if vgi not in ds._gene_pools or len(ds._gene_pools[vgi]) == 0:
            continue
        cgi = name_to_ckpt_idx[name]
        remap_pool[cgi] = ds._gene_pools[vgi]
        remap_ch[cgi] = ds._gene_ch_ids[vgi]
        kept_genes.append(name)

    ds.perturbation_list = kept_genes
    ds._gene_pools = remap_pool
    ds._gene_ch_ids = remap_ch
    if ds._gene_dump_meta is not None:
        new_meta: dict = {}
        for name in kept_genes:
            vgi = val_gene_to_idx[name]
            cgi = name_to_ckpt_idx[name]
            if vgi in ds._gene_dump_meta:
                new_meta[cgi] = ds._gene_dump_meta[vgi]
        ds._gene_dump_meta = new_meta if new_meta else None


def _snapshot_mixed_val(
    ds: MixedChannelDataset,
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor], list[str]]:
    """Snapshot pools/ch_ids/perturbation_list so we can restore between channel iters."""
    pools = {g: p.clone() for g, p in ds._gene_pools.items()}
    ch_ids = {g: c.clone() for g, c in ds._gene_ch_ids.items()}
    perts = list(ds.perturbation_list)
    return pools, ch_ids, perts


def _restrict_mixed_val_to_channel(
    ds: MixedChannelDataset,
    target_ckpt_ch_id: int,
    snapshot_pools: dict[int, torch.Tensor],
    snapshot_ch_ids: dict[int, torch.Tensor],
    snapshot_perturbation_list: list[str],
    active_gene_to_idx: dict[str, int],
) -> None:
    """Restore ds from snapshot then drop all cells not on ``target_ckpt_ch_id``.

    Genes with zero cells of the target channel are dropped from ``perturbation_list``
    so they don't pollute accuracy stats with fully-masked padded sets.
    """
    new_pools: dict[int, torch.Tensor] = {}
    new_ch_ids: dict[int, torch.Tensor] = {}
    kept_gene_names: list[str] = []
    for name in snapshot_perturbation_list:
        g_idx = active_gene_to_idx[name]
        pool = snapshot_pools[g_idx]
        ch_ids = snapshot_ch_ids[g_idx]
        keep_mask = ch_ids == target_ckpt_ch_id
        if not bool(keep_mask.any()):
            continue
        new_pools[g_idx] = pool[keep_mask]
        new_ch_ids[g_idx] = ch_ids[keep_mask]
        kept_gene_names.append(name)
    # Genes outside perturbation_list still need entries (used elsewhere by index).
    for g_idx, pool in snapshot_pools.items():
        if g_idx in new_pools:
            continue
        new_pools[g_idx] = torch.zeros(0, ds.emb_dim)
        new_ch_ids[g_idx] = torch.zeros(0, dtype=torch.long)
    ds._gene_pools = new_pools
    ds._gene_ch_ids = new_ch_ids
    ds.perturbation_list = kept_gene_names


def _run_sweep(
    *,
    val_ds: Dataset,
    loader: DataLoader,
    model: nn.Module,
    device: torch.device,
    mixed_mode: bool,
    n_cells_list: list[int],
    n_repetitions: int,
    base_seed: int,
    n_classes: int,
    n_active_genes: int,
    gene_indices_in_dataset_order: torch.Tensor,
    loader_idx_to_ch: dict[int, str] | None,
    n_parquet_channels: int | None,
    overrides_raw,
    show_progress: bool,
    show_batch_pbar: bool,
    desc_prefix: str = "",
) -> tuple[
    list[float],
    list[float],
    list[float],
    list[float],
    dict[str, dict[str, list[int]]],
    dict[str, dict[str, list[float]]],
]:
    """Run the n_cells × n_repetitions sweep and return aggregate + per-gene results."""
    results: dict[str, dict[str, list[float]]] = {}
    means_top1: list[float] = []
    stderrs_top1: list[float] = []
    means_top5: list[float] = []
    stderrs_top5: list[float] = []
    per_gene_results: dict[str, dict[str, list[int]]] = {}

    outer_desc = f"{desc_prefix}n_cells" if desc_prefix else "n_cells"
    n_cells_iter = (
        tqdm(n_cells_list, desc=outer_desc) if show_progress else n_cells_list
    )
    for n_cells in n_cells_iter:
        accs1: list[float] = []
        accs5: list[float] = []
        key = str(n_cells)
        results[key] = {"top1": [], "top5": []}
        cum_pc_top1 = torch.zeros(n_active_genes, dtype=torch.long)
        cum_pc_top5 = torch.zeros(n_active_genes, dtype=torch.long)
        cum_pc_n = torch.zeros(n_active_genes, dtype=torch.long)
        rep_iter = (
            tqdm(
                range(n_repetitions),
                desc=f"{desc_prefix}n_cells={n_cells} reps",
                leave=False,
                unit="rep",
            )
            if show_progress
            else range(n_repetitions)
        )
        for rep in rep_iter:
            rseed = base_seed + rep * 1_000_003 + n_cells * 17
            torch.manual_seed(rseed)
            np.random.seed(rseed % (2**32))
            random.seed(rseed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(rseed)
            if mixed_mode:
                assert isinstance(val_ds, MixedChannelDataset)
                val_ds.set_n_cells(n_cells)
            else:
                raise ValueError("Only mixed_channels_mode=true is supported.")

            batch_desc = f"{desc_prefix}n={n_cells} rep {rep + 1}/{n_repetitions}"
            if mixed_mode:
                _, acc1, acc5, pc_top1, pc_top5, pc_n = _evaluate_mixed_with_topk(
                    cast(MixedChannelClassifier, model),
                    loader,
                    device,
                    n_classes=n_classes,
                    n_track_genes=n_active_genes,
                    gene_indices_in_dataset_order=gene_indices_in_dataset_order,
                    desc=batch_desc,
                    show_batches=show_batch_pbar,
                )
            else:
                raise ValueError("Only mixed_channels_mode=true is supported.")
            accs1.append(acc1)
            accs5.append(acc5)
            results[key]["top1"].append(acc1)
            results[key]["top5"].append(acc5)
            cum_pc_top1 += pc_top1
            cum_pc_top5 += pc_top5
            cum_pc_n += pc_n

        m1 = float(np.mean(accs1))
        m5 = float(np.mean(accs5))
        if n_repetitions > 1:
            se1 = float(np.std(accs1, ddof=1) / np.sqrt(len(accs1)))
            se5 = float(np.std(accs5, ddof=1) / np.sqrt(len(accs5)))
        else:
            se1 = 0.0
            se5 = 0.0
        means_top1.append(m1)
        stderrs_top1.append(se1)
        means_top5.append(m5)
        stderrs_top5.append(se5)
        per_gene_results[key] = {
            "top1_correct": cum_pc_top1.tolist(),
            "top5_correct": cum_pc_top5.tolist(),
            "n_samples": cum_pc_n.tolist(),
        }

    return (
        means_top1,
        stderrs_top1,
        means_top5,
        stderrs_top5,
        per_gene_results,
        results,
    )


def run(cfg: DictConfig) -> None:
    ckpt = _load_checkpoint(cfg.get("checkpoint_path"))
    config = ckpt["config"]
    ckpt_gene_to_idx: dict[str, int] = ckpt["gene_to_idx"]
    ckpt_channel_to_idx: dict[str, int] = ckpt["channel_to_idx"]
    n_ckpt_channels = len(ckpt_channel_to_idx)
    state_dict = ckpt["model_state_dict"]
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

    # ---- Label map (e.g. gene → pathway) ----
    label_to_idx: dict[str, int] | None = None
    label_remap: dict[int, int] | None = None
    gene_to_label: dict[str, str] | None = None

    label_map_path = cfg.get("label_map_path", None)
    if label_map_path is not None:
        # Supersedes any gene→label mapping stored in the checkpoint, letting us
        # evaluate on a different gene set. The label SPACE (label_to_idx) must
        # match training though, otherwise model output indices misalign — so we
        # take label_to_idx from ckpt when it's saved there. CSV labels not in
        # the trained label space are dropped (model can't predict them).
        gene_col = cfg.get("label_map_gene_col", "gene_name")
        label_col = cfg.get("label_map_label_col", "pathway")
        gene_to_label = _load_label_map(label_map_path, gene_col, label_col)
        if "label_to_idx" in ckpt:
            label_to_idx = cast(dict[str, int], ckpt["label_to_idx"])
            new_labels = set(gene_to_label.values()) - set(label_to_idx)
            if new_labels:
                print(
                    f"  Warning: {len(new_labels)} labels in CSV not in trained "
                    f"label set (genes dropped): {sorted(new_labels)[:10]}"
                )
                gene_to_label = {
                    g: lab for g, lab in gene_to_label.items() if lab in label_to_idx
                }
        else:
            unique_labels = sorted(set(gene_to_label.values()))
            label_to_idx = {label: i for i, label in enumerate(unique_labels)}
        print(
            f"Label map from config: {len(gene_to_label)} genes in CSV → "
            f"{len(label_to_idx)} classes ({label_col})"
        )
    elif "label_to_idx" in ckpt:
        label_to_idx = cast(dict[str, int], ckpt["label_to_idx"])
        if "label_remap" in ckpt:
            label_remap = cast(dict[int, int], ckpt["label_remap"])
            print(
                f"Label map from checkpoint: {len(label_remap)} genes → "
                f"{len(label_to_idx)} classes"
            )
        else:
            raise ValueError(
                "Checkpoint has label_to_idx but no label_remap (older checkpoint). "
                "Set label_map_path in the config to the same CSV used during training "
                "so the gene→label mapping can be reconstructed."
            )

    n_classes = state_dict["head.1.weight"].shape[0]
    if label_to_idx is not None and len(label_to_idx) != n_classes:
        raise ValueError(
            f"label_to_idx has {len(label_to_idx)} classes but model output dim "
            f"is {n_classes}. The CSV's label set must match what the model was "
            f"trained on."
        )

    mixed_mode = bool(
        cfg.get("mixed_channels_mode", config.get("mixed_channels_mode", True))
    )

    device = torch.device(
        cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    )
    base_seed = int(cfg.get("seed", 42))

    n_cells_list = [int(x) for x in list(cfg.n_cells_list)]
    if not n_cells_list:
        raise ValueError("n_cells_list must be non-empty")
    n_repetitions = int(cfg["n_repetitions"])
    if n_repetitions < 1:
        raise ValueError("n_repetitions must be >= 1")

    batch_size = int(cfg.get("batch_size", 64))
    num_workers = int(cfg.get("num_workers", 0))

    # Checkpoint stores Hydra config as a plain dict after torch.load — not an OmegaConf node.
    mcfg = OmegaConf.to_container(OmegaConf.create(config["model"]), resolve=True)
    assert isinstance(mcfg, dict)

    # ---- Model ----
    emb_dim = int(state_dict["input_proj.weight"].shape[1])
    if mixed_mode:
        model = MixedChannelClassifier(
            emb_dim=emb_dim,
            n_classes=n_classes,
            n_channels=n_ckpt_channels,
            d_model=mcfg.get("d_model", 256),
            n_heads=mcfg.get("n_heads", 4),
            n_layers=mcfg.get("n_layers_cell", 2),
            n_inducing=mcfg.get("n_inducing_cell", 32),
            d_ff=mcfg.get("d_ff", None),
            dropout=mcfg.get("dropout", 0.1),
            cosine_classifier=mcfg.get("cosine_classifier", False),
            channel_conditioning=mcfg.get("channel_conditioning", "none"),
        )
    else:
        raise ValueError("Only mixed_channels_mode=true is supported.")

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    # ---- Data ----
    if mixed_mode:
        vd = cfg.get("val_dump_dir")
        if vd in (None, "", "null"):
            raise ValueError(
                "mixed_channels_mode=true requires val_dump_dir (directory with metadata.pt from dump_val_dir)."
            )
        val_dir = Path(str(vd)).expanduser()
        meta_path = val_dir / "metadata.pt"
        if not meta_path.is_file():
            raise FileNotFoundError(f"Expected val export at {meta_path}")
        val_meta = torch.load(meta_path, map_location="cpu", weights_only=False)
        val_gene_to_idx: dict[str, int] = val_meta["gene_to_idx"]
        val_channel_to_idx: dict[str, int] = val_meta["channel_to_idx"]

        # Gene set check: with label_map_path, val genes only need to be in the
        # CSV (not in ckpt). Otherwise, val genes must be a subset of ckpt's.
        if gene_to_label is None:
            val_only = set(val_gene_to_idx) - set(ckpt_gene_to_idx)
            if val_only:
                raise ValueError(
                    "Val export has genes not present in checkpoint "
                    f"(cannot be evaluated): {sorted(val_only)[:20]}"
                )
            ckpt_only = set(ckpt_gene_to_idx) - set(val_gene_to_idx)
            if ckpt_only:
                print(
                    f"  {len(ckpt_only)} genes in checkpoint absent from val export "
                    f"(skipped): {sorted(ckpt_only)[:10]}{'...' if len(ckpt_only) > 10 else ''}"
                )

        ch_remap_tensor, dropped_val_ch = _build_channel_alignment(
            val_channel_to_idx, ckpt_channel_to_idx
        )

        only_channels_raw = cfg.get("only_channels")
        if only_channels_raw is not None:
            only_channels = set(only_channels_raw)
            unknown = only_channels - set(val_channel_to_idx.keys())
            if unknown:
                raise ValueError(
                    f"only_channels names not found in val export: {unknown}. "
                    f"Available: {sorted(val_channel_to_idx.keys())}"
                )
            extra_drop = frozenset(
                val_channel_to_idx[n]
                for n in val_channel_to_idx
                if n not in only_channels
            )
            dropped_val_ch = dropped_val_ch | extra_drop
            print(f"Restricting validation to channels: {sorted(only_channels)}")

        max_n = max(n_cells_list)
        val_ds = load_val_dataset(str(val_dir), n_cells=max_n)
        val_ds.replacement = not bool(cfg.get("sample_without_replacement", False))

        if gene_to_label is not None:
            # label_map_path mode: keep val's gene indices, build label_remap
            # from val_gene_to_idx + CSV. Val genes can be entirely new wrt ckpt.
            assert label_to_idx is not None
            _filter_mixed_val_channels(val_ds, ch_remap_tensor, dropped_val_ch)
            kept = [
                g
                for g in val_ds.perturbation_list
                if g in gene_to_label
                and val_gene_to_idx[g] in val_ds._gene_pools
                and len(val_ds._gene_pools[val_gene_to_idx[g]]) > 0
            ]
            if len(kept) < len(val_ds.perturbation_list):
                dropped = len(val_ds.perturbation_list) - len(kept)
                print(f"  Label map: dropped {dropped} genes without label mapping")
            val_ds.perturbation_list = kept
            label_remap = {
                val_gene_to_idx[g]: label_to_idx[gene_to_label[g]] for g in kept
            }
            val_ds.label_remap = label_remap
            active_gene_to_idx: dict[str, int] = val_gene_to_idx
        else:
            _filter_and_remap_mixed_val(
                val_ds, ch_remap_tensor, dropped_val_ch, ckpt_gene_to_idx
            )
            if label_remap is not None:
                val_ds.label_remap = label_remap
                kept = [
                    g
                    for g in val_ds.perturbation_list
                    if ckpt_gene_to_idx[g] in label_remap
                ]
                if len(kept) < len(val_ds.perturbation_list):
                    dropped = len(val_ds.perturbation_list) - len(kept)
                    print(f"  Label map: dropped {dropped} genes without label mapping")
                    val_ds.perturbation_list = kept
            active_gene_to_idx = ckpt_gene_to_idx

        if len(val_ds.perturbation_list) == 0:
            raise ValueError(
                "No genes left in val set after channel filtering (empty pools)."
            )

        loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
        )
        gene_indices_in_dataset_order = torch.tensor(
            [active_gene_to_idx[name] for name in val_ds.perturbation_list],
            dtype=torch.long,
        )
        loader_idx_to_ch: dict[int, str] | None = None
        n_parquet_channels: int | None = None
    else:
        raise ValueError("Only mixed_channels_mode=true is supported.")

    # ---- Eval sweeps ----
    show_progress = bool(cfg.get("show_progress", True))
    show_batch_pbar = show_progress
    n_active_genes = len(active_gene_to_idx)
    split_channels = bool(cfg.get("split_channels", False))
    overrides_raw = cfg.get("n_cells_per_set_overrides", {})

    # Per-channel sweep results when split_channels=true; otherwise a single "all" entry.
    sweep_by_channel: dict[
        str,
        tuple[
            list[float],
            list[float],
            list[float],
            list[float],
            dict[str, dict[str, list[int]]],
            dict[str, dict[str, list[float]]],
        ],
    ] = {}

    if split_channels:
        if not mixed_mode:
            raise ValueError("split_channels=true requires mixed_channels_mode=true.")
        assert isinstance(val_ds, MixedChannelDataset)
        # Snapshot the val_ds state after the ckpt-wide channel filter+remap, so each
        # per-channel iteration starts from the same clean baseline.
        snap_pools, snap_ch_ids, snap_perts = _snapshot_mixed_val(val_ds)

        if only_channels_raw is not None:
            eval_channel_names = list(only_channels_raw)
        else:
            eval_channel_names = sorted(
                set(ckpt_channel_to_idx.keys()) & set(val_channel_to_idx.keys())
            )
        if not eval_channel_names:
            raise ValueError(
                "split_channels=true but no channels available to evaluate "
                "(empty intersection of checkpoint and val export)."
            )
        unknown_ch = [n for n in eval_channel_names if n not in ckpt_channel_to_idx]
        if unknown_ch:
            raise ValueError(
                f"split_channels: channel names not in checkpoint vocab: {unknown_ch}"
            )
        # Optional channel sharding for parallel eval jobs: each job handles a
        # deterministic stride of the (sorted) channel list; combine CSVs after.
        n_channel_shards = int(cfg.get("n_channel_shards", 1))
        channel_shard_id = int(cfg.get("channel_shard_id", 0))
        if n_channel_shards > 1:
            eval_channel_names = eval_channel_names[channel_shard_id::n_channel_shards]
            print(
                f"channel shard {channel_shard_id}/{n_channel_shards}: "
                f"{len(eval_channel_names)} channels"
            )
        print(f"split_channels: evaluating channels {eval_channel_names}")

        for ch_name in eval_channel_names:
            target_ch_id = ckpt_channel_to_idx[ch_name]
            _restrict_mixed_val_to_channel(
                val_ds,
                target_ch_id,
                snap_pools,
                snap_ch_ids,
                snap_perts,
                active_gene_to_idx,
            )
            if not val_ds.perturbation_list:
                print(
                    f"  Skipping channel {ch_name!r}: no genes with cells in this channel"
                )
                continue
            gene_indices_in_dataset_order = torch.tensor(
                [active_gene_to_idx[name] for name in val_ds.perturbation_list],
                dtype=torch.long,
            )
            sweep_out = _run_sweep(
                val_ds=val_ds,
                loader=loader,
                model=model,
                device=device,
                mixed_mode=mixed_mode,
                n_cells_list=n_cells_list,
                n_repetitions=n_repetitions,
                base_seed=base_seed,
                n_classes=n_classes,
                n_active_genes=n_active_genes,
                gene_indices_in_dataset_order=gene_indices_in_dataset_order,
                loader_idx_to_ch=loader_idx_to_ch,
                n_parquet_channels=n_parquet_channels,
                overrides_raw=overrides_raw,
                show_progress=show_progress,
                show_batch_pbar=show_batch_pbar,
                desc_prefix=f"[{ch_name}] ",
            )
            sweep_by_channel[ch_name] = sweep_out
    else:
        sweep_out = _run_sweep(
            val_ds=val_ds,
            loader=loader,
            model=model,
            device=device,
            mixed_mode=mixed_mode,
            n_cells_list=n_cells_list,
            n_repetitions=n_repetitions,
            base_seed=base_seed,
            n_classes=n_classes,
            n_active_genes=n_active_genes,
            gene_indices_in_dataset_order=gene_indices_in_dataset_order,
            loader_idx_to_ch=loader_idx_to_ch,
            n_parquet_channels=n_parquet_channels,
            overrides_raw=overrides_raw,
            show_progress=show_progress,
            show_batch_pbar=show_batch_pbar,
        )
        # Use a single synthetic channel name for the unified-eval case.
        sweep_by_channel["__all__"] = sweep_out

    # ---- Plot ----
    out_raw = cfg.get("output_plot_path")
    out_plot = (
        Path(str(out_raw)).expanduser() if out_raw not in (None, "", "null") else None
    )
    plot_x_log = bool(cfg.get("plot_x_log", True))
    x = np.array(sorted(n_cells_list), dtype=float)
    plot_title = cfg.get("plot_title", "Set classifier accuracy vs. N cells")

    if out_plot is None:
        print("output_plot_path not set; skipping plot and sidecar JSON.")
    else:
        out_plot.parent.mkdir(parents=True, exist_ok=True)
        if split_channels:
            fig, (ax1, ax5) = plt.subplots(1, 2, figsize=(14, 5))
            for i, (ch_name, sweep_out) in enumerate(sweep_by_channel.items()):
                m1, e1, m5, e5, *_ = sweep_out
                color = f"C{i % 10}"
                ax1.errorbar(
                    x,
                    np.array(m1),
                    yerr=np.array(e1),
                    fmt="o-",
                    capsize=3,
                    capthick=1.0,
                    linewidth=1.8,
                    markersize=5,
                    label=ch_name,
                    color=color,
                )
                ax5.errorbar(
                    x,
                    np.array(m5),
                    yerr=np.array(e5),
                    fmt="s-",
                    capsize=3,
                    capthick=1.0,
                    linewidth=1.8,
                    markersize=5,
                    label=ch_name,
                    color=color,
                )
            for ax, title in ((ax1, "Top-1"), (ax5, "Top-5")):
                ax.set_xlabel("N cells per set")
                ax.set_ylabel("Validation accuracy")
                ax.set_title(f"{plot_title} ({title})")
                if plot_x_log:
                    ax.set_xscale("log")
                tick_vals = sorted({float(n) for n in n_cells_list})
                ax.set_xticks(tick_vals)
                ax.set_xticklabels(
                    [
                        str(int(v)) if v >= 1.0 and v == int(v) else f"{v:g}"
                        for v in tick_vals
                    ]
                )
                ax.grid(True, alpha=0.3)
                ax.set_ylim(0.0, 1.0)
                ax.legend(loc="lower right", fontsize=8)
        else:
            m1, e1, m5, e5, *_ = sweep_by_channel["__all__"]
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.errorbar(
                x,
                np.array(m1),
                yerr=np.array(e1),
                fmt="o-",
                capsize=4,
                capthick=1.5,
                linewidth=2,
                markersize=6,
                label="Top-1",
                color="C0",
            )
            ax.errorbar(
                x,
                np.array(m5),
                yerr=np.array(e5),
                fmt="s-",
                capsize=4,
                capthick=1.5,
                linewidth=2,
                markersize=6,
                label="Top-5",
                color="C1",
            )
            ax.set_xlabel("N cells per set")
            ax.set_ylabel("Validation accuracy")
            ax.set_title(plot_title)
            if plot_x_log:
                ax.set_xscale("log")
            tick_vals = sorted({float(n) for n in n_cells_list})
            ax.set_xticks(tick_vals)
            ax.set_xticklabels(
                [
                    str(int(v)) if v >= 1.0 and v == int(v) else f"{v:g}"
                    for v in tick_vals
                ]
            )
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0.0, 1.0)
            ax.legend(loc="lower right")
        fig.tight_layout()
        fig.savefig(out_plot, dpi=150)
        plt.close(fig)
        print(f"Saved plot to {out_plot}")

        # ---- Sidecar JSON ----
        sidecar = out_plot.with_suffix(".json")
        payload: dict = {
            "n_cells_list": n_cells_list,
            "n_repetitions": n_repetitions,
            "plot_x_log": plot_x_log,
            "device": str(device),
            "mixed_channels_mode": mixed_mode,
            "split_channels": split_channels,
            "n_classes": n_classes,
        }
        if split_channels:
            payload["per_channel"] = {
                ch_name: {
                    "mean_accuracy_top1": m1,
                    "stderr_top1": e1,
                    "mean_accuracy_top5": m5,
                    "stderr_top5": e5,
                    "repetitions": reps,
                }
                for ch_name, (m1, e1, m5, e5, _, reps) in sweep_by_channel.items()
            }
        else:
            m1, e1, m5, e5, _, reps = sweep_by_channel["__all__"]
            payload["mean_accuracy_top1"] = m1
            payload["stderr_top1"] = e1
            payload["mean_accuracy_top5"] = m5
            payload["stderr_top5"] = e5
            payload["repetitions"] = reps
        if label_to_idx is not None:
            idx_to_label = {v: k for k, v in label_to_idx.items()}
            payload["label_to_idx"] = label_to_idx
            payload["class_names"] = [idx_to_label[i] for i in range(len(idx_to_label))]
        sidecar.write_text(json.dumps(payload, indent=2))
        print(f"Saved metrics to {sidecar}")

    # ---- Per-class CSV ----
    per_class_path = cfg.get("per_class_output_file")
    if per_class_path not in (None, "", "null"):
        per_class_out = Path(str(per_class_path)).expanduser()
        per_class_out.parent.mkdir(parents=True, exist_ok=True)
        idx_to_gene_name = {v: k for k, v in active_gene_to_idx.items()}
        has_label_map = label_remap is not None
        idx_to_label_name: dict[int, str] = {}
        if has_label_map:
            assert label_to_idx is not None and label_remap is not None
            idx_to_label_name = {v: k for k, v in label_to_idx.items()}
            gene_indices_to_write = sorted(label_remap.keys())
        else:
            gene_indices_to_write = sorted(active_gene_to_idx.values())
        with open(per_class_out, "w", newline="") as f:
            writer = csv.writer(f)
            header: list[str] = []
            if split_channels:
                header.append("channel_name")
            header += ["n_cells", "gene_idx", "gene_name"]
            if has_label_map:
                header += ["label_name"]
            header += [
                "n_repetitions",
                "n_samples",
                "top1_correct",
                "top5_correct",
                "top1_acc",
                "top5_acc",
            ]
            writer.writerow(header)
            for ch_name, sweep_out in sweep_by_channel.items():
                _, _, _, _, per_gene_results, _ = sweep_out
                for n_cells in n_cells_list:
                    pc = per_gene_results[str(n_cells)]
                    for gene_idx in gene_indices_to_write:
                        n_samples = pc["n_samples"][gene_idx]
                        top1_c = pc["top1_correct"][gene_idx]
                        top5_c = pc["top5_correct"][gene_idx]
                        top1_acc = top1_c / n_samples if n_samples > 0 else 0.0
                        top5_acc = top5_c / n_samples if n_samples > 0 else 0.0
                        row: list = []
                        if split_channels:
                            row.append(ch_name)
                        row += [n_cells, gene_idx, idx_to_gene_name[gene_idx]]
                        if has_label_map:
                            assert label_remap is not None
                            row.append(idx_to_label_name[label_remap[gene_idx]])
                        row += [
                            n_repetitions,
                            n_samples,
                            top1_c,
                            top5_c,
                            f"{top1_acc:.6f}",
                            f"{top5_acc:.6f}",
                        ]
                        writer.writerow(row)
        print(f"Saved per-gene metrics to {per_class_out}")


@hydra.main(
    version_base="1.3.0",
    config_path=_CONFIG_DIR,
    config_name="eval_set_classifier",
)
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()  # type: ignore[call-arg]
