#!/usr/bin/env python
"""Train a set-transformer classifier on per-cell embeddings.

Architecture
------------
1. For each perturbation (gene), sample ``n_cells`` cell embeddings per channel.
2. Add a learned channel embedding to every cell.
3. Per-channel Set Transformer (ISAB + PMA) pools each channel's cells
   into a single vector.
4. A cross-channel Set Transformer aggregates channel vectors into a
   final representation.
5. A linear head predicts the gene class.

Uses inducing-point set attention (ISAB) so that cost is O(N·m) instead
of O(N²), where m is the number of inducing points.

Usage
-----
.. code-block:: bash

    python train_set_classifier.py
    python train_set_classifier.py n_cells=500 model.d_model=256
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm import tqdm

_CONFIG_DIR = str(
    Path(os.environ.get("CONFIG_PATH", Path(__file__).resolve().parent / "configs"))
)

# ---------------------------------------------------------------------------
# Set Transformer building blocks
# ---------------------------------------------------------------------------


def _cond_kw(cond: torch.Tensor | None) -> dict[str, torch.Tensor]:
    """Build kwargs dict for AdaLNMAB (empty dict when no conditioning)."""
    return {"cond": cond} if cond is not None else {}


class CrossAttnBlock(nn.Module):
    """Pre-norm cross-attention block.

    Only the query stream (X) is normalized; KV (Y) passes through raw.
    H = X + drop(Attn(LN(X), Y, Y))
    out = H + drop(FF(LN(H)))
    """

    def __init__(
        self, d_model: int, n_heads: int, d_ff: int | None = None, dropout: float = 0.0
    ):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_ff = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        y_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x_norm = self.norm_q(x)
        h = (
            x
            + self.attn(
                x_norm, y, y, key_padding_mask=y_key_padding_mask, need_weights=False
            )[0]
        )
        return h + self.ff(self.norm_ff(h))


class AdaLNCrossAttnBlock(nn.Module):
    """Pre-norm cross-attention block with adaptive LayerNorm conditioning.

    Only the query stream is normalized; KV passes through raw.
    AdaLN modulates the two LN outputs (query norm, FF norm) via
    per-sample scale/shift from a conditioning vector.

    Accepts cond as either (B, D) for global conditioning (broadcast across
    all query positions) or (B, N, D) for per-token conditioning.
    """

    def __init__(
        self, d_model: int, n_heads: int, d_ff: int | None = None, dropout: float = 0.0
    ):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm_q = nn.LayerNorm(d_model, elementwise_affine=False)
        self.norm_ff = nn.LayerNorm(d_model, elementwise_affine=False)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.adaln_proj = nn.Linear(d_model, 4 * d_model)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        y_key_padding_mask: torch.Tensor | None = None,
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if cond is not None:
            params = self.adaln_proj(cond)
            if params.dim() == 2:
                params = params.unsqueeze(1)
            gamma_q, beta_q, gamma_ff, beta_ff = params.chunk(4, dim=-1)
            x_norm = self.norm_q(x) * (1 + gamma_q) + beta_q
            h = (
                x
                + self.attn(
                    x_norm,
                    y,
                    y,
                    key_padding_mask=y_key_padding_mask,
                    need_weights=False,
                )[0]
            )
            return h + self.ff(self.norm_ff(h) * (1 + gamma_ff) + beta_ff)
        else:
            x_norm = self.norm_q(x)
            h = (
                x
                + self.attn(
                    x_norm,
                    y,
                    y,
                    key_padding_mask=y_key_padding_mask,
                    need_weights=False,
                )[0]
            )
            return h + self.ff(self.norm_ff(h))


class SelfAttnBlock(nn.Module):
    """Pre-norm self-attention block.

    The single input is normalized once, then used for Q, K, V.
    H = X + drop(Attn(LN(X), LN(X), LN(X)))
    out = H + drop(FF(LN(H)))
    """

    def __init__(
        self, d_model: int, n_heads: int, d_ff: int | None = None, dropout: float = 0.0
    ):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm_attn = nn.LayerNorm(d_model)
        self.norm_ff = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x_norm = self.norm_attn(x)
        h = (
            x
            + self.attn(
                x_norm,
                x_norm,
                x_norm,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )[0]
        )
        return h + self.ff(self.norm_ff(h))


class ISAB(nn.Module):
    """Inducing-point Set Attention Block.

    Reduces O(N²) self-attention to O(N·m) by routing through *m*
    learnable inducing points.  Both sub-blocks are cross-attention:
    mab1 attends inducing→input, mab2 attends input→inducing summary.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_inducing: int,
        d_ff: int | None = None,
        adaln: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.inducing = nn.Parameter(torch.randn(1, n_inducing, d_model) * 0.02)
        cross_cls = AdaLNCrossAttnBlock if adaln else CrossAttnBlock
        self.cross1 = cross_cls(d_model, n_heads, d_ff, dropout=dropout)
        self.cross2 = cross_cls(d_model, n_heads, d_ff, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # cross1: inducing points query the input tokens. When cond is per-token
        # (B, N, D), average to (B, D) for the m inducing-point queries.
        cond1 = cond.mean(dim=1) if cond is not None and cond.dim() == 3 else cond
        h = self.cross1(
            self.inducing.expand(x.size(0), -1, -1),
            x,
            y_key_padding_mask=key_padding_mask,
            **_cond_kw(cond1),
        )
        # cross2: input tokens query the inducing summaries. Per-token cond
        # (B, N, D) is passed directly so each token gets its own modulation.
        return self.cross2(x, h, **_cond_kw(cond))


class PMA(nn.Module):
    """Pooling by Multihead Attention (cross-attention from seeds to set)."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_seeds: int = 1,
        d_ff: int | None = None,
        adaln: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.seeds = nn.Parameter(torch.randn(1, n_seeds, d_model) * 0.02)
        cross_cls = AdaLNCrossAttnBlock if adaln else CrossAttnBlock
        self.cross = cross_cls(d_model, n_heads, d_ff, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Seeds are learnable queries, not input tokens. When cond is per-token
        # (B, N, D), average to (B, D) for the seed queries.
        cond_pool = cond.mean(dim=1) if cond is not None and cond.dim() == 3 else cond
        return self.cross(
            self.seeds.expand(x.size(0), -1, -1),
            x,
            y_key_padding_mask=key_padding_mask,
            **_cond_kw(cond_pool),
        )


class MILAttentionPool(nn.Module):
    """Gated-attention multiple-instance-learning pooling (Ilse et al. 2018).

    Computes a scalar attention weight per instance,
    ``a_i = softmax_i(w^T (tanh(V h_i) ⊙ sigmoid(U h_i)))``, and returns the
    convex combination ``sum_i a_i h_i``.

    Unlike PMA, each ``a_i`` depends only on cell ``i`` (no inter-cell
    interaction) and pooling is a plain convex combination, so the weights are a
    clean, faithful per-cell importance score for downstream cell selection. The
    most recent (masked, softmaxed) weights are stored on ``last_attn`` (B, N)
    so selection code can read them off after a forward pass.
    """

    def __init__(self, d_model: int, d_attn: int | None = None):
        super().__init__()
        d_attn = d_attn or d_model
        self.V = nn.Linear(d_model, d_attn)
        self.U = nn.Linear(d_model, d_attn)
        self.w = nn.Linear(d_attn, 1)
        self.last_attn = None

    def forward(self, x, key_padding_mask=None):
        scores = self.w(torch.tanh(self.V(x)) * torch.sigmoid(self.U(x))).squeeze(-1)
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask, float("-inf"))
        attn = torch.softmax(scores, dim=1)
        self.last_attn = attn.detach()
        return torch.bmm(attn.unsqueeze(1), x).squeeze(1)


class ISABBlock(nn.Module):
    """Stack of ISAB layers followed by pooling.

    pool_type:
        - "pma": Pooling by Multihead Attention (learnable seed tokens).
        - "mean": masked mean over the set axis.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        n_inducing: int,
        n_pool_seeds: int = 1,
        d_ff: int | None = None,
        adaln: bool = False,
        dropout: float = 0.0,
        pool_type: str = "pma",
        grad_checkpoint: bool = False,
    ):
        super().__init__()
        assert pool_type in ("pma", "mean")
        self.pool_type = pool_type
        self.grad_checkpoint = grad_checkpoint
        self.layers = nn.ModuleList(
            [
                ISAB(d_model, n_heads, n_inducing, d_ff, adaln=adaln, dropout=dropout)
                for _ in range(n_layers)
            ]
        )
        if pool_type == "pma":
            self.pool: nn.Module | None = PMA(
                d_model, n_heads, n_pool_seeds, d_ff, adaln=adaln, dropout=dropout
            )
        else:
            self.pool = None
        self.final_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            if self.grad_checkpoint and self.training:
                # Discard the ISAB layer's activations and recompute them in
                # backward -- trades ~25-30% compute for much lower memory,
                # enabling larger bags. Identical math; loss is unaffected.
                x = torch.utils.checkpoint.checkpoint(
                    layer, x, key_padding_mask, cond, use_reentrant=False
                )
            else:
                x = layer(x, key_padding_mask=key_padding_mask, cond=cond)
        if self.pool_type == "pma":
            assert self.pool is not None
            pooled = self.pool(x, key_padding_mask=key_padding_mask, cond=cond).squeeze(
                1
            )
        else:
            # key_padding_mask convention: True = pad.
            if key_padding_mask is not None:
                valid = (~key_padding_mask).unsqueeze(-1).to(x.dtype)
                pooled = (x * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
            else:
                pooled = x.mean(dim=1)
        return self.final_norm(pooled)


class SABBlock(nn.Module):
    """Stack of self-attention layers followed by PMA pooling."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        n_pool_seeds: int = 1,
        d_ff: int | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                SelfAttnBlock(d_model, n_heads, d_ff, dropout=dropout)
                for _ in range(n_layers)
            ]
        )
        self.pool = PMA(d_model, n_heads, n_pool_seeds, d_ff, dropout=dropout)
        self.final_norm = nn.LayerNorm(d_model)

    def forward(
        self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask)
        return self.final_norm(
            self.pool(x, key_padding_mask=key_padding_mask).squeeze(1)
        )


# ---------------------------------------------------------------------------
# Full classifier
# ---------------------------------------------------------------------------


class CosineClassifier(nn.Module):
    """Cosine-similarity head with learned temperature.

    Normalizes both the input and weight vectors so the logit for each class
    is ``scale * cos(x, w_c)``.
    """

    def __init__(self, in_dim: int, num_classes: int, init_scale: float = 20.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_classes, in_dim))
        self.log_scale = nn.Parameter(torch.tensor(math.log(init_scale)))
        nn.init.normal_(self.weight, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.normalize(x, dim=-1)
        w = F.normalize(self.weight, dim=-1)
        return torch.exp(self.log_scale) * (x @ w.t())


class SetClassifier(nn.Module):
    """Two-stage set transformer classifier.

    Stage 1 (per-channel): pools N_cells cell embeddings per channel into
    one vector, using ISAB for efficiency.

    Stage 2 (cross-channel): aggregates the per-channel vectors into a
    single representation and classifies.

    Channel conditioning modes:
        - "add": add channel embedding to each cell embedding (default).
        - "adaln": modulate LayerNorm outputs via adaptive scale/shift from channel embedding.
        - "none": no channel conditioning.
    """

    def __init__(
        self,
        emb_dim: int,
        n_channels: int,
        n_classes: int,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers_cell: int = 2,
        n_layers_channel: int = 1,
        n_inducing_cell: int = 32,
        d_ff: int | None = None,
        dropout: float = 0.1,
        channel_conditioning: str = "add",
        cosine_classifier: bool = False,
    ):
        super().__init__()
        assert channel_conditioning in ("add", "adaln", "none")
        self.channel_conditioning = channel_conditioning

        self.input_proj = nn.Linear(emb_dim, d_model)
        if channel_conditioning in ("add", "adaln"):
            self.channel_embeddings = nn.Embedding(n_channels, d_model)

        use_adaln = channel_conditioning == "adaln"
        self.cell_encoder = ISABBlock(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers_cell,
            n_inducing=n_inducing_cell,
            n_pool_seeds=1,
            d_ff=d_ff,
            adaln=use_adaln,
            dropout=dropout,
        )

        self.channel_encoder = SABBlock(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers_channel,
            n_pool_seeds=1,
            d_ff=d_ff,
            dropout=dropout,
        )

        classifier: nn.Module
        if cosine_classifier:
            classifier = CosineClassifier(d_model, n_classes)
        else:
            classifier = nn.Linear(d_model, n_classes)
        self.head = nn.Sequential(nn.Dropout(dropout), classifier)

    def forward(
        self,
        channel_groups: list[tuple[list[int], torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            channel_groups: list of (ch_indices, embs, masks) where
                embs is (B, G, N, D) and masks is (B, G, N) bool.
                Channels within a group share the same N so they can
                be processed by the cell encoder in one batched call.

        Returns:
            Logits of shape (B, n_classes).
        """
        all_vecs: list[tuple[int, torch.Tensor, torch.Tensor]] = []

        for ch_indices, embs, masks in channel_groups:
            B, G, N, _ = embs.shape
            x = self.input_proj(embs.reshape(B * G, N, -1))

            cond = None
            ch_ids = torch.tensor(ch_indices, device=embs.device)
            if self.channel_conditioning == "add":
                ch_emb = self.channel_embeddings(ch_ids)
                ch_emb = ch_emb.unsqueeze(0).unsqueeze(2).expand(B, G, N, -1)
                x = x + ch_emb.reshape(B * G, N, -1)
            elif self.channel_conditioning == "adaln":
                cond = self.channel_embeddings(ch_ids)
                cond = cond.unsqueeze(0).expand(B, G, -1).reshape(B * G, -1)

            kpm = ~masks.reshape(B * G, N)
            vecs = self.cell_encoder(x, key_padding_mask=kpm, cond=cond)
            vecs = vecs.reshape(B, G, -1)
            valid = masks.any(dim=2)

            for i, ch_idx in enumerate(ch_indices):
                all_vecs.append((ch_idx, vecs[:, i], valid[:, i]))

        all_vecs.sort(key=lambda t: t[0])
        stacked = torch.stack([v for _, v, _ in all_vecs], dim=1)
        ch_valid = torch.stack([v for _, _, v in all_vecs], dim=1)
        z = self.channel_encoder(stacked, key_padding_mask=~ch_valid)
        return self.head(z)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


CellIndex = dict[tuple[int, int], torch.Tensor]
"""Mapping from (gene_idx, channel_idx) → (N, D) embedding tensor."""

CellDumpMetaChunk = dict[str, list]
"""One embedding-chunk's metadata: column → flat list of per-cell values (Python scalars/str)."""

CellDumpMetaLoL = dict[str, list[list]]
"""Column → list of segments; each segment is a list aligned with one concatenated emb block."""

CellDumpMetaFlat = dict[str, list]
"""Column → one flat list per cell (used after ``load_val_dataset``)."""

VAL_DUMP_META_COLS = (
    "well",
    "x_pheno",
    "y_pheno",
    "segmentation_id",
    "channel_type",
    "index",
)
"""Parquet columns read only when ``load_val_dump_metadata`` is True (plus ``experiment``, already required)."""

CELL_METADATA_DUMP_KEY = "cell_metadata"
CELL_DUMP_KEYS = (
    "experiment",
    "well",
    "x_pheno",
    "y_pheno",
    "segmentation_id",
    "channel_type",
    "index",
)


def _meta_lol_n_cells(m: CellDumpMetaLoL) -> int:
    return sum(len(seg) for seg in m["experiment"])


def _is_cell_meta_lol(dm: CellDumpMetaLoL | CellDumpMetaFlat) -> bool:
    ex = dm["experiment"]
    return bool(ex) and isinstance(ex[0], list)


def _pack_val_dump_meta(meta: pd.DataFrame, rows: np.ndarray) -> CellDumpMetaChunk:
    """Build one chunk's metadata as Python lists (no numpy allocation for the lists)."""
    sub = meta.iloc[rows]
    seg_f = np.asarray(
        pd.to_numeric(sub["segmentation_id"], errors="coerce"),
        dtype=np.float64,
    )
    seg_int = np.nan_to_num(seg_f, nan=-1.0).astype(np.int64)
    chix_f = np.asarray(
        pd.to_numeric(sub["index"], errors="coerce"),
        dtype=np.float64,
    )
    chix_int = np.nan_to_num(chix_f, nan=-1.0).astype(np.int64)
    return {
        "experiment": sub["experiment"].astype("string").fillna("").tolist(),
        "well": sub["well"].astype("string").fillna("").tolist(),
        "x_pheno": [float(x) for x in sub["x_pheno"].tolist()],
        "y_pheno": [float(x) for x in sub["y_pheno"].tolist()],
        "segmentation_id": [int(x) for x in seg_int.tolist()],
        "channel_type": sub["channel_type"].astype("string").fillna("").tolist(),
        "index": [int(x) for x in chix_int.tolist()],
    }


def _chunks_to_lol(chunks: list[CellDumpMetaChunk]) -> CellDumpMetaLoL:
    """Many parquet chunks for one (g,c,e) → column → [seg0, seg1, …] (extend-style concat)."""
    cols = chunks[0].keys()
    return {col: [list(c[col]) for c in chunks] for col in cols}


def _merge_lol_for_experiments(meta_list: list[CellDumpMetaLoL]) -> CellDumpMetaLoL:
    """Concat along embedding axis: extend each column's list-of-lists with more segments."""
    cols = meta_list[0].keys()
    out: CellDumpMetaLoL = {col: [] for col in cols}
    for m in meta_list:
        for col in cols:
            out[col].extend(m[col])
    return out


def _permute_meta_lol(m: CellDumpMetaLoL, perm: torch.Tensor) -> CellDumpMetaLoL:
    """Flatten, permute, store as a single segment (still list-of-lists: one inner list)."""
    flat = {col: [x for seg in m[col] for x in seg] for col in m}
    idx = perm.detach().cpu().long().tolist()
    return {col: [[flat[col][i] for i in idx]] for col in m}


def _merge_experiments_val_with_meta(
    gce: dict[tuple[int, int, int], torch.Tensor],
    meta_gce: dict[tuple[int, int, int], CellDumpMetaLoL],
    max_cells_per_group: int | None,
    desc: str,
) -> tuple[CellIndex, dict[tuple[int, int], CellDumpMetaLoL]]:
    """Like merging val (gene,ch,exp)→(gene,ch), keeping per-cell metadata aligned."""
    gc_tensor_lists: dict[tuple[int, int], list[torch.Tensor]] = {}
    gc_meta_lists: dict[tuple[int, int], list[CellDumpMetaLoL]] = {}
    for (g_idx, ch_idx, _exp_idx), t in gce.items():
        gc_key = (g_idx, ch_idx)
        gc_tensor_lists.setdefault(gc_key, []).append(t)
        gc_meta_lists.setdefault(gc_key, []).append(meta_gce[(g_idx, ch_idx, _exp_idx)])

    index: CellIndex = {}
    meta_index: dict[tuple[int, int], CellDumpMetaLoL] = {}
    n_capped = 0
    for gc_key, tensors in tqdm(
        gc_tensor_lists.items(), desc=desc, total=len(gc_tensor_lists), unit="group"
    ):
        metas = gc_meta_lists[gc_key]
        combined = torch.cat(tensors) if len(tensors) > 1 else tensors[0]
        combined_meta = _merge_lol_for_experiments(metas)
        assert len(combined) == _meta_lol_n_cells(combined_meta), gc_key
        if max_cells_per_group is not None and len(combined) > max_cells_per_group:
            perm = torch.randperm(len(combined))[:max_cells_per_group]
            combined = combined[perm]
            combined_meta = _permute_meta_lol(combined_meta, perm)
            n_capped += 1
        index[gc_key] = combined
        meta_index[gc_key] = combined_meta
    if max_cells_per_group is not None and n_capped > 0:
        print(f"  Capped {n_capped} groups to {max_cells_per_group} cells")
    return index, meta_index


class PerturbationDataset(Dataset):
    """Each item is one perturbation → sample n_cells per channel.

    Returns dicts keyed by channel index since each channel can have a
    different number of sampled cells.
    """

    def __init__(
        self,
        cell_index: CellIndex,
        emb_dim: int,
        gene_to_idx: dict[str, int],
        n_channels: int,
        perturbation_list: list[str],
        n_cells_per_channel: dict[int, int],
        channels_subset: list[int] | None = None,
        channel_drop_fraction: float = 0.0,
        protected_channels: set[int] | None = None,
        label_remap: dict[int, int] | None = None,
    ):
        self.cell_index = cell_index
        self.emb_dim = emb_dim
        self.gene_to_idx = gene_to_idx
        self.n_channels = n_channels
        self.perturbation_list = perturbation_list
        self.n_cells_per_channel = n_cells_per_channel
        self.active_channels = (
            channels_subset if channels_subset is not None else list(range(n_channels))
        )
        self.channel_drop_fraction = channel_drop_fraction
        self.protected_channels = protected_channels or set()
        self._droppable_channels = [
            ch for ch in self.active_channels if ch not in self.protected_channels
        ]
        self.label_remap = label_remap

    def __len__(self) -> int:
        return len(self.perturbation_list)

    def __getitem__(
        self, idx: int
    ) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor], int]:
        """Returns (cell_embs, masks, label).

        cell_embs: {ch_idx: (N_ch, D)}
        masks: {ch_idx: (N_ch,) bool} — True = valid, False = padding.
        """
        gene_name = self.perturbation_list[idx]
        g_idx = self.gene_to_idx[gene_name]

        dropped: set[int] = set()
        if self.channel_drop_fraction > 0 and self._droppable_channels:
            n_drop = int(len(self._droppable_channels) * self.channel_drop_fraction)
            if n_drop > 0:
                dropped = set(random.sample(self._droppable_channels, n_drop))

        cell_embs: dict[int, torch.Tensor] = {}
        masks: dict[int, torch.Tensor] = {}
        for ch_idx in self.active_channels:
            n_cells = self.n_cells_per_channel[ch_idx]
            if ch_idx in dropped:
                cell_embs[ch_idx] = torch.zeros(n_cells, self.emb_dim)
                masks[ch_idx] = torch.zeros(n_cells, dtype=torch.bool)
                continue
            pool = self.cell_index.get((g_idx, ch_idx))
            if pool is not None and len(pool) > 0:
                n_available = len(pool)
                if n_available >= n_cells:
                    selected = torch.randperm(n_available)[:n_cells]
                    cell_embs[ch_idx] = pool[selected]
                    masks[ch_idx] = torch.ones(n_cells, dtype=torch.bool)
                else:
                    padded = torch.zeros(n_cells, self.emb_dim)
                    perm = torch.randperm(n_available)
                    padded[:n_available] = pool[perm]
                    cell_embs[ch_idx] = padded
                    mask = torch.zeros(n_cells, dtype=torch.bool)
                    mask[:n_available] = True
                    masks[ch_idx] = mask
            else:
                cell_embs[ch_idx] = torch.zeros(n_cells, self.emb_dim)
                masks[ch_idx] = torch.zeros(n_cells, dtype=torch.bool)

        label = self.label_remap[g_idx] if self.label_remap is not None else g_idx
        return cell_embs, masks, label


ChannelGroup = tuple[list[int], torch.Tensor, torch.Tensor]
"""(channel_indices, embs (B, G, N, D), masks (B, G, N))."""

CollatedBatch = tuple[list[ChannelGroup], torch.Tensor]
"""(groups, labels)."""


def collate_perturbation(
    batch: list[tuple[dict[int, torch.Tensor], dict[int, torch.Tensor], int]],
) -> CollatedBatch:
    """Group channels with the same n_cells into fused tensors.

    Returns a list of :data:`ChannelGroup` tuples plus labels. Each group
    contains channels that share the same sequence length so the cell
    encoder can process them in a single batched call.
    """
    all_channels = sorted(batch[0][0].keys())

    n_cells_map: dict[int, list[int]] = {}
    for ch in all_channels:
        n = batch[0][0][ch].shape[0]
        n_cells_map.setdefault(n, []).append(ch)

    groups: list[ChannelGroup] = []
    for _n_cells, ch_indices in n_cells_map.items():
        embs = torch.stack(
            [torch.stack([item[0][ch] for ch in ch_indices], dim=0) for item in batch]
        )
        masks = torch.stack(
            [torch.stack([item[1][ch] for ch in ch_indices], dim=0) for item in batch]
        )
        groups.append((ch_indices, embs, masks))

    labels = torch.tensor([item[2] for item in batch], dtype=torch.long)
    return groups, labels


class RepeatSampler(Sampler[int]):
    """Repeats each index exactly ``multiplier`` times per epoch, shuffled."""

    def __init__(self, n_items: int, multiplier: int = 1):
        self.n_items = n_items
        self.multiplier = multiplier

    def __len__(self) -> int:
        return self.n_items * self.multiplier

    def __iter__(self):
        indices = list(range(self.n_items)) * self.multiplier
        random.shuffle(indices)
        yield from indices


# ---------------------------------------------------------------------------
# Mixed-channel mode
# ---------------------------------------------------------------------------


def _compute_ch_indices(ch_ids: torch.Tensor) -> dict[int, torch.Tensor]:
    """Group cell positions by channel: ch_idx → LongTensor of indices into the pool."""
    if len(ch_ids) == 0:
        return {}
    unique = torch.unique(ch_ids).tolist()
    return {int(c): (ch_ids == c).nonzero(as_tuple=False).squeeze(-1) for c in unique}


def _normalize_cps_choices(
    channels_per_set: int | list[int | None] | None,
) -> list[int | None]:
    """Normalize ``channels_per_set`` to a list of per-set choices.

    A single int or None acts as a one-element list. A list value is used as-is,
    and ``None`` entries within it represent "use all channels for this set".
    """
    if channels_per_set is None:
        return [None]
    if isinstance(channels_per_set, int):
        return [channels_per_set]
    return list(channels_per_set)


class MixedChannelDataset(Dataset):
    """Each item samples n_cells across all channels proportionally.

    Tracks per-cell channel IDs so the model can apply channel conditioning.

    ``channels_per_set`` controls how many channels each set is sampled from:

        - ``None`` (default): no subsetting — sample from cells across all channels.
        - ``int`` (e.g. ``3``): every set picks exactly that many random channels.
        - ``list[int | None]`` (e.g. ``[1, 2, None]``): every set independently
          picks one value from the list; ``None`` in the list means "all channels"
          for that set.

    If a gene has fewer channels than the chosen value, all available are used.
    Applies to whichever dataset (train and/or val) it is set on.
    """

    def __init__(
        self,
        cell_index: CellIndex,
        emb_dim: int,
        gene_to_idx: dict[str, int],
        perturbation_list: list[str],
        n_cells: int,
        cell_dump_index: dict[tuple[int, int], CellDumpMetaLoL] | None = None,
        label_remap: dict[int, int] | None = None,
        replacement: bool = True,
        channels_per_set: int | list[int | None] | None = None,
    ):
        self.emb_dim = emb_dim
        self.gene_to_idx = gene_to_idx
        self.perturbation_list = perturbation_list
        self.n_cells = n_cells
        self.label_remap = label_remap
        self.replacement = replacement
        self.channels_per_set = channels_per_set
        self._cps_choices = _normalize_cps_choices(channels_per_set)

        self._gene_pools: dict[int, torch.Tensor] = {}
        self._gene_ch_ids: dict[int, torch.Tensor] = {}
        self._gene_ch_indices: dict[int, dict[int, torch.Tensor]] = {}
        self._gene_dump_meta: dict[int, CellDumpMetaLoL | CellDumpMetaFlat] | None = (
            None
        )
        if cell_dump_index is not None:
            self._gene_dump_meta = {}

        for g_name in tqdm(perturbation_list, desc="Building gene pools", unit="gene"):
            g_idx = gene_to_idx[g_name]
            if g_idx in self._gene_pools:
                continue
            emb_chunks: list[torch.Tensor] = []
            ch_chunks: list[torch.Tensor] = []
            meta_chunks: list[CellDumpMetaLoL] = []
            for key, t in cell_index.items():
                if key[0] == g_idx:
                    emb_chunks.append(t)
                    ch_chunks.append(torch.full((len(t),), key[1], dtype=torch.long))
                    if cell_dump_index is not None:
                        m = cell_dump_index[key]
                        assert len(t) == _meta_lol_n_cells(m), (
                            key,
                            len(t),
                            _meta_lol_n_cells(m),
                        )
                        meta_chunks.append(m)
            if emb_chunks:
                self._gene_pools[g_idx] = torch.cat(emb_chunks)
                self._gene_ch_ids[g_idx] = torch.cat(ch_chunks)
                if cell_dump_index is not None:
                    assert self._gene_dump_meta is not None
                    self._gene_dump_meta[g_idx] = _merge_lol_for_experiments(
                        meta_chunks
                    )
            else:
                self._gene_pools[g_idx] = torch.zeros(0, emb_dim)
                self._gene_ch_ids[g_idx] = torch.zeros(0, dtype=torch.long)
                if cell_dump_index is not None:
                    assert self._gene_dump_meta is not None
                    self._gene_dump_meta[g_idx] = {k: [] for k in CELL_DUMP_KEYS}
            self._gene_ch_indices[g_idx] = _compute_ch_indices(self._gene_ch_ids[g_idx])

        self._t_randperm = 0.0
        self._t_index = 0.0
        self._t_total = 0.0
        self._n_calls = 0

    def reset_timers(self) -> None:
        self._t_randperm = 0.0
        self._t_index = 0.0
        self._t_total = 0.0
        self._n_calls = 0

    def print_timers(self) -> None:
        print(
            f"  [__getitem__] {self._n_calls} calls: "
            f"total={self._t_total:.3f}s  "
            f"randperm={self._t_randperm:.3f}s  "
            f"index={self._t_index:.3f}s  "
            f"other={self._t_total - self._t_randperm - self._t_index:.3f}s"
        )

    def set_n_cells(self, n_cells: int) -> None:
        self.n_cells = n_cells

    def __len__(self) -> int:
        return len(self.perturbation_list)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Returns (cell_embs, ch_ids, mask, label).

        cell_embs: (n_cells, D)
        ch_ids: (n_cells,) long — channel index per cell
        mask: (n_cells,) bool
        """
        t_start = time.perf_counter()
        g_idx = self.gene_to_idx[self.perturbation_list[idx]]
        label = self.label_remap[g_idx] if self.label_remap is not None else g_idx
        pool = self._gene_pools[g_idx]
        ch_ids = self._gene_ch_ids[g_idx]

        if len(pool) > 0:
            cps_choice = random.choice(self._cps_choices)
            if cps_choice is not None:
                ch_indices = self._gene_ch_indices[g_idx]
                avail = list(ch_indices.keys())
                n_sel = min(cps_choice, len(avail))
                if n_sel < len(avail):
                    sel = random.sample(avail, n_sel)
                    sel_idx = torch.cat([ch_indices[c] for c in sel])
                    pool = pool[sel_idx]
                    ch_ids = ch_ids[sel_idx]

        n_available = len(pool)

        if n_available >= self.n_cells:
            t0 = time.perf_counter()
            if self.replacement:
                selected = torch.randint(n_available, (self.n_cells,))
            else:
                selected = torch.randperm(n_available)[: self.n_cells]
            t1 = time.perf_counter()
            embs_out = pool[selected]
            ch_out = ch_ids[selected]
            t2 = time.perf_counter()
            self._t_randperm += t1 - t0
            self._t_index += t2 - t1
            self._t_total += t2 - t_start
            self._n_calls += 1
            return (
                embs_out,
                ch_out,
                torch.ones(self.n_cells, dtype=torch.bool),
                label,
            )

        padded = torch.zeros(self.n_cells, self.emb_dim)
        padded_ch = torch.zeros(self.n_cells, dtype=torch.long)
        if n_available > 0:
            padded[:n_available] = pool
            padded_ch[:n_available] = ch_ids
        mask = torch.zeros(self.n_cells, dtype=torch.bool)
        mask[:n_available] = True
        self._t_total += time.perf_counter() - t_start
        self._n_calls += 1
        return padded, padded_ch, mask, label


class MixedChannelClassifier(nn.Module):
    """Single-level set transformer: ISAB pools all cells → classify.

    Channel conditioning modes:
        - "add": add learned channel embedding to each cell.
        - "concat": concatenate channel embedding with projected cell embedding
          and linearly project back to d_model.
        - "adaln": modulate LayerNorm via adaptive scale/shift from the *mean*
          channel embedding (global conditioning, shape (B, D)).
        - "adaln-token": modulate LayerNorm via adaptive scale/shift from
          *per-token* channel embeddings (shape (B, N, D)). Each cell gets
          conditioning from its own channel rather than an average.
        - "none": no channel conditioning.
    """

    def __init__(
        self,
        emb_dim: int,
        n_classes: int,
        n_channels: int = 0,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 2,
        n_inducing: int = 32,
        d_ff: int | None = None,
        dropout: float = 0.1,
        cosine_classifier: bool = False,
        channel_conditioning: str = "none",
        pool_type: str = "pma",
        grad_checkpoint: bool = False,
    ):
        super().__init__()
        assert channel_conditioning in ("add", "concat", "adaln", "adaln-token", "none")
        self.channel_conditioning = channel_conditioning

        self.input_proj = nn.Linear(emb_dim, d_model)
        if channel_conditioning in ("add", "concat", "adaln", "adaln-token"):
            self.channel_embeddings = nn.Embedding(n_channels, d_model)
        if channel_conditioning == "concat":
            self.concat_proj = nn.Linear(2 * d_model, d_model)

        use_adaln = channel_conditioning in ("adaln", "adaln-token")
        self.encoder = ISABBlock(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            n_inducing=n_inducing,
            n_pool_seeds=1,
            d_ff=d_ff,
            adaln=use_adaln,
            dropout=dropout,
            pool_type=pool_type,
            grad_checkpoint=grad_checkpoint,
        )
        classifier: nn.Module
        if cosine_classifier:
            classifier = CosineClassifier(d_model, n_classes)
        else:
            classifier = nn.Linear(d_model, n_classes)
        self.head = nn.Sequential(nn.Dropout(dropout), classifier)

    def encode(
        self,
        embs: torch.Tensor,
        ch_ids: torch.Tensor,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        """Pool a set of cells into the penultimate representation.

        Args:
            embs: (B, N, D) cell embeddings.
            ch_ids: (B, N) long — per-cell channel index.
            masks: (B, N) bool — True = valid.

        Returns:
            Pooled set vector of shape (B, d_model) — the input to the head.
        """
        x = self.input_proj(embs)

        cond = None
        if self.channel_conditioning == "add":
            x = x + self.channel_embeddings(ch_ids)
        elif self.channel_conditioning == "concat":
            x = self.concat_proj(
                torch.cat([x, self.channel_embeddings(ch_ids)], dim=-1)
            )
        elif self.channel_conditioning == "adaln":
            cond = self.channel_embeddings(ch_ids).mean(dim=1)
        elif self.channel_conditioning == "adaln-token":
            cond = self.channel_embeddings(ch_ids)

        return self.encoder(x, key_padding_mask=~masks, cond=cond)

    def forward(
        self,
        embs: torch.Tensor,
        ch_ids: torch.Tensor,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            embs: (B, N, D) cell embeddings.
            ch_ids: (B, N) long — per-cell channel index.
            masks: (B, N) bool — True = valid.

        Returns:
            Logits of shape (B, n_classes).
        """
        return self.head(self.encode(embs, ch_ids, masks))


def train_one_epoch_mixed(
    model: MixedChannelClassifier,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_grad_norm: float | None = None,
) -> tuple[float, float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    grad_norm_sum = 0.0
    n_steps = 0

    t_data_sum = 0.0
    t_transfer_sum = 0.0
    t_forward_sum = 0.0
    t_backward_sum = 0.0
    t_step_sum = 0.0

    print(
        f"  [loader] len={len(loader)} batches, dataset={len(loader.dataset)} samples"
    )

    t_batch_start = time.perf_counter()
    for embs, ch_ids, masks, labels in loader:
        if n_steps == 0:
            print(
                f"  [batch shape] embs={list(embs.shape)} ch_ids={list(ch_ids.shape)} masks={list(masks.shape)} labels={list(labels.shape)}"
            )
        t_data = time.perf_counter()

        embs = embs.to(device)
        ch_ids = ch_ids.to(device)
        masks = masks.to(device)
        labels = labels.to(device)
        t_transfer = time.perf_counter()

        logits = model(embs, ch_ids, masks)
        loss = F.cross_entropy(logits, labels)
        t_forward = time.perf_counter()

        optimizer.zero_grad()
        loss.backward()
        if max_grad_norm is not None:
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        grad_norm_sum += _grad_norm(model)
        t_backward = time.perf_counter()

        n_steps += 1
        optimizer.step()
        t_step = time.perf_counter()

        t_data_sum += t_data - t_batch_start
        t_transfer_sum += t_transfer - t_data
        t_forward_sum += t_forward - t_transfer
        t_backward_sum += t_backward - t_forward
        t_step_sum += t_step - t_backward

        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=-1) == labels).sum().item()
        total += labels.size(0)

        t_batch_start = time.perf_counter()

    print(
        f"  [timing] {n_steps} steps: "
        f"data={t_data_sum:.2f}s  "
        f"transfer={t_transfer_sum:.2f}s  "
        f"forward={t_forward_sum:.2f}s  "
        f"backward={t_backward_sum:.2f}s  "
        f"optim_step={t_step_sum:.2f}s"
    )

    if total == 0:
        raise ValueError(
            "Mixed-channel training epoch saw zero samples (no batches). "
            "Check for an empty dataset or train DataLoader drop_last dropping "
            "all indices when len(dataset)*train_multiplier < batch_size."
        )

    return total_loss / total, correct / total, grad_norm_sum / max(n_steps, 1)


@torch.no_grad()
def evaluate_mixed(
    model: MixedChannelClassifier,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for embs, ch_ids, masks, labels in loader:
        embs = embs.to(device)
        ch_ids = ch_ids.to(device)
        masks = masks.to(device)
        labels = labels.to(device)
        logits = model(embs, ch_ids, masks)
        loss = F.cross_entropy(logits, labels)

        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=-1) == labels).sum().item()
        total += labels.size(0)

    if total == 0:
        raise ValueError(
            "Mixed-channel eval saw zero samples (no batches). "
            "Check for an empty val dataset or val DataLoader configuration."
        )

    return total_loss / total, correct / total


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_label_map(
    label_map_path: str,
    gene_col: str = "gene_name",
    label_col: str = "pathway",
) -> dict[str, str]:
    """Load an external CSV that maps gene names to class labels (e.g. pathways).

    Empty/NaN values in ``gene_col`` are filled as ``"NTC"`` to match the
    ``fillna("NTC")`` convention used during data loading.
    """
    df = pd.read_csv(label_map_path)
    if gene_col not in df.columns:
        raise ValueError(f"Column {gene_col!r} not found in {label_map_path}")
    if label_col not in df.columns:
        raise ValueError(f"Column {label_col!r} not found in {label_map_path}")
    df[gene_col] = df[gene_col].fillna("NTC")
    return dict(zip(df[gene_col], df[label_col]))


def _build_channel_label(meta: pd.DataFrame) -> pd.Series:
    """Build channel label from biological annotations, falling back to ``name``.

    Rows with ``name == "Phase2D"`` are always labeled ``Phase2D`` regardless of
    annotation columns (some batches store sentinel strings like ``'no label'``
    instead of NULL there).  Otherwise: if either
    ``biological_annotation.organelle`` or ``biological_annotation.marker`` is
    non-null, the channel label is their concatenation (separated by ``_``,
    skipping nulls).  Otherwise falls back to the ``name`` column.
    """
    name = meta["name"]
    org = meta["biological_annotation.organelle"]
    marker = meta["biological_annotation.marker"]
    has_annotation = org.notna() | marker.notna()

    parts = org.fillna("").str.cat(marker.fillna(""), sep="_").str.strip("_")
    label = parts.where(has_annotation, name)
    return label.where(name != "Phase2D", "Phase2D")


def _resolve_label_map_genes(
    unique_genes: list[str],
    gene_to_label: dict[str, str],
    fallback_to_gene: bool,
) -> tuple[list[str], dict[str, str]]:
    """Resolve kept genes and their effective labels under an external label map.

    Parameters
    ----------
    unique_genes
        All gene names present in the dataset.
    gene_to_label
        Mapping from gene name to class label (e.g. EBI complex), from
        :func:`_load_label_map`.
    fallback_to_gene
        If True, genes absent from ``gene_to_label`` are kept and labeled by
        their own gene name (each forms its own class). If False, such genes are
        dropped (train only on genes present in the map).

    Returns
    -------
    tuple[list[str], dict[str, str]]
        ``(kept_genes, effective_gene_to_label)`` where every gene in
        ``kept_genes`` has an entry in ``effective_gene_to_label``.
    """
    mapped = [g for g in unique_genes if g in gene_to_label]
    if fallback_to_gene:
        effective = dict(gene_to_label)
        for g in unique_genes:
            effective.setdefault(g, g)
        return sorted(unique_genes), effective
    return sorted(mapped), gene_to_label


def _cell_stratify_val_mask(
    experiment: np.ndarray,
    well: np.ndarray,
    segmentation_id: np.ndarray,
    val_fraction: float,
    seed: int,
) -> np.ndarray:
    """Assign each unique cell to the val (True) or train (False) split.

    A cell is keyed by ``(experiment, well, segmentation_id)``. The assignment is
    a deterministic hash of that key, so every row belonging to a given cell
    receives the same assignment regardless of row ordering, which row group it
    came from, or which parquet (modality) it lives in. This prevents train/val
    leakage when a single cell contributes multiple embeddings (e.g. one row per
    fluorescent marker / channel, or the same cell imaged in phenotyping, Cell
    Painting and 4i).

    Parameters
    ----------
    experiment, well, segmentation_id
        Per-row cell identifier components (any dtype; coerced to a canonical
        string form — ``segmentation_id`` is normalized to an integer).
    val_fraction
        Target fraction of *cells* (not rows) assigned to val.
    seed
        Seed folded into the hash so the split is reproducible yet re-seedable.

    Returns
    -------
    np.ndarray
        Boolean mask, True where the row's cell is assigned to val.
    """
    seg_num = pd.Series(pd.to_numeric(pd.Series(segmentation_id), errors="coerce"))
    seg = seg_num.fillna(-1).astype(np.int64).astype(str)
    keys = (
        pd.Series(experiment)
        .astype(str)
        .str.cat([pd.Series(well).astype(str), seg], sep="|")
    )
    # Deterministic per-cell hash: factorize to unique cell keys, hash each once
    # (few uniques per call), then broadcast back to rows. blake2b is stable
    # across processes (unlike the salted builtin hash()).
    codes, uniques = pd.factorize(keys, sort=False)
    threshold = round(val_fraction * 1_000_000)
    unique_val = np.array(
        [
            int.from_bytes(
                hashlib.blake2b(f"{seed}|{k}".encode(), digest_size=8).digest(),
                "big",
            )
            % 1_000_000
            < threshold
            for k in uniques
        ],
        dtype=bool,
    )
    return unique_val[codes]


def load_data(
    parquet_entries: list[dict],
    max_row_groups: int | None,
    max_cells_per_group: int | None,
    val_fraction: float = 0.2,
    seed: int = 42,
    cell_stratify: bool = False,
    max_genes: int | None = None,
    max_channels: int | None = None,
    min_cells_per_group: int | None = None,
    min_cells_drop_val: bool = False,
    load_val_dump_metadata: bool = False,
    load_train_dump_metadata: bool = False,
    exclude_channel_names: list[str] | None = None,
    include_channel_names: list[str] | None = None,
    z_standardize: bool = True,
    z_standardize_control_only: bool = False,
) -> tuple[
    CellIndex,
    CellIndex,
    int,
    dict[str, int],
    dict[str, int],
    dict[tuple[int, int], CellDumpMetaLoL] | None,
    dict[tuple[int, int], CellDumpMetaLoL] | None,
]:
    """Load embeddings from parquet files into train/val :data:`CellIndex` dicts.

    Each row is randomly assigned to train or val as it is read, so no
    post-hoc pass over the full data is needed.  Null ``gene_name``
    values are filled as ``"NTC"`` (non-targeting control).

    Args:
        parquet_entries: List of dicts, each with ``"path"`` (str) and optional
            per-file filters ``"exclude_experiments"`` (list[str] | None) and
            ``"exclude_fluorescent_experiments"`` (list[str] | None).
        exclude_channel_names: If set, drop any row whose channel label (same string
            as :func:`_build_channel_label` / keys in ``channel_to_idx``) is in this
            list. Matching is exact and case-sensitive. Applied globally to all files.
        include_channel_names: If set, keep ONLY rows whose channel label is in this
            list (same string semantics as ``exclude_channel_names``). Applied before
            the exclude filter, so a channel in both lists is dropped. Use e.g.
            ``["Phase2D"]`` to train a single-channel phase-only classifier.
        z_standardize: If True (default), z-standardize embeddings per
            (channel, experiment). If False, skip standardization entirely
            and return raw embeddings.
        z_standardize_control_only: If True, compute z-standardization statistics
            (mean/std per channel×experiment) only from NTC (non-targeting control)
            cells. All cells are still standardized using these control-derived stats.
            Ignored when ``z_standardize`` is False.

    Returns:
        (train_index, val_index, emb_dim, gene_to_idx, channel_to_idx,
        val_cell_dump_meta, train_cell_dump_meta).
        Each dump meta is None unless the corresponding ``load_*_dump_metadata`` flag is True.
    """
    base_meta_cols = [
        "gene_name",
        "experiment",
        "name",
        "biological_annotation.organelle",
        "biological_annotation.marker",
    ]
    if load_val_dump_metadata or load_train_dump_metadata:
        base_meta_cols = base_meta_cols + list(VAL_DUMP_META_COLS)

    any_fluor_exclude = any(
        entry.get("exclude_fluorescent_experiments") for entry in parquet_entries
    )
    if any_fluor_exclude and "channel_type" not in base_meta_cols:
        base_meta_cols.append("channel_type")

    exclude_set: frozenset[str] | None = (
        frozenset(exclude_channel_names) if exclude_channel_names else None
    )
    if exclude_set is not None:
        print(f"Excluding channel labels (exact match): {sorted(exclude_set)}")
    include_set: frozenset[str] | None = (
        frozenset(include_channel_names) if include_channel_names else None
    )
    if include_set is not None:
        print(f"Including ONLY channel labels (exact match): {sorted(include_set)}")
    if not z_standardize:
        print("Z-standardization: DISABLED (returning raw embeddings)")
    elif z_standardize_control_only:
        print("Z-standardization: computing stats from NTC (control) cells only")

    rng = np.random.RandomState(seed)
    gene_to_idx: dict[str, int] = {}
    channel_to_idx: dict[str, int] = {}
    experiment_to_idx: dict[str, int] = {}
    train_buf: dict[tuple[int, int, int], list[torch.Tensor]] = {}
    val_buf: dict[tuple[int, int, int], list[torch.Tensor]] = {}
    val_dump_buf: dict[tuple[int, int, int], list[CellDumpMetaChunk]] = {}
    train_dump_buf: dict[tuple[int, int, int], list[CellDumpMetaChunk]] = {}
    ce_sums: dict[tuple[int, int], torch.Tensor] = {}
    ce_sq_sums: dict[tuple[int, int], torch.Tensor] = {}
    ce_counts: dict[tuple[int, int], int] = {}
    emb_dim: int | None = None
    total_rows = 0
    t0 = time.time()

    rg_tasks: list[tuple] = []
    for entry in parquet_entries:
        parquet_path = entry["path"]
        file_exclude_exp: frozenset[str] | None = (
            frozenset(entry["exclude_experiments"])
            if entry.get("exclude_experiments")
            else None
        )
        file_exclude_fluor: frozenset[str] | None = (
            frozenset(entry["exclude_fluorescent_experiments"])
            if entry.get("exclude_fluorescent_experiments")
            else None
        )
        file_col_remap: dict[str, str] = dict(entry.get("column_remap", {}))
        pf = pq.ParquetFile(parquet_path)
        n_rg_file = pf.metadata.num_row_groups
        if max_row_groups is not None:
            remaining = max_row_groups - len(rg_tasks)
            if remaining <= 0:
                break
            n_rg_file = min(n_rg_file, remaining)
        print(
            f"Loading {n_rg_file}/{pf.metadata.num_row_groups} "
            f"row groups from {parquet_path}"
        )
        if file_exclude_exp is not None:
            print(f"  Excluding experiments: {sorted(file_exclude_exp)}")
        if file_exclude_fluor is not None:
            print(
                f"  Excluding fluorescent channels from: {sorted(file_exclude_fluor)}"
            )
        schema_names = {f.name for f in pf.schema_arrow}
        _legacy = "index" not in schema_names and "channel_index" in schema_names
        fmc = list(base_meta_cols)
        if _legacy and "index" in fmc:
            fmc = ["channel_index" if c == "index" else c for c in fmc]
        remap_reverse: dict[str, str] = {}
        for canonical, source in file_col_remap.items():
            if source in schema_names:
                fmc = [source if c == canonical else c for c in fmc]
                remap_reverse[source] = canonical
        missing = frozenset(c for c in fmc if c not in schema_names)
        _optional_cols = frozenset(VAL_DUMP_META_COLS)
        required_missing = missing - _optional_cols
        if cell_stratify:
            # These are otherwise optional (part of VAL_DUMP_META_COLS) but are
            # mandatory for the cell-level split — fail loud rather than silently
            # collapsing every cell into one hash bucket via the -1 default.
            required_missing = required_missing | (
                missing & {"well", "segmentation_id"}
            )
        if required_missing:
            raise KeyError(
                f"Required columns missing from {parquet_path}: {sorted(required_missing)}. "
                f"Available: {sorted(schema_names)}"
            )
        if missing:
            fmc = [c for c in fmc if c in schema_names]
            print(
                f"  Columns missing from schema (will fill defaults): {sorted(missing)}"
            )
        for rg_idx in range(n_rg_file):
            rg_tasks.append(
                (
                    pf,
                    rg_idx,
                    fmc,
                    _legacy,
                    file_exclude_exp,
                    file_exclude_fluor,
                    missing,
                    remap_reverse,
                )
            )

    _MISSING_COL_DEFAULTS: dict[str, object] = {
        "x_pheno": float("nan"),
        "y_pheno": float("nan"),
        "segmentation_id": -1,
        "index": -1,
        "well": "",
        "channel_type": "",
    }

    n_total_rg = len(rg_tasks)
    for task_idx, (
        pf,
        rg_idx,
        file_meta_cols,
        _legacy_channel_index,
        exclude_exp_set,
        exclude_fluor_exp_set,
        missing_cols,
        col_remap,
    ) in enumerate(rg_tasks):
        table = pf.read_row_group(rg_idx)
        meta = table.select(file_meta_cols).to_pandas()
        if _legacy_channel_index and "channel_index" in meta.columns:
            meta = meta.rename(columns={"channel_index": "index"})
        if col_remap:
            meta = meta.rename(columns=col_remap)
        for col in missing_cols:
            meta[col] = _MISSING_COL_DEFAULTS.get(col, "")
        meta["gene_name"] = meta["gene_name"].astype("string").fillna("NTC")
        n_rows = len(meta)

        channel_label = _build_channel_label(meta)

        emb_col = table.column("embeddings")
        flat = emb_col.combine_chunks().values.to_numpy(zero_copy_only=False)
        dim = len(flat) // n_rows
        if emb_dim is None:
            emb_dim = dim
        emb_2d = flat.reshape(n_rows, dim)

        if include_set is not None:
            keep = channel_label.isin(include_set)
            keep_arr = keep.to_numpy()
            if not keep_arr.any():
                print(
                    f"  Row group {task_idx + 1}/{n_total_rg}: 0 rows after channel include, skipping"
                )
                del table, emb_col, flat, emb_2d
                continue
            if not keep_arr.all():
                meta = meta.loc[keep].reset_index(drop=True)
                channel_label = channel_label.loc[keep].reset_index(drop=True)
                emb_2d = emb_2d[keep_arr]
                n_rows = len(meta)

        if exclude_set is not None:
            keep = ~channel_label.isin(exclude_set)
            keep_arr = keep.to_numpy()
            if not keep_arr.any():
                print(
                    f"  Row group {task_idx + 1}/{n_total_rg}: 0 rows after channel exclude, skipping"
                )
                del table, emb_col, flat, emb_2d
                continue
            if not keep_arr.all():
                meta = meta.loc[keep].reset_index(drop=True)
                channel_label = channel_label.loc[keep].reset_index(drop=True)
                emb_2d = emb_2d[keep_arr]
                n_rows = len(meta)

        if exclude_exp_set is not None:
            keep_exp = ~meta["experiment"].isin(exclude_exp_set)
            keep_exp_arr = keep_exp.to_numpy()
            if not keep_exp_arr.any():
                print(
                    f"  Row group {task_idx + 1}/{n_total_rg}: 0 rows after experiment exclude, skipping"
                )
                del table, emb_col, flat, emb_2d
                continue
            if not keep_exp_arr.all():
                meta = meta.loc[keep_exp].reset_index(drop=True)
                channel_label = channel_label.loc[keep_exp].reset_index(drop=True)
                emb_2d = emb_2d[keep_exp_arr]
                n_rows = len(meta)

        if exclude_fluor_exp_set is not None:
            is_target_exp = meta["experiment"].isin(exclude_fluor_exp_set)
            is_fluorescent = meta["channel_type"] == "fluorescent"
            keep_fluor = ~(is_target_exp & is_fluorescent)
            keep_fluor_arr = keep_fluor.to_numpy()
            if not keep_fluor_arr.any():
                print(
                    f"  Row group {task_idx + 1}/{n_total_rg}: "
                    f"0 rows after fluorescent exclude, skipping"
                )
                del table, emb_col, flat, emb_2d
                continue
            if not keep_fluor_arr.all():
                meta = meta.loc[keep_fluor].reset_index(drop=True)
                channel_label = channel_label.loc[keep_fluor].reset_index(drop=True)
                emb_2d = emb_2d[keep_fluor_arr]
                n_rows = len(meta)

        genes = meta["gene_name"].values
        ch_labels = channel_label.values
        experiments = meta["experiment"].values
        for g in set(genes):
            if g not in gene_to_idx:
                gene_to_idx[g] = len(gene_to_idx)
        for c in set(ch_labels):
            if c not in channel_to_idx:
                channel_to_idx[c] = len(channel_to_idx)
        for e in set(experiments):
            if e not in experiment_to_idx:
                experiment_to_idx[e] = len(experiment_to_idx)

        gene_ids = np.array([gene_to_idx[g] for g in genes], dtype=np.int64)
        ch_ids = np.array([channel_to_idx[c] for c in ch_labels], dtype=np.int64)
        exp_ids = np.array([experiment_to_idx[e] for e in experiments], dtype=np.int64)
        n_ch = len(channel_to_idx)
        n_exp = len(experiment_to_idx)
        group_keys = gene_ids * (n_ch * n_exp) + ch_ids * n_exp + exp_ids

        if cell_stratify:
            # Assign whole cells (experiment, well, segmentation_id) to train/val
            # so all rows of a cell — across channels/markers and modalities —
            # stay on one side. No train/val leakage.
            is_val = _cell_stratify_val_mask(
                meta["experiment"].to_numpy(),
                meta["well"].to_numpy(),
                meta["segmentation_id"].to_numpy(),
                val_fraction,
                seed,
            )
        else:
            is_val = rng.random(n_rows) < val_fraction

        order = np.argsort(group_keys, kind="mergesort")
        sorted_keys = group_keys[order]
        split_points = np.flatnonzero(np.diff(sorted_keys)) + 1
        for chunk_indices in np.split(order, split_points):
            first = chunk_indices[0]
            key = (int(gene_ids[first]), int(ch_ids[first]), int(exp_ids[first]))
            ce_key = (key[1], key[2])
            chunk_val = is_val[chunk_indices]
            val_rows = chunk_indices[chunk_val]
            train_rows = chunk_indices[~chunk_val]
            if len(train_rows) > 0:
                train_chunk = torch.from_numpy(emb_2d[train_rows])
                train_buf.setdefault(key, []).append(train_chunk)
                use_for_stats = z_standardize and (
                    not z_standardize_control_only or genes[first] == "NTC"
                )
                if use_for_stats:
                    ce_sums[ce_key] = ce_sums.get(
                        ce_key, torch.zeros(dim)
                    ) + train_chunk.sum(0)
                    ce_sq_sums[ce_key] = ce_sq_sums.get(
                        ce_key, torch.zeros(dim)
                    ) + train_chunk.pow(2).sum(0)
                    ce_counts[ce_key] = ce_counts.get(ce_key, 0) + len(train_rows)
                if load_train_dump_metadata:
                    train_dump_buf.setdefault(key, []).append(
                        _pack_val_dump_meta(meta, train_rows)
                    )
            if len(val_rows) > 0:
                val_buf.setdefault(key, []).append(torch.from_numpy(emb_2d[val_rows]))
                if load_val_dump_metadata:
                    val_dump_buf.setdefault(key, []).append(
                        _pack_val_dump_meta(meta, val_rows)
                    )

        total_rows += n_rows
        elapsed = time.time() - t0
        print(
            f"  Row group {task_idx + 1}/{n_total_rg}: {n_rows:,} rows ({elapsed:.1f}s)"
        )

        del table, emb_col, flat, emb_2d

    if emb_dim is None:
        raise ValueError("No valid rows found in any parquet file.")

    if max_genes is not None and len(gene_to_idx) > max_genes:
        idx_to_gene = {v: k for k, v in gene_to_idx.items()}
        gene_cells: dict[int, int] = {}
        for (g_idx, _ch, _exp), chunks in [*train_buf.items(), *val_buf.items()]:
            gene_cells[g_idx] = gene_cells.get(g_idx, 0) + sum(
                c.shape[0] for c in chunks
            )
        top_genes = sorted(gene_cells, key=lambda g: gene_cells[g], reverse=True)[
            :max_genes
        ]
        keep = set(top_genes)
        train_buf = {k: v for k, v in train_buf.items() if k[0] in keep}
        val_buf = {k: v for k, v in val_buf.items() if k[0] in keep}
        old_to_new = {old: new for new, old in enumerate(sorted(keep))}
        gene_to_idx = {idx_to_gene[old]: new for old, new in old_to_new.items()}
        train_buf = {(old_to_new[k[0]], k[1], k[2]): v for k, v in train_buf.items()}
        val_buf = {(old_to_new[k[0]], k[1], k[2]): v for k, v in val_buf.items()}
        if load_val_dump_metadata:
            val_dump_buf = {k: v for k, v in val_dump_buf.items() if k[0] in keep}
            val_dump_buf = {
                (old_to_new[k[0]], k[1], k[2]): v for k, v in val_dump_buf.items()
            }
        if load_train_dump_metadata:
            train_dump_buf = {k: v for k, v in train_dump_buf.items() if k[0] in keep}
            train_dump_buf = {
                (old_to_new[k[0]], k[1], k[2]): v for k, v in train_dump_buf.items()
            }
        print(f"  Filtered to top {max_genes} genes by cell count")

    if max_channels is not None and len(channel_to_idx) > max_channels:
        idx_to_channel = {v: k for k, v in channel_to_idx.items()}
        ch_cells: dict[int, int] = {}
        for (_g, ch_idx, _exp), chunks in [*train_buf.items(), *val_buf.items()]:
            ch_cells[ch_idx] = ch_cells.get(ch_idx, 0) + sum(c.shape[0] for c in chunks)
        top_channels = sorted(ch_cells, key=lambda c: ch_cells[c], reverse=True)[
            :max_channels
        ]
        keep_ch = set(top_channels)
        train_buf = {k: v for k, v in train_buf.items() if k[1] in keep_ch}
        val_buf = {k: v for k, v in val_buf.items() if k[1] in keep_ch}
        old_to_new_ch = {old: new for new, old in enumerate(sorted(keep_ch))}
        channel_to_idx = {
            idx_to_channel[old]: new for old, new in old_to_new_ch.items()
        }
        train_buf = {(k[0], old_to_new_ch[k[1]], k[2]): v for k, v in train_buf.items()}
        val_buf = {(k[0], old_to_new_ch[k[1]], k[2]): v for k, v in val_buf.items()}
        if load_val_dump_metadata:
            val_dump_buf = {k: v for k, v in val_dump_buf.items() if k[1] in keep_ch}
            val_dump_buf = {
                (k[0], old_to_new_ch[k[1]], k[2]): v for k, v in val_dump_buf.items()
            }
        if load_train_dump_metadata:
            train_dump_buf = {
                k: v for k, v in train_dump_buf.items() if k[1] in keep_ch
            }
            train_dump_buf = {
                (k[0], old_to_new_ch[k[1]], k[2]): v for k, v in train_dump_buf.items()
            }
        ce_sums = {
            (old_to_new_ch[k[0]], k[1]): v
            for k, v in ce_sums.items()
            if k[0] in keep_ch
        }
        ce_sq_sums = {
            (old_to_new_ch[k[0]], k[1]): v
            for k, v in ce_sq_sums.items()
            if k[0] in keep_ch
        }
        ce_counts = {
            (old_to_new_ch[k[0]], k[1]): v
            for k, v in ce_counts.items()
            if k[0] in keep_ch
        }
        print(f"  Filtered to top {max_channels} channels by cell count")

    if min_cells_per_group is not None:
        gc_cells: dict[tuple[int, int], int] = {}
        for (g_idx, ch_idx, _exp), chunks in [*train_buf.items(), *val_buf.items()]:
            gc_key = (g_idx, ch_idx)
            gc_cells[gc_key] = gc_cells.get(gc_key, 0) + sum(c.shape[0] for c in chunks)
        drop_gc: set[tuple[int, int]] = {
            k for k, n in gc_cells.items() if n < min_cells_per_group
        }
        if drop_gc:
            train_buf = {
                k: v for k, v in train_buf.items() if (k[0], k[1]) not in drop_gc
            }
            if load_train_dump_metadata:
                train_dump_buf = {
                    k: v
                    for k, v in train_dump_buf.items()
                    if (k[0], k[1]) not in drop_gc
                }
            if min_cells_drop_val:
                val_buf = {
                    k: v for k, v in val_buf.items() if (k[0], k[1]) not in drop_gc
                }
                if load_val_dump_metadata:
                    val_dump_buf = {
                        k: v
                        for k, v in val_dump_buf.items()
                        if (k[0], k[1]) not in drop_gc
                    }
            print(
                f"  Dropped {len(drop_gc)} (gene, channel) groups"
                f" with < {min_cells_per_group} cells"
                f" (train{' + val' if min_cells_drop_val else ', val kept'})"
            )

    print(f"Total: {total_rows:,} rows, dim={emb_dim}")

    # Compute mean/std per (channel, experiment) from train moments
    ce_mean: dict[tuple[int, int], torch.Tensor] = {}
    ce_std: dict[tuple[int, int], torch.Tensor] = {}
    ch_fallback_mean: dict[int, torch.Tensor] = {}
    ch_fallback_std: dict[int, torch.Tensor] = {}
    if z_standardize:
        for ce_key in ce_sums:
            n = ce_counts[ce_key]
            mean = ce_sums[ce_key] / n
            ce_mean[ce_key] = mean
            ce_std[ce_key] = (
                (ce_sq_sums[ce_key] / n - mean**2).clamp(min=0).sqrt().clamp(min=1e-6)
            )
        for ch_idx in channel_to_idx.values():
            ch_means = [ce_mean[k] for k in ce_mean if k[0] == ch_idx]
            if ch_means:
                ch_fallback_mean[ch_idx] = torch.stack(ch_means).mean(0)
                ch_fallback_std[ch_idx] = torch.stack(
                    [ce_std[k] for k in ce_std if k[0] == ch_idx]
                ).mean(0)
    del ce_sums, ce_sq_sums, ce_counts

    # Concatenate chunks per (gene, channel, experiment)
    print("Concatenating per (gene, channel, experiment)...")

    def _concat_gce(
        buf: dict[tuple[int, int, int], list[torch.Tensor]],
        desc: str = "Concat",
    ) -> dict[tuple[int, int, int], torch.Tensor]:
        return {
            key: (torch.cat(chunks) if len(chunks) > 1 else chunks[0])
            for key, chunks in tqdm(
                buf.items(), desc=desc, total=len(buf), unit="group"
            )
        }

    train_gce = _concat_gce(train_buf, desc="Concat train")
    del train_buf
    val_gce = _concat_gce(val_buf, desc="Concat val")
    del val_buf

    val_dump_gce: dict[tuple[int, int, int], CellDumpMetaLoL] | None = None
    if load_val_dump_metadata:
        val_dump_gce = {}
        for key, chunks in tqdm(
            val_dump_buf.items(),
            desc="Concat val meta",
            total=len(val_dump_buf),
            unit="group",
        ):
            val_dump_gce[key] = _chunks_to_lol(chunks)
        del val_dump_buf

    train_dump_gce: dict[tuple[int, int, int], CellDumpMetaLoL] | None = None
    if load_train_dump_metadata:
        train_dump_gce = {}
        for key, chunks in tqdm(
            train_dump_buf.items(),
            desc="Concat train meta",
            total=len(train_dump_buf),
            unit="group",
        ):
            train_dump_gce[key] = _chunks_to_lol(chunks)
        del train_dump_buf

    # Build (ch, exp) → list of gce keys for efficient lookup
    train_by_ce: dict[tuple[int, int], list[tuple[int, int, int]]] = {}
    for key in train_gce:
        train_by_ce.setdefault((key[1], key[2]), []).append(key)
    val_by_ce: dict[tuple[int, int], list[tuple[int, int, int]]] = {}
    for key in val_gce:
        val_by_ce.setdefault((key[1], key[2]), []).append(key)

    if z_standardize:
        # Z-standardize by looping over (channel, experiment) pairs
        ce_pairs = sorted(set(train_by_ce.keys()) | set(val_by_ce.keys()))
        for ch_idx, exp_idx in tqdm(ce_pairs, desc="Z-standardize", unit="(ch,exp)"):
            ce_key = (ch_idx, exp_idx)
            mean = ce_mean.get(
                ce_key, ch_fallback_mean.get(ch_idx, torch.zeros(emb_dim))
            )
            std = ce_std.get(ce_key, ch_fallback_std.get(ch_idx, torch.ones(emb_dim)))
            for gce_key in train_by_ce.get(ce_key, []):
                train_gce[gce_key].sub_(mean).div_(std)
            for gce_key in val_by_ce.get(ce_key, []):
                val_gce[gce_key].sub_(mean).div_(std)

        n_ce_groups = len(ce_mean)
        stats_source = "NTC-only" if z_standardize_control_only else "all-train"
        print(
            f"  Z-standardized per (channel, experiment): "
            f"{n_ce_groups} groups (stats from {stats_source})"
        )
    else:
        print("  Skipping z-standardization (z_standardize=False)")

    # Merge across experiments: (gene, channel, experiment) → (gene, channel)
    def _merge_experiments(
        gce: dict[tuple[int, int, int], torch.Tensor],
        desc: str = "Merge",
    ) -> CellIndex:
        gc_groups: dict[tuple[int, int], list[torch.Tensor]] = {}
        for (g_idx, ch_idx, _exp_idx), t in gce.items():
            gc_groups.setdefault((g_idx, ch_idx), []).append(t)
        index: CellIndex = {}
        n_capped = 0
        for gc_key, tensors in tqdm(
            gc_groups.items(), desc=desc, total=len(gc_groups), unit="group"
        ):
            combined = torch.cat(tensors) if len(tensors) > 1 else tensors[0]
            if max_cells_per_group is not None and len(combined) > max_cells_per_group:
                combined = combined[torch.randperm(len(combined))[:max_cells_per_group]]
                n_capped += 1
            index[gc_key] = combined
        if max_cells_per_group is not None and n_capped > 0:
            print(f"  Capped {n_capped} groups to {max_cells_per_group} cells")
        return index

    if load_train_dump_metadata:
        assert train_dump_gce is not None
        train_index, train_cell_dump_meta = _merge_experiments_val_with_meta(
            train_gce,
            train_dump_gce,
            max_cells_per_group,
            desc="Merge train",
        )
        del train_gce, train_dump_gce
    else:
        train_index = _merge_experiments(train_gce, desc="Merge train")
        train_cell_dump_meta = None
        del train_gce

    if load_val_dump_metadata:
        assert val_dump_gce is not None
        val_index, val_cell_dump_meta = _merge_experiments_val_with_meta(
            val_gce,
            val_dump_gce,
            max_cells_per_group,
            desc="Merge val",
        )
        del val_gce, val_dump_gce
    else:
        val_index = _merge_experiments(val_gce, desc="Merge val")
        val_cell_dump_meta = None
        del val_gce

    gene_to_idx = dict(sorted(gene_to_idx.items(), key=lambda kv: kv[1]))
    channel_to_idx = dict(sorted(channel_to_idx.items(), key=lambda kv: kv[1]))
    total_train = sum(t.shape[0] for t in train_index.values())
    total_val = sum(t.shape[0] for t in val_index.values())
    print(
        f"  {len(gene_to_idx)} genes, {len(channel_to_idx)} channels"
        f" | train: {total_train:,} cells, val: {total_val:,} cells"
    )
    print(f"  Total load time: {time.time() - t0:.1f}s")

    return (
        train_index,
        val_index,
        emb_dim,
        gene_to_idx,
        channel_to_idx,
        val_cell_dump_meta,
        train_cell_dump_meta,
    )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def _grad_norm(model: nn.Module) -> float:
    total_sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_sq += p.grad.data.square().sum().item()
    return total_sq**0.5


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_grad_norm: float | None = None,
) -> tuple[float, float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    grad_norm_sum = 0.0
    n_steps = 0

    for groups, labels in loader:
        groups = [(ch, e.to(device), m.to(device)) for ch, e, m in groups]
        labels = labels.to(device)

        logits = model(groups)
        loss = F.cross_entropy(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        if max_grad_norm is not None:
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        grad_norm_sum += _grad_norm(model)
        n_steps += 1
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=-1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total, grad_norm_sum / max(n_steps, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for groups, labels in loader:
        groups = [(ch, e.to(device), m.to(device)) for ch, e, m in groups]
        labels = labels.to(device)

        logits = model(groups)
        loss = F.cross_entropy(logits, labels)

        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=-1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def _dump_dataset(
    ds: Dataset,
    gene_to_idx: dict[str, int],
    channel_to_idx: dict[str, int],
    emb_dim: int,
    dump_dir: str,
    label: str = "dataset",
) -> None:
    """Save a MixedChannelDataset's gene pools to disk: one .pt per gene + metadata.pt."""
    assert isinstance(ds, MixedChannelDataset)
    out = Path(dump_dir)
    out.mkdir(parents=True, exist_ok=True)

    idx_to_gene = {v: k for k, v in gene_to_idx.items()}

    for g_idx, pool in tqdm(
        ds._gene_pools.items(), desc=f"Dumping {label} genes", unit="gene"
    ):
        gene_name = idx_to_gene[g_idx]
        payload: dict = {
            "embeddings": pool,
            "channel_ids": ds._gene_ch_ids[g_idx],
        }
        if ds._gene_dump_meta is not None:
            dm = ds._gene_dump_meta[g_idx]
            if _is_cell_meta_lol(dm):
                payload[CELL_METADATA_DUMP_KEY] = dm
            elif not dm["experiment"]:
                payload[CELL_METADATA_DUMP_KEY] = {k: [] for k in CELL_DUMP_KEYS}
            else:
                flat: CellDumpMetaFlat = dm  # type: ignore[assignment]
                payload[CELL_METADATA_DUMP_KEY] = {k: [v] for k, v in flat.items()}
        torch.save(payload, out / f"{gene_name}.pt")

    torch.save(
        {
            "gene_to_idx": gene_to_idx,
            "channel_to_idx": channel_to_idx,
            "emb_dim": emb_dim,
            "perturbation_list": ds.perturbation_list,
            "n_cells": ds.n_cells,
        },
        out / "metadata.pt",
    )
    print(f"Dumped {label} dataset ({len(ds._gene_pools)} genes) to {out}")

    ds._gene_dump_meta = None


def load_dataset(
    dump_dir: str,
    n_cells: int | None = None,
    channels_per_set: int | list[int | None] | None = None,
) -> MixedChannelDataset:
    """Recreate a MixedChannelDataset from files written by :func:`_dump_dataset`.

    Per-gene ``.pt`` files may include ``cell_metadata`` (column -> list of segment lists).
    When present, it is flattened into ``_gene_dump_meta`` as one list per column.
    """
    root = Path(dump_dir)
    meta = torch.load(root / "metadata.pt", map_location="cpu", weights_only=False)

    gene_to_idx: dict[str, int] = meta["gene_to_idx"]
    idx_to_gene = {v: k for k, v in gene_to_idx.items()}
    emb_dim: int = meta["emb_dim"]
    perturbation_list: list[str] = meta["perturbation_list"]
    if n_cells is None:
        n_cells = meta["n_cells"]

    ds = MixedChannelDataset.__new__(MixedChannelDataset)
    ds.emb_dim = emb_dim
    ds.gene_to_idx = gene_to_idx
    ds.perturbation_list = perturbation_list
    ds.n_cells = n_cells
    ds.replacement = True
    ds.channels_per_set = channels_per_set
    ds._cps_choices = _normalize_cps_choices(channels_per_set)
    ds._gene_pools = {}
    ds._gene_ch_ids = {}
    ds._gene_ch_indices = {}
    ds._gene_dump_meta = None
    ds.label_remap = None
    ds._t_randperm = 0.0
    ds._t_index = 0.0
    ds._t_total = 0.0
    ds._n_calls = 0

    for g_idx, gene_name in tqdm(
        idx_to_gene.items(), desc="Loading genes", unit="gene"
    ):
        pt_path = root / f"{gene_name}.pt"
        if not pt_path.exists():
            ds._gene_pools[g_idx] = torch.zeros(0, emb_dim)
            ds._gene_ch_ids[g_idx] = torch.zeros(0, dtype=torch.long)
            ds._gene_ch_indices[g_idx] = {}
            continue
        data = torch.load(pt_path, map_location="cpu", weights_only=False)
        ds._gene_pools[g_idx] = data["embeddings"]
        ds._gene_ch_ids[g_idx] = data["channel_ids"]
        ds._gene_ch_indices[g_idx] = _compute_ch_indices(data["channel_ids"])
        if CELL_METADATA_DUMP_KEY in data:
            if ds._gene_dump_meta is None:
                ds._gene_dump_meta = {}
            lol = data[CELL_METADATA_DUMP_KEY]
            # Backwards compat: old dumps use "channel_index" instead of "index"
            if "channel_index" in lol and "index" not in lol:
                lol["index"] = lol.pop("channel_index")
            col_order = [c for c in CELL_DUMP_KEYS if c in lol]
            col_order += [c for c in lol if c not in col_order]
            ds._gene_dump_meta[g_idx] = {
                col: [x for seg in lol[col] for x in seg] for col in col_order
            }

    print(
        f"Loaded dataset from {root}: "
        f"{len(ds._gene_pools)} genes, {len(ds.perturbation_list)} perturbations, "
        f"n_cells={ds.n_cells}"
    )
    return ds


load_val_dataset = load_dataset


def _subset_train_cells(
    ds: MixedChannelDataset,
    max_train_cells: int,
    seed: int,
) -> tuple[int, int]:
    """Uniformly subsample a mixed-channel train dataset to ``max_train_cells`` cells.

    Cells are drawn at random across all genes (and channels), so the natural
    per-gene / per-channel distribution is preserved in expectation. Pass only
    the *training* dataset -- validation is always kept intact.

    Parameters
    ----------
    ds
        The mixed-channel training dataset to subsample in place.
    max_train_cells
        Target total number of training cells. If the dataset already has fewer
        cells, it is left unchanged.
    seed
        Seed for the global permutation, so the subset is reproducible.

    Returns
    -------
    tuple[int, int]
        ``(original_total, new_total)`` cell counts.
    """
    keys = list(ds._gene_pools.keys())
    sizes = [len(ds._gene_pools[k]) for k in keys]
    total = sum(sizes)
    if total <= max_train_cells:
        return total, total

    g = torch.Generator().manual_seed(seed)
    keep = torch.zeros(total, dtype=torch.bool)
    keep[torch.randperm(total, generator=g)[:max_train_cells]] = True

    offset = 0
    for k, sz in zip(keys, sizes):
        mask = keep[offset : offset + sz]
        offset += sz
        ds._gene_pools[k] = ds._gene_pools[k][mask]
        ds._gene_ch_ids[k] = ds._gene_ch_ids[k][mask]
        ds._gene_ch_indices[k] = _compute_ch_indices(ds._gene_ch_ids[k])
    # Per-cell dump metadata (if present) is only consumed for validation
    # analysis, never during training, so drop it rather than re-aligning.
    ds._gene_dump_meta = None
    return total, max_train_cells


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run(cfg: DictConfig) -> None:
    wandb_config = OmegaConf.to_container(cfg, resolve=True)
    run_name = wandb_config.get("name") if isinstance(wandb_config, dict) else None
    print(f"Resolved run name: {run_name!r}")
    # Default to offline/disabled so the script runs standalone with no W&B login;
    # set `wandb_mode: online` (+ wandb_project/entity) in the config to log to W&B.
    wandb_run = wandb.init(
        project=cfg.get("wandb_project", "set-classifier"),
        entity=cfg.get("wandb_entity", None),
        settings=wandb.Settings(silent=True),
        config=wandb_config,  # type: ignore[arg-type]
        name=run_name,
        mode=cfg.get("wandb_mode", "disabled"),
    )

    device = torch.device(
        cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")

    seed = cfg.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    mixed_mode: bool = cfg.get("mixed_channels_mode", False)
    dump_val_dir = cfg.get("dump_val_dir", None)
    dump_train_dir = cfg.get("dump_train_dir", None)
    load_train_dir = cfg.get("load_train_dir", None)
    load_val_dir = cfg.get("load_val_dir", None)

    for _dname, _dval in [
        ("dump_val_dir", dump_val_dir),
        ("dump_train_dir", dump_train_dir),
    ]:
        if _dval is not None:
            _dp = Path(_dval)
            if _dp.exists() and any(_dp.iterdir()):
                raise RuntimeError(
                    f"{_dname} {_dval!r} already exists and is not empty. "
                    f"Remove it or choose a different directory."
                )
            _dp.mkdir(parents=True, exist_ok=True)

    use_preloaded = load_train_dir is not None and load_val_dir is not None
    if use_preloaded and not mixed_mode:
        raise ValueError(
            "load_train_dir / load_val_dir require mixed_channels_mode=true"
        )

    load_val_dump_metadata = dump_val_dir is not None and mixed_mode
    load_train_dump_metadata = dump_train_dir is not None and mixed_mode

    # ---- Load data ----
    if use_preloaded:
        # Fast path: load pre-dumped datasets directly
        n_cells_cfg: int = cfg.get("n_cells_per_set", 500)
        print(f"Loading pre-dumped train dataset from {load_train_dir}")
        preloaded_train_ds = load_dataset(load_train_dir, n_cells=n_cells_cfg)
        print(f"Loading pre-dumped val dataset from {load_val_dir}")
        preloaded_val_ds = load_dataset(load_val_dir, n_cells=n_cells_cfg)

        train_meta = torch.load(
            Path(load_train_dir) / "metadata.pt", map_location="cpu", weights_only=False
        )
        gene_to_idx = train_meta["gene_to_idx"]
        channel_to_idx = train_meta["channel_to_idx"]
        emb_dim = train_meta["emb_dim"]

        train_index = None
        val_index = None
        val_cell_dump_meta = None
        train_cell_dump_meta = None
    else:
        preloaded_train_ds = None
        preloaded_val_ds = None

        val_frac = cfg.get("val_fraction", 0.2)
        exclude_ch_cfg = cfg.data.get("exclude_channel_names")
        exclude_channel_names = (
            list(exclude_ch_cfg) if exclude_ch_cfg is not None else None
        )
        include_ch_cfg = cfg.data.get("include_channel_names")
        include_channel_names = (
            list(include_ch_cfg) if include_ch_cfg is not None else None
        )

        parquet_entries: list[dict] = []
        for entry in cfg.data.parquet_entries:
            pe: dict = {"path": entry.path}
            exc_exp = entry.get("exclude_experiments")
            if exc_exp is not None:
                pe["exclude_experiments"] = list(exc_exp)
            exc_fluor = entry.get("exclude_fluorescent_experiments")
            if exc_fluor is not None:
                pe["exclude_fluorescent_experiments"] = list(exc_fluor)
            col_remap = entry.get("column_remap")
            if col_remap is not None:
                pe["column_remap"] = dict(col_remap)
            parquet_entries.append(pe)

        (
            train_index,
            val_index,
            emb_dim,
            gene_to_idx,
            channel_to_idx,
            val_cell_dump_meta,
            train_cell_dump_meta,
        ) = load_data(
            parquet_entries=parquet_entries,
            max_row_groups=cfg.data.get("max_row_groups", None),
            max_cells_per_group=cfg.data.get("max_cells_per_group", None),
            val_fraction=val_frac,
            seed=seed,
            cell_stratify=cfg.data.get("cell_stratify", False),
            max_genes=cfg.data.get("max_genes", None),
            max_channels=cfg.data.get("max_channels", None),
            min_cells_per_group=cfg.data.get("min_cells_per_group", None),
            min_cells_drop_val=cfg.data.get("min_cells_drop_val", False),
            load_val_dump_metadata=load_val_dump_metadata,
            load_train_dump_metadata=load_train_dump_metadata,
            exclude_channel_names=exclude_channel_names,
            include_channel_names=include_channel_names,
            z_standardize=cfg.data.get("z_standardize", True),
            z_standardize_control_only=cfg.data.get(
                "z_standardize_control_only", False
            ),
        )

    unique_genes = sorted(gene_to_idx.keys())
    n_channels = len(channel_to_idx)

    # ---- Optional label map (e.g. gene → pathway) ----
    label_map_path = cfg.data.get("label_map_path", None)
    label_to_idx: dict[str, int] | None = None
    label_remap: dict[int, int] | None = None
    if label_map_path is not None:
        gene_col = cfg.data.get("label_map_gene_col", "gene_name")
        label_col = cfg.data.get("label_map_label_col", "pathway")
        fallback_to_gene = cfg.data.get("label_map_fallback_to_gene", False)
        gene_to_label = _load_label_map(label_map_path, gene_col, label_col)
        n_before = len(unique_genes)
        unique_genes, gene_to_label = _resolve_label_map_genes(
            unique_genes, gene_to_label, fallback_to_gene
        )
        n_dropped = n_before - len(unique_genes)
        if n_dropped:
            print(f"  Label map: dropping {n_dropped} genes not in {label_map_path}")
        unique_labels = sorted(set(gene_to_label[g] for g in unique_genes))
        label_to_idx = {label: i for i, label in enumerate(unique_labels)}
        label_remap = {
            gene_to_idx[g]: label_to_idx[gene_to_label[g]] for g in unique_genes
        }
        n_classes = len(label_to_idx)
        print(
            f"  Label map: {len(unique_genes)} genes → {n_classes} classes "
            f"from {label_map_path}"
        )
        print(f"  Classes: {unique_labels}")
    else:
        n_classes = len(unique_genes)

    # ---- Print per-channel cell counts ----
    idx_to_channel = {v: k for k, v in channel_to_idx.items()}
    gene_indices = [gene_to_idx[g] for g in unique_genes]
    if train_index is not None and val_index is not None:
        cells_per_gene_channel: dict[int, list[int]] = {
            i: [] for i in range(n_channels)
        }
        for g_idx in gene_indices:
            for ch_idx in range(n_channels):
                n = 0
                if (g_idx, ch_idx) in train_index:
                    n += len(train_index[(g_idx, ch_idx)])
                if (g_idx, ch_idx) in val_index:
                    n += len(val_index[(g_idx, ch_idx)])
                cells_per_gene_channel[ch_idx].append(n)

        print(f"\nCells per channel ({n_channels} channels):")
        for ch_idx in range(n_channels):
            counts = cells_per_gene_channel[ch_idx]
            total = sum(counts)
            sorted_counts = sorted(counts)
            mid = len(sorted_counts) // 2
            median = (
                sorted_counts[mid]
                if len(sorted_counts) % 2 == 1
                else (sorted_counts[mid - 1] + sorted_counts[mid]) // 2
            )
            print(
                f"  {idx_to_channel[ch_idx]}: {total:,} cells "
                f"(per gene: min={min(counts):,}, median={median:,}, max={max(counts):,})"
            )

    train_multiplier = cfg.get("train_n_cell_sets_per_gene", 1)
    val_multiplier = cfg.get("val_n_cell_sets_per_gene", 1)
    batch_size = cfg.get("batch_size", 32)
    mcfg = cfg.model

    phase2d_val: bool = cfg.get("phase2d_val", True)
    phase2d_ch_idx: int | None = None
    if phase2d_val:
        for ch_name, ch_idx in channel_to_idx.items():
            if ch_name == "Phase2D":
                phase2d_ch_idx = ch_idx
                break

    val_phase2d_loader: DataLoader | None = None
    val_n_cells_list: list[int] = []
    mixed_val_ds: MixedChannelDataset | None = None
    mixed_val_p2d_ds: MixedChannelDataset | None = None

    if mixed_mode:
        # -- Mixed-channel mode: pool all channels into one set --
        n_cells: int = cfg.get("n_cells_per_set", 500)
        _cps_raw = cfg.get("channels_per_set", None)
        channels_per_set: int | list[int | None] | None
        if _cps_raw is None or isinstance(_cps_raw, int):
            channels_per_set = _cps_raw
        else:
            channels_per_set = list(_cps_raw)
        if channels_per_set is not None:
            print(
                f"  channels_per_set={channels_per_set}: each set picks a value "
                f"(None=all channels) and samples cells only from that many "
                f"randomly-chosen channels per gene (applied to both train and val)"
            )

        if preloaded_train_ds is not None and preloaded_val_ds is not None:
            # Fast path: datasets already loaded from dump dirs
            train_ds: Dataset = preloaded_train_ds
            val_ds: Dataset = preloaded_val_ds
            preloaded_train_ds.channels_per_set = channels_per_set
            preloaded_train_ds._cps_choices = _normalize_cps_choices(channels_per_set)
            preloaded_val_ds.channels_per_set = channels_per_set
            preloaded_val_ds._cps_choices = _normalize_cps_choices(channels_per_set)
            if label_remap is not None:
                preloaded_train_ds.label_remap = label_remap
                preloaded_val_ds.label_remap = label_remap
                # The dump's perturbation_list covers every dumped gene, but
                # label_remap only has entries for genes the new label map
                # covers. Drop the rest so __getitem__ never hits a KeyError.
                mapped_set = set(unique_genes)
                preloaded_train_ds.perturbation_list = [
                    g for g in preloaded_train_ds.perturbation_list if g in mapped_set
                ]
                preloaded_val_ds.perturbation_list = [
                    g for g in preloaded_val_ds.perturbation_list if g in mapped_set
                ]
        else:
            assert train_index is not None and val_index is not None

            def _median(vals: list[int]) -> int:
                s = sorted(vals)
                mid = len(s) // 2
                return s[mid] if len(s) % 2 == 1 else (s[mid - 1] + s[mid]) // 2

            # Stats: total cells per gene across all channels
            total_per_gene: list[int] = []
            phase2d_per_gene: list[int] = []
            for g_idx in gene_indices:
                total = 0
                p2d = 0
                for ch_idx in range(n_channels):
                    n = 0
                    if (g_idx, ch_idx) in train_index:
                        n += len(train_index[(g_idx, ch_idx)])
                    if (g_idx, ch_idx) in val_index:
                        n += len(val_index[(g_idx, ch_idx)])
                    total += n
                    if ch_idx == phase2d_ch_idx:
                        p2d = n
                total_per_gene.append(total)
                phase2d_per_gene.append(p2d)

            print(
                f"\nMixed-channel mode: {n_cells} cells per set"
                f"\n  Total cells per gene: "
                f"min={min(total_per_gene):,}, "
                f"median={_median(total_per_gene):,}, "
                f"max={max(total_per_gene):,}"
            )
            if phase2d_ch_idx is not None:
                print(
                    f"  Phase2D cells per gene: "
                    f"min={min(phase2d_per_gene):,}, "
                    f"median={_median(phase2d_per_gene):,}, "
                    f"max={max(phase2d_per_gene):,}"
                )

            train_ds = MixedChannelDataset(
                train_index,
                emb_dim,
                gene_to_idx,
                unique_genes,
                n_cells,
                cell_dump_index=train_cell_dump_meta,
                label_remap=label_remap,
                channels_per_set=channels_per_set,
            )
            val_ds = MixedChannelDataset(
                val_index,
                emb_dim,
                gene_to_idx,
                unique_genes,
                n_cells,
                cell_dump_index=val_cell_dump_meta,
                label_remap=label_remap,
                channels_per_set=channels_per_set,
            )

            if dump_train_dir is not None:
                _dump_dataset(
                    train_ds,
                    gene_to_idx,
                    channel_to_idx,
                    emb_dim,
                    dump_train_dir,
                    label="train",
                )
                train_cell_dump_meta = None
            if dump_val_dir is not None:
                _dump_dataset(
                    val_ds,
                    gene_to_idx,
                    channel_to_idx,
                    emb_dim,
                    dump_val_dir,
                    label="val",
                )
                val_cell_dump_meta = None

            del train_index, val_index
            gc.collect()

        # Optionally cap the TOTAL number of training cells (val kept full).
        assert isinstance(train_ds, MixedChannelDataset)
        max_train_cells = cfg.data.get("max_train_cells", None)
        if max_train_cells is not None:
            orig, new = _subset_train_cells(train_ds, int(max_train_cells), seed)
            print(
                f"Train-cell subset: {orig:,} -> {new:,} "
                f"(max_train_cells={int(max_train_cells):,})"
            )
            wandb.log({"data/train_cells_original": orig, "data/train_cells": new})
        else:
            n_train_cells = sum(len(p) for p in train_ds._gene_pools.values())
            print(f"Using full train set: {n_train_cells:,} cells")
            wandb.log({"data/train_cells": n_train_cells})

        # If len(dataset)*multiplier < batch_size, drop_last=True yields zero batches
        # (e.g. max_genes=1 with batch_size>1).
        _mixed_train_indices = len(train_ds) * train_multiplier
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=RepeatSampler(len(train_ds), train_multiplier),
            num_workers=cfg.get("num_workers", 0),
            drop_last=_mixed_train_indices >= batch_size,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            sampler=RepeatSampler(len(val_ds), val_multiplier),
            num_workers=0,
            pin_memory=True,
        )

        channel_conditioning = mcfg.get("channel_conditioning", "none")
        model: nn.Module = MixedChannelClassifier(
            emb_dim=emb_dim,
            n_classes=n_classes,
            n_channels=n_channels,
            d_model=mcfg.get("d_model", 256),
            n_heads=mcfg.get("n_heads", 4),
            n_layers=mcfg.get("n_layers_cell", 2),
            n_inducing=mcfg.get("n_inducing_cell", 32),
            d_ff=mcfg.get("d_ff", None),
            dropout=mcfg.get("dropout", 0.1),
            cosine_classifier=mcfg.get("cosine_classifier", False),
            channel_conditioning=channel_conditioning,
            pool_type=mcfg.get("pool_type", "pma"),
        ).to(device)

        _train_fn = train_one_epoch_mixed
        _eval_fn = evaluate_mixed

        val_p2d_ds: MixedChannelDataset | None = None
        if phase2d_ch_idx is not None and val_index is not None:
            val_p2d_index: CellIndex = {
                k: v for k, v in val_index.items() if k[1] == phase2d_ch_idx
            }
            if val_p2d_index:
                val_p2d_ds = MixedChannelDataset(
                    val_p2d_index,
                    emb_dim,
                    gene_to_idx,
                    unique_genes,
                    n_cells,
                    label_remap=label_remap,
                )
                val_phase2d_loader = DataLoader(
                    val_p2d_ds,
                    batch_size=batch_size,
                    sampler=RepeatSampler(len(val_p2d_ds), val_multiplier),
                    num_workers=0,
                    pin_memory=True,
                )

        val_n_cells_raw = cfg.get("val_n_cells_per_set", None)
        val_n_cells_list = list(val_n_cells_raw) if val_n_cells_raw else []
        mixed_val_ds = val_ds  # type: ignore[assignment]
        mixed_val_p2d_ds = val_p2d_ds

    else:
        # -- Per-channel mode (original) --
        assert train_index is not None and val_index is not None
        default_n_cells: int = cfg.get("n_cells_per_set", 500)
        overrides_raw = cfg.get("n_cells_per_set_overrides", {})
        overrides: dict[str, int] = dict(overrides_raw) if overrides_raw else {}
        idx_to_channel = {v: k for k, v in channel_to_idx.items()}
        n_cells_per_channel: dict[int, int] = {}
        for ch_idx in range(n_channels):
            ch_name = idx_to_channel[ch_idx]
            n_cells_per_channel[ch_idx] = overrides.get(ch_name, default_n_cells)
        if overrides:
            for ch_name, n in overrides.items():
                print(f"  n_cells_per_set override: {ch_name} = {n}")

        channel_drop_fraction: float = cfg.get("channel_drop_fraction", 0.0)
        protected_channels: set[int] = set()
        if phase2d_ch_idx is not None:
            protected_channels.add(phase2d_ch_idx)
        if channel_drop_fraction > 0:
            print(
                f"  Channel dropout: {channel_drop_fraction:.0%} of droppable channels"
                f" (protected: {[idx_to_channel[c] for c in sorted(protected_channels)]})"
            )

        train_ds = PerturbationDataset(
            train_index,
            emb_dim,
            gene_to_idx,
            n_channels,
            unique_genes,
            n_cells_per_channel,
            channel_drop_fraction=channel_drop_fraction,
            protected_channels=protected_channels,
            label_remap=label_remap,
        )
        val_ds = PerturbationDataset(
            val_index,
            emb_dim,
            gene_to_idx,
            n_channels,
            unique_genes,
            n_cells_per_channel,
            label_remap=label_remap,
        )
        _pc_train_indices = len(train_ds) * train_multiplier
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=RepeatSampler(len(train_ds), train_multiplier),
            collate_fn=collate_perturbation,
            num_workers=cfg.get("num_workers", 0),
            drop_last=_pc_train_indices >= batch_size,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            sampler=RepeatSampler(len(val_ds), val_multiplier),
            collate_fn=collate_perturbation,
            num_workers=0,
            pin_memory=True,
        )

        if phase2d_ch_idx is not None:
            val_phase2d_ds = PerturbationDataset(
                val_index,
                emb_dim,
                gene_to_idx,
                n_channels,
                unique_genes,
                n_cells_per_channel,
                channels_subset=[phase2d_ch_idx],
                label_remap=label_remap,
            )
            val_phase2d_loader = DataLoader(
                val_phase2d_ds,
                batch_size=batch_size,
                sampler=RepeatSampler(len(val_phase2d_ds), val_multiplier),
                collate_fn=collate_perturbation,
                num_workers=0,
                pin_memory=True,
            )
            print(f"Phase2D-only validation enabled (channel idx={phase2d_ch_idx})")
        else:
            print(
                "Warning: 'Phase2D' channel not found — skipping Phase2D-only validation"
            )

        model = SetClassifier(
            emb_dim=emb_dim,
            n_channels=n_channels,
            n_classes=n_classes,
            d_model=mcfg.get("d_model", 256),
            n_heads=mcfg.get("n_heads", 4),
            n_layers_cell=mcfg.get("n_layers_cell", 2),
            n_layers_channel=mcfg.get("n_layers_channel", 1),
            n_inducing_cell=mcfg.get("n_inducing_cell", 32),
            d_ff=mcfg.get("d_ff", None),
            dropout=mcfg.get("dropout", 0.1),
            channel_conditioning=mcfg.get("channel_conditioning", "add"),
            cosine_classifier=mcfg.get("cosine_classifier", False),
        ).to(device)

        _train_fn = train_one_epoch
        _eval_fn = evaluate

    n_gpus = torch.cuda.device_count()
    if n_gpus > 1:
        print(f"Using DataParallel across {n_gpus} GPUs")
        model = nn.DataParallel(model)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")
    wandb.log({"model/n_params": n_params})

    # ---- Optimizer & scheduler ----
    lr = cfg.get("learning_rate", 1e-3)
    wd = cfg.get("weight_decay", 1e-4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    n_epochs = cfg.get("num_epochs", 100)
    warmup_epochs = cfg.get("warmup_epochs", 5)

    def lr_schedule(epoch: int) -> float:
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, n_epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)
    max_grad_norm: float | None = cfg.get("max_grad_norm", None)

    eval_every = cfg.get("eval_every", 1)

    # ---- Training ----
    best_val_acc = 0.0
    # Per-N validation accuracies of the SELECTED (best primary-N) model, captured
    # at the epoch the best checkpoint is saved. Written to `metrics_out` if set.
    best_metrics: dict[str, float] = {}
    for epoch in range(n_epochs):
        t0 = time.time()

        if isinstance(train_loader.dataset, MixedChannelDataset):
            train_loader.dataset.reset_timers()

        train_loss, train_acc, grad_norm = _train_fn(
            model, train_loader, optimizer, device, max_grad_norm=max_grad_norm
        )

        if isinstance(train_loader.dataset, MixedChannelDataset):
            train_loader.dataset.print_timers()

        scheduler.step()
        elapsed = time.time() - t0

        log_dict: dict[str, float] = {
            "epoch": epoch + 1,
            "train/loss": train_loss,
            "train/accuracy": train_acc,
            "train/grad_norm": grad_norm,
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_time_s": elapsed,
        }

        do_eval = (epoch + 1) % eval_every == 0 or (epoch + 1) == n_epochs
        val_elapsed = 0.0
        if do_eval:
            t_val = time.time()
            val_loss, val_acc = _eval_fn(model, val_loader, device)
            log_dict["val/loss"] = val_loss
            log_dict["val/accuracy"] = val_acc

            if val_phase2d_loader is not None:
                p2d_loss, p2d_acc = _eval_fn(model, val_phase2d_loader, device)
                log_dict["val_phase2d/loss"] = p2d_loss
                log_dict["val_phase2d/accuracy"] = p2d_acc

            for nc in val_n_cells_list:
                assert mixed_val_ds is not None
                mixed_val_ds.set_n_cells(nc)
                nc_loss, nc_acc = _eval_fn(model, val_loader, device)
                log_dict[f"val_n{nc}/loss"] = nc_loss
                log_dict[f"val_n{nc}/accuracy"] = nc_acc
                if mixed_val_p2d_ds is not None and val_phase2d_loader is not None:
                    mixed_val_p2d_ds.set_n_cells(nc)
                    nc_p2d_loss, nc_p2d_acc = _eval_fn(
                        model, val_phase2d_loader, device
                    )
                    log_dict[f"val_phase2d_n{nc}/loss"] = nc_p2d_loss
                    log_dict[f"val_phase2d_n{nc}/accuracy"] = nc_p2d_acc

            if val_n_cells_list and mixed_val_ds is not None:
                mixed_val_ds.set_n_cells(n_cells)
                if mixed_val_p2d_ds is not None:
                    mixed_val_p2d_ds.set_n_cells(n_cells)

            val_elapsed = time.time() - t_val

        extra_parts = ""
        if do_eval:
            extra_parts += f"| val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
            if val_phase2d_loader is not None:
                extra_parts += f"| phase2d_acc={p2d_acc:.4f} "
            for nc in val_n_cells_list:
                extra_parts += f"| val_n{nc}_acc={log_dict[f'val_n{nc}/accuracy']:.4f} "

        print(
            f"Epoch {epoch + 1:3d}/{n_epochs} "
            f"| train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            + extra_parts
            + f"| lr={optimizer.param_groups[0]['lr']:.2e} "
            f"| train={elapsed:.1f}s val={val_elapsed:.1f}s"
        )

        wandb.log(log_dict)

        if do_eval and val_acc > best_val_acc:
            best_val_acc = val_acc
            best_metrics = {
                "val_accuracy": val_acc,
                **{
                    f"val_n{nc}_accuracy": log_dict[f"val_n{nc}/accuracy"]
                    for nc in val_n_cells_list
                },
            }
            save_path = cfg.get("save_path", "best_set_classifier.pt")
            state_dict = (
                model.module.state_dict()
                if isinstance(model, nn.DataParallel)
                else model.state_dict()
            )
            ckpt: dict = {
                "model_state_dict": state_dict,
                "gene_to_idx": gene_to_idx,
                "channel_to_idx": channel_to_idx,
                "config": wandb_config,
                "epoch": epoch + 1,
                "val_acc": val_acc,
            }
            if label_to_idx is not None:
                ckpt["label_to_idx"] = label_to_idx
                ckpt["label_remap"] = label_remap
            torch.save(ckpt, save_path)
            print(f"  Saved best model (val_acc={val_acc:.4f}) to {save_path}")
            if cfg.get("wandb_mode", "disabled") != "disabled":
                artifact = wandb.Artifact(
                    name=f"model-{wandb_run.id}",
                    type="model",
                    metadata={"val_acc": val_acc, "epoch": epoch + 1},
                )
                artifact.add_file(save_path)
                wandb_run.log_artifact(artifact)

    print(f"\nBest val accuracy: {best_val_acc:.4f}")
    wandb.log({"best_val_accuracy": best_val_acc})

    # Optionally persist the selected model's per-N validation accuracies so a
    # wrapping eval (e.g. CsPhenoEvaluator) can read them back without scraping
    # stdout / the wandb run.
    metrics_out = cfg.get("metrics_out", None)
    if metrics_out is not None:
        out_path = Path(metrics_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"n_cells_per_set": cfg.get("n_cells_per_set", None), **best_metrics}
        out_path.write_text(json.dumps(payload, indent=2))
        print(f"Wrote metrics to {out_path}")

    wandb.finish()


def build_model(ckpt: dict, device: torch.device) -> MixedChannelClassifier:
    """Reconstruct a trained ``MixedChannelClassifier`` from a checkpoint dict.

    Used by ``eval.py`` and ``shap.py``. Infers ``emb_dim`` / ``n_classes`` /
    ``n_channels`` from the saved state dict so no separate metadata is required.
    """
    config = ckpt["config"]
    mcfg = OmegaConf.to_container(OmegaConf.create(config["model"]), resolve=True)
    assert isinstance(mcfg, dict)
    state_dict = ckpt["model_state_dict"]
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    emb_dim = int(state_dict["input_proj.weight"].shape[1])
    n_classes = int(state_dict["head.1.weight"].shape[0])
    n_channels = len(ckpt["channel_to_idx"])
    model = MixedChannelClassifier(
        emb_dim=emb_dim,
        n_classes=n_classes,
        n_channels=n_channels,
        d_model=mcfg.get("d_model", 256),
        n_heads=mcfg.get("n_heads", 4),
        n_layers=mcfg.get("n_layers_cell", 2),
        n_inducing=mcfg.get("n_inducing_cell", 32),
        d_ff=mcfg.get("d_ff", None),
        dropout=mcfg.get("dropout", 0.1),
        cosine_classifier=mcfg.get("cosine_classifier", False),
        channel_conditioning=mcfg.get("channel_conditioning", "none"),
    )
    model.load_state_dict(state_dict)
    return model.to(device).eval()


@hydra.main(
    version_base="1.3.0",
    config_path=_CONFIG_DIR,
    config_name="train_set_classifier",
)
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()  # type: ignore[call-arg]
