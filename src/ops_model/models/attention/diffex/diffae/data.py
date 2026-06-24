"""Broad phase-crop sampler for DiffAE training.

Samples ~n_crops cells uniformly across the whole geneKO phase parquet (all genes
incl NTC, all attention ranks), then reuses the classifier's crop materialization
so the DiffAE trains on the same crop pipeline. Normalization: per-image z-score
then /3 + clip to [-1, 1] (diffusion-friendly, intensity-invariant like CellDINO).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ..classifier.celldino_features import embed_crops  # frozen CellDINO encoder
from ..classifier.data import make_labels_df, materialize_crops  # reuse

_COLS = ["gene", "experiment", "well", "segmentation", "x_pheno", "y_pheno"]


def build_broad_table(cfg) -> pd.DataFrame:
    """Uniform fraction-sample across all row groups (covers all genes/ranks)."""
    pf = pq.ParquetFile(cfg.pma_parquet)
    total = pf.metadata.num_rows
    frac = min(1.0, cfg.n_crops / total * 1.15)
    rng = np.random.default_rng(cfg.seed)
    parts = []
    for batch in pf.iter_batches(columns=_COLS, batch_size=250_000):
        df = batch.to_pandas()
        keep = rng.random(len(df)) < frac
        if keep.any():
            parts.append(df.loc[keep])
    df = pd.concat(parts, ignore_index=True)
    if len(df) > cfg.n_crops:
        df = df.sample(n=cfg.n_crops, random_state=cfg.seed).reset_index(drop=True)
    df = df.rename(columns={"gene": "cls"})
    df["label"] = 0  # unconditional generator — label unused
    print(f"broad table: {len(df)} crops across {df['cls'].nunique()} classes")
    return df


def normalize(images: np.ndarray) -> np.ndarray:
    """Per-image z-score, /3, clip to [-1, 1]. images: (N,1,H,W)."""
    x = images.astype(np.float32)
    mu = x.mean(axis=(-2, -1), keepdims=True)
    sd = x.std(axis=(-2, -1), keepdims=True) + 1e-6
    return np.clip((x - mu) / sd / 3.0, -1.0, 1.0)


def load_diffae_crops(cfg, crops_cache, emb_cache):
    """Returns (images_norm, celldino_embs).

    images_norm: (N,1,H,W) normalized for diffusion.
    celldino_embs: (N, cond_dim) FROZEN CellDINO embeddings of the SAME crops —
    the conditioning signal (Alex's design). Embeddings come from the raw crops
    (CellDINO does its own resize + z-score internally), cached separately.
    """
    df = build_broad_table(cfg)
    labels_df = make_labels_df(df, cfg)
    images_raw, _, _ = materialize_crops(labels_df, cfg, cache_path=crops_cache)
    embs = embed_crops(images_raw, cfg, cache_path=emb_cache)  # frozen CellDINO
    return normalize(images_raw), embs
