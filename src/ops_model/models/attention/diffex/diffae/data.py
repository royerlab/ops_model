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
    """Uniform fraction-sample across all row groups (covers all genes/ranks).
    Fluorescent mode (cfg.marker_channel set): sample that marker's cells from the fluor
    attention CSV; the generator reads cfg.channel (GFP/mCherry) — the raw pheno-zarr channel
    carrying that marker."""
    if getattr(cfg, "anndata_paths", ()):          # no-PMA markers: cell table from per-exp anndata
        import anndata as ad
        parts = []
        for p in cfg.anndata_paths:
            o = ad.read_h5ad(p, backed="r").obs[["perturbation", "well", "x_position", "y_position", "experiment"]].copy()
            parts.append(o)
        df = pd.concat(parts, ignore_index=True)
        df["well"] = df["well"].astype(str).str.split("_").str[0]     # "A/2/0_ops.." -> "A/2/0"
        df = df.rename(columns={"perturbation": "cls", "x_position": "x_pheno", "y_position": "y_pheno"})
        df["segmentation"] = 0                                        # unused (mask_cell=False)
        if len(df) > cfg.n_crops:
            df = df.sample(n=cfg.n_crops, random_state=cfg.seed)
        df = df.reset_index(drop=True); df["label"] = 0
        print(f"anndata broad table [raw {cfg.channel}]: {len(df)} crops across {df['cls'].nunique()} genes")
        return df

    if getattr(cfg, "marker_channel", None):
        cols = ["gene", "channel", "experiment", "well", "segmentation", "x_pheno", "y_pheno", "rank_type"]
        df = pd.read_csv(cfg.fluor_csv, usecols=cols)
        df = df[(df["channel"] == cfg.marker_channel) & (df["rank_type"] == "top")]
        if df.empty:
            raise ValueError(f"no 'top' cells for marker_channel={cfg.marker_channel!r}")
        if len(df) > cfg.n_crops:
            df = df.sample(n=cfg.n_crops, random_state=cfg.seed)
        df = df.reset_index(drop=True).rename(columns={"gene": "cls"})
        df["label"] = 0
        print(f"fluor broad table [{cfg.marker_channel} -> raw {cfg.channel}]: "
              f"{len(df)} crops across {df['cls'].nunique()} genes")
        return df

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
