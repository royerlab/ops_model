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
    if getattr(cfg, "marker_channel", None):
        pre = getattr(cfg, "_fluor_rows", None)      # preloaded rows (multi-marker: read the 12GB CSV ONCE)
        if pre is not None:
            df = pre[pre["channel"] == cfg.marker_channel] if "channel" in pre.columns else pre
            if "rank_type" in df.columns:
                df = df[df["rank_type"] == "top"]
        else:
            src = cfg.fluor_csv
            if src.endswith(".parquet"):                       # per-marker rankings parquet (channel_name == marker)
                df = pd.read_parquet(src)
                if "channel_name" in df.columns and "channel" not in df.columns:
                    df = df.rename(columns={"channel_name": "channel"})
            else:
                cols = ["gene", "channel", "experiment", "well", "segmentation", "x_pheno", "y_pheno", "rank_type"]
                df = pd.read_csv(src, usecols=cols)
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


def scale_imgs(images: np.ndarray, cfg) -> np.ndarray:
    """DiffAE target scaling. Default per-image z-score (prod). intensity_norm='global' uses a
    SINGLE global mu/sd over the whole sample (one constant, not per-image, not per-plate) so
    absolute cross-cell brightness survives — the no-normalization experiment. Still lands in
    [-1,1] for the diffusion model."""
    if getattr(cfg, "intensity_norm", "per_image") == "global":
        x = images.astype(np.float32)
        mu, sd = float(x.mean()), float(x.std()) + 1e-6   # one stat for all cells → intensity preserved
        return np.clip((x - mu) / sd / 3.0, -1.0, 1.0)
    return normalize(images)


def load_diffae_crops(cfg, crops_cache, emb_cache, cond_cache=None, return_cond_images=False):
    """Returns (images_norm, celldino_embs[, cond_images_norm]).

    images_norm: (N,1,H,W) generation target, normalized for diffusion.
    celldino_embs: (N, cond_dim) FROZEN CellDINO conditioning embeddings.

    Same-channel (default): embeddings come from the SAME crops as the target (Alex's design).
    Virtual staining (cfg.cond_channel set): the target is `cfg.channel` (e.g. mCherry) while the
    conditioning is CellDINO of the co-registered `cfg.cond_channel` crop (e.g. Phase2D) at the SAME
    cell locations — same labels_df → aligned index. return_cond_images also returns the normalized
    conditioning (phase) crops for eval montages.
    """
    import dataclasses
    df = build_broad_table(cfg)
    labels_df = make_labels_df(df, cfg)
    images_raw, _, _ = materialize_crops(labels_df, cfg, cache_path=crops_cache)   # target (cfg.channel)
    if getattr(cfg, "cond_channel", None):                          # virtual staining: embed a DIFFERENT channel
        cond_cfg = dataclasses.replace(cfg, channel=cfg.cond_channel)
        cond_raw, _, _ = materialize_crops(labels_df, cond_cfg, cache_path=cond_cache)
        embs = embed_crops(cond_raw, cfg, cache_path=emb_cache)     # CellDINO of the conditioning channel
        if return_cond_images:
            return scale_imgs(images_raw, cfg), embs, scale_imgs(cond_raw, cfg)
        return scale_imgs(images_raw, cfg), embs
    embs = embed_crops(images_raw, cfg, cache_path=emb_cache)       # frozen CellDINO (same-channel)
    if return_cond_images:
        return scale_imgs(images_raw, cfg), embs, scale_imgs(images_raw, cfg)
    return scale_imgs(images_raw, cfg), embs
