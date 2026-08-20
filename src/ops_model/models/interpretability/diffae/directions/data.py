"""Gather target (KD) + control (NTC) top-attention cells, with CellDINO embeddings.

Returns raw crops (for traversal inversion), CellDINO embeddings (direction
discovery + ranking), and labels (1=target, 0=control)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..classifier.celldino_features import embed_crops
from ..classifier.config import GRAINS
from ..classifier.data import _BASE_COLS, make_labels_df, materialize_crops


def _top_cells(parquet: str, class_col: str, value: str, n: int) -> pd.DataFrame:
    df = pd.read_parquet(
        parquet, filters=[(class_col, "==", value), ("rank_type", "==", "top")],
        columns=[class_col] + _BASE_COLS,
    )
    if df.empty:
        raise ValueError(f"no 'top' rows for {class_col}={value!r}")
    return df.sort_values("rank").head(n).rename(columns={class_col: "cls"})


def _gather_df(cfg):
    """target+control cell table — phase: grain parquet; fluor: marker CSV (cfg.marker_channel)."""
    cc = GRAINS[cfg.grain]["class_col"]
    if getattr(cfg, "marker_channel", None):        # fluorescent mode
        acc = getattr(cfg, "accuracy_fluor_csv", None)
        if acc:                                     # accuracy variant: per-channel parquet (channel-filtered, class_col renamed)
            rows = pd.read_parquet(acc)
        else:
            cols = list(dict.fromkeys([cc, *_BASE_COLS, "channel", "rank_type"]))
            rows = pd.read_csv(cfg.fluor_csv, usecols=cols)
            rows = rows[(rows["channel"] == cfg.marker_channel) & (rows["rank_type"] == "top")]

        def top(value, label):
            d = rows[rows[cc] == value].sort_values("rank").head(cfg.n_per_class)
            if d.empty:
                raise ValueError(f"no fluor 'top' rows for {cc}={value!r} channel={cfg.marker_channel!r}")
            d = d.rename(columns={cc: "cls"}).copy(); d["label"] = label
            return d
        return pd.concat([top(cfg.target, 1), top(cfg.control, 0)], ignore_index=True)

    g = GRAINS[cfg.grain]                            # phase mode
    pq = getattr(cfg, "accuracy_parquet", None) or g["parquet"]   # accuracy variant overrides the attention parquet (both A & B)
    pos = _top_cells(pq, g["class_col"], cfg.target, cfg.n_per_class); pos["label"] = 1
    ctl = _top_cells(pq, g["class_col"], cfg.control, cfg.n_per_class); ctl["label"] = 0
    return pd.concat([pos, ctl], ignore_index=True)


def gather(cfg, crops_cache, emb_cache):
    """-> (images_raw (N,1,H,W), embs (N,cond_dim), labels (N,))."""
    df = _gather_df(cfg)
    print(f"gather: {int(df.label.sum())} {cfg.target} + {int((df.label==0).sum())} {cfg.control}"
          + (f" [fluor {cfg.marker_channel}->raw {cfg.channel}]" if getattr(cfg, "marker_channel", None) else ""))

    labels_df = make_labels_df(df, cfg)
    images, labels, _ = materialize_crops(labels_df, cfg, cache_path=crops_cache)
    embs = embed_crops(images, cfg, cache_path=emb_cache)
    return images, embs, labels
