"""Data layer: build the cell table, materialize phase crops, split.

Positives  = top-N attention cells of cfg.gene.
Negatives  = top-`neg_rank_max` attention cells of every OTHER gene (the
             "distinct" contrast), sampled to N — like-with-like (strong vs strong).

Crops are read once via the shared `BaseDataset` loader and cached, so both
option B (ResNet on crops) and option C (CellDINO features on the SAME crops)
train off one identical cell set.
"""
from __future__ import annotations

import os
import re
import warnings

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from iohub import open_ome_zarr
from ops_utils.data.bbox_utils import BaseDataset
from ops_utils.data.experiment import OpsDataset
from ops_utils.data.filesystem import resolve_experiment_name

_BASE_COLS = [
    "experiment", "well", "segmentation", "x_pheno", "y_pheno", "pma_attention", "rank",
]


def _normalize_well(well_str) -> str:
    """'A3' -> 'A/3/0' (OPS zarr positions are 3-level row/col/fov)."""
    w = str(well_str).strip()
    if w.count("/") == 2:
        return w
    if w.count("/") == 1:
        return f"{w}/0"
    m = re.match(r"^([A-Za-z]+)(\d+)$", w)
    if not m:
        raise ValueError(f"Unknown well format: {well_str!r}")
    return f"{m.group(1)}/{m.group(2)}/0"


def build_cell_table(cfg) -> pd.DataFrame:
    """Filtered reads off the parquet (no full load). Class column is cfg.class_col
    ('gene' for geneKO, 'predicted_class' for complexes)."""
    cc = cfg.class_col
    cols = [cc] + _BASE_COLS
    pos = pd.read_parquet(
        cfg.pma_parquet,
        filters=[(cc, "==", cfg.gene), ("rank_type", "==", "top")],
        columns=cols,
    )
    if pos.empty:
        raise ValueError(f"No 'top' attention rows for {cc}={cfg.gene!r}")
    pos = pos.sort_values("rank").head(cfg.n_per_class).copy()
    pos["label"] = 1

    neg_pool = pd.read_parquet(
        cfg.pma_parquet,
        filters=[("rank_type", "==", "top"), ("rank", "<=", cfg.neg_rank_max)],
        columns=cols,
    )
    neg_pool = neg_pool[neg_pool[cc] != cfg.gene]
    neg = neg_pool.sample(
        n=min(cfg.n_per_class, len(neg_pool)), random_state=cfg.seed
    ).copy()
    neg["label"] = 0

    df = pd.concat([pos, neg], ignore_index=True).rename(columns={cc: "cls"})
    print(f"cell table: {int(df.label.sum())} pos ({cfg.gene}) + "
          f"{int((df.label == 0).sum())} neg (other classes' top-{cfg.neg_rank_max})")
    return df


def make_labels_df(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """Cell table -> BaseDataset labels_df (bbox from x/y_pheno, like the atlas)."""
    half = cfg.crop_size // 2
    recs = []
    for i, r in df.reset_index(drop=True).iterrows():
        y, x = int(r["y_pheno"]), int(r["x_pheno"])
        recs.append({
            "experiment": r["experiment"],
            "store_key": r["experiment"],
            "well": _normalize_well(r["well"]),
            # clamp low end so border cells don't negative-wrap the zarr slice;
            # SpatialPadd pads any short side back to crop_size.
            "bbox": [max(0, y - half), max(0, x - half), y + half, x + half],
            "segmentation_id": r["segmentation"],
            "gene_name": r["cls"],
            "label": int(r["label"]),
            "total_index": i,
        })
    return pd.DataFrame(recs)


def _open_stores(experiments):
    stores = {}
    for exp in experiments:
        try:
            ds = OpsDataset(resolve_experiment_name(exp))
            path = ds.store_paths["pheno_assembled_v3"]
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                stores[exp] = open_ome_zarr(str(path), mode="r")
        except Exception as e:  # noqa: BLE001 - surface and skip the experiment
            print(f"[store] failed {exp}: {e}")
    return stores


def _collate(batch):
    data = torch.stack([torch.as_tensor(b["data"]) for b in batch])
    ti = torch.tensor([b["total_index"] for b in batch])
    return {"data": data, "total_index": ti}


def materialize_crops(labels_df: pd.DataFrame, cfg, cache_path=None):
    """Read every crop once via BaseDataset; return (images, labels, experiment).

    images: (N, 1, crop, crop) float32. Cached to ``cache_path`` (.npz).
    """
    if cache_path and os.path.exists(cache_path):
        d = np.load(cache_path, allow_pickle=True)
        print(f"[cache] crops <- {cache_path}  {d['images'].shape}")
        return d["images"], d["labels"], d["experiment"]

    stores = _open_stores(labels_df["experiment"].unique())
    df = labels_df[labels_df["experiment"].isin(stores)].reset_index(drop=True)
    df["total_index"] = range(len(df))
    if len(df) < len(labels_df):
        print(f"[store] dropped {len(labels_df) - len(df)} cells from failed stores")

    ds = BaseDataset(
        stores=stores,
        labels_df=df,
        initial_yx_patch_size=(cfg.crop_size, cfg.crop_size),
        final_yx_patch_size=(cfg.crop_size, cfg.crop_size),
        out_channels=[cfg.channel],
        mask_cell=cfg.mask_cell,
    )
    loader = DataLoader(
        ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, collate_fn=_collate,
    )
    images = np.zeros((len(df), 1, cfg.crop_size, cfg.crop_size), np.float32)
    for batch in loader:
        idx = batch["total_index"].numpy()
        images[idx] = batch["data"].numpy().astype(np.float32)
    labels = df["label"].to_numpy(np.int64)
    experiment = df["experiment"].to_numpy()

    if cache_path:
        np.savez(cache_path, images=images, labels=labels, experiment=experiment)
        print(f"[cache] crops -> {cache_path}  {images.shape}")
    return images, labels, experiment


def split_train_val_test(labels: np.ndarray, experiment: np.ndarray, cfg):
    """3-way split (train/val/test). Grouped by experiment by default (confound
    guard: val & test cells come from experiments the model never trained on).
    val = model selection; test = clean held-out number. Falls back to
    stratified random if the grouped split leaves a class missing from any side.

    Returns three boolean masks (train, val, test).
    """
    rng = np.random.default_rng(cfg.seed)
    n = len(labels)

    def _ok(which) -> bool:
        for s in ("train", "val", "test"):
            m = which == s
            if m.sum() == 0 or labels[m].min() != 0 or labels[m].max() != 1:
                return False
        return True

    def _stratified_random():
        which = np.array(["train"] * n, dtype="<U5")
        for cls in np.unique(labels):
            idx = np.flatnonzero(labels == cls)
            rng.shuffle(idx)
            n_val = max(1, int(len(idx) * cfg.val_fraction))
            n_test = max(1, int(len(idx) * cfg.test_fraction))
            which[idx[:n_val]] = "val"
            which[idx[n_val:n_val + n_test]] = "test"
        return which

    if cfg.split_mode == "experiment":
        exps = np.unique(experiment)
        rng.shuffle(exps)
        n_val = max(1, int(len(exps) * cfg.val_fraction))
        n_test = max(1, int(len(exps) * cfg.test_fraction))
        val_exps = set(exps[:n_val])
        test_exps = set(exps[n_val:n_val + n_test])
        which = np.array([
            "val" if e in val_exps else "test" if e in test_exps else "train"
            for e in experiment
        ])
        if not _ok(which):
            print("[split] grouped split degenerate (class missing) -> stratified random")
            which = _stratified_random()
    else:
        which = _stratified_random()

    tr, va, te = which == "train", which == "val", which == "test"
    print(f"[split] train={int(tr.sum())} val={int(va.sum())} test={int(te.sum())} "
          f"(mode={cfg.split_mode})")
    return tr, va, te
