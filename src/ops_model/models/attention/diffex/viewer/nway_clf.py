"""N-way per-(marker, grain) single-cell classifiers for the DiffEx viewer score.

The viewer badge should read "does the classifier call this generated cell the target
CLASS, out of all classes" (1-of-N distinctiveness) — NOT "target vs NTC". So for each
(marker, grain) we train one MLPHead over ALL classes on CellDINO features of top-attention
cells. A generated cell is then scored: image → CellDINO (embed_crops) → MLP → softmax →
P(target).

Trained on embed_crops features so it lives in the SAME space the viewer re-encodes
generated cells into (no domain mismatch). Outputs under
<out_root>/_clf/<modality>/<grain>/: mlp.pt, classes.json, metrics.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ..classifier.celldino_features import embed_crops
from ..classifier.config import Config, GRAINS, slugify
from ..classifier.data import _BASE_COLS, make_labels_df, materialize_crops
from ..classifier.models import MLPHead


def _all_class_table(cfg, marker_channel, fluor_csv, n_per_class):
    """Top-attention cells of EVERY class (incl NTC) with integer class labels."""
    cc = cfg.class_col
    if marker_channel:
        cols = list(dict.fromkeys([cc, *_BASE_COLS, "channel", "rank_type"]))
        rows = pd.read_csv(fluor_csv, usecols=cols)
        rows = rows[(rows["channel"] == marker_channel) & (rows["rank_type"] == "top")]
    else:
        rows = pd.read_parquet(cfg.pma_parquet, filters=[("rank_type", "==", "top")],
                               columns=[cc, *_BASE_COLS])
    rows = rows.sort_values("rank").groupby(cc, group_keys=False).head(n_per_class)
    classes = sorted(map(str, rows[cc].unique()))
    idx = {c: i for i, c in enumerate(classes)}
    rows = rows.rename(columns={cc: "cls"}).copy()
    rows["cls"] = rows["cls"].astype(str)
    rows["label"] = rows["cls"].map(idx)
    return rows, classes


def _grouped_split(experiment, seed=0, val_frac=0.15):
    """Hold out whole experiments for val (confound guard); fall back to random if too few."""
    exps = np.array(sorted(set(experiment)))
    rng = np.random.RandomState(seed)
    if len(exps) >= 4:
        rng.shuffle(exps)
        n_val = max(1, int(round(len(exps) * val_frac)))
        val_exps = set(exps[:n_val])
        va = np.array([e in val_exps for e in experiment])
    else:
        va = rng.rand(len(experiment)) < val_frac
    return ~va, va


def _topk(model, X, y, dev, bs=512, k=5):
    model.eval()
    t1 = t5 = 0
    with torch.no_grad():
        for i in range(0, len(X), bs):
            logits = model(torch.as_tensor(X[i:i + bs]).to(dev))
            top = logits.topk(min(k, logits.shape[1]), -1).indices.cpu().numpy()
            yb = y[i:i + bs]
            t1 += int((top[:, 0] == yb).sum())
            t5 += int([yy in tr for yy, tr in zip(yb, top)].count(True))
    return t1 / len(X), t5 / len(X)


def train_nway(grain, out_root, marker_channel=None, channel="Phase2D", fluor_csv=None,
               n_per_class=100, epochs=30, device="cuda", load_workers=12, hidden=256):
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    cfg = Config(class_col=GRAINS[grain]["class_col"], pma_parquet=GRAINS[grain]["parquet"],
                 channel=channel, num_workers=load_workers)
    modality = slugify(marker_channel) if marker_channel else "phase"
    out = Path(out_root) / "_clf" / modality / grain
    out.mkdir(parents=True, exist_ok=True)

    df, classes = _all_class_table(cfg, marker_channel, fluor_csv, n_per_class)
    print(f"[nway] {modality}/{grain}: {len(classes)} classes, {len(df)} cells")
    ldf = make_labels_df(df, cfg)
    images, labels, experiment = materialize_crops(ldf, cfg, cache_path=str(out / "crops.npz"))
    X = embed_crops(images, cfg, cache_path=str(out / "celldino.npz")).astype(np.float32)
    y = labels.astype(np.int64)

    tr, va = _grouped_split(experiment, seed=cfg.seed)
    net = MLPHead(in_dim=X.shape[1], hidden=hidden, n_classes=len(classes)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    crit = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(torch.as_tensor(X[tr]), torch.as_tensor(y[tr])),
                        batch_size=cfg.batch_size, shuffle=True)
    best = {"top1": -1, "state": None}
    for ep in range(epochs):
        net.train()
        for xb, yb in loader:
            opt.zero_grad(); loss = crit(net(xb.to(dev)), yb.to(dev)); loss.backward(); opt.step()
        t1, t5 = _topk(net, X[va], y[va], dev)
        if t1 > best["top1"]:
            best = {"top1": t1, "top5": t5, "epoch": ep,
                    "state": {k: v.detach().cpu() for k, v in net.state_dict().items()}}
        print(f"  ep{ep:02d}: val top1={t1:.3f} top5={t5:.3f} (best {best['top1']:.3f})")
    net.load_state_dict(best["state"])
    torch.save({"state": best["state"], "in_dim": int(X.shape[1]), "hidden": hidden,
                "n_classes": len(classes)}, out / "mlp.pt")
    (out / "classes.json").write_text(json.dumps(classes))
    meta = {"grain": grain, "modality": modality, "marker_channel": marker_channel,
            "channel": channel, "n_classes": len(classes), "n_cells": int(len(X)),
            "n_per_class": n_per_class, "val_top1": best["top1"], "val_top5": best["top5"],
            "best_epoch": best["epoch"]}
    (out / "metrics.json").write_text(json.dumps(meta, indent=2))
    # crops cache is large + one-time; drop it (keep celldino features, small + reusable)
    (out / "crops.npz").unlink(missing_ok=True)
    print(json.dumps(meta, indent=2))
    return meta
