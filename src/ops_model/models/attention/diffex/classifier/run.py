"""Orchestrator for the DiffEx single-cell classifier PoC.

    python -m ops_model.models.attention.diffex.classifier.run --model B --gene HSPA5
    python -m ops_model.models.attention.diffex.classifier.run --model C --gene HSPA5

Shared steps: build cell table -> materialize phase crops (cached) -> split.
Then B trains a ResNet on crops; C embeds the crops with CellDINO and trains an
MLP. Writes model_<X>.pt + metrics_<X>.json. The crops/features cache is shared,
so running C after B reuses the same crops.

``run_poc`` is the importable entry point used by ``submit.py`` (SLURM).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .celldino_features import embed_crops
from .config import DEFAULT_OUT_ROOT, GRAINS, Config, slugify
from .data import build_cell_table, make_labels_df, materialize_crops, split_train_val_test
from .models import MLPHead, build_resnet
from .train import train_classifier


def run_poc(cfg: Config, model: str, out_dir: str) -> dict:
    """Run one PoC (model 'B' or 'C') and write artifacts to ``out_dir``."""
    out = Path(out_dir)
    cache = out / "cache"
    cache.mkdir(parents=True, exist_ok=True)

    # ---- shared: cells -> crops -> split (crops cache is model-agnostic) ----
    tag = f"{slugify(cfg.gene)}_{cfg.crop_size}_{'mask' if cfg.mask_cell else 'nomask'}"
    df = build_cell_table(cfg)
    labels_df = make_labels_df(df, cfg)
    images, labels, experiment = materialize_crops(
        labels_df, cfg, cache_path=str(cache / f"crops_{tag}.npz")
    )
    tr, va, te = split_train_val_test(labels, experiment, cfg)

    # ---- model-specific features + model ----
    if model == "B":
        X = images
        net = build_resnet(in_channels=1)
    elif model == "C":
        X = embed_crops(images, cfg, cache_path=str(cache / f"celldino_{tag}.npz"))
        net = MLPHead(in_dim=X.shape[1])
    else:
        raise ValueError(f"model must be 'B' or 'C', got {model!r}")

    best = train_classifier(
        net, X[tr], labels[tr], X[va], labels[va], X[te], labels[te], cfg
    )

    torch.save(best["state"], out / f"model_{model}.pt")
    metrics = {
        "model": model, "gene": cfg.gene, "class_col": cfg.class_col,
        "n_pos": int((labels == 1).sum()), "n_neg": int((labels == 0).sum()),
        "n_train": int(tr.sum()), "n_val": int(va.sum()), "n_test": int(te.sum()),
        "split_mode": cfg.split_mode,
        "test_auroc": best["test_auroc"],
        "val_auroc": best["val_auroc"], "train_auroc_at_best": best["train_auroc"],
        "best_epoch": best["epoch"],
        "feature_dim": int(X.shape[1]) if model == "C" else None,
    }
    (out / f"metrics_{model}.json").write_text(json.dumps(metrics, indent=2))
    (out / f"history_{model}.json").write_text(json.dumps(best["history"], indent=2))
    print(json.dumps(metrics, indent=2))
    return metrics


def main():
    ap = argparse.ArgumentParser(description="DiffEx single-cell classifier PoC (B/C)")
    ap.add_argument("--model", choices=["B", "C"], required=True,
                    help="B = ResNet on crops; C = MLP on CellDINO features")
    ap.add_argument("--grain", choices=list(GRAINS), default="geneKO")
    ap.add_argument("--gene", default="HSPA5", help="class value (gene or complex name)")
    ap.add_argument("--n-per-class", type=int, default=1000)
    ap.add_argument("--crop-size", type=int, default=160)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--split-mode", choices=["experiment", "random"], default="experiment")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    grain = GRAINS[args.grain]
    cfg = Config(
        gene=args.gene, class_col=grain["class_col"], pma_parquet=grain["parquet"],
        n_per_class=args.n_per_class, crop_size=args.crop_size,
        epochs=args.epochs, split_mode=args.split_mode, device=args.device,
        num_workers=args.num_workers,
    )
    run_poc(cfg, args.model, args.out_dir or f"{DEFAULT_OUT_ROOT}/{args.grain}/{slugify(cfg.gene)}")


if __name__ == "__main__":
    main()
