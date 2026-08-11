"""Aggregate per-class classifier metrics into a ranked table.

    python -m ops_model.models.attention.diffex.classifier.aggregate --grain complex

Collects every <root>/<grain>/<slug>/metrics_<model>.json into one CSV ranked by
test AUROC (how cleanly/distinctly each class's top-attention cells classify), plus
a histogram. NTC is flagged as the negative-control reference.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .config import DEFAULT_OUT_ROOT


def collect(root: Path, model: str) -> pd.DataFrame:
    rows = []
    for mp in sorted(root.glob(f"*/metrics_{model}.json")):
        try:
            rows.append(json.loads(mp.read_text()))
        except Exception as e:  # noqa: BLE001
            print(f"[skip] {mp}: {e}")
    if not rows:
        raise FileNotFoundError(f"no metrics_{model}.json under {root}")
    df = pd.DataFrame(rows)
    df["is_ntc"] = df["gene"].astype(str).eq("NTC")
    return df.sort_values("test_auroc", ascending=False).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description="Aggregate classifier sweep metrics")
    ap.add_argument("--grain", default="complex")
    ap.add_argument("--model", default="C", choices=["B", "C"])
    ap.add_argument("--out-dir", default=DEFAULT_OUT_ROOT)
    args = ap.parse_args()

    root = Path(args.out_dir) / args.grain
    df = collect(root, args.model)

    cols = ["gene", "test_auroc", "val_auroc", "train_auroc_at_best",
            "n_pos", "n_neg", "n_test", "best_epoch"]
    cols = [c for c in cols if c in df.columns]
    csv = root / f"auroc_ranking_{args.model}.csv"
    df[cols + ["is_ntc"]].to_csv(csv, index=False)

    print(f"\n{len(df)} classes  (model {args.model}, grain {args.grain})")
    print(f"test AUROC: median={df.test_auroc.median():.3f}  "
          f"mean={df.test_auroc.mean():.3f}  "
          f"range {df.test_auroc.min():.3f}..{df.test_auroc.max():.3f}")
    if df.is_ntc.any():
        ntc = df[df.is_ntc].iloc[0]
        rank = int(df.index[df.is_ntc][0]) + 1
        print(f"NTC control: test_auroc={ntc.test_auroc:.3f}  (rank {rank}/{len(df)} — "
              f"should be near the bottom)")
    print("\nTop 10 most-distinct:")
    print(df.head(10)[cols].to_string(index=False))
    print("\nBottom 10 (least distinct):")
    print(df.tail(10)[cols].to_string(index=False))

    try:
        import matplotlib
        matplotlib.use("Agg")
        matplotlib.rcParams["pdf.fonttype"] = 42
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(df.test_auroc, bins=25, color="steelblue", edgecolor="white")
        if df.is_ntc.any():
            ax.axvline(df[df.is_ntc].test_auroc.iloc[0], color="red", ls="--",
                       label="NTC control")
            ax.legend()
        ax.set_xlabel("test AUROC (distinctiveness)")
        ax.set_ylabel("# classes")
        ax.set_title(f"{args.grain} {args.model}: per-class classifier AUROC (n={len(df)})")
        fig.tight_layout()
        png = root / f"auroc_hist_{args.model}.png"
        fig.savefig(png, dpi=150)
        print(f"\nwrote {csv}\nwrote {png}")
    except Exception as e:  # noqa: BLE001
        print(f"\nwrote {csv}  (histogram skipped: {e})")


if __name__ == "__main__":
    main()
