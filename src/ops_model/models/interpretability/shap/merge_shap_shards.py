"""Merge per-shard ko_shap_features_shardNN.csv files into ko_shap_features.csv.

Usage:
    python scripts/ko_shap/merge_shap_shards.py --out-dir /path/to/output
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True, help="Directory containing shard CSVs")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    shards = sorted(out_dir.glob("ko_shap_features_shard*.csv"))
    if not shards:
        raise FileNotFoundError(f"No shard CSVs found in {out_dir}")

    # Skip 0-byte or header-only shard CSVs without exploding. Two
    # legitimate ways to land here:
    #   • --n-shards > n_genes: empty shards have a 0-byte CSV (this is
    #     normal for CHAD level where 90 complexes < default 200 shards
    #     — clamped to gene count, leaving the extras 0-byte).
    #   • A shard timed out mid-write and left a partial / header-only
    #     CSV. EmptyDataError is the pandas signal.
    frames = []
    n_skipped = 0
    for s in shards:
        if s.stat().st_size == 0:
            n_skipped += 1
            continue
        try:
            frames.append(pd.read_csv(s))
        except pd.errors.EmptyDataError:
            n_skipped += 1
            continue
    if n_skipped:
        print(f"  Skipped {n_skipped}/{len(shards)} empty / header-only shards "
              f"(normal when --n-shards exceeds gene count)")
    if not frames:
        raise RuntimeError(
            f"All {len(shards)} shards empty in {out_dir}. Re-run step 1."
        )
    df  = pd.concat(frames, ignore_index=True).sort_values(["gene", "shap_rank"])
    out = out_dir / "ko_shap_features.csv"
    df.to_csv(out, index=False)

    top1 = df[df["shap_rank"] == 1]
    print(f"Merged {len(frames)}/{len(shards)} shards → {df['gene'].nunique()} genes")
    print(f"Median AUROC: {float(np.median(top1['auroc'])):.3f}  F1: {float(np.median(top1['f1'])):.3f}")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
