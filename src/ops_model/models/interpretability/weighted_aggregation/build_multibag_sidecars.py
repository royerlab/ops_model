"""Build 3 v5-multibag sidecar parquets (gko/ebionly/ebifb) from the
multi-bag SHAP ranking CSVs/parquet in v5/multi_rank/.

Output schema per sidecar (matches per_experiment_v5_{name}.parquet):
    experiment, well, segmentation_id,
    v5m_{name}_gene, v5m_{name}_rank,
    v5m_{name}                  = 1 - rank / N (higher-ranked cells contribute more)
    v5m_{name}_cutoff_20k       = v5m_{name} if rank <= 20000 else 0

N = per-perturbation total cell count (from the ``n_cells`` column when present,
else computed as the per-gene rank_max).

Usage
-----
    # Build one at a time (parallel-friendly):
    python -m ops_model.models.interpretability.weighted_aggregation.build_multibag_sidecars gko
    python -m ops_model.models.interpretability.weighted_aggregation.build_multibag_sidecars ebionly
    python -m ops_model.models.interpretability.weighted_aggregation.build_multibag_sidecars ebifb
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.csv as pa_csv
import pyarrow.parquet as pq

SRC_DIR = Path("/hpc/projects/icd.fast.ops/models/alex_lin_attention/v5/multi_rank")
OUT_DIR = Path("/hpc/projects/icd.fast.ops/models/alex_lin_attention/v5/expansion_v1")

SOURCES = {
    "gko":     ("shap_screen_phase_all.parquet",     "parquet"),
    "ebionly": ("shap_screen_ebi_phase_all.csv",     "csv"),
    "ebifb":   ("shap_screen_ebifb_phase_all.csv",   "csv"),
}


def _read_source(name: str) -> pd.DataFrame:
    fname, kind = SOURCES[name]
    path = SRC_DIR / fname
    print(f"[{name}] reading {path} ({kind})")
    t0 = time.time()
    keep = ["gene", "rank", "experiment", "well", "segmentation_id"]
    optional = ["n_cells"]
    if kind == "parquet":
        cols = pq.ParquetFile(path).schema.names
        use = [c for c in keep + optional if c in cols]
        df = pd.read_parquet(path, columns=use)
    else:
        header = pd.read_csv(path, nrows=1).columns.tolist()
        use = [c for c in keep + optional if c in header]
        tbl = pa_csv.read_csv(
            path,
            convert_options=pa_csv.ConvertOptions(include_columns=use),
        )
        df = tbl.to_pandas()
        del tbl
    print(f"[{name}] loaded {len(df):,} rows in {time.time()-t0:.1f}s (cols={list(df.columns)})")
    return df


def build_sidecar(name: str) -> Path:
    df = _read_source(name)

    if "n_cells" not in df.columns:
        print(f"[{name}] deriving N per gene from max(rank)…")
        N_per_gene = df.groupby("gene")["rank"].transform("max").astype(np.int32)
    else:
        N_per_gene = df["n_cells"].astype(np.int32)

    print(f"[{name}] computing weight cols…")
    rank = df["rank"].astype(np.int32)
    w = 1.0 - rank.astype(np.float32) / N_per_gene.astype(np.float32)
    w = w.clip(lower=0.0).astype(np.float32)
    w_cutoff_20k = np.where(rank <= 20000, w, np.float32(0.0)).astype(np.float32)

    out = pd.DataFrame({
        "experiment":      df["experiment"].astype(str),
        "well":            df["well"].astype(str),
        "segmentation_id": df["segmentation_id"].astype(np.int64),
        f"v5m_{name}_gene": df["gene"].astype(str),
        f"v5m_{name}_rank": rank,
        f"v5m_{name}":      w,
        f"v5m_{name}_cutoff_20k": w_cutoff_20k,
    })

    out_path = OUT_DIR / f"per_experiment_v5_multibag_{name}.parquet"
    print(f"[{name}] writing {out_path}")
    t0 = time.time()
    out.to_parquet(out_path, index=False)
    print(f"[{name}] wrote {len(out):,} rows in {time.time()-t0:.1f}s")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("name", choices=list(SOURCES), help="which sidecar to build")
    args = ap.parse_args()
    build_sidecar(args.name)


if __name__ == "__main__":
    main()
