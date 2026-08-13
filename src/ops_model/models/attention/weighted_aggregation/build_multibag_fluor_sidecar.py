"""Build the v5-multibag FLUOR sidecar parquet — one row per
(experiment, well, segmentation_id, channel_name), with weight columns for
both classifier variants (geneKO / EBI).

Reads the two multi-rank CSVs and merges them on the shared cell/channel key.

Output schema matches the shape of per_experiment_v4_attn_fluor.parquet, so the
weighted_pca runner can consume it via ``--signal-set no_phase``:

    experiment, well, segmentation_id, channel,
    v5m_<head>_fluor_gene, v5m_<head>_fluor_rank,
    v5m_<head>_fluor_cutoff_500, v5m_<head>_fluor_cutoff_100

for head in {gko, ebionly}. cutoff_500 ≈ phase cutoff_20k proportion (top ~31%
of median ~1600 fluor cells/gene/channel); cutoff_100 is a tighter sweep point.
Weight column = (1 − rank/N) clipped to ≥0, zeroed outside the cutoff.

Dedup: as with phase, a small tail of cells appear in more than one gene's
ranking (barcode / dual-guide crosstalk). We keep the lowest (best) rank per
(exp, well, seg, channel) per head.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.csv as pa_csv

SRC = Path("/hpc/projects/icd.fast.ops/models/alex_lin_attention/v5/multi_rank")
OUT = Path("/hpc/projects/icd.fast.ops/models/alex_lin_attention/v5/expansion_v1/"
           "per_experiment_v5_multibag_fluor.parquet")
FILES = {"gko":     "shap_screen_fluor_all.csv",
         "ebionly": "shap_screen_ebi_fluor_all.csv"}
KEY = ["experiment", "well", "segmentation_id", "channel"]


def _load(head: str) -> pd.DataFrame:
    p = SRC / FILES[head]
    print(f"[{head}] reading {p}")
    t0 = time.time()
    tbl = pa_csv.read_csv(p, convert_options=pa_csv.ConvertOptions(
        include_columns=["gene", "channel_name", "rank",
                          "experiment", "well", "segmentation_id", "n_cells"]))
    df = tbl.to_pandas()
    del tbl
    print(f"[{head}] loaded {len(df):,} rows in {time.time()-t0:.1f}s")

    rank = df["rank"].astype(np.int32)
    N = df["n_cells"].astype(np.int32)
    w = (1.0 - rank.astype(np.float32) / N.astype(np.float32)).clip(lower=0.0)
    w_cutoff_500 = np.where(rank <= 500, w, np.float32(0.0)).astype(np.float32)
    w_cutoff_100 = np.where(rank <= 100, w, np.float32(0.0)).astype(np.float32)

    # Multi-rank CSV uses space-separated channel names (e.g.
    # 'Fe2+_FeRhoNox live-cell dye'); V4 fluor sidecar uses underscored
    # ('Fe2+_FeRhoNox_live-cell_dye'). Normalize to the V4 format so the
    # runner's lookup key matches.
    channel = df["channel_name"].astype(str).str.replace(" ", "_", regex=False)
    out = pd.DataFrame({
        "experiment":      df["experiment"].astype(str),
        "well":            df["well"].astype(str),
        "segmentation_id": df["segmentation_id"].astype(np.int64),
        "channel":         channel,
        f"v5m_{head}_fluor_gene": df["gene"].astype(str),
        f"v5m_{head}_fluor_rank": rank,
        f"v5m_{head}_fluor_cutoff_500":   w_cutoff_500,
        f"v5m_{head}_fluor_cutoff_100":   w_cutoff_100,
    })
    # dedup — best rank per (exp, well, seg, channel_name)
    out = (out.sort_values(KEY + [f"v5m_{head}_fluor_rank"])
              .drop_duplicates(subset=KEY, keep="first"))
    return out


def build_sidecar() -> Path:
    dfs = {head: _load(head) for head in FILES}
    print("merging heads on (exp, well, seg, channel_name)…")
    t0 = time.time()
    merged = dfs["gko"].merge(dfs["ebionly"], on=KEY, how="outer")
    print(f"merged: {len(merged):,} rows in {time.time()-t0:.1f}s")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    print(f"writing {OUT}")
    t0 = time.time()
    merged.to_parquet(OUT, index=False)
    print(f"wrote {len(merged):,} rows in {time.time()-t0:.1f}s")
    return OUT


if __name__ == "__main__":
    build_sidecar()
