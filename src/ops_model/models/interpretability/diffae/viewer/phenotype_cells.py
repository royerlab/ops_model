"""Enumerate the PHENOTYPE cells the viewer traverses toward: for every (marker × perturbation),
the top-`n_cells` attention-ranked real cells. Perturbations = geneKOs + EBI complexes; markers =
phase + all fluor channels. A handoff list for Ritvik to compute SetTransformer attention pixel-patches
on the real phenotype cells (to compare what the classifier attends to vs the generative morph).

Grid = n_cells × (1000 geneKO + ~100 complex) × markers. `segmentation_id` = the pma-source
`segmentation` value → (experiment, well, segmentation_id) is the unique cell lookup key.
Sources: phase = v4 parquets (rank pushdown); fluor = v4 CSVs (geneKO=pma_fluorescent_cells_all,
complex=pma_fluorescent_cells_ebi_all), read chunked + filtered to the top ranks.
"""
from __future__ import annotations

import glob
import os

import pandas as pd
import pyarrow.parquet as pq

from ..classifier.config import PMA_PHASE_EBI, PMA_PHASE_GENEKO
from ..classifier.data import _BASE_COLS
from . import catalog as C

_V4 = os.path.dirname(C.EBI_FLUOR_CSV)
FLUOR_GENEKO_CSV = f"{_V4}/pma_fluorescent_cells_all.csv"
FLUOR_COMPLEX_CSV = C.EBI_FLUOR_CSV
OUT_CSV = f"{C.OUT}/viewer_assets/phenotype_cells_for_attention.csv"
# rank_source: "model" = ranked by the marker's SetTransformer attention (rank/pma_attention are real);
# "fallback" = marker/perturbation not in the attention model (e.g. cisGolgi) → cells from another source,
# rank/pma_attention/map_score are NOT model-derived (marked so consumers don't trust them as attention).
COLS = ["marker_channel", "grain", "perturbation", "geneKO", "ebi_complex", "map_score", "rank_source",
        "cell_index", "experiment", "well", "segmentation_id", "x_pheno", "y_pheno", "rank", "pma_attention"]


def _gene2cx():
    import yaml
    y = yaml.safe_load(open(C.EBI_YAML)) or {}
    return {g: v["name"] for v in y.values() if isinstance(v, dict) for g in (v.get("genes") or [])}


def _fmt(df, mc, grain, pert_col, gene2cx, n_cells, score=None, rank_source="model"):
    df = df.sort_values("rank").copy()
    df["marker_channel"] = mc; df["grain"] = grain; df["rank_source"] = rank_source
    df["perturbation"] = df[pert_col].astype(str)
    if "gene" not in df:
        df["gene"] = df["perturbation"]
    df["ebi_complex"] = df["gene"].astype(str).map(gene2cx).fillna("")
    df["map_score"] = df["perturbation"].map(score) if score is not None else float("nan")  # dist(geneKO)/EBI mAP(complex)
    df["cell_index"] = df.groupby(["marker_channel", "grain", "perturbation"]).cumcount()
    df = df[df["cell_index"] < n_cells]
    return df.rename(columns={"segmentation": "segmentation_id", "gene": "geneKO"})[COLS]


def _csv_top(path, group_cols, n_cells, chunk=3_000_000):
    """Chunked read of a big fluor CSV → each MARKER's top-`n_cells` cells per group. `rank` is GLOBAL
    per-geneKO (not per marker), so we re-rank WITHIN each (channel, group): take the n_cells
    lowest-rank (highest-attention) cells present in that channel. Two-pass head() keeps memory bounded."""
    use = list(dict.fromkeys(["gene", "channel", "predicted_class", "rank_type", *_BASE_COLS]))
    keep = []
    for ch in pd.read_csv(path, usecols=use, chunksize=chunk):
        ch = ch[ch["rank_type"] == "top"]
        keep.append(ch.sort_values("rank").groupby(group_cols, sort=False).head(n_cells))
    return pd.concat(keep, ignore_index=True).sort_values("rank").groupby(group_cols, sort=False).head(n_cells)


def build(n_cells=20, out_csv=OUT_CSV, fluor=True, map_thr=0.01):
    """Phase = ALL perturbations. Fluor = only perturbations the marker distinguishes: geneKO by gene
    distinctiveness >= map_thr, complex by EBI complex mAP >= map_thr (per the marker's reporter)."""
    g2c = _gene2cx()
    dist = C.dist_matrix(); cdist = C.complex_dist()
    parts = []
    filt = [("rank_type", "==", "top"), ("rank", "<=", n_cells)]

    gk = pq.read_table(PMA_PHASE_GENEKO, columns=["gene", *_BASE_COLS], filters=filt).to_pandas()
    parts.append(_fmt(gk, "phase", "geneKO", "gene", g2c, n_cells, score=dist.get("Phase")))
    cx = pq.read_table(PMA_PHASE_EBI, columns=["gene", "predicted_class", *_BASE_COLS], filters=filt).to_pandas()
    parts.append(_fmt(cx, "phase", "complex", "predicted_class", g2c, n_cells, score=cdist.get("Phase")))
    print(f"[phenotype] phase (ALL): {len(parts[0])} geneKO + {len(parts[1])} complex cells")

    if fluor:
        fg = _csv_top(FLUOR_GENEKO_CSV, ["channel", "gene"], n_cells)
        for mc, sub in fg.groupby("channel"):                    # geneKO: keep genes with distinctiveness >= thr
            rep = C.rep_of(dist, str(mc))
            if rep not in dist.columns:                          # no distinctiveness mAP (e.g. 4i antibodies) → exclude
                continue
            sc = dist[rep]
            sub = sub[sub["gene"].astype(str).isin(set(sc.index[sc >= map_thr]))]
            parts.append(_fmt(sub, str(mc), "geneKO", "gene", g2c, n_cells, score=sc))
        fc = _csv_top(FLUOR_COMPLEX_CSV, ["channel", "predicted_class"], n_cells)
        for mc, sub in fc.groupby("channel"):                    # complex: keep complexes with EBI mAP >= thr
            rep = C.rep_of(dist, str(mc))
            if not rep or rep not in cdist.columns:              # no EBI mAP → exclude
                continue
            sc = cdist[rep]
            sub = sub[sub["predicted_class"].astype(str).isin(set(sc.index[sc >= map_thr]))]
            parts.append(_fmt(sub, str(mc), "complex", "predicted_class", g2c, n_cells, score=sc))
        print(f"[phenotype] fluor filtered @ mAP>={map_thr}")

    df = pd.concat(parts, ignore_index=True)
    df.to_csv(out_csv, index=False)
    print(f"[phenotype] {len(df)} cells | {df['marker_channel'].nunique()} markers | "
          f"{df.groupby(['marker_channel', 'grain'])['perturbation'].nunique().sum()} (marker,grain,pert) groups -> {out_csv}")
    return out_csv


if __name__ == "__main__":
    build()
