"""Enumerate every viewer ANCHOR cell (the NTC/anchor cells each traversal morphs) with the metadata
needed to recreate the crop — a handoff list for Ritvik to compute SetTransformer attention-head
pixel-patches on the real cells (to compare what the classifier attends to vs the generative morph).

Anchor cells are the top-N-by-attention cells per (marker, anchor), so this is complete for ALL
markers/anchor types regardless of whether that traversal's frames are cached yet. Phase geneKO +
complex share ONE NTC phase-anchor set; each fluor marker has its own NTC set (its channel's ranking).
"""
from __future__ import annotations

import pandas as pd

from ..classifier.config import PMA_PHASE_EBI
from ..classifier.data import _BASE_COLS
from . import catalog as C

OUT_CSV = f"{C.OUT}/viewer_assets/anchor_cells_for_attention.csv"
# geneKO (`gene`) + pathway membership (`ebi_complex`) columns included per your ask.
# NOTE: guideRNA is NOT in the pma source (no sgRNA column) — would need a per-cell join to
# guide-call data by (experiment, well, segmentation); omitted rather than faked.
# segmentation_id is the pma-source `segmentation` value → (experiment, well, segmentation_id) is the
# unique cell lookup key in the pma CSVs/parquets.
COLS = ["marker_channel", "anchor", "gene", "ebi_complex", "cell_index", "experiment", "well",
        "segmentation_id", "x_pheno", "y_pheno", "rank", "pma_attention"]


def build(n_cells=20, n_anchors=8, out_csv=OUT_CSV):
    """Every (marker, anchor) top-`n_cells` anchor set. Markers = phase + ALL fluor channels in the
    data (not just trained ones). Anchors = NTC + the marker's top-`n_anchors` complexes by
    EBI complex mAP (cross-phenotype A→B comparison). Future-proof: covers everything the cache holds."""
    import yaml
    cdist = C.complex_dist()                                       # complex(name) × reporter EBI mAP
    dist = C.dist_matrix()
    y = yaml.safe_load(open(C.EBI_YAML)) or {}
    gene2cx = {g: v["name"] for v in y.values() if isinstance(v, dict) for g in (v.get("genes") or [])}
    parts = []

    def top_cx(reporter):
        if not reporter or reporter not in cdist.columns:
            return []
        return list(cdist[reporter].dropna().sort_values(ascending=False).head(n_anchors).index)

    def add(df, mc, anchor):
        if df is None or not len(df):
            return
        d = df.sort_values("rank").head(n_cells).copy()
        d["marker_channel"] = mc; d["anchor"] = anchor
        d["ebi_complex"] = d["gene"].astype(str).map(gene2cx).fillna("")   # the geneKO's own EBI complex membership
        d["cell_index"] = range(len(d)); parts.append(d)

    # PHASE: NTC + top EBI-mAP complexes (from the EBI phase parquet; reporter col = "Phase")
    ph = pd.read_parquet(PMA_PHASE_EBI, columns=["gene", "predicted_class", "rank_type", *_BASE_COLS])
    ph = ph[ph["rank_type"] == "top"]
    add(ph[ph["gene"].astype(str) == "NTC"], "phase", "NTC")
    for cx in top_cx("Phase"):
        add(ph[ph["predicted_class"].astype(str) == cx], "phase", cx)

    # FLUOR: every channel × (NTC + its top EBI-mAP complexes for that reporter)
    fr = pd.read_csv(C.EBI_FLUOR_CSV, usecols=["gene", "channel", "predicted_class", "rank_type", *_BASE_COLS])
    fr = fr[fr["rank_type"] == "top"]
    for mc in sorted(fr["channel"].dropna().astype(str).unique()):
        sub = fr[fr["channel"] == mc]
        add(sub[sub["gene"].astype(str) == "NTC"], mc, "NTC")
        for cx in top_cx(C.rep_of(dist, mc)):
            add(sub[sub["predicted_class"].astype(str) == cx], mc, cx)

    df = pd.concat(parts, ignore_index=True).rename(columns={"segmentation": "segmentation_id"})[COLS]
    df.to_csv(out_csv, index=False)
    print(f"[anchor_cells] {len(df)} cells | {df['marker_channel'].nunique()} markers | "
          f"{df.groupby('marker_channel')['anchor'].nunique().mean():.1f} anchors/marker -> {out_csv}")
    return out_csv


if __name__ == "__main__":
    build()
