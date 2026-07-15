"""Export per-gene OP/CP morphometric feature values for the Embedding tab's 'Color by' feature coloring.

Reads gene_feature_means.h5ad (genes × 3113 OP/CP phase features), collapses variants to base names
(same _base_name dedup as the PC features), robustly normalizes each feature to 0–1 across genes (2–98
pct) for the colormap, and writes a compact lookup the montage colors points by.

  viewer_assets/montage_features.json
    {"features": [base names], "range": {feat: [lo, hi]}, "values": {gene: [0..1 per feature | null]}}

  python -m ops_model.models.attention.diffex.viewer.build_montage_features
"""
from __future__ import annotations

import json
import os
from collections import OrderedDict

import numpy as np

from . import catalog as C
from .build_pc_features import _base_name

FM = "/hpc/projects/icd.fast.ops/analysis/pc_feature_correlation/phase_only/gene_feature_means.h5ad"
OUT = f"{C.OUT}/viewer_assets/montage_features.json"


def build(fm_path=FM, out=OUT):
    import anndata as ad
    fm = ad.read_h5ad(fm_path)
    genes = [str(g) for g in fm.obs_names]
    feats = [str(v) for v in fm.var_names]
    X = np.asarray(fm.X, dtype=np.float64)                  # genes × features

    groups = OrderedDict()                                  # collapse variants → base name (mean of members)
    for j, f in enumerate(feats):
        groups.setdefault(_base_name(f), []).append(j)
    base = sorted(groups)
    D = np.column_stack([np.nanmean(X[:, groups[b]], axis=1) for b in base])   # genes × base
    print(f"[montage-feat] {len(feats)} features → {len(base)} deduped base names over {len(genes)} genes")

    lo = np.nanpercentile(D, 2, axis=0)
    hi = np.nanpercentile(D, 98, axis=0)
    span = np.where(hi > lo, hi - lo, 1.0)
    N = np.clip((D - lo) / span, 0, 1)                      # robust 0–1 per feature for the colormap

    fin = lambda v, nd: (round(float(v), nd) if np.isfinite(v) else None)   # non-finite → null (valid JSON; NaN is not)
    values = {}
    for i, g in enumerate(genes):
        values[g] = [fin(N[i, j], 3) for j in range(len(base))]
    rng = {b: [fin(lo[j], 4), fin(hi[j], 4)] for j, b in enumerate(base)}

    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump({"features": base, "range": rng, "values": values}, f, allow_nan=False)   # fail loud if any NaN slips through
    print(f"[montage-feat] -> {out} ({os.path.getsize(out) / 1e6:.1f} MB)")
    return out


if __name__ == "__main__":
    build()
