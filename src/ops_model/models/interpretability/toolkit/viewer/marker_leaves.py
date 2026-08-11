"""Resolve a viewer marker (manifest `marker_channel`) → its paper_v2 per-marker embedding leaf.

55 of 58 viewer markers have an isolated single-channel embedding under paper_v2/markers/:
    live fluorescent  markers/<leaf>/all_livecell/fixed_80%/cosine/
    Cell Painting     markers/<leaf>_cp/only_cp/all_livecell/fixed_80%/cosine/
    4i                markers/<leaf>_4i/only_4i/all_livecell/fixed_80%/cosine/
Phase is separate: paper_v2/phase_only/fixed_80%/cosine/.
(NFkB, Rb, gH2AX have no leaf — they keep the phase embedding.)

Each leaf holds gene_embedding_pca_optimized.h5ad (genes×101 PCs, obsm X_umap/X_phate/X_pca) +
per_signal/<name>_cells.h5ad (per-cell) + leiden_cache.pkl + metrics/.
"""
from __future__ import annotations

import os
import re

PAPER_V2 = "/hpc/projects/icd.fast.ops/organelle_attribution/pca_optimized_v0.3/cell_dino/zscore_per_exp/paper_v2"
MARKERS_ROOT = f"{PAPER_V2}/markers"
PHASE_LEAF = f"{PAPER_V2}/phase_only/fixed_80%/cosine"

# viewer marker_channel names whose CP leaf is spelled differently than the auto-normalizer would guess
_ALIAS = {
    "Endoplasmic Reticulum_Concanavalin A": "ER_ConA_cp",
    "Nucleus_Hoechst": "nuclei_Hoechst_cp",
    "Plasma Membrane_Wheat Germ Agglutinin": "plasma_membrane_WGA_cp",
}


def _norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _leaf_base(leaf):
    return re.sub(r"_(cp|4i)$", "", leaf)


def build_map(markers_root=MARKERS_ROOT):
    """{viewer_marker_channel: leaf_dir_name} for every viewer marker that has a paper_v2 leaf."""
    leaves = sorted(os.listdir(markers_root))
    by_base = {_norm(_leaf_base(l)): l for l in leaves}
    return leaves, by_base


def resolve_leaf(marker_channel, markers_root=MARKERS_ROOT):
    """Viewer marker_channel → leaf dir name (or None if no per-marker embedding exists)."""
    if marker_channel in _ALIAS:
        return _ALIAS[marker_channel]
    _, by_base = build_map(markers_root)
    for key in (_norm(marker_channel), _norm(marker_channel.split("_")[0])):   # full, then compartment token (4i = <gene>_<gene>)
        if key in by_base:
            return by_base[key]
    return None


def leaf_dir(leaf, markers_root=MARKERS_ROOT):
    """Leaf name → its cosine embedding directory (handles the only_cp / only_4i / all_livecell nesting)."""
    sub = "only_cp/all_livecell" if leaf.endswith("_cp") else "only_4i/all_livecell" if leaf.endswith("_4i") else "all_livecell"
    return f"{markers_root}/{leaf}/{sub}/fixed_80%/cosine"


def embedding_h5ad(marker_channel):
    """Viewer marker_channel → its gene_embedding_pca_optimized.h5ad path (phase falls back to the phase leaf)."""
    if not marker_channel or marker_channel.lower() in ("phase", "phase2d"):
        return f"{PHASE_LEAF}/gene_embedding_pca_optimized.h5ad"
    leaf = resolve_leaf(marker_channel)
    return f"{leaf_dir(leaf)}/gene_embedding_pca_optimized.h5ad" if leaf else None


if __name__ == "__main__":
    import json
    m = json.load(open("/hpc/projects/icd.fast.ops/models/diffex/viewer_assets/manifest.json"))
    n = miss = 0
    for x in m["markers"]:
        mc = x.get("marker_channel")
        if not mc:
            continue
        p = embedding_h5ad(mc)
        ok = bool(p and os.path.exists(p))
        n += ok
        if not ok:
            miss += 1; print(f"  no leaf: {mc}")
    print(f"{n} markers resolve to an existing embedding h5ad; {miss} without")
