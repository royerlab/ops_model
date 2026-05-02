"""Hand-annotated cluster animation on the gene UMAP.

Reads a plain-text annotation file (cluster name on its own line ending in
``:``, followed by one gene symbol per line; clusters separated by blank
lines), looks each gene up in ``gene_embedding_pca_optimized.h5ad``, and
writes a GIF that walks through the clusters in file order — each frame
shows the full gene UMAP in gray with that cluster's genes highlighted.

The UMAP coords are read straight from the h5ad's ``obsm["X_umap"]`` (no
refit), so the animation matches whatever layout you most recently locked
in via ``--overlays-only`` / ``--sweep-seed`` / etc.

Run with::

    # default annotation file + default run dir
    python -m ops_model.post_process.combination.hand_annotated_umap_animation

    # custom paths / cadence
    python -m ops_model.post_process.combination.hand_annotated_umap_animation \\
        --annotations /path/to/clusters.txt --run-dir /path/to/run \\
        --frame-ms 2000
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import anndata as ad
import numpy as np


DEFAULT_RUN_DIR = (
    "/hpc/projects/icd.fast.ops/organelle_attribution/pca_optimized_v0.3/"
    "cell_dino/zscore_per_exp/paper_v1/all/fixed_80%/cosine/second_pca_consensus"
)
DEFAULT_ANNOTATIONS = "/hpc/mydata/gav.sturm/ops_mono/hand_annotated_cluster.txt"

# Per-cluster representative cell is pulled from Alex's attention CSVs. The
# cluster's `# rep_modality` (fluor / phase) decides which CSV is queried.
DEFAULT_FLUOR_CSV = (
    "/home/gav.sturm/linked_folders/icd.fast.ops/models/alex_lin_attention/"
    "pma_top_fluorescent_cells_v2.csv"
)
DEFAULT_PHASE_CSV = (
    "/home/gav.sturm/linked_folders/icd.fast.ops/models/alex_lin_attention/"
    "pma_top_phase_cells_v2.csv"
)
# Reuse cell-loading machinery (StoreCache, BaseDataset wrapper, channel
# resolvers, contrast policies) from attention_atlas — same modality, dataset
# family, conventions. attention_atlas is a script (not a package), so we add
# its dir to sys.path on first use.
_ATTENTION_ATLAS_DIR = (
    "/hpc/mydata/gav.sturm/ops_mono/ops_process/ops_analysis/napari"
)


def parse_clusters(path: Path) -> List[Dict]:
    """Parse the annotation file into a list of cluster dicts in file order.

    Each cluster header is a non-comment line ending in ``:``; the colon is
    stripped to give the cluster name. Subsequent ``# key: value`` lines
    become metadata fields (``well_known``, ``rep_gene``, ``rep_channel``,
    ``ch_rank``, ``rep_modality``, ``supercluster``); plain non-blank lines
    are gene symbols. Blank lines separate clusters.

    Returns dicts with keys ``name`` and ``genes`` plus whatever metadata
    fields were present.
    """
    META_KEYS = {
        "supercluster", "well_known", "rep_gene",
        "rep_channel", "ch_rank", "rep_modality",
    }
    clusters: List[Dict] = []
    current: Optional[Dict] = None

    def _flush():
        nonlocal current
        if current and current.get("genes"):
            clusters.append(current)
        current = None

    for raw in Path(path).read_text().splitlines():
        line = raw.strip()
        if not line:
            _flush()
            continue
        if line.endswith(":") and not line.startswith("#"):
            _flush()
            current = {"name": line[:-1].strip(), "genes": []}
        elif line.startswith("#"):
            if current is None:
                continue
            body = line.lstrip("#").strip()
            if ":" not in body:
                continue
            key, val = body.split(":", 1)
            key = key.strip()
            if key in META_KEYS:
                current[key] = val.strip()
        else:
            if current is not None:
                current["genes"].append(line)
    _flush()
    return clusters


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


_NTC_COLOR = "#ff5a5a"
_BG_COLOR = "#aab1c2"   # background scatter dots — light gray on dark canvas
_DARK_BG = "#0c1018"    # figure + axes face color
_AXIS_FG = "#cfd6e0"    # axis label / tick / title color on dark
_GRID_COLOR = "#3a4253"

# 8 super-categories from organelle_profiler/configs/gene_supercategory_mapping.yaml.
# Each hand-annotated cluster is assigned to exactly one. Colors distinct enough
# to read on the dark-gray background.
# Brightened palette tuned to read against the dark canvas — purples /
# blues / greens lifted toward pastel so they pop instead of muddying.
SUPERCATEGORY_COLORS: Dict[str, str] = {
    "Translation":               "#ffa84a",  # light orange
    "Transcription":             "#e056d8",  # vivid magenta — distinct from
                                              #                 the cyan used
                                              #                 for Membrane
                                              #                 Trafficking
    "Cell Cycle & DNA":          "#c79deb",  # light purple
    "Signaling":                 "#c8907a",  # light brown (unused; clusters
                                              #             folded into Metabolism)
    "Membrane Trafficking":      "#5fdee8",  # light cyan
    "Metabolism":                "#6cd86c",  # light green
    "Protein Homeostasis":       "#dadc4e",  # light olive
    "Cytoskeleton & Morphology": "#f5a6dc",  # light pink
}

# Hand-annotated cluster -> set of compartments to highlight in the cell
# schematic inset. Each compartment id matches a key in CELL_SCHEMATIC below.
HAND_CLUSTER_TO_COMPARTMENT: Dict[str, List[str]] = {
    # Cytoskeleton & Morphology
    "Actin Organization":                ["cytoskeleton"],
    "Cytoskeleton Organization":         ["cytoskeleton"],
    "Intermediate Filament Cytoskeleton": ["cytoskeleton"],
    "Actin Nucleation":                  ["cytoskeleton"],
    "Actin Projections":                 ["cytoskeleton", "plasma_membrane"],
    # Mitochondria
    "Mitochondria Inner Membrane OXPHOS": ["mitochondria"],
    "Mitochondrial Import":              ["mitochondria"],
    "Cristae Formation":                 ["mitochondria"],
    "Mitochondrial Outer Membrane":      ["mitochondria"],
    # Other organelles / metabolism
    "Peroxisome":                        ["peroxisome"],
    "Galactose Metabolism":              ["cytoplasm"],
    "Lipid Metabolism":                  ["er", "cytoplasm"],
    "Purine Metabolism":                 ["cytoplasm"],
    "Triglyceride Metabolism":           ["er", "cytoplasm"],
    # Signaling
    "Hypoxia":                           ["cytoplasm", "nucleus"],
    "Calcium Signaling":                 ["er", "plasma_membrane"],
    "Calcium-dependent Exocytosis":      ["vesicles", "plasma_membrane"],
    "MTOR Signaling":                    ["lysosome", "cytoplasm"],
    "Potassium Channels":                ["plasma_membrane"],
    "Sodium Channels":                   ["plasma_membrane"],
    # Membrane Trafficking
    "Tight Junctions":                   ["plasma_membrane"],
    "Golgi Organization":                ["golgi"],
    "Exocytic Vesicles":                 ["vesicles", "plasma_membrane"],
    "Vacuolar Acidification":            ["lysosome"],
    "Exosomal Secretion":                ["vesicles", "plasma_membrane"],
    "Endocytosis":                       ["vesicles", "plasma_membrane"],
    "ER membrane":                       ["er"],
    "ER Tubular Network":                ["er"],
    "ER-Golgi Transport":                ["er", "golgi"],
    "Lysosome":                          ["lysosome"],
    # Gene Expression
    "snRNA":                             ["nucleus"],
    "Transcription Initiation":          ["nucleus"],
    "DNA-templated Transcription":       ["nucleus"],
    "Spliceosome Assembly":              ["nucleus"],
    "Splicing":                          ["nucleus"],
    "RNA Polymerase":                    ["nucleolus"],
    "Nuclear Export":                    ["nucleus"],
    # Translation
    "Translation Initiation":            ["er", "cytoplasm"],
    "Ribosomal Large Subunit":           ["cytoplasm"],
    "Ribosomal Small Subunit":           ["cytoplasm"],
    "Ribosomal Biogenesis":              ["nucleolus"],
    "RNA Degradation":                   ["cytoplasm"],
    # Cell Cycle & DNA
    "DNA Methylation":                   ["nucleus"],
    "DNA recombination":                 ["nucleus"],
    "DNA Replication":                   ["nucleus"],
    "Spindle Assembly":                  ["cytoskeleton", "nucleus"],
    "Microtubule Nucleation":            ["cytoskeleton"],
    "Nuclear Deformation":               ["nucleus"],
}


# Hand-annotated cluster name -> super-category. Lookups are exact (not
# substring) so renames in the annotation file must be reflected here.
HAND_CLUSTER_TO_SUPER: Dict[str, str] = {
    # Cytoskeleton & Morphology
    "Actin Organization":                "Cytoskeleton & Morphology",
    "Cytoskeleton Organization":         "Cytoskeleton & Morphology",
    "Intermediate Filament Cytoskeleton": "Cytoskeleton & Morphology",
    "Actin Nucleation":                  "Cytoskeleton & Morphology",
    "Actin Projections":                 "Cytoskeleton & Morphology",
    # Metabolism
    "Mitochondria Inner Membrane OXPHOS": "Metabolism",
    "Mitochondrial Import":              "Metabolism",
    "Cristae Formation":                 "Metabolism",
    "Mitochondrial Outer Membrane":      "Metabolism",
    "Peroxisome":                        "Membrane Trafficking",
    "Galactose Metabolism":              "Metabolism",
    "Lipid Metabolism":                  "Metabolism",
    "Purine Metabolism":                 "Metabolism",
    "Triglyceride Metabolism":           "Metabolism",
    # Signaling — folded into Metabolism for this animation
    "Hypoxia":                           "Metabolism",
    "Calcium Signaling":                 "Metabolism",
    "Calcium-dependent Exocytosis":      "Metabolism",
    "MTOR Signaling":                    "Metabolism",
    "Potassium Channels":                "Metabolism",
    "Sodium Channels":                   "Metabolism",
    # Membrane Trafficking
    "Tight Junctions":                   "Metabolism",
    "Golgi Organization":                "Membrane Trafficking",
    "Exocytic Vesicles":                 "Membrane Trafficking",
    "Vacuolar Acidification":            "Membrane Trafficking",
    "Exosomal Secretion":                "Membrane Trafficking",
    "Endocytosis":                       "Membrane Trafficking",
    "ER membrane":                       "Membrane Trafficking",
    "ER Tubular Network":                "Membrane Trafficking",
    "ER-Golgi Transport":                "Membrane Trafficking",
    "Lysosome":                          "Membrane Trafficking",
    # Gene Expression
    "snRNA":                             "Transcription",
    "Transcription Initiation":          "Transcription",
    "DNA-templated Transcription":       "Transcription",
    "Spliceosome Assembly":              "Transcription",
    "Splicing":                          "Transcription",
    "RNA Polymerase":                    "Translation",
    "Nuclear Export":                    "Transcription",
    # Translation
    "Translation Initiation":            "Translation",
    "Ribosomal Large Subunit":           "Translation",
    "Ribosomal Small Subunit":           "Translation",
    "Ribosomal Biogenesis":              "Translation",
    "RNA Degradation":                   "Translation",
    # Cell Cycle & DNA
    "DNA Methylation":                   "Transcription",
    "DNA recombination":                 "Cell Cycle & DNA",
    "DNA Replication":                   "Cell Cycle & DNA",
    "Spindle Assembly":                  "Cell Cycle & DNA",
    "Microtubule Nucleation":            "Cytoskeleton & Morphology",
    "Nuclear Deformation":               "Cell Cycle & DNA",
}


def _wrap_label(text: str, width: int = 20) -> str:
    """Wrap ``text`` on word boundaries so each rendered line is <= ``width``
    characters. ``break_long_words=False`` keeps multi-syllable words intact."""
    import textwrap

    return "\n".join(textwrap.wrap(
        text, width=width, break_long_words=False, break_on_hyphens=False,
    )) or text


def _centroid_label(ax, coords_subset: np.ndarray, text: str, color: str,
                    fontsize: float = 14, fontweight: str = "bold"):
    """Place ``text`` at the median (x, y) of ``coords_subset`` with a soft
    white halo so it stays legible against the scatter. Long names wrap on
    word boundaries at ~20 chars per line, centered."""
    if coords_subset.size == 0:
        return
    cx, cy = np.median(coords_subset, axis=0)
    wrapped = _wrap_label(text, width=20)
    txt = ax.text(
        cx, cy, wrapped,
        ha="center", va="center", multialignment="center",
        fontsize=fontsize, fontweight=fontweight, color=color,
        zorder=10,
    )
    import matplotlib.patheffects as pe

    # Dark halo so labels stay legible against the dark canvas.
    txt.set_path_effects([
        pe.Stroke(linewidth=3.5, foreground="#0c1018", alpha=0.95),
        pe.Normal(),
    ])


# Cell-schematic geometry. Each entry: (kind, params). Coordinates are in
# the inset axes' [0,1] x [0,1] space. Drawn in order, so later entries
# render on top of earlier ones (membrane → cytoplasm → organelles → labels).
_BG_FILL = "#e9ecef"
_OUTLINE = "#4a4a4a"


def _bean_patch(cx, cy, length, width, angle_deg, **patch_kw):
    """Kidney-bean shape via the parametric curve
    ``y = sin(t) + 0.6 * sin(2t)`` (asymmetric profile with one indented
    side). Returns a single ``PathPatch`` that can be added to the axes."""
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path as MplPath

    n = 80
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    a = length / 2.0
    b = width / 2.0
    # Local coords (centered at origin, indent on the left).
    x_local = a * np.cos(t)
    y_local = b * (np.sin(t) + 0.6 * np.sin(2 * t))

    # Rotate into place around (cx, cy).
    th = np.deg2rad(angle_deg)
    cos_th, sin_th = np.cos(th), np.sin(th)
    xs = cx + x_local * cos_th - y_local * sin_th
    ys = cy + x_local * sin_th + y_local * cos_th

    verts = list(zip(xs, ys)) + [(xs[0], ys[0])]
    codes = [MplPath.MOVETO] + [MplPath.LINETO] * (len(verts) - 2) + [MplPath.CLOSEPOLY]
    return PathPatch(MplPath(verts, codes), **patch_kw)


def _irregular_cell_outline_path(n: int = 240):
    """Closed Path for an amoeboid cell with two elongated ends. Fixed seed
    so the silhouette is identical across frames."""
    from matplotlib.path import Path as MplPath

    rng = np.random.default_rng(7)
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    # Base radius with a bilobed stretch (longer along x than y) — gives the
    # shape two elongated ends without breaking convexity wildly.
    r = (
        0.40
        + 0.05 * np.cos(2 * theta)         # bilobed elongation
        + 0.025 * np.sin(3 * theta)        # gentle asymmetry
        + 0.015 * np.cos(5 * theta)        # high-frequency wobble
        + 0.012 * rng.standard_normal(n)   # tiny noise so it doesn't look mathematical
    )
    x = 0.50 + r * np.cos(theta) * 1.25
    y = 0.50 + r * np.sin(theta) * 0.95
    verts = list(zip(x, y)) + [(x[0], y[0])]
    codes = [MplPath.MOVETO] + [MplPath.LINETO] * (len(verts) - 2) + [MplPath.CLOSEPOLY]
    return MplPath(verts, codes)


def _arc_polygon(cx, cy, r_inner, r_outer, theta0_deg, theta1_deg, n=40):
    """Filled annular sector polygon (used for ER arcs wrapping the nucleus)."""
    from matplotlib.path import Path as MplPath

    t0, t1 = np.deg2rad(theta0_deg), np.deg2rad(theta1_deg)
    outer_t = np.linspace(t0, t1, n)
    inner_t = np.linspace(t1, t0, n)
    outer_pts = np.column_stack([cx + r_outer * np.cos(outer_t), cy + r_outer * np.sin(outer_t) * 0.85])
    inner_pts = np.column_stack([cx + r_inner * np.cos(inner_t), cy + r_inner * np.sin(inner_t) * 0.85])
    pts = np.vstack([outer_pts, inner_pts, outer_pts[:1]])
    codes = [MplPath.MOVETO] + [MplPath.LINETO] * (len(pts) - 2) + [MplPath.CLOSEPOLY]
    return MplPath(pts, codes)


_HL_SCALE = 1.55           # how much bigger highlighted organelles render
_DIM_ALPHA = 0.18          # fill alpha for non-highlighted organelles —
                            # very faint so the active compartment dominates


def _draw_cell_schematic(
    fig,
    highlighted: List[str],
    color: str,
    *,
    rect: Tuple[float, float, float, float] = (0.78, 0.06, 0.21, 0.21),
):
    """Render a cartoon cell in a fixed-position inset, with the given
    compartment ids drawn in the highlight color (and enlarged) and
    everything else in faded pastel. ``rect`` is (left, bottom, w, h) in
    figure coords."""
    from matplotlib.patches import Circle, Ellipse, PathPatch

    ax = fig.add_axes(rect)
    # Slight padding around [0,1] so the irregular cell outline (which can
    # extend a bit past the unit square along its elongated axes) renders
    # in full without being clipped at the inset edges.
    ax.set_xlim(-0.10, 1.10)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect("equal")
    ax.axis("off")

    hl = set(highlighted or [])
    is_hl = lambda n: n in hl  # noqa: E731

    def _fc(name, default):
        return color if is_hl(name) else default

    def _ec(name, default):
        return color if is_hl(name) else default

    def _lw(name, default):
        return 2.0 if is_hl(name) else default

    def _alpha(name, dim=_DIM_ALPHA):
        return 1.0 if is_hl(name) else dim

    def _zoom(name):
        return _HL_SCALE if is_hl(name) else 1.0

    # ---- Cell outline (irregular, elongated) + cytoplasm fill ------------
    cell_path = _irregular_cell_outline_path()
    cytoplasm_fill = color if is_hl("cytoplasm") else "#1a2030"
    pm_edge = color if is_hl("plasma_membrane") else "#cfd6e0"
    pm_lw = 2.4 if is_hl("plasma_membrane") else 1.2
    ax.add_patch(PathPatch(
        cell_path,
        facecolor=cytoplasm_fill,
        edgecolor=pm_edge,
        linewidth=pm_lw,
        alpha=_alpha("cytoplasm") if is_hl("cytoplasm") else 1.0,
        zorder=1,
    ))

    # Nucleus is centered. ER, mitochondria, etc. position relative to it.
    nuc_cx, nuc_cy = 0.50, 0.50
    nuc_rx, nuc_ry = 0.16, 0.14

    # ---- ER (3 arc-bands wrapping the nucleus) ---------------------------
    er_fc = _fc("er", "#cfe7d3")
    er_ec = _ec("er", "#3e7c52")
    er_lw = _lw("er", 0.9)
    er_alpha = _alpha("er")
    er_zoom = _zoom("er")
    er_inner = max(nuc_rx, nuc_ry) + 0.02
    for r_in, r_out, t0, t1 in [
        (er_inner + 0.000, er_inner + 0.030 * er_zoom,  20, 160),
        (er_inner + 0.040, er_inner + 0.065 * er_zoom, 200, 340),
        (er_inner + 0.080, er_inner + 0.105 * er_zoom,  -5, 110),
    ]:
        p = _arc_polygon(nuc_cx, nuc_cy, r_in, r_out, t0, t1)
        ax.add_patch(PathPatch(
            p, facecolor=er_fc, edgecolor=er_ec, linewidth=er_lw,
            alpha=er_alpha, zorder=3,
        ))

    # ---- Nucleus + nucleolus (centered, fixed size — no zoom-on-highlight
    # because enlarging the nucleus pushes ER/cytoskeleton out of place and
    # looks worse than just the color change) -----------------------------
    ax.add_patch(Ellipse(
        (nuc_cx, nuc_cy), nuc_rx * 2, nuc_ry * 2,
        facecolor=_fc("nucleus", "#d6def0"),
        edgecolor=_ec("nucleus", "#3a4a80"),
        linewidth=_lw("nucleus", 1.2),
        alpha=_alpha("nucleus"), zorder=5,
    ))
    ax.add_patch(Ellipse(
        (nuc_cx + 0.02, nuc_cy + 0.005), 0.08, 0.06,
        facecolor=_fc("nucleolus", "#7e8db8"),
        edgecolor=_ec("nucleolus", "#2a3460"),
        linewidth=_lw("nucleolus", 0.8),
        alpha=_alpha("nucleolus"), zorder=6,
    ))

    # ---- Golgi (stacked curved bands, near top of nucleus) ---------------
    golgi_fc = _fc("golgi", "#ffe1b3")
    golgi_ec = _ec("golgi", "#b06b14")
    golgi_lw = _lw("golgi", 0.9)
    golgi_alpha = _alpha("golgi")
    golgi_zoom = _zoom("golgi")
    for dy in [0.0, 0.025, 0.050, 0.075]:
        ax.add_patch(Ellipse(
            (0.78, 0.66 - dy),
            0.13 * golgi_zoom, 0.018 * golgi_zoom,
            facecolor=golgi_fc, edgecolor=golgi_ec,
            linewidth=golgi_lw, alpha=golgi_alpha, zorder=4,
        ))

    # ---- Mitochondria (bean shapes scattered around the nucleus) --------
    mito_fc = _fc("mitochondria", "#f5c0c0")
    mito_ec = _ec("mitochondria", "#9c2f2f")
    mito_lw = _lw("mitochondria", 0.9)
    mito_alpha = _alpha("mitochondria")
    mito_zoom = _zoom("mitochondria")
    bean_kw = dict(
        facecolor=mito_fc, edgecolor=mito_ec, linewidth=mito_lw,
        alpha=mito_alpha, zorder=4,
    )
    for cx, cy, length, width, ang in [
        (0.16, 0.32, 0.13, 0.045,  20),
        (0.20, 0.72, 0.11, 0.040, -25),
        (0.50, 0.18, 0.13, 0.045,  10),
        (0.82, 0.32, 0.11, 0.040,  60),
        (0.86, 0.78, 0.10, 0.038, -45),
        (0.36, 0.83, 0.10, 0.038,  10),
    ]:
        ax.add_patch(_bean_patch(
            cx, cy, length * mito_zoom, width * mito_zoom, ang, **bean_kw,
        ))

    # ---- Lysosome (one circle) -------------------------------------------
    ax.add_patch(Circle(
        (0.74, 0.30), 0.040 * _zoom("lysosome"),
        facecolor=_fc("lysosome", "#dcb8ec"),
        edgecolor=_ec("lysosome", "#6f3a8c"),
        linewidth=_lw("lysosome", 0.9),
        alpha=_alpha("lysosome"), zorder=4,
    ))

    # ---- Peroxisome (smaller circle) -------------------------------------
    ax.add_patch(Circle(
        (0.28, 0.50), 0.028 * _zoom("peroxisome"),
        facecolor=_fc("peroxisome", "#cfe1f5"),
        edgecolor=_ec("peroxisome", "#26528c"),
        linewidth=_lw("peroxisome", 0.9),
        alpha=_alpha("peroxisome"), zorder=4,
    ))

    # ---- Vesicles (many small spheres scattered through the cytoplasm) ---
    ves_fc = _fc("vesicles", "#f0e2ff")
    ves_ec = _ec("vesicles", "#5a3aa6")
    ves_lw = _lw("vesicles", 0.6)
    ves_alpha = _alpha("vesicles")
    ves_zoom = _zoom("vesicles")
    # Positions chosen between the nucleus (~r=0.16 from 0.5,0.5) and the
    # cell periphery, avoiding overlap with the bean cluster of mitochondria
    # and the Golgi stack on the right.
    for cx, cy, r in [
        (0.30, 0.42, 0.011),
        (0.32, 0.58, 0.012),
        (0.32, 0.66, 0.010),
        (0.40, 0.72, 0.013),
        (0.50, 0.72, 0.011),
        (0.60, 0.68, 0.012),
        (0.65, 0.55, 0.011),
        (0.66, 0.42, 0.013),
        (0.55, 0.34, 0.011),
        (0.43, 0.34, 0.012),
        (0.36, 0.50, 0.010),
        (0.70, 0.50, 0.011),
        (0.45, 0.62, 0.010),
        (0.55, 0.62, 0.011),
    ]:
        ax.add_patch(Circle(
            (cx, cy), r * ves_zoom,
            facecolor=ves_fc, edgecolor=ves_ec, linewidth=ves_lw,
            alpha=ves_alpha, zorder=4,
        ))

    # ---- Cytoskeleton (short scattered fragments in the cytoplasm) -------
    # NOT radial from the nucleus — those looked like spokes and overshot
    # the membrane. Instead: short tilted segments scattered through the
    # cytoplasm at irregular positions and angles, like loose filament
    # fragments. Hand-picked positions/lengths keep the look stable across
    # frames while reading as 'less regular'.
    cyto_color = color if is_hl("cytoskeleton") else "#888888"
    cyto_lw = 1.6 if is_hl("cytoskeleton") else 0.7
    cyto_alpha = 0.80 if is_hl("cytoskeleton") else 0.30
    # (cx, cy, length, angle_deg)
    cyto_segments = [
        (0.30, 0.40, 0.07,   25),
        (0.66, 0.65, 0.06,  -15),
        (0.40, 0.72, 0.08,   55),
        (0.62, 0.40, 0.07,  110),
        (0.34, 0.62, 0.06,  -50),
        (0.55, 0.35, 0.05,   80),
        (0.72, 0.55, 0.06,   30),
        (0.28, 0.55, 0.05, -100),
    ]
    for cx, cy, length, ang in cyto_segments:
        th = np.deg2rad(ang)
        dx = (length / 2) * np.cos(th)
        dy = (length / 2) * np.sin(th)
        ax.plot([cx - dx, cx + dx], [cy - dy, cy + dy],
                color=cyto_color, linewidth=cyto_lw, alpha=cyto_alpha,
                solid_capstyle="round", zorder=2)

    # No labels overlaid on the cell — the highlighted compartment's color +
    # size change is the only signal in this inset.


def _load_representative_cells(
    clusters: List[Dict],
    fluor_csv: Path,
    phase_csv: Path,
    crop_size: int = 306,
) -> Dict[str, Dict]:
    """Resolve and load the representative cell per cluster.

    For each cluster's ``rep_gene`` / ``rep_channel`` / ``rep_modality``,
    pulls the top-attention row from the matching CSV (fluor: top by
    ``pma_attention`` for that (gene, viz_channel); phase: rank=1 for that
    gene), then loads the crop + segmentation mask via attention_atlas's
    ``BaseDataset`` wrapper. Returns a dict keyed by cluster name with
    ``crop``, ``mask``, ``vmin``, ``vmax``, ``gene``, ``channel``,
    ``modality``. Missing entries silently fall through — the renderer draws
    an "n/a" placeholder.
    """
    import sys
    if _ATTENTION_ATLAS_DIR not in sys.path:
        sys.path.insert(0, _ATTENTION_ATLAS_DIR)
    from attention_atlas import (  # type: ignore[import-not-found]
        StoreCache, _build_base_dataset,
        _resolve_fluor_channel, _resolve_phase_channel,
        FLUOR_TILE_PERCENTILES, PHASE_CONTRAST,
    )
    import pandas as pd

    fluor_df = pd.read_csv(fluor_csv)
    if "rank_type" in fluor_df.columns:
        fluor_df = fluor_df[fluor_df["rank_type"] == "top"]
    phase_df = pd.read_csv(phase_csv)

    # Oversample N candidates per cluster (top-N by attention); the picker
    # below selects the first with a valid cell mask. Top-1 cells sometimes
    # land on a seg ID that's missing from cell_seg / cp_cell_seg, leaving
    # the inverse-mask overlay either empty (no cell) or all-True (masking
    # everything as background). Falling through to the next-attention cell
    # avoids those failures.
    N_CANDIDATES = 5
    MIN_MASK_PIXELS = 25  # match attention_atlas's NTC_MIN_MASK_PIXELS

    cell_rows: List[Dict] = []
    candidate_meta: List[Dict] = []
    for c in clusters:
        gene = c.get("rep_gene")
        ch = c.get("rep_channel")
        mod = c.get("rep_modality", "fluor")
        if not gene or not ch:
            continue
        if mod == "phase":
            sub = phase_df[phase_df["gene"] == gene]
            if "rank" in sub.columns:
                sub = sub.sort_values("rank")
            sub = sub.head(N_CANDIDATES)
        else:
            sub = fluor_df[
                (fluor_df["gene"] == gene)
                & (fluor_df["viz_channel"] == ch)
            ].sort_values("pma_attention", ascending=False).head(N_CANDIDATES)
        if sub.empty:
            print(f"  [{c['name']}] no top cell for {gene}/{ch}/{mod}", flush=True)
            continue
        for within_rank, (_, row) in enumerate(sub.iterrows()):
            cell_rows.append({
                "experiment": str(row["experiment"]),
                "well": str(row["well"]),
                "y_pheno": float(row["y_pheno"]),
                "x_pheno": float(row["x_pheno"]),
                "segmentation": int(row["segmentation"]),
                "gene": gene,
                "kind": "phase" if mod == "phase" else "fluor",
            })
            candidate_meta.append({
                "cluster_name": c["name"],
                "gene": gene, "ch": ch, "mod": mod,
                "within_rank": within_rank,
            })

    if not cell_rows:
        return {}

    store_cache = StoreCache()
    for r in cell_rows:
        store_cache.get(r["experiment"])

    ds, input_indices = _build_base_dataset(cell_rows, store_cache, crop_size)
    if ds is None:
        return {}

    # Replace attention_atlas's mask loader with one that tries ALL label
    # preferences and only "succeeds" when the seg-ID lookup returns
    # non-empty pixels. The upstream version caches the first label that
    # successfully *loads* — which for CP experiments is `cp_cell_seg`, but
    # the fluor CSV's seg-ID is often the live-cell `cell_seg` ID, so the
    # mask comes back all-False (empty).
    import types
    import pandas as pd_local  # avoid shadowing outer pd

    def _smart_mask(self, ci, bbox):
        mh = bbox[2] - bbox[0]
        mw = bbox[3] - bbox[1]
        if pd_local.isna(ci.segmentation_id):
            return np.ones((1, mh, mw), dtype=bool)
        seg_id = int(ci.segmentation_id)
        for label in ("cell_seg", "cp_cell_seg", "4i_cell_seg"):
            try:
                arr = self.load_label_array(ci.store_key, ci.well, label, bbox)
                if arr is None:
                    continue
                sc = (arr == seg_id)
                if sc.any():
                    return np.expand_dims(sc, axis=0)
            except Exception:
                continue
        return np.ones((1, mh, mw), dtype=bool)

    ds.add_mask_to_batch = types.MethodType(_smart_mask, ds)

    # Group all loaded candidates by cluster, then pick the first with a
    # valid mask (>= MIN_MASK_PIXELS but not the all-True fallback).
    candidates_by_cluster: Dict[str, List[Dict]] = {}
    for ds_idx, in_idx in enumerate(input_indices):
        meta = candidate_meta[in_idx]
        try:
            batch = ds[ds_idx]
            data = batch["data"].numpy()
            mask = batch["mask"].numpy()[0].astype(bool)
        except Exception as e:
            print(
                f"  [{meta['cluster_name']}] cell load failed at rank "
                f"{meta['within_rank']}: {e}", flush=True,
            )
            continue
        n_mask = int(mask.sum())
        valid = MIN_MASK_PIXELS <= n_mask < mask.size
        candidates_by_cluster.setdefault(meta["cluster_name"], []).append({
            "in_idx": in_idx,
            "data": data,
            "mask": mask,
            "valid": valid,
            "n_mask": n_mask,
            **meta,
        })

    out: Dict[str, Dict] = {}
    for cluster_name, cands in candidates_by_cluster.items():
        cands.sort(key=lambda x: x["within_rank"])
        chosen = next((c for c in cands if c["valid"]), None)
        if chosen is None:
            # All candidates had bad masks — render top-1 anyway. The
            # inverse-mask overlay will fall back gracefully.
            chosen = cands[0]
            print(
                f"  [{cluster_name}] no valid mask in top {len(cands)} "
                f"(pixels: {[c['n_mask'] for c in cands]}); using rank-1 anyway",
                flush=True,
            )
        elif chosen["within_rank"] > 0:
            print(
                f"  [{cluster_name}] using rank-{chosen['within_rank'] + 1} "
                f"(rank-{chosen['within_rank']} had invalid mask)",
                flush=True,
            )

        in_idx = chosen["in_idx"]
        data = chosen["data"]
        mask = chosen["mask"]
        gene = chosen["gene"]
        ch = chosen["ch"]
        mod = chosen["mod"]

        store = store_cache.get(cell_rows[in_idx]["experiment"])
        if store is None:
            continue
        ch_names = list(store.channel_names)

        if mod == "phase":
            zarr_ch = _resolve_phase_channel(store)
            vmin, vmax = PHASE_CONTRAST
        else:
            cm = store_cache.channel_map(cell_rows[in_idx]["experiment"])
            zarr_ch = _resolve_fluor_channel(store, cm, ch)
            if zarr_ch is None or zarr_ch not in ch_names:
                print(
                    f"  [{cluster_name}] could not resolve channel "
                    f"{ch!r} -> zarr (channels: {ch_names})",
                    flush=True,
                )
                continue
            crop_full = data[ch_names.index(zarr_ch)]
            if mask.any() and mask.sum() < mask.size:
                pixels = crop_full[mask]
            else:
                # Fall back to full-crop percentiles when the mask is empty
                # or all-True so we still get a usable contrast range.
                pixels = crop_full.ravel()
            if pixels.size:
                lo, hi = np.percentile(pixels, FLUOR_TILE_PERCENTILES)
                vmin, vmax = float(lo), float(hi)
                if vmax <= vmin:
                    vmax = vmin + 1e-6
            else:
                vmin, vmax = None, None

        if zarr_ch is None or zarr_ch not in ch_names:
            continue
        crop = data[ch_names.index(zarr_ch)]

        # Same cell, Phase2D channel — for the side-by-side phase view. We
        # require *exactly* "Phase2D" (not the legacy "Phase" fallback) because
        # that's the modality these attention scores were trained against.
        # Use the same masked-pixel percentile policy as fluor (vs. a fixed
        # PHASE_CONTRAST) — the dynamic range varies by experiment, and tile-
        # specific clims keep faint cells from washing out and bright ones
        # from clipping.
        phase_crop = (
            data[ch_names.index("Phase2D")] if "Phase2D" in ch_names else None
        )
        if phase_crop is not None:
            if mask.any() and mask.sum() < mask.size:
                pixels = phase_crop[mask]
            else:
                pixels = phase_crop.ravel()
            if pixels.size:
                p_lo, p_hi = np.percentile(pixels, FLUOR_TILE_PERCENTILES)
                phase_vmin = float(p_lo)
                phase_vmax = float(p_hi)
                if phase_vmax <= phase_vmin:
                    phase_vmax = phase_vmin + 1e-6
            else:
                phase_vmin = phase_vmax = None
        else:
            phase_vmin = phase_vmax = None

        # Segmentation prediction channels — overlaid on top of Phase2D so the
        # nuclear envelope + plasma membrane prediction probabilities are
        # visible alongside the raw phase image. Names taken from the
        # cellpose/segmentation prediction zarr convention; we accept a few
        # spellings and skip silently if absent.
        def _first_present(*candidates):
            for c in candidates:
                if c in ch_names:
                    return data[ch_names.index(c)]
            return None

        nuclei_pred = _first_present(
            "nuclei_prediction", "nucleus_prediction", "nuclei_pred",
        )
        membrane_pred = _first_present(
            "membrane_prediction", "cell_membrane_prediction", "membrane_pred",
        )

        out[cluster_name] = {
            "crop": np.asarray(crop),
            "mask": np.asarray(mask, dtype=bool),
            "vmin": vmin,
            "vmax": vmax,
            "phase_crop": np.asarray(phase_crop) if phase_crop is not None else None,
            "phase_vmin": phase_vmin,
            "phase_vmax": phase_vmax,
            "nuclei_pred": np.asarray(nuclei_pred) if nuclei_pred is not None else None,
            "membrane_pred": np.asarray(membrane_pred) if membrane_pred is not None else None,
            "gene": gene,
            "channel": ch,
            "modality": mod,
        }
    return out


# Fluor reporter cmap — inferno (perceptually uniform, black → dark purple →
# red → orange → yellow) gives much more dynamic range for varied intensity
# than the linear black→green ramp.
_FLUOR_CMAP = "inferno"


_NUCLEI_OVERLAY_RGB = (0.20, 0.80, 1.00)   # cyan
_MEMBRANE_OVERLAY_RGB = (1.00, 0.30, 0.65)  # magenta-pink
_PRED_OVERLAY_ALPHA = 0.45                  # peak alpha for prediction overlays
# Nuclei prediction is *not* a sigmoid probability — it's a model logit-ish
# scale that runs ~0..60. Clipping to [0,1] saturates everywhere; fixed
# (0, 60) clims match what we use elsewhere for this channel.
_NUCLEI_PRED_CLIM = (0.0, 60.0)
_MEMBRANE_PRED_CLIM = (-1.0, 10.0)


def _draw_representative_cell(
    fig,
    cell_info: Dict,
    color: str,
    *,
    rect: Tuple[float, float, float, float] = (0.74, 0.06, 0.25, 0.52),
):
    """Render the cluster's reporter channel + Phase2D as a vertical stack on
    the right of the figure. Top panel = fluorescent reporter; bottom panel =
    Phase2D with nuclei / membrane prediction probabilities overlaid (cyan
    for nuclei, magenta for plasma membrane). Both panels share an inverse-
    mask blue overlay so the target cell pops against the dimmed background.
    Caption: per-panel channel name above each tile, gene KO label below.
    """
    from scipy.ndimage import binary_dilation

    left, bottom, w, h = rect
    gap = 0.022  # leaves room for per-panel caption between the two tiles
    half_h = (h - gap) / 2
    bottom_rect = (left, bottom, w, half_h)               # phase
    top_rect = (left, bottom + half_h + gap, w, half_h)   # fluor

    mask = cell_info["mask"]
    if mask is not None and mask.any():
        dilated = binary_dilation(mask, iterations=15)
        inv = ~dilated
        inverse_mask_overlay = np.zeros((*inv.shape, 4), dtype=np.float32)
        inverse_mask_overlay[..., 0] = 0.30
        inverse_mask_overlay[..., 1] = 0.40
        inverse_mask_overlay[..., 2] = 0.85
        inverse_mask_overlay[..., 3] = np.where(inv, 0.55, 0.0)
    else:
        inverse_mask_overlay = None

    def _pred_overlay(arr, rgb, clim):
        if arr is None:
            return None
        vmin, vmax = clim
        a = np.clip(
            (arr.astype(np.float32) - vmin) / max(vmax - vmin, 1e-9),
            0.0, 1.0,
        )
        ov = np.zeros((*a.shape, 4), dtype=np.float32)
        ov[..., 0], ov[..., 1], ov[..., 2] = rgb
        ov[..., 3] = a * _PRED_OVERLAY_ALPHA
        return ov

    def _panel(panel_rect, crop, vmin, vmax, caption, pred_overlays=(),
               show_mask=True, cmap="gray"):
        ax = fig.add_axes(panel_rect)
        if crop is None:
            ax.set_facecolor(_DARK_BG)
            ax.text(
                0.5, 0.5, "n/a",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=12, color=color,
            )
        else:
            ax.imshow(
                crop, cmap=cmap, vmin=vmin, vmax=vmax,
                interpolation="nearest", aspect="equal",
            )
            for ov in pred_overlays:
                if ov is not None:
                    ax.imshow(ov, interpolation="nearest", aspect="equal")
            if show_mask and inverse_mask_overlay is not None:
                ax.imshow(inverse_mask_overlay, interpolation="nearest", aspect="equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor(color)
            s.set_linewidth(1.6)
        # Per-panel caption above the image. Wrap long channel names so they
        # fit in the inset column (~15 chars per line, words preserved when
        # possible).
        wrapped = _wrap_label(caption, width=15)
        fig.text(
            panel_rect[0] + panel_rect[2] / 2,
            panel_rect[1] + panel_rect[3] + 0.004, wrapped,
            ha="center", va="bottom", multialignment="center",
            fontsize=8.5, fontweight="bold", color=color,
        )

    _panel(
        top_rect,
        cell_info["crop"],
        cell_info.get("vmin"),
        cell_info.get("vmax"),
        str(cell_info["channel"]).replace("_", " "),
        cmap=_FLUOR_CMAP,
    )
    # Phase2D shows the FOV in full (no inverse-mask dimming) so the nuclei +
    # membrane prediction overlays read against the raw phase context.
    _panel(
        bottom_rect,
        cell_info.get("phase_crop"),
        cell_info.get("phase_vmin"),
        cell_info.get("phase_vmax"),
        "Phase2D",
        pred_overlays=(
            _pred_overlay(
                cell_info.get("nuclei_pred"),
                _NUCLEI_OVERLAY_RGB, _NUCLEI_PRED_CLIM,
            ),
        ),
        show_mask=False,
    )

    # Shared gene-KO label below the bottom panel.
    fig.text(
        left + w / 2, bottom - 0.012, f"{cell_info['gene']} KO",
        ha="center", va="top",
        fontsize=10, fontweight="bold", color="#cfd6e0",
    )


def _draw_cell_placeholder(
    fig,
    color: str,
    *,
    rect: Tuple[float, float, float, float] = (0.74, 0.06, 0.25, 0.52),
):
    """Drawn when cell loading failed — keeps the layout stable."""
    ax = fig.add_axes(rect)
    ax.set_facecolor(_DARK_BG)
    ax.text(
        0.5, 0.5, "n/a",
        ha="center", va="center", transform=ax.transAxes,
        fontsize=14, color=color,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor(color)
        s.set_linewidth(1.0)


def _render_frame(
    coords: np.ndarray,
    pert_index: Dict[str, np.ndarray],
    is_ntc: np.ndarray,
    cluster_name: str,
    cluster_genes: List[str],
    cluster_idx: int,
    n_clusters: int,
    figsize: Tuple[float, float],
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    plt,
    cell_info: Optional[Dict] = None,
    cluster_match_idx_list: Optional[List[np.ndarray]] = None,
    cluster_color_list: Optional[List[str]] = None,
    passed_cluster: Optional[np.ndarray] = None,
    canvas_dpi: int = 180,
):
    """Render a single frame with FIXED axes/figure layout — every frame
    has identical xlim/ylim and identical subplots_adjust margins, so the
    UMAP itself never shifts as the title text changes.

    ``passed_cluster`` (length-N array, value = most-recent cluster index that
    claimed each point, -1 = never claimed) lets already-shown clusters keep
    their super-category color at normal size, so the embedding fills up over
    the course of the animation instead of reverting to gray.
    """
    if cluster_match_idx_list is not None:
        match_idx = cluster_match_idx_list[cluster_idx]
    else:
        matched_idx_lists = [pert_index[g] for g in cluster_genes if g in pert_index]
        if matched_idx_lists:
            match_idx = np.unique(np.concatenate(matched_idx_lists))
        else:
            match_idx = np.empty(0, dtype=np.int64)

    n_matched = int(match_idx.size)
    n_missing = len(cluster_genes) - sum(1 for g in cluster_genes if g in pert_index)

    super_cat = HAND_CLUSTER_TO_SUPER.get(cluster_name, "Protein Homeostasis")
    color = SUPERCATEGORY_COLORS.get(super_cat, "#1f77b4")

    # canvas_dpi controls the GIF pixel resolution (canvas pixels = figsize *
    # dpi). 180 dpi at 8" → 1440 px wide, giving the cell-image inset enough
    # pixels to render ~290 px wide tiles vs. the 306 px source crops.
    fig, ax = plt.subplots(figsize=figsize, dpi=canvas_dpi)
    fig.subplots_adjust(left=0.10, right=0.97, top=0.86, bottom=0.10)
    fig.patch.set_facecolor(_DARK_BG)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_facecolor(_DARK_BG)
    ax.grid(True, color=_GRID_COLOR, alpha=0.45, linewidth=0.6, zorder=0)
    ax.set_xlabel("UMAP 1", fontsize=11, color=_AXIS_FG)
    ax.set_ylabel("UMAP 2", fontsize=11, color=_AXIS_FG)
    ax.tick_params(
        axis="both", which="both", labelsize=9,
        color=_AXIS_FG, labelcolor=_AXIS_FG,
    )
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
        spine.set_edgecolor(_AXIS_FG)

    n_pts = coords.shape[0]
    current_mask_arr = np.zeros(n_pts, dtype=bool)
    current_mask_arr[match_idx] = True

    # "Passed" = points claimed by an earlier cluster (passed_cluster[i] >= 0).
    # We render them at small size in their cluster's super-cat color so the
    # embedding fills up over time instead of reverting to gray.
    if passed_cluster is not None and cluster_color_list is not None:
        is_passed = passed_cluster >= 0
    else:
        is_passed = np.zeros(n_pts, dtype=bool)

    # 1. Gray: not current, not ntc, not passed
    gray_mask = ~current_mask_arr & ~is_ntc & ~is_passed
    if gray_mask.any():
        ax.scatter(
            coords[gray_mask, 0], coords[gray_mask, 1],
            s=6, c=_BG_COLOR, alpha=0.40, linewidths=0, zorder=1,
        )

    # 2. Passed (excluding current): small dots, super-cat color of the
    # most-recent cluster that claimed each point. One scatter call per
    # cluster index so each color renders correctly.
    if is_passed.any() and cluster_color_list is not None:
        passed_excl = is_passed & ~current_mask_arr & ~is_ntc
        for c_idx in np.unique(passed_cluster[passed_excl]):
            sub_mask = (passed_cluster == c_idx) & passed_excl
            if not sub_mask.any():
                continue
            ax.scatter(
                coords[sub_mask, 0], coords[sub_mask, 1],
                s=14, c=cluster_color_list[int(c_idx)],
                alpha=0.95, linewidths=0, zorder=1.5,
            )

    # 3. NTC: red x markers
    if is_ntc.any():
        ax.scatter(
            coords[is_ntc, 0], coords[is_ntc, 1],
            s=14, c=_NTC_COLOR, marker="x", alpha=0.85, linewidths=0.9, zorder=2,
        )
        _centroid_label(
            ax, coords[is_ntc], "Controls", color=_NTC_COLOR,
            fontsize=13, fontweight="bold",
        )

    # 4. Current cluster: large + super-cat color + white edge
    if match_idx.size:
        ax.scatter(
            coords[match_idx, 0], coords[match_idx, 1],
            s=140, c=color, alpha=0.95,
            edgecolors="white", linewidths=1.0, zorder=3,
        )
        _centroid_label(
            ax, coords[match_idx], cluster_name, color=color,
            fontsize=16, fontweight="bold",
        )

    # The figure title shows the SUPER-CATEGORY (the colored bucket this
    # cluster belongs to). The specific cluster name is already overlaid at
    # its centroid on the UMAP — repeating it in the title is redundant.
    progress = f"[{cluster_idx + 1}/{n_clusters}]  {super_cat}"
    sub = f"{n_matched} gene{'s' if n_matched != 1 else ''} matched"
    if n_missing:
        sub += f"  ·  {n_missing} not in dataset"
    # Title is colored by super-category; positioned by suptitle so it sits
    # in the fixed top margin (subplots_adjust top=0.86) and never reshapes
    # the axes between frames.
    fig.text(
        0.535, 0.965, progress,
        ha="center", va="top",
        fontsize=14, fontweight="bold", color=color,
    )
    fig.text(
        0.535, 0.92, sub,
        ha="center", va="top",
        fontsize=10, color="#9aa0aa",
    )

    # Representative-cell inset (bottom-right) — top-attention crop for this
    # cluster's rep_gene in its rep_channel. Same color as title + highlighted
    # dots. Falls back to a placeholder if the cell failed to load.
    if cell_info is not None:
        _draw_representative_cell(fig, cell_info, color)
    else:
        _draw_cell_placeholder(fig, color)
    return fig


def make_animation(
    run_dir: Path,
    annotations_path: Path,
    out_path: Optional[Path] = None,
    frame_ms: int = 750,
    figsize: Tuple[float, float] = (8.0, 8.0),
    dpi: int = 110,
    save_static_panels: bool = False,
    fluor_csv: Optional[Path] = None,
    phase_csv: Optional[Path] = None,
    crop_size: int = 306,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import imageio.v2 as imageio

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    _logger = logging.getLogger(__name__)

    run_dir = Path(run_dir)
    gene_h5ad = run_dir / "gene_embedding_pca_optimized.h5ad"
    if not gene_h5ad.exists():
        raise SystemExit(f"ERROR: {gene_h5ad} does not exist")

    _logger.info("Loading %s", gene_h5ad)
    adata = ad.read_h5ad(gene_h5ad)
    if "X_umap" not in adata.obsm:
        raise SystemExit(
            f"ERROR: {gene_h5ad} has no obsm['X_umap'] — run --overlays-only first"
        )
    coords = np.asarray(adata.obsm["X_umap"], dtype=np.float32)

    perts = (
        adata.obs["perturbation"].astype(str).values
        if "perturbation" in adata.obs.columns
        else np.asarray(adata.obs_names.values, dtype=str)
    )
    is_ntc = np.array([p.startswith("NTC") for p in perts])
    pert_index: Dict[str, np.ndarray] = {}
    for i, p in enumerate(perts):
        pert_index.setdefault(p, []).append(i)
    pert_index = {k: np.asarray(v, dtype=np.int64) for k, v in pert_index.items()}

    clusters = parse_clusters(annotations_path)
    if not clusters:
        raise SystemExit(f"ERROR: parsed 0 clusters from {annotations_path}")
    _logger.info("Parsed %d clusters from %s", len(clusters), annotations_path)

    out_dir = run_dir / "plots" / "hand_annotated_animation"
    out_dir.mkdir(parents=True, exist_ok=True)
    if out_path is None:
        out_path = out_dir / "hand_annotated_clusters.gif"

    frames_dir = out_dir / "frames"
    if save_static_panels:
        frames_dir.mkdir(parents=True, exist_ok=True)

    # Pre-load representative cell crops up front (~30s for 48 clusters across
    # ~30 unique experiments). Done once so each frame just renders.
    fluor_csv_path = Path(fluor_csv) if fluor_csv else Path(DEFAULT_FLUOR_CSV)
    phase_csv_path = Path(phase_csv) if phase_csv else Path(DEFAULT_PHASE_CSV)
    _logger.info(
        "Loading representative cells (fluor=%s, phase=%s)",
        fluor_csv_path.name, phase_csv_path.name,
    )
    cell_info_by_cluster = _load_representative_cells(
        clusters, fluor_csv_path, phase_csv_path, crop_size=crop_size,
    )
    _logger.info(
        "Loaded %d/%d representative cells",
        len(cell_info_by_cluster), len(clusters),
    )

    # Fixed axes limits — shared across every frame so the UMAP never shifts.
    xpad = (coords[:, 0].max() - coords[:, 0].min()) * 0.04
    ypad = (coords[:, 1].max() - coords[:, 1].min()) * 0.04
    xlim = (float(coords[:, 0].min() - xpad), float(coords[:, 0].max() + xpad))
    ylim = (float(coords[:, 1].min() - ypad), float(coords[:, 1].max() + ypad))

    # Precompute per-cluster point indices + super-cat colors. Used by the
    # render loop to (a) avoid recomputing match_idx every frame, and (b)
    # render already-shown clusters in their persistent color.
    cluster_match_idx_list: List[np.ndarray] = []
    cluster_color_list: List[str] = []
    for cluster in clusters:
        matched_idx_lists = [
            pert_index[g] for g in cluster["genes"] if g in pert_index
        ]
        if matched_idx_lists:
            cmi = np.unique(np.concatenate(matched_idx_lists))
        else:
            cmi = np.empty(0, dtype=np.int64)
        cluster_match_idx_list.append(cmi)
        super_cat = HAND_CLUSTER_TO_SUPER.get(
            cluster["name"], "Protein Homeostasis",
        )
        cluster_color_list.append(SUPERCATEGORY_COLORS.get(super_cat, "#1f77b4"))

    # State array: passed_cluster[i] = most-recent cluster index that claimed
    # point i, -1 if never claimed. Updated AFTER each frame renders, so
    # frame k shows clusters 0..k-1 as passed (small + colored) and cluster k
    # as current (large + colored).
    passed_cluster = np.full(coords.shape[0], -1, dtype=np.int32)

    images = []
    for k, cluster in enumerate(clusters):
        name = cluster["name"]
        genes = cluster["genes"]
        cell_info = cell_info_by_cluster.get(name)
        fig = _render_frame(
            coords, pert_index, is_ntc, name, genes, k, len(clusters),
            figsize, xlim, ylim, plt, cell_info=cell_info,
            cluster_match_idx_list=cluster_match_idx_list,
            cluster_color_list=cluster_color_list,
            passed_cluster=passed_cluster,
        )
        fig.canvas.draw()
        # Read pixels straight from the canvas — avoids a temp file roundtrip.
        rgba = np.asarray(fig.canvas.buffer_rgba())
        rgb = rgba[..., :3].copy()
        images.append(rgb)
        if save_static_panels:
            fig.savefig(
                frames_dir / f"{k:03d}_{_slug(name)}.png",
                dpi=dpi, bbox_inches="tight",
            )
        plt.close(fig)

        # Mark this cluster's points as "passed" — most-recent-wins, so a
        # gene shared across two clusters takes its color from the cluster
        # that just rendered.
        passed_cluster[cluster_match_idx_list[k]] = k

        n_matched = int(cluster_match_idx_list[k].size)
        _logger.info(
            "  [%d/%d] %s — %d/%d genes matched",
            k + 1, len(clusters), name, n_matched, len(genes),
        )

    # imageio's PIL/GIF backend takes ``duration`` in MILLISECONDS per frame
    # (not seconds — the docs are confusing). Passing 0.5 here would render
    # a half-millisecond frame, hence the "way too fast" complaint.
    imageio.mimsave(
        out_path, images, format="GIF", duration=int(frame_ms), loop=0,
    )
    _logger.info("Wrote %s (%d frames, %d ms each)", out_path, len(images), frame_ms)
    return out_path


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Walk through hand-annotated gene clusters on the gene UMAP and "
            "save an animated GIF — one frame per cluster, in file order."
        ),
    )
    p.add_argument(
        "--run-dir", type=str, default=DEFAULT_RUN_DIR,
        help=f"Run directory containing gene_embedding_pca_optimized.h5ad. "
             f"Default: {DEFAULT_RUN_DIR}",
    )
    p.add_argument(
        "--annotations", type=str, default=DEFAULT_ANNOTATIONS,
        help=f"Path to the annotation text file. Default: {DEFAULT_ANNOTATIONS}",
    )
    p.add_argument(
        "--out", type=str, default=None,
        help="Output GIF path. Default: <run_dir>/plots/hand_annotated_animation/"
             "hand_annotated_clusters.gif",
    )
    p.add_argument(
        "--frame-ms", type=int, default=750,
        help="Per-frame duration in milliseconds (default: 750 = 0.75 s).",
    )
    p.add_argument(
        "--figsize", type=str, default="8,8",
        help="Figure size in inches as W,H (default: 8,8)",
    )
    p.add_argument(
        "--save-frames", action="store_true",
        help="Also save each frame as a numbered PNG alongside the GIF.",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()
    w, h = (float(v) for v in args.figsize.split(","))
    make_animation(
        run_dir=Path(args.run_dir),
        annotations_path=Path(args.annotations),
        out_path=Path(args.out) if args.out else None,
        frame_ms=args.frame_ms,
        figsize=(w, h),
        save_static_panels=args.save_frames,
    )


if __name__ == "__main__":
    main()
