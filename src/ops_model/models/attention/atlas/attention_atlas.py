"""
Generate a multi-page landscape PDF atlas of top-attention cells per gene KO.

For each gene (alphabetical):
  Row 1      : top 10 phase cells (Phase2D), ranked by pma_attention.
  Rows 2..4  : top 10 cells in each of the 3 fluorescent viz_channels with the
               highest pma_attention for that gene.

The fluor CSV's `viz_channel` is a biological marker label (e.g.
`autophagosome_MAP1LC3B`). The zarr stores channels by fluorophore (e.g. `GFP`).
We resolve them per-experiment via `OpsDataset.channel_map_data`
(e.g. ops0054: `{'GFP': 'autophagosome, MAP1LC3B'}`), normalizing ", " -> "_".

Modes:
  Local:  python attention_atlas.py <args>
  SLURM:  python attention_atlas.py --submit <args>
          Submits an array of per-chunk render jobs (default 1 gene/task),
          waits, then submits a dependent merge job that stitches per-gene
          PDFs into --output. Uses ops_utils.hpc.slurm_batch_utils for
          submission + live job tracking.

Example:
  python ops_analysis/napari/attention_atlas.py --submit \\
    --phase-csv /home/gav.sturm/linked_folders/icd.fast.ops/models/alex_lin_attention/pma_top_phase_cells_v2.csv \\
    --fluor-csv /home/gav.sturm/linked_folders/icd.fast.ops/models/alex_lin_attention/pma_top_fluorescent_cells_v1.csv \\
    --output /hpc/mydata/gav.sturm/attention_atlas.pdf
"""

import argparse
import multiprocessing as mp
import os
import pickle
import re
import shutil
import subprocess
import sys
import tempfile
import time
import warnings
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, os.getcwd())

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, VPacker, TextArea
from matplotlib.offsetbox import AnchoredText
import numpy as np
import pandas as pd
from iohub import open_ome_zarr
from scipy.ndimage import binary_dilation, gaussian_filter
from skimage.measure import find_contours


# Qualitative palette for unique-per-key title highlights. Varied dark colors:
# hues evenly spaced via the golden-ratio conjugate so neighboring palette
# indices land far apart in hue, combined with 4 rotating (S, V) variants for
# extra visual diversity. All values are dark enough to stay readable on a
# white background at ~5 pt font.
def _make_dark_palette(n=80):
    import colorsys
    golden = 0.61803398875
    sv_variants = (
        (0.95, 0.40),
        (0.75, 0.55),
        (0.88, 0.30),
        (0.85, 0.50),
    )
    palette = []
    for i in range(n):
        h = (i * golden) % 1.0
        s, v = sv_variants[i % len(sv_variants)]
        palette.append(colorsys.hsv_to_rgb(h, s, v))
    return palette


_QUAL_PALETTE = _make_dark_palette(80)


# Dedicated vivid palette for wells. Deterministic LINEAR mapping (row*12 +
# col) ensures A1/A2/A3/... never collide regardless of experiment. 16 slots
# covers any realistic subset of a plate (A1-A12 + first rows of B/C/...).
_WELL_PALETTE = [
    "#C81E1E",  # red
    "#1F46A6",  # blue
    "#158939",  # green
    "#6E23B8",  # purple
    "#E06B00",  # orange
    "#0F8891",  # teal
    "#7A4A22",  # brown
    "#B5268A",  # magenta
    "#4A5A00",  # olive
    "#A03030",  # dark red
    "#2F6F2F",  # dark green
    "#005E99",  # navy
    "#9A3A8A",  # plum
    "#FF4500",  # vermilion
    "#6B004F",  # maroon
    "#336B87",  # slate
]


def _normalize_well_tag(w):
    if w is None:
        return ""
    return str(w).strip().upper().replace("/", "").replace(" ", "")


def _hash_color(key, fallback="black"):
    if not key:
        return fallback
    h = zlib.crc32(str(key).encode("utf-8")) & 0xFFFFFFFF
    return _QUAL_PALETTE[h % len(_QUAL_PALETTE)]


def _well_color(well, fallback="black"):
    """Deterministic, collision-free well color. A1→slot 0, A2→slot 1, ...,
    B1→slot 12, etc. Uses the well-dedicated palette."""
    w = _normalize_well_tag(well)
    if not w:
        return fallback
    m = re.match(r"^([A-Z]+)(\d+)$", w)
    if m is None:
        # Non-standard well name — fall back to crc32 hash so it still has some color.
        h = zlib.crc32(w.encode("utf-8")) & 0xFFFFFFFF
        return _WELL_PALETTE[h % len(_WELL_PALETTE)]
    row_char, col_str = m.group(1), m.group(2)
    row_idx = ord(row_char[0]) - ord("A")  # A=0, B=1, ...
    col_idx = int(col_str) - 1              # 1-indexed -> 0-indexed
    idx = row_idx * 12 + col_idx
    return _WELL_PALETTE[idx % len(_WELL_PALETTE)]


def _tile_tag(x, y, tile_size=100):
    """Short xy-region label. (857, 412) @ tile=100 -> 'T8,4'."""
    try:
        return f"T{int(x) // tile_size},{int(y) // tile_size}"
    except (ValueError, TypeError):
        return ""


def _set_structured_title(ax, line1, line2, exp, well, x, y, fontsize=5):
    """Render a 3-line title above `ax`:
        line1, line2   (black)
        <exp>  <well>  <tile>   (each in its own hash-derived color)
    """
    tp = dict(fontsize=fontsize, color="black")

    def ta(text, color=None):
        p = dict(tp)
        if color is not None:
            p["color"] = color
        return TextArea(str(text) if text is not None else "", textprops=p)

    tile = _tile_tag(x, y)
    parts = []
    if exp:
        parts.append(ta(exp, _hash_color(exp)))
    if well:
        parts.append(ta(well, _well_color(well)))
    if tile:
        parts.append(ta(tile, _hash_color(tile)))
    interleaved = []
    for i, p in enumerate(parts):
        if i > 0:
            interleaved.append(ta("  "))
        interleaved.append(p)
    if not interleaved:
        interleaved = [ta("")]
    line3 = HPacker(children=interleaved, pad=0, sep=0, align="center")
    vp = VPacker(children=[ta(line1), ta(line2), line3],
                 pad=0, sep=1, align="center")
    anchored = AnchoredOffsetbox(
        loc="lower center", child=vp, frameon=False,
        bbox_to_anchor=(0.5, 1.0), bbox_transform=ax.transAxes,
        pad=0, borderpad=0,
    )
    ax.add_artist(anchored)

from ops_utils.data.bbox_utils import BaseDataset
from ops_utils.data.experiment import OpsDataset
from ops_utils.data.filesystem import resolve_experiment_name


DEFAULT_SUPERCATEGORY_CONFIG = Path(
    "/hpc/mydata/gav.sturm/ops_mono/organelle_profiler/configs/gene_supercategory_mapping.yaml"
)


# ──────────────────────────────────────────────────────────────────────────
# Layout constants
# ──────────────────────────────────────────────────────────────────────────
# 4 rows: phase KO, phase NTC, fluor KO (top attn, any channel), fluor NTC
# (random, rendered in the most-common viz_channel from the fluor KO row).
N_COLS = 10
N_FLUOR_CHANNELS_PER_GENE = 3   # Alex v2: every gene has exactly 3 viz_channels
N_TOTAL_ROWS = 2 + 2 * N_FLUOR_CHANNELS_PER_GENE   # 8 rows
ROW_PHASE_KO = 0
ROW_PHASE_NTC = 1
# Rows 2..7 alternate KO / NTC for each of the 3 fluor channels
def _fluor_ko_row(i):  # i = 0..2
    return 2 + 2 * i
def _fluor_ntc_row(i):
    return 3 + 2 * i
PHASE_CHANNEL_CANDIDATES = ("Phase2D", "Phase")
PHASE_CONTRAST = (-0.45, 0.60)  # fixed for all phase tiles
# Fluor per-viz_channel clims: for each tile of that reporter, take the
# (low, high) intra-tile percentile, then take the MEDIAN across tiles.
# Median-of-maxes is robust to one outlier-bright cell and keeps most tiles
# visible while preserving relative dim/bright appearance across tiles.
FLUOR_TILE_PERCENTILES = (1.0, 99.9)
# Outlier-clim override: if a single tile's top percentile is way above or
# below the reporter's median, render it with its own (vmin, vmax) and flag
# it in red so the viewer can tell it escaped the shared scale.
FLUOR_OUTLIER_HI_RATIO = 2.0   # tile_hi > this * median_hi -> outlier-bright
FLUOR_OUTLIER_LO_RATIO = 0.4   # tile_hi < this * median_hi -> outlier-dim
# NTC sanity check: oversample candidates, then keep only ones whose cell
# is actually there (mask has pixels) and image has structure.
NTC_OVERSAMPLE_FACTOR = 5
NTC_MIN_MASK_PIXELS = 25
NTC_MIN_INTENSITY_STD = 1e-3
MASK_DILATION = 15


# ──────────────────────────────────────────────────────────────────────────
# Store cache (per-process) + channel_map resolution
# ──────────────────────────────────────────────────────────────────────────
class StoreCache:
    """Lazy cache of pheno_assembled_v3 stores + channel_map keyed by experiment."""

    def __init__(self):
        self.stores = {}
        self.channel_maps = {}

    def get(self, experiment):
        if experiment in self.stores:
            return self.stores[experiment]
        try:
            resolved = resolve_experiment_name(experiment)
            ds = OpsDataset(resolved)
            path = ds.store_paths["pheno_assembled_v3"]
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", module="zarr")
                store = open_ome_zarr(str(path), mode="r")
            self.stores[experiment] = store
            # Parse ops_channel_maps.yaml directly — OpsDataset.channel_map_data
            # drops all CP / 4i channels.
            self.channel_maps[experiment] = _channel_map_for_experiment(experiment)
            return store
        except Exception as e:
            print(f"  [store] failed {experiment}: {e}", flush=True)
            self.stores[experiment] = None
            self.channel_maps[experiment] = {}
            return None

    def channel_map(self, experiment):
        if experiment not in self.channel_maps:
            self.get(experiment)
        return self.channel_maps.get(experiment, {})


def _normalize_well(well_str):
    w = str(well_str).strip()
    if w.count("/") == 2:
        return w
    if w.count("/") == 1:
        return f"{w}/0"
    m = re.match(r"^([A-Za-z]+)(\d+)$", w)
    if not m:
        raise ValueError(f"Unknown well format: {well_str!r}")
    return f"{m.group(1)}/{m.group(2)}/0"


# Long-form marker names used by Alex's attention CSV versus the abbreviated
# forms used by ops_channel_maps.yaml. Apply as substring substitutions on
# normalized text so the two namings always compare equal.
_MARKER_ALIASES = (
    ("wheat germ agglutinin", "wga"),
    ("concanavalin a", "cona"),
    ("endoplasmic reticulum", "er"),
    ("nucleus", "nuclei"),
)


def _norm_channel(name):
    """Normalize a biological channel name for cross-source matching.

    Bridges the pma CSV's `p21_p21 (rabbit-647)` / `gH2AX_gH2AX (rabbit-647)`
    / `Mitochondria_TOMM20` style with the channel_map yaml's bare-name
    style (`p21`, `gH2AX`, `TOMM20`). Without the trailing-paren strip
    + duplicate-token collapse, CP/4i fluor rows fail
    `_experiment_images_viz_channel` and get dropped at load time even
    though the CP/4i seg ID and tile data are perfectly valid — so the
    atlas renders blank cells for those channels.

    'autophagosome, MAP1LC3B'               -> 'autophagosome_map1lc3b'
    'Plasma Membrane_Wheat Germ Agglutinin' -> 'plasma membrane_wga'   (alias)
    'Endoplasmic Reticulum_Concanavalin A'  -> 'er_cona'               (alias)
    'p21_p21 (rabbit-647)'                  -> 'p21'                   (CP/4i)
    'Mitochondria_TOMM20'                   -> 'mitochondria_tomm20'   (no-op duplicate)
    """
    if name is None:
        return ""
    n = str(name).replace(", ", "_").replace(",", "_").strip().lower()
    # Strip every trailing parenthetical: csv " (rabbit-647)" / " (mouse-488)"
    # antibody descriptors AND any matrix-style " (cp)" / " (4i)" tags.
    while n.endswith(")"):
        i = n.rfind(" (")
        if i == -1:
            break
        n = n[:i].rstrip()
    # Collapse duplicate-token format: `p21_p21` → `p21`,
    # `b-catenin_b-catenin` → `b-catenin`. The pma CSV mirrors the
    # marker into both the pre-_ position and the probe-name suffix
    # post-_; the channel_map yaml only lists the bare marker.
    if "_" in n:
        head, tail = n.split("_", 1)
        if head and head == tail:
            n = head
    for long_form, short_form in _MARKER_ALIASES:
        n = n.replace(long_form, short_form)
    return n


def _resolve_phase_channel(store):
    if store is None:
        return None
    names = list(store.channel_names)
    for cand in PHASE_CHANNEL_CANDIDATES:
        if cand in names:
            return cand
    return None


def _resolve_fluor_channel(store, channel_map, viz_channel):
    """Resolve CSV viz_channel (marker name) -> zarr channel name (fluorophore).

    Tries: (1) direct match, (2) channel_map lookup with normalization.
    Returns None if no match.
    """
    if store is None:
        return None
    names = list(store.channel_names)
    if viz_channel in names:
        return viz_channel
    viz_norm = _norm_channel(viz_channel)
    for zarr_ch, bio_name in (channel_map or {}).items():
        if bio_name and _norm_channel(bio_name) == viz_norm and zarr_ch in names:
            return zarr_ch
    return None


def _ntc_linked_paths(ds, well, channel_map):
    """Return (linked_csv_path, seg_col, fluor_seg_col, y_col, x_col) for this
    experiment+well. CP / 4i experiments carry BOTH a live-cell `segmentation_id`
    (for phase tile / cell_seg mask) AND a fluor_seg_col (`cp_cell_seg_id` or
    `4i_segmentation_id`) for the CP/4i fluor tiles. Live-cell experiments
    just have one seg column.

    Path layout (3-assembly/, current as of 2026-05-09):
      live-cell : <W><col>_linked_pheno_iss.csv      (y_pheno / x_pheno)
      CP        : <W><col>_linked_pheno_iss_cp.csv   (y_pheno_centroid / x_pheno_centroid)
      4i        : <W><col>_linked_pheno_iss_4i.csv   (y_pheno_centroid / x_pheno_centroid)
    Older naming (`cell_painting_linked_<w>_0.csv` / `four_i_linked_<w>_0.csv`)
    was a stale path — silently missed every CP/4i file, leaving NTC pools
    without CP/4i rows and atlas pages with NaN tiles for those rows.
    """
    keys = list(channel_map.keys()) if channel_map else []
    is_cp = any(str(k).startswith(("CP1_", "CP2_")) for k in keys)
    is_4i = any(str(k).startswith("4i_") for k in keys)
    base = ds.append_well("linked_results", well)  # <fast>/<W><col>_linked_pheno_iss.csv
    if is_cp:
        return (base.with_name(base.stem + "_cp" + base.suffix),
                "segmentation_id", "cp_cell_seg_id",
                "y_pheno_centroid", "x_pheno_centroid")
    if is_4i:
        return (base.with_name(base.stem + "_4i" + base.suffix),
                "segmentation_id", "4i_segmentation_id",
                "y_pheno_centroid", "x_pheno_centroid")
    return (base, "segmentation_id", None, "y_pheno", "x_pheno")


def _load_ntc_pool_from_pma(phase_csv: Path | None,
                             fluor_csv: Path | None) -> pd.DataFrame:
    """Build the NTC image-tile pool from the PMA top-attention NTC CSVs
    (`pma_top_*_cells_chad_ntc_v3.csv`) instead of per-experiment
    `linked_results.csv`.

    Use case: when the model has emitted attention rankings for NTC cells
    (high-attention NTC cohort), prefer those as the on-page NTC strip
    so the violin's BG reference and the image rows draw from the SAME
    cohort. Falls back to an empty DataFrame if neither path exists; the
    caller then routes to the legacy `_build_ntc_pool` linked_results
    sampler.

    Output schema matches `_build_ntc_pool`:
      [experiment, well, segmentation, fluor_segmentation, x_pheno,
       y_pheno, sgRNA, barcode, gene_label]
    """
    frames = []
    for path in (phase_csv, fluor_csv):
        if path is None or not Path(path).exists():
            continue
        df = pd.read_csv(path)
        # Schema sanity: must have the columns we need to seed the tile.
        need = {"experiment", "well", "segmentation", "x_pheno", "y_pheno"}
        missing = need - set(df.columns)
        if missing:
            print(f"  [ntc-pma] {Path(path).name} missing columns "
                  f"{missing} — skipping", flush=True)
            continue
        df = df.copy()
        # PMA wells are "A1", "A2"; downstream expects "A1" too (no
        # slashes), so no normalization needed here.
        # `fluor_segmentation` mirrors `segmentation` because the PMA
        # NTC CSV's `segmentation` column already carries the imaging-
        # modality-specific seg-ID for whatever channel that row
        # represents: live-cell seg for phase / live-fluor rows, 4i
        # seg for 4i fluor rows, CP seg for CP fluor rows. Downstream
        # the atlas's per-row code overwrites `segmentation` with
        # `fluor_segmentation` for is_fixed_mod (CP/4i) channels — so
        # we keep the two equal here so that overwrite is a no-op for
        # live channels and a same-value reset for fixed channels
        # (the right seg ID was already in `segmentation`).
        df["fluor_segmentation"] = df["segmentation"]
        # Synthetic identity columns (legacy schema). sgRNA/barcode
        # come from the source CSVs only for control-guide bookkeeping;
        # the renderer just shows them on the per-tile title block.
        if "sgRNA" not in df.columns:
            df["sgRNA"] = ""
        if "barcode" not in df.columns:
            df["barcode"] = df["segmentation"].astype(str)
        df["gene_label"] = "NTC"
        # Preserve `channel` / `rank` / `pma_attention` so downstream
        # per-channel top-N selection (`_sample_ntc_rows`) can sort by
        # attention rank instead of random-sampling the pool.
        if "channel" not in df.columns:
            df["channel"] = "Phase2D"
        if "rank" not in df.columns:
            df["rank"] = np.iinfo(np.int64).max
        if "pma_attention" not in df.columns:
            df["pma_attention"] = 0.0
        # Preserve `rank_type` (top|bottom|ntc_ko_typical) so
        # `_sample_ntc_rows` can restrict to high-attention NTCs (PMA)
        # or KO-resembling NTCs (picker output). Bottom-rank PMA NTCs
        # are uninformative as "what an unperturbed cell looks like".
        if "rank_type" not in df.columns:
            df["rank_type"] = "top"
        # `target_gene` is written by ntc_pick_cells.py — keeps each
        # picker NTC row grouped with the target KO page. PMA pools
        # don't have it; downstream `_sample_ntc_rows` detects column
        # presence to pick the right code path.
        if "target_gene" not in df.columns:
            df["target_gene"] = ""
        # `viz_channel` is the picker's per-row channel label (the
        # `channel` column carries `"Phase"` / `"Phase2D"` / fluor
        # marker names depending on source; viz_channel is normalized).
        if "viz_channel" not in df.columns:
            df["viz_channel"] = df["channel"].astype(str)
        frames.append(df[[
            "experiment", "well", "segmentation", "fluor_segmentation",
            "x_pheno", "y_pheno", "sgRNA", "barcode", "gene_label",
            "channel", "viz_channel", "rank", "rank_type",
            "pma_attention", "target_gene",
        ]])
    if not frames:
        return pd.DataFrame()
    pool = pd.concat(frames, ignore_index=True)
    pool = pool.dropna(subset=["x_pheno", "y_pheno"])
    seg_ok = pd.to_numeric(pool["segmentation"], errors="coerce") > 0
    pool = pool[seg_ok].reset_index(drop=True)
    # Dedup INCLUDES channel so the same physical cell can appear in
    # multiple channel rows (each channel's top-N selection sees its
    # own ranked set). Picker output also keys per `target_gene` (the
    # same physical cell can be a "KO-typical NTC" for >1 gene), so
    # include that in the subset when present — without it, the first
    # gene to surface a cell would steal it from all other genes.
    dedup_keys = ["experiment", "well", "segmentation", "channel"]
    if "target_gene" in pool.columns:
        dedup_keys.append("target_gene")
    pool = pool.drop_duplicates(
        subset=dedup_keys, keep="first",
    ).reset_index(drop=True)
    return pool


def _build_ntc_pool(experiments, max_per_exp=500, wells=("A/1", "A/2", "A/3")):
    """Collect NTC rows across experiments. CP/4i NTCs carry both seg IDs so
    phase NTC rows can use the live-cell `segmentation` (matching cell_seg)
    while CP/4i fluor NTC rows can swap to the CP/4i-specific seg ID.
    """
    pieces = []
    for exp in experiments:
        try:
            ds = OpsDataset(resolve_experiment_name(exp))
        except Exception as e:
            print(f"  [ntc] resolve {exp}: {e}", flush=True)
            continue
        cmap = _channel_map_for_experiment(exp)
        for w in wells:
            try:
                path, seg_col, fluor_seg_col, y_col, x_col = _ntc_linked_paths(ds, w, cmap)
                if not path.exists():
                    continue
                cols = ["gene_name", "sgRNA", "barcode", seg_col, y_col, x_col]
                if fluor_seg_col:
                    cols.append(fluor_seg_col)
                df = pd.read_csv(path, usecols=cols)
            except Exception as e:
                print(f"  [ntc] read {exp} {w}: {e}", flush=True)
                continue
            # NTC encoding: NaN (live/4i) OR literal "NTC" (CP)
            gn = df["gene_name"]
            ntc = df[gn.isna() | (gn.astype(str).str.upper() == "NTC")].copy()
            if ntc.empty:
                continue
            if len(ntc) > max_per_exp:
                ntc = ntc.sample(n=max_per_exp, random_state=42)
            rename = {seg_col: "segmentation_id", y_col: "y_pheno", x_col: "x_pheno"}
            if fluor_seg_col:
                rename[fluor_seg_col] = "fluor_segmentation"
            ntc = ntc.rename(columns=rename)
            if "fluor_segmentation" not in ntc.columns:
                ntc["fluor_segmentation"] = pd.NA
            ntc["experiment"] = exp
            ntc["well"] = w.replace("/", "")
            ntc["gene_label"] = "NTC_" + ntc["barcode"].astype(str)
            pieces.append(
                ntc[[
                    "experiment", "well", "segmentation_id", "fluor_segmentation",
                    "x_pheno", "y_pheno", "sgRNA", "barcode", "gene_label",
                ]]
            )
    if not pieces:
        return pd.DataFrame()
    pool = pd.concat(pieces, ignore_index=True)
    pool = pool.rename(columns={"segmentation_id": "segmentation"})
    pool = pool.dropna(subset=["x_pheno", "y_pheno"])
    # Keep cells where AT LEAST ONE seg-id (live or CP/4i) is valid. The
    # per-NTC-row code at task-build time applies the correct seg filter
    # for that row's target mask (cell_seg / cp_cell_seg / 4i_cell_seg).
    seg_live_ok = pd.to_numeric(pool["segmentation"], errors="coerce") > 0
    seg_fluor_ok = pd.to_numeric(pool["fluor_segmentation"], errors="coerce") > 0
    pool = pool[seg_live_ok | seg_fluor_ok]
    pool = pool.reset_index(drop=True)
    # Skip the O(N) xy-bin pass on the 40K+ full pool — random sampling
    # won't systematically hit cells that share only an xy bin. The smaller
    # per-gene display subsets (<=100 rows) still do the full xy-tol dedup.
    pool = _dedup_cells(pool, xy_tol=0)
    return pool


def _pick_ntc_experiments(fluor_df, phase_df, channel_map_cache,
                          target_total, per_channel=3):
    """Pick NTC-pool experiments ensuring each unique viz_channel in the fluor
    CSV is represented by at least `per_channel` experiments (if that many
    exist in channel_map_cache), then fill to `target_total` with random
    extras from phase CSV experiments for row-2 diversity.
    """
    picked = set()
    viz_channels = sorted(fluor_df["viz_channel"].dropna().unique())
    for ch in viz_channels:
        eligible = sorted([
            e for e, cm in channel_map_cache.items()
            if _cm_has_viz_channel(cm, ch)
        ])
        if not eligible:
            continue
        rng = np.random.default_rng(zlib.crc32(ch.encode("utf-8")) & 0xFFFFFFFF)
        k = min(per_channel, len(eligible))
        for e in rng.choice(eligible, size=k, replace=False):
            picked.add(str(e))

    # Fill up with random extras (prioritizing phase CSV experiments for variety)
    if len(picked) < target_total:
        candidates = sorted(
            set(phase_df["experiment"].dropna().unique()) - picked
        )
        if candidates:
            rng = np.random.default_rng(42)
            n_extra = min(target_total - len(picked), len(candidates))
            for e in rng.choice(candidates, size=n_extra, replace=False):
                picked.add(str(e))
    return sorted(picked)


# Persistent on-disk cache for the mygene.info fallback. Negative
# results (gene not found / network error) are cached as `None` so we
# don't keep retrying every render. Lives in the user's HOME so it
# survives across runs and isn't tied to a specific working dir.
_LONGNAME_CACHE_PATH = (
    Path.home() / ".cache" / "ops_atlas" / "gene_longname_cache.json"
)


def _load_longname_cache():
    """Load the persistent cache of {gene_symbol: longname-or-None}
    populated by `_fetch_gene_longname_via_mygene`. Returns {} on
    missing/corrupt cache."""
    import json
    p = _LONGNAME_CACHE_PATH
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def _save_longname_cache(cache):
    import json
    p = _LONGNAME_CACHE_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        p.write_text(json.dumps(cache, indent=0, sort_keys=True))
    except Exception as e:
        print(f"  [longname-cache] save failed: {e}", flush=True)


def _fetch_gene_longname_via_mygene(gene_symbol, timeout=5.0):
    """Look up a gene's `name` + `summary` from mygene.info (NCBI Gene
    aggregator) via plain urllib. Returns "{name}. {summary}" on hit,
    or None on miss / network error. No extra deps.

    Used as a fallback for genes missing from the local longnames
    file. Cached results live in `_LONGNAME_CACHE_PATH`.
    """
    import json
    import urllib.parse
    import urllib.request

    if not gene_symbol or not isinstance(gene_symbol, str):
        return None
    url = (
        "https://mygene.info/v3/query?"
        + urllib.parse.urlencode({
            "q": f"symbol:{gene_symbol}",
            "species": "human",
            "fields": "name,summary",
            "size": 1,
        })
    )
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            data = json.loads(resp.read())
    except Exception:
        return None
    hits = (data or {}).get("hits") or []
    if not hits:
        return None
    h = hits[0]
    name = (h.get("name") or "").strip()
    summary = (h.get("summary") or "").strip()
    if not name and not summary:
        return None
    if name and summary:
        return f"{name}. {summary}"
    return name or summary


def _enrich_gene_longnames(gene_to_longname, all_genes):
    """Fill in missing longnames via mygene.info lookup, with on-disk
    caching of both hits and misses. Mutates `gene_to_longname`
    in-place (and returns it). Skips entirely if `all_genes` is empty
    or if every gene already has a longname.

    Cache semantics: a `None` value in the cache means "we already
    looked this up and got nothing" — avoids re-hitting the API on
    every atlas run. Delete `_LONGNAME_CACHE_PATH` to force re-fetch.
    """
    missing = [g for g in all_genes
               if g and not gene_to_longname.get(g)]
    if not missing:
        return gene_to_longname

    cache = _load_longname_cache()
    api_targets, n_hits, n_already_cached_miss = [], 0, 0
    for g in missing:
        if g in cache:
            v = cache[g]
            if v:
                gene_to_longname[g] = v
                n_hits += 1
            else:
                n_already_cached_miss += 1
        else:
            api_targets.append(g)

    print(
        f"  [longname-fallback] {len(missing)} missing — "
        f"{n_hits} from cache, {n_already_cached_miss} cached-miss, "
        f"{len(api_targets)} to fetch via mygene.info",
        flush=True,
    )
    if not api_targets:
        return gene_to_longname

    n_filled = 0
    for g in api_targets:
        ln = _fetch_gene_longname_via_mygene(g)
        cache[g] = ln  # store both hits and misses so we don't retry
        if ln:
            gene_to_longname[g] = ln
            n_filled += 1
    _save_longname_cache(cache)
    print(f"  [longname-fallback] filled {n_filled}/{len(api_targets)} "
          f"via mygene.info; cache → {_LONGNAME_CACHE_PATH}", flush=True)
    return gene_to_longname


def _load_gene_longnames():
    """Return {gene_name: function_description} from the annotated gene panel.

    Richer than just `LongName` — combines LongName (the short Entrez/UniProt
    title like 'KRAS proto-oncogene, GTPase') with the top GO biological-process
    terms for a fuller functional description suitable for a subtitle line.
    """
    import ast
    try:
        from ops_utils.analysis.gene_supercategories import DEFAULT_GENE_PANEL_PATH
        df = pd.read_csv(
            DEFAULT_GENE_PANEL_PATH,
            usecols=["Gene.name", "LongName", "In_go_pathways", "In_REACT_pathways"],
        )
    except Exception as e:
        print(f"  [gene longnames] load failed: {e}", flush=True)
        return {}

    def _parse_list(val):
        if not isinstance(val, str) or not val.strip():
            return []
        try:
            parsed = ast.literal_eval(val)
            return [str(x).strip() for x in parsed if str(x).strip()] if isinstance(parsed, list) else []
        except Exception:
            return []

    result = {}
    for _, r in df.iterrows():
        g = r.get("Gene.name")
        if not isinstance(g, str) or not g.strip():
            continue
        g = g.strip()
        parts = []
        ln = r.get("LongName")
        if isinstance(ln, str) and ln.strip():
            parts.append(ln.strip())
        go_terms = _parse_list(r.get("In_go_pathways"))
        if go_terms:
            parts.append("GO: " + "; ".join(go_terms[:4]))
        rx_terms = _parse_list(r.get("In_REACT_pathways"))
        if rx_terms:
            parts.append("Reactome: " + "; ".join(rx_terms[:3]))
        result[g] = ". ".join(parts)
    return result


def _load_one_eval_csv(eval_csv: Path, n_cells: int = 100) -> dict:
    """Load a single eval CSV (one modality) and return
    {key: {'top1_acc': float, 'top5_acc': float, 'n_samples': int}}.
    `key` is `class_name` (gene-level) or `label_name` (complex-level,
    aggregated as MEAN across member genes).
    """
    if eval_csv is None or not Path(eval_csv).exists():
        print(f"  [eval] {eval_csv} not found; skipping",
              flush=True)
        return {}
    try:
        df = pd.read_csv(eval_csv)
    except Exception as e:
        print(f"  [eval] read failed: {e!r}; skipping", flush=True)
        return {}
    sub = df[df["n_cells"] == n_cells]
    if sub.empty:
        print(f"  [eval] no rows with n_cells={n_cells} in {eval_csv.name}",
              flush=True)
        return {}
    result: dict = {}
    if "label_name" in sub.columns:
        agg = sub.groupby("label_name").agg(
            top1_acc=("top1_acc", "mean"),
            top5_acc=("top5_acc", "mean"),
            n_samples=("n_samples", "sum"),
        )
        for label, r in agg.iterrows():
            if not isinstance(label, str) or not label.strip():
                continue
            result[label.strip()] = {
                "top1_acc": float(r["top1_acc"]),
                "top5_acc": float(r["top5_acc"]),
                "n_samples": int(r["n_samples"]),
            }
        return result
    # Schema drift: legacy CSVs key by `class_name`; the v3/attention_v3/
    # cdino/ CSVs use `gene_name`. Accept either so the auto-default
    # path keeps working when the eval source rolls forward.
    key_col = "class_name" if "class_name" in sub.columns else "gene_name"
    for _, r in sub.iterrows():
        g = r.get(key_col)
        if not isinstance(g, str) or not g.strip():
            continue
        result[g.strip()] = {
            "top1_acc": float(r.get("top1_acc", float("nan"))),
            "top5_acc": float(r.get("top5_acc", float("nan"))),
            "n_samples": int(r.get("n_samples", 0)),
        }
    return result


def _load_eval_accuracy(eval_csv: Path, n_cells: int = 100) -> dict:
    """Load attention-model accuracy for BOTH modalities (phase + fluor).

    `eval_csv` names the PHASE CSV (e.g. `cdino_eval_phase_50.csv`); the
    fluor sibling is auto-discovered by replacing the first 'phase' →
    'fluorescent' in the filename. Both load with the same n_cells
    threshold so the two top1_acc numbers are comparable.

    Returns:
        {key: {
            'top1_acc':       phase top1_acc (legacy alias; kept so
                              downstream code that reads `top1_acc`
                              still gets the phase value),
            'top1_acc_phase': phase top1_acc (explicit),
            'top1_acc_fluor': fluor top1_acc or NaN if missing,
            'top5_acc':       phase top5_acc (legacy alias),
            'n_samples':      phase n_samples (legacy alias),
        }, ...}
    """
    phase_result = _load_one_eval_csv(eval_csv, n_cells)
    # Resolve fluor sibling — same dir, replace 'phase' → 'fluorescent'.
    fluor_csv = None
    if eval_csv is not None:
        ec = Path(eval_csv)
        if "phase" in ec.name:
            fluor_csv = ec.with_name(ec.name.replace("phase", "fluorescent", 1))
    fluor_result = _load_one_eval_csv(fluor_csv, n_cells) if fluor_csv else {}

    out: dict = {}
    keys = set(phase_result) | set(fluor_result)
    for k in keys:
        p = phase_result.get(k, {})
        f = fluor_result.get(k, {})
        out[k] = {
            # Legacy alias: `top1_acc` continues to mean PHASE so the
            # existing --threshold filter (which reads top1_acc) keeps
            # the same behavior as before.
            "top1_acc":       p.get("top1_acc", float("nan")),
            "top5_acc":       p.get("top5_acc", float("nan")),
            "n_samples":      p.get("n_samples", 0),
            "top1_acc_phase": p.get("top1_acc", float("nan")),
            "top1_acc_fluor": f.get("top1_acc", float("nan")),
        }
    return out


def _load_chad_annotations(config_path):
    """Build gene -> (chad_cluster, chad_supercluster) annotations
    AND complex_name -> [member_genes] for the page-header member list.

    Returns (gene_to_cluster, gene_to_supercluster, cluster_to_genes).
    On any loading error, returns ({}, {}, {}) so the page falls back
    cleanly.
    """
    try:
        import yaml
        from ops_utils.analysis.gene_supercategories import (
            build_gene_supercategory_map,
            _load_chad_hierarchy,
            DEFAULT_CHAD_PATH,
        )
    except Exception as e:
        print(f"  [chad] annotation imports failed: {e}", flush=True)
        return {}, {}, {}

    try:
        with open(config_path) as f:
            supercat_config = yaml.safe_load(f) or {}
    except Exception as e:
        print(f"  [chad] could not load {config_path}: {e}", flush=True)
        return {}, {}, {}

    # Gene -> CHAD protein complex / cluster name (first cluster encountered)
    # AND cluster_name -> [member_genes] for complex pages.
    gene_to_cluster = {}
    cluster_to_genes = {}
    try:
        chad_path = Path(
            supercat_config.get("chad_hierarchy_path", str(DEFAULT_CHAD_PATH))
        )
        chad = _load_chad_hierarchy(chad_path)
        for _id, cluster in chad.items():
            if not isinstance(cluster, dict) or "name" not in cluster:
                continue
            cluster_name = cluster["name"]
            members = list(cluster.get("genes", []) or [])
            cluster_to_genes.setdefault(cluster_name, members)
            for gene in members:
                gene_to_cluster.setdefault(gene, cluster_name)
    except Exception as e:
        print(f"  [chad] cluster map failed: {e}", flush=True)

    # Gene -> CHAD-boosted supercluster/supercategory
    gene_to_supercluster = {}
    try:
        gene_to_supercluster = dict(
            build_gene_supercategory_map(supercat_config, boosted=True) or {}
        )
    except Exception as e:
        print(f"  [chad] supercluster map failed: {e}", flush=True)

    return gene_to_cluster, gene_to_supercluster, cluster_to_genes


# ──────────────────────────────────────────────────────────────────────────
# Cell loading via BaseDataset
# ──────────────────────────────────────────────────────────────────────────
PHASE_MASK_PREFERENCE = ("cell_seg",)
FLUOR_MASK_PREFERENCE = ("4i_cell_seg", "cp_cell_seg", "cell_seg")
# Nuclear segmentation labels — overlaid as a thin blue contour on every
# rendered tile to give the reader a stable nuclear reference. Live-cell
# experiments use plain `nuclear_seg`; CP / 4i experiments carry their own
# variants (`CP1_nuclear_seg`, `4i_R1_nuclear_seg`). Tiles from CP/4i
# stores fall back through all three; phase tiles only use the live-cell
# label since the others are physically absent from those zarrs.
NUCLEAR_MASK_PREFERENCE_PHASE = ("nuclear_seg",)
NUCLEAR_MASK_PREFERENCE_FLUOR = ("4i_R1_nuclear_seg", "CP1_nuclear_seg", "nuclear_seg")
# Nuclear contour matches the cell-mask "negative overlay" color/alpha
# (the translucent halo drawn outside each cell at line ~770:
# RGB(0.30, 0.40, 0.85) with alpha=0.55) so the two blue elements
# read as part of the same visual system rather than competing hues.
NUCLEAR_OUTLINE_COLOR = "#4D66D9"   # RGB(0.30, 0.40, 0.85) hex
NUCLEAR_OUTLINE_LW = 0.55
NUCLEAR_OUTLINE_ALPHA = 0.55
# Gaussian sigma (in pixels) applied to the binary nuclear mask before
# contour extraction. Smooths the pixel-grid jaggies without shifting the
# boundary noticeably (~0.5–0.8 px @ sigma=3). Set to 0 to disable.
# 3.0 is still cheap (a single ndimage pass per crop) and gives a notably
# rounder outline than 1.5.
NUCLEAR_OUTLINE_SMOOTH_SIGMA = 3.0
# Virtual-stain ('vs') overlay parameters. The nuclei_prediction channel
# in the OPS zarrs has most signal in the 0–30 intensity range, so
# clipping to that window saturates the brighter pixels (alpha→max) and
# leaves the long upper tail untouched — gives the glow a stronger
# nuclear-body read without raising the global max alpha excessively.
NUCLEAR_VS_CLIP_LO = 0.0
NUCLEAR_VS_CLIP_HI = 30.0
NUCLEAR_VS_MAX_ALPHA = 0.35


def _bbox_center_crop(data, mask, target_size, nuc_mask=None, nuc_pred=None):
    """Slice a wider-than-needed load down to (target_size × target_size)
    centered on the segmentation mask's bbox center.

    Inputs are the arrays loaded by BaseDataset at the WIDER pad-factor
    size; output is the tight DISPLAY crop. Returns mask center when
    mask is empty (defensive fallback so blank-mask cells still render
    a centered tile).

    All arrays are sliced on their last two axes — works uniformly for
    `data` shape (C, Y, X) and 2D `mask`/`nuc_mask`/`nuc_pred` (Y, X).
    """
    H, W = mask.shape[-2:]
    if mask.any():
        ys, xs = np.where(mask)
        cy = (int(ys.min()) + int(ys.max())) // 2
        cx = (int(xs.min()) + int(xs.max())) // 2
    else:
        cy, cx = H // 2, W // 2
    half = target_size // 2
    # Clamp so the slice window stays inside the loaded array. This
    # also auto-handles cells whose bbox would otherwise stretch past
    # the load edge — they get re-centered to the closest fit.
    y0 = max(0, min(H - target_size, cy - half))
    x0 = max(0, min(W - target_size, cx - half))
    y1, x1 = y0 + target_size, x0 + target_size

    def _slice(arr):
        if arr is None:
            return None
        return arr[..., y0:y1, x0:x1]
    return _slice(data), _slice(mask), _slice(nuc_mask), _slice(nuc_pred)


def _build_base_dataset(cell_rows, store_cache, crop_size, load_pad_factor=1.0):
    # Internal load size — wider than the display crop so we can re-center
    # each tile on the cell's segmentation bbox after loading.
    load_size = int(crop_size * load_pad_factor)
    half = load_size // 2
    records = []
    input_indices = []
    per_record_mask_pref = []
    for in_idx, cell in enumerate(cell_rows):
        exp = cell["experiment"]
        store = store_cache.get(exp)
        if store is None:
            continue
        y = int(cell["y_pheno"])
        x = int(cell["x_pheno"])
        records.append(
            {
                "experiment": exp,
                "store_key": exp,
                "well": _normalize_well(cell["well"]),
                "bbox": [y - half, x - half, y + half, x + half],
                "segmentation_id": cell["segmentation"],
                "gene_name": cell["gene"],
                "total_index": len(records),
            }
        )
        # row kind: "phase" -> cell_seg only; else fall back to 4i/cp/cell_seg
        per_record_mask_pref.append(
            PHASE_MASK_PREFERENCE if cell.get("kind") == "phase" else FLUOR_MASK_PREFERENCE
        )
        input_indices.append(in_idx)

    if not records:
        return None, []

    df = pd.DataFrame(records)
    stores_needed = {
        exp: store_cache.stores[exp]
        for exp in df["experiment"].unique()
        if store_cache.stores.get(exp) is not None
    }
    ds = BaseDataset(
        stores=stores_needed,
        labels_df=df,
        initial_yx_patch_size=(load_size, load_size),
        final_yx_patch_size=(load_size, load_size),
        out_channels="all",
        mask_cell=False,
    )

    # Monkey-patch mask loader. Phase cells always use plain cell_seg; fluor
    # cells try 4i_cell_seg / cp_cell_seg / cell_seg in preference order so
    # 4i and Cell Painting experiments get the correct cell outline.
    # Caches the working label per (store_key, well, preference_tuple).
    import types
    label_cache = {}

    def _add_mask_multi(self, ci, bbox):
        mh = bbox[2] - bbox[0]
        mw = bbox[3] - bbox[1]
        if pd.isna(ci.segmentation_id):
            return np.ones((1, mh, mw), dtype=bool)
        pref = per_record_mask_pref[int(ci.total_index)]
        cache_key = (ci.store_key, ci.well, pref)
        candidates = (
            (label_cache[cache_key],)
            if cache_key in label_cache
            else pref
        )
        last_err = None
        for label in candidates:
            try:
                arr = self.load_label_array(ci.store_key, ci.well, label, bbox)
                if arr is None:
                    continue
                label_cache[cache_key] = label
                sc_mask = (arr == int(ci.segmentation_id))
                return np.expand_dims(sc_mask, axis=0)
            except Exception as e:
                last_err = e
                continue
        print(
            f"  Warning: no {'/'.join(pref)} mask for "
            f"{ci.store_key} {ci.well}: {last_err}",
            flush=True,
        )
        return np.ones((1, mh, mw), dtype=bool)

    ds.add_mask_to_batch = types.MethodType(_add_mask_multi, ds)
    return ds, input_indices


# ──────────────────────────────────────────────────────────────────────────
# Rendering
# ──────────────────────────────────────────────────────────────────────────
def _load_nuc_mask(ds, store_key, well, bbox, cell_mask, pref, label_cache=None):
    """Resolve the nuclear segmentation crop for a single cell.

    Tries `pref` labels in order against `ds.load_label_array(...)`; the
    first label that exists at this position wins (cached per `(store_key,
    well, pref)` so we don't re-probe the same store for every cell).
    Returns a binary mask of the SINGLE nucleus that overlaps `cell_mask`
    most — the corresponding nucleus of this cell — or None if no
    nuclear seg is available or the cell tile contains no nucleus.
    """
    cache_key = (store_key, well, pref)
    candidates = (
        (label_cache[cache_key],)
        if label_cache is not None and cache_key in label_cache
        else pref
    )
    for label in candidates:
        try:
            arr = ds.load_label_array(store_key, well, label, tuple(bbox))
        except Exception:
            continue
        if arr is None:
            continue
        if arr.shape != cell_mask.shape:
            continue
        if label_cache is not None:
            label_cache[cache_key] = label
        # Restrict to the nucleus most overlapping this cell — avoids
        # outlining a neighbor's nucleus that happens to clip the bbox.
        overlap = arr * cell_mask.astype(arr.dtype)
        if not (overlap > 0).any():
            return None
        ids, counts = np.unique(overlap[overlap > 0], return_counts=True)
        return (arr == int(ids[np.argmax(counts)]))
    return None


def _render_cell(ax, crop, mask, title, vmin=None, vmax=None,
                 nuc_mask=None, nuc_pred=None):
    if crop is None:
        ax.text(
            0.5, 0.5, "n/a",
            ha="center", va="center", transform=ax.transAxes,
            fontsize=6, color="gray",
        )
    else:
        if vmin is None or vmax is None:
            if crop.size > 0:
                auto_vmin, auto_vmax = np.percentile(crop, FLUOR_TILE_PERCENTILES)
                if auto_vmax <= auto_vmin:
                    auto_vmax = float(auto_vmin) + 1e-6
            else:
                auto_vmin, auto_vmax = 0.0, 1.0
            if vmin is None:
                vmin = auto_vmin
            if vmax is None:
                vmax = auto_vmax
        ax.imshow(crop, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")

        if mask is not None and mask.any():
            dilated = binary_dilation(mask, iterations=MASK_DILATION)
            inv = ~dilated
            overlay = np.zeros((*inv.shape, 4), dtype=np.float32)
            overlay[..., 0] = 0.30
            overlay[..., 1] = 0.40
            overlay[..., 2] = 0.85
            overlay[..., 3] = np.where(inv, 0.55, 0.0)
            ax.imshow(overlay, interpolation="nearest")

        # Nuclear-overlay layers, picked at the call site via the
        # `--nuclear-overlay` CLI flag:
        #   'seg'  → `nuc_mask` only — thin blue contour traced from
        #            nuclear_seg. Gaussian-smoothed before tracing so
        #            the curve follows a continuous level set (kills
        #            the staircase look at PDF zoom).
        #   'vs'   → `nuc_pred` only — low-alpha blue glow from the
        #            nuclei_prediction virtual-stain channel, clipped to
        #            a fixed 0–50 intensity window for stable scaling.
        #   'both' → BOTH crops are populated upstream so we render
        #            the VS glow first (zorder=4) and the contour on
        #            top (zorder=5) — contour stays crisp over the
        #            translucent fill.
        if nuc_pred is not None and nuc_pred.size:
            p = np.asarray(nuc_pred, dtype=np.float32)
            p = np.clip(
                (p - NUCLEAR_VS_CLIP_LO)
                / max(NUCLEAR_VS_CLIP_HI - NUCLEAR_VS_CLIP_LO, 1e-6),
                0.0, 1.0,
            )
            overlay = np.zeros((*p.shape, 4), dtype=np.float32)
            overlay[..., 0] = 0.30
            overlay[..., 1] = 0.40
            overlay[..., 2] = 0.85
            overlay[..., 3] = p * NUCLEAR_VS_MAX_ALPHA
            ax.imshow(overlay, interpolation="nearest", zorder=4)
        if nuc_mask is not None and nuc_mask.any():
            mask_f = nuc_mask.astype(np.float32)
            if NUCLEAR_OUTLINE_SMOOTH_SIGMA > 0:
                mask_f = gaussian_filter(mask_f, sigma=NUCLEAR_OUTLINE_SMOOTH_SIGMA)
            for c in find_contours(mask_f, 0.5):
                ax.plot(
                    c[:, 1], c[:, 0],
                    color=NUCLEAR_OUTLINE_COLOR,
                    linewidth=NUCLEAR_OUTLINE_LW,
                    alpha=NUCLEAR_OUTLINE_ALPHA, zorder=5,
                    solid_capstyle="round", solid_joinstyle="round",
                )

    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    # title may be None (n/a), a str (legacy), or a (line1, line2, exp, well, x, y) tuple
    if title:
        if isinstance(title, tuple):
            line1, line2, exp, well, x, y = title
            _set_structured_title(ax, line1, line2, exp, well, x, y)
        else:
            ax.set_title(str(title), fontsize=5, pad=2)


def _format_title(row, rank_key, zarr_ch_label=None):
    """Phase-KO title spec: (line1, line2, exp, well, x, y).

    When `pma_attention` is NaN (e.g. picker-selected cells in NTC mode,
    where cells are chosen by SHAP-feature-mean distance rather than
    attention rank), drop the `attn=` segment so the label stays
    meaningful rather than printing "attn=nan".
    """
    rank = int(row[rank_key]) if pd.notna(row[rank_key]) else 0
    ch_suffix = f"  [{zarr_ch_label}]" if zarr_ch_label else ""
    attn = row.get("pma_attention")
    if pd.notna(attn):
        line1 = f"#{rank}  attn={attn:.4f}{ch_suffix}"
    else:
        line1 = f"#{rank}{ch_suffix}"
    line2 = f"x={int(row['x_pheno'])} y={int(row['y_pheno'])}"
    return (line1, line2, row["experiment"], row["well"], row["x_pheno"], row["y_pheno"])


def _format_phase_ntc_title(row, zarr_ch):
    """Phase-NTC title spec: (line1, line2, exp, well, x, y)."""
    ch_suffix = f" [{zarr_ch}]" if zarr_ch else ""
    label = row.get("gene_label", "NTC")
    line1 = f"{label}{ch_suffix}"
    line2 = f"x={int(row['x_pheno'])} y={int(row['y_pheno'])}"
    return (line1, line2, row["experiment"], row["well"], row["x_pheno"], row["y_pheno"])


def _format_fluor_ntc_title(row, zarr_ch):
    """Fluor-NTC title spec: (line1, line2, exp, well, x, y)."""
    viz_ch = row.get("viz_channel") or ""
    label = row.get("gene_label", "NTC")
    line1 = label
    line2 = _channel_label(viz_ch, zarr_ch)
    return (line1, line2, row["experiment"], row["well"], row["x_pheno"], row["y_pheno"])


_DYE_TRAILERS = (
    " Live Cell Dye", " Live-Cell Dye",
    " live cell dye", " live-cell dye",
    " Live Cell dye", " live Cell dye",
    " live cell dyes", " Live Cell Dyes",
    " live cell stain", " Live Cell Stain",
    " excitation",
)


def _strip_dye_trailer(name):
    """Strip generic dye/probe trailers ("live cell dye", "excitation",
    etc.) from marker names so atlas labels read "BODIPY" / "FeRhoNox"
    instead of "BODIPY live cell dye". Returns name unchanged when no
    known trailer matches.
    """
    if not isinstance(name, str):
        return name
    for suf in _DYE_TRAILERS:
        if name.lower().endswith(suf.lower()):
            return name[: -len(suf)].strip()
    return name


def _channel_label(viz_channel, zarr_ch):
    """Compose the channel label shown on fluor tiles. Underscores in the
    marker name are replaced with spaces for readability. CP / 4i tiles get
    a fixed-modality tag at the end; live-cell tiles get the fluorophore.
        CP     : 'Mitochondria TOMM20 [CP fixed]'
        4i     : 'gH2AX [4i fixed]'
        live   : 'autophagosome MAP1LC3B [GFP]'
    Dye/probe trailers ("live cell dye", "excitation") are stripped so
    the visible label stays compact ("FeRhoNox" instead of "FeRhoNox
    live-cell dye").
    """
    marker = _strip_dye_trailer(str(viz_channel)).replace("_", " ")
    if zarr_ch is None:
        return f"{marker} (no img)"
    z = str(zarr_ch)
    if z.startswith(("CP1_", "CP2_")):
        return f"{marker} [CP fixed]"
    if z.startswith("4i_"):
        return f"{marker} [4i fixed]"
    return f"{marker} [{zarr_ch}]"


def _format_fluor_title(row, zarr_ch, rank_key):
    """Fluor-KO title spec: (line1, line2, exp, well, x, y).

    When `pma_attention` is NaN (NTC variant picker rows), drop the
    `attn=` segment — those cells were selected by SHAP-feature-mean
    distance, not attention rank, so showing "attn=nan" is misleading.
    """
    rank = int(row[rank_key]) if pd.notna(row[rank_key]) else 0
    attn = row.get("pma_attention")
    if pd.notna(attn):
        line1 = f"#{rank}  attn={attn:.4f}"
    else:
        line1 = f"#{rank}"
    line2 = _channel_label(row["viz_channel"], zarr_ch)
    return (line1, line2, row["experiment"], row["well"], row["x_pheno"], row["y_pheno"])


def render_gene_page(gene, phase_rows, ntc_phase_rows, fluor_pairs,
                     chad_cluster, chad_supercluster, uniprot_function,
                     store_cache, crop_size, strict=True,
                     fig_factory=None, post_render_hook=None,
                     eval_acc=None,
                     channel_colors=None, chad_members=None,
                     chad_description=None,
                     mAP_all_combined=None, mAP_metric_label=None,
                     nuclear_overlay="none",
                     n_fluor_channels=None,
                     load_pad_factor=1.0):
    """fluor_pairs: list of `n_fluor_channels` dicts, each with
    'viz_channel' (str or None), 'fluor_rows' (DataFrame), 'ntc_fluor_rows' (DataFrame).

    `n_fluor_channels` overrides the global `N_FLUOR_CHANNELS_PER_GENE`
    (3) on a per-page basis — `attention_atlas_shap` passes a dynamic
    count based on the per-gene mAP-threshold filter, so a high-signal
    complex with 9 above-threshold markers renders 9 fluor rows while
    a low-signal one might render only 1. Defaults to the global
    constant when not provided (legacy callers + tests).

    NTC rows oversample by `NTC_OVERSAMPLE_FACTOR` and only fill slots with
    candidates whose mask actually covers the cell (rejects fake/stale cells
    whose seg_id doesn't appear in the segmentation mask at that bbox).

    `fig_factory()` -> (fig, axes_NxM) overrides the default figure layout so
    callers (e.g. attention_atlas_shap) can embed the image grid in a wider
    figure with their own side panels. `post_render_hook(fig, axes)` runs
    after all cell tiles are drawn and lets the caller fill those side panels.
    """
    # Per-page channel-count override. Locals shadow the module globals
    # for all loops/indexing in this function. `_n_fluor` is the number
    # of (KO + NTC) row-pairs for fluor channels; `_n_total_rows` is the
    # total grid height (2 for phase + 2 per fluor channel).
    _n_fluor = n_fluor_channels if n_fluor_channels is not None else N_FLUOR_CHANNELS_PER_GENE
    _n_total_rows = 2 + 2 * _n_fluor
    cells_to_load = []
    # slots: list of (in_idx, grid_row, grid_col, title, zarr_channel, viz_channel_or_None)
    # in_idx is the index into cells_to_load. For NTC rows, slots are assigned
    # AFTER validation (so cells_to_load can have more entries than slots).
    slots = []
    row_fills = {r: 0 for r in range(_n_total_rows)}
    failures = []
    # ntc_candidates[ntc_row_idx] -> list of dicts { in_idx, title, zarr_ch, viz_ch }
    ntc_candidates = {}

    def _add_cell(payload):
        cells_to_load.append(payload)
        return len(cells_to_load) - 1

    # Row 0: phase KO (from phase CSV, top 10) — direct slots
    row_fills[ROW_PHASE_KO] = min(len(phase_rows), N_COLS)
    for c in range(row_fills[ROW_PHASE_KO]):
        row = phase_rows.iloc[c]
        in_idx = _add_cell({
            "experiment": row["experiment"], "well": row["well"],
            "y_pheno": row["y_pheno"], "x_pheno": row["x_pheno"],
            "segmentation": row["segmentation"], "gene": gene, "kind": "phase",
        })
        zarr_ch = _resolve_phase_channel(store_cache.get(row["experiment"]))
        slots.append((in_idx, ROW_PHASE_KO, c,
                      _format_title(row, "rank", zarr_ch), zarr_ch, None))

    # Row 1: phase NTC — oversample candidates, filter later
    phase_ntc_limit = min(len(ntc_phase_rows), N_COLS * NTC_OVERSAMPLE_FACTOR)
    ntc_candidates[ROW_PHASE_NTC] = []
    for c in range(phase_ntc_limit):
        row = ntc_phase_rows.iloc[c]
        in_idx = _add_cell({
            "experiment": row["experiment"], "well": row["well"],
            "y_pheno": row["y_pheno"], "x_pheno": row["x_pheno"],
            "segmentation": row["segmentation"],
            "gene": row.get("gene_label", "NTC"), "kind": "phase",
        })
        zarr_ch = _resolve_phase_channel(store_cache.get(row["experiment"]))
        ntc_candidates[ROW_PHASE_NTC].append({
            "in_idx": in_idx,
            "title": _format_phase_ntc_title(row, zarr_ch),
            "zarr_ch": zarr_ch,
            "viz_ch": None,
        })

    # Rows 2..7: 3 (fluor KO, fluor NTC) pairs — KO direct, NTC candidates.
    # Track explicitly-skipped rows (e.g. --skip-cp-4i) so the strict check
    # below doesn't treat their 0-fill as a "short row" failure.
    skipped_rows = set()
    for ch_idx_pair, pair in enumerate(fluor_pairs[:_n_fluor]):
        viz_ch = pair.get("viz_channel")
        if pair.get("skipped"):
            skipped_rows.add(_fluor_ko_row(ch_idx_pair))
            skipped_rows.add(_fluor_ntc_row(ch_idx_pair))
            continue
        ko_df = pair.get("fluor_rows")
        if ko_df is None:
            ko_df = pd.DataFrame()
        ntc_df = pair.get("ntc_fluor_rows")
        if ntc_df is None:
            ntc_df = pd.DataFrame()

        ko_row_idx = _fluor_ko_row(ch_idx_pair)
        row_fills[ko_row_idx] = min(len(ko_df), N_COLS)
        for c in range(row_fills[ko_row_idx]):
            row = ko_df.iloc[c]
            in_idx = _add_cell({
                "experiment": row["experiment"], "well": row["well"],
                "y_pheno": row["y_pheno"], "x_pheno": row["x_pheno"],
                "segmentation": row["segmentation"], "gene": gene,
            })
            store = store_cache.get(row["experiment"])
            cm = store_cache.channel_map(row["experiment"])
            zarr_ch = _resolve_fluor_channel(store, cm, row["viz_channel"])
            rank_key = "rank" if "rank" in row.index else "channel_rank"
            slots.append((in_idx, ko_row_idx, c,
                          _format_fluor_title(row, zarr_ch, rank_key),
                          zarr_ch, row["viz_channel"]))

        ntc_row_idx = _fluor_ntc_row(ch_idx_pair)
        ntc_cands = []
        ntc_limit = min(len(ntc_df), N_COLS * NTC_OVERSAMPLE_FACTOR)
        for c in range(ntc_limit):
            row = ntc_df.iloc[c]
            in_idx = _add_cell({
                "experiment": row["experiment"], "well": row["well"],
                "y_pheno": row["y_pheno"], "x_pheno": row["x_pheno"],
                "segmentation": row["segmentation"],
                "gene": row.get("gene_label", "NTC"),
            })
            store = store_cache.get(row["experiment"])
            cm = store_cache.channel_map(row["experiment"])
            zarr_ch = _resolve_fluor_channel(store, cm, viz_ch)
            ntc_cands.append({
                "in_idx": in_idx,
                "title": _format_fluor_ntc_title(row, zarr_ch),
                "zarr_ch": zarr_ch,
                "viz_ch": viz_ch,
            })
        ntc_candidates[ntc_row_idx] = ntc_cands

    # ── Load all crops (KO slots + NTC candidates) ────────────────────────
    ds, input_indices = _build_base_dataset(
        cells_to_load, store_cache, crop_size,
        load_pad_factor=load_pad_factor,
    )
    input_indices_set = set(input_indices)
    cell_load_errors = {}
    crops_by_input = {}
    # Nuclear-overlay crops per cell. Only ONE of these is filled per
    # tile, picked by the --nuclear-overlay flag:
    #   'seg' → `nuc_masks_by_input` holds a binary nuclear_seg crop
    #           (or CP/4i variant); rendered as a smoothed contour.
    #   'vs'  → `nuc_preds_by_input` holds the raw `nuclei_prediction`
    #           channel from this cell's data; rendered as a soft blue
    #           glow with alpha ∝ intensity.
    #   'none' → both stay empty; no overlay drawn.
    nuc_masks_by_input: dict[int, np.ndarray | None] = {}
    nuc_preds_by_input: dict[int, np.ndarray | None] = {}
    nuc_label_cache: dict = {}
    if ds is not None:
        for ds_idx, in_idx in enumerate(input_indices):
            try:
                batch = ds[ds_idx]
                data = batch["data"].numpy()
                mask = batch["mask"].numpy()[0].astype(bool)
                # Stash the LOAD-sized arrays; we'll bbox-center-slice
                # them down to crop_size after the optional nuc overlays
                # are also loaded at the same wide size.
                crops_by_input[in_idx] = (data, mask)
                cell = cells_to_load[in_idx]
                rec = ds.labels_df.iloc[ds_idx]
                if nuclear_overlay in ("seg", "both"):
                    pref = (
                        NUCLEAR_MASK_PREFERENCE_PHASE
                        if cell.get("kind") == "phase"
                        else NUCLEAR_MASK_PREFERENCE_FLUOR
                    )
                    try:
                        nuc_masks_by_input[in_idx] = _load_nuc_mask(
                            ds, rec["store_key"], rec["well"], rec["bbox"],
                            mask, pref, label_cache=nuc_label_cache,
                        )
                    except Exception:
                        nuc_masks_by_input[in_idx] = None
                if nuclear_overlay in ("vs", "both"):
                    # Pluck `nuclei_prediction` straight from the already-
                    # loaded multi-channel `data` array — no separate zarr
                    # read needed. None when the experiment doesn't carry
                    # that channel (CP/4i can be missing it).
                    store = store_cache.get(rec["store_key"])
                    if store is not None:
                        ch_names = list(store.channel_names)
                        if "nuclei_prediction" in ch_names:
                            nuc_preds_by_input[in_idx] = data[
                                ch_names.index("nuclei_prediction")
                            ]
            except Exception as e:
                cell = cells_to_load[in_idx]
                cell_load_errors[in_idx] = f"BaseDataset __getitem__ raised: {e}"
                print(
                    f"    [{gene}] cell load failed "
                    f"(exp={cell['experiment']} well={cell['well']} "
                    f"seg={cell['segmentation']}): {e}",
                    flush=True,
                )

    # ── Bbox-center re-crop ────────────────────────────────────────────────
    # BaseDataset loaded each tile at `load_size = crop_size * load_pad_factor`;
    # now slice every cell's data + mask + nuc overlays down to
    # `crop_size × crop_size` centered on the segmentation mask's bbox center.
    # No-op when load == crop (--load-pad-factor 1.0).
    target_size = crop_size
    for in_idx, pair in list(crops_by_input.items()):
        data, mask = pair
        if data.shape[-1] == target_size and data.shape[-2] == target_size:
            continue  # load == display, nothing to slice
        nuc_mask = nuc_masks_by_input.get(in_idx)
        nuc_pred = nuc_preds_by_input.get(in_idx)
        data, mask, nuc_mask, nuc_pred = _bbox_center_crop(
            data, mask, target_size, nuc_mask=nuc_mask, nuc_pred=nuc_pred,
        )
        crops_by_input[in_idx] = (data, mask)
        if in_idx in nuc_masks_by_input:
            nuc_masks_by_input[in_idx] = nuc_mask
        if in_idx in nuc_preds_by_input:
            nuc_preds_by_input[in_idx] = nuc_pred

    # ── NTC candidate validation (strict: crop + mask + intensity) ───────
    # With per-row pool pre-filtering (live-seg for phase, fluor-seg for
    # CP/4i), every sampled candidate should have a seg-ID that matches its
    # target mask — so mask.sum() >= NTC_MIN_MASK_PIXELS should rarely fail.
    def _ntc_is_real(in_idx, zarr_ch):
        pair = crops_by_input.get(in_idx)
        if pair is None:
            return False
        data, mask = pair
        if int(mask.sum()) < NTC_MIN_MASK_PIXELS:
            return False
        if zarr_ch is None:
            return False
        exp = cells_to_load[in_idx]["experiment"]
        store = store_cache.get(exp)
        if store is None:
            return False
        ch_names = list(store.channel_names)
        if zarr_ch not in ch_names:
            return False
        crop = data[ch_names.index(zarr_ch)]
        inside = crop[mask]
        if inside.size < 1 or float(inside.std()) < NTC_MIN_INTENSITY_STD:
            return False
        return True

    for ntc_row_idx, cands in ntc_candidates.items():
        valid_count = 0
        for cand in cands:
            if valid_count >= N_COLS:
                break
            if not _ntc_is_real(cand["in_idx"], cand["zarr_ch"]):
                continue
            slots.append((
                cand["in_idx"], ntc_row_idx, valid_count,
                cand["title"], cand["zarr_ch"], cand["viz_ch"],
            ))
            valid_count += 1
        row_fills[ntc_row_idx] = valid_count

    # ── Per-viz_channel clims (from final slots only, post-NTC-validation) ──
    # All fluor clim statistics are computed over the CELL PIXELS (crop[mask])
    # not the full crop, so neighboring cells and background don't bias the
    # (vmin, vmax) for the target cell.
    def _tile_masked_bounds(in_idx, channel):
        pair = crops_by_input.get(in_idx)
        if pair is None:
            return None
        data, mask = pair
        if mask is None or not mask.any():
            return None
        store = store_cache.get(cells_to_load[in_idx]["experiment"])
        if store is None:
            return None
        ch_names = list(store.channel_names)
        if channel not in ch_names:
            return None
        pixels = data[ch_names.index(channel)][mask]
        if pixels.size == 0:
            return None
        lo, hi = np.percentile(pixels, (lo_pct, hi_pct))
        return float(lo), float(hi)

    viz_channel_clims = {}
    tile_bounds = {}
    lo_pct, hi_pct = FLUOR_TILE_PERCENTILES
    for (in_idx, gr, gc, title, channel, viz_ch) in slots:
        if viz_ch is None or channel is None:
            continue
        bounds = _tile_masked_bounds(in_idx, channel)
        if bounds is None:
            continue
        tile_bounds.setdefault(viz_ch, []).append(bounds)
    for viz_ch, pairs in tile_bounds.items():
        lows = np.array([p[0] for p in pairs])
        highs = np.array([p[1] for p in pairs])
        lo = float(np.median(lows))
        hi = float(np.median(highs))
        if hi <= lo:
            hi = lo + 1e-6
        viz_channel_clims[viz_ch] = (lo, hi)

    # Outlier-clim override: any fluor tile whose own masked hi is way above
    # or below the viz_channel's median hi gets its own (vmin, vmax) instead
    # of the shared one, plus a red/blue marker in the corner.
    #
    # For UPPER outliers we use the masked pixel max as vmax (no clipping at
    # all), since 99.9 percentile on very-bright cells still leaves visible
    # saturation. For LOWER outliers the tile's 99.9 percentile is fine.
    slot_override = {}
    for (in_idx, gr, gc, title, channel, viz_ch) in slots:
        if viz_ch is None or channel is None:
            continue
        if channel in PHASE_CHANNEL_CANDIDATES:
            continue
        if viz_ch not in viz_channel_clims:
            continue
        bounds = _tile_masked_bounds(in_idx, channel)
        if bounds is None:
            continue
        t_lo, t_hi = bounds
        med_lo, med_hi = viz_channel_clims[viz_ch]
        # For upper outliers we may need (a) the masked max and (b) the full
        # crop max — both are needed because the crop often contains a
        # neighbor cell or halo pixel that's brighter than the target cell's
        # own max, which would clip to white under the blue inverse-mask
        # overlay and look "saturated" despite the cell itself being in range.
        pair = crops_by_input.get(in_idx)
        masked_max = None
        crop_max = None
        if pair is not None:
            data, mask = pair
            store = store_cache.get(cells_to_load[in_idx]["experiment"])
            if store is not None:
                ch_names = list(store.channel_names)
                if channel in ch_names:
                    full_crop = data[ch_names.index(channel)]
                    crop_max = float(full_crop.max())
                    if mask is not None and mask.any():
                        masked_max = float(full_crop[mask].max())
        if med_hi <= 0:
            continue
        ratio = float(t_hi) / float(med_hi)
        if ratio >= FLUOR_OUTLIER_HI_RATIO:
            # Use 1.15 * max(crop_max, masked_max) so NO pixel in the crop
            # (cell + background halo + neighbor cells) clips to white.
            # Gives bright cells an unclipped render and the background
            # under the blue overlay no longer has white patches.
            ceiling = None
            if crop_max is not None or masked_max is not None:
                ceiling = max(v for v in (crop_max, masked_max) if v is not None)
            vmax_up = ceiling * 1.15 if ceiling is not None else float(t_hi)
            slot_override[in_idx] = {
                "vmin": float(t_lo), "vmax": vmax_up,
                "marker": f"*↑{int(round(ratio))}×", "direction": "up",
                "color": "#D11616",   # red = too-bright outlier
            }
        elif ratio <= FLUOR_OUTLIER_LO_RATIO and ratio > 0:
            # Format as an integer denominator for readability: 0.25x → "4×"
            denom = int(round(1.0 / ratio)) if ratio > 0 else 0
            slot_override[in_idx] = {
                "vmin": float(t_lo), "vmax": float(t_hi),
                "marker": f"*↓{denom}×", "direction": "down",
                "color": "#1F46A6",   # blue = too-dim outlier
            }

    if fig_factory is None:
        fig, axes = plt.subplots(
            _n_total_rows, N_COLS,
            figsize=(N_COLS * 1.9, _n_total_rows * 2.25),
            squeeze=False,
            gridspec_kw={"hspace": 0.25, "wspace": 0.06},
            constrained_layout=False,
        )
        fig.subplots_adjust(left=0.025, right=0.995, top=0.955, bottom=0.01,
                            hspace=0.25, wspace=0.06)
    else:
        fig, axes = fig_factory()
    annot_bits = []
    if chad_cluster:
        annot_bits.append(f"Protein Complex: {chad_cluster}")
    if chad_supercluster:
        annot_bits.append(f"Supercluster: {chad_supercluster}")

    # Quality-score badges immediately after the gene name: the
    # all-markers-combined mAP (a single distinctiveness score for
    # this gene/complex) and the attention model's top1 accuracy at
    # the n_cells=100 eval bin (when available). Both render as
    # plain "key=value" strings so the suptitle reads as one
    # uniform left-flush block.
    quality_bits = []
    if isinstance(mAP_all_combined, (int, float)) and mAP_all_combined == mAP_all_combined:
        # Prefix the badge with the metric type ("distinct" for gene
        # level, "consist" for complex level) so readers can tell
        # which signal we're reporting.
        mlabel = mAP_metric_label or "mAP"
        quality_bits.append(f"mAP_{mlabel}={float(mAP_all_combined):.2f}")
    if isinstance(eval_acc, dict) and eval_acc:
        t1_phase = eval_acc.get("top1_acc_phase", eval_acc.get("top1_acc"))
        t1_fluor = eval_acc.get("top1_acc_fluor")
        bits = []
        if isinstance(t1_phase, (int, float)) and t1_phase == t1_phase:
            bits.append(f"{t1_phase:.0%} Phase")
        if isinstance(t1_fluor, (int, float)) and t1_fluor == t1_fluor:
            bits.append(f"{t1_fluor:.0%} Fluorescence")
        if bits:
            quality_bits.append("Attention Accuracy " + " / ".join(bits))
        au = eval_acc.get("auroc")
        if isinstance(au, (int, float)) and au == au:
            quality_bits.append(f"AUROC={au:.2f}")

    # Plain-text suptitle (no mathtext) so the gene name + annotations
    # always render — previously a stray LaTeX-special character in the
    # gene/complex name could silently break the whole title via the
    # `$\mathbf{}$` block. Bold across the whole title is fine; the
    # gene name is the leftmost token so reads as the anchor regardless.
    title_parts = [f"Attention Atlas — {gene}"]
    title_parts.extend(quality_bits)
    title_parts.extend(annot_bits)
    line1 = "    |    ".join(title_parts)
    # Anchor the header lines in absolute INCHES from the top of the
    # figure so they stay at a fixed visual distance from the mAP bar
    # / image grid (which are also inch-anchored in the SHAP variant).
    # Calibrated to the FH=18 baseline:
    #   suptitle    y=0.995 → 0.09" from top
    #   ontology    y=0.978 → 0.396" from top
    # Without this rebase a tall variable-page-size run (FH=50)
    # leaves the suptitle and ontology drifting far down into the
    # canvas — the ontology line could even end up BELOW the bar.
    _FH = fig.get_size_inches()[1]
    suptitle_y = 1.0 - 0.09 / _FH
    fig.suptitle(line1, fontsize=11, y=suptitle_y, fontweight="bold",
                 x=0.025, ha="left")
    # The ontology subtitle line lives BELOW the mAP heatmap bar (when
    # the SHAP wrapper is in use). Default 0.396" from top (= 0.978
    # at FH=18) is overridden to ~0.918 fig-frac when chad_members or
    # a custom y is in play. For complex pages, swap uniprot_function
    # for "Members (N): GENE1, GENE2, ..." so the page tells the
    # reader which genes the complex pools cells from.
    ontology_y = 1.0 - 0.396 / _FH

    if chad_members:
        # Complex page: "{description}    Members (N): GENE1, GENE2, ..."
        members_text = (
            f"Members ({len(chad_members)}): " + ", ".join(chad_members)
        )
        if chad_description:
            ontology_text = f"{chad_description}    {members_text}"
        else:
            ontology_text = members_text
    elif uniprot_function:
        ontology_text = uniprot_function
    else:
        ontology_text = None
        ontology_y = None
    if ontology_text:
        # Wrap the description line so it doesn't overflow the page.
        # Figure is 19" wide; at ~10.6pt the wrap budget shrinks to
        # ~175 char per line. Left-aligned at the same x as the
        # suptitle so the page header reads as a single block flush
        # to the image-grid's left edge.
        import textwrap
        wrapped = "\n".join(textwrap.wrap(ontology_text, width=175))
        fig.text(
            0.025, ontology_y, wrapped,
            ha="left", va="top",
            fontsize=10.6, style="italic", color="#333333",
        )

    rendered = set()
    for (in_idx, gr, gc, title, channel, viz_ch) in slots:
        ax = axes[gr, gc]
        rendered.add((gr, gc))
        cell = cells_to_load[in_idx]
        cell_ctx = (
            f"exp={cell['experiment']} well={cell['well']} "
            f"seg={cell['segmentation']}"
        )

        def _fail(reason):
            failures.append((gr, gc, f"{reason} ({cell_ctx})"))
            _render_cell(ax, None, None, title)

        # 1. BaseDataset didn't include this cell (store failed to open)
        if in_idx not in input_indices_set:
            store = store_cache.stores.get(cell["experiment"], "missing")
            if store is None:
                _fail("store failed to open")
            else:
                _fail("cell not loaded by BaseDataset")
            continue
        # 2. BaseDataset tried but raised
        if in_idx in cell_load_errors:
            _fail(cell_load_errors[in_idx])
            continue
        # 3. No zarr channel resolved (channel_map mismatch / missing Phase2D)
        if channel is None:
            _fail("no zarr channel resolved")
            continue
        pair = crops_by_input.get(in_idx)
        if pair is None:
            _fail("crop not in crops_by_input (unexpected)")
            continue
        data, mask = pair
        store = store_cache.get(cell["experiment"])
        if store is None:
            _fail("store cache returned None at render time")
            continue
        ch_names = list(store.channel_names)
        if channel not in ch_names:
            _fail(f"zarr does not have channel {channel!r} (has {ch_names})")
            continue
        ch_idx = ch_names.index(channel)
        # Contrast policy:
        #   - All phase tiles share the fixed PHASE_CONTRAST
        #   - Fluor tiles: all tiles sharing the same viz_channel share a
        #     median-of-per-tile-bounds clim (viz_channel_clims). A tile
        #     that's dramatically brighter / dimmer than the reporter's
        #     median gets its own (vmin, vmax) via slot_override and a
        #     red corner marker.
        override = slot_override.get(in_idx)
        if channel in PHASE_CHANNEL_CANDIDATES:
            vmin, vmax = PHASE_CONTRAST
        elif override is not None:
            vmin, vmax = override["vmin"], override["vmax"]
        elif viz_ch is not None and viz_ch in viz_channel_clims:
            vmin, vmax = viz_channel_clims[viz_ch]
        else:
            vmin, vmax = None, None
        _render_cell(
            ax, data[ch_idx], mask, title,
            vmin=vmin, vmax=vmax,
            nuc_mask=nuc_masks_by_input.get(in_idx),
            nuc_pred=nuc_preds_by_input.get(in_idx),
        )
        if override is not None:
            # Draw the outlier marker as a FIGURE-level text artist so it
            # stays as crisp vector text in the PDF (axes-level text inside
            # an imshow region can get rasterized with the image layer and
            # come out low-res).
            # Anchored at top-left of the FOV, fully inside the axes so the
            # badge never overlaps the title row above. y=0.985 with
            # va="top" snaps the badge flush to the top-left corner
            # (was 0.92 — visibly low). Alpha 0.5 on the white backing
            # box per style request.
            fig_ax = ax.figure
            x_disp = ax.transAxes.transform((0.02, 0.985))
            x_fig, y_fig = fig_ax.transFigure.inverted().transform(x_disp)
            fig_ax.text(
                x_fig, y_fig, override["marker"],
                ha="left", va="top",
                fontsize=9, fontweight="bold", color=override["color"],
                bbox=dict(
                    facecolor="white", edgecolor=override["color"],
                    linewidth=0.8, boxstyle="round,pad=0.12",
                    alpha=0.5,
                ),
                zorder=1000,
            )

    for r in range(_n_total_rows):
        for c in range(N_COLS):
            if (r, c) not in rendered:
                _render_cell(axes[r, c], None, None, "")

    # Per-gene row labels (rows 2..N alternate KO/NTC for the K viz_channels).
    row_labels = [f"Phase2D\n{gene} KO", "Phase2D\nNTC"]
    for pair in fluor_pairs[:_n_fluor]:
        vc_raw = pair.get("viz_channel") or "fluor"
        vc = _strip_dye_trailer(vc_raw)
        if pair.get("skipped"):
            reason = pair.get("skip_reason") or "skipped"
            row_labels.append(f"{vc}\n{gene} KO\n[{reason}]")
            row_labels.append(f"{vc}\nNTC\n[{reason}]")
        else:
            row_labels.append(f"{vc}\n{gene} KO")
            row_labels.append(f"{vc}\nNTC")
    while len(row_labels) < _n_total_rows:
        row_labels.append("")
    channel_colors = channel_colors or {}
    fluor_row_colors = []
    for pair in fluor_pairs[:_n_fluor]:
        vc = pair.get("viz_channel")
        c = channel_colors.get(vc) if vc else None
        fluor_row_colors.append(c or "black")
    while len(fluor_row_colors) < _n_fluor:
        fluor_row_colors.append("black")
    for r, label in enumerate(row_labels):
        if r < 2:
            label_color = "black"
        else:
            label_color = fluor_row_colors[(r - 2) // 2]
        axes[r, 0].set_ylabel(
            label, fontsize=8, fontweight="bold", rotation=90, labelpad=10,
            color=label_color,
        )

    # Thin horizontal separator lines between each (KO + NTC) pair.
    # Generalized to the dynamic row count: pair separators sit above
    # the start of each fluor pair (rows 2, 4, 6, ... → separator under
    # the NTC row of the pair above).
    #
    # The line is positioned at a fixed small offset below the BOTTOM
    # of the FOVs above it — independent of the inter-pair gap size.
    # The asymmetric nested gridspec (SHAP variant) makes that gap
    # large enough for the line to sit clearly in the middle of the
    # whitespace, and the offset just nudges it off the FOV edge so
    # there's a visible ~half-line-width of clear pixels above.
    #
    # Variable-page-size mode (n_fluor != 3): the row positions don't
    # line up with the fixed 0.0025 fig-fraction offset that's tuned
    # for the 3-channel layout — the lines drift onto the FOV edges or
    # disappear into the gap. Disable them in that mode; the row
    # ylabels (Phase / channel name) already mark the pair boundaries.
    if _n_fluor == N_FLUOR_CHANNELS_PER_GENE:
        from matplotlib.lines import Line2D
        for above_row in range(1, _n_total_rows - 1, 2):
            bb_above = axes[above_row, 0].get_position()
            # Center-ish within the gap: with SHAP hspace=0.18 the inter-
            # row gap is ≈0.017 fig-frac, the title eats ≈0.013 at the
            # bottom of that gap, leaving ≈0.004 of clear space above the
            # title. 0.0025 below upper FOV bottom puts the line just past
            # the midpoint of that clear space — clear of both the FOV
            # above and the title below.
            y_sep = bb_above.y0 - 0.0025
            x0 = bb_above.x0
            x1 = axes[above_row, -1].get_position().x1
            fig.add_artist(Line2D(
                [x0, x1], [y_sep, y_sep],
                transform=fig.transFigure,
                color="black", linewidth=1.4, alpha=0.8, zorder=200,
            ))

    row_names = {
        ROW_PHASE_KO: "phase-KO",
        ROW_PHASE_NTC: "phase-NTC",
    }
    for i, pair in enumerate(fluor_pairs[:_n_fluor]):
        vc = pair.get("viz_channel") or f"ch{i+1}"
        row_names[_fluor_ko_row(i)] = f"{vc} KO"
        row_names[_fluor_ntc_row(i)] = f"{vc} NTC"
    short_rows = [
        (r, row_fills[r]) for r in range(_n_total_rows)
        if r not in skipped_rows and row_fills[r] < N_COLS
    ]

    if strict and (failures or short_rows):
        lines = [
            f"[{gene}] {_n_total_rows * N_COLS} slots expected; "
            f"{len(failures)} failed + {len(short_rows)} short rows:"
        ]
        for (r, fill) in short_rows:
            lines.append(
                f"  [{row_names.get(r, r)}] only {fill}/{N_COLS} cells available upstream"
            )
        for (r, c, reason) in failures:
            lines.append(f"  [{row_names.get(r, r)} col {c}] {reason}")
        plt.close(fig)
        raise RuntimeError("\n".join(lines))

    if post_render_hook is not None:
        post_render_hook(fig, axes)

    return fig


# ──────────────────────────────────────────────────────────────────────────
# Worker: render a single gene's page
# ──────────────────────────────────────────────────────────────────────────
def _worker_render_gene(task):
    """Render one gene to its own PDF file. Top-level for pickling."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: F401

    gene = task["gene"]
    try:
        def _as_df(payload):
            return pd.DataFrame(payload) if payload else pd.DataFrame()

        phase_rows = _as_df(task["phase_rows"])
        ntc_phase_rows = _as_df(task["ntc_phase_rows"])
        fluor_pairs_raw = task.get("fluor_pairs") or []
        fluor_pairs = [
            {
                "viz_channel": p.get("viz_channel"),
                "fluor_rows": _as_df(p.get("fluor_rows")),
                "ntc_fluor_rows": _as_df(p.get("ntc_fluor_rows")),
            }
            for p in fluor_pairs_raw
        ]
        chad_cluster = task.get("chad_cluster")
        chad_supercluster = task.get("chad_supercluster")
        uniprot_function = task.get("uniprot_function")
        eval_acc = task.get("eval_acc")
        chad_members = task.get("chad_members")
        chad_description = None
        mAP_all_combined = None
        mAP_metric_label = None
        # The SHAP wrapper plants per-channel colors (Phase + 3 fluor),
        # the chad_members fallback, the one-line functional description,
        # the all-markers-combined mAP score, and a metric-type label
        # ("distinct" for gene-level, "consist" for complex-level) so
        # the title badge prefixes the score with the right metric name.
        channel_colors = None
        shap_payload = task.get("shap_data") or {}
        if isinstance(shap_payload, dict):
            channel_colors = shap_payload.get("channel_colors")
            if not chad_members:
                chad_members = shap_payload.get("chad_members")
            chad_description = shap_payload.get("chad_description")
            mAP_all_combined = shap_payload.get("mAP_all_combined")
            mAP_metric_label = shap_payload.get("mAP_metric_label")
        crop_size = task["crop_size"]
        load_pad_factor = task.get("load_pad_factor", 1.0)
        strict = task.get("strict", True)
        nuclear_overlay = task.get("nuclear_overlay", "none")
        dpi = task["dpi"]
        out_path = Path(task["out_path"])

        store_cache = StoreCache()
        exps = set()
        for df in (phase_rows, ntc_phase_rows):
            if not df.empty:
                exps.update(df["experiment"].astype(str).unique())
        for pair in fluor_pairs:
            for df in (pair["fluor_rows"], pair["ntc_fluor_rows"]):
                if not df.empty:
                    exps.update(df["experiment"].astype(str).unique())
        for exp in exps:
            store_cache.get(exp)

        # Optional side-panel hook (e.g. SHAP lollipops): the caller stuffs a
        # fully-qualified function name into the task; we import it here in
        # the worker process and let it return (fig_factory, post_render_hook)
        # given the task dict (so the hook can read per-gene side data).
        fig_factory = None
        post_render_hook = None
        hook_fqn = task.get("hook_factory_fqn")
        if hook_fqn:
            napari_dir = str(Path(__file__).resolve().parent)
            if napari_dir not in sys.path:
                sys.path.insert(0, napari_dir)
            import importlib
            mod_name, fn_name = hook_fqn.rsplit(".", 1)
            mod = importlib.import_module(mod_name)
            factory_fn = getattr(mod, fn_name)
            fig_factory, post_render_hook = factory_fn(task)

        fig = render_gene_page(
            gene=gene,
            phase_rows=phase_rows,
            ntc_phase_rows=ntc_phase_rows,
            fluor_pairs=fluor_pairs,
            chad_cluster=chad_cluster,
            chad_supercluster=chad_supercluster,
            uniprot_function=uniprot_function,
            store_cache=store_cache,
            crop_size=crop_size,
            load_pad_factor=load_pad_factor,
            strict=strict,
            fig_factory=fig_factory,
            post_render_hook=post_render_hook,
            eval_acc=eval_acc,
            channel_colors=channel_colors,
            chad_members=chad_members,
            chad_description=chad_description,
            mAP_all_combined=mAP_all_combined,
            mAP_metric_label=mAP_metric_label,
            nuclear_overlay=nuclear_overlay,
            n_fluor_channels=task.get("n_fluor_channels"),
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Ensure the rasterized tile resolution is >= crop_size so we never
        # downsample source pixels. Each tile is ~1.9" wide; 2.25" tall.
        min_dpi = int(np.ceil(crop_size / 1.9)) + 5
        effective_dpi = max(dpi, min_dpi)
        fig.savefig(str(out_path), format="pdf", dpi=effective_dpi)
        plt.close(fig)
        return (gene, str(out_path), None)
    except Exception as e:
        import traceback
        return (gene, None, f"{type(e).__name__}: {e}\n{traceback.format_exc()}")


# ──────────────────────────────────────────────────────────────────────────
# PDF merge (pdfunite, chunked to avoid ARG_MAX)
# ──────────────────────────────────────────────────────────────────────────
def _merge_pdfs(input_paths, output_path, chunk_size=5000, max_parallel=4):
    """Concatenate many PDFs with pdfunite, chunked + parallelized when needed.

    `chunk_size=5000` is well under ARG_MAX (~25K paths on Linux) so a
    1000-gene atlas merges in a single pdfunite call (no intermediate files).
    When chunking is required, chunk merges run in parallel (I/O-bound).
    """
    input_paths = [str(p) for p in input_paths]
    output_path = str(output_path)
    if not input_paths:
        raise RuntimeError("No PDFs to merge")

    if len(input_paths) <= chunk_size:
        subprocess.run(["pdfunite", *input_paths, output_path], check=True)
        return

    from concurrent.futures import ThreadPoolExecutor

    with tempfile.TemporaryDirectory() as td:
        jobs = []
        for i in range(0, len(input_paths), chunk_size):
            chunk = input_paths[i : i + chunk_size]
            co = str(Path(td) / f"chunk_{i // chunk_size:04d}.pdf")
            jobs.append((chunk, co))

        def _one(job):
            chunk, co = job
            subprocess.run(["pdfunite", *chunk, co], check=True)
            return co

        with ThreadPoolExecutor(max_workers=max_parallel) as ex:
            chunk_outs = list(ex.map(_one, jobs))

        if len(chunk_outs) > chunk_size:
            _merge_pdfs(chunk_outs, output_path,
                        chunk_size=chunk_size, max_parallel=max_parallel)
        else:
            subprocess.run(["pdfunite", *chunk_outs, output_path], check=True)


# ──────────────────────────────────────────────────────────────────────────
# Task building (shared between local + SLURM paths)
# ──────────────────────────────────────────────────────────────────────────
_YAML_CMAP_CACHE = None


def _parse_ops_channel_maps_yaml(yaml_path=None):
    """Parse ops_channel_maps.yaml directly into {ops_prefix: {zarr_channel_name: label}}.

    Bypasses `OpsDataset.channel_map_data`, which drops every CP / 4i channel
    for experiments whose yaml entry ends with a metadata dict (cell_painting
    / four_i) that lacks a `channel_name` field — leaving only `{BF: Phase}`.
    Parsing the yaml ourselves keeps every `channel_name + label` pair.
    """
    import yaml
    if yaml_path is None:
        try:
            # Use any experiment just to discover the canonical path.
            yaml_path = Path(OpsDataset(resolve_experiment_name("ops0107_20251208")).channel_maps)
        except Exception:
            yaml_path = Path(
                "/hpc/projects/intracellular_dashboard/fast_ops/configs/ops_channel_maps.yaml"
            )
    try:
        with open(yaml_path) as f:
            data = yaml.safe_load(f) or {}
    except Exception as e:
        print(f"  [channel-map yaml] failed to load {yaml_path}: {e}", flush=True)
        return {}
    result = {}
    for exp_prefix, entries in data.items():
        if not isinstance(entries, list):
            continue
        cm = {}
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            ch = entry.get("channel_name")
            lbl = entry.get("label")
            if ch and lbl:
                cm[str(ch)] = str(lbl)
        if cm:
            result[str(exp_prefix)] = cm
    return result


def _get_yaml_channel_map_cache():
    """Lazy per-process singleton so workers only parse the yaml once."""
    global _YAML_CMAP_CACHE
    if _YAML_CMAP_CACHE is None:
        _YAML_CMAP_CACHE = _parse_ops_channel_maps_yaml()
    return _YAML_CMAP_CACHE


def _channel_map_for_experiment(experiment):
    """{zarr_channel_name: label} for this experiment, parsed from yaml directly
    (correct for CP / 4i experiments, unlike OpsDataset.channel_map_data)."""
    y = _get_yaml_channel_map_cache()
    # yaml key is the 'ops0094' prefix; experiment names come through as 'ops0094_20251217'.
    prefix = experiment.split("_")[0]
    return dict(y.get(prefix, {}))


def _build_experiment_channel_map_cache(experiments):
    """Load channel_map per experiment from the yaml (direct parse).
    Returns dict: experiment -> {zarr_channel -> biological_marker}.
    """
    cache = {}
    for exp in experiments:
        cache[exp] = _channel_map_for_experiment(exp)
    return cache


def _cm_has_viz_channel(cm, viz_channel):
    """True iff a pre-loaded channel_map dict contains `viz_channel` (normalized)."""
    if not cm:
        return False
    viz_norm = _norm_channel(viz_channel)
    for _zarr_ch, bio_name in cm.items():
        if bio_name and _norm_channel(bio_name) == viz_norm:
            return True
    return False


def _experiment_images_viz_channel(experiment, viz_channel, channel_map_cache):
    """True iff this experiment's channel_map contains a marker that normalizes
    to `viz_channel`. Used to drop cells whose experiment never imaged the marker
    their CSV row is labeled with.
    """
    return _cm_has_viz_channel(channel_map_cache.get(experiment), viz_channel)


def _load_and_filter_csvs(phase_csv, fluor_csv, genes_subset=None,
                          max_genes=None, skip=0, ntc_max_experiments=60,
                          ntc_max_per_experiment=300, ntc_experiments=None,
                          skip_ntc=False,
                          supercategory_config_path=DEFAULT_SUPERCATEGORY_CONFIG,
                          eval_csv=None, eval_n_cells=100, threshold=0.0,
                          aggregation_level="gene",
                          ntc_pma_phase_csv=None,
                          ntc_pma_fluor_csv=None):
    phase_df = pd.read_csv(phase_csv)
    fluor_df = pd.read_csv(fluor_csv)
    # CHAD complex level: relabel `gene` from source-gene → predicted_class
    # (the complex name) so all downstream per-gene grouping renders one
    # page per complex, with cell crops sampled across all member sgRNAs.
    # The CHAD attention CSVs (pma_top_*_chad_v1.csv) carry both `gene` and
    # `predicted_class`; here we just swap the columns.
    if aggregation_level == "complex":
        for _df in (phase_df, fluor_df):
            if "predicted_class" not in _df.columns:
                raise SystemExit(
                    "--aggregation-level complex requires a 'predicted_class' "
                    "column in the attention CSVs (pma_top_*_chad_v1.csv "
                    "supplies it). Pass the chad_v1 --phase-csv / --fluor-csv."
                )
            _df["gene"] = _df["predicted_class"].astype(str)
    # v3 schema: rename `channel` -> `viz_channel` for downstream compat.
    for _df in (phase_df, fluor_df):
        if "viz_channel" not in _df.columns and "channel" in _df.columns:
            _df.rename(columns={"channel": "viz_channel"}, inplace=True)
    # Keep 3x the row budget so per-gene dedup has fallback cells to pull
    # forward when duplicate cells (same segmentation) appear in the top N.
    phase_df = phase_df[phase_df["rank"] <= N_COLS * 3].copy()
    if "rank_type" in fluor_df.columns:
        fluor_df = fluor_df[fluor_df["rank_type"] == "top"].copy()
    # No channel_rank cap: keep all ranks so dedup has fallback cells available.

    # Drop cells with missing / non-positive segmentation IDs: BaseDataset
    # would return empty masks for them and tiles would render blank.
    for _name, _df in (("phase", phase_df), ("fluor", fluor_df)):
        seg_col = "segmentation" if "segmentation" in _df.columns else None
        if seg_col is None:
            continue
        pre = len(_df)
        seg_ok = pd.to_numeric(_df[seg_col], errors="coerce") > 0
        _df.drop(_df.index[~seg_ok], inplace=True)
        dropped = pre - len(_df)
        if dropped:
            print(
                f"  [{_name}] dropped {dropped:,}/{pre:,} rows with "
                f"missing/non-positive segmentation",
                flush=True,
            )

    # Drop fluor rows whose experiment didn't actually image the viz_channel.
    # Alex's CSV pools cells across a cohort; some experiments in the cohort
    # imaged a different marker, so they'd render as n/a if we kept them.
    all_exps = sorted(
        set(phase_df["experiment"].dropna().unique())
        | set(fluor_df["experiment"].dropna().unique())
    )
    print(f"  Loading channel_map for {len(all_exps)} experiments...", flush=True)
    channel_map_cache = _build_experiment_channel_map_cache(all_exps)

    pre_n = len(fluor_df)
    keep = fluor_df.apply(
        lambda r: _experiment_images_viz_channel(
            r["experiment"], r["viz_channel"], channel_map_cache,
        ),
        axis=1,
    )
    fluor_df = fluor_df[keep].copy()
    dropped = pre_n - len(fluor_df)
    if dropped:
        print(
            f"  Dropped {dropped:,}/{pre_n:,} fluor rows whose experiment "
            f"didn't image the labeled viz_channel",
            flush=True,
        )

    # Keep all filtered cells so per-(gene, viz_channel) dedup has fallback
    # rows to pull forward when duplicates are dropped. _build_tasks picks the
    # top N_COLS unique cells per viz_channel downstream.

    if genes_subset:
        genes = sorted(set(genes_subset))
    else:
        genes = sorted(
            set(phase_df["gene"].dropna()) | set(fluor_df["gene"].dropna())
        )
    if skip:
        genes = genes[skip:]
    if max_genes:
        genes = genes[:max_genes]

    # Per-gene attention-model accuracy from Alex's eval CSV. Used for both
    # (a) the optional --threshold filter, dropping low-confidence genes from
    # the atlas, and (b) the per-page bold accuracy annotation in the title.
    eval_acc = _load_eval_accuracy(eval_csv, eval_n_cells) if eval_csv else {}
    # Per-complex AUROC (from the auroc_per_reporter_guide.csv at the repo
    # root, mean across reporters) — folded into eval_acc as the `auroc`
    # key so the existing render path picks it up alongside top1_acc.
    # The CSV's `complex` field uses bare names ("CCAN", "DNA Replication"),
    # but eval_acc is keyed by the prefixed CHAD-YAML names ("subu CCAN",
    # "GOI KRAS"). We normalize both sides (lowercase, strip "subu "/"goi "
    # prefixes, collapse separators) to bridge them.
    # Robust to script moves: find the monorepo root via the shared helper,
    # then locate the auroc CSV under coding_exps/.
    from ops_utils.data.filesystem import find_monorepo_root
    try:
        auroc_csv = find_monorepo_root(Path(__file__)) / "coding_exps" / "auroc_per_reporter_guide.csv"
    except FileNotFoundError:
        auroc_csv = Path("/nonexistent")
    if auroc_csv.exists():
        try:
            adf = pd.read_csv(auroc_csv)
            def _norm(s: str) -> str:
                s = str(s).strip().lower()
                for prefix in ("subu ", "goi ", "ncc "):
                    if s.startswith(prefix):
                        s = s[len(prefix):]
                return s.replace("/", " ").replace("-", " ").strip()
            au_norm = {
                _norm(cx): float(v)
                for cx, v in adf.groupby("complex")["auroc"].mean().items()
            }
            matched = 0
            for key in list(eval_acc.keys()):
                v = au_norm.get(_norm(key))
                if v is not None:
                    eval_acc[key]["auroc"] = v
                    matched += 1
            print(f"  [auroc] joined {matched}/{len(eval_acc)} complexes "
                  f"({len(au_norm)} in {auroc_csv.name})", flush=True)
        except Exception as e:
            print(f"  [auroc] read failed: {e!r}; skipping AUROC badge",
                  flush=True)
    if eval_acc and threshold and threshold > 0:
        before = len(genes)
        # Pass if EITHER phase or fluor classifier clears the threshold —
        # the two modalities are highly correlated but a small handful of
        # genes are only readable in one. NaN-safe via `or 0.0`.
        def _gene_max_acc(g):
            ea = eval_acc.get(g, {})
            p = ea.get("top1_acc_phase", ea.get("top1_acc", 0.0)) or 0.0
            f = ea.get("top1_acc_fluor", 0.0) or 0.0
            return max(p, f)
        genes = [g for g in genes if _gene_max_acc(g) >= threshold]
        dropped = before - len(genes)
        print(f"  --threshold {threshold}: kept {len(genes):,}/{before:,} genes "
              f"with max(top1_acc_phase, top1_acc_fluor) >= {threshold} at "
              f"n_cells={eval_n_cells} (dropped {dropped:,})", flush=True)

    # Build NTC pool. Two sources, in priority:
    #   1. PMA NTC CSV (high-attention NTC cells) — when paths to
    #      `pma_top_*_cells_chad_ntc_v3.csv` are provided. The model's
    #      attention model ranked NTCs, so we use those rankings to pick
    #      the on-page NTC strip — same cohort the violin BG uses, so
    #      "what the model thinks is a discriminative NTC" is what
    #      readers see in the FOVs.
    #   2. Legacy linked_results — random NTC sampling across a
    #      channel-aware experiment pool. Used when PMA NTC CSV paths
    #      aren't provided.
    if skip_ntc:
        ntc_pool = pd.DataFrame()
    else:
        ntc_pool = pd.DataFrame()
        if ntc_pma_phase_csv or ntc_pma_fluor_csv:
            ntc_pool = _load_ntc_pool_from_pma(
                ntc_pma_phase_csv, ntc_pma_fluor_csv,
            )
            if not ntc_pool.empty:
                print(
                    f"  NTC pool (high-attention PMA): {len(ntc_pool):,} cells "
                    f"from {ntc_pool['experiment'].nunique()} experiments",
                    flush=True,
                )
        if ntc_pool.empty:
            if ntc_experiments:
                ntc_exps = list(ntc_experiments)
            else:
                ntc_exps = _pick_ntc_experiments(
                    fluor_df=fluor_df,
                    phase_df=phase_df,
                    channel_map_cache=channel_map_cache,
                    target_total=ntc_max_experiments,
                    per_channel=3,
                )
            print(
                f"  Loading NTC pool from {len(ntc_exps)} experiments "
                f"(channel-aware, max {ntc_max_per_experiment} cells/exp)...",
                flush=True,
            )
            ntc_pool = _build_ntc_pool(ntc_exps, max_per_exp=ntc_max_per_experiment)
            print(f"  NTC pool (random linked_results): {len(ntc_pool):,} cells",
                  flush=True)

        # Extend channel_map_cache to cover any NTC experiments we haven't seen.
        ntc_exps_set = set(ntc_pool["experiment"].unique()) if not ntc_pool.empty else set()
        new_exps = [e for e in ntc_exps_set if e not in channel_map_cache]
        if new_exps:
            channel_map_cache.update(_build_experiment_channel_map_cache(new_exps))

    # CHAD annotations: gene → cluster, gene → supercluster, AND
    # cluster → member genes (used for the "Members (N): ..." page-
    # header line on complex-aggregation pages).
    print(f"  Loading CHAD annotations from {supercategory_config_path}...", flush=True)
    gene_to_cluster, gene_to_supercluster, cluster_to_genes = _load_chad_annotations(
        supercategory_config_path
    )
    print(
        f"  CHAD: {len(gene_to_cluster):,} genes with cluster, "
        f"{len(gene_to_supercluster):,} with supercluster, "
        f"{len(cluster_to_genes):,} clusters with member lists",
        flush=True,
    )

    gene_to_longname = _load_gene_longnames()
    print(f"  Gene LongNames: {len(gene_to_longname):,} entries", flush=True)

    # Fill in genes missing from the local longnames file via the
    # mygene.info API (NCBI Gene aggregator). Persistent cache means
    # this only fires the first time a new gene is seen; subsequent
    # runs hit the cache. Genuinely-missing genes (e.g. PRELID3B
    # before the first run) get a real description after one call;
    # the cache also stores negative results so dead-end lookups
    # don't repeat. Login-node-only — workers reuse the dict.
    gene_to_longname = _enrich_gene_longnames(gene_to_longname, genes)

    return (phase_df, fluor_df, ntc_pool, channel_map_cache,
            gene_to_cluster, gene_to_supercluster, gene_to_longname,
            cluster_to_genes, genes, eval_acc)


def _stable_seed(*parts):
    """Deterministic, cross-session 32-bit seed from arbitrary string parts.

    Python's built-in hash() is randomized per-process via PYTHONHASHSEED, so
    we use zlib.crc32 so the same (gene, suffix) always hashes to the same
    integer across runs/machines.
    """
    key = "|".join(str(p) for p in parts).encode("utf-8")
    return zlib.crc32(key) & 0xFFFFFFFF


XY_DEDUP_TOL_PX = 10  # two centroids within this distance (in pheno pixels)
# on the same (experiment, well) are treated as the same cell. Cells are
# typically ~30-80 px across, so ~10 px is well within "same cell" territory.


def _dedup_cells(df, xy_tol=XY_DEDUP_TOL_PX):
    """Drop duplicate cells within the pool. A cell is identified by
    (experiment, well, segmentation) OR by (experiment, well, xy-bin) so a
    resegmentation that gives the same cell a different seg_id still dedups.

    Keep first occurrence so the top-ranked copy is preserved.
    """
    if df is None or df.empty:
        return df

    # First pass: exact seg-id dedup
    pre = len(df)
    subset = [c for c in ("experiment", "well", "segmentation") if c in df.columns]
    if subset:
        df = df.drop_duplicates(subset=subset, keep="first")

    # Second pass: xy-bin dedup (tolerates different seg_ids for the same
    # physical cell when centroids fall in the same `xy_tol`-px bin).
    if (
        xy_tol
        and "x_pheno" in df.columns
        and "y_pheno" in df.columns
        and "experiment" in df.columns
        and "well" in df.columns
    ):
        df = df.copy()
        df["_xb"] = (df["x_pheno"].astype(float) / xy_tol).round().astype("Int64")
        df["_yb"] = (df["y_pheno"].astype(float) / xy_tol).round().astype("Int64")
        df = df.drop_duplicates(subset=["experiment", "well", "_xb", "_yb"], keep="first")
        df = df.drop(columns=["_xb", "_yb"])

    return df.reset_index(drop=True)


def _sample_ntc_rows(ntc_pool, gene, n, seed_suffix="", viz_channel=None):
    """Pick `n` NTC rows from the pool. Two pool layouts are supported:

    (1) **PMA top-attention pool** (legacy / top-attention variants).
        Schema: `channel`, `rank`, `rank_type∈{top,bottom}`. We sample
        randomly from `rank_type=='top'` (high-attention NTCs only)
        for the requested `viz_channel`, seeded per-(gene, channel)
        so each page gets a different cut of the same cohort.

    (2) **ntc_pick_cells output** (all-cells variants).
        Schema adds `target_gene` and `rank_type∈{top, ntc_ko_typical}`.
        Each row is keyed to a specific gene — `rank_type='top'` rows
        are the KO image rows (consumed by `--phase-csv`/`--fluor-csv`)
        and `rank_type='ntc_ko_typical'` rows are KO-resembling NTCs
        for that target gene's NTC strip. We filter to
        `target_gene==gene` AND `rank_type=='ntc_ko_typical'` —
        no further sampling needed (the picker already selected the
        top-N most KO-resembling NTCs).

    Detection is duck-typed on `target_gene` column presence — keeps
    both call sites unchanged.

    Older random-sample behavior (no `channel` column at all — legacy
    `_build_ntc_pool` source) is the final fallback.
    """
    if ntc_pool is None or ntc_pool.empty or n <= 0:
        return ntc_pool.iloc[0:0] if ntc_pool is not None else pd.DataFrame()

    # Picker pool: rows are tagged per target gene + role; no random
    # sampling — picker already ranked them. Take top-`n`.
    if "target_gene" in ntc_pool.columns:
        sub = ntc_pool[ntc_pool["target_gene"].astype(str) == str(gene)]
        if viz_channel is not None and "viz_channel" in sub.columns:
            sub_ch = sub[sub["viz_channel"].astype(str) == str(viz_channel)]
            if sub_ch.empty and str(viz_channel).lower() == "phase":
                sub_ch = sub[sub["viz_channel"].astype(str) == "Phase"]
            if not sub_ch.empty:
                sub = sub_ch
        if "rank_type" in sub.columns:
            sub = sub[sub["rank_type"].astype(str) == "ntc_ko_typical"]
        if not sub.empty:
            sort_col = "rank" if "rank" in sub.columns else None
            sub = sub.sort_values(sort_col) if sort_col else sub
            return sub.head(n).reset_index(drop=True)
        # No picker rows for this gene/channel — fall through to legacy
        # paths so the page still renders.

    has_rank = "rank" in ntc_pool.columns and "channel" in ntc_pool.columns
    if viz_channel is not None and has_rank:
        sub = ntc_pool[ntc_pool["channel"].astype(str) == str(viz_channel)]
        if sub.empty and str(viz_channel).lower() == "phase":
            # PMA phase CSV labels every row `channel="Phase2D"`; map the
            # atlas's lowercase "phase" to that.
            sub = ntc_pool[ntc_pool["channel"].astype(str) == "Phase2D"]
        # Restrict to the high-attention NTC pool (rank_type=='top') —
        # bottom-rank NTCs are model-deemed "uninteresting" and not what
        # we want to show as exemplars. Done only when the column is
        # present; older CSVs without it are unaffected.
        if not sub.empty and "rank_type" in sub.columns:
            top_sub = sub[sub["rank_type"].astype(str) == "top"]
            if not top_sub.empty:
                sub = top_sub
        if not sub.empty:
            rng = np.random.default_rng(
                _stable_seed(gene, f"{seed_suffix}:{viz_channel}")
            )
            k = min(n, len(sub))
            idx = rng.choice(len(sub), size=k, replace=False)
            return sub.iloc[idx].reset_index(drop=True)
        # No rows for this channel — fall through to whole-pool random
        # sample so the row still renders (legacy behavior).

    rng = np.random.default_rng(_stable_seed(gene, seed_suffix))
    k = min(n, len(ntc_pool))
    idx = rng.choice(len(ntc_pool), size=k, replace=False)
    return ntc_pool.iloc[idx].reset_index(drop=True)


def _pick_most_common_viz_channel(fluor_rows):
    """Return the viz_channel that appears most often in the top-N fluor rows.
    Ties broken by the highest total pma_attention for that channel.
    Returns None if no fluor rows.
    """
    if fluor_rows is None or fluor_rows.empty:
        return None
    ranked = (
        fluor_rows.groupby("viz_channel")
        .agg(count=("viz_channel", "size"), total_attn=("pma_attention", "sum"))
        .sort_values(["count", "total_attn"], ascending=False)
    )
    return ranked.index[0] if len(ranked) else None


def _build_tasks(phase_df, fluor_df, ntc_pool, channel_map_cache,
                 gene_to_cluster, gene_to_supercluster, gene_to_longname, genes,
                 crop_size, dpi, page_dir, strict=True,
                 extra_per_gene_data=None, hook_factory_fqn=None,
                 skip_cp_4i=False, eval_acc=None,
                 cluster_to_genes=None, aggregation_level="gene",
                 nuclear_overlay="none", load_pad_factor=1.0):
    # Dedup cells (same seg = same physical cell) so duplicate rows don't
    # steal slots; when one is dropped the next-ranked cell fills in.
    phase_by_gene = {
        g: _dedup_cells(grp.sort_values("rank")).head(N_COLS)
        for g, grp in phase_df.groupby("gene")
    }
    # Fluor: keep full per-gene subset so we can slice by viz_channel downstream.
    fluor_by_gene = {g: grp for g, grp in fluor_df.groupby("gene")}

    # Pre-compute: for each viz_channel, which NTC-pool experiments image it?
    ntc_exps_by_channel = {}
    if not ntc_pool.empty:
        for viz_ch in fluor_df["viz_channel"].dropna().unique():
            ntc_exps_by_channel[viz_ch] = {
                e for e in ntc_pool["experiment"].unique()
                if _cm_has_viz_channel(channel_map_cache.get(e), viz_ch)
            }

    # Per-gene viz_channel ordering: use Alex's channel_rank (1..3, lowest first)
    # so the atlas rows follow Alex's ordering of the gene's top reporters.
    gene_channel_ranks = {}
    for (g, viz_ch), grp in fluor_df.groupby(["gene", "viz_channel"]):
        cr = int(grp["channel_rank"].min()) if "channel_rank" in grp.columns else 99
        gene_channel_ranks.setdefault(g, []).append((cr, viz_ch))
    for g in gene_channel_ranks:
        gene_channel_ranks[g].sort()

    tasks = []
    for gene in genes:
        phase_rows = phase_by_gene.get(gene, phase_df.iloc[0:0])
        gene_fluor = fluor_by_gene.get(gene, fluor_df.iloc[0:0])

        # Phase-NTC must have a live-cell seg-ID (matches cell_seg mask).
        phase_pool = ntc_pool[
            pd.to_numeric(ntc_pool["segmentation"], errors="coerce") > 0
        ] if not ntc_pool.empty else ntc_pool
        ntc_phase_rows = _sample_ntc_rows(
            phase_pool, gene, N_COLS * NTC_OVERSAMPLE_FACTOR,
            seed_suffix="phase",
            viz_channel="Phase2D",
        )

        # Build up to 3 (viz_channel, ko_rows, ntc_rows) entries.
        # Default ordering is by Alex's `channel_rank` (1,2,3 — the
        # "best-attended" markers per gene). When the SHAP wrapper
        # planted a `viz_channel_override` for this gene (via
        # extra_per_gene_data["shap_data"]["viz_channel_override"]),
        # use that order instead so the image rows match the SHAP
        # bar's mAP-driven selection — and so the row labels' colors
        # line up with the bar's selected-channel colors.
        override = None
        if extra_per_gene_data and gene in extra_per_gene_data:
            shap_payload = extra_per_gene_data[gene].get("shap_data") or {}
            override = (shap_payload or {}).get("viz_channel_override")
        # Per-gene fluor-channel count. When atlas_shap supplies an
        # mAP-thresholded override list of length K (e.g. 1..10), this
        # gene's page renders K fluor row-pairs. Without an override,
        # fall back to the global N_FLUOR_CHANNELS_PER_GENE (3) for
        # backward compat with legacy callers / standalone atlas.py.
        if override:
            n_fluor_for_gene = max(1, min(len(override), 10))
            fluor_channels_ordered = list(override)[:n_fluor_for_gene]
        else:
            n_fluor_for_gene = N_FLUOR_CHANNELS_PER_GENE
            fluor_channels_ordered = [vc for _, vc in gene_channel_ranks.get(gene, [])][:n_fluor_for_gene]
        # Pad with None if the gene has fewer than expected viz_channels
        while len(fluor_channels_ordered) < n_fluor_for_gene:
            fluor_channels_ordered.append(None)

        fluor_pairs = []
        for viz_ch in fluor_channels_ordered:
            if viz_ch is None:
                fluor_pairs.append({
                    "viz_channel": None,
                    "fluor_rows": {},
                    "ntc_fluor_rows": {},
                })
                continue
            sub = gene_fluor[gene_fluor["viz_channel"] == viz_ch]
            ko_rows = _dedup_cells(
                sub.sort_values("pma_attention", ascending=False)
            ).head(N_COLS)
            valid_exps = ntc_exps_by_channel.get(viz_ch, set())
            # Detect CP/4i modality from BOTH the gene's KO experiments AND
            # the NTC-pool experiments — so `--skip-cp-4i` triggers correctly
            # even when the NTC pool is empty (--no-ntc) or when the cohort's
            # only CP/4i exposure for this channel is in the KO arm.
            is_fixed_mod = False
            ko_exps = sub["experiment"].dropna().astype(str).unique() if not sub.empty else []
            for e in (*ko_exps, *valid_exps):
                cm = channel_map_cache.get(e, {}) or {}
                for zch, lbl in cm.items():
                    if _norm_channel(lbl) == _norm_channel(viz_ch):
                        if str(zch).startswith(("CP1_", "CP2_", "4i_")):
                            is_fixed_mod = True
                        break
                if is_fixed_mod:
                    break

            if skip_cp_4i and is_fixed_mod:
                # Drop image rows for this viz_channel; render_gene_page reads
                # `skipped: True` to leave the row blank and exclude it from
                # the strict short-row check. SHAP / caption side panels (if
                # in use) still render normally — the SHAP signal doesn't
                # depend on the broken CP/4i coordinate space.
                fluor_pairs.append({
                    "viz_channel": viz_ch,
                    "fluor_rows": {},
                    "ntc_fluor_rows": {},
                    "skipped": True,
                    "skip_reason": "CP/4i — coord space broken",
                })
                continue

            if valid_exps:
                pool_subset = ntc_pool[ntc_pool["experiment"].isin(valid_exps)]
                # Filter to rows whose RELEVANT seg-ID is valid so the
                # sampled NTC will always have a mask to match against.
                if is_fixed_mod:
                    pool_subset = pool_subset[
                        pd.to_numeric(pool_subset["fluor_segmentation"], errors="coerce") > 0
                    ]
                else:
                    pool_subset = pool_subset[
                        pd.to_numeric(pool_subset["segmentation"], errors="coerce") > 0
                    ]
            else:
                pool_subset = ntc_pool.iloc[0:0]

            ntc_rows_sub = _sample_ntc_rows(
                pool_subset, gene, N_COLS * NTC_OVERSAMPLE_FACTOR,
                seed_suffix=f"fluor_{viz_ch}",
                viz_channel=viz_ch,
            )
            if not ntc_rows_sub.empty:
                ntc_rows_sub = ntc_rows_sub.copy()
                ntc_rows_sub["viz_channel"] = viz_ch
                if is_fixed_mod:
                    # Use the CP/4i seg-ID so the mask look-up against
                    # cp_cell_seg / 4i_cell_seg actually finds the cell.
                    ntc_rows_sub["segmentation"] = ntc_rows_sub["fluor_segmentation"]
            fluor_pairs.append({
                "viz_channel": viz_ch,
                "fluor_rows": ko_rows.to_dict("list"),
                "ntc_fluor_rows": ntc_rows_sub.to_dict("list"),
            })

        # CHAD complex member-gene list — at complex-aggregation level,
        # `gene` IS a complex name and the page-header line shows
        # "Members (N): GENE1, GENE2, …" sourced from the CHAD YAML.
        # None at gene-level (the existing uniprot_function path runs
        # instead).
        chad_members = None
        if aggregation_level == "complex" and cluster_to_genes:
            chad_members = cluster_to_genes.get(gene)

        task = {
            "gene": gene,
            "phase_rows": phase_rows.to_dict("list"),
            "ntc_phase_rows": ntc_phase_rows.to_dict("list"),
            "fluor_pairs": fluor_pairs,
            "n_fluor_channels": n_fluor_for_gene,
            "chad_cluster": gene_to_cluster.get(gene),
            "chad_supercluster": gene_to_supercluster.get(gene),
            "uniprot_function": gene_to_longname.get(gene),
            "chad_members": chad_members,
            "eval_acc": (eval_acc or {}).get(gene),
            "strict": strict,
            "nuclear_overlay": nuclear_overlay,
            "crop_size": crop_size,
            "load_pad_factor": load_pad_factor,
            "dpi": dpi,
            "out_path": str(page_dir / f"{gene}.pdf"),
        }
        if extra_per_gene_data and gene in extra_per_gene_data:
            task.update(extra_per_gene_data[gene])
        if hook_factory_fqn:
            task["hook_factory_fqn"] = hook_factory_fqn
        tasks.append(task)
    return tasks


# ──────────────────────────────────────────────────────────────────────────
# Local (in-process) pipeline
# ──────────────────────────────────────────────────────────────────────────
def _run_atlas_local(phase_csv, fluor_csv, output, crop_size, genes_subset,
                     max_genes, skip, dpi, workers, keep_pages,
                     ntc_max_experiments=60, ntc_max_per_experiment=300,
                     skip_ntc=False, strict=True,
                     extra_per_gene_data=None, hook_factory_fqn=None,
                     skip_cp_4i=False,
                     eval_csv=None, eval_n_cells=100, threshold=0.0,
                     aggregation_level="gene",
                     nuclear_overlay="none",
                     ntc_pma_phase_csv=None,
                     ntc_pma_fluor_csv=None,
                     load_pad_factor=1.0):
    from ops_utils.hpc.resource_manager import get_optimal_workers

    output = Path(output)
    print(f"Loading CSVs...", flush=True)
    (phase_df, fluor_df, ntc_pool, channel_map_cache,
     gene_to_cluster, gene_to_supercluster, gene_to_longname,
     cluster_to_genes, genes, eval_acc) = _load_and_filter_csvs(
        phase_csv, fluor_csv, genes_subset, max_genes, skip,
        ntc_max_experiments=ntc_max_experiments,
        ntc_max_per_experiment=ntc_max_per_experiment,
        skip_ntc=skip_ntc,
        eval_csv=eval_csv, eval_n_cells=eval_n_cells, threshold=threshold,
        aggregation_level=aggregation_level,
        ntc_pma_phase_csv=ntc_pma_phase_csv,
        ntc_pma_fluor_csv=ntc_pma_fluor_csv,
    )
    print(f"Rendering {len(genes):,} genes locally", flush=True)

    if workers is None:
        workers = get_optimal_workers(
            use_gpu=False, model_ram_gb=0.3, data_ram_gb=1.0, verbose=False,
        )
    print(f"Workers: {workers}", flush=True)

    output.parent.mkdir(parents=True, exist_ok=True)
    page_dir = output.parent / f"{output.stem}_pages"
    page_dir.mkdir(exist_ok=True)

    tasks = _build_tasks(phase_df, fluor_df, ntc_pool, channel_map_cache,
                         gene_to_cluster, gene_to_supercluster, gene_to_longname, genes,
                         crop_size, dpi, page_dir, strict=strict,
                         extra_per_gene_data=extra_per_gene_data,
                         hook_factory_fqn=hook_factory_fqn,
                         skip_cp_4i=skip_cp_4i,
                         eval_acc=eval_acc,
                         cluster_to_genes=cluster_to_genes,
                         aggregation_level=aggregation_level,
                         nuclear_overlay=nuclear_overlay,
                         load_pad_factor=load_pad_factor)

    success = {}
    fail = {}
    t0 = time.time()

    if workers <= 1:
        for i, t in enumerate(tasks):
            gene, path, err = _worker_render_gene(t)
            if err:
                fail[gene] = err
                print(f"  [{gene}] FAILED:\n{err}", flush=True)
            else:
                success[gene] = path
            if (i + 1) % 10 == 0 or (i + 1) == len(tasks):
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                eta = (len(tasks) - i - 1) / rate / 60 if rate > 0 else 0
                print(
                    f"  {i+1}/{len(tasks)} ({rate:.2f}/s, eta {eta:.1f}m, fail={len(fail)})",
                    flush=True,
                )
    else:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
            futures = {ex.submit(_worker_render_gene, t): t["gene"] for t in tasks}
            for i, fut in enumerate(as_completed(futures)):
                gene, path, err = fut.result()
                if err:
                    fail[gene] = err
                    print(f"  [{gene}] FAILED: {err.splitlines()[0]}", flush=True)
                else:
                    success[gene] = path
                if (i + 1) % 50 == 0 or (i + 1) == len(tasks):
                    elapsed = time.time() - t0
                    rate = (i + 1) / elapsed if elapsed > 0 else 0
                    eta = (len(tasks) - i - 1) / rate / 60 if rate > 0 else 0
                    print(
                        f"  {i+1}/{len(tasks)} ({rate:.1f}/s, eta {eta:.1f}m, fail={len(fail)})",
                        flush=True,
                    )

    paths = [success[g] for g in genes if g in success]
    if not paths:
        print("No PDFs to merge.", flush=True)
        return
    print(f"Merging {len(paths)} per-gene PDFs...", flush=True)
    _merge_pdfs(paths, output)
    print(f"Wrote {output}", flush=True)
    if not keep_pages:
        shutil.rmtree(page_dir, ignore_errors=True)
    print(f"Done. {len(success)}/{len(genes)} pages ({len(fail)} failures).", flush=True)


# ──────────────────────────────────────────────────────────────────────────
# SLURM-side top-level functions (called via submitit)
# ──────────────────────────────────────────────────────────────────────────
def _slurm_render_chunk(chunk_pkl_path):
    """Array-task entry point. Renders every gene in the pickled task list.

    Prints the FULL error text for each failed gene and writes a summary JSON
    next to the chunk pickle. RAISES at the end if any gene failed so that
    submitit / SLURM reports the array task as failed (otherwise a chunk with
    a strict-mode gene failure still returns normally and the failure is
    invisible in the batch summary).
    """
    import matplotlib
    matplotlib.use("Agg")
    import json

    with open(chunk_pkl_path, "rb") as f:
        tasks = pickle.load(f)
    print(f"Loaded {len(tasks)} gene tasks from {chunk_pkl_path}", flush=True)

    results = []
    n_ok = 0
    failures = []  # list of (gene, full_error_text)
    for task in tasks:
        gene, path, err = _worker_render_gene(task)
        if err:
            failures.append((gene, err))
            print(f"  [{gene}] FAILED:\n{err}", flush=True)
            results.append({"gene": gene, "ok": False, "error": err})
        else:
            n_ok += 1
            print(f"  [{gene}] OK -> {path}", flush=True)
            results.append({"gene": gene, "ok": True, "path": path})

    # Per-chunk summary JSON alongside the pickle
    summary_path = Path(chunk_pkl_path).with_suffix(".result.json")
    try:
        with open(summary_path, "w") as f:
            json.dump({"ok": n_ok, "failed": len(failures), "results": results}, f, indent=2)
    except Exception as e:
        print(f"  [chunk] could not write summary {summary_path}: {e}", flush=True)

    print(f"Chunk done: {n_ok} ok, {len(failures)} failed", flush=True)
    if failures:
        # Raise a concise summary so SLURM/submitit marks this array task as
        # failed. The full per-gene errors are already printed above and in
        # the .result.json file, so the traceback itself stays short.
        names = ", ".join(g for g, _ in failures[:5])
        more = f" (+{len(failures)-5} more)" if len(failures) > 5 else ""
        raise RuntimeError(
            f"{len(failures)}/{len(tasks)} gene(s) failed in chunk "
            f"{Path(chunk_pkl_path).name}: {names}{more}"
        )
    return {"ok": n_ok, "failed": 0, "chunk": chunk_pkl_path}


def _slurm_merge(gene_list, page_dir, output_pdf, keep_pages):
    """Dependent merge-task entry point."""
    page_dir = Path(page_dir)
    output_pdf = Path(output_pdf)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    paths = [
        str(page_dir / f"{g}.pdf")
        for g in gene_list
        if (page_dir / f"{g}.pdf").exists()
    ]
    missing = [g for g in gene_list if not (page_dir / f"{g}.pdf").exists()]
    print(f"Merging {len(paths)} PDFs ({len(missing)} missing)", flush=True)
    if missing[:10]:
        print(f"  first missing: {missing[:10]}", flush=True)
    if not paths:
        raise RuntimeError("No per-gene PDFs to merge — all render tasks failed?")
    _merge_pdfs(paths, output_pdf)
    print(f"Wrote {output_pdf}", flush=True)
    if not keep_pages:
        shutil.rmtree(page_dir, ignore_errors=True)
    return str(output_pdf)


# ──────────────────────────────────────────────────────────────────────────
# SLURM submission (array + dependent merge, via submit_parallel_jobs)
# ──────────────────────────────────────────────────────────────────────────
def _submit_via_slurm(args, extra_per_gene_data=None, hook_factory_fqn=None):
    from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs

    output = Path(args.output)
    print(f"Loading CSVs and building per-gene tasks...", flush=True)
    (phase_df, fluor_df, ntc_pool, channel_map_cache,
     gene_to_cluster, gene_to_supercluster, gene_to_longname,
     cluster_to_genes, genes, eval_acc) = _load_and_filter_csvs(
        args.phase_csv, args.fluor_csv,
        genes_subset=args.genes, max_genes=args.max_genes, skip=args.skip,
        ntc_max_experiments=args.ntc_experiments,
        ntc_max_per_experiment=args.ntc_per_experiment,
        skip_ntc=args.no_ntc,
        eval_csv=getattr(args, "eval_csv", None),
        eval_n_cells=getattr(args, "eval_n_cells", 100),
        threshold=getattr(args, "threshold", 0.0),
        aggregation_level=getattr(args, "aggregation_level", "gene"),
        ntc_pma_phase_csv=getattr(args, "ntc_pma_phase_csv", None),
        ntc_pma_fluor_csv=getattr(args, "ntc_pma_fluor_csv", None),
    )
    print(f"  {len(genes):,} genes to render", flush=True)
    if not genes:
        print("Nothing to render. Exiting.")
        return

    work_dir = output.parent / f"{output.stem}_work"
    work_dir.mkdir(parents=True, exist_ok=True)
    page_dir = work_dir / "pages"
    page_dir.mkdir(exist_ok=True)
    chunks_dir = work_dir / "chunks"
    chunks_dir.mkdir(exist_ok=True)

    tasks = _build_tasks(phase_df, fluor_df, ntc_pool, channel_map_cache,
                         gene_to_cluster, gene_to_supercluster, gene_to_longname, genes,
                         args.crop_size, args.dpi, page_dir,
                         strict=args.strict,
                         extra_per_gene_data=extra_per_gene_data,
                         hook_factory_fqn=hook_factory_fqn,
                         skip_cp_4i=getattr(args, "skip_cp_4i", False),
                         eval_acc=eval_acc,
                         cluster_to_genes=cluster_to_genes,
                         aggregation_level=getattr(args, "aggregation_level", "gene"),
                         nuclear_overlay=getattr(args, "nuclear_overlay", "none"),
                         load_pad_factor=getattr(args, "load_pad_factor", 1.0))

    n_per = max(1, args.genes_per_task)
    chunk_groups = [tasks[i : i + n_per] for i in range(0, len(tasks), n_per)]
    print(f"  {len(chunk_groups)} array tasks ({n_per} genes/task)", flush=True)

    chunk_paths = []
    for i, chunk in enumerate(chunk_groups):
        p = chunks_dir / f"chunk_{i:05d}.pkl"
        with open(p, "wb") as f:
            pickle.dump(chunk, f)
        chunk_paths.append(p)

    array_jobs = [
        {
            "name": f"atlas_{output.stem}_chunk_{i:05d}",
            "func": _slurm_render_chunk,
            "kwargs": {"chunk_pkl_path": str(cp)},
            "metadata": {"chunk_index": i, "n_genes": len(chunk_groups[i])},
        }
        for i, cp in enumerate(chunk_paths)
    ]

    array_slurm_params = {
        "timeout_min": args.time_minutes_per_task,
        "slurm_partition": args.partition,
        "cpus_per_task": args.cpus_per_task,
        "mem": f"{args.mem_gb_per_task}G",
    }

    print(f"\nSubmitting render array ({len(array_jobs)} tasks)...\n", flush=True)
    render_result = submit_parallel_jobs(
        jobs_to_submit=array_jobs,
        experiment=f"attention_atlas_render_{output.stem}",
        slurm_params=array_slurm_params,
        log_dir=f"attention_atlas/{output.stem}",
        manifest_prefix=f"atlas_render_{output.stem}",
        wait_for_completion=True,
        verbose=True,
    )

    failed_render = render_result.get("failed") or []
    if failed_render:
        print(
            f"\nNote: {len(failed_render)} render task(s) failed; merge will "
            f"use whatever per-gene PDFs exist.",
            flush=True,
        )

    merge_slurm_params = {
        "timeout_min": max(30, args.time_minutes_per_task),
        "slurm_partition": args.partition,
        "cpus_per_task": 2,
        "mem": f"{max(16, args.mem_gb_per_task)}G",
    }
    merge_job = {
        "name": f"atlas_{output.stem}_merge",
        "func": _slurm_merge,
        "kwargs": {
            "gene_list": genes,
            "page_dir": str(page_dir),
            "output_pdf": str(output),
            "keep_pages": args.keep_pages,
        },
        "metadata": {"n_genes": len(genes)},
    }

    print(f"\nSubmitting merge job...\n", flush=True)
    merge_result = submit_parallel_jobs(
        jobs_to_submit=[merge_job],
        experiment=f"attention_atlas_merge_{output.stem}",
        slurm_params=merge_slurm_params,
        log_dir=f"attention_atlas/{output.stem}_merge",
        manifest_prefix=f"atlas_merge_{output.stem}",
        wait_for_completion=True,
        verbose=True,
    )

    print(f"\nAtlas generation complete.")
    print(f"  Render array: {render_result.get('base_job_id')}")
    print(f"  Merge job:    {merge_result.get('base_job_id')}")
    print(f"  Output:       {output}")


# ──────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────
def _make_arg_parser():
    """Build the shared argparse for both the bare atlas and the SHAP-augmented
    variant — keeping all flag definitions in one place."""
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _ATTN_V3 = Path("/hpc/projects/icd.fast.ops/models/alex_lin_attention/v3/attention_v3")
    ap.add_argument(
        "--phase-csv", type=Path,
        default=_ATTN_V3 / "pma_top_phase_cells_v3.csv",
        help="Top-phase attention CSV (default: v3/attention_v3/pma_top_phase_cells_v3.csv)",
    )
    ap.add_argument(
        "--fluor-csv", type=Path,
        default=_ATTN_V3 / "pma_top_fluorescent_cells_v3.csv",
        help="Top-fluor attention CSV (default: v3/attention_v3/pma_top_fluorescent_cells_v3.csv)",
    )
    ap.add_argument(
        "--eval-csv", type=Path,
        default=Path("/hpc/projects/icd.fast.ops/models/alex_lin_attention/"
                     "v3/attention_v3/cdino/cdino_eval_phase_50.csv"),
        help="Per-gene attention-model accuracy CSV from Alex's eval. Used to "
             "annotate the atlas page header with top1_acc and (optionally) "
             "filter low-confidence genes via --threshold. Default: "
             "v3/attention_v3/cdino_eval_phase_50.csv.",
    )
    ap.add_argument(
        "--eval-n-cells", type=int, default=100,
        help="Which n_cells subset of --eval-csv to use for the accuracy "
             "lookup (CSV has 100/200/500/1000). Default 100.",
    )
    ap.add_argument(
        "--threshold", type=float, default=0.0,
        help="Drop genes whose --eval-csv top1_acc is below this threshold "
             "(at --eval-n-cells). Default 0.0 (no filter).",
    )
    ap.add_argument(
        "--aggregation-level", choices=("gene", "complex"), default="gene",
        help="Render one page per gene (default) or one page per CHAD protein "
             "complex. At complex level: --phase-csv / --fluor-csv default to "
             "pma_top_*_chad_v1.csv (CHAD-trained attention model output); "
             "predicted_class column relabels `gene` to the complex name; cell "
             "crops naturally sample across complex members (since the CHAD "
             "attention model already selected representative cells per "
             "complex). --threshold filter is gene-level only and is no-op "
             "at complex level.",
    )
    ap.add_argument(
        "--output", type=Path,
        default=Path("/hpc/projects/icd.fast.ops/models/alex_lin_attention/atlas/attention_atlas.pdf"),
        help="Output PDF path (default: icd.fast.ops/models/alex_lin_attention/atlas/attention_atlas.pdf)",
    )
    ap.add_argument("--crop-size", type=int, default=200,
                    help="Cell-tile DISPLAY crop size in pixels. Default 200. "
                         "Atlas loads a wider window (--load-pad-factor × "
                         "crop_size) so the renderer can re-center each "
                         "tile on the cell's segmentation bbox instead of "
                         "the raw (x_pheno, y_pheno) position.")
    ap.add_argument("--load-pad-factor", type=float, default=1.5,
                    help="Multiplier for the underlying zarr load relative "
                         "to --crop-size. Default 1.5 → load 300² for a "
                         "200² display crop, leaving 50 px of slack on each "
                         "side so the bbox-center re-crop never falls "
                         "outside the loaded data. Set to 1.0 to disable "
                         "bbox re-centering (load = display = same size).")
    ap.add_argument("--genes", nargs="*", default=None,
                    help="Explicit gene list (default: all, alphabetical)")
    ap.add_argument("--max-genes", type=int, default=None,
                    help="Render only the first N genes after alphabetizing")
    ap.add_argument("--skip", type=int, default=0, help="Skip the first N genes")
    ap.add_argument(
        "--dpi", type=int, default=200,
        help="PDF DPI (default 200). Auto-bumped to avoid downsampling if "
             "crop_size exceeds what the default can fit in a cell tile.",
    )
    ap.add_argument("--workers", type=int, default=None,
                    help="Local workers for in-process mode (default: auto)")
    ap.add_argument("--delete-pages", dest="keep_pages", action="store_false",
                    default=True,
                    help="Delete the per-gene PDFs after merging "
                         "(default: keep them in <output>_work/pages/)")
    ap.add_argument("--no-strict", dest="strict", action="store_false",
                    default=True,
                    help="Render n/a placeholders instead of failing the page "
                         "when any cell can't be rendered (default: strict)")

    ntc = ap.add_argument_group("NTC row (row 3)")
    ntc.add_argument("--ntc-experiments", type=int, default=60,
                     help="Target # of NTC-pool experiments. Channel-aware: "
                          "guarantees >=3 experiments per unique viz_channel, "
                          "then fills to this number (default: 60).")
    ntc.add_argument("--ntc-per-experiment", type=int, default=300,
                     help="Max NTC cells to pull from each experiment (default: 300)")
    ntc.add_argument("--no-ntc", action="store_true",
                     help="Skip NTC row entirely (leave row 3 blank)")
    ap.add_argument(
        "--skip-cp-4i", dest="skip_cp_4i", action="store_true", default=False,
        help="Skip image rows for any fluor viz_channel that resolves to a CP "
             "or 4i zarr channel (those modalities currently have a broken "
             "coordinate space). Affected rows render as n/a with a "
             "'CP/4i — coord space broken' annotation; SHAP / caption side "
             "panels (when present) still render normally.",
    )
    ap.add_argument(
        "--nuclear-overlay", dest="nuclear_overlay",
        choices=("none", "seg", "vs", "both"),
        default="none",
        help="How to display the nucleus on each tile. Default: none. "
             "'seg' draws a thin blue outline traced from `nuclear_seg` "
             "(or CP/4i variants); "
             "'vs' renders the `nuclei_prediction` virtual-stain channel "
             "as a low-alpha blue glow — softer and avoids the hard outline; "
             "'both' overlays the VS glow with the seg contour on top — "
             "useful for verifying the seg labels actually match the "
             "virtual-stain signal.",
    )

    slurm = ap.add_argument_group("SLURM submission (default)")
    slurm.add_argument("--local", action="store_true",
                       help="Run in-process on this node instead of submitting to SLURM")
    slurm.add_argument("--genes-per-task", type=int, default=1,
                       help="Genes rendered per array task (default: 1)")
    slurm.add_argument("--partition", default="cpu")
    slurm.add_argument("--cpus-per-task", type=int, default=2)
    slurm.add_argument("--mem-gb-per-task", type=int, default=16)
    slurm.add_argument(
        "--time-minutes-per-task", type=int, default=15,
        help="Per-task SLURM timeout. Bumped from 5→15 because some genes "
             "consistently timed out at 5 min when the cache + violin "
             "extraction step landed in attention_atlas_shap (37/1000 "
             "failures observed at default=5).",
    )

    return ap


_ATTN_V3 = Path("/hpc/projects/icd.fast.ops/models/alex_lin_attention/v3/attention_v3")
_GENE_PHASE = _ATTN_V3 / "pma_top_phase_cells_v3.csv"
_GENE_FLUOR = _ATTN_V3 / "pma_top_fluorescent_cells_v3.csv"
_CHAD_PHASE = _ATTN_V3 / "pma_top_phase_cells_chad_v1.csv"
_CHAD_FLUOR = _ATTN_V3 / "pma_top_fluorescent_cells_chad_v1.csv"


# Per-modality + per-aggregation eval CSV layout (all in one folder):
#   cdino_eval/
#     cdino_eval_phase_50.csv          (gene-level, phase classifier)
#     cdino_eval_fluorescent_50.csv    (gene-level, fluor classifier)
#     cdino_eval_phase_chad_50.csv     (CHAD-level, phase classifier)
#     cdino_eval_fluorescent_chad_50.csv (CHAD-level, fluor classifier)
# Each `--eval-csv` arg names the PHASE CSV; the fluor sibling is
# auto-discovered by replacing "phase" → "fluorescent" in the filename.
# NEW cdino_eval dir has 9 cell-count bins (10–5000) vs the legacy
# 4-bin variant under …/cdino_eval/. Both phase + fluor siblings live
# here so the auto-sibling resolver finds them.
_CDINO_EVAL_DIR = Path(
    "/hpc/projects/icd.fast.ops/models/alex_lin_attention/"
    "v3/attention_v3/cdino"
)
_GENE_EVAL = _CDINO_EVAL_DIR / "cdino_eval_phase_50.csv"
_CHAD_EVAL = _CDINO_EVAL_DIR / "cdino_eval_phase_chad_50.csv"


def apply_aggregation_level_defaults(args):
    """Swap default --phase-csv / --fluor-csv to chad_v1 variants and
    suffix --output with `_chad` when --aggregation-level complex AND
    the user left the CSVs at default. Kept as a top-level helper so
    `attention_atlas_shap.main()` (which bypasses
    `attention_atlas.main()`) can call it too — without this, atlas_shap
    in CHAD mode reads the gene-level pma CSVs, can't find the
    predicted_class column, and renders blank cell crops.
    """
    if args.aggregation_level != "complex":
        return
    if args.phase_csv == _GENE_PHASE:
        args.phase_csv = _CHAD_PHASE
        print(f"[CHAD] swapped --phase-csv default to: {args.phase_csv.name}")
    if args.fluor_csv == _GENE_FLUOR:
        args.fluor_csv = _CHAD_FLUOR
        print(f"[CHAD] swapped --fluor-csv default to: {args.fluor_csv.name}")
    # Switch eval CSV to the CHAD complex-level eval (same n_cells=100
    # as gene-level for consistency — the CHAD eval CSV carries every
    # bin from 10..1000 so the existing gene-level default works
    # without a separate n_cells swap).
    if Path(args.eval_csv) == _GENE_EVAL:
        args.eval_csv = _CHAD_EVAL
        print(f"[CHAD] swapped --eval-csv default to: {args.eval_csv.name}")
    # Swap NTC PMA CSV defaults (high-attention NTC pool) gene → chad
    # variants. Defined on attention_atlas_shap (which adds the
    # corresponding CLI args); accessed via getattr so attention_atlas's
    # own main() — which doesn't expose these args — stays a no-op.
    _GENE_NTC_PHASE = Path(
        "/hpc/projects/icd.fast.ops/models/alex_lin_attention/"
        "v3/attention_v3/pma_top_phase_cells_ntc_v3.csv"
    )
    _CHAD_NTC_PHASE = Path(
        "/hpc/projects/icd.fast.ops/models/alex_lin_attention/"
        "v3/attention_v3/pma_top_phase_cells_chad_ntc_v3.csv"
    )
    _GENE_NTC_FLUOR = Path(
        "/hpc/projects/icd.fast.ops/models/alex_lin_attention/"
        "v3/attention_v3/pma_top_fluorescent_cells_ntc_v3.csv"
    )
    _CHAD_NTC_FLUOR = Path(
        "/hpc/projects/icd.fast.ops/models/alex_lin_attention/"
        "v3/attention_v3/pma_top_fluorescent_cells_chad_ntc_v3.csv"
    )
    pma_phase = getattr(args, "ntc_pma_phase_csv", None)
    if pma_phase is not None and Path(pma_phase) == _GENE_NTC_PHASE:
        args.ntc_pma_phase_csv = _CHAD_NTC_PHASE
        print(f"[CHAD] swapped --ntc-pma-phase-csv → "
              f"{args.ntc_pma_phase_csv.name}")
    pma_fluor = getattr(args, "ntc_pma_fluor_csv", None)
    if pma_fluor is not None and Path(pma_fluor) == _GENE_NTC_FLUOR:
        args.ntc_pma_fluor_csv = _CHAD_NTC_FLUOR
        print(f"[CHAD] swapped --ntc-pma-fluor-csv → "
              f"{args.ntc_pma_fluor_csv.name}")

    # SHAP feature/caption CSVs and SHAP caches — atlas_shap-only args.
    # Same pattern as the NTC swaps above: only swap when the user kept
    # the gene-level default. Hardcoded gene/chad paths mirror the
    # constants in attention_atlas_shap.py.
    _GENE_SHAP_FEAT = Path(
        "/hpc/projects/icd.fast.ops/models/alex_lin_attention/"
        "top20_v4/ko_shap_features.csv"
    )
    _CHAD_SHAP_FEAT = Path(
        "/hpc/projects/icd.fast.ops/models/alex_lin_attention/"
        "top20_v4_chad/ko_shap_features.csv"
    )
    _GENE_SHAP_CAP = Path(
        "/hpc/projects/icd.fast.ops/models/alex_lin_attention/"
        "top20_v4/ko_shap_captions.csv"
    )
    _CHAD_SHAP_CAP = Path(
        "/hpc/projects/icd.fast.ops/models/alex_lin_attention/"
        "top20_v4_chad/ko_shap_captions.csv"
    )
    _GENE_CACHE_PHASE = Path(
        "/hpc/projects/icd.fast.ops/models/alex_lin_attention/"
        "shap_caches/v4/phase"
    )
    _CHAD_CACHE_PHASE = Path(
        "/hpc/projects/icd.fast.ops/models/alex_lin_attention/"
        "shap_caches/v4_chad/phase"
    )
    _GENE_CACHE_FLUOR = Path(
        "/hpc/projects/icd.fast.ops/models/alex_lin_attention/"
        "shap_caches/v4/fluor"
    )
    _CHAD_CACHE_FLUOR = Path(
        "/hpc/projects/icd.fast.ops/models/alex_lin_attention/"
        "shap_caches/v4_chad/fluor"
    )
    sf = getattr(args, "shap_features_csv", None)
    if sf is not None and Path(sf) == _GENE_SHAP_FEAT:
        args.shap_features_csv = _CHAD_SHAP_FEAT
        print(f"[CHAD] swapped --shap-features-csv → "
              f"{args.shap_features_csv.parent.name}/{args.shap_features_csv.name}")
    sc = getattr(args, "shap_captions_csv", None)
    if sc is not None and Path(sc) == _GENE_SHAP_CAP:
        args.shap_captions_csv = _CHAD_SHAP_CAP
        print(f"[CHAD] swapped --shap-captions-csv → "
              f"{args.shap_captions_csv.parent.name}/{args.shap_captions_csv.name}")
    cp = getattr(args, "shap_cache_phase", None)
    if cp is not None and Path(cp) == _GENE_CACHE_PHASE:
        args.shap_cache_phase = _CHAD_CACHE_PHASE
        print(f"[CHAD] swapped --shap-cache-phase → "
              f"{args.shap_cache_phase.parent.name}/{args.shap_cache_phase.name}")
    cf = getattr(args, "shap_cache_fluor", None)
    if cf is not None and Path(cf) == _GENE_CACHE_FLUOR:
        args.shap_cache_fluor = _CHAD_CACHE_FLUOR
        print(f"[CHAD] swapped --shap-cache-fluor → "
              f"{args.shap_cache_fluor.parent.name}/{args.shap_cache_fluor.name}")
    # Suffix output PDF with `_chad` so the gene-level run isn't clobbered.
    if args.output.stem and "_chad" not in args.output.stem:
        args.output = args.output.with_name(
            args.output.stem + "_chad" + args.output.suffix
        )
        print(f"[CHAD] suffixed --output to: {args.output.name}")


def main():
    args = _make_arg_parser().parse_args()
    apply_aggregation_level_defaults(args)

    if args.local:
        _run_atlas_local(
            phase_csv=args.phase_csv,
            fluor_csv=args.fluor_csv,
            output=args.output,
            crop_size=args.crop_size,
            genes_subset=args.genes,
            max_genes=args.max_genes,
            skip=args.skip,
            dpi=args.dpi,
            workers=args.workers,
            keep_pages=args.keep_pages,
            ntc_max_experiments=args.ntc_experiments,
            ntc_max_per_experiment=args.ntc_per_experiment,
            skip_ntc=args.no_ntc,
            strict=args.strict,
            skip_cp_4i=args.skip_cp_4i,
            eval_csv=args.eval_csv,
            eval_n_cells=args.eval_n_cells,
            threshold=args.threshold,
            aggregation_level=args.aggregation_level,
            nuclear_overlay=args.nuclear_overlay,
            load_pad_factor=args.load_pad_factor,
        )
    else:
        _submit_via_slurm(args)


if __name__ == "__main__":
    main()
