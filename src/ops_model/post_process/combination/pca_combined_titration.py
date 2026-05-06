"""Combined-reporter cell-count titration.

Like ``pca_titration`` but instead of titrating each reporter independently,
this samples cells across N reporters at each cell budget, h-concatenates
their NTC-normalized guide-level features into one combined matrix, and scores
that combined matrix. Produces one curve per metric for the *group* (e.g. all
cp markers together, all 4i markers together), so you can compare how the cp
panel and the 4i panel collectively degrade with cell count.

Built-in groups (resolved from --output-dir + --paper-v1 + --cell-dino):
  --groups cp                → cells.h5ads under paper_v1/only_cp/<...>/per_signal/
  --groups 4i                → cells.h5ads under paper_v1/only_4i/<...>/per_signal/
  --groups all               → cells.h5ads under paper_v1/with_cp/with_4i/<...>/per_signal/
  --groups matched_livecell  → 7 live-cell signals under paper_v1/all_livecell/<...>/per_signal/
                              (one per CP organelle in --matching-config)
  --groups custom + --custom-paths a.h5ad,b.h5ad,...

After --compare cp,4i (or any pair) the script overlays their mean-mAP curves
in 4 figures (one per metric), in the same style as compare_pca_titration_versions.

Usage::

    # Run cp + 4i in parallel SLURM, then compare
    python -m ops_model.post_process.combination.pca_combined_titration \\
        --cell-dino --paper-v1 --per-guide-max-titration --slurm \\
        --groups cp,4i --compare
"""
from __future__ import annotations

import argparse
import logging
import math
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import anndata as ad
import numpy as np
import pandas as pd

from ops_model.features.anndata_utils import (
    aggregate_to_level,
    hconcat_by_perturbation,
    normalize_guide_adata,
)
from ops_model.post_process.combination.pca_titration import (
    DOWNSAMPLE_RATIO,
    METRICS,
    MIN_CELLS,
    SCALES,
    SCALE_LABEL_SHORT,
    TITRATION_MAP_LABELS,
    TITRATION_RATIO_LABELS,
    _apply_x_scale,
    _build_parser as _titr_parser,  # reuse common nesting flags
    _build_per_ko_schedule,
    _prepare_for_copairs,
    _resolve_output_dir,
    _score_all_metrics,
    _subsample_and_aggregate,
    _subsample_per_guide_and_aggregate,
    _subsample_per_ko_and_aggregate,
    titration_x_axis_base_label,
)

logger = logging.getLogger(__name__)

# Default cp_challenge config — defines organelle ↔ live-cell channel pairings
DEFAULT_MATCHING_CONFIG = (
    "/home/gav.sturm/linked_folders/mydata/ops_mono/organelle_profiler/configs/"
    "cp_challenge_config.yaml"
)
_GENERIC_FLUORS = {"GFP", "mCherry", "BFP", "RFP", "YFP", "Cy5", "Cy3"}


# ---------------------------------------------------------------------------
# Group resolution
# ---------------------------------------------------------------------------


def _per_signal_dir(args: argparse.Namespace, group: str) -> Path:
    """Resolve the per_signal/ dir for a given group given the same flag
    space as pca_titration / pca_optimization."""
    ns = argparse.Namespace(**vars(args))
    # Reset channel-selection flags to match the requested group
    ns.include_4i = False
    ns.include_cp = False
    ns.only_4i = False
    ns.only_cp = False
    if group == "cp":
        ns.include_cp = True
        ns.only_cp = True
    elif group == "4i":
        ns.include_4i = True
        ns.only_4i = True
    elif group == "all":
        ns.include_cp = True
        ns.include_4i = True
    elif (
        group == "custom"
        or group == BEST_GROUP_NAME
        or group == LIVECELL_ALL_GROUP_NAME
        or _parse_matched_livecell_group(group) is not None
        or _parse_matched_combo_group(group) is not None
    ):
        # No channel-set flags — use the all-livecell dir
        pass
    else:
        raise ValueError(f"Unknown group: {group!r}")
    variant_dir = _resolve_output_dir(ns)
    return variant_dir / "per_signal"


def _extract_marker(entry: dict) -> str:
    """Best-effort marker token from a cp_challenge_config live_cell entry.

    Prefers ``dino_channel`` when it's specific; otherwise extracts the marker
    name from ``notes`` (drops the leading fluorophore + trailing annotations).
    """
    ch = (entry.get("dino_channel") or "").strip()
    if ch and ch not in _GENERIC_FLUORS:
        return ch
    notes = (entry.get("notes") or "").strip()
    parts = notes.split(maxsplit=1)
    if len(parts) != 2:
        return ch  # last resort
    marker = parts[1].split(",")[0].strip()
    return marker.replace(" ", "_")


def _matched_livecell_organelle_lists(
    args: argparse.Namespace, config_path: str,
) -> Tuple[Dict[str, List[Tuple[str, Path]]], Path]:
    """Resolve every (organelle → ordered list of [(signal, path)]) match in the
    cp_challenge_config. Order follows the ``live_cell`` entries in the YAML;
    duplicates within an organelle are removed (so two YAML rows that both
    point to the same signal collapse to one entry).
    """
    import yaml

    cfg_path = Path(config_path)
    if not cfg_path.is_file():
        raise SystemExit(f"--matching-config not found: {cfg_path}")
    cfg = yaml.safe_load(cfg_path.read_text()) or {}
    comparisons = cfg.get("comparisons", {}) or {}

    per_signal = _per_signal_dir(args, "matched_livecell")
    available = {p.stem.replace("_cells", ""): p for p in per_signal.glob("*_cells.h5ad")}
    if not available:
        raise SystemExit(f"No *_cells.h5ad in {per_signal}")
    available_lc = {sig.lower(): (sig, path) for sig, path in available.items()}

    matchings: Dict[str, List[Tuple[str, Path]]] = {}
    for organelle, comp in comparisons.items():
        ordered: List[Tuple[str, Path]] = []
        seen_paths: set = set()
        for entry in (comp.get("live_cell", []) or []):
            marker = _extract_marker(entry)
            if not marker:
                continue
            mlc = marker.lower()
            for sig_lc, (sig, path) in available_lc.items():
                if path in seen_paths:
                    continue
                if sig_lc.endswith(f"_{mlc}") or sig_lc == mlc:
                    ordered.append((sig, path))
                    seen_paths.add(path)
                    break  # next YAML entry
        matchings[organelle] = ordered
    return matchings, cfg_path


def _matched_livecell_paths(
    args: argparse.Namespace,
    config_path: str,
    pick_index: int = 0,
) -> Tuple[List[Path], Dict[str, str]]:
    """Pick the ``pick_index``-th unique matching live-cell signal per
    organelle. Falls back to the last available match if ``pick_index`` exceeds
    the matches for an organelle.

    Returns (sorted h5ad paths, {organelle: chosen_signal_name}).
    """
    matchings, cfg_path = _matched_livecell_organelle_lists(args, config_path)
    print(
        f"  Matching live-cell signals against {len(matchings)} CP organelles "
        f"using {cfg_path.name} (pick_index={pick_index})"
    )

    picked: List[Path] = []
    organelle_to_sig: Dict[str, str] = {}
    for organelle, ordered in matchings.items():
        if not ordered:
            print(f"    {organelle:>16}: no live-cell match — skipped")
            continue
        idx = min(pick_index, len(ordered) - 1)  # fall back to last available
        sig, path = ordered[idx]
        fallback = " (fallback to last)" if pick_index >= len(ordered) else ""
        picked.append(path)
        organelle_to_sig[organelle] = sig
        print(f"    {organelle:>16} → {sig}{fallback}")
    if not picked:
        raise SystemExit("No live-cell signals matched any CP organelle")
    return sorted(picked), organelle_to_sig


_MATCHED_SET_RE = re.compile(r"^matched_livecell(?:_set(\d+))?$")
_MATCHED_COMBO_RE = re.compile(r"^matched_livecell_combo(\d+)$")
ALL_COMBOS_KEYWORD = "all_combos"
BEST_GROUP_NAME = "matched_livecell_best"
LIVECELL_ALL_GROUP_NAME = "livecell"  # all standard live-cell signals (40 in paper_v1)


def _parse_matched_livecell_group(group: str) -> Optional[int]:
    """Parse 'matched_livecell' / 'matched_livecell_set1' / 'matched_livecell_set2'
    etc. Returns the 0-based pick_index, or None if the group isn't a matched
    variant.
    """
    m = _MATCHED_SET_RE.match(group)
    if not m:
        return None
    n = m.group(1)
    return 0 if n is None else max(int(n) - 1, 0)


def _parse_matched_combo_group(group: str) -> Optional[int]:
    """Parse 'matched_livecell_combo<N>' (1-based). Returns the combo index,
    or None if the group isn't a combo variant."""
    m = _MATCHED_COMBO_RE.match(group)
    return int(m.group(1)) if m else None


def _resolve_best_matched_livecell(
    args: argparse.Namespace, config_path: str,
) -> Tuple[List[Path], Dict[str, str]]:
    """Resolve the curated best-per-organelle live-cell signal set, sourced
    from the ``best_matched_livecell:`` block of the cp_challenge config.
    Returns (sorted h5ad paths, {organelle: chosen_signal_name}).
    """
    import yaml

    cfg_path = Path(config_path)
    if not cfg_path.is_file():
        raise SystemExit(f"--matching-config not found: {cfg_path}")
    cfg = yaml.safe_load(cfg_path.read_text()) or {}
    best = cfg.get("best_matched_livecell") or {}
    if not best:
        raise SystemExit(
            f"`best_matched_livecell:` block missing or empty in {cfg_path}. "
            f"Add it to define the curated best-per-organelle marker mapping."
        )

    per_signal = _per_signal_dir(args, BEST_GROUP_NAME)
    available = {p.stem.replace("_cells", ""): p for p in per_signal.glob("*_cells.h5ad")}
    if not available:
        raise SystemExit(f"No *_cells.h5ad in {per_signal}")
    available_lc = {sig.lower(): (sig, path) for sig, path in available.items()}

    picked: List[Path] = []
    organelle_to_sig: Dict[str, str] = {}
    print(f"  Best-marker mapping from {cfg_path.name}:")
    for organelle, sig_name in best.items():
        # Accept exact or case-insensitive match against the per_signal stems
        target = available_lc.get(str(sig_name).lower())
        if target is None:
            print(f"    {organelle:>16}: {sig_name} → NOT FOUND in {per_signal.name}")
            continue
        sig, path = target
        picked.append(path)
        organelle_to_sig[organelle] = sig
        print(f"    {organelle:>16} → {sig}")
    if not picked:
        raise SystemExit("None of the `best_matched_livecell` entries resolved.")
    return sorted(picked), organelle_to_sig


def _enumerate_marker_combinations(
    args: argparse.Namespace, config_path: str,
) -> List[Dict]:
    """Cartesian product across CP organelles of unique live-cell signals.

    Returns a list of dicts (one per combination) with:
        combo_index       — 1-based ordinal
        group_name        — 'matched_livecell_combo<N>'
        paths             — sorted list of cells.h5ad Paths
        organelle_to_signal — {organelle: signal_name}
    """
    from itertools import product

    matchings, _ = _matched_livecell_organelle_lists(args, config_path)
    organelles_with_opts = [(o, opts) for o, opts in matchings.items() if opts]
    if not organelles_with_opts:
        return []

    organelle_names = [o for o, _ in organelles_with_opts]
    options_per_organelle = [opts for _, opts in organelles_with_opts]

    combos: List[Dict] = []
    for idx, picks in enumerate(product(*options_per_organelle), 1):
        organelle_to_sig = {
            organelle_names[i]: picks[i][0] for i in range(len(picks))
        }
        paths = sorted(picks[i][1] for i in range(len(picks)))
        combos.append({
            "combo_index": idx,
            "group_name": f"matched_livecell_combo{idx}",
            "paths": paths,
            "organelle_to_signal": organelle_to_sig,
        })
    return combos


def _expand_groups_with_combos(
    args: argparse.Namespace, groups: List[str],
) -> List[str]:
    """Expand the literal 'all_combos' / 'matched_livecell_all_combos' tokens
    into the full list of `matched_livecell_combo<N>` groups. Preserves order
    and deduplicates."""
    expanded: List[str] = []
    seen: set = set()

    def _add(g: str) -> None:
        if g not in seen:
            expanded.append(g)
            seen.add(g)

    for g in groups:
        if g in (ALL_COMBOS_KEYWORD, "matched_livecell_all_combos"):
            for combo in _enumerate_marker_combinations(args, args.matching_config):
                _add(combo["group_name"])
        else:
            _add(g)
    return expanded


def _resolve_group_paths(
    args: argparse.Namespace,
    group: str,
    custom_paths: Optional[List[str]] = None,
    matching_config: Optional[str] = None,
) -> List[Path]:
    """Return sorted list of *_cells.h5ad paths for the group."""
    if group == "custom":
        if not custom_paths:
            raise SystemExit("--group custom requires --custom-paths")
        paths = [Path(p) for p in custom_paths]
        for p in paths:
            if not p.is_file():
                raise SystemExit(f"Missing cells h5ad: {p}")
        return sorted(paths)
    if group == BEST_GROUP_NAME:
        paths, _ = _resolve_best_matched_livecell(
            args, matching_config or DEFAULT_MATCHING_CONFIG,
        )
        return paths
    if group == LIVECELL_ALL_GROUP_NAME:
        per_signal = _per_signal_dir(args, group)
        paths = sorted(per_signal.glob("*_cells.h5ad"))
        if not paths:
            raise SystemExit(f"No *_cells.h5ad found in {per_signal}")
        return paths
    combo_idx = _parse_matched_combo_group(group)
    if combo_idx is not None:
        combos = _enumerate_marker_combinations(
            args, matching_config or DEFAULT_MATCHING_CONFIG,
        )
        if combo_idx < 1 or combo_idx > len(combos):
            raise SystemExit(
                f"combo index {combo_idx} out of range [1, {len(combos)}]"
            )
        return list(combos[combo_idx - 1]["paths"])
    pick_idx = _parse_matched_livecell_group(group)
    if pick_idx is not None:
        paths, _ = _matched_livecell_paths(
            args, matching_config or DEFAULT_MATCHING_CONFIG, pick_index=pick_idx,
        )
        return paths
    per_signal = _per_signal_dir(args, group)
    paths = sorted(per_signal.glob("*_cells.h5ad"))
    if not paths:
        raise SystemExit(f"No *_cells.h5ad found in {per_signal}")
    return paths


def _resolve_matched_set_membership(
    args: argparse.Namespace,
    group: str,
    matching_config: Optional[str] = None,
) -> Optional[Dict[str, str]]:
    """For matched_livecell / matched_livecell_setN / matched_livecell_combo<N>
    / matched_livecell_best groups, return the organelle → chosen signal name
    mapping. Returns None for other groups."""
    if group == BEST_GROUP_NAME:
        _, organelle_to_sig = _resolve_best_matched_livecell(
            args, matching_config or DEFAULT_MATCHING_CONFIG,
        )
        return organelle_to_sig
    combo_idx = _parse_matched_combo_group(group)
    if combo_idx is not None:
        combos = _enumerate_marker_combinations(
            args, matching_config or DEFAULT_MATCHING_CONFIG,
        )
        if 1 <= combo_idx <= len(combos):
            return dict(combos[combo_idx - 1]["organelle_to_signal"])
        return None
    pick_idx = _parse_matched_livecell_group(group)
    if pick_idx is None:
        return None
    _, organelle_to_sig = _matched_livecell_paths(
        args, matching_config or DEFAULT_MATCHING_CONFIG, pick_index=pick_idx,
    )
    return organelle_to_sig


def _mode_subdir(sampling_mode: str) -> str:
    """Filesystem-safe subdir name per titration method (keeps modes from clobbering)."""
    return {
        "per_guide": "per_guide_max",
        "per_guide_median": "per_guide_median",
        "per_ko": "per_ko",
        "total": "total_cells",
    }.get(sampling_mode, sampling_mode)


def _resolve_group_output_dir(
    args: argparse.Namespace, group: str, sampling_mode: str = "per_guide",
) -> Path:
    """Output dir for the combined titration of one group, scoped by titration method."""
    return (
        _per_signal_dir(args, group).parent
        / "combined_titration"
        / _mode_subdir(sampling_mode)
        / group
    )


def _compact_groups_tag(groups: Sequence[str], max_len: int = 200) -> str:
    """Build a filesystem-safe tag joining group names with `_vs_`. Collapses
    runs of `matched_livecell_combo<N>` into a compact `combos<count>` token
    (with the explicit indices in `combos<a>-<b>` form when contiguous) so the
    leaf directory name stays well under the 255-byte filename limit even with
    32+ combo groups."""
    combo_idxs: List[int] = []
    others: List[str] = []
    for g in groups:
        idx = _parse_matched_combo_group(g)
        if idx is not None:
            combo_idxs.append(idx)
        else:
            others.append(g)
    if combo_idxs:
        sorted_idxs = sorted(combo_idxs)
        if sorted_idxs == list(range(sorted_idxs[0], sorted_idxs[-1] + 1)):
            combo_tag = f"combos{sorted_idxs[0]}-{sorted_idxs[-1]}"
        else:
            combo_tag = f"combos_n{len(sorted_idxs)}"
        tag = "_vs_".join(others + [combo_tag])
    else:
        tag = "_vs_".join(groups)
    if len(tag) > max_len:
        # Last resort — truncate and append a deterministic short hash.
        import hashlib
        h = hashlib.sha1("|".join(groups).encode()).hexdigest()[:8]
        tag = tag[: max_len - 12] + f"__{h}"
    return tag


def _resolve_compare_dir(
    args: argparse.Namespace, groups: Sequence[str], sampling_mode: str = "per_guide",
) -> Path:
    """Top-level dir for cross-group comparison plots, scoped by titration method.

    The dir nests under the *union* of channel-sets needed by the groups so the
    comparison output sits alongside (or above) every group it covers. E.g.
    comparing cp + 4i + matched_livecell lands under with_cp/with_4i/all_livecell/.
    """
    ns = argparse.Namespace(**vars(args))
    ns.only_4i = False
    ns.only_cp = False
    ns.include_cp = any(g in ("cp", "all") for g in groups)
    ns.include_4i = any(g in ("4i", "all") for g in groups)
    base = _resolve_output_dir(ns)
    return (
        base / "combined_titration_compare"
        / _mode_subdir(sampling_mode)
        / _compact_groups_tag(groups)
    )


# ---------------------------------------------------------------------------
# Schedule
# ---------------------------------------------------------------------------


def _per_guide_pool(paths: List[Path]) -> np.ndarray:
    """Pooled non-NTC sgRNA cell counts across all reporters (or all sgRNAs if no NTC info)."""
    pooled: List[int] = []
    for p in paths:
        a = ad.read_h5ad(p, backed="r")
        if "sgRNA" not in a.obs.columns:
            raise ValueError(f"{p.name}: per-guide titration requires 'sgRNA' obs col")
        pert_col = "perturbation" if "perturbation" in a.obs.columns else "label_str"
        sg_counts = a.obs.groupby("sgRNA", observed=True).size()
        sg_pert = a.obs.groupby("sgRNA", observed=True)[pert_col].first()
        non_ntc_sg = sg_pert[~sg_pert.astype(str).str.startswith("NTC")].index
        pool = sg_counts.loc[sg_counts.index.intersection(non_ntc_sg)]
        if len(pool) == 0:
            pool = sg_counts
        pooled.extend(int(v) for v in pool.values)
    return np.asarray(pooled, dtype=int)


def _build_per_guide_max_schedule(paths: List[Path]) -> List[int]:
    """cells/guide schedule starting at p90 of pooled non-NTC sgRNA cell counts."""
    pool = _per_guide_pool(paths)
    start = int(np.percentile(pool, 90))
    schedule: List[int] = []
    n = start
    while n >= 1:
        schedule.append(n)
        n = int(n * DOWNSAMPLE_RATIO)
    return schedule


def _build_per_guide_median_schedule(
    paths: List[Path],
    start_override: Optional[int] = None,
) -> List[int]:
    """cells/guide schedule from the MEDIAN of pooled non-NTC sgRNA counts
    down to 1 (mirrors max-mode but caps the high end at median instead of
    p90). ``start_override`` clamps the starting cells/guide value (used to
    align starts across groups in cross-group comparisons).
    """
    pool = _per_guide_pool(paths)
    start = start_override if start_override is not None else int(np.median(pool))
    return _build_per_ko_schedule(start)


def _per_guide_median(paths: List[Path]) -> int:
    """Median of pooled non-NTC sgRNA cell counts across all reporters in the group."""
    return int(np.median(_per_guide_pool(paths)))


def _build_per_ko_max_schedule(paths: List[Path]) -> List[int]:
    """cells/KO schedule from largest reporter's max non-NTC perturbation count."""
    starts = []
    for p in paths:
        a = ad.read_h5ad(p, backed="r")
        pert_col = "perturbation" if "perturbation" in a.obs.columns else "label_str"
        counts = a.obs.groupby(pert_col, observed=True).size()
        non_ntc = counts.loc[~counts.index.astype(str).str.startswith("NTC")]
        starts.append(int(non_ntc.max() if len(non_ntc) else counts.max()))
    start = int(np.max(starts))
    schedule = []
    n = start
    while n >= 1:
        schedule.append(n)
        n = int(n * DOWNSAMPLE_RATIO)
    return schedule


def _build_total_schedule(paths: List[Path]) -> List[int]:
    """Plain n_cells schedule starting at the SMALLEST reporter's total cell count."""
    totals = [ad.read_h5ad(p, backed="r").n_obs for p in paths]
    start = int(min(totals))
    schedule = []
    n = start
    while n >= MIN_CELLS:
        schedule.append(n)
        n = int(n * DOWNSAMPLE_RATIO)
    return schedule or [start]


# ---------------------------------------------------------------------------
# Core: subsample → aggregate → NTC-normalize → h-concat → score
# ---------------------------------------------------------------------------


def _subsample_one(
    adata_cells: ad.AnnData, target: int, sampling_mode: str, rng: np.random.RandomState,
) -> ad.AnnData:
    if sampling_mode in ("per_guide", "per_guide_median"):
        return _subsample_per_guide_and_aggregate(adata_cells, target, rng)
    if sampling_mode == "per_ko":
        return _subsample_per_ko_and_aggregate(adata_cells, target, rng)
    return _subsample_and_aggregate(adata_cells, target, rng)


def _build_combined_at_target(
    cells_blocks: List[ad.AnnData],
    target: int,
    sampling_mode: str,
    norm_method: str,
    rng: np.random.RandomState,
) -> ad.AnnData:
    """For each reporter: subsample → aggregate → NTC-normalize, then h-concat."""
    blocks = []
    for adata in cells_blocks:
        g_sub = _subsample_one(adata, target, sampling_mode, rng)
        g_norm = normalize_guide_adata(g_sub, norm_method)
        # Tag features so concat doesn't collide
        sig = str(adata.obs.get("signal", pd.Series(["?"])).iloc[0])
        g_norm.var_names = [f"{sig}::{v}" for v in g_norm.var_names]
        blocks.append(g_norm)
    return hconcat_by_perturbation(blocks, level="guide")


def run_combined_titration(
    cells_h5ad_paths: List[str],
    output_dir: str,
    sampling_mode: str = "per_guide",
    norm_method: str = "ntc",
    distance: str = "cosine",
    n_bootstraps: int = 1,
    random_seed: int = 42,
    schedule: Optional[List[int]] = None,
    group_label: str = "combined",
    cache: bool = True,
) -> str:
    """Run the combined-titration loop for one group and write CSV + plots.

    sampling_mode: 'per_guide' | 'per_ko' | 'total' — interprets ``schedule``
        targets as cells/sgRNA, cells/perturbation, or absolute n_cells.
    cache: when True (default), reuse already-scored rows from any existing
        combined_titration_<group>.csv and only score the missing schedule
        targets. Pass --no-cache (CLI) to force a full recompute.
    """
    _logger = logging.getLogger(f"combined_titration.{group_label}")
    if not _logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        _logger.addHandler(h)
        _logger.setLevel(logging.INFO)

    paths = [Path(p) for p in cells_h5ad_paths]
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(random_seed)

    _logger.info(
        f"[{group_label}] Loading {len(paths)} reporter cells.h5ads "
        f"(mode={sampling_mode}, bootstrap={n_bootstraps})..."
    )
    cells_blocks: List[ad.AnnData] = []
    for p in paths:
        a = ad.read_h5ad(p)
        if "signal" not in a.obs.columns:
            sig_guess = p.stem.replace("_cells", "")
            a.obs["signal"] = sig_guess
        cells_blocks.append(a)
        _logger.info(f"  {p.name}: {a.n_obs:,} cells x {a.n_vars} PCs")

    if schedule is None:
        if sampling_mode == "per_guide":
            schedule = _build_per_guide_max_schedule(paths)
        elif sampling_mode == "per_guide_median":
            schedule = _build_per_guide_median_schedule(paths)
        elif sampling_mode == "per_ko":
            schedule = _build_per_ko_max_schedule(paths)
        else:
            schedule = _build_total_schedule(paths)
    _logger.info(f"[{group_label}] Schedule ({len(schedule)} pts): {schedule}")

    csv_path = out_dir / f"combined_titration_{group_label}.csv"
    target_col = (
        "cells_per_guide" if sampling_mode in ("per_guide", "per_guide_median")
        else "cells_per_perturbation" if sampling_mode == "per_ko"
        else "n_cells"
    )
    cached_rows: List[Dict] = []
    targets_to_run = list(schedule)
    if cache and csv_path.is_file():
        try:
            df_old = pd.read_csv(csv_path)
            if target_col in df_old.columns:
                done = {int(v) for v in df_old[target_col].dropna().astype(int).tolist()}
                cached_rows = df_old.to_dict(orient="records")
                targets_to_run = [t for t in schedule if int(t) not in done]
                if cached_rows:
                    _logger.info(
                        f"[{group_label}] Cache hit: {len(cached_rows)} existing "
                        f"rows in {csv_path.name}; "
                        f"{len(targets_to_run)}/{len(schedule)} targets need scoring "
                        f"(skipped: {sorted(done & set(int(t) for t in schedule))})"
                    )
        except Exception as exc:
            _logger.warning(f"[{group_label}] Cache read failed ({exc}); recomputing all.")
            cached_rows = []
            targets_to_run = list(schedule)

    metric_cols = [
        "activity_ratio", "activity_map_mean",
        "distinctiveness_ratio", "distinctiveness_map_mean",
        "corum_ratio", "corum_map_mean",
        "chad_ratio", "chad_map_mean",
    ]
    base_seed = int(rng.randint(0, 2**31 - 1))
    rows = []

    for target in targets_to_run:
        unit = (
            "cells/guide" if sampling_mode in ("per_guide", "per_guide_median")
            else "cells/KO" if sampling_mode == "per_ko"
            else "cells"
        )
        _logger.info(
            f"[{group_label}] Scoring at {target:,} {unit} "
            f"({n_bootstraps} draw{'s' if n_bootstraps > 1 else ''})..."
        )
        t_step = time.time()

        draws: List[Dict[str, float]] = []
        last_combined = None
        for b in range(n_bootstraps):
            draw_rng = np.random.RandomState(base_seed + b * 9973 + target)
            combined = _build_combined_at_target(
                cells_blocks, target, sampling_mode, norm_method, draw_rng,
            )
            scores_b = _score_all_metrics(combined, _logger)
            draws.append(scores_b)
            last_combined = combined

        scores: Dict[str, float] = {}
        for k in metric_cols:
            vals = np.array([d.get(k, float("nan")) for d in draws], dtype=float)
            finite = vals[np.isfinite(vals)]
            if len(finite) == 0:
                scores[k] = float("nan")
                scores[f"{k}_sem"] = float("nan")
                scores[f"{k}_std"] = float("nan")
            else:
                scores[k] = float(np.mean(finite))
                if len(finite) > 1:
                    std = float(np.std(finite, ddof=1))
                    scores[f"{k}_std"] = std
                    scores[f"{k}_sem"] = std / np.sqrt(len(finite))
                else:
                    scores[f"{k}_std"] = 0.0
                    scores[f"{k}_sem"] = 0.0
            if n_bootstraps > 1:
                scores[f"{k}_draws"] = "|".join(
                    "nan" if not np.isfinite(v) else f"{v:.6g}" for v in vals
                )

        # x-axis bookkeeping
        n_guides = last_combined.n_obs if last_combined is not None else 0
        pert_col = (
            "perturbation" if "perturbation" in last_combined.obs.columns
            else "label_str"
        )
        n_perts = last_combined.obs[pert_col].nunique()
        n_reporters = len(cells_blocks)
        if sampling_mode in ("per_guide", "per_guide_median"):
            scores["cells_per_guide"] = target
            scores["n_cells"] = target * n_guides * n_reporters
            scores["cells_per_perturbation"] = (
                scores["n_cells"] / max(n_perts * n_reporters, 1)
            )
        elif sampling_mode == "per_ko":
            scores["cells_per_perturbation"] = target
            scores["n_cells"] = target * n_perts * n_reporters
            scores["cells_per_guide"] = (
                scores["n_cells"] / max(n_guides * n_reporters, 1)
            )
        else:
            scores["n_cells"] = target * n_reporters
            scores["cells_per_guide"] = target / max(n_guides, 1)
            scores["cells_per_perturbation"] = target / max(n_perts, 1)
        scores["n_guides"] = int(n_guides)
        scores["n_perturbations"] = int(n_perts)
        scores["n_reporters"] = n_reporters
        scores["n_bootstraps"] = n_bootstraps
        scores["group"] = group_label
        rows.append(scores)

        _logger.info(
            f"  act={scores['activity_map_mean']:.3f} "
            f"dist={scores['distinctiveness_map_mean']:.3f} "
            f"corum={scores['corum_map_mean']:.3f} "
            f"chad={scores['chad_map_mean']:.3f} "
            f"({time.time() - t_step:.0f}s)"
        )

    # Merge cached + newly-scored, dedupe on the target column, sort descending
    all_rows = (cached_rows or []) + rows
    df = pd.DataFrame(all_rows)
    if target_col in df.columns:
        df = (
            df.dropna(subset=[target_col])
              .drop_duplicates(subset=[target_col], keep="last")
              .sort_values(target_col, ascending=False)
              .reset_index(drop=True)
        )
    df.to_csv(csv_path, index=False)
    _logger.info(
        f"[{group_label}] Wrote {csv_path} ({len(df)} rows: "
        f"{len(rows)} new + {len(cached_rows)} cached)"
    )

    # Per-metric plot
    _plot_group_curves(df, group_label, out_dir, sampling_mode)
    return f"SUCCESS: {csv_path}"


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def _x_col_for_mode(sampling_mode: str) -> str:
    return {
        "per_guide": "cells_per_guide",
        "per_guide_median": "cells_per_guide",
        "per_ko": "cells_per_perturbation",
        "total": "n_cells",
    }[sampling_mode]


def _plot_group_curves(
    df: pd.DataFrame, group_label: str, out_dir: Path, sampling_mode: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x_col = _x_col_for_mode(sampling_mode)
    if x_col not in df.columns or df.empty:
        return
    x = df[x_col].values
    out_dir.mkdir(parents=True, exist_ok=True)

    for scale in SCALES:
        fig, axes = plt.subplots(1, 4, figsize=(22, 5), sharex=True)
        for ax, metric in zip(axes, METRICS):
            ycol = f"{metric}_map_mean"
            sem_col = f"{ycol}_sem"
            if ycol not in df.columns:
                continue
            y = df[ycol].values
            ax.plot(x, y, marker="o", lw=2.5, color="darkorange", label=group_label)
            if sem_col in df.columns:
                sem = df[sem_col].values
                ax.fill_between(x, y - sem, y + sem, color="darkorange", alpha=0.25, lw=0)
            ax.set_title(TITRATION_MAP_LABELS[metric], fontsize=12)
            ax.set_xlabel(titration_x_axis_base_label(x_col), fontsize=11)
            ax.set_ylabel("mean mAP", fontsize=11)
            _apply_x_scale(ax, x, scale, tick_fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10)
        fig.suptitle(
            f"Combined titration — {group_label} "
            f"(n_reporters={int(df['n_reporters'].iloc[0])}, "
            f"x={SCALE_LABEL_SHORT[scale]})",
            fontsize=13, fontweight="bold",
        )
        fig.tight_layout()
        stem = out_dir / f"combined_titration_{group_label}_{scale}"
        fig.savefig(stem.with_suffix(".png"), dpi=160, bbox_inches="tight")
        fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
        plt.close(fig)


def plot_group_comparison(
    csvs_by_group: Dict[str, Path],
    output_dir: Path,
    sampling_mode: str,
    title_prefix: str = "Combined titration",
    matched_set_membership: Optional[Dict[str, Dict[str, str]]] = None,
) -> None:
    """One canvas per x-axis scale, with all 4 metrics as subplots and all
    groups overlaid in each subplot.

    ``matched_set_membership`` (when provided): {group → {organelle → signal}}
    for every matched_livecell_setN group. The membership is rendered as a
    text annotation under the legend so the reader sees which markers each
    set used.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    dfs: Dict[str, pd.DataFrame] = {}
    for g, csv in csvs_by_group.items():
        if not Path(csv).is_file():
            logger.warning("Missing CSV for %s: %s", g, csv)
            continue
        dfs[g] = pd.read_csv(csv)
    if len(dfs) < 2:
        logger.warning("Need >=2 groups for comparison; got %d", len(dfs))
        return

    x_col = _x_col_for_mode(sampling_mode)

    # Dump the exact (x, y, sem) points plotted, long format, one row per
    # (group, metric, x). Same data for every scale variant of the plot.
    long_rows: List[Dict] = []
    for g, df in dfs.items():
        if x_col not in df.columns:
            continue
        n_rep = int(df["n_reporters"].iloc[0]) if "n_reporters" in df.columns else 0
        for metric in METRICS:
            ycol = f"{metric}_map_mean"
            sem_col = f"{ycol}_sem"
            if ycol not in df.columns:
                continue
            for _, row in df.iterrows():
                long_rows.append({
                    "group": g,
                    "n_reporters": n_rep,
                    "metric": metric,
                    "x_col": x_col,
                    "x": float(row[x_col]),
                    "y": float(row[ycol]),
                    "sem": float(row[sem_col]) if sem_col in df.columns and pd.notna(row.get(sem_col)) else float("nan"),
                })
    if long_rows:
        compare_csv = output_dir / f"compare_all_metrics_{x_col}.csv"
        pd.DataFrame(long_rows).to_csv(compare_csv, index=False)
        logger.info("Wrote %s", compare_csv)
    x_label = titration_x_axis_base_label(x_col)
    palette = {
        "cp": "#d97706",                # amber
        "4i": "#2563eb",                # blue
        "all": "#6b7280",               # gray
        "matched_livecell": "#10b981",  # green
    }
    fallback = ["#dc2626", "#7c3aed", "#0891b2", "#a16207"]

    group_labels: Dict[str, str] = {}
    for i, (g, df) in enumerate(dfs.items()):
        n_rep = int(df["n_reporters"].iloc[0]) if "n_reporters" in df.columns else 0
        group_labels[g] = f"{g} (n={n_rep})"

    x_all = np.concatenate([
        d[x_col].values for d in dfs.values() if x_col in d.columns
    ])

    for scale in SCALES:
        fig, axes = plt.subplots(1, 4, figsize=(22, 6), sharex=True)
        for ax, metric in zip(axes, METRICS):
            ycol = f"{metric}_map_mean"
            sem_col = f"{ycol}_sem"
            for i, (g, df) in enumerate(dfs.items()):
                if x_col not in df.columns or ycol not in df.columns:
                    continue
                x = df[x_col].values
                y = df[ycol].values
                color = palette.get(g) or fallback[i % len(fallback)]
                ax.plot(
                    x, y, marker="o", lw=3.0, ms=7, color=color,
                    label=group_labels[g],
                )
                if sem_col in df.columns:
                    sem = df[sem_col].values
                    ax.fill_between(x, y - sem, y + sem, color=color, alpha=0.2, lw=0)
            ax.set_title(TITRATION_MAP_LABELS[metric], fontsize=13)
            ax.set_xlabel(x_label, fontsize=12)
            ax.set_ylabel("mean mAP", fontsize=12)
            _apply_x_scale(ax, x_all, scale, tick_fontsize=10)
            ax.grid(True, alpha=0.3)

        # Single shared legend at the top
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(
                handles, labels, loc="upper center", ncol=len(labels),
                fontsize=12, frameon=False, bbox_to_anchor=(0.5, 0.99),
            )
        fig.suptitle(
            f"{title_prefix} — {x_label} ({SCALE_LABEL_SHORT[scale]})",
            fontsize=14, fontweight="bold", y=0.93,
        )

        # Footer text: matched_livecell membership.
        # - For matched_livecell_combo<N>: show only the markers that DIFFER
        #   from the first combo group present (which serves as baseline).
        # - For matched_livecell_setN / matched_livecell: show full mapping.
        # - When there are > FOOTER_MAX_LINES combo groups, drop the per-combo
        #   footer (it would otherwise be illegible) and let the reader use
        #   matched_set_membership.csv next to the plots.
        FOOTER_MAX_LINES = 6
        footer_lines: List[str] = []
        if matched_set_membership:
            combo_groups = [g for g in dfs.keys()
                            if g in matched_set_membership and "combo" in g]
            baseline_combo = combo_groups[0] if combo_groups else None
            baseline_map = (
                matched_set_membership[baseline_combo] if baseline_combo else {}
            )
            if len(combo_groups) > FOOTER_MAX_LINES:
                if baseline_combo:
                    items = ", ".join(
                        f"{o}={s}" for o, s in baseline_map.items()
                    )
                    footer_lines.append(f"{baseline_combo} (baseline): {items}")
                footer_lines.append(
                    f"+ {len(combo_groups) - 1} additional combo groups — see "
                    f"matched_set_membership.csv for per-combo swap details"
                )
                # Plus any non-combo matched_livecell groups
                for g in dfs.keys():
                    if g not in matched_set_membership or "combo" in g:
                        continue
                    mp = matched_set_membership[g]
                    items = ", ".join(f"{o}={s}" for o, s in mp.items())
                    footer_lines.append(f"{g}: {items}")
            else:
                for g in dfs.keys():
                    if g not in matched_set_membership:
                        continue
                    mp = matched_set_membership[g]
                    if g == baseline_combo:
                        items = ", ".join(f"{o}={s}" for o, s in mp.items())
                        footer_lines.append(f"{g} (baseline): {items}")
                    elif baseline_map and "combo" in g:
                        swaps = [
                            f"{o}={s}"
                            for o, s in mp.items()
                            if baseline_map.get(o) != s
                        ]
                        diff = "(no swaps)" if not swaps else "swaps: " + ", ".join(swaps)
                        footer_lines.append(f"{g} {diff}")
                    else:
                        items = ", ".join(f"{o}={s}" for o, s in mp.items())
                        footer_lines.append(f"{g}: {items}")
        bottom_pad = 0.08 + 0.025 * len(footer_lines)
        if footer_lines:
            fig.text(
                0.5, 0.005 + 0.022 * (len(footer_lines) - 1),
                "\n".join(footer_lines),
                ha="center", va="bottom", fontsize=9, family="monospace",
            )
            fig.tight_layout(rect=(0, bottom_pad, 1, 0.91))
        else:
            fig.tight_layout(rect=(0, 0, 1, 0.91))
        stem = output_dir / f"compare_all_metrics_{x_col}_{scale}"
        fig.savefig(stem.with_suffix(".png"), dpi=160, bbox_inches="tight")
        fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
        plt.close(fig)
        logger.info("Wrote %s.png/svg", stem)

    # When there are several matched_livecell_combo<N> groups, the all-curves
    # canvas above gets visually crowded. Add two alternative views below:
    # an envelope summary (combos collapsed to median + min/max) and a
    # by-swap-count view (combos colored by Hamming distance from baseline).
    combo_dfs = {g: d for g, d in dfs.items() if _parse_matched_combo_group(g) is not None}
    other_dfs = {g: d for g, d in dfs.items() if g not in combo_dfs}
    if len(combo_dfs) >= 3 and matched_set_membership:
        _plot_combo_envelope(
            combo_dfs, other_dfs, output_dir, x_col, x_label, x_all,
            title_prefix=title_prefix, palette=palette, fallback=fallback,
        )
        _plot_combo_by_swap_count(
            combo_dfs, other_dfs, matched_set_membership, output_dir,
            x_col, x_label, x_all,
            title_prefix=title_prefix, palette=palette, fallback=fallback,
        )
        _plot_marker_swap_effects(
            combo_dfs, matched_set_membership, output_dir,
            x_col, x_label, x_all, title_prefix=title_prefix,
        )


def _plot_combo_envelope(
    combo_dfs: Dict[str, pd.DataFrame],
    other_dfs: Dict[str, pd.DataFrame],
    output_dir: Path,
    x_col: str,
    x_label: str,
    x_all,
    *,
    title_prefix: str,
    palette: Dict[str, str],
    fallback: List[str],
) -> None:
    """Combos collapsed into a min/max band + median; cp/4i (and any other
    non-combo groups) drawn as solid prominent curves on top.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Build per-x stats across combo curves on a shared sorted x-grid
    x_grid = np.array(sorted({float(v) for d in combo_dfs.values() for v in d[x_col].values}))
    if x_grid.size < 2:
        return

    def _series_at_grid(d: pd.DataFrame, ycol: str) -> np.ndarray:
        if ycol not in d.columns:
            return np.full_like(x_grid, np.nan, dtype=float)
        ord_ = d.sort_values(x_col)
        x = ord_[x_col].to_numpy(dtype=float)
        y = ord_[ycol].to_numpy(dtype=float)
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 2:
            return np.full_like(x_grid, np.nan, dtype=float)
        # Use linear interpolation on log10 of x to match how the curves are read
        lx, ly = np.log10(np.clip(x[ok], 1e-9, None)), y[ok]
        lt = np.log10(np.clip(x_grid, 1e-9, None))
        return np.interp(lt, lx, ly, left=np.nan, right=np.nan)

    # Long-format CSV with combo aggregate stats + named-group raw points
    long_rows: List[Dict] = []
    for scale in SCALES:
        fig, axes = plt.subplots(1, 4, figsize=(22, 6), sharex=True)
        for ax, metric in zip(axes, METRICS):
            ycol = f"{metric}_map_mean"
            stack = np.vstack([_series_at_grid(d, ycol) for d in combo_dfs.values()])
            with np.errstate(all="ignore"):
                med = np.nanmedian(stack, axis=0)
                lo = np.nanmin(stack, axis=0)
                hi = np.nanmax(stack, axis=0)
            for j, (xv, m, l, h) in enumerate(zip(x_grid, med, lo, hi)):
                if scale == "log10":
                    long_rows.append({
                        "metric": metric, x_col: float(xv),
                        "combo_min": float(l), "combo_median": float(m),
                        "combo_max": float(h),
                    })
            ax.fill_between(x_grid, lo, hi, color="#9ca3af", alpha=0.25, lw=0,
                            label=f"combos min–max (n={len(combo_dfs)})")
            ax.plot(x_grid, med, color="#374151", lw=2.5, ls="--",
                    label="combos median")
            for i, (g, df) in enumerate(other_dfs.items()):
                if x_col not in df.columns or ycol not in df.columns:
                    continue
                color = palette.get(g) or fallback[i % len(fallback)]
                xx = df[x_col].to_numpy()
                yy = df[ycol].to_numpy()
                ax.plot(xx, yy, marker="o", lw=3.5, ms=8, color=color, label=g)
                sem_col = f"{ycol}_sem"
                if sem_col in df.columns:
                    sem = df[sem_col].to_numpy()
                    ax.fill_between(xx, yy - sem, yy + sem, color=color, alpha=0.2, lw=0)
            ax.set_title(TITRATION_MAP_LABELS[metric], fontsize=13)
            ax.set_xlabel(x_label, fontsize=12)
            ax.set_ylabel("mean mAP", fontsize=12)
            _apply_x_scale(ax, x_all, scale, tick_fontsize=10)
            ax.grid(True, alpha=0.3)
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center", ncol=len(labels),
                       fontsize=11, frameon=False, bbox_to_anchor=(0.5, 0.99))
        fig.suptitle(
            f"{title_prefix} — combo envelope (median + min/max) — "
            f"{x_label} ({SCALE_LABEL_SHORT[scale]})",
            fontsize=14, fontweight="bold", y=0.93,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.91))
        stem = output_dir / f"compare_envelope_{x_col}_{scale}"
        fig.savefig(stem.with_suffix(".png"), dpi=160, bbox_inches="tight")
        fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
        plt.close(fig)
        logger.info("Wrote %s.png/svg", stem)
    if long_rows:
        pd.DataFrame(long_rows).to_csv(
            output_dir / f"compare_envelope_{x_col}.csv", index=False,
        )


def _plot_marker_swap_effects(
    combo_dfs: Dict[str, pd.DataFrame],
    matched_set_membership: Dict[str, Dict[str, str]],
    output_dir: Path,
    x_col: str,
    x_label: str,
    x_all,
    *,
    title_prefix: str,
) -> None:
    """Per-organelle "swap effect" view: for each organelle that has >1
    distinct marker across combos, group the combos by which marker they use
    for that organelle, and plot mean ± SEM mAP (across the OTHER swap
    dimensions). Reveals how much each individual marker swap moves the
    needle, marginalizing across all other choices.

    Layout: rows = organelle (variable markers only), cols = metric (4).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # organelle → list of distinct markers seen across combos
    organelle_markers: Dict[str, List[str]] = {}
    for g, mp in matched_set_membership.items():
        if g not in combo_dfs:
            continue
        for organelle, sig in mp.items():
            organelle_markers.setdefault(organelle, [])
            if sig not in organelle_markers[organelle]:
                organelle_markers[organelle].append(sig)
    swap_organelles = [o for o, ms in organelle_markers.items() if len(ms) > 1]
    if not swap_organelles:
        return

    # Long-format: organelle × marker × metric × x → mean, sem, n_combos
    long_rows: List[Dict] = []
    for organelle in swap_organelles:
        for marker in organelle_markers[organelle]:
            combos_with_marker = [
                g for g, mp in matched_set_membership.items()
                if g in combo_dfs and mp.get(organelle) == marker
            ]
            if not combos_with_marker:
                continue
            stack_per_metric = {m: [] for m in METRICS}
            x_grid: Optional[np.ndarray] = None
            for g in combos_with_marker:
                d = combo_dfs[g].sort_values(x_col)
                if x_col not in d.columns:
                    continue
                if x_grid is None:
                    x_grid = d[x_col].to_numpy(dtype=float)
                for metric in METRICS:
                    ycol = f"{metric}_map_mean"
                    if ycol in d.columns:
                        stack_per_metric[metric].append(d[ycol].to_numpy(dtype=float))
            if x_grid is None:
                continue
            for metric in METRICS:
                if not stack_per_metric[metric]:
                    continue
                arr = np.vstack(stack_per_metric[metric])
                with np.errstate(all="ignore"):
                    mean = np.nanmean(arr, axis=0)
                    sem = (np.nanstd(arr, axis=0, ddof=1)
                           / np.sqrt(np.sum(np.isfinite(arr), axis=0).clip(min=1)))
                for j, xv in enumerate(x_grid):
                    long_rows.append({
                        "organelle": organelle, "marker": marker,
                        "metric": metric, x_col: float(xv),
                        "y_mean": float(mean[j]),
                        "y_sem": float(sem[j]) if np.isfinite(sem[j]) else float("nan"),
                        "n_combos": int(arr.shape[0]),
                    })
    if not long_rows:
        return
    csv_path = output_dir / f"swap_effects_{x_col}.csv"
    pd.DataFrame(long_rows).to_csv(csv_path, index=False)
    logger.info(f"  Wrote {csv_path}")

    df_long = pd.DataFrame(long_rows)
    nrows = len(swap_organelles)
    ncols = len(METRICS)
    for scale in SCALES:
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(5.5 * ncols, 3.5 * nrows),
            sharex=True, squeeze=False,
        )
        for r, organelle in enumerate(swap_organelles):
            markers = organelle_markers[organelle]
            colors = ["#d97706", "#2563eb", "#10b981", "#7c3aed"][: len(markers)]
            for c, metric in enumerate(METRICS):
                ax = axes[r][c]
                sub = df_long[
                    (df_long["organelle"] == organelle)
                    & (df_long["metric"] == metric)
                ]
                for marker, color in zip(markers, colors):
                    sm = sub[sub["marker"] == marker].sort_values(x_col)
                    if sm.empty:
                        continue
                    n = int(sm["n_combos"].iloc[0])
                    ax.plot(
                        sm[x_col].to_numpy(), sm["y_mean"].to_numpy(),
                        marker="o", lw=2.0, color=color,
                        label=f"{marker} (n={n})",
                    )
                    if "y_sem" in sm.columns:
                        y = sm["y_mean"].to_numpy()
                        e = sm["y_sem"].to_numpy()
                        ax.fill_between(
                            sm[x_col].to_numpy(), y - e, y + e,
                            color=color, alpha=0.2, lw=0,
                        )
                ax.grid(True, alpha=0.3)
                _apply_x_scale(ax, x_all, scale, tick_fontsize=9)
                ax.set_ylabel("mean mAP", fontsize=10)
                if r == 0:
                    ax.set_title(TITRATION_MAP_LABELS[metric], fontsize=12)
                if r == nrows - 1:
                    ax.set_xlabel(x_label, fontsize=10)
                if c == 0:
                    ax.text(
                        -0.18, 0.5, organelle, transform=ax.transAxes,
                        ha="right", va="center", fontsize=12, fontweight="bold",
                    )
                ax.legend(fontsize=8, loc="best")
        fig.suptitle(
            f"{title_prefix} — marker swap effects (mean ± SEM across the other "
            f"{nrows - 1} organelle's markers) — {x_label} ({SCALE_LABEL_SHORT[scale]})",
            fontsize=13, fontweight="bold", y=0.995,
        )
        fig.tight_layout(rect=(0.04, 0, 1, 0.97))
        stem = output_dir / f"compare_swap_effects_{x_col}_{scale}"
        fig.savefig(stem.with_suffix(".png"), dpi=160, bbox_inches="tight")
        fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
        plt.close(fig)
        logger.info("Wrote %s.png/svg", stem)


def _plot_combo_by_swap_count(
    combo_dfs: Dict[str, pd.DataFrame],
    other_dfs: Dict[str, pd.DataFrame],
    matched_set_membership: Dict[str, Dict[str, str]],
    output_dir: Path,
    x_col: str,
    x_label: str,
    x_all,
    *,
    title_prefix: str,
    palette: Dict[str, str],
    fallback: List[str],
) -> None:
    """Color combo curves by Hamming distance (# swaps) from the baseline combo
    so visually-close combos differ from baseline at similar dimensions.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.colors import Normalize

    # First combo group becomes the baseline (consistent with footer logic)
    baseline_combo = next(iter(combo_dfs))
    baseline_map = matched_set_membership.get(baseline_combo, {})
    swap_counts: Dict[str, int] = {}
    for g in combo_dfs:
        mp = matched_set_membership.get(g, {})
        n_swaps = sum(
            1 for o, s in mp.items() if baseline_map.get(o) != s
        )
        swap_counts[g] = n_swaps
    if not swap_counts:
        return
    max_swaps = max(swap_counts.values())
    cmap = cm.get_cmap("viridis")
    norm = Normalize(vmin=0, vmax=max(max_swaps, 1))

    # Dump the swap-count manifest so the legend's color encoding is reproducible
    pd.DataFrame([
        {"group": g, "n_swaps_from_baseline": n}
        for g, n in swap_counts.items()
    ]).to_csv(output_dir / "compare_by_swap_count.csv", index=False)

    for scale in SCALES:
        fig, axes = plt.subplots(1, 4, figsize=(22, 6), sharex=True)
        for ax, metric in zip(axes, METRICS):
            ycol = f"{metric}_map_mean"
            for g, df in combo_dfs.items():
                if x_col not in df.columns or ycol not in df.columns:
                    continue
                color = cmap(norm(swap_counts[g]))
                ax.plot(
                    df[x_col].to_numpy(), df[ycol].to_numpy(),
                    color=color, lw=1.0, alpha=0.6,
                )
            for i, (g, df) in enumerate(other_dfs.items()):
                if x_col not in df.columns or ycol not in df.columns:
                    continue
                color = palette.get(g) or fallback[i % len(fallback)]
                xx = df[x_col].to_numpy()
                yy = df[ycol].to_numpy()
                ax.plot(xx, yy, marker="o", lw=3.5, ms=8, color=color, label=g)
            ax.set_title(TITRATION_MAP_LABELS[metric], fontsize=13)
            ax.set_xlabel(x_label, fontsize=12)
            ax.set_ylabel("mean mAP", fontsize=12)
            _apply_x_scale(ax, x_all, scale, tick_fontsize=10)
            ax.grid(True, alpha=0.3)
        # cp/4i legend
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center",
                       ncol=len(labels), fontsize=11, frameon=False,
                       bbox_to_anchor=(0.45, 0.99))
        # Shared colorbar for swap count
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(
            sm, ax=axes, location="right", pad=0.01, fraction=0.018,
            ticks=range(0, max_swaps + 1),
        )
        cbar.set_label(f"# marker swaps from {baseline_combo}", fontsize=11)
        fig.suptitle(
            f"{title_prefix} — combos colored by swap distance from baseline — "
            f"{x_label} ({SCALE_LABEL_SHORT[scale]})",
            fontsize=14, fontweight="bold", y=0.95,
        )
        stem = output_dir / f"compare_by_swap_count_{x_col}_{scale}"
        fig.savefig(stem.with_suffix(".png"), dpi=160, bbox_inches="tight")
        fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
        plt.close(fig)
        logger.info("Wrote %s.png/svg", stem)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """Reuse pca_titration's parser to share path-resolution flags, then add ours."""
    p = _titr_parser()
    p.description = "Combined-reporter cell-count titration."
    p.add_argument(
        "--groups", type=str, default="cp,4i,matched_livecell",
        help="Comma-separated groups to titrate. Built-ins: cp, 4i, all, "
             "matched_livecell, custom (default: cp,4i,matched_livecell)",
    )
    p.add_argument(
        "--custom-paths", type=str, default=None,
        help="When --groups includes 'custom', comma-separated absolute paths to *_cells.h5ad",
    )
    p.add_argument(
        "--matching-config", type=str, default=DEFAULT_MATCHING_CONFIG,
        help=f"YAML mapping CP organelles to matching live-cell channels for "
             f"the 'matched_livecell' group (default: {DEFAULT_MATCHING_CONFIG})",
    )
    p.add_argument(
        "--no-compare", action="store_true",
        help="Skip the cross-group comparison plot at the end",
    )
    p.add_argument(
        "--compare-only", action="store_true",
        help="Skip the per-group titration step entirely and just regenerate "
             "the comparison plots from the existing per-group CSVs. "
             "Errors out if any expected combined_titration_<group>.csv is missing.",
    )
    p.add_argument(
        "--no-shared-start", dest="shared_start", action="store_false", default=True,
        help="In --per-guide-median-titration mode, let each group start at its "
             "own median instead of the smallest median across groups (default: "
             "share the start so curves align at the top).",
    )
    # --no-cache is inherited from pca_titration's parser (same semantics).
    p.add_argument(
        "--seed", type=int, default=42, help="Random seed for cell subsampling",
    )
    return p


def _sampling_mode(args: argparse.Namespace) -> str:
    if getattr(args, "per_guide_median_titration", False):
        return "per_guide_median"
    if args.per_guide_max_titration or args.per_guide_min_titration:
        return "per_guide"
    if args.per_ko_max_titration or args.per_ko_min_titration:
        return "per_ko"
    return "total"


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = _build_parser()
    args = parser.parse_args()
    groups = [g.strip() for g in args.groups.split(",") if g.strip()]
    groups = _expand_groups_with_combos(args, groups)
    custom_paths = (
        [p.strip() for p in args.custom_paths.split(",") if p.strip()]
        if args.custom_paths else None
    )
    sampling_mode = _sampling_mode(args)

    # Resolve paths + output dirs per group up front so we can submit jobs.
    group_paths: Dict[str, List[Path]] = {}
    group_outdirs: Dict[str, Path] = {}
    matched_set_membership: Dict[str, Dict[str, str]] = {}
    compare_only = getattr(args, "compare_only", False)
    # In --compare-only mode we skip group_paths resolution (which calls
    # _enumerate_marker_combinations once per group → quadratic in #combos).
    if compare_only:
        # Enumerate combos once and reuse for membership lookups.
        all_combos = _enumerate_marker_combinations(args, args.matching_config)
        combo_lookup = {c["group_name"]: c for c in all_combos}
    for g in groups:
        if compare_only:
            group_paths[g] = []
            group_outdirs[g] = _resolve_group_output_dir(args, g, sampling_mode)
            print(f"[{g}] → {group_outdirs[g]}")
            combo_idx = _parse_matched_combo_group(g)
            if combo_idx is not None and g in combo_lookup:
                matched_set_membership[g] = dict(combo_lookup[g]["organelle_to_signal"])
            elif _parse_matched_livecell_group(g) is not None:
                m = _resolve_matched_set_membership(args, g, args.matching_config)
                if m:
                    matched_set_membership[g] = m
            continue
        paths = _resolve_group_paths(
            args, g, custom_paths=custom_paths,
            matching_config=args.matching_config,
        )
        group_paths[g] = paths
        group_outdirs[g] = _resolve_group_output_dir(args, g, sampling_mode)
        print(f"[{g}] {len(paths)} reporters → {group_outdirs[g]}")
        m = _resolve_matched_set_membership(args, g, args.matching_config)
        if m:
            matched_set_membership[g] = m

    # For median mode with multiple groups, by default cap every group's start
    # (the median) at the smallest median across groups so curves align at the
    # top of the x-axis. Pass --no-shared-start to let each group start at its
    # own median instead. Every group titrates down to 1 cell/guide regardless.
    # In --compare-only mode we never run the schedule, so skip the expensive
    # median computation (which opens every reporter h5ad in backed mode).
    group_schedules: Dict[str, Optional[List[int]]] = {g: None for g in groups}
    if (
        sampling_mode == "per_guide_median"
        and len(groups) > 1
        and not getattr(args, "compare_only", False)
    ):
        medians = {g: _per_guide_median(group_paths[g]) for g in groups}
        if args.shared_start:
            shared_start = min(medians.values())
            print(
                f"\nMedian-mode shared start: min median across groups = "
                f"{shared_start} (per-group medians: {medians})"
            )
            for g in groups:
                group_schedules[g] = _build_per_guide_median_schedule(
                    group_paths[g], start_override=shared_start,
                )
                print(f"  [{g}] schedule ({len(group_schedules[g])} pts): {group_schedules[g]}")
        else:
            print(
                f"\nMedian-mode per-group starts: each group runs from its own "
                f"median to 1 (medians: {medians})"
            )
            for g in groups:
                group_schedules[g] = _build_per_guide_median_schedule(group_paths[g])
                print(f"  [{g}] schedule ({len(group_schedules[g])} pts): {group_schedules[g]}")

    # Run each group (locally or in parallel SLURM)
    csvs_by_group: Dict[str, Path] = {}
    if getattr(args, "compare_only", False):
        # Skip the per-group step entirely; just collect existing CSV paths.
        missing = []
        for g in groups:
            csv = group_outdirs[g] / f"combined_titration_{g}.csv"
            if csv.is_file():
                csvs_by_group[g] = csv
            else:
                missing.append(str(csv))
        if missing:
            raise SystemExit(
                f"--compare-only: {len(missing)} expected per-group CSV(s) missing. "
                f"Run without --compare-only first. Missing:\n  "
                + "\n  ".join(missing[:10])
                + (f"\n  ... (+{len(missing) - 10} more)" if len(missing) > 10 else "")
            )
        print(f"\n--compare-only: skipping per-group step; loading {len(csvs_by_group)} cached CSVs.")
    elif args.slurm:
        from ops_utils.hpc.slurm_batch_utils import (
            submit_parallel_jobs,
            wait_for_multiple_job_arrays,
        )

        base_slurm_params = {
            "mem": args.slurm_memory,
            "cpus_per_task": args.slurm_cpus,
            "slurm_partition": args.slurm_partition,
        }
        # Auto-bump timeout for any group that includes the Phase reporter
        # (~60M cells / 25GB) to args.phase_slurm_time (default 240min). All
        # reporters in a combined-titration group are h-concatted into one job,
        # so we can't split Phase out the way pca_titration does.
        def _has_phase(paths: List[Path]) -> bool:
            return any("phase" in p.stem.lower() for p in paths)

        job_arrays = []
        for g in groups:
            phase_in_group = _has_phase(group_paths[g])
            timeout_min = args.phase_slurm_time if phase_in_group else args.slurm_time
            # Mirror pca_titration's bootstrap autoscaling for the Phase budget
            if (
                phase_in_group and args.bootstrap > 1
                and timeout_min == parser.get_default("phase_slurm_time")
            ):
                timeout_min = timeout_min * int(args.bootstrap)
            slurm_params = {**base_slurm_params, "timeout_min": timeout_min}
            job = {
                "name": f"combtitr_{g}",
                "func": run_combined_titration,
                "kwargs": {
                    "cells_h5ad_paths": [str(p) for p in group_paths[g]],
                    "output_dir": str(group_outdirs[g]),
                    "sampling_mode": sampling_mode,
                    "norm_method": args.norm_method,
                    "distance": args.distance,
                    "n_bootstraps": int(args.bootstrap),
                    "random_seed": int(args.seed),
                    "group_label": g,
                    "cache": bool(args.cache),
                    "schedule": group_schedules[g],
                },
            }
            phase_tag = " (Phase reporter present — bumped time)" if phase_in_group else ""
            print(
                f"\nSubmitting combined-titration SLURM job for group={g} "
                f"({timeout_min}min, {args.slurm_memory}){phase_tag}..."
            )
            result = submit_parallel_jobs(
                jobs_to_submit=[job],
                experiment="pca_combined_titration",
                slurm_params=slurm_params,
                log_dir="pca_optimization",
                manifest_prefix=f"pca_combtitr_{g}",
                wait_for_completion=False,
            )
            if result.get("submitted_jobs"):
                job_arrays.append({
                    "submitted_jobs": result["submitted_jobs"],
                    "base_job_id": result["base_job_id"],
                    "label": g,
                    "slurm_params": slurm_params,
                })
            csvs_by_group[g] = (
                group_outdirs[g] / f"combined_titration_{g}.csv"
            )

        if job_arrays:
            wait_for_multiple_job_arrays(job_arrays, experiment="pca_combined_titration")
    else:
        for g in groups:
            run_combined_titration(
                cells_h5ad_paths=[str(p) for p in group_paths[g]],
                output_dir=str(group_outdirs[g]),
                sampling_mode=sampling_mode,
                norm_method=args.norm_method,
                distance=args.distance,
                n_bootstraps=int(args.bootstrap),
                random_seed=int(args.seed),
                group_label=g,
                cache=bool(args.cache),
                schedule=group_schedules[g],
            )
            csvs_by_group[g] = group_outdirs[g] / f"combined_titration_{g}.csv"

    # Cross-group comparison
    if not args.no_compare and len(csvs_by_group) >= 2:
        compare_dir = _resolve_compare_dir(args, groups, sampling_mode)
        compare_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nComparison plots → {compare_dir}")
        # Drop a manifest listing every group included (since the leaf folder
        # name collapses combo runs into `combosN-M`).
        (compare_dir / "groups.txt").write_text("\n".join(list(csvs_by_group.keys())) + "\n")
        plot_group_comparison(
            {g: p for g, p in csvs_by_group.items()},
            output_dir=compare_dir,
            sampling_mode=sampling_mode,
            title_prefix=f"Combined titration ({sampling_mode})",
            matched_set_membership=matched_set_membership,
        )
        # Always dump set membership next to the comparison plot, even when
        # no matched_livecell groups are present (for full provenance).
        if matched_set_membership:
            rows = []
            for g, mp in matched_set_membership.items():
                for organelle, sig in mp.items():
                    rows.append({"group": g, "organelle": organelle, "signal": sig})
            pd.DataFrame(rows).to_csv(compare_dir / "matched_set_membership.csv", index=False)

    print("\nDone.")


if __name__ == "__main__":
    main()
