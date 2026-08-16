"""Native-160 LOSSLESS-float morpho measurement + single real-NTC baseline check + seg-overlay debug panel,
for the 3 Fig-4 groups (mTOR lysosome, POLR1B phase-nucleoli, TIM23 mito).

Both sides measured IDENTICALLY: same 160px window, same production seg config (adaptive=False), same dilated
real cell_seg mask, same feature extractor (process_single_cell at spacing (1,1) → pixel units), so any gen-vs-real
offset is real, not a units/pipeline artifact.
  generated → cell{c}/frames_f32.npz (model's native 160, no webp/upsample) → parametrized full_features
              (crop=160, float_frames=True, cell_masks=<dilated real cell_seg>) → read the mini-zarr labels back.
  real      → the stored production org seg (gfp_seg / nucleoli_phase2d_seg / mcherry_seg) cropped at the cell's
              160 window, same dilated mask.

STANDARD (per feature): (1) gen α0 within ~5% of real NTC, (2) real KO >25% over real NTC,
(3) gen α1 within 10% of real KO, (4) α3 > α1.

Run (SLURM, all 3): python morpho_native.py --submit    |    local one group: python morpho_native.py mtor
Render panels (after jobs): python morpho_native.py --panel
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import zarr
from scipy.ndimage import binary_dilation
from skimage.transform import resize

from _setacc_common import _materialize
from ops_model.models.attention.diffex.viewer.build_pc_crops_masked import BASE, _crop, _zarr_patch
from ops_model.models.attention.diffex.viewer.morpho_pipeline import (
    GEN_CROP, PAD, MO_PARAMS, MORPHO_TARGETS, _clip_border, _seg_masked_object, _vs_h2b_nucleus_npz, full_features)

NUCLEOLI_MO = {**MO_PARAMS, "min_object_size": 25, "mo_object_min_area_px": 25}   # size-exclude VS-NPM3 noise blobs
NUCLEOLI_MO_STRICT = {**NUCLEOLI_MO, "mo_local_adjust": 1.35}                     # higher mo_local_adjust = stricter local threshold (MO_PARAMS default 1.3); gen-only, real stays on NUCLEOLI_MO.
                                                                                   # sweet spot confirmed via single-cell preview: 1.5+ detects nothing at any α; 1.4 matches lenient at α=0/1 but
                                                                                   # correctly drops the unreliable high-α signal (zero at α>=2.5) instead of lenient's noisy partial detections there


VA = "/hpc/projects/icd.fast.ops/models/diffex/viewer_assets_v5"
OUT = "/hpc/projects/icd.fast.ops/analysis/figure4_traversals_violin/_native"
SYN = "/hpc/projects/icd.fast.ops/models/diffex/morpho_synth_native"
HALF = GEN_CROP // 2                                 # 80 → 160px native window
_ALPHA_GRID = [-5, -4, -3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5]   # the real 17-pt grid: 0.5-step in [-3,3], 1.0-step outside -- NOT uniform
_ALPHA_IDX = {a: i for i, a in enumerate(_ALPHA_GRID)}
_aidx = lambda a: _ALPHA_IDX[a]                       # alpha value -> alpha_idx (exact match required -- the grid is fixed, not continuous)
DEFAULT_ALPHAS = (0, 0.5, 1, 1.5, 2, 2.5, 3)
TIM23 = "TIM23 mitochondrial inner membrane pre-sequence translocase complex, TIM17A variant"
N_PANEL = 100                                         # cells cached with full img/lab/mask into panel.npz (candidate pool for downstream picking)
PANEL_GRID_COLS = 6                                   # debug-image grid width — kept small regardless of N_PANEL so _render_panel stays a quick sanity check
NET = {"connectivity": "largest_connected_component_size", "degree": "average_degree", "branches": "num_branches", "nodes": "num_nodes"}

_LYSO = dict(modality="lysosome_LysoTracker_live_cell_dye", ch="GFP", network=False,
             ntc_rank="fluor_shap/geneKO/lysosome_LysoTracker_live_cell_dye",
             ko_rank="fluor_shap/geneKO/lysosome_LysoTracker_live_cell_dye", ko_gene="MTOR", real_label="gfp_seg", location=True)
_POLR1B = dict(mt="POLR1B_NUCLEOLI_PHASE", modality="phase", ch="Phase2D", network=False,
               ntc_rank="pma_shap_phase_geneKO", ko_rank="pma_shap_phase_geneKO", ko_gene="POLR1B",
               real_label="nucleoli_phase2d_seg", shape=True, mask_by_cell=False, drop_empty=True)   # VS-nucleus localizes; drop failed-nucleus cells
_TIM23 = dict(mt="CHROMALIVE_TIM23", modality="mitochondria_ChromaLIVE_561_excitation", ch="mCherry", network=True,
              ntc_rank="fluor_shap/geneKO/mitochondria_ChromaLIVE_561_excitation",
              ko_rank="fluor_shap/complex/mitochondria_ChromaLIVE_561_excitation", ko_gene=TIM23, real_label="mcherry_seg",
              alpha_range=(-1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3),   # extend below 0 -> real NTC (degree) sits above the α=0..3 curve, need negative α to show the crossing
              hist_match=True,   # rebalance gen intensity profile -> real-NTC reference (same rescaling mTOR used); untried for TIM23 so far
              min_obj_px=15)   # drop sub-visual segmentation-noise fragments (<15px) from "count" -- real NTC picks up ~2x more of these than gen (sensor noise vs. diffusion smoothing), inflating count without being visually apparent (15 replaces an earlier 25 trial; 10 also tested in a separate min_obj10 variant)
# SAMM50/MICOS13 (MICOS complex, mitochondria ChromaLIVE561) -- same seg params/checkpoint/shared NTC anchor
# pool as TIM23, individual geneKO ranking (not complex-pooled) for both ntc_rank and ko_rank.
_SAMM50 = dict(mt="CHROMALIVE_SAMM50", modality="mitochondria_ChromaLIVE_561_excitation", ch="mCherry", network=True,
              ntc_rank="fluor_shap/geneKO/mitochondria_ChromaLIVE_561_excitation",
              ko_rank="fluor_shap/geneKO/mitochondria_ChromaLIVE_561_excitation", ko_gene="SAMM50", real_label="mcherry_seg",
              hist_match=True, min_obj_px=15)
_MICOS13 = dict(mt="CHROMALIVE_MICOS13", modality="mitochondria_ChromaLIVE_561_excitation", ch="mCherry", network=True,
              ntc_rank="fluor_shap/geneKO/mitochondria_ChromaLIVE_561_excitation",
              ko_rank="fluor_shap/geneKO/mitochondria_ChromaLIVE_561_excitation", ko_gene="MICOS13", real_label="mcherry_seg",
              hist_match=True, min_obj_px=15)

# group → measurement config; seg params pulled from MORPHO_TARGETS[mt]. These 3 are the final Fig-4 picks
# (mtor_mo_hm_100, polr1b_vsnpm3_100cpu, tim23_100 in _native/) that raw_alpha_violins.py/raw_alpha_panels.py read.
GROUPS = {
    "mtor_mo_hm": dict(mt="MTOR_LYSO", **_LYSO,                                   # back to MTOR_LYSO's default blob detection (masked_object detour reverted, was working fine before).
                       seg_override=dict(frangi_override=dict(threshold=0.05))),  # light touch: default 0.03 lets spurious weak local maxima blow up α=0's count specifically; 0.05 calms
                                                                                   # that down (confirmed close to the other alphas) while keeping overall density close to the original 0.03 look
                                                                                   # (0.1 also fixes the α=0 spike but pushes overall density down much further from how it looked before)
    "mtor_mo_hm_histmatch": dict(mt="MTOR_LYSO", **_LYSO, hist_match=True,        # same threshold fix + hist-match rescale, to compare against threshold alone
                       seg_override=dict(frangi_override=dict(threshold=0.05))),
    "polr1b_vsnpm3": dict(mt="POLR1B", modality="vs_npm3_from_phase", ch="Phase2D", network=False,   # VS-NPM3 is the MEASUREMENT TOOL, applied symmetrically to real-phase AND gen-phase
                          ntc_rank="pma_shap_phase_geneKO", ko_rank="pma_shap_phase_geneKO",           # phase ranking (same pool as plain polr1b) — NOT the NPM3-fluor ranking (different modality/cells)
                          ko_gene="POLR1B", shape=True, mask_by_cell=False, drop_empty=True, vs_real=True, vs_real_n=100,   # match gen ncell=100 exactly
                          vs_nuc_from="phase", mo_params=NUCLEOLI_MO, panel_src="phase", gen_cell_offset=200,   # panel shows PHASE under the VS-NPM3-derived seg; cells 200-299 = multirank top-100
                          seg_override=dict(marker_dir="vs_npm3_from_phase", seg_method="masked_object", structure_type="vesicular",
                                            mo_nucleus=True, mo_vs_nucleus=True, mo_vs_erode=4, frangi_override=None)),
    "polr1b_vsnpm3_stringent": dict(mt="POLR1B", modality="vs_npm3_from_phase", ch="Phase2D", network=False,   # same VS method as polr1b_vsnpm3 (vs_real, seg_override, mo_params for REAL all unchanged) --
                          ntc_rank="pma_shap_phase_geneKO", ko_rank="pma_shap_phase_geneKO",                     # ONLY gen_mo_params differs, applying stricter MO to generated cells alone
                          ko_gene="POLR1B", shape=True, mask_by_cell=False, drop_empty=True, vs_real=True, vs_real_n=1000,   # top-1000, matching mTOR/TIM23's real population size
                          vs_nuc_from="phase", mo_params=NUCLEOLI_MO, gen_mo_params=NUCLEOLI_MO_STRICT, panel_src="phase", gen_cell_offset=200,   # real=1.3, gen=1.35
                          seg_override=dict(marker_dir="vs_npm3_from_phase", seg_method="masked_object", structure_type="vesicular",
                                            mo_nucleus=True, mo_vs_nucleus=True, mo_vs_erode=4, frangi_override=None)),
    "taf1b_vsnpm3_stringent": dict(mt="TAF1B", modality="vs_npm3_from_phase", ch="Phase2D", network=False,   # same VS-NPM3/stringent-MO method as polr1b_vsnpm3_stringent, f=1.25
                          ntc_rank="pma_shap_phase_geneKO", ko_rank="pma_shap_phase_geneKO",
                          ko_gene="TAF1B", shape=True, mask_by_cell=False, drop_empty=True, vs_real=True, vs_real_n=1000,
                          vs_nuc_from="phase", mo_params=NUCLEOLI_MO, gen_mo_params=NUCLEOLI_MO_STRICT, panel_src="phase", gen_cell_offset=200,   # real=1.3, gen=1.35 -- anchor slots 200-299 (0-99 are stale, predate the current ranking)
                          seg_override=dict(marker_dir="vs_npm3_from_phase", seg_method="masked_object", structure_type="vesicular",
                                            mo_nucleus=True, mo_vs_nucleus=True, mo_vs_erode=4, frangi_override=None)),
    "taf1b_vsnpm3_stringent_oldanchor": dict(mt="TAF1B", modality="vs_npm3_from_phase", ch="Phase2D", network=False,   # gen-only comparison: SAME everything, gen_cell_offset=0 (the stale anchor range) instead of 200
                          ntc_rank="pma_shap_phase_geneKO", ko_rank="pma_shap_phase_geneKO",
                          ko_gene="TAF1B", shape=True, mask_by_cell=False, drop_empty=True, vs_real=True, vs_real_n=1000,
                          vs_nuc_from="phase", mo_params=NUCLEOLI_MO, gen_mo_params=NUCLEOLI_MO_STRICT, panel_src="phase", gen_cell_offset=0,
                          seg_override=dict(marker_dir="vs_npm3_from_phase", seg_method="masked_object", structure_type="vesicular",
                                            mo_nucleus=True, mo_vs_nucleus=True, mo_vs_erode=4, frangi_override=None)),
    "polr1b_vsnpm3_simplenuc14": dict(mt="POLR1B", modality="vs_npm3_from_phase", ch="Phase2D", network=False,   # test: SIMPLE auto-detected nucleus (no VS-H2B) for gen, mo_local_adjust=1.4 -- matches the quick single-cell preview methodology exactly, to see what the FULL 100-cell run gives with that approach
                          ntc_rank="pma_shap_phase_geneKO", ko_rank="pma_shap_phase_geneKO",
                          ko_gene="POLR1B", shape=True, mask_by_cell=False, drop_empty=True, vs_real=True, vs_real_n=100,
                          vs_nuc_from="phase", mo_params=NUCLEOLI_MO, gen_mo_params={**NUCLEOLI_MO, "mo_local_adjust": 1.4}, panel_src="phase", gen_cell_offset=200,
                          seg_override=dict(marker_dir="vs_npm3_from_phase", seg_method="masked_object", structure_type="vesicular",
                                            mo_nucleus=True, mo_vs_nucleus=False, frangi_override=None)),   # mo_vs_nucleus=False -> gen falls back to _nucleus_mask() auto-detect, same as the quick preview
    "polr1b_vsnpm3_histmatch": dict(mt="POLR1B", modality="vs_npm3_from_phase", ch="Phase2D", network=False, hist_match=True,   # same VS method as polr1b_vsnpm3 (vs_real, seg_override untouched), + hist-match rescale
                          ntc_rank="pma_shap_phase_geneKO", ko_rank="pma_shap_phase_geneKO",
                          ko_gene="POLR1B", shape=True, mask_by_cell=False, drop_empty=True, vs_real=True, vs_real_n=100,
                          vs_nuc_from="phase", mo_params=NUCLEOLI_MO, panel_src="phase", gen_cell_offset=200,
                          seg_override=dict(marker_dir="vs_npm3_from_phase", seg_method="masked_object", structure_type="vesicular",
                                            mo_nucleus=True, mo_vs_nucleus=True, mo_vs_erode=4, frangi_override=None)),
    "tim23":     dict(**_TIM23),
    "samm50_chromalive":  dict(**_SAMM50),
    "micos13_chromalive": dict(**_MICOS13),

    # F-rescale variants (MORPH_F_RESCALE_HANDOFF.md): φ={0,1,2,3} = α/f at each perturbation's own centroid-
    # recovery f (POLR1B=2.45, MTOR=1.38, TIM23=2.25 -- from f_centroid_recovery/f_all.json + centroid_recovery_fluor/*.json).
    # α 0/1.5/2.5/3.0 are already measured in the main group's stats.npz -- alpha_range here is ONLY the ONE
    # genuinely new grid point each perturbation needs (α=4 for mTOR, α=5 for POLR1B/TIM23); fscore_violins.py
    # merges this with the existing stats.npz instead of remeasuring the overlap. Same seg settings as the
    # validated main group in each case.
    "mtor_mo_hm_fscore": dict(mt="MTOR_LYSO", **_LYSO, alpha_range=(4.0,),                     # φ3 = α 4.0 (φ0,1,2 = α 0,1.5,3.0 already in mtor_mo_hm_100)
                       seg_override=dict(frangi_override=dict(threshold=0.05))),
    "polr1b_vsnpm3_fscore": dict(mt="POLR1B", modality="vs_npm3_from_phase", ch="Phase2D", network=False, alpha_range=(4.0, 5.0),   # φ1.5 = α 4.0, φ2 = α 5.0 (φ0,1 = α 0,2.5 -- now sourced from polr1b_vsnpm3_stringentcpu); φ3 (α=7.35) capped to the same 5.0 at render time
                          ntc_rank="pma_shap_phase_geneKO", ko_rank="pma_shap_phase_geneKO",
                          ko_gene="POLR1B", shape=True, mask_by_cell=False, drop_empty=True, vs_real=True, vs_real_n=100,
                          vs_nuc_from="phase", mo_params=NUCLEOLI_MO, gen_mo_params=NUCLEOLI_MO_STRICT, panel_src="phase", gen_cell_offset=200,   # gen_mo_params: match the validated stringent (1.4) gen-side seg
                          seg_override=dict(marker_dir="vs_npm3_from_phase", seg_method="masked_object", structure_type="vesicular",
                                            mo_nucleus=True, mo_vs_nucleus=True, mo_vs_erode=4, frangi_override=None)),
    "tim23_fscore": dict(mt="CHROMALIVE_TIM23", modality="mitochondria_ChromaLIVE_561_excitation", ch="mCherry", network=True,   # φ2 = α 5.0 (φ0,1 = α 0,2.5 already in tim23_100); φ3 capped to the same 5.0
                       ntc_rank="fluor_shap/geneKO/mitochondria_ChromaLIVE_561_excitation",
                       ko_rank="fluor_shap/complex/mitochondria_ChromaLIVE_561_excitation", ko_gene=TIM23, real_label="mcherry_seg",
                       alpha_range=(5.0,), min_obj_px=15),
    "taf1b_vsnpm3_fscore": dict(mt="TAF1B", modality="vs_npm3_from_phase", ch="Phase2D", network=False, alpha_range=(5.0,),   # f=2.25 (corrected from 1.25): only phi2 (alpha=5.0) is a new point; phi0/0.5/1/1.5 fall in taf1b_vsnpm3_stringent's base 0-3 range
                          ntc_rank="pma_shap_phase_geneKO", ko_rank="pma_shap_phase_geneKO",
                          ko_gene="TAF1B", shape=True, mask_by_cell=False, drop_empty=True, vs_real=True, vs_real_n=1000,
                          vs_nuc_from="phase", mo_params=NUCLEOLI_MO, gen_mo_params=NUCLEOLI_MO_STRICT, panel_src="phase", gen_cell_offset=200,
                          seg_override=dict(marker_dir="vs_npm3_from_phase", seg_method="masked_object", structure_type="vesicular",
                                            mo_nucleus=True, mo_vs_nucleus=True, mo_vs_erode=4, frangi_override=None)),
    "taf1b_vsnpm3_fscore_oldanchor": dict(mt="TAF1B", modality="vs_npm3_from_phase", ch="Phase2D", network=False, alpha_range=(5.0,),   # same as above but gen_cell_offset=0, for the old-anchor/combined comparison plots
                          ntc_rank="pma_shap_phase_geneKO", ko_rank="pma_shap_phase_geneKO",
                          ko_gene="TAF1B", shape=True, mask_by_cell=False, drop_empty=True, vs_real=True, vs_real_n=1000,
                          vs_nuc_from="phase", mo_params=NUCLEOLI_MO, gen_mo_params=NUCLEOLI_MO_STRICT, panel_src="phase", gen_cell_offset=0,
                          seg_override=dict(marker_dir="vs_npm3_from_phase", seg_method="masked_object", structure_type="vesicular",
                                            mo_nucleus=True, mo_vs_nucleus=True, mo_vs_erode=4, frangi_override=None)),
    # SAMM50 f=1.55, MICOS13 f=2.0 (own centroid-recovery peak-alpha, MICOS complex, real values -- not
    # borrowed from TIM23). alpha_range = the ONE new grid point each needs beyond the base 0-3 measurement.
    "samm50_chromalive_fscore":  dict(**_SAMM50, alpha_range=(5.0,)),
    "micos13_chromalive_fscore": dict(**_MICOS13, alpha_range=(4.0,)),
}

FSCORE_F = {"mtor_mo_hm_fscore": 1.38, "polr1b_vsnpm3_fscore": 2.45, "tim23_fscore": 2.25, "taf1b_vsnpm3_fscore": 2.25,
            "samm50_chromalive_fscore": 1.55, "micos13_chromalive_fscore": 2.0}   # centroid-recovery f (α units) per perturbation
FSCORE_PHI_ALPHA = {                                                                         # φ -> resolved α per group (post-snap); None = not measured (leave blank, never substitute another φ's data)
    "mtor_mo_hm_fscore":    {0: 0, 0.5: 0.5, 1: 1.5, 1.5: 2.0, 2: 3.0, 3: None},   # φ3 (α=4.0) dropped from display by request
    "polr1b_vsnpm3_fscore": {0: 0, 0.5: 1.0, 1: 2.5, 1.5: 4.0, 2: 5.0, 3: None},   # φ3 (true α=7.35) exceeds the generated range
    "tim23_fscore":         {0: 0, 0.5: 1.0, 1: 2.5, 1.5: 3.0, 2: 5.0, 3: None},    # φ3 (true α=6.75) exceeds the generated range
    "taf1b_vsnpm3_fscore":  {0: 0, 0.5: 1.0, 1: 2.5, 1.5: 3.0, 2: 5.0, 3: None},      # f=2.25 (corrected from 1.25) -- identical mapping to tim23_fscore (same f); φ1 (true α=2.25) ties 2.0/2.5 -- snapped to 2.5 (larger); φ2 (true α=4.5) ties 4.0/5.0 -- snapped to 5.0 (larger); φ3 (true α=6.75) exceeds the generated range
    "samm50_chromalive_fscore":  {0: 0, 0.5: 1.0, 1: 1.5, 1.5: 2.5, 2: 3.0, 3: None},   # f=1.55: true alphas 0,0.775,1.55,2.325,3.1; φ3 (α=5.0, has data) dropped from display by request, matching every other group's page-wide convention
    "micos13_chromalive_fscore": {0: 0, 0.5: 1.0, 1: 2.0, 1.5: 3.0, 2: 4.0, 3: None},  # f=2.0: exact grid hits at phi<=2 (0,1,2,3,4); phi3 (true α=6) exceeds the generated range
}


def _rank(path, gene, n):
    d = pd.read_parquet(f"{VA}/_rankings/{path}.parquet")
    rt = d["rank_type"].astype(str) == "top" if "rank_type" in d.columns else np.ones(len(d), bool)
    d = d[(d["gene"].astype(str) == gene) & rt].sort_values("rank")
    return d.head(n).reset_index(drop=True)


def _open(exp, well, lab):
    return zarr.open(f"{BASE}/{exp}/3-assembly/phenotyping_v3.zarr/{well[0]}/{well[1:]}/0/labels/{lab}/0", mode="r")


_CHAN = {}


def _marker_crop(exp, well, ch, x, y):
    """Native-160 crop of the marker-channel intensity from phenotyping_v3 (for panel display)."""
    key = (exp, well)
    if key not in _CHAN:
        from iohub import open_ome_zarr
        p = f"{BASE}/{exp}/3-assembly/phenotyping_v3.zarr/{well[0]}/{well[1:]}/0"
        pos = open_ome_zarr(p, mode="r")
        names = list(pos.channel_names)
        idx = next((i for i, n in enumerate(names) if ch.lower() in n.lower() or n.lower() in ch.lower()), 0)
        _CHAN[key] = (zarr.open(f"{p}/0", mode="r"), idx)
    arr, ci = _CHAN[key]
    return _crop(arr, ci, x, y, HALF).astype(np.float32)


def _cell_nuc(exp, well, x, y, sid, dilate):
    """Native-160 real cell mask (dilated) + nucleus mask (nuclear_seg ∩ cell) for the same window."""
    cell = _crop(_open(exp, well, "cell_seg"), None, x, y, HALF) == sid
    if not cell.any():
        return None, None
    nuc = cell & (_crop(_open(exp, well, "nuclear_seg"), None, x, y, HALF) > 0)
    cd = binary_dilation(cell, iterations=dilate) if dilate else cell
    return cd, (nuc if nuc.any() else None)


def _size_filter(org, min_obj_px):
    """Drop connected components < min_obj_px px. Factored out of _measure() so panel/debug-image
    construction can apply the IDENTICAL filter -- otherwise the visual overlay (built from the raw,
    pre-filter labels) never reflects the fix, even though the measured count/area do."""
    if not min_obj_px:
        return org
    from skimage.measure import label as relabel
    lbl = relabel(org > 0)
    if lbl.max():
        sizes = np.bincount(lbl.ravel())
        keep = np.zeros(len(sizes), bool)
        keep[1:] = sizes[1:] >= min_obj_px
        return np.where(keep[lbl], lbl, 0).astype(np.int32)
    return lbl.astype(np.int32)


def _measure(lab_crop, mask, nuc, feats, marker_channel, min_obj_px=0):
    """Same feature extractor for real & gen: process_single_cell at spacing (1,1) → pixel units.
    Returns a {feat: value} dict for the requested feats (count/area/connectivity/degree/branches/location).
    min_obj_px: drop connected components smaller than this (px) before measuring -- _segment_blob_log has
    NO min-size filter of its own (always floors each blob to >=2px radius regardless of min_object_size),
    so tile-stitching leaves tiny (down to ~2px) disk remnants that blow up count/area on noisy gen frames."""
    from organelle_profiler.feature_extraction.fe_workers import process_single_cell
    org = _clip_border(np.where(mask, lab_crop, 0)).astype(np.int32)
    org = _size_filter(org, min_obj_px)
    out = {f: (0.0 if (f in ("count", "area") or f in NET) else np.nan) for f in feats}   # mean-type (shape/location) → NaN when empty
    if org.max() == 0:
        return out
    network = any(k in feats for k in NET)
    chans = ["Phase2D"] if marker_channel == "Phase2D" else ["Phase2D", marker_channel]
    inten = np.zeros((len(chans), *org.shape), np.float32)                       # intensity features unused here
    cf, of, _ = process_single_cell(
        cell_info={"global_cell_id": "x", "well": "A/0/0"}, cell_specific_mask=mask.astype(np.uint8),
        organelle_mask_arrays={"org": org}, intensity_image=inten, frangi_image_arrays={},
        organelles_to_process=["org"], network_organelles=(["org"] if network else []),
        spacing=(1.0, 1.0), channel_names=chans, organelle_map={"org": marker_channel}, full_features=True)
    odf = of.get("org") if of else None
    if odf is not None and len(odf):
        num = odf.select_dtypes(include="number")
        out["count"] = int(num["area_filled"].count()) if "area_filled" in num else len(odf)
        if "area" in feats:
            out["area"] = float(num["area_filled"].sum()) if "area_filled" in num else 0.0
        for sf in ("eccentricity", "aspect_ratio"):                              # per-object shape means (nucleolar shape)
            col = sf if sf in num else (f"{sf}_approx" if f"{sf}_approx" in num else None)
            if sf in feats and col is not None:
                out[sf] = float(num[col].mean())
        if "circularity" in feats:
            # organelle_profiler's circularity_approx uses a Ramanujan ellipse-perimeter approximation
            # (cheap, but unbounded — blows up past 1.0 on thin/degenerate objects). Same metric as
            # cct_nucleoli_roundness.py::measure_roundness: real perimeter_crofton, clipped to [0,1].
            from skimage.measure import regionprops_table
            rp = regionprops_table(org, properties=["area", "perimeter_crofton"])
            if len(rp["area"]):
                p = np.asarray(rp["perimeter_crofton"], float); a = np.asarray(rp["area"], float)
                circ = np.clip(np.where(p > 0, 4 * np.pi * a / (p ** 2), np.nan), 0, 1)
                out["circularity"] = float(np.nanmean(circ))
    for short in NET:
        if short in feats:
            k = next((kk for kk in cf if kk.endswith(NET[short])), None)
            out[short] = float(cf[k]) if k is not None else 0.0
    if "location" in feats and nuc is not None and org.max() > 0:                 # dispersion: mean normalized radial position
        from organelle_profiler.feature_extraction.localization_features import compute_localization_features
        loc = compute_localization_features(org, mask, nuc, spacing=(1.0, 1.0))
        rp = loc["normalized_radial_position"].to_numpy(float)
        out["location"] = float(np.nanmean(rp)) if np.isfinite(rp).any() else np.nan
    return out


def gen_masks(g, ncell, dilate, corr_min=0.5):
    """Per generated cell → dilated real cell mask, recovered by matching its lossless anchor_img (GFP/phase/mCherry
    corr) to the NTC ranking. Returns {c: (mask, rec_row)}."""
    _zarr_patch()
    mt = MORPHO_TARGETS[g["mt"]]
    anc = np.load(f"{VA}/{g['modality']}/_anchors/NTC/ctrl.npz")["anchor_imgs"][:ncell, 0]   # (ncell,160,160)
    raw, recs = _materialize(_rank(g["ntc_rank"], "NTC", 800), mt["marker_channel"], g["ch"], "NTC")   # 800 candidates enough to match the top-~100 anchors (was 2500 → 3× less I/O)
    crops = raw[:, 0]
    zc = lambda a: (a - a.mean()) / (a.std() + 1e-6)
    r160 = lambda a: a if a.shape == anc[0].shape else resize(a, anc[0].shape, preserve_range=True)
    G = np.stack([zc(r160(c)).ravel() for c in crops])
    out = {}
    for c in range(ncell):
        cc = G @ zc(anc[c]).ravel() / G.shape[1]
        b = int(cc.argmax())
        if cc[b] < corr_min:
            continue
        r = recs.iloc[b]
        cell, nuc = _cell_nuc(r["experiment"], str(r["well"]), int(round(r["x_pheno"])), int(round(r["y_pheno"])),
                              int(r["segmentation"]), dilate)
        if cell is not None:
            out[c] = (cell, nuc, r)
    print(f"  [gen_masks] {len(out)}/{ncell} recovered (dilate={dilate}, corr≥{corr_min})", flush=True)
    return out


def _gen_labels(zpath, na):
    root = zarr.open(zpath, mode="r")
    lname = os.listdir(f"{zpath}/A/0/0/labels")
    lname = [x for x in lname if x != "zarr.json"][0]
    return root, lname


def _build_hist_ref(g, n=25):
    """Pooled real-NTC marker crops (percentile-normalized [0,1]) → histogram-match reference for gen frames."""
    df = _rank(g["ntc_rank"], "NTC", n)
    refs = []
    for _, r in df.iterrows():
        try:
            ci = _marker_crop(r["experiment"], str(r["well"]), g["ch"], int(round(r["x_pheno"])), int(round(r["y_pheno"])))
        except Exception:
            continue
        lo, hi = np.percentile(ci, [1, 99.5])
        refs.append(np.clip((ci - lo) / max(hi - lo, 1e-6), 0, 1).astype(np.float32))
    return np.concatenate(refs, axis=0) if refs else None                        # 2D (n*H, W) pooled grayscale reference (match_histograms needs matching ndim)


def gen_measure(gname, g, ncell, dilate):
    from concurrent.futures import ThreadPoolExecutor
    mt = {**MORPHO_TARGETS[g["mt"]], **g.get("seg_override", {})}                   # per-group seg override (e.g. mTOR blob → MO)
    mask_by_cell = g.get("mask_by_cell", True)
    drop_empty = g.get("drop_empty", False)                                        # skip cells with no seg (failed nucleus) so false-0s don't dilute
    masks = gen_masks(g, ncell, dilate) if mask_by_cell else {}
    hm_mode = g.get("hm_mode", "global")
    hist_ref = _build_hist_ref(g, n=g.get("hm_ref_n", 25)) if (g.get("hist_match") and hm_mode != "clahe") else None
    base_dir = f"{SYN}/{gname}"
    cell_offset = g.get("gen_cell_offset", 0)                                       # e.g. 200 → multirank top-100 anchors, rank-aligned w/ real _rank()
    alpha_range = g.get("alpha_range", DEFAULT_ALPHAS)
    aidxs = [_aidx(a) for a in alpha_range]
    vs = None
    if mt.get("mo_vs_nucleus"):
        vs = f"{OUT}/{gname}/vs_nucleus.npz"; os.makedirs(os.path.dirname(vs), exist_ok=True)
        vs_src = g.get("vs_nuc_from", mt["marker_dir"])                             # H2B nucleus from this modality (phase for VS-NPM3, whose frames are phase-derived)
        _vs_h2b_nucleus_npz(vs_src, mt["target"], mt["grain"], vs, ncell, crop=GEN_CROP, float_frames=True,
                            alpha_idxs=aidxs, force=False, cell_offset=cell_offset)   # subset aligned to the seg staging positions 0..len(alpha_range)-1;
                                                                                       # force=False reuses the cached vs_nucleus.npz (GPU DDIM+Cellpose, expensive) when re-running the SAME group/alpha_range/ncell -- only the downstream MO seg logic changed
    full_features(mt["marker_dir"], mt["target"], mt["real_exp"], mt["marker_channel"], grain=mt["grain"],
                  n_cells=ncell, adaptive=True, seg_method=mt.get("seg_method"), structure_type=mt.get("structure_type"),
                  org_label=mt["org_label"], mo_nucleus=mt.get("mo_nucleus", False), vs_nucleus_npz=vs,
                  vs_erode=mt.get("mo_vs_erode", 0), frangi_override=mt.get("frangi_override"), crop=GEN_CROP, float_frames=True,
                  cell_masks=({c: cell for c, (cell, nuc, _) in masks.items()} if mask_by_cell else None),
                  mo_params=g.get("gen_mo_params", g.get("mo_params")), hist_ref=hist_ref, hm_mode=hm_mode, alpha_idxs=aidxs,   # seg ONLY the measured α; gen_mo_params (if set) overrides mo_params for GEN only -- real_measure()/_real_measure_vs() keep reading mo_params, unaffected
                  clahe_params=g.get("clahe_params"),
                  cell_offset=cell_offset, base_dir=base_dir, out_root=f"{OUT}/{gname}")
    zpath = f"{base_dir}/{mt['real_exp']}/3-assembly/phenotyping_v3.zarr"
    root, lname = _gen_labels(zpath, len(_ALPHAS))
    feats = _feats(g)
    res = {f: {} for f in feats}
    panel = {}
    ONES = np.ones((GEN_CROP, GEN_CROP), bool)                                     # no cell mask → measure the (nucleus-bounded) seg directly
    items = ([(c, cell, nuc) for c, (cell, nuc, _) in masks.items()] if mask_by_cell
             else [(c, ONES, None) for c in range(ncell)])
    pcells = list(range(min(N_PANEL, ncell)))                                     # FIXED rank positions 0..N-1 (local index == rank-1) — same 6 cells
                                                                                    # across every α AND matched to real's rank-position panel (not "first N survivors", which drifts per-α and per-side)
    nthr = int(os.environ.get("MNAT_THREADS", "8"))
    for pos, a in enumerate(alpha_range):                                          # subset staged at positions 0..len(alpha_range)-1
        lab = np.asarray(root[f"A/{pos}/0/labels/{lname}/0"][0, 0, 0]).astype(np.int32)
        img = np.asarray(root[f"A/{pos}/0/0"][0, -1, 0]).astype(np.float32)

        def _work(item):
            c, cell, nuc = item
            lc = lab[:, c * (GEN_CROP + PAD):c * (GEN_CROP + PAD) + GEN_CROP]
            return c, cell, lc, _measure(lc, cell, nuc, feats, mt["marker_channel"], min_obj_px=g.get("min_obj_px", 0))

        with ThreadPoolExecutor(max_workers=nthr) as ex:
            out = list(ex.map(_work, items))
        kept = [(c, cell, lc, vals) for (c, cell, lc, vals) in out if not (drop_empty and vals["count"] == 0)]
        acc = {f: [vals[f] for _, _, _, vals in kept] for f in feats}
        for c, cell, lc, vals in out:                                             # panel from ALL cells (not just kept) — rank position must show even if empty
            if c in pcells:
                x0 = c * (GEN_CROP + PAD)
                pimg = img[:, x0:x0 + GEN_CROP]
                if g.get("panel_src"):                                            # show the SOURCE image (e.g. phase) under the seg, not the seg-input (VS-NPM3)
                    zr = _aidx(a)
                    pimg = np.clip((np.load(f"{VA}/{g['panel_src']}/{mt['grain']}/{mt['target']}/cell{c + cell_offset}/frames_f32.npz")["gen"][zr] + 1) / 2, 0, 1)
                lc_show = _size_filter(_clip_border(np.where(cell, lc, 0)).astype(np.int32), g.get("min_obj_px", 0))   # same preprocessing _measure() applies, so the overlay matches what's actually measured
                panel.setdefault(pcells.index(c), {})[f"gen_a{a}"] = (np.asarray(pimg).copy(), lc_show.copy(), cell.copy())
        for f in feats:
            res[f][a] = np.array(acc[f], float)
    return res, panel


def _reseg(crop, mt, nuc, mo_params=None):
    """Seg a (blurred) real crop with the group's gen seg method → for resolution-matched real measurement."""
    from scipy import ndimage as ndi
    lo, hi = np.percentile(crop, [1, 99.5])
    cn = np.clip((crop - lo) / max(hi - lo, 1e-6), 0, 1).astype(np.float32)
    if mt.get("seg_method") == "masked_object":
        return _seg_masked_object(cn, tp=(mo_params or MO_PARAMS), nucleus=mt.get("mo_nucleus", False),
                                  nucleus_override=(nuc if (mt.get("mo_nucleus") and nuc is not None) else None),
                                  override_erode=mt.get("mo_vs_erode", 0))
    from skimage.filters import frangi                                            # tubular/blob → frangi ridge + threshold
    f = frangi(cn, sigmas=np.arange(1, 4), black_ridges=False)
    return ndi.label(f > np.percentile(f, 92))[0].astype(np.int32)


_VS_MODEL = {}


def _vs_npm3_model():
    """Lazy-load the multi-marker spatial-cond DiffAE used to VS-predict NPM3 from phase (shared w/ build_vs_npm3)."""
    if "m" not in _VS_MODEL:
        import torch
        from ops_model.models.attention.diffex.diffae.config import DiffAEConfig
        from ops_model.models.attention.diffex.diffae.model import DiffAE
        VOUT = "/hpc/projects/icd.fast.ops/analysis/virtual_staining/multi_marker"
        markers = json.load(open(f"{VOUT}/markers.json")); mi = markers.index("nucleolus-GC_NPM3")
        dev = torch.device("cuda")
        cfg = DiffAEConfig(spatial_cond=True, n_markers=len(markers), device="cuda")
        model = DiffAE(cfg).to(dev).eval(); model.load_state_dict(torch.load(f"{VOUT}/diffae_best.pt", map_location=dev))
        _VS_MODEL["m"] = (model, cfg, mi, dev)
    return _VS_MODEL["m"]


def _vs_npm3_apply(phase01_stack, batch=48):
    """phase01_stack: (N,H,H) in [0,1] → VS-NPM3 (N,H,H) in [0,1]. Batched DDIM sample (GPU) — the real-side
    counterpart of build_vs_npm3._vs_job, so 'real' gets the SAME VS measurement tool as 'gen' (no fluor shortcut)."""
    import torch
    from diffusers import DDIMScheduler
    from ops_model.models.attention.diffex.classifier.celldino_features import embed_crops
    model, cfg, mi, dev = _vs_npm3_model()
    out = np.empty_like(phase01_stack, dtype=np.float32)
    for i0 in range(0, len(phase01_stack), batch):
        P = (phase01_stack[i0:i0 + batch, None] * 2 - 1).astype(np.float32)
        emb = embed_crops(P, cfg)
        with torch.no_grad():
            fwd = DDIMScheduler(num_train_timesteps=cfg.train_timesteps); fwd.set_timesteps(cfg.ddim_steps)
            ci = torch.as_tensor(P, device=dev); e = torch.as_tensor(emb, device=dev)
            mk = torch.full((P.shape[0],), mi, dtype=torch.long, device=dev)
            c = model.cond(e, mk); x = torch.randn(P.shape[0], 1, *P.shape[-2:], device=dev)
            for t in fwd.timesteps:
                x = fwd.step(model.denoise(x, t, c, ci), t, x).prev_sample
        out[i0:i0 + batch] = np.clip((x.cpu().numpy()[:, 0] + 1) / 2, 0, 1)
    return out


def _real_measure_vs(g, mt, rows, dilate, feats, mask_by_cell, drop_empty, want_panel):
    """Real side of a vs_real group: gather real PHASE crops + nucleus masks (threaded I/O), batch-VS them to
    NPM3 (GPU), then seg+measure (threaded) — the exact same VS tool the gen side uses, on real phase cells."""
    from concurrent.futures import ThreadPoolExecutor
    ONES = np.ones((2 * HALF, 2 * HALF), bool)

    def _gather(rw):
        exp, well, x, y, sid = rw
        cell, nuc = _cell_nuc(exp, well, x, y, sid, dilate)
        if cell is None and mask_by_cell:
            return None
        try:
            ci = _marker_crop(exp, well, "Phase2D", x, y)
        except Exception:
            return None
        lo, hi = np.percentile(ci, [1, 99.5])
        crop01 = np.clip((ci - lo) / max(hi - lo, 1e-6), 0, 1).astype(np.float32)
        return crop01, (cell if mask_by_cell else ONES), nuc

    nthr = int(os.environ.get("MNAT_THREADS", "8"))
    with ThreadPoolExecutor(max_workers=nthr) as ex:
        gathered = list(ex.map(_gather, rows))
    keep = [x for x in gathered if x is not None]
    if not keep:
        return {f: np.array([], float) for f in feats}, []
    crops = np.stack([k[0] for k in keep])
    vs_npm3 = _vs_npm3_apply(crops)                                              # batched GPU VS — same tool as gen

    tp = g.get("mo_params") or MO_PARAMS

    def _seg_measure(i):
        V = vs_npm3[i]
        lo, hi = np.percentile(V, [1, 99.5]); Vs = np.clip((V - lo) / max(hi - lo, 1e-6), 0, 1).astype(np.float32)
        _, mask, nuc = keep[i]
        lab = _seg_masked_object(Vs, tp=tp, nucleus=True, nucleus_override=(nuc if nuc is not None else None), override_erode=g.get("vs_erode", 4))
        vals = _measure(lab, mask, nuc, feats, mt["marker_channel"], min_obj_px=g.get("min_obj_px", 0))
        pit = (crops[i].copy(), lab.copy(), np.asarray(mask).copy()) if want_panel else None
        return vals, pit

    with ThreadPoolExecutor(max_workers=nthr) as ex:
        results = list(ex.map(_seg_measure, range(len(keep))))
    acc = {f: [] for f in feats}; panel = []                                       # panel picks by RANK POSITION (same reasoning as real_measure/gen_measure)
    for vals, pit in results:
        if want_panel and pit is not None and len(panel) < N_PANEL:
            panel.append(pit)
        if drop_empty and vals["count"] == 0:
            continue
        for f in feats:
            acc[f].append(vals[f])
    return {f: np.array(acc[f], float) for f in feats}, panel


def real_measure(g, gene, rank_path, n, dilate, want_panel=False):
    from concurrent.futures import ThreadPoolExecutor
    mt = {**MORPHO_TARGETS[g["mt"]], **g.get("seg_override", {})}
    mask_by_cell = g.get("mask_by_cell", True)
    drop_empty = g.get("drop_empty", False)
    ONES = np.ones((2 * HALF, 2 * HALF), bool)
    _zarr_patch()
    df = _rank(rank_path, gene, n)
    feats = _feats(g)
    rows = [(r["experiment"], str(r["well"]), int(round(r["x_pheno"])), int(round(r["y_pheno"])), int(r["segmentation"]))
            for _, r in df.iterrows()]

    if g.get("vs_real"):                                                         # VS-NPM3 IS the measurement tool → apply to real PHASE cells too (symmetric w/ gen)
        return _real_measure_vs(g, mt, rows, dilate, feats, mask_by_cell, drop_empty, want_panel)

    def _work(rw):
        exp, well, x, y, sid = rw
        cell, nuc = _cell_nuc(exp, well, x, y, sid, dilate)
        if cell is None and mask_by_cell:
            return None
        mask = cell if mask_by_cell else ONES                                     # no cell mask → measure the (nucleus-bounded) label directly
        if g.get("blur_real"):                                                   # RESOLUTION MATCH: blur real → gen's smoothness, then re-seg (group's method)
            from scipy.ndimage import gaussian_filter
            ci = gaussian_filter(_marker_crop(exp, well, g["ch"], x, y).astype(np.float32), g["blur_real"])
            lc = _reseg(ci, mt, nuc, g.get("mo_params"))
        elif g.get("real_mo"):                                                   # MO-seg the real image too (symmetric w/ gen MO)
            ci = _marker_crop(exp, well, g["ch"], x, y)
            lo, hi = np.percentile(ci, [1, 99.5])
            lc = _seg_masked_object(np.clip((ci - lo) / max(hi - lo, 1e-6), 0, 1).astype(np.float32),
                                    tp=(g.get("mo_params") or MO_PARAMS), nucleus=mt.get("mo_nucleus", False))
        else:
            lc = _crop(_open(exp, well, g["real_label"]), None, x, y, HALF)
        vals = _measure(lc, mask, nuc, feats, mt["marker_channel"], min_obj_px=g.get("min_obj_px", 0))
        pit = None
        if want_panel:
            try:
                int_c = _marker_crop(exp, well, g["ch"], x, y)
            except Exception:
                int_c = (np.asarray(lc) > 0).astype(np.float32)
            pit = (int_c.copy(), np.asarray(lc).copy(), np.asarray(mask).copy())
        return vals, pit

    nthr = int(os.environ.get("MNAT_THREADS", "8"))
    with ThreadPoolExecutor(max_workers=nthr) as ex:
        results = list(ex.map(_work, rows))
    acc = {f: [] for f in feats}; panel = []
    for res in results:                                                           # rank order preserved (ex.map) — panel picks by RANK POSITION, not
        if res is None:                                                           # "first N stats-survivors" (that drifts independently per side/per-α, see gen_measure)
            continue
        vals, pit = res
        if want_panel and pit is not None and len(panel) < N_PANEL:
            panel.append(pit)
        if drop_empty and vals["count"] == 0:                                     # skip failed-detection cells (dilute the median) — stats only, not panel
            continue
        for f in feats:
            acc[f].append(vals[f])
    return {f: np.array(acc[f], float) for f in feats}, panel


SHAPE = ["circularity", "eccentricity", "aspect_ratio"]


def _feats(g):
    return (["count", "area"] + (list(NET) if g["network"] else [])
            + (SHAPE if g.get("shape") else []) + (["location"] if g.get("location") else []))


def _crit(feat, rn_a, rk_a, gen):
    md = lambda a: float(np.nanmedian(a)) if len(a) else float("nan")
    rn, rk = md(rn_a), md(rk_a)
    g0, g1, g2, g3 = md(gen[0]), md(gen[1]), md(gen[2]), md(gen[3])
    ko = (rk - rn) / rn * 100 if rn else float("nan")
    a0 = (g0 - rn) / rn * 100 if rn else float("nan")
    a1 = (g1 - rk) / rk * 100 if rk else float("nan")
    print(f"\n== {feat} (medians) ==")
    print(f"  real NTC {rn:.2f} | real KO {rk:.2f} | gen α0 {g0:.2f} | α1 {g1:.2f} | α2 {g2:.2f} | α3 {g3:.2f}")
    print(f"  (1) gen α0 vs real NTC: {a0:+.1f}%  {'OK' if abs(a0) <= 5 else 'FAIL'}")
    print(f"  (2) real KO vs real NTC: {ko:+.1f}%  {'OK' if ko >= 25 else 'FAIL'}")
    print(f"  (3) gen α1 vs real KO: {a1:+.1f}%  {'OK' if abs(a1) <= 10 else 'FAIL'}")
    print(f"  (4) α3 > α1: {g3:.2f} > {g1:.2f}  {'OK' if g3 > g1 else 'FAIL'}")


_ALPHAS = list(range(17))


def run(gname, ncell=100, dilate=8, tag=""):
    g = GROUPS[gname]
    odir = f"{gname}{tag}"                                                          # output subdir (tagged so runs don't clobber each other)
    os.makedirs(f"{OUT}/{odir}", exist_ok=True)
    print(f"[native] {gname} -> {odir}: ncell={ncell} dilate={dilate}", flush=True)
    gen, gpanel = gen_measure(odir, g, ncell, dilate)
    # real_n must match gen's exact aligned population when gen is a specific N-cell set (gen_cell_offset, e.g.
    # multirank top-100 anchors) — pulling a bigger pool (e.g. 1000) then slicing [:100] is WRONG: drop_empty
    # compacts/reorders before the slice, so "[:100]" isn't the true top-100 rank population gen reconstructs.
    real_n = g.get("vs_real_n", ncell) if g.get("vs_real") else (ncell if g.get("gen_cell_offset") else 1000)
    rn, rn_panel = real_measure(g, "NTC", g["ntc_rank"], real_n, dilate, want_panel=True)
    rk, rk_panel = real_measure(g, g["ko_gene"], g["ko_rank"], real_n, dilate, want_panel=True)
    labels = {"count": "count", "area": "area (px)", "connectivity": "connectivity (LCC)",
              "degree": "network degree", "branches": "branch count", "nodes": "node count", "location": "radial position",
              "circularity": "circularity", "eccentricity": "eccentricity", "aspect_ratio": "aspect ratio"}
    feats = _feats(g)
    stats = {}
    for feat in feats:
        tag100 = f" [real top-{real_n}]" if real_n <= 100 else " [real top-1000]"
        _crit(feat + tag100, rn[feat], rk[feat], gen[feat])
        if real_n > 100:                                                          # only meaningful as a SEPARATE cut when the pool is bigger than 100
            _crit(feat + " [real top-100]", rn[feat][:100], rk[feat][:100], gen[feat])
        stats[f"rn_{feat}"] = rn[feat]; stats[f"rk_{feat}"] = rk[feat]
        stats[f"rn100_{feat}"] = rn[feat][:100]; stats[f"rk100_{feat}"] = rk[feat][:100]
        for a in g.get("alpha_range", DEFAULT_ALPHAS):
            stats[f"gen_{feat}_a{a}"] = gen[feat][a]
        _violin(odir, feat, labels[feat], rn[feat], rk[feat], gen[feat])
    np.savez_compressed(f"{OUT}/{odir}/stats.npz", **stats)
    np.savez_compressed(f"{OUT}/{odir}/panel.npz",
                        gpanel=np.array(gpanel, dtype=object), rn=np.array(rn_panel, dtype=object),
                        rk=np.array(rk_panel, dtype=object))
    _render_panel(odir)                                                            # render the debug panel IN-job (parallel across groups, not serial in the monitor)
    print(f"[native] {odir} saved panel.npz + stats.npz + violins + panel.png", flush=True)


def run_gen_only(gname, ncell=100, dilate=8, tag=""):
    """Lean variant of run() -- skips real_measure() entirely (real doesn't depend on alpha/gen-side seg
    tweaks, already in the base stats.npz) and skips violin/debug-panel rendering. tag MUST match whatever
    the base full run used (run()'s odir = f"{gname}{tag}") so this reuses the SAME vs_nucleus.npz cache
    (force=False in gen_measure) instead of paying for GPU DDIM+Cellpose nucleus prediction again.
    Saves just {gen_<feat>_a<alpha>: array} + gpanel to new_alpha.npz/new_alpha_panel.npz, for
    fscore_violins.py (or a manual merge) to combine with the base group's existing stats.npz/panel.npz."""
    g = GROUPS[gname]
    odir = f"{gname}{tag}"
    os.makedirs(f"{OUT}/{odir}", exist_ok=True)
    print(f"[native] {gname} (gen-only) -> {odir}: ncell={ncell} dilate={dilate}", flush=True)
    gen, gpanel = gen_measure(odir, g, ncell, dilate)
    feats = _feats(g)
    stats = {}
    for feat in feats:
        for a in g.get("alpha_range", DEFAULT_ALPHAS):
            stats[f"gen_{feat}_a{a}"] = gen[feat][a]
    np.savez_compressed(f"{OUT}/{odir}/new_alpha.npz", **stats)
    np.savez_compressed(f"{OUT}/{odir}/new_alpha_panel.npz", gpanel=np.array(gpanel, dtype=object))
    print(f"[native] {odir} saved new_alpha.npz + new_alpha_panel.npz", flush=True)


def run_real_only(gname, ncell=100, dilate=8, tag=""):
    """Lean variant of run() -- skips gen_measure() entirely, only redoes real_measure() (e.g. when the
    REAL-side mo_params changed). Overwrites rn_*/rk_*/rn100_*/rk100_* in stats.npz and the rn/rk entries
    in panel.npz IN PLACE, preserving every gen_* key and gpanel entry already there."""
    g = GROUPS[gname]
    odir = f"{gname}{tag}"
    stats_path = f"{OUT}/{odir}/stats.npz"
    panel_path = f"{OUT}/{odir}/panel.npz"
    stats = dict(np.load(stats_path))
    panel = dict(np.load(panel_path, allow_pickle=True))
    real_n = g.get("vs_real_n", ncell) if g.get("vs_real") else (ncell if g.get("gen_cell_offset") else 1000)
    rn, rn_panel = real_measure(g, "NTC", g["ntc_rank"], real_n, dilate, want_panel=True)
    rk, rk_panel = real_measure(g, g["ko_gene"], g["ko_rank"], real_n, dilate, want_panel=True)
    feats = _feats(g)
    for feat in feats:
        stats[f"rn_{feat}"] = rn[feat]; stats[f"rk_{feat}"] = rk[feat]
        stats[f"rn100_{feat}"] = rn[feat][:100]; stats[f"rk100_{feat}"] = rk[feat][:100]
    np.savez_compressed(stats_path, **stats)
    np.savez_compressed(panel_path, gpanel=panel["gpanel"], rn=np.array(rn_panel, dtype=object), rk=np.array(rk_panel, dtype=object))
    print(f"[native] {odir} real-only: updated rn/rk in stats.npz + panel.npz", flush=True)


def _job_gen_only(gname, ncell, dilate, threads, tag=""):
    os.environ["OPS_DIFFEX_ASSETS"] = "viewer_assets_v5"
    os.environ["SLURM_CPUS_PER_TASK"] = "1"
    os.environ["MNAT_THREADS"] = str(threads)
    run_gen_only(gname, ncell, dilate, tag=tag)


def submit_gen_only(groups, ncell=100, dilate=8, cpus=16, tag=""):
    import pathlib
    from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs
    os.environ["PYTHONPATH"] = str(pathlib.Path(__file__).resolve().parent) + os.pathsep + os.environ.get("PYTHONPATH", "")
    os.environ.setdefault("OPS_DIFFEX_ASSETS", "viewer_assets_v5")
    jobs = [{"name": f"mnatgen_{gp}", "func": _job_gen_only, "kwargs": {"gname": gp, "ncell": ncell, "dilate": dilate, "threads": cpus, "tag": tag}} for gp in groups]
    sp = {"slurm_partition": "gpu", "gpus_per_node": 1, "cpus_per_task": cpus, "mem_gb": 64, "timeout_min": 180,
          "slurm_setup": ["export OPS_DIFFEX_ASSETS=viewer_assets_v5"]}
    submit_parallel_jobs(jobs, experiment="diffex_mnat", slurm_params=sp, log_dir="diffex_mnat", wait_for_completion=False)


def _violin(gname, feat, ylab, rn_a, rk_a, gen):
    """Raw-units violin: real NTC, real KO, gen α0/α1/α3 — same window/seg/mask so bars are directly comparable."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["pdf.fonttype"] = 42
    C = {"real": "#999999", "KO": "#2e8b57", "α=0": "#c6dbef", "α=0.5": "#9ecae1", "α=1": "#6baed6",
        "α=1.5": "#4292c6", "α=2": "#3182bd", "α=2.5": "#1c5c94", "α=3": "#08519c"}
    data = [np.asarray(rn_a, float), np.asarray(rk_a, float), gen[0], gen[0.5], gen[1], gen[1.5], gen[2], gen[2.5], gen[3]]
    data = [d[np.isfinite(d)] for d in data]                                      # drop NaN (location) for the KDE/median
    labs = ["real", "KO", "α=0", "α=0.5", "α=1", "α=1.5", "α=2", "α=2.5", "α=3"]
    keep = [i for i, d in enumerate(data) if len(d)]
    fig, ax = plt.subplots(figsize=(5.4, 5.2), facecolor="white")
    parts = ax.violinplot([data[i] for i in keep], positions=keep, showmeans=False, showextrema=False, showmedians=False, widths=0.82)
    for pc, i in zip(parts["bodies"], keep):
        pc.set_facecolor(C[labs[i]]); pc.set_alpha(0.6); pc.set_edgecolor(C[labs[i]]); pc.set_linewidth(1.5)
    for i in keep:
        ax.hlines(np.median(data[i]), i - 0.34, i + 0.34, color="#222", lw=3, zorder=5)
    ax.set_xticks(range(len(labs))); ax.set_xticklabels(labs, fontsize=24)
    ax.set_ylabel(f"{ylab} (raw)", fontsize=20)
    ax.tick_params(axis="y", labelsize=18, width=2.5, length=8); ax.tick_params(axis="x", length=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_linewidth(2.5)
    for ext in ("png", "svg"):
        fig.savefig(f"{OUT}/{gname}/violin_{feat}_raw.{ext}", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _force_cpu_morphology():
    """Poison the GPU morphology module BEFORE fe_workers' top-level auto-detect ever runs, forcing the CPU
    feature-extraction path (real perimeter-based circularity_approx=4πA/P², not GPU's circularity=1-eccentricity).
    Pure process-local sys.modules trick — does NOT touch the shared organelle_profiler package/other jobs."""
    import sys
    sys.modules["organelle_profiler.feature_extraction.morphology_features_gpu"] = None


def _job(gname, ncell, dilate, threads, tag, force_cpu_circularity=False):
    if force_cpu_circularity:
        _force_cpu_morphology()
    os.environ["OPS_DIFFEX_ASSETS"] = "viewer_assets_v5"
    os.environ["SLURM_CPUS_PER_TASK"] = "1"              # serial run_seg → avoid nested-pool loky crashes
    os.environ["MNAT_THREADS"] = str(threads)            # thread the per-cell measurement (real ~2000 cells) across the alloc
    run(gname, ncell, dilate, tag=tag)


def submit(groups=("mtor_mo_hm", "polr1b_vsnpm3", "tim23"), ncell=100, dilate=8, cpus=16, tag="", force_cpu_circularity=False):
    import pathlib
    from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs
    os.environ["PYTHONPATH"] = str(pathlib.Path(__file__).resolve().parent) + os.pathsep + os.environ.get("PYTHONPATH", "")
    os.environ.setdefault("OPS_DIFFEX_ASSETS", "viewer_assets_v5")
    jobs = [{"name": f"mnat_{gp}", "func": _job, "kwargs": {"gname": gp, "ncell": ncell, "dilate": dilate, "threads": cpus, "tag": tag, "force_cpu_circularity": force_cpu_circularity}} for gp in groups]
    sp = {"slurm_partition": "gpu", "gpus_per_node": 1, "cpus_per_task": cpus, "mem_gb": 64, "timeout_min": 180,
          "slurm_setup": ["export OPS_DIFFEX_ASSETS=viewer_assets_v5"]}   # gpu: POLR1B vs_nucleus needs the DiffAE/Cellpose
    submit_parallel_jobs(jobs, experiment="diffex_mnat", slurm_params=sp, log_dir="diffex_mnat", wait_for_completion=False)


def _render_panel(gname):
    """One debug figure (gname already includes any tag). Rows [real KO, real NTC, gen α0/α1/α3] × N cells."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from skimage.segmentation import find_boundaries
    plt.rcParams["pdf.fonttype"] = 42
    pf = f"{OUT}/{gname}/panel.npz"
    if not os.path.exists(pf):
        print(f"skip {gname}: no panel.npz"); return
    d = np.load(pf, allow_pickle=True)
    gpanel = d["gpanel"].item()
    rn, rk = list(d["rn"]), list(d["rk"])
    rows = [("real KO", rk), ("real NTC", rn), ("gen α0", 0), ("gen α1", 1), ("gen α3", 3)]   # KO→NTC→α0: NTC & α0 adjacent
    nc = min(N_PANEL, PANEL_GRID_COLS)
    fig, axes = plt.subplots(len(rows), nc, figsize=(nc * 2.0, len(rows) * 2.0), facecolor="white")
    for ri, (label, src) in enumerate(rows):
        for ci in range(nc):
            ax = axes[ri, ci]; ax.axis("off")
            if ri == 0:
                ax.set_title(f"cell {ci}", fontsize=9)
            if ci == 0:
                ax.text(-0.15, 0.5, label, transform=ax.transAxes, rotation=90, va="center", ha="center", fontsize=12, fontweight="bold")
            if isinstance(src, list):                                        # real: (int_crop, lab, mask)
                if ci >= len(src):
                    continue
                img, lc, mask = src[ci]
            else:                                                            # gen: gpanel[cell][f"gen_a{a}"] = (img, lab, mask)
                key = f"gen_a{src}"
                if ci not in gpanel or key not in gpanel[ci]:
                    continue
                img, lc, mask = gpanel[ci][key]
            img = np.asarray(img, np.float32); lc = np.asarray(lc); mask = np.asarray(mask) > 0
            img = (img - np.min(img)) / (np.ptp(img) + 1e-9)
            ax.imshow(img, cmap="gray", vmin=0, vmax=1)
            b = find_boundaries(np.where(mask, lc, 0) > 0, mode="outer")
            ov = np.zeros((*b.shape, 4)); ov[b] = [1, 0.3, 0, 1]
            ax.imshow(ov)
            mb = find_boundaries(mask, mode="inner")
            ov2 = np.zeros((*mb.shape, 4)); ov2[mb] = [0.2, 0.8, 1, 0.7]      # cell-mask outline (cyan)
            ax.imshow(ov2)
    fig.suptitle(f"{gname} — native-160 seg (orange=organelle, cyan=dilated cell mask)", fontsize=13)
    fig.tight_layout(rect=[0.02, 0, 1, 0.97])
    out = f"{OUT}/{gname}/DEBUG_seg_panel.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print(f"saved {out}")


def render_panels(groups=("mtor_mo_hm", "polr1b_vsnpm3", "tim23"), tag=""):
    for gname0 in groups:
        _render_panel(f"{gname0}{tag}")


def vsnpm3_panel(n=6, min_size=25, erode=4):
    """Standalone (no SLURM) standard-layout panel for VS-NPM3: rows [real KO, real NTC, gen α0/α1/α3] × cells.
    real = NPM3 marker crop + stored gfp_seg; gen = PHASE frame + VS-NPM3-derived MO seg (nucleus-masked)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from skimage.segmentation import find_boundaries
    from ops_model.models.attention.diffex.viewer.morpho_pipeline import _seg_masked_object, MO_PARAMS
    _zarr_patch()
    tp = {**MO_PARAMS, "min_object_size": min_size, "mo_object_min_area_px": min_size}
    nuc = np.load(f"{OUT}/polr1b_100/vs_nucleus.npz")["masks"]
    ph = f"{VA}/phase/geneKO/POLR1B"; vs = f"{VA}/vs_npm3_from_phase/geneKO/POLR1B"
    norm = lambda a: np.clip((a + 1) / 2, 0, 1)

    def real_row(gene):                                                          # REAL = phase cells + stored phase-nucleoli seg (all-phase panel)
        df = _rank("pma_shap_phase_geneKO", gene, n * 4)
        out = []
        for _, r in df.iterrows():
            exp, well = r["experiment"], str(r["well"]); x, y = int(round(r["x_pheno"])), int(round(r["y_pheno"]))
            try:
                img = _marker_crop(exp, well, "Phase2D", x, y)
            except Exception:
                continue
            lab = _crop(_open(exp, well, "nucleoli_phase2d_seg"), None, x, y, HALF)
            out.append((img, lab))
            if len(out) >= n:
                break
        return out

    gen = {a: [] for a in (0, 1, 3)}
    for a, ai in ((0, _aidx(0)), (1, _aidx(1)), (3, _aidx(3))):
        for c in range(n):
            V = norm(np.load(f"{vs}/cell{c}/frames_f32.npz")["gen"][ai])
            lo, hi = np.percentile(V, [1, 99.5]); Vs = np.clip((V - lo) / max(hi - lo, 1e-6), 0, 1).astype(np.float32)
            lab = _seg_masked_object(Vs, tp=tp, nucleus=True, nucleus_override=nuc[c, ai] > 0, override_erode=erode)
            gen[a].append((norm(np.load(f"{ph}/cell{c}/frames_f32.npz")["gen"][ai]), lab))
    rows = [("real KO", real_row("POLR1B")), ("real NTC", real_row("NTC")),
            ("gen α0", gen[0]), ("gen α1", gen[1]), ("gen α3", gen[3])]
    fig, axes = plt.subplots(len(rows), n, figsize=(n * 2.0, len(rows) * 2.0), facecolor="white")
    for ri, (label, cells) in enumerate(rows):
        for ci in range(n):
            ax = axes[ri, ci]; ax.axis("off")
            if ri == 0:
                ax.set_title(f"cell {ci}", fontsize=9)
            if ci == 0:
                ax.text(-0.18, 0.5, label, transform=ax.transAxes, rotation=90, va="center", ha="center", fontsize=11, fontweight="bold")
            if ci >= len(cells):
                continue
            img, lab = cells[ci]; im = (img - img.min()) / (np.ptp(img) + 1e-9)
            ax.imshow(im, cmap="gray", vmin=0, vmax=1)
            b = find_boundaries(np.asarray(lab) > 0, mode="outer")
            ov = np.zeros((*b.shape, 4)); ov[b] = [1, 0.3, 0, 1]; ax.imshow(ov)
    fig.suptitle("VS-NPM3 — real PHASE (nucleoli_phase2d_seg) vs gen PHASE w/ VS-NPM3-derived nucleoli seg (orange)", fontsize=12)
    fig.tight_layout(rect=[0.02, 0, 1, 0.97])
    out = f"{OUT}/polr1b_vsnpm3_100/DEBUG_seg_panel.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print(f"saved {out}")


def phase_diag(n=6, min_size=15, erode=4):
    """Debug: POLR1B phase gen frames (α0/α1/α3 × cells) with the phase-nucleoli MO seg (inside the VS-H2B nucleus)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from skimage.segmentation import find_boundaries
    from ops_model.models.attention.diffex.viewer.morpho_pipeline import _seg_masked_object, MO_PARAMS
    tp = {**MO_PARAMS, "min_object_size": min_size, "mo_object_min_area_px": min_size}
    nuc = np.load(f"{OUT}/polr1b_100/vs_nucleus.npz")["masks"]                            # (100,17,160,160) VS-H2B nucleus
    ph = f"{VA}/phase/geneKO/POLR1B"
    norm = lambda a: np.clip((a + 1) / 2, 0, 1)
    alphas = [(0, _aidx(0)), (1, _aidx(1)), (3, _aidx(3))]
    fig, axes = plt.subplots(len(alphas), n, figsize=(n * 2.0, len(alphas) * 2.0), facecolor="white")
    for ri, (a, ai) in enumerate(alphas):
        for ci in range(n):
            ax = axes[ri, ci]; ax.axis("off")
            if ri == 0:
                ax.set_title(f"cell {ci}", fontsize=9)
            if ci == 0:
                ax.text(-0.18, 0.5, f"phase α{a}", transform=ax.transAxes, rotation=90, va="center", ha="center", fontsize=11, fontweight="bold")
            P = norm(np.load(f"{ph}/cell{ci}/frames_f32.npz")["gen"][ai]).astype(np.float32)
            lab = _seg_masked_object(P, tp=tp, nucleus=True, nucleus_override=nuc[ci, ai] > 0, override_erode=erode)
            im = (P - P.min()) / (np.ptp(P) + 1e-9)
            ax.imshow(im, cmap="gray", vmin=0, vmax=1)
            b = find_boundaries(lab > 0, mode="outer")
            ov = np.zeros((*b.shape, 4)); ov[b] = [1, 0.3, 0, 1]; ax.imshow(ov)
    fig.suptitle("POLR1B phase — gen frames w/ phase-nucleoli MO seg (orange, inside VS-nucleus)", fontsize=13)
    fig.tight_layout(rect=[0.02, 0, 1, 0.97])
    out = f"{OUT}/polr1b_100/DEBUG_phase_seg.png"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print(f"saved {out}")


def vsnpm3_diag(n=6, min_size=20, erode=4):
    """Diagnostic for VS-NPM3: per cell × α, show the PHASE input and the VS-NPM3 image side by side, BOTH with
    the SAME seg (MO on VS-NPM3, MASKED to the VS-H2B nucleus + size-excluded) overlaid."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from skimage.segmentation import find_boundaries
    from ops_model.models.attention.diffex.viewer.morpho_pipeline import _seg_masked_object, MO_PARAMS
    tp = {**MO_PARAMS, "min_object_size": min_size, "mo_object_min_area_px": min_size}    # size-exclude noise
    nuc = np.load(f"{OUT}/polr1b_100/vs_nucleus.npz")["masks"]                            # (100,17,160,160) VS-H2B nucleus from phase
    ph = f"{VA}/phase/geneKO/POLR1B"; vs = f"{VA}/vs_npm3_from_phase/geneKO/POLR1B"
    alphas = [(0, _aidx(0)), (1, _aidx(1)), (3, _aidx(3))]
    norm = lambda a: np.clip((a + 1) / 2, 0, 1)
    rows = []
    for a, ai in alphas:
        prow, vrow = [], []
        for c in range(n):
            P = norm(np.load(f"{ph}/cell{c}/frames_f32.npz")["gen"][ai])
            V = norm(np.load(f"{vs}/cell{c}/frames_f32.npz")["gen"][ai])
            lo, hi = np.percentile(V, [1, 99.5]); Vs = np.clip((V - lo) / max(hi - lo, 1e-6), 0, 1).astype(np.float32)
            lab = _seg_masked_object(Vs, tp=tp, nucleus=True, nucleus_override=nuc[c, ai] > 0, override_erode=erode)
            prow.append((P, lab))
        rows.append((f"phase α{a}", prow))                                        # ONLY phase + the VS-NPM3-derived seg (intermediate VS image removed)
    fig, axes = plt.subplots(len(rows), n, figsize=(n * 2.0, len(rows) * 2.0), facecolor="white")
    for ri, (label, cells) in enumerate(rows):
        for ci, (img, lab) in enumerate(cells):
            ax = axes[ri, ci]; ax.axis("off")
            if ri == 0:
                ax.set_title(f"cell {ci}", fontsize=9)
            if ci == 0:
                ax.text(-0.18, 0.5, label, transform=ax.transAxes, rotation=90, va="center", ha="center", fontsize=11, fontweight="bold")
            im = (img - img.min()) / (np.ptp(img) + 1e-9)
            ax.imshow(im, cmap="gray", vmin=0, vmax=1)
            b = find_boundaries(lab > 0, mode="outer")
            ov = np.zeros((*b.shape, 4)); ov[b] = [1, 0.3, 0, 1]; ax.imshow(ov)
    fig.suptitle("VS-NPM3 diagnostic — phase input vs VS-NPM3 image, both w/ the VS-NPM3 MO seg (orange)", fontsize=13)
    fig.tight_layout(rect=[0.02, 0, 1, 0.97])
    out = f"{OUT}/polr1b_vsnpm3_100/DEBUG_phase_vs_npm3.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    if "--submit" in sys.argv:
        submit()
    elif "--panel" in sys.argv:
        render_panels()
    else:
        run(sys.argv[1] if len(sys.argv) > 1 else "mtor_mo_hm")
