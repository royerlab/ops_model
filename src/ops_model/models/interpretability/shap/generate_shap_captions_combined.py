"""Generate captions from two-modality SHAP features (phase + per-channel fluorescence).

Input : reports/ko_shap_features_targeted.csv  (has 'modality' column: 'phase' / 'fluor',
        and 'channel_rank' 0/1/2/3 for phase/ch1/ch2/ch3)
Output: reports/ko_shap_captions_targeted.csv

Caption structure per gene:
  "[GENE] KO cells (phase auroc=X.XX, ch1 auroc=Y.YY, ...):
   Phase — [phase organelle phrases];
   [ch1 channel name] — [fluor ch1 phrases];
   [ch2 channel name] — [fluor ch2 phrases];
   [ch3 channel name] — [fluor ch3 phrases]."

Usage:
    python scripts/generate_shap_captions_combined.py
    python scripts/generate_shap_captions_combined.py \
        --features reports/ko_shap_features_combined.csv \
        --captions reports/ko_shap_captions_combined.csv
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Organelle lookup tables
# ---------------------------------------------------------------------------

# Phase-derived organelles drop the modality parenthetical entirely —
# the panel title already says "Phase" so "(phase)" on every row would
# just be noise. Structure-name carries the meaning (cell body vs
# nucleoli vs dark vacuoles vs ...) and the reader knows which modality
# the panel covers from the title.
ORGANELLE_DISPLAY: dict[str, tuple[str, str]] = {
    # `*_tubular` features describe the FILAMENT-NETWORK segmentation
    # (long thin connected structures detected in phase contrast / focus
    # stack), NOT the whole cell silhouette. Earlier versions labelled
    # these as "cell body" / "cells" which mis-reads `tubular_area_mean`
    # as cell shape — they're tubules. The actual cell shape lives in
    # `cp_cell_*` features which still map to "whole cell" below.
    "phase2d_tubular":        ("tubules",           "tubules"),
    "focus3d_tubular":        ("tubules",           "tubules"),
    "nucleoli_focus3d":       ("nucleolus",         "nucleoli"),
    "nucleoli_phase2d":       ("nucleolus",         "nucleoli"),
    "focus3d_vesicular_dark": ("dark vacuoles",     "dark vacuoles"),
    "focus3d_vesicular":      ("light vacuoles",    "light vacuoles"),
    "phase2d_vesicular":      ("light vacuoles",    "light vacuoles"),
    "phase2d_vesicular_dark": ("dark vacuoles",     "dark vacuoles"),
    "nuclei":                 ("nucleus",           "nucleus"),
    "cell":                   ("whole cell",        "whole-cell morphology"),
    "cp_cell":                ("whole cell",        "whole-cell morphology"),
    # CellProfiler compartments backfilled in `ko_shap_features._build_cache`
    # when the h5ad var metadata is missing. Cytoplasm groups with whole
    # cell (cytoplasm = cell − nucleus); nucleus stays nuclear.
    "cp_cytoplasm":           ("whole cell",        "whole-cell morphology"),
    "cp_nucleus":             ("nucleus",           "nucleus"),
}

# ---------------------------------------------------------------------------
# Count-tail support (opt-in via --include-counts)
# ---------------------------------------------------------------------------
# Used only when `generate_captions(..., include_counts=True)` is set.
# Brings back the prior unconditional `~52% more X objects ***` tails
# appended to each channel section. The default caption flow now relies
# on count features that land in the top SHAP rows themselves, but the
# canonical-phenotype validation showed that for some KOs (TIMM mito
# loss, HSPA5 ERAD, POLR1B nucleolar collapse, GBF1 LD accumulation) the
# count tail is what actually surfaces the literature-known phenotype.
ORG_COUNT_FEATURE: dict[str, str] = {
    "nucleoli_phase2d":       "op_nucleoli_phase2d_count",
    "nucleoli_focus3d":       "op_nucleoli_focus3d_count",
    "focus3d_vesicular_dark": "op_focus3d_vesicular_dark_count",
    "focus3d_vesicular":      "op_focus3d_vesicular_count",
    "phase2d_vesicular":      "op_phase2d_vesicular_count",
    "phase2d_vesicular_dark": "op_phase2d_vesicular_dark_count",
    "phase2d_tubular":        "op_phase2d_tubular_count",
    "focus3d_tubular":        "op_focus3d_tubular_count",
    "nuclei":                 "op_nuclei_count",
}
# Per-channel count feature for fluor (single feature, no per-organelle
# disambiguation needed).
FLUOR_COUNT_FEATURE = "op_count"

# Display short name → list of count features whose raw token in
# `ORG_COUNT_FEATURE` resolves to that display name. Phase has multiple
# count metrics per organelle (e.g. nucleolus has both phase2d and
# focus3d count features); the lookup picks the one with the largest
# |delta| so the caption surfaces the strongest abundance signal.
_DISPLAY_TO_COUNT_FEATURES: dict[str, list[str]] = {}
for _raw_token, _count_feat in ORG_COUNT_FEATURE.items():
    if _raw_token in ORGANELLE_DISPLAY:
        _display_short = ORGANELLE_DISPLAY[_raw_token][0]
        _DISPLAY_TO_COUNT_FEATURES.setdefault(_display_short, []).append(_count_feat)
del _raw_token, _count_feat, _display_short

# Non-punctate fluor markers — count is unreliable for these because the
# segmenter fragments a continuous structure (membrane / filament /
# diffuse stain) into a brightness-dependent number of pieces. Kept in
# the module for future callers; current `_fluor_lookup` does NOT filter
# on these (matches the prior "emit count tail for ALL channels" policy).
NON_PUNCTATE_KEYWORDS = (
    "actin", "filament", "microtubule", "tubulin",
    "plasma membrane", "membrane",
    "endoplasmic reticulum", "er/golgi",
    "chromatin", "chaperone", "laminin", "lamina",
    "proliferation",
    "oxidative stress", "cellrox",
    "hoechst",
    "live cell dye", "chromalive",
)


def _structure_supports_count(structure):
    """True iff the structure name suggests discrete punctate objects so
    the count abundance reading is meaningful (and not dominated by
    fragmentation/brightness artifacts)."""
    s = (structure or "").lower()
    return not any(kw in s for kw in NON_PUNCTATE_KEYWORDS)


# (prefix_to_match, org_key, display_name) — checked in order.
# Display names omit probe-target parentheticals (e.g. "(TOMM20)",
# "(FeRhoNox)", "(CellROX)") because the atlas panel title already
# carries the raw channel ID. Compartment/isoform parentheticals like
# "(COP-II)" and "(Lamin B)" are kept since they convey biology beyond
# the probe label.
_VIZ_CHANNEL_MAP: list[tuple[str, str, str]] = [
    ("nucleolus",                  "nucleolus",       "nucleolus"),
    ("actin filament",             "actin",           "actin filament"),
    ("F-actin",                    "actin",           "actin filament"),
    ("Mitochondria_TOMM20",        "mitochondria",    "mitochondria"),
    ("mitochondria",               "mitochondria",    "mitochondria"),
    ("Mitochondria",               "mitochondria",    "mitochondria"),
    ("lysosome",                   "lysosome",        "lysosome"),
    ("Lysosome",                   "lysosome",        "lysosome"),
    ("autophagosome",              "autophagosome",   "autophagosome"),
    ("lipid droplet",              "lipid droplet",   "lipid droplet"),
    ("stress granule",             "stress granule",  "stress granule"),
    ("nuclear speckles",           "nuclear speckles","nuclear speckles"),
    ("peroxisome",                 "peroxisome",      "peroxisome"),
    ("ER/Golgi COP-II",            "COPII",           "ER/Golgi (COP-II)"),
    ("ER/Golgi",                   "golgi",           "ER/Golgi"),
    ("ER/golgi",                   "golgi",           "ER/Golgi"),
    ("trans-Golgi",                "golgi",           "trans-Golgi"),
    ("cis-Golgi",                  "golgi",           "cis-Golgi"),
    ("ER_",                        "ER",              "ER"),
    ("plasma membrane",            "plasma membrane", "plasma membrane"),
    ("Plasma Membrane",            "plasma membrane", "plasma membrane"),
    ("late endosome",              "endosome",        "late endosome"),
    ("recycling endosome",         "endosome",        "recycling endosome"),
    ("endocytic vesicle",          "endosome",        "endocytic vesicle"),
    ("clathrin vesicles",          "endosome",        "clathrin vesicles"),
    ("early endosome",             "endosome",        "early endosome"),
    ("endosome",                   "endosome",        "endosome"),
    ("microtubules",               "microtubules",    "microtubules"),
    ("Microtubules",               "microtubules",    "microtubules"),
    ("laminin",                    "nuclear lamina",  "nuclear lamina (Lamin B)"),
    ("Fe2+",                       "fe2+",            "Fe²⁺"),
    ("caspase activity",           "apoptosis",       "caspase activity"),
    ("oxidative stress",           "oxidative stress","oxidative stress"),
    ("cell proliferation",         "proliferation",   "proliferation"),
    ("chaperone",                  "chaperone",       "chaperone"),
    ("ChromaLIVE 561",             "mitochondria",    "mitochondria"),
    ("ChromaLIVE 488",             "gfp_live",        "live-cell fluorescence"),
    ("5xUPRE",                     "UPR",             "UPR reporter"),
    ("nucleus",                    "nucleus",         "nucleus"),
    ("Nucleus",                    "nucleus",         "nucleus"),
    ("proteasome",                 "proteasome",      "proteasome"),
]

_ORG_KEY_PRIORITY: list[str] = [
    "nucleolus", "mitochondria", "actin", "ER", "plasma membrane", "golgi",
    "COPII", "microtubules", "nuclear lamina", "lipid droplet", "autophagosome",
    "lysosome", "endosome", "stress granule", "nuclear speckles", "peroxisome",
    "proteasome", "nucleus", "phase contrast", "focus-3D", "proliferation",
    "chaperone", "apoptosis", "oxidative stress", "UPR", "fe2+", "gfp_live", "gfp",
]

_LOCALIZATION_TOKENS = ("shifted", "redistributed", "concentrated near", "toward", "away from", "perinuclear")
_SIGNAL_TOKENS = ("signal", "intensity")


# ---------------------------------------------------------------------------
# Viz-channel parsing
# ---------------------------------------------------------------------------

def _org_key_rank(ok: str) -> int:
    try:
        return _ORG_KEY_PRIORITY.index(ok)
    except ValueError:
        return len(_ORG_KEY_PRIORITY)


def _parse_viz_channel(vc: str) -> tuple[str, str]:
    # Probe-name parenthetical (e.g. "(SRRM2)" / "(G3BP1)") removed from
    # the in-caption display string — the atlas panel title already shows
    # the raw channel ID (`nuclear speckles_SRRM2`) and the section
    # header in the caption echoes it, so repeating the marker inside
    # the descriptive text was just noise.
    for prefix, org_key, display in _VIZ_CHANNEL_MAP:
        if vc.startswith(prefix):
            return org_key, display
    parts = vc.split("_", 1)
    return parts[0].lower().replace(" ", "_"), vc


def _resolve_fluor(viz_channels_str: str) -> tuple[str, str]:
    """Map viz_channels string → (org_key, org_long) for fluor organelle label resolution."""
    if not viz_channels_str or pd.isna(viz_channels_str):
        return "gfp", "fluorescent reporter"
    channels = [c.strip() for c in viz_channels_str.split("|") if c.strip() != "Phase"]
    if not channels:
        return "gfp", "fluorescent reporter"
    parsed = [_parse_viz_channel(c) for c in channels]
    groups: dict[str, list[str]] = {}
    for ok, display in parsed:
        groups.setdefault(ok, []).append(display)
    merged: dict[str, str] = {}
    for ok, displays in groups.items():
        if len(displays) == 1:
            merged[ok] = displays[0]
        else:
            base = re.sub(r"\s*\(.*\)$", "", displays[0]).strip()
            probes = [m.group(1) for d in displays if (m := re.search(r"\(([^)]+)\)$", d))]
            merged[ok] = f"{base} ({' / '.join(probes)})" if probes else base
    best_key = min(groups.keys(), key=_org_key_rank)
    other_keys = sorted([k for k in groups if k != best_key], key=_org_key_rank)
    all_displays = [merged[best_key]] + [merged[k] for k in other_keys]
    return best_key, " / ".join(all_displays)


# ---------------------------------------------------------------------------
# Feature → phrase rules
# ---------------------------------------------------------------------------

# (regex_pattern, phrase_if_up, phrase_if_down) — matched in order
MEASURE_RULES: list[tuple[str, str, str]] = [
    # _std variants before generic fallbacks
    # Note: _std = std across organelle instances within a single cell (within-cell heterogeneity),
    # NOT cell-to-cell variability. E.g. intensity_mean_std = some organelles bright, others dim within same cell.
    # Note: previously appended "within cell" to disambiguate "across
    # organelle instances within a single cell" from cell-to-cell. But
    # when the natural-phrase layer pairs this with an organelle name
    # ("...within cell in cell body"), the result reads as a doubled noun.
    # Drop "within cell" — the natural-phrase "in {organelle}" prepositional
    # already carries the "within {structure} of a cell" meaning.
    (r"intensity_.*_std$",              "heterogeneous signal intensity",                "uniform signal intensity"),
    (r"distance_from_cell_edge.*_std$", "scattered organelle positioning",              "uniform organelle positioning"),
    (r"distance_from_nucleus.*_std$",   "scattered perinuclear distribution",           "uniform perinuclear distribution"),
    (r"normalized_radial_position_std", "scattered radial positioning",                 "uniform radial positioning"),
    (r"_std$",                          "more heterogeneous",                           "more uniform"),
    # network topology
    (r"num_branches",               "more network branches",              "fewer network branches"),
    (r"euler_number",               "more fragmented network",            "more interconnected network"),
    (r"largest_connected_component","larger connected network component", "smaller connected network component"),
    (r"branch_thickness",           "thicker network filaments",          "thinner network filaments"),
    (r"branch_length",              "longer network filaments",           "shorter network filaments"),
    (r"network_length_density",     "denser network",                     "sparser network"),
    (r"branching_density",          "more branched network",              "less branched network"),
    (r"tortuosity",                 "more tortuous, curved filaments",    "straighter filaments"),
    (r"skeleton_pixel_count",       "longer network",                     "shorter network"),
    (r"total_branch_length",        "longer network filaments",           "shorter network filaments"),
    (r"num_nodes",                  "more network branch points",         "fewer network branch points"),
    (r"num_endpoints",              "more free filament ends",            "fewer free filament ends"),
    # localization
    (r"distance_from_cell_edge",    "shifted toward cell center",                       "shifted toward cell periphery"),
    (r"distance_from_nucleus",      "redistributed away from nucleus",                  "concentrated near nucleus"),
    (r"normalized_radial_position", "shifted toward cell periphery",                    "concentrated near cell center (perinuclear)"),
    (r"centroid",                   "altered spatial distribution",                     "altered spatial distribution"),
    # intensity
    (r"intensity_mean_sum",         "more organelles or denser signal",                 "fewer organelles or sparser signal"),
    (r"intensity_min_sum",          "increased total organelle signal",                 "decreased total organelle signal"),
    (r"intensity_max_sum",          "more / brighter organelles",                       "fewer / dimmer organelles"),
    (r"intensity_mean",             "more abundant / denser signal",                    "less abundant / sparser signal"),
    (r"intensity_max",              "brighter peak signal",                             "dimmer peak signal"),
    (r"intensity_min",              "elevated baseline signal",                         "reduced baseline signal"),
    (r"intensity_range",            "higher peak signal relative to background",        "lower peak signal relative to background"),
    (r"intensity_std",              "more punctate signal",                             "more diffuse signal"),
    (r"intensity_iqr",              "wider intensity spread",                           "narrower intensity spread"),
    (r"intensity_q75",              "brighter signal",                                  "dimmer signal"),
    # size / shape
    (r"^area\b",                    "enlarged cells",                                   "smaller cells"),
    (r"^axis_major_length",         "more elongated",                                   "more compact"),
    (r"equivalent_diameter_area",   "larger structures",                                "smaller structures"),
    (r"eccentricity",               "more elongated",                                   "rounder"),
    (r"circularity",                "rounder",                                          "more irregular"),
    (r"solidity",                   "rounder / more convex",                            "more irregular / concave"),
    (r"^extent",                    "more compact shape",                               "more elongated / irregular shape"),
    (r"aspect_ratio",               "more elongated",                                   "rounder"),
    (r"perimeter",                  "more irregular boundary",                          "smoother boundary"),
    (r"area\b",                     "larger structures",                                "smaller structures"),
    (r"axis_major_length",          "more elongated",                                   "more compact"),
    (r"axis_minor_length",          "wider",                                            "narrower"),
    (r"inertia_eigval_0",           "rounder",                                          "more elongated"),
    (r"inertia_eigval_1",           "more elongated",                                   "rounder"),
    # moments
    (r"moments_weighted_hu_0",      "more dispersed signal",                            "more concentrated signal"),
    (r"moments_weighted_hu_1",      "more elongated signal distribution",               "rounder signal distribution"),
    (r"moments_weighted_hu",        "more complex signal distribution",                 "more regular signal distribution"),
    (r"hu_moment_0",                "more spread-out",                                  "more compact"),
    (r"hu_moment_1",                "more elongated",                                   "rounder"),
    (r"hu_moment",                  "more complex shape",                               "more regular shape"),
    # vesicular
    (r"vesicular.*dark.*area",      "more / larger dark vacuoles",                      "fewer / smaller dark vacuoles"),
    (r"vesicular.*dark",            "increased dark vacuolar content",                  "decreased dark vacuolar content"),
    (r"vesicular.*area",            "more / larger vesicles",                           "fewer / smaller vesicles"),
    (r"vesicular",                  "more vesicular structures",                        "fewer vesicular structures"),
    # extra coverage to plug common fall-throughs to generic "elevated signal"
    (r"orientation_std",            "more disordered orientation",                      "more aligned orientation"),
    (r"^orientation",               "rotated orientation",                              "rotated orientation"),
    (r"_q\d{1,2}\b|quantile",       "shifted higher in the value distribution",         "shifted lower in the value distribution"),
    (r"branching_density",          "more branched / dense network",                    "less branched / sparser network"),
    (r"num_branches",               "more network branches",                            "fewer network branches"),
    (r"network_length",             "longer total network length",                      "shorter total network length"),
    (r"focus_score|sharpness",      "sharper focal structures",                         "softer focal structures"),
]


def _measure_phrase(measure_token: str, direction: float) -> str:
    """Map a single feature measure token + direction to a plain-English phrase."""
    up = direction > 0
    if "vesicular" in measure_token.lower():
        for pattern, up_text, down_text in MEASURE_RULES:
            if "vesicular" in pattern and re.search(pattern, measure_token, re.IGNORECASE):
                return up_text if up else down_text
    for pattern, up_text, down_text in MEASURE_RULES:
        if re.search(pattern, measure_token, re.IGNORECASE):
            return up_text if up else down_text
    return "elevated signal" if up else "reduced signal"


def _parse_measure(feat: str, organelle: str) -> str:
    return feat.removeprefix("op_").removeprefix(organelle + "_")


# ---------------------------------------------------------------------------
# Multi-feature synthesis per organelle
# ---------------------------------------------------------------------------

def _synthesize_organelle_phrase(org_features: list[dict], org_key: str = "") -> str | None:
    """Collapse all features for one organelle into a single descriptive phrase.

    Priority order: size → shape → intensity → network → localization.
    Returns None when all effects are near-zero (noise floor).
    """
    size_votes: list[float] = []
    count_votes: list[float] = []
    shape_traits: list[str] = []
    # Up to 2 DISTINCT phrases per category for network + localization
    # so distinct top-SHAP features (e.g. AGAP9 mitochondria's
    # branch_thickness + num_nodes; or a gene with both nuclear-distance
    # and cell-edge-distance signals) don't silently drop whichever
    # ranks lower in SHAP.
    network_phrases: list[str] = []
    localization_phrases: list[str] = []
    bright_effects: list[float] = []  # intensity level (mean, max, range, quantile)
    var_effects: list[float] = []     # within-cell heterogeneity (std/iqr across organelle instances)

    for f in org_features:
        m = f["measure"]
        # Prefer effect_size sign over SHAP direction sign — more reliable
        effect = f.get("effect_size")
        d = float(np.sign(effect)) if effect is not None and effect != 0 else float(f["direction"])
        eff_val = float(effect) if effect is not None else d

        # Count — only described when count is itself a top-SHAP feature for
        # the gene/channel (the unconditional "~X% more nucleoli" tail was
        # removed, so count info now flows exclusively through the synth).
        # Exclude network-topology counts (branch_count, skeleton_pixel_count,
        # num_*) — those are network metrics, not organelle abundance, and
        # voting them as "more abundant" produces false claims like
        # "more abundant mitochondria, ...; 44% fewer mitochondria objects".
        if re.search(r"(^|_)count$", m, re.I) \
                and not re.search(
                    r"branch_count|skeleton_pixel_count|skeleton.*count|"
                    r"^num_|_num_", m, re.I,
                ):
            count_votes.append(d)

        # Size
        if re.search(r"^area|area_filled|equivalent_diameter_area|axis_major_length|axis_minor_length", m, re.I) \
                and not re.search(r"_std(?:_|$)", m, re.I):
            size_votes.append(d)

        # Shape — collect up to 2 distinct traits
        has_round = any(t in ("rounder", "more elongated") for t in shape_traits)
        if not has_round and re.search(r"circularity|eccentricity|aspect_ratio|inertia_eigval", m, re.I):
            if re.search(r"circularity|inertia_eigval_0", m, re.I):
                shape_traits.append("rounder" if d > 0 else "more elongated")
            else:
                shape_traits.append("more elongated" if d > 0 else "rounder")
        if re.search(r"solidity", m, re.I) and "rounder / more convex" not in shape_traits:
            shape_traits.append("rounder / more convex" if d > 0 else "more irregular / concave")
        # extent = object_area / bounding_box_area; up=compact, down=elongated/irregular
        if re.search(r"^extent", m, re.I) and not re.search(r"_std(?:_|$)", m, re.I) \
                and not any(t in shape_traits for t in ("more compact", "more elongated / irregular")):
            shape_traits.append("more compact shape" if d > 0 else "more elongated / irregular shape")

        # Network — collect up to 2 distinct phrases. Eligible features
        # cover topology (branches, nodes, endpoints), thickness/length,
        # density, fragmentation, tortuosity, and connected-component
        # extent. Dedupe by the rendered phrase, so e.g. both
        # `branch_length` and `total_branch_length` (which collapse to the
        # same wording) only count once but `branch_thickness` +
        # `num_nodes` (different wording) BOTH appear.
        if len(network_phrases) < 2 and re.search(
                r"branch_thickness|num_branches|num_nodes|num_endpoints|"
                r"tortuosity|network_length|branching_density|"
                r"skeleton_pixel_count|euler_number|"
                r"largest_connected_component|"
                r"total_branch_length|branch_length", m, re.I):
            phrase = _measure_phrase(m, d)
            if phrase not in network_phrases:
                network_phrases.append(phrase)

        # Localization — collect up to 2 distinct phrases. Three
        # eligible patterns map to biologically distinct readings:
        # `normalized_radial_position` (radial), `distance_from_nucleus`
        # (perinuclear), `distance_from_cell_edge` (cortical). When two
        # of them are top features, we want both surfaced. Skip
        # `distance_from_nucleus` for nuclear markers since "redistributed
        # away from the nucleus" is meaningless for a nuclear reporter.
        is_nuclear_marker = org_key in ("nucleus", "nucleolus", "proliferation", "nuclear speckles", "nuclear lamina")
        if len(localization_phrases) < 2 \
                and re.search(r"normalized_radial_position|distance_from_nucleus|distance_from_cell_edge", m, re.I) \
                and not re.search(r"_std(?:_|$)", m, re.I) \
                and not (is_nuclear_marker and re.search(r"distance_from_nucleus", m, re.I)):
            phrase = _measure_phrase(m, d)
            if phrase not in localization_phrases:
                localization_phrases.append(phrase)

        # Intensity: brightness (mean/max/range/quantile) vs within-cell heterogeneity (std/iqr).
        # Use (?:_|$) instead of \b because _ is a word character in Python regex,
        # so intensity_max_max would not match a \b after "max".
        # intensity_range (peak-to-background per object) = brightness, not variability.
        # intensity_std / intensity_iqr = spread of pixel values within each object = punctateness.
        # intensity_.*_std = std of a feature across organelle instances within one cell = heterogeneity.
        if re.search(r"intensity_(?:std|iqr)(?:_|$)|intensity_.*_std$", m, re.I):
            var_effects.append(eff_val)
        elif re.search(r"intensity_(?:mean|max|range|q\d+)(?:_|$)", m, re.I) \
                and not re.search(r"_sum$|_std$", m, re.I):
            bright_effects.append(eff_val)

    # Synthesize intensity into one phrase. Floor lowered to 0.05 so
    # subtle but real shifts still surface (with the "very subtly"
    # qualifier conveying magnitude). Removes a second redundant
    # gate — the qualifier ladder alone communicates effect size.
    intensity_phrase: str | None = None
    all_intensity = bright_effects + var_effects
    if all_intensity and max(abs(e) for e in all_intensity) >= 0.05:
        bright_dir = float(np.sign(sum(bright_effects))) if bright_effects else None
        var_dir    = float(np.sign(sum(var_effects)))    if var_effects    else None
        if bright_dir is not None and var_dir is not None:
            b = "elevated" if bright_dir > 0 else "reduced"
            v = "heterogeneous" if var_dir > 0 else "uniform"
            # Use "and" when the two directions agree, " yet " when they
            # disagree — both keep the bright+var info bound to "signal"
            # in a single phrase. We avoid a bare "," joiner because
            # _natural_phrase splits the synth's parts on commas to
            # separate morph from signal — a "," would orphan the bright
            # word ("reduced, heterogeneous signal" → bare "reduced").
            joiner = " and " if (bright_dir > 0) == (var_dir > 0) else " yet "
            intensity_phrase = f"{b}{joiner}{v} signal"
        elif bright_dir is not None:
            intensity_phrase = "elevated signal" if bright_dir > 0 else "reduced signal"
        elif var_dir is not None:
            intensity_phrase = "heterogeneous signal" if var_dir > 0 else "uniform signal"

    parts: list[str] = []
    if size_votes:
        pos = sum(1 for v in size_votes if v > 0)
        neg = len(size_votes) - pos
        if not (pos > 0 and neg > 0):  # conflicting votes → skip size
            parts.append("smaller" if sum(size_votes) / len(size_votes) < 0 else "larger")
    if count_votes:
        # Renders as "more abundant {organelle}" / "less abundant {organelle}"
        # via the default _natural_phrase suffix path.
        s = sum(count_votes)
        if s != 0:
            parts.append("more abundant" if s > 0 else "less abundant")
    parts.extend(shape_traits[:2])
    if intensity_phrase:
        parts.append(intensity_phrase)
    if network_phrases:
        parts.extend(network_phrases)
    if localization_phrases:
        parts.extend(localization_phrases)
    parts = _dedupe_concepts(parts)
    # Cap bumped 4 → 5 so a typical "size + shape + intensity + 2 network"
    # caption keeps both network phrases without dropping localization.
    if parts:
        return ", ".join(parts[:5])

    # No generic-direction fallback. If none of the structured rules
    # above (size / count / shape / intensity / network / localization)
    # produced a phrase for this organelle group, return None and let
    # the caller skip the group entirely. The old fallback path
    # mapped unrecognized features (SpatialMoment, CentralMoment,
    # orientation, etc.) to a meaningless "elevated / reduced signal"
    # phrase — biologically opaque AND, for whole-cell rows whose
    # only SHAP features were spatial moments, directly contradicted
    # what the violin showed for the channel's interpretable features.
    return None


def _effect_qualifier(max_abs_effect: float | None) -> str:
    """Adverb prefix based on largest |effect_size| in the organelle group.

    Tiers tuned for sensitivity to subtle effects — many real
    perturbations land in the 0.05–0.2 range and the prior thresholds
    (0.1/0.3/0.8) drowned those in "very subtly" or skipped them
    entirely. Current ladder: |eff|≥0.5 unqualified (strong);
    ≥0.2 mildly; ≥0.05 subtly; below floor very subtly. The
    structured-rule gate at the synth level still controls whether a
    feature gets a phrase at all.
    """
    if max_abs_effect is None:
        return ""
    if max_abs_effect >= 0.5:
        return ""
    if max_abs_effect >= 0.2:
        return "mildly "
    if max_abs_effect >= 0.05:
        return "subtly "
    return "very subtly "


# Phrases that, once present in the parts list, suppress later phrases
# carrying the same core concept. Multiple SHAP rules can fire for the
# same shape category (e.g. eccentricity → "more elongated" AND extent →
# "more elongated / irregular shape"), so the synth's parts list ends up
# with redundant overlapping descriptions. This map collapses them: when
# any phrase containing one of these tokens is added, later phrases
# containing it are skipped.
_DEDUPE_TOKENS = (
    "elongated", "rounder", "compact", "irregular",
    "convex", "concave", "smaller", "larger",
)


def _dedupe_concepts(parts: list[str]) -> list[str]:
    """Drop later phrases whose core shape concept is already represented.

    Walks `parts` in order; for each phrase, finds which `_DEDUPE_TOKENS`
    it contains. Skips the phrase if any of those tokens have already
    appeared in an earlier phrase. Preserves the first occurrence so
    earlier (more specific) rules win.
    """
    seen_tokens: set[str] = set()
    out: list[str] = []
    for p in parts:
        p_low = p.lower()
        tokens_here = {t for t in _DEDUPE_TOKENS if t in p_low}
        if tokens_here & seen_tokens:
            continue
        seen_tokens |= tokens_here
        out.append(p)
    return out


# ---------------------------------------------------------------------------
# Per-gene phrase building
# ---------------------------------------------------------------------------

def _resolve_organelle(org_col: str, feat_str: str, fluor_key: str, fluor_long: str) -> tuple[str, str]:
    """Map organelle column value + feature name → (org_key, org_long).

    Resolution order:
      1. fluor sentinels (`gfp` / `fluor_unified`) → fluor reporter resolution
      2. explicit organelle in `ORGANELLE_DISPLAY` (now includes
         `cp_cell`, `cp_cytoplasm`, `cp_nucleus` after the v3 backfill)
      3. legacy fallback: feature-name inference for empty / unknown
         CP-prefixed organelle values (kept for old caches / partially-
         labeled CSVs).
    """
    if org_col in ("gfp", "fluor_unified"):
        return fluor_key, fluor_long
    # Explicit organelle wins — order matters: cp_cell / cp_cytoplasm /
    # cp_nucleus are now in ORGANELLE_DISPLAY thanks to the v3 backfill,
    # so we hit this branch before the legacy cp_-prefix inference.
    if org_col in ORGANELLE_DISPLAY:
        return ORGANELLE_DISPLAY[org_col]
    # Legacy path: organelle column missing or has an unknown cp_* token.
    if org_col in ("nan", "None", "") or org_col.startswith("cp_"):
        feat_low = feat_str.lower()
        if org_col.startswith("cp_cytoplasm") or "cp_cytoplasm" in feat_low:
            return "whole cell", "whole-cell morphology"
        if org_col.startswith("cp_cell") or "cp_cell" in feat_low:
            return "whole cell", "whole-cell morphology"
        if org_col.startswith("cp_nucleus") or "cp_nucleus" in feat_low \
                or org_col == "CP1_nuclear":
            return "nucleus", "nucleus"
        if "cytoplasm" in feat_low:
            return "whole cell", "whole-cell morphology"
        if "nucleus" in feat_low or "nuclei" in feat_low:
            return "nucleus", "nucleus"
        if "cell" in feat_low:
            return "whole cell", "whole-cell morphology"
        return "nucleus", "nucleus"
    return ORGANELLE_DISPLAY.get(org_col, (org_col, org_col))


def _is_organelle_specific(org_col: str, feat_str: str) -> bool:
    """True iff a feature describes the channel's reporter (the organelle)
    rather than generic cell- or nucleus-level morphology.

    Used to break the cell/nucleus bias in fluor-channel captions: we
    prioritize features extracted from the fluorescent signal itself
    (organelle == "fluor_unified" / "gfp") over CellProfiler whole-cell or
    nucleus shape/intensity features that happen to also be SHAP-important.
    """
    org_low = str(org_col).lower()
    feat_low = str(feat_str).lower()
    if org_low in ("fluor_unified", "gfp"):
        return True
    if org_low in ("cell", "nuclei", "whole cell", "nucleus"):
        return False
    if org_low.startswith("cp_") or "cp_cell" in feat_low \
            or "cp_cytoplasm" in feat_low or "cp_nucleus" in feat_low:
        return False
    # Otherwise treat as organelle-specific (org column carries an
    # explicit organelle name from the OPS feature pipeline).
    return True


def _build_phrases(modality_df: pd.DataFrame, viz_channels_str: str) -> list[tuple]:
    """Build [(org_key, org_long, phrase, net_dir), ...] for one modality.

    Returns at most 3 entries, one per distinct organelle group.

    Bias fix: for fluor channels with a known organelle reporter, features
    describing the organelle itself (organelle column = "fluor_unified" /
    "gfp" / matching the fluor key) come first, ahead of generic CP cell/
    nucleus features. This prevents organelle-specific channel captions
    from being dominated by cell-shape morphology that happens to also be
    SHAP-discriminative.

    Direction frame: per-feature `direction` field is taken from the
    `effect_size` column (sign of gene mean − global median, feature-value
    space) — NOT the SHAP `direction` column (prediction-logit space).
    These can disagree for tree classifiers and we always want
    feature-value direction in the synthesized text.
    """
    fluor_key, fluor_long = _resolve_fluor(viz_channels_str)
    has_effect  = "effect_size" in modality_df.columns
    has_pct     = "pct_cells" in modality_df.columns

    # Drop off-channel nuclear features from non-nuclear fluor channels.
    # The fluor SHAP cache stores `nuclei_*` and `cp_nucleus_*` features
    # alongside the channel-specific ones, and they routinely rank in
    # the top-5 because they capture cohort-wide nuclear-stress
    # signatures — but they describe biology orthogonal to the channel's
    # reporter and produce misleading "elevated signal in nucleus"
    # phrases in actin / autophagosome / ER captions.
    #
    # Filter strictly by the `organelle` column (which the v3 cache
    # populates correctly via _infer_cp_metadata + adata.var). Don't
    # touch features by feature-name regex — too easy to over-match
    # adjacent organelles (e.g. `nucleoli_*` differs from `nuclei_*` by
    # one character; nuclear-lamina features have `lamin` not `nuclei`;
    # etc.). If the organelle column says "nuclei" or "cp_nucleus", the
    # row IS nuclear; otherwise leave it alone.
    NUCLEAR_FLUOR_KEYS = {
        "nucleus", "nucleolus", "nuclear speckles",
        "nuclear lamina", "proliferation", "chromatin",
    }
    is_fluor = (
        "modality" in modality_df.columns
        and len(modality_df) > 0
        and (modality_df["modality"] == "fluor").all()
    )
    # Stash the unfiltered rows so we can restore them if the nuclear-drop
    # rule would otherwise leave the caption empty (e.g. retromer ×
    # endosome_VPS35: top SHAP rows are all nuclei features, the few
    # remaining rows are moments / centroids that the synth suppresses,
    # and the channel ends up with no caption text at all). Better to
    # describe nuclear-leaning biology than emit nothing.
    df_pre_nuclear_drop = modality_df
    if is_fluor and fluor_key not in NUCLEAR_FLUOR_KEYS:
        nuclear_org_mask = modality_df["organelle"].astype(str).str.lower().isin(
            {"nuclei", "cp_nucleus"}
        )
        modality_df = modality_df[~nuclear_org_mask].copy()

    # Order: organelle-specific features first (within shap_rank), then
    # generic cell/nucleus features. For phase modality `fluor_key` is the
    # gene's best fluor channel, which still works as a tiebreaker but
    # most phase rows aren't tagged "fluor_unified" so the partition just
    # mirrors shap_rank order.
    def _synthesize(df_local: pd.DataFrame) -> list[tuple]:
        rows_with_priority = []
        for _, row in df_local.iterrows():
            org_col = str(row["organelle"])
            feat_str = str(row["feature"])
            is_specific = _is_organelle_specific(org_col, feat_str)
            rows_with_priority.append((
                0 if is_specific else 1,
                int(row.get("shap_rank", 999)),
                row,
            ))
        rows_with_priority.sort(key=lambda t: (t[0], t[1]))

        org_groups: dict[str, dict] = {}
        for _, _, row in rows_with_priority:
            org_col  = str(row["organelle"])
            feat_str = str(row["feature"])
            org_key, org_long = _resolve_organelle(org_col, feat_str, fluor_key, fluor_long)
            measure  = _parse_measure(feat_str, org_col)
            effect   = float(row["effect_size"]) if has_effect else None
            pct      = float(row["pct_cells"])   if has_pct and not pd.isna(row["pct_cells"]) else None  # type: ignore[arg-type]
            # Use effect_size sign (feature-value space) as the canonical
            # direction; fall back to SHAP `direction` only if effect_size is
            # missing or exactly 0.
            if effect is not None and effect != 0:
                direction = float(np.sign(effect))
            else:
                direction = float(row["direction"])
            if org_key not in org_groups:
                org_groups[org_key] = {"long": org_long, "features": []}
            org_groups[org_key]["features"].append({
                "measure":    measure,
                "direction":  direction,
                "effect_size": effect,
                "pct_cells":  pct,
            })

        # Compute per-group max-|effect| so we can sort phrases by signal
        # strength (instead of the previous shap-rank insertion order).
        group_strength = {
            k: max(
                (abs(f["effect_size"]) for f in v["features"] if f.get("effect_size") is not None),
                default=0.0,
            )
            for k, v in org_groups.items()
        }

        out: list[tuple] = []
        for org_key in sorted(org_groups, key=lambda k: -group_strength[k]):
            info = org_groups[org_key]
            phrase = _synthesize_organelle_phrase(info["features"], org_key=org_key)
            if phrase is None:
                continue
            effects = [abs(f["effect_size"]) for f in info["features"] if f.get("effect_size") is not None]
            max_effect = max(effects) if effects else None
            qualifier = _effect_qualifier(max_effect)

            # Per-cell penetrance suffix removed: it duplicates the pct_cells
            # number that the atlas violin already shows at the right edge of
            # each row. Inline phrases now stay clean ("rounder nucleoli"
            # instead of "rounder nucleoli (~80% of cells)").
            net_dir = sum(f["direction"] for f in info["features"]) / len(info["features"])
            out.append((org_key, info["long"], qualifier + phrase, net_dir))
            if len(out) >= 3:
                break
        return out

    result = _synthesize(modality_df)
    # Fallback: nuclear-drop above can leave channels (e.g. retromer ×
    # endosome_VPS35) with only moments / centroids, which the synth
    # suppresses → empty caption section. Retry with the unfiltered rows
    # so SHAP's actual top features still get described, even if they're
    # nuclear. Better mildly off-target wording than a blank channel.
    if not result and len(modality_df) < len(df_pre_nuclear_drop):
        result = _synthesize(df_pre_nuclear_drop)
    return result


# ---------------------------------------------------------------------------
# Natural-language rendering
# ---------------------------------------------------------------------------

def _natural_phrase(org_key: str, org_long: str, phrase: str) -> str:
    """Convert (org_key, org_long, phrase) into natural word order.

    Preferred forms:
      - "{phrase} {org_long}"          — default
      - "{phrase} in {org_long}"       — when phrase describes signal/intensity
      - "{morph} {org}, {signal}"      — when both morph and signal traits coexist
      - "{phrase} cells (modality)"    — cell body
    """
    if phrase.startswith("no discriminating"):
        return phrase

    # Replace generic "organelle(s)" token with the specific structure name
    if re.search(r"\borganelles?\b", phrase, re.I):
        org_short = re.sub(r"\s*\(.*\)$", "", org_long).strip()
        replacement = "nuclear organelles" if org_key == "nucleus" else org_short
        new_phrase = re.sub(r"\borganelles?\b", replacement, phrase, flags=re.I)
        if new_phrase != phrase:
            probe_m = re.search(r"\(([^)]+)\)$", org_long)
            if probe_m and probe_m.group(1) not in new_phrase and not new_phrase.rstrip().endswith(")"):
                return f"{new_phrase} ({probe_m.group(1)})"
            return new_phrase

    # Cell body: "{phrase} cells (phase-2D)" or "{phrase} in cell body (phase-2D)"
    if "cell body" in org_long:
        mod_m = re.search(r"\((.+?)\)", org_long)
        suffix = f" ({mod_m.group(1)})" if mod_m else ""
        # Use "in cell body" when phrase describes signal or already ends with a structural noun
        _STRUCT_NOUNS = r"\b(filaments?|branches?|network|vacuoles?|vesicles?|structures?|component)\s*$"
        if any(t in phrase.lower() for t in _SIGNAL_TOKENS) or re.search(_STRUCT_NOUNS, phrase, re.I):
            return f"{phrase} in cell body{suffix}"
        return f"{phrase} cells{suffix}"

    # Whole-cell morphology
    if org_key == "whole cell":
        phrase_low = phrase.lower()
        if any(t in phrase_low for t in _SIGNAL_TOKENS):
            return f"{phrase} (whole-cell)"
        if "boundary" in phrase_low:
            return phrase.replace("boundary", "cell boundary", 1)
        if "shape" in phrase_low:
            return phrase.replace("shape", "cell shape", 1)
        if any(w in phrase_low for w in ("rounder", "elongated", "convex", "irregular", "concave")):
            return f"{phrase} cell shape"
        return f"{phrase} cells"

    # Split comma-separated sub-parts and classify each as morph / signal / localization
    sub_parts = [p.strip() for p in phrase.split(",")]
    loc_idx   = {i for i, p in enumerate(sub_parts)
                 if any(t in p.lower() for t in _LOCALIZATION_TOKENS)}
    sig_idx   = {i for i, p in enumerate(sub_parts)
                 if any(t in p.lower() for t in _SIGNAL_TOKENS)} - loc_idx
    morph_idx = set(range(len(sub_parts))) - loc_idx - sig_idx

    # Both morph and signal: attach org to morph, leave signal as trailing clause
    if morph_idx and sig_idx:
        morph_str = ", ".join(sub_parts[i] for i in sorted(morph_idx))
        sig_str   = ", ".join(sub_parts[i] for i in sorted(sig_idx))
        loc_str   = (", " + ", ".join(sub_parts[i] for i in sorted(loc_idx))) if loc_idx else ""
        return f"{morph_str} {org_long}, {sig_str}{loc_str}"

    # Shape + localization
    shape_idx = set(range(len(sub_parts))) - loc_idx
    if loc_idx and shape_idx:
        shape_str = ", ".join(sub_parts[i] for i in sorted(shape_idx))
        loc_str   = ", ".join(sub_parts[i] for i in sorted(loc_idx))
        return f"{shape_str} {org_long}, {loc_str}"
    if loc_idx and not shape_idx:
        return f"{org_long} {phrase}"

    # Signal/intensity → "in {org}"
    if any(t in phrase.lower() for t in _SIGNAL_TOKENS):
        return f"{phrase} in nucleus" if org_key == "nucleus" else f"{phrase} in {org_long}"

    # Default
    return f"{phrase} {org_long}"


def _format_phrases(phrases: list[tuple]) -> str:
    return "; ".join(
        _natural_phrase(org_key, org_long, phrase)
        for org_key, org_long, phrase, _ in phrases
    )


# ---------------------------------------------------------------------------
# Caption assembly
# ---------------------------------------------------------------------------


class _CountIndex:
    """Per-modality count-feature lookup against the SHAP cache.

    Built once per (cache_dir, modality) when --include-counts is set.
    `delta(gene, feature_name)` returns (pct_delta, z_score) or None,
    where pct_delta = (gene_mean − cohort_mean)/cohort_mean and the
    z-score is (gene_mean − cohort_mean) / SEM(gene), the large-cohort
    limit of Welch's t. Caller maps |z| to */**/*** stars and suppresses
    the tail when not statistically significant.

    Two layers of caching:
      1. In-memory cohort-mean cache (`_cohort_mean_cache`) — avoids
         recomputing nanmean over the same scope/column twice within a
         single run. Cohort mean depends only on (col, channel_rank);
         repeated across all genes.
      2. On-disk result cache (`load_disk_cache` / `save_disk_cache`) —
         persists the full {(gene, feature, channel_rank): (pct, z)} map
         to a sidecar pickle keyed by `X.npy` mtime + `obs.parquet` mtime.
         Reruns that don't rebuild the cache (e.g. tweaking caption text)
         load all hits from disk and skip the SHAP-cache reads entirely.
    """

    def __init__(self, cache_dir, modality):
        self.modality = modality
        self.cache_dir = Path(cache_dir)
        cdir = self.cache_dir
        self.X = np.load(cdir / "X.npy", mmap_mode="r")
        self.obs = pd.read_parquet(cdir / "obs.parquet")
        self.fi = {f: i for i, f in enumerate(
            (cdir / "features.txt").read_text().splitlines()
        )}
        # Restrict fluor index to top-attention rows so per-channel means
        # match what SHAP saw as "this gene's attended cells"; phase
        # cache is already top-only.
        if modality == "fluor" and "rank_type" in self.obs.columns:
            mask = self.obs["rank_type"] == "top"
            self._scope_idx = np.where(mask)[0]
        else:
            self._scope_idx = np.arange(len(self.obs))
        # Pre-compute scope_idx per (channel_rank) for fluor — avoids
        # rebuilding the boolean mask on every gene lookup.
        self._scope_idx_by_ch = {}
        if modality == "fluor" and "channel_rank" in self.obs.columns and "rank_type" in self.obs.columns:
            top_mask = self.obs["rank_type"] == "top"
            ch_arr = self.obs["channel_rank"].to_numpy()
            for cr in np.unique(ch_arr):
                self._scope_idx_by_ch[int(cr)] = np.where(
                    top_mask & (ch_arr == cr)
                )[0]
        # In-memory cohort-mean cache: (col, channel_rank or None) → mean
        self._cohort_mean_cache: dict = {}
        # On-disk result cache: (gene, feature, channel_rank) → (pct, z) | None
        self._disk_cache: dict = {}
        # Cache invalidation signature — derived from cache_dir mtimes.
        self._cache_signature = self._compute_signature()

    def _compute_signature(self) -> str:
        """File-mtime fingerprint used to detect when X.npy / obs.parquet
        have been rebuilt (and the disk cache must be discarded)."""
        try:
            x_mtime = (self.cache_dir / "X.npy").stat().st_mtime
            obs_mtime = (self.cache_dir / "obs.parquet").stat().st_mtime
            return f"{x_mtime:.3f}|{obs_mtime:.3f}"
        except FileNotFoundError:
            return ""

    def _disk_cache_path(self, sidecar_dir: Path) -> Path:
        return Path(sidecar_dir) / f".count_cache_{self.modality}.pkl"

    def load_disk_cache(self, sidecar_dir: Path) -> int:
        """Load disk cache from `sidecar_dir`. Returns # of entries loaded
        (0 if file missing, mismatched signature, or unreadable)."""
        path = self._disk_cache_path(sidecar_dir)
        if not path.exists():
            return 0
        import pickle
        try:
            with open(path, "rb") as f:
                payload = pickle.load(f)
            if payload.get("signature") != self._cache_signature:
                return 0   # stale — cache_dir was rebuilt
            self._disk_cache = dict(payload.get("entries", {}))
            return len(self._disk_cache)
        except Exception:
            return 0

    def save_disk_cache(self, sidecar_dir: Path) -> None:
        """Persist the in-memory result cache to `sidecar_dir`."""
        Path(sidecar_dir).mkdir(parents=True, exist_ok=True)
        path = self._disk_cache_path(sidecar_dir)
        import pickle
        payload = {
            "signature": self._cache_signature,
            "entries":   self._disk_cache,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    def _gene_idx(self, gene, channel_rank=None):
        obs = self.obs
        gene_mask = obs["gene"].astype(str) == str(gene)
        if self.modality == "fluor" and channel_rank is not None:
            ch_mask = obs["channel_rank"] == int(channel_rank)
            mask = gene_mask & ch_mask & (obs["rank_type"] == "top")
        else:
            mask = gene_mask
        return np.where(np.asarray(mask))[0]

    def _cohort_mean(self, col, channel_rank=None) -> float:
        """Cohort mean of the feature column over the cohort scope. For
        fluor, scope is top-attention rows in this channel_rank; for
        phase, scope is all rows. Cached per (col, channel_rank)."""
        key = (col, channel_rank if self.modality == "fluor" else None)
        if key in self._cohort_mean_cache:
            return self._cohort_mean_cache[key]
        if self.modality == "fluor" and channel_rank is not None:
            scope = self._scope_idx_by_ch.get(int(channel_rank), self._scope_idx)
        else:
            scope = self._scope_idx
        cohort_vals = np.asarray(self.X[scope, col], dtype=np.float64)
        cohort_mean = float(np.nanmean(cohort_vals))
        self._cohort_mean_cache[key] = cohort_mean
        return cohort_mean

    def delta(self, gene, feature, channel_rank=None):
        # Disk-cache hit fast path
        cache_key = (str(gene), str(feature), int(channel_rank) if channel_rank is not None else -1)
        if cache_key in self._disk_cache:
            return self._disk_cache[cache_key]

        col = self.fi.get(feature)
        if col is None:
            self._disk_cache[cache_key] = None
            return None
        idx = self._gene_idx(gene, channel_rank=channel_rank)
        if len(idx) < 5:
            self._disk_cache[cache_key] = None
            return None
        gene_vals = np.asarray(self.X[idx, col], dtype=np.float64)
        gene_finite = gene_vals[np.isfinite(gene_vals)]
        if len(gene_finite) < 5:
            self._disk_cache[cache_key] = None
            return None
        gene_mean = float(np.mean(gene_finite))
        gene_std  = float(np.std(gene_finite, ddof=1)) if len(gene_finite) > 1 else 0.0
        cohort_mean = self._cohort_mean(col, channel_rank=channel_rank)
        if not np.isfinite(cohort_mean) or cohort_mean == 0:
            self._disk_cache[cache_key] = None
            return None
        pct = (gene_mean - cohort_mean) / cohort_mean
        sem = gene_std / np.sqrt(len(gene_finite))
        if sem == 0 or not np.isfinite(sem):
            self._disk_cache[cache_key] = None
            return None
        z = (gene_mean - cohort_mean) / sem
        result = (pct, float(z))
        self._disk_cache[cache_key] = result
        return result


def _pluralize(word):
    """Lightweight English/Latin plural for the structure-name family."""
    w = (word or "").strip()
    if not w:
        return w
    lw = w.lower()
    if lw in {"er/golgi", "plasma membrane", "proliferation"} \
            or lw.startswith("er/golgi") or "/" in lw:
        return w
    if lw.endswith("speckles"): return w
    if lw.endswith("speckle"):  return w + "s"
    if lw.endswith("us"):       return w[:-2] + "i"
    if lw.endswith("um"):       return w[:-2] + "a"
    if lw.endswith(("s", "ia", "ae")):  return w
    return w + "s"


def _sig_stars(z):
    """|z| → significance markers. *** p<0.001, ** p<0.01, * p<0.05, '' otherwise."""
    if z is None:
        return ""
    az = abs(z)
    if az >= 3.29: return "***"
    if az >= 2.58: return "**"
    if az >= 1.96: return "*"
    return ""


def _format_count_tail(result, structure):
    """Render '52% more mitochondria objects' when |pct|≥5% AND |z|≥1.96.

    Significance is still enforced (|z|≥1.96 = p<0.05) — but the */**/***
    star annotation is dropped from the visible caption since readers
    found it noisy. The cell-level z-score still drives the show/suppress
    threshold so non-significant shifts don't get an empty claim."""
    if result is None:
        return ""
    pct, z = result
    if pct is None or z is None or abs(pct) < 0.05:
        return ""
    if abs(z) < 1.96:    # not significant at p < 0.05
        return ""
    word = "more" if pct > 0 else "fewer"
    return f"{abs(pct) * 100:.0f}% {word} {structure} objects"


def _confidence(auroc: float) -> str:
    if auroc >= 0.80: return "high confidence"
    if auroc >= 0.70: return "reliable"
    if auroc >= 0.65: return "moderate confidence"
    if auroc >= 0.60: return "weakly supported"
    return "near-chance"


def _section_count_tails(phrases, count_lookup):
    """Build "52% more X objects" tail(s) for a channel section.
    `count_lookup(org_key, org_long)` returns (tail, structure) | None.
    Dedupes by structure so a section with three organelle phrases reads
    e.g. "...; 12% fewer dark vacuoles, 8% more nucleoli".
    """
    if count_lookup is None:
        return ""
    seen_structures = set()
    tails = []
    for org_key, org_long, _phrase, _net_dir in phrases:
        result = count_lookup(org_key, org_long)
        if result is None:
            continue
        tail, structure = result
        if not tail or structure in seen_structures:
            continue
        seen_structures.add(structure)
        tails.append(tail)
    return ", ".join(tails)


def _make_caption(
    gene: str,
    phase_phrases: list[tuple],
    fluor_sections: list[tuple[str, float, list[tuple]]],  # (ch_name, auroc, phrases)
    phase_auroc: float,
    phase_count_lookup=None,    # callable(org_key, org_long) -> (tail, structure) | None
    fluor_count_lookup=None,    # callable(channel_rank) -> (tail, structure) | None
    fluor_channel_ranks=None,   # list[int] aligned with fluor_sections
    header_label: str = "geneKO",
) -> str:
    """Caption header: `{gene} {header_label}:`. AUROC parenthetical
    lives in the atlas violin panel title, not here.

    `header_label` defaults to "geneKO" (distinctiveness atlas) and is
    overridden by the NTC/median callers to "KO vs NTC" / "KO vs
    cohort" framing.

    When `*_count_lookup` is provided (--include-counts opt-in), each
    channel section gets a "~XX% more|fewer {structure} objects ***"
    abundance tail computed against the SHAP cache. Default flow leaves
    these silent and lets count info flow through the synth's
    `count_votes` path when a `_count` feature lands in top SHAP.
    """
    header = f"{gene} {header_label}"

    sections = []
    if phase_phrases:
        body = _format_phrases(phase_phrases)
        tails = _section_count_tails(phase_phrases, phase_count_lookup)
        if tails:
            body = f"{body}; {tails}"
        sections.append(f"Phase — {body}")
    fluor_channel_ranks = fluor_channel_ranks or [None] * len(fluor_sections)
    for (ch_name, _ch_auroc, ch_phrases), ch_rank in zip(fluor_sections, fluor_channel_ranks):
        if not ch_phrases:
            continue
        body = _format_phrases(ch_phrases)
        if fluor_count_lookup is not None and ch_rank is not None:
            ftail = fluor_count_lookup(ch_rank)
            if ftail:
                tail, _structure = ftail
                if tail:
                    body = f"{body}; {tail}"
        sections.append(f"{ch_name} — {body}")

    if not sections:
        return f"{header}: no distinctive morphological features detected."
    return header + ": " + "; ".join(sections) + "."


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

CONTRAST_HEADER_LABEL = {
    "distinct": "geneKO",
    "ntc":      "KO vs NTC",
    "global":   "KO vs cohort",
}


def generate_captions(in_csv: Path, out_csv: Path,
                      include_counts: bool = False,
                      cache_phase: Path | None = None,
                      cache_fluor: Path | None = None,
                      contrast: str = "distinct") -> pd.DataFrame:
    """Generate per-gene captions from SHAP features CSV.

    Args:
        in_csv:  Path to ko_shap_features*.csv with columns
                 [gene, modality, channel_rank, shap_rank, feature, organelle,
                  direction, auroc, effect_size, viz_channel, viz_channels, ...]
        out_csv: Destination path for ko_shap_captions*.csv
        include_counts: When True, append "~XX% more|fewer {structure}
                 objects ***" tails to each channel section using the
                 cohort cache. Requires `cache_phase` and `cache_fluor`.
                 Default False — count info flows through the synth's
                 `count_votes` path only when `_count` features land in
                 top SHAP rows.
        cache_phase / cache_fluor: SHAP cache directories
                 (`X.npy` / `obs.parquet` / `features.txt`). Required
                 when `include_counts=True`; ignored otherwise.
        contrast: One of "distinct" (default — original ko_shap atlas),
                 "ntc" (filter rows to contrast=='ntc' from
                 ntc_shap_features.csv; reframe header as "KO vs NTC"),
                 or "global" (filter to contrast=='global'; reframe
                 "KO vs cohort").

    Returns:
        DataFrame written to out_csv.
    """
    if not in_csv.exists():
        raise FileNotFoundError(f"Input not found: {in_csv}. Run ko_shap_features_combined.py first.")

    if contrast not in CONTRAST_HEADER_LABEL:
        raise ValueError(
            f"Unknown contrast={contrast!r}; expected one of "
            f"{tuple(CONTRAST_HEADER_LABEL)}."
        )
    header_label = CONTRAST_HEADER_LABEL[contrast]

    df = pd.read_csv(in_csv)
    if contrast in ("ntc", "global"):
        if "contrast" not in df.columns:
            # Top-attention CSVs (ko_shap_features.py) don't carry a
            # `contrast` column — each contrast writes to its own
            # out_dir, so the file IS the contrast filter. Only the
            # all-cells CSV (ntc_shap_features.py) multiplexes
            # contrasts in one file via the `contrast` column.
            print(f"  contrast={contrast}: no `contrast` column in "
                  f"{Path(in_csv).name} (top-attention CSV); using all rows.",
                  flush=True)
        else:
            before = len(df)
            df = df[df["contrast"].astype(str) == contrast].copy()
            print(f"  contrast={contrast}: filtered {len(df):,}/{before:,} rows",
                  flush=True)
            if not len(df):
                raise ValueError(
                    f"{in_csv}: no rows with contrast='{contrast}'."
                )
    auroc_col = "auroc" if "auroc" in df.columns else "auroc_vs_ko"
    n_col     = "n_pos_cells" if "n_pos_cells" in df.columns else "n_ko_cells"
    print(f"Loaded {df['gene'].nunique()} genes, modalities: {df['modality'].unique().tolist()}")

    # Build count-cache indices once if --include-counts requested.
    phase_idx = None
    fluor_idx = None
    if include_counts:
        if cache_phase is None or cache_fluor is None:
            raise ValueError(
                "include_counts=True requires both cache_phase and cache_fluor"
            )
        print(f"  --include-counts: loading phase cache from {cache_phase}", flush=True)
        phase_idx = _CountIndex(cache_phase, "phase")
        print(f"  --include-counts: loading fluor cache from {cache_fluor}", flush=True)
        fluor_idx = _CountIndex(cache_fluor, "fluor")
        # Load any previously-computed (gene, feature, channel_rank) →
        # (pct, z) results from a sidecar pickle next to out_csv. Reruns
        # that don't rebuild the SHAP cache (e.g. tweaking caption text)
        # short-circuit every delta() call from disk.
        sidecar = Path(out_csv).parent
        n_phase = phase_idx.load_disk_cache(sidecar)
        n_fluor = fluor_idx.load_disk_cache(sidecar)
        if n_phase or n_fluor:
            print(
                f"  --include-counts: loaded count cache from {sidecar} "
                f"(phase={n_phase}, fluor={n_fluor})",
                flush=True,
            )
        else:
            print(
                f"  --include-counts: no usable count cache at {sidecar} "
                "(missing or stale) — recomputing from SHAP cache",
                flush=True,
            )

    rows = []
    for gene, gene_df in df.groupby("gene", observed=True):
        gene = str(gene)
        gene_df = gene_df.sort_values(["modality", "channel_rank", "shap_rank"])

        phase_df: pd.DataFrame = gene_df[gene_df["modality"] == "phase"].copy()  # type: ignore[assignment]
        fluor_df: pd.DataFrame = gene_df[gene_df["modality"] == "fluor"].copy()  # type: ignore[assignment]

        phase_auroc = float(phase_df[auroc_col].iat[0]) if len(phase_df) else 0.0
        phase_n     = int(phase_df[n_col].iat[0]) if len(phase_df) else 0
        viz_channels_str = str(gene_df["viz_channels"].iat[0]) if "viz_channels" in gene_df.columns else ""

        phase_phrases = _build_phrases(phase_df, viz_channels_str) if len(phase_df) else []

        fluor_sections: list[tuple[str, float, list[tuple]]] = []
        fluor_channel_ranks: list[int] = []
        for ch_rank, ch_df in fluor_df.groupby("channel_rank", observed=True):
            ch_df = ch_df.sort_values("shap_rank")  # type: ignore[assignment]
            ch_name  = str(ch_df["viz_channel"].iat[0]) if "viz_channel" in ch_df.columns else f"ch{ch_rank}"
            ch_auroc = float(ch_df[auroc_col].iat[0])
            ch_phrases = _build_phrases(ch_df, ch_name)
            fluor_sections.append((ch_name, ch_auroc, ch_phrases))
            fluor_channel_ranks.append(int(ch_rank))

        # When --include-counts is set, build per-section count-tail
        # closures that read from the SHAP cache. Phase walks
        # _DISPLAY_TO_COUNT_FEATURES and picks the largest |delta|; fluor
        # uses the channel-rank-keyed `op_count` feature.
        _phase_lookup = None
        _fluor_lookup = None
        if include_counts and phase_idx is not None:
            def _phase_lookup(org_key, org_long, _gene=gene):  # noqa: F811
                candidates = _DISPLAY_TO_COUNT_FEATURES.get(org_key, [])
                if not candidates:
                    return None
                best_result = None
                for feat in candidates:
                    result = phase_idx.delta(_gene, feat)
                    if result is None:
                        continue
                    if best_result is None or abs(result[0]) > abs(best_result[0]):
                        best_result = result
                if best_result is None:
                    return None
                structure = re.sub(r"\s*\(.*\)$", "", org_long).strip() or org_key
                tail = _format_count_tail(best_result, structure)
                return (tail, structure) if tail else None
        if include_counts and fluor_idx is not None:
            def _fluor_lookup(ch_rank, _gene=gene,                   # noqa: F811
                               _ranks_list=fluor_channel_ranks,
                               _sections=fluor_sections):
                try:
                    pos = _ranks_list.index(ch_rank)
                    ch_name = _sections[pos][0]
                    _key, structure = _resolve_fluor(ch_name)
                    structure = re.sub(r"\s*\(.*\)$", "", structure).strip() or ch_name
                except (ValueError, IndexError):
                    structure = "this organelle"
                result = fluor_idx.delta(
                    _gene, FLUOR_COUNT_FEATURE, channel_rank=ch_rank,
                )
                tail = _format_count_tail(result, structure)
                return (tail, structure) if tail else None

        caption = _make_caption(
            gene, phase_phrases, fluor_sections, phase_auroc,
            phase_count_lookup=_phase_lookup,
            fluor_count_lookup=_fluor_lookup,
            fluor_channel_ranks=fluor_channel_ranks,
            header_label=header_label,
        )

        # Arrow uses effect_size sign (feature-value space) when available
        # — same convention as the synthesized caption text and the atlas
        # violin colors. Falls back to SHAP `direction` if effect_size is
        # missing or exactly 0.
        def _arrow(row) -> str:
            es = row.get("effect_size")
            if pd.notna(es) and float(es) != 0:
                return "↑" if float(es) > 0 else "↓"
            return "↑" if float(row["direction"]) > 0 else "↓"

        phase_feats = "; ".join(
            f"{str(r['feature']).removeprefix('op_')}:{_arrow(r)}"
            for _, r in phase_df.head(5).iterrows()
        )
        fluor_feats = "; ".join(
            f"ch{int(r['channel_rank'])}/{str(r['feature']).removeprefix('op_')}:{_arrow(r)}"
            for _, r in fluor_df.head(9).iterrows()
        )
        best_fluor_auroc = max((a for _, a, _ in fluor_sections), default=0.0)
        rows.append({
            "gene":             gene,
            "phase_n_cells":    phase_n,
            "phase_auroc":      round(phase_auroc, 3),
            "best_fluor_auroc": round(best_fluor_auroc, 3),
            "phase_confidence": _confidence(phase_auroc),
            "viz_channels":     viz_channels_str,
            "caption":          caption,
            "phase_top5":       phase_feats,
            "fluor_top9":       fluor_feats,
        })

    out = (
        pd.DataFrame(rows)
        .assign(_best=lambda d: d[["phase_auroc", "best_fluor_auroc"]].max(axis=1))
        .sort_values("_best", ascending=False)
        .drop(columns="_best")
    )
    out.to_csv(out_csv, index=False)
    # Persist the count cache so reruns short-circuit the per-gene
    # SHAP-cache reads. Sidecar lives next to the captions CSV; keyed on
    # the underlying X.npy/obs.parquet mtime so it auto-invalidates if
    # the SHAP cache is rebuilt.
    if include_counts:
        sidecar = Path(out_csv).parent
        if phase_idx is not None:
            phase_idx.save_disk_cache(sidecar)
        if fluor_idx is not None:
            fluor_idx.save_disk_cache(sidecar)
        print(
            f"Saved count cache → {sidecar} "
            f"(.count_cache_phase.pkl, .count_cache_fluor.pkl)",
            flush=True,
        )
    print(f"Saved: {out_csv}  ({len(out)} genes)")
    print()
    print(f"{'GENE':<12} {'PHASE':>6} {'FLUOR':>6}  CAPTION")
    print("-" * 120)
    for _, r in out.iterrows():
        print(f"{r['gene']:<12} {r['phase_auroc']:>6.3f} {r['best_fluor_auroc']:>6.3f}  {r['caption'][:100]}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", default="reports/ko_shap_features_targeted.csv")
    parser.add_argument("--captions", default="reports/ko_shap_captions_targeted.csv")
    parser.add_argument(
        "--include-counts", action="store_true",
        help="Append '~XX% more|fewer {structure} objects ***' tails to "
             "each channel section using the SHAP cache (requires "
             "--cache-phase and --cache-fluor). Disabled by default since "
             "abundance info now flows via the synth's count_votes path; "
             "enable to recover canonical-phenotype sensitivity for KOs "
             "whose count signal matters (mito mass, nucleolar collapse, "
             "lipid droplet accumulation, ERAD upregulation, etc.).",
    )
    parser.add_argument(
        "--cache-phase", default=None,
        help="Phase SHAP cache dir (X.npy/obs.parquet/...). Required with "
             "--include-counts.",
    )
    parser.add_argument(
        "--cache-fluor", default=None,
        help="Fluor SHAP cache dir. Required with --include-counts.",
    )
    parser.add_argument(
        "--contrast", choices=("distinct", "ntc", "global"), default="distinct",
        help="Atlas variant. `distinct` (default): legacy ko_shap atlas, "
             "header `{gene} geneKO`. `ntc`: filter to contrast=='ntc' rows "
             "of an ntc_shap_features.csv, header `{gene} KO vs NTC`. "
             "`global`: filter to contrast=='global', header `{gene} KO vs cohort`.",
    )
    args = parser.parse_args()
    generate_captions(
        Path(args.features), Path(args.captions),
        include_counts=args.include_counts,
        cache_phase=Path(args.cache_phase) if args.cache_phase else None,
        cache_fluor=Path(args.cache_fluor) if args.cache_fluor else None,
        contrast=args.contrast,
    )


if __name__ == "__main__":
    main()
