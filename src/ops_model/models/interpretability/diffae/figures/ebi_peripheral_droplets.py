"""ISOLATED PROBE — peripheral lipid-droplet counts for the EMC/BODIPY panel (panel A of
figure_multirank_ebi_grid / figure_ebi_morpho_violin).

Deliberately standalone: the op_cp_features stores only hold per-CELL aggregates of the localization
features (distance_from_cell_edge mean/min/max/...), so a "droplets within X µm of the cell boundary"
count has to be measured per object. This script re-measures it for the panel's top-SHAP cells straight
from phenotyping_v3.zarr (gfp_seg droplets + cell_seg + nuclear_seg, same stitched coords the image panel
crops from) using organelle_profiler's own localization code, so `distance_from_cell_edge` here means
exactly what the store's column means.

Nothing in the working figure scripts is modified — it only imports read-only helpers from them, writes
its own cache + its own figure, and can be deleted to go back to the simple droplet count.

Definitions per cell (droplets = gfp_seg objects inside the cell mask). "Peripheral" = the droplet's
centroid is within d µm of the cell boundary, or at normalized_radial_position >= t (0 = nucleus,
1 = cell edge). Counts and areas both, since abundance and size move independently:
  count / area_sum          : all droplets (cross-check vs the store's op_gfp_count / op_gfp_area_sum)
  peri_edge_<d>um           : # peripheral droplets            frac_edge_<d>um  : / total count
  peri_shell_<t>            : # peripheral droplets            frac_shell_<t>   : / total count
  periarea_edge_<d>um (µm²) : peripheral droplet area          fracarea_edge_<d>um : / total area
  periarea_shell_<t>  (µm²) : peripheral droplet area          fracarea_shell_<t>  : / total area

Cells: the top-1000 SHAP-ranked cells per class from the EBI multi_rank screen (rank order, no store
matching needed — the zarr is the source here). Cells whose mask is missing or clipped by the crop window
are skipped and counted, never silently dropped.

Run: python ebi_peripheral_droplets.py            # measure (cached) + plot
     python ebi_peripheral_droplets.py --refresh   # re-measure from the zarrs
     python ebi_peripheral_droplets.py --limit 50  # quick smoke test (50 cells/class)
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import zarr

from figure_ebi_morpho_violin import draw_violin
from figure_multirank_ebi_grid import CACHE, OUT, ebi_rows, top_rows
from ops_model.models.interpretability.diffae.traversal.build_pc_crops_masked import BASE, _crop, _zarr_patch

from organelle_profiler.feature_extraction.localization_features import compute_localization_features

CH_NAME = "lipid droplet_BODIPY live cell dye"     # multi_rank channel_name (panel A)
GENES = ["EMC1", "EMC2", "EMC3"]                   # same rows as the image panel
ORG_LABEL = "gfp_seg"                              # BODIPY droplets in phenotyping_v3.zarr
N_TOP = 1000
PX_UM = 0.325                 # NATIVE phenotype pixel size; the zarr's declared 0.65 is the known bug
HALF = 192                    # crop half-window (px) — 125 µm box, comfortably larger than a HeLa cell
EDGE_UM = (1.0, 2.0, 3.0)     # "within d µm of the cell boundary"
SHELL = (0.6, 0.8)            # normalized nucleus→edge position threshold
POUT = f"{OUT}/peripheral"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]   # Arial first: no Illustrator substitution

FEATS = ([("count", "Lipid droplet count"), ("area_sum", "Total lipid droplet area (µm²)")]
         + [(f"peri_edge_{d:g}um", f"Peripheral droplets\n(≤{d:g} µm from cell edge)") for d in EDGE_UM]
         + [(f"frac_edge_{d:g}um", f"Peripheral droplet fraction\n(≤{d:g} µm from cell edge)") for d in EDGE_UM]
         + [(f"periarea_edge_{d:g}um", f"Peripheral droplet area (µm²)\n(≤{d:g} µm from cell edge)") for d in EDGE_UM]
         + [(f"fracarea_edge_{d:g}um", f"Peripheral droplet area fraction\n(≤{d:g} µm from cell edge)") for d in EDGE_UM]
         + [(f"peri_shell_{t:g}", f"Peripheral droplets\n(radial position ≥ {t:g})") for t in SHELL]
         + [(f"frac_shell_{t:g}", f"Peripheral droplet fraction\n(radial position ≥ {t:g})") for t in SHELL]
         + [(f"periarea_shell_{t:g}", f"Peripheral droplet area (µm²)\n(radial position ≥ {t:g})") for t in SHELL]
         + [(f"fracarea_shell_{t:g}", f"Peripheral droplet area fraction\n(radial position ≥ {t:g})") for t in SHELL])


def _pos(exp, well):
    """phenotyping_v3.zarr position group for one (experiment, well)."""
    w = str(well)
    return f"{BASE}/{exp}/3-assembly/phenotyping_v3.zarr/{w[0]}/{w[1:]}/0"


class Stores:
    """cell_seg / organelle / nuclear_seg label arrays per (experiment, well), opened once."""

    def __init__(self):
        self.cache = {}

    def get(self, exp, well):
        k = (exp, str(well))
        if k not in self.cache:
            p = _pos(exp, well)
            try:
                self.cache[k] = tuple(zarr.open(f"{p}/labels/{n}/0", mode="r")
                                      for n in ("cell_seg", ORG_LABEL, "nuclear_seg"))
            except Exception as e:                      # noqa: BLE001 - report, keep going
                print(f"  [store] {exp}/{well}: {type(e).__name__}: {e}")
                self.cache[k] = None
        return self.cache[k]


def cell_measure(stores, r):
    """Per-object localization for one cell → dict of peripheral counts/fractions, or a skip reason."""
    z = stores.get(r["experiment"], r["well"])
    if z is None:
        return None, "no_store"
    cz, oz, nz = z
    x, y, sid = int(round(r["x_pheno"])), int(round(r["y_pheno"])), int(r["segmentation_id"])
    cell = _crop(cz, None, x, y, HALF) == sid
    if not cell.any():
        return None, "cell_label_absent"
    if cell[0, :].any() or cell[-1, :].any() or cell[:, 0].any() or cell[:, -1].any():
        return None, "cell_clipped"                     # edge distances would be wrong — never guess
    org = np.where(cell, _crop(oz, None, x, y, HALF), 0)
    if not org.any():
        return dict(count=0), None                      # a real zero-droplet cell
    nuc = cell & (_crop(nz, None, x, y, HALF) > 0)      # nuclear_seg has its own IDs -> intersect spatially
    loc = compute_localization_features(org, cell, nuc if nuc.any() else None, spacing=(PX_UM, PX_UM))
    n = len(loc)
    out = {"count": float(n)}
    if n:
        px = np.bincount(org.ravel())                   # per-object pixel count -> µm² by label id
        area = px[loc["label"].to_numpy(int)] * PX_UM ** 2
        atot = float(area.sum())
        out["area_sum"] = atot

        def _both(mask, ctag, atag):
            c = float(np.sum(mask))
            out[f"peri_{ctag}"] = c
            out[f"frac_{ctag}"] = c / n
            a = float(area[mask].sum())
            out[f"periarea_{atag}"] = a
            out[f"fracarea_{atag}"] = a / atot if atot > 0 else np.nan

        ed = loc["distance_from_cell_edge"].to_numpy(float)
        for d in EDGE_UM:
            _both(ed <= d, f"edge_{d:g}um", f"edge_{d:g}um")
        if "normalized_radial_position" in loc and np.isfinite(loc["normalized_radial_position"]).any():
            rp = np.nan_to_num(loc["normalized_radial_position"].to_numpy(float), nan=-1.0)
            for t in SHELL:
                _both(rp >= t, f"shell_{t:g}", f"shell_{t:g}")
    return out, None


def measure(limit=None):
    """Measure every class's top-N cells. Returns the tidy per-cell table."""
    _zarr_patch()
    screen = ebi_rows("fluor")
    stores = Stores()
    rows, skips = [], {}
    for gene in GENES + ["NTC"]:
        sel = top_rows(screen, gene, CH_NAME, limit or N_TOP)
        for i, (_, r) in enumerate(sel.iterrows()):
            m, why = cell_measure(stores, r)
            if m is None:
                skips[why] = skips.get(why, 0) + 1
                continue
            rows.append(dict(gene=gene, rank=int(r["rank"]), experiment=r["experiment"], well=r["well"],
                             segmentation_id=int(r["segmentation_id"]), **m))
            if (i + 1) % 250 == 0:
                print(f"  [{gene}] {i + 1}/{len(sel)} cells", flush=True)
        n = sum(1 for x in rows if x["gene"] == gene)
        print(f"  [{gene}] measured {n}/{len(sel)} cells", flush=True)
    if skips:
        print(f"  skipped: {skips}  (clipped cells are excluded, not guessed)", flush=True)
    df = pd.DataFrame(rows)
    long = df.melt(id_vars=["gene", "rank", "experiment", "well", "segmentation_id"],
                   var_name="feature", value_name="value").dropna(subset=["value"])
    long["unit"] = "count"                              # counts + fractions: unit-less for the label helper
    return long


def table(refresh=False, limit=None):
    p = f"{CACHE}/peripheral_bodipy_{limit or N_TOP}_f{len(FEATS)}.parquet"   # feature count keys the cache
    if os.path.exists(p) and not refresh:
        return pd.read_parquet(p)
    df = measure(limit)
    os.makedirs(CACHE, exist_ok=True)
    df.to_parquet(p)
    print(f"  cached {p}  ({df['gene'].nunique()} classes)", flush=True)
    return df


def main(refresh=False, limit=None):
    tab = table(refresh, limit)
    os.makedirs(POUT, exist_ok=True)
    for feat, ylab in FEATS:
        if not (tab["feature"] == feat).any():
            print(f"skip {feat}: not measured"); continue
        fig, ax = plt.subplots(figsize=(1.35 * (len(GENES) + 1) + 1.6, 4.8), facecolor="white")
        draw_violin(ax, tab, GENES, feat, ylab, title="EMC complex — lipid droplet (BODIPY)")
        for ext in ("png", "svg"):
            fig.savefig(f"{POUT}/violin_peripheral_{feat}.{ext}", dpi=220, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"saved {POUT}/violin_peripheral_{feat}.png/.svg", flush=True)
    for feat, _ in FEATS:                               # Δmedian table, to compare against the plain count
        d = tab[tab["feature"] == feat]
        if d.empty:
            continue
        med = lambda g: np.nanmedian(d.loc[d["gene"] == g, "value"])
        nmed = med("NTC")
        print(f"{feat:20s} NTC {nmed:8.3g} | "
              + ", ".join(f"{g} {(med(g) - nmed) / (abs(nmed) or 1e-9) * 100:+.0f}%" for g in GENES))


if __name__ == "__main__":
    lim = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
    main(refresh="--refresh" in sys.argv, limit=lim)
