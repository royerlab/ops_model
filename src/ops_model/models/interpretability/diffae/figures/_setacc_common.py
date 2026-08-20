"""Shared machinery for the fig-4 set-accuracy panels: the C/D figure-group registry, on-demand
cropping of a specific rank from the v5 set-accuracy rankings (KO class or NTC), marker-global
normalization, and the inverse-blue-mask composite. Used by figure4_setacc_panel.py (final panel)
and debug_setacc_top100.py (per-group montages to pick cells from).

Cells are picked by their parquet `rank` (the badge shown in the debug montage). Ranks beyond the
cached top-30 are cropped live from phenotyping_v3.zarr via materialize_crops (same path as
viewer/_fluor_topcells)."""
import numpy as np
import pandas as pd
import zarr

from ops_model.models.interpretability.diffae.classifier.config import slugify
from ops_model.models.interpretability.diffae.classifier.data import make_labels_df, materialize_crops
from ops_model.models.interpretability.diffae.directions.config import DirConfig
from ops_model.models.interpretability.diffae.viewer._fluor_topcells import _overlay_rgba
from ops_model.models.interpretability.diffae.viewer.build_pc_crops_masked import BASE, CROP_SIZE, _crop, _zarr_patch

OUT = "/hpc/projects/icd.fast.ops/analysis/figure4_setacc_panel"
RANK_BASE = "/hpc/projects/icd.fast.ops/models/diffex/viewer_assets_v5/_rankings/fluor_shap"

TIM23 = "TIM23 mitochondrial inner membrane pre-sequence translocase complex, TIM17A variant"
COPI = "COPI vesicle coat complex, COPG1-COPZ1 variant"

# Published C/D figure groups. key = class in the ranking parquet (POLR1H stored under alias ZNRD1);
# ko_rank / ntc_rank = the rank badge picked from the debug montage (default 1 = top set-accuracy).
GENE_COLS = [
    dict(slug="Mitochondria_TOMM20", mc="Mitochondria_TOMM20", ch="CP1_mitochondria_TOMM20",
         block="genes", key="TOMM20", top_label="TOMM20", marker_label="Mitochondria\n(TOMM20)",
         ko_rank=1, ntc_rank=18),
    dict(slug="nucleolus_GC_NPM3", mc="nucleolus-GC_NPM3", ch="GFP",
         block="genes", key="ZNRD1", top_label="POLR1H", marker_label="Nucleoli\n(NPM3-GFP)",
         ko_rank=1, ntc_rank=5),
    dict(slug="5xUPRE", mc="5xUPRE", ch="GFP",
         block="genes", key="HSPA5", top_label="HSPA5", marker_label="UPR\n(5xUPRE)",
         ko_rank=100, ntc_rank=3),
    dict(slug="ER_Golgi_COP_II_SEC23A", mc="ER/Golgi COP-II_SEC23A", ch="GFP",
         block="genes", key="GBF1", top_label="GBF1", marker_label="ER-Golgi\n(GFP-SEC23A)",
         ko_rank=7, ntc_rank=2),
]
COMPLEX_COLS = [
    dict(slug="mitochondria_ChromaLIVE_561_excitation", mc="mitochondria_ChromaLIVE 561 excitation",
         ch="mCherry", block="complexes", key=TIM23, top_label="TIM23",
         marker_label="Mitochondria\n(ChromaLIVE 561)", ko_rank=27, ntc_rank=1),
    dict(slug="cell_proliferation_marker_MKI67", mc="cell proliferation marker_MKI67", ch="GFP",
         block="complexes", key="DNA polymerase alpha:primase complex", top_label="DNA Pol α",
         marker_label="Proliferation\n(GFP-MKI67)", ko_rank=2, ntc_rank=2),
    dict(slug="actin_filament_FastAct_SPY555_Live_Cell_Dye", mc="actin filament_FastAct_SPY555 Live Cell Dye",
         ch="mCherry", block="complexes", key="Chaperonin-containing T-complex", top_label="CCT",
         marker_label="Actin\n(FastAct SPY555)", ko_rank=4, ntc_rank=13),
    dict(slug="lysosome_LysoTracker_live_cell_dye", mc="lysosome_LysoTracker live-cell dye", ch="GFP",
         block="complexes", key="mTORC1 complex", top_label="mTORC1", marker_label="Lysosome\n(LysoTracker)",
         ko_rank=18, ntc_rank=2),   # Rab-GGTase slot -> mTORC1 · Lysosome
]


def _rankdir(block):
    return f"{RANK_BASE}/{'geneKO' if block == 'genes' else 'complex'}"


def _materialize(sel, mc, ch, cls):
    """Crop the given (already row-selected) ranking rows. Returns (raw[N,1,H,W], recs realigned)."""
    _zarr_patch()
    recs = sel.rename(columns={"gene": "cls", "pma_attention": "score"}).copy()
    recs["label"] = 0
    cfg = DirConfig(grain="geneKO", target=cls, device="cpu")
    cfg.marker_channel = mc; cfg.channel = ch; cfg.num_workers = 8
    raw, _, exps = materialize_crops(make_labels_df(recs, cfg), cfg, cache_path=None)
    recs = recs[recs["experiment"].isin(set(exps))].reset_index(drop=True)
    n = min(len(raw), len(recs))
    return raw[:n], recs.iloc[:n].reset_index(drop=True)


def _class_df(mc, block, cls):
    df = pd.read_parquet(f"{_rankdir(block)}/{slugify(mc)}.parquet")
    df = df[df["gene"] == cls].sort_values("rank").reset_index(drop=True)
    if df.empty:
        raise ValueError(f"{cls!r} absent from {mc!r} [{block}]")
    return df


def materialize_class(mc, ch, block, cls, top_n):
    """Crop the top-`top_n` cells (by rank order/position) of one class. For the debug montages."""
    return _materialize(_class_df(mc, block, cls).head(top_n), mc, ch, cls)


def crop_pick_from_df(df, pick_rank, mc, ch, cls, win=40):
    """Crop the cell with rank == pick_rank from a rank-sorted class df (the unique cell picked from
    the montage), PLUS a top-`win` positional sample for a stable intensity window. Returns
    (raw, recs, pos). Raises loudly if the picked rank is absent or its store dropped. Channel-agnostic
    (ch=Phase2D for phase, marker channel for fluor) so both panels reuse it."""
    pick = df[df["rank"] == pick_rank]
    if pick.empty:
        raise ValueError(f"rank {pick_rank} not found for {cls!r} (max rank {int(df['rank'].max())})")
    sel = pd.concat([df.head(win), pick]).drop_duplicates(["experiment", "well", "x_pheno", "y_pheno"])
    raw, recs = _materialize(sel, mc, ch, cls)
    w = np.where(recs["rank"].values == pick_rank)[0]
    if not len(w):
        raise RuntimeError(f"rank {pick_rank} cell for {cls!r} dropped by materialize_crops (store missing) — pick another")
    return raw, recs, int(w[0])


def crop_pick(mc, ch, block, cls, pick_rank, win=40):
    """Fluor: crop rank==pick_rank of one class from its per-marker parquet."""
    return crop_pick_from_df(_class_df(mc, block, cls), pick_rank, mc, ch, cls, win)


_SEGCACHE = {}


def seg_crop(exp, well, x, y, half):
    ek = (exp, well)
    if ek not in _SEGCACHE:
        pos = f"{BASE}/{exp}/3-assembly/phenotyping_v3.zarr/{str(well)[0]}/{str(well)[1:]}/0"
        try:
            _SEGCACHE[ek] = zarr.open(f"{pos}/labels/cell_seg/0", mode="r")
        except Exception:
            _SEGCACHE[ek] = None
    z = _SEGCACHE[ek]
    if z is None:
        return None
    try:
        return _crop(z, None, int(round(x)), int(round(y)), half)
    except Exception:
        return None


def composite(gray, seg, half):
    """gray (0-255 float) + inverse blue seg mask → RGB uint8 (blue outside the center cell)."""
    rgb = np.stack([gray] * 3, -1).astype(np.float32)
    if seg is not None:
        ov = _overlay_rgba(seg, half).astype(np.float32)
        a = ov[..., 3:4] / 255.0
        rgb = rgb * (1 - a) + ov[..., :3] * a
    return rgb.clip(0, 255).astype(np.uint8)


def tile_at(raw, recs, pos, lo, hi):
    half = CROP_SIZE // 2
    r = recs.iloc[pos]
    gray = np.clip((raw[pos, 0] - lo) / (hi - lo), 0, 1) * 255
    seg = seg_crop(r["experiment"], r["well"], r["x_pheno"], r["y_pheno"], half)
    return composite(gray, seg, half), r


def column_tiles(col):
    """For one figure-group column dict, crop the chosen KO-rank and NTC-rank cells (selected by
    rank value) with a shared marker-global (KO+NTC) 1-99 pct window. Returns
    (ko_rgb, ntc_rgb, ko_conf, ntc_conf)."""
    ko_raw, ko_recs, ko_pos = crop_pick(col["mc"], col["ch"], col["block"], col["key"], col["ko_rank"])
    ntc_raw, ntc_recs, ntc_pos = crop_pick(col["mc"], col["ch"], "genes", "NTC", col["ntc_rank"])   # NTC always from the marker's geneKO ranking (complex parquet has no NTC)
    lo, hi = np.percentile(np.concatenate([ko_raw.ravel(), ntc_raw.ravel()]), (1, 99))
    if hi - lo < 1e-6:
        hi = lo + 1
    ko_im, ko_r = tile_at(ko_raw, ko_recs, ko_pos, lo, hi)
    ntc_im, ntc_r = tile_at(ntc_raw, ntc_recs, ntc_pos, lo, hi)
    return ko_im, ntc_im, float(ko_r["score"]), float(ntc_r["score"])
