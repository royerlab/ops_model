"""Real-cell morphometrics for the EBI-complex predictive-cell panels (figure_multirank_ebi_grid) — NTC vs
the SAME 3 gene KOs, on the phenotype-relevant organelle feature.

Nothing is re-measured: the per-cell organelle features already exist in the op_cp_features stores
(op_cp_features_<store>.h5ad, one row per real cell, op_<organelle>_<feature> columns), so this just
gathers them for the panel's genes and plots the distributions:

  A  EMC     · BODIPY lipid droplets   -> op_gfp_count / area          (droplet number, size)
  B  EMC     · phase LIGHT vesicles    -> op_phase2d_vesicular_*       (count, area)
  C  Dynein-1· ER/Golgi COPE           -> op_gfp_normalized_radial_position_mean / distance_from_nucleus
                                          / area_sum                  (localization, then size)
  D  U7 snRNP· phase DARK vesicles     -> op_phase2d_vesicular_dark_*  (count, area)

Cells per group: the top-1000 SHAP-RANKED cells of that gene (the multi_rank screen's own rank order,
walked until 1000 are matched into the store by experiment/well/segmentation — same matching as
morpho_pipeline.real_percell), with NTC restricted to the experiments its KO cells come from so batch is
matched. The cells displayed in the image panel are ranks 1-3, marked as dots.

Run: python figure_ebi_morpho_violin.py            # extract (cached) + plot
     python figure_ebi_morpho_violin.py --refresh   # re-read the stores
"""
import os
import re
import sys

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from figure_multirank_ebi_grid import (BGX, CACHE, COMBINED_ORDER, FOOT, GAP, OUT, SUP, T, TITLE, block_h,
                                      block_w, build_blocks, draw_block, ebi_rows, top_rows, windows)

# paper-v2 stores first; the v2 dir is fluor-only, so the phase store still comes from the v1 dir (loud).
OPCP_DIRS = ["/hpc/projects/icd.fast.ops/analysis/op_cp_features_paper_v2",
             "/hpc/projects/icd.fast.ops/analysis/op_cp_features"]
VOUT = f"{OUT}/morpho"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["svg.fonttype"] = "none"

N_TOP = 1000         # cells per group = top-N SHAP-ranked cells matched into the store
BGY_C = 0.2          # vertical gap between composite rows
CAPTION_C = ("Images: top-SHAP cells (badge = rank). Violins: organelle morphometrics (op_cp_features store) of the "
             "top-1000 SHAP-ranked real cells per class (n≈1000 each); bar = median, % = Δmedian vs NTC.")
VIEW_PCT = (2, 96)   # y-view clip — tails cropped from the VIEW only, medians/KDE stay on the full data
C_NTC, C_KO = "#8a8a8a", "#c1272d"

# per panel-block (keyed on the grid module's stem): store, organelle prefix, candidate features
SPECS = {
    "fluor_EMC_BODIPY": dict(store="lipid_droplet_bodipy_live_cell_dye", org="gfp", primary="area_sum", feats=[
        ("area_sum", "Total lipid droplet area (px²)"), ("count", "Lipid droplet count"),
        ("area_mean", "Mean droplet area (px²)")]),
    "phase_EMC": dict(store="phase", org="phase2d_vesicular", primary="count", feats=[
        ("count", "Light vesicle count"), ("area_mean", "Mean vesicle area (px²)"),
        ("area_sum", "Total vesicle area (px²)")]),
    # dynein KO expands + disperses the ER/Golgi: total COPE area is the strongest readout (+53-66%),
    # ~5x the two localization metrics (distance-from-nucleus / radial position), which are kept as variants
    "fluor_Dynein1_COPE": dict(store="er_golgi_cope", org="gfp", primary="area_sum", feats=[
        ("area_sum", "Total ER/Golgi (COPE) area (px²)"),
        ("distance_from_nucleus_mean", "COPE distance from nucleus (px)"),
        ("normalized_radial_position_mean", "COPE radial position\n(0 = nucleus, 1 = cell edge)")]),
    "phase_U7_snRNP": dict(store="phase", org="phase2d_vesicular_dark", primary="count", feats=[
        ("count", "Dark vesicle count"), ("area_mean", "Mean dark vesicle area (px²)"),
        ("area_sum", "Total dark vesicle area (px²)")]),
}


def store_path(store):
    for i, d in enumerate(OPCP_DIRS):
        p = f"{d}/op_cp_features_{store}.h5ad"
        if os.path.exists(p):
            if i:
                print(f"  [store] {store}: absent from op_cp_features_paper_v2 — falling back to {d}")
            return p
    raise FileNotFoundError(f"no op_cp_features store for {store!r} in {OPCP_DIRS}")


def _obs(h, key, idx=None):
    """One obs column as an array (categorical / nullable-integer / plain dataset), optionally row-subset."""
    o = h[f"obs/{key}"]
    take = (lambda d: d[:]) if idx is None else (lambda d: d[idx])
    if isinstance(o, h5py.Group):
        if "categories" in o:                                            # categorical
            cats = np.array([c.decode() if isinstance(c, bytes) else str(c) for c in o["categories"][:]])
            return cats[take(o["codes"])]
        return take(o["values"])                                         # nullable-integer
    v = take(o)
    return np.array([x.decode() if isinstance(x, bytes) else x for x in v]) if v.dtype.kind in "SO" else v


def _nw(w):
    """Store well form: 'A2' -> 'A/2/0' (matches morpho_pipeline.real_percell)."""
    w = str(w).strip()
    if w.count("/") == 2:
        return w
    m = re.match(r"^([A-Za-z]+)(\d+)$", w)
    return f"{m.group(1)}/{m.group(2)}/0" if m else w


def _cat(h, key):
    """(categories, codes) of an obs categorical — codes stay integer, so no 56M-string materialization."""
    o = h[f"obs/{key}"]
    cats = np.array([c.decode() if isinstance(c, bytes) else str(c) for c in o["categories"][:]])
    return cats, o["codes"][:]


def _numkey(seg, ecode, wcode):
    """(segmentation, experiment, well) packed into one int64 so cells can be matched vectorized."""
    return np.asarray(seg, np.int64) * 10_000 + np.asarray(ecode, np.int64) * 100 + np.asarray(wcode, np.int64)


def _screen_numkeys(df, e2c, w2c):
    """Screen rows (already rank-ordered) → int64 store keys, dropping rows whose exp/well isn't in the store."""
    e = df["experiment"].astype(str).map(e2c)
    w = df["well"].astype(str).map(lambda x: w2c.get(_nw(x)))
    ok = e.notna() & w.notna()
    return _numkey(df.loc[ok, "segmentation_id"].astype("int64"), e[ok], w[ok])


def _match_ranked(store_keys, rows, screen_keys, n):
    """Walk screen_keys in rank order, keep the store rows they hit, stop at n. Returns (rows, n_hit)."""
    o = np.argsort(store_keys, kind="stable")
    sk = store_keys[o]
    pos = np.clip(np.searchsorted(sk, screen_keys), 0, max(len(sk) - 1, 0))
    hit = (sk[pos] == screen_keys) if len(sk) else np.zeros(len(screen_keys), bool)
    matched = rows[o[pos[hit]]]
    _, first = np.unique(matched, return_index=True)              # a store cell can be hit twice; keep rank order
    matched = matched[np.sort(first)]
    return matched[:n], int(hit.sum())


def extract(blk, n_top=N_TOP):
    """Tidy per-cell table (gene, feature, value, shown) for one block: the top-n_top SHAP-ranked cells of
    each gene KO + of NTC (NTC restricted to its KO cells' experiments), from the op_cp_features store."""
    b, genes = blk["b"], blk["genes"]
    spec = SPECS[b["stem"]]
    cols = [f"op_{spec['org']}_{f}" for f, _ in spec["feats"]]
    screen = ebi_rows(b["modality"])
    with h5py.File(store_path(spec["store"]), "r") as h:
        if isinstance(h["X"], h5py.Group):
            raise TypeError(f"{spec['store']}: sparse X not supported by this reader")
        var = [v.decode() for v in h["var/_index"][:]]
        missing = [c for c in cols if c not in var]
        if missing:
            raise KeyError(f"{spec['store']}: missing feature columns {missing}")
        cidx = [var.index(c) for c in cols]
        gcats, gcodes = _cat(h, "gene_name")
        ecats, ecodes = _cat(h, "experiment")
        wcats, wcodes = _cat(h, "well")
        if len(ecats) > 99 or len(wcats) > 99:
            raise ValueError(f"{spec['store']}: {len(ecats)} exps / {len(wcats)} wells — key packing overflows")
        e2c = {e: i for i, e in enumerate(ecats)}
        w2c = {_nw(w): i for i, w in enumerate(wcats)}
        allkeys = _numkey(_obs(h, "segmentation"), ecodes, wcodes)
        gcode = {g: int(np.where(gcats == g)[0][0]) for g in genes + [""]}

        def panel_keys(recs):                       # the cells actually drawn in the image panel
            e = recs["experiment"].astype(str).map(e2c)
            w = recs["well"].astype(str).map(lambda x: w2c.get(_nw(x)))
            ok = e.notna() & w.notna()
            return set(_numkey(recs.loc[ok, "segmentation"].astype("int64"), e[ok], w[ok]).tolist())

        pick, shown = {}, {}
        for i, g in enumerate(genes):
            gi = np.where(gcodes == gcode[g])[0]
            sk = _screen_numkeys(top_rows(screen, g, b["ch_name"], 10 * n_top), e2c, w2c)
            pick[g], nhit = _match_ranked(allkeys[gi], gi, sk, n_top)
            pk = panel_keys(blk["ko"][i][1])
            shown[g] = {r for r in pick[g] if allkeys[r] in pk}
            if len(pick[g]) < n_top:
                print(f"  [{b['stem']}] {g}: only {len(pick[g])} of the top-{n_top} ranked cells are in the store "
                      f"({nhit} matched overall)")
            if len(shown[g]) < len(pk):
                print(f"  [{b['stem']}] {g}: {len(shown[g])}/{len(pk)} panel cells in the store")
        koe = set(ecodes[np.concatenate([pick[g] for g in genes])].tolist())
        ni = np.where((gcodes == gcode[""]) & np.isin(ecodes, list(koe)))[0]        # NTC, KO experiments only
        nsk = _screen_numkeys(top_rows(screen, "NTC", b["ch_name"], 20 * n_top), e2c, w2c)
        pick["NTC"], nhit = _match_ranked(allkeys[ni], ni, nsk, n_top)
        npk = panel_keys(blk["ntc"][1])
        shown["NTC"] = {r for r in pick["NTC"] if allkeys[r] in npk}
        print(f"  [{b['stem']}] NTC {len(pick['NTC'])} ranked cells from {len(koe)} KO experiments", flush=True)

        rows = np.unique(np.concatenate([pick[k] for k in ["NTC"] + genes]))   # h5py needs strictly increasing
        X = h["X"][rows, :][:, cidx].astype(float)
    r2p = {r: i for i, r in enumerate(rows)}
    recs = [(grp, f, float(X[r2p[r]][j]), r in shown[grp])
            for grp in ["NTC"] + genes for r in pick[grp] for j, (f, _) in enumerate(spec["feats"])]
    print(f"  [{b['stem']}] {len(rows)} cells x {len(cols)} features", flush=True)
    return pd.DataFrame(recs, columns=["gene", "feature", "value", "shown"])


def table(blocks, refresh=False):
    """Cached per-cell tables for every block: {stem: DataFrame}."""
    out = {}
    for blk in blocks:
        p = f"{CACHE}/morpho_{blk['b']['stem']}.parquet"
        if os.path.exists(p) and not refresh:
            out[blk["b"]["stem"]] = pd.read_parquet(p)
            continue
        df = extract(blk)
        df.to_parquet(p)
        out[blk["b"]["stem"]] = df
    return out


def draw_violin(ax, df, genes, feat, ylab, title=None, fs=15, show_n=True, xrot=0):
    """NTC + one violin per gene KO, medians as bars, %Δ median vs NTC (annotated at the top of the axes)."""
    groups = ["NTC"] + genes
    data = [df.loc[(df["gene"] == g) & (df["feature"] == feat), "value"].dropna().values for g in groups]
    parts = ax.violinplot(data, positions=range(len(groups)), widths=0.82,
                          showmeans=False, showextrema=False, showmedians=False)
    for i, pc in enumerate(parts["bodies"]):
        c = C_NTC if i == 0 else C_KO
        pc.set_facecolor(c); pc.set_edgecolor(c); pc.set_alpha(0.55); pc.set_linewidth(1.5)
    nmed = float(np.median(data[0])) if len(data[0]) else np.nan
    for i, d in enumerate(data):
        if not len(d):
            continue
        ax.hlines(np.median(d), i - 0.34, i + 0.34, color="#222", lw=2.5, zorder=5)
        if i:
            pct = (np.median(d) - nmed) / (abs(nmed) or 1e-9) * 100
            ax.text(i, 0.995 if i % 2 else 0.885, f"{pct:+.0f}%", transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=fs - 4, fontweight="bold", color=C_KO)  # staggered: narrow axes
    pooled = np.concatenate([d for d in data if len(d)])
    ylo, yhi = np.percentile(pooled, VIEW_PCT)
    pad = 0.08 * (yhi - ylo + 1e-9)
    ax.set_ylim(ylo - pad, yhi + pad * 3.4)                       # headroom for the staggered %Δ rows
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    ax.set_xlim(-0.62, len(groups) - 0.38)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([f"{g}\nn={len(d):,}" if show_n else g for g, d in zip(groups, data)],
                       fontsize=fs, rotation=xrot, ha="right" if xrot else "center")
    for t, c in zip(ax.get_xticklabels(), [C_NTC] + [C_KO] * len(genes)):
        t.set_color(c)
        t.set_fontweight("bold")
    ax.set_ylabel(ylab, fontsize=fs + 1)
    ax.tick_params(axis="y", labelsize=fs - 1, width=2, length=7)
    ax.tick_params(axis="x", length=0)
    if title:
        ax.set_title(title, fontsize=fs + 3, fontweight="bold", pad=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_linewidth(2)


def save(fig, stem, outdir=VOUT):
    os.makedirs(outdir, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(f"{outdir}/{stem}.{ext}", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {outdir}/{stem}.png/.svg", flush=True)


# composite geometry (inches): violin axes width, gap after the image block, y-label gutter, and the
# x-label band — which is taken INSIDE the image block's height (violin is shorter, top-aligned) so the
# rotated gene labels don't push the next panel row down.
VW, VGAP, VLAB, VXLAB, VBOT = 2.45, 0.12, 1.18, 1.0, 0.12
FS_V = 22


def composite(blocks, tabs, ncol=2, x0=0.45):
    """Image block + its violin side by side, 2 blocks per row, in the panel order of the image figure."""
    win = windows(blocks)
    order = [blocks[i] for i in COMBINED_ORDER]
    rows = [order[i:i + ncol] for i in range(0, len(order), ncol)]
    tot_w = lambda blk: block_w(blk["genes"]) + VGAP + VLAB + VW
    colw = [max(tot_w(r[c]) for r in rows if c < len(r)) for c in range(ncol)]
    rowh = [max(block_h(len(blk["genes"])) for blk in r) + VBOT for r in rows]
    W = x0 + sum(colw) + (ncol - 1) * BGX + 0.12
    H = SUP + sum(rowh) + BGY_C * (len(rows) - 1) + FOOT
    fig = plt.figure(figsize=(W, H), facecolor="white")
    y = SUP
    for r, row in enumerate(rows):
        for c, blk in enumerate(row):
            b, genes = blk["b"], blk["genes"]
            spec = SPECS[b["stem"]]
            x = x0 + sum(colw[:c]) + c * BGX
            draw_block(fig, blk, x, y, *win[b["ch_name"]], W, H, letter="ABCDEFGH"[r * ncol + c])
            th = len(genes) * T + (len(genes) - 1) * GAP - VXLAB          # room for the rotated x labels
            ax = fig.add_axes([(x + block_w(genes) + VGAP + VLAB) / W, 1 - (y + TITLE + th) / H, VW / W, th / H])
            draw_violin(ax, tabs[b["stem"]], genes, spec["primary"], dict(spec["feats"])[spec["primary"]],
                        fs=FS_V, show_n=False, xrot=45)
        y += rowh[r] + BGY_C
    fig.text(0.5, 1 - 0.3 / H, "Top Predictive cells per protein complex", fontsize=26, fontweight="bold",
             ha="center", va="center")
    fig.text(0.5, (FOOT - 0.3) / H, CAPTION_C, fontsize=13, style="italic", color="#333", ha="center", va="center")
    save(fig, "ebi_composite_grid_violin", OUT)


def main(refresh=False):
    blocks = build_blocks()
    tabs = table(blocks, refresh)
    by_stem = {blk["b"]["stem"]: blk for blk in blocks}

    for blk in blocks:                                          # every candidate feature, to pick from
        stem, spec = blk["b"]["stem"], SPECS[blk["b"]["stem"]]
        for feat, ylab in spec["feats"]:
            fig, ax = plt.subplots(figsize=(1.35 * (len(blk["genes"]) + 1) + 1.6, 4.6), facecolor="white")
            draw_violin(ax, tabs[stem], blk["genes"], feat, ylab,
                        title=f"{blk['b']['label']} — {blk['b']['marker_label']}")
            save(fig, f"violin_{stem}_{feat}")

    order = [blocks[i] for i in COMBINED_ORDER]                 # combined: primary feature, panel order A-D
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.4), facecolor="white")
    for ax, blk, letter in zip(axes.flat, order, "ABCD"):
        stem = blk["b"]["stem"]
        spec = SPECS[stem]
        ylab = dict(spec["feats"])[spec["primary"]]
        draw_violin(ax, tabs[stem], blk["genes"], spec["primary"], ylab,
                    title=f"{blk['b']['label']} — {blk['b']['marker_label']}")
        ax.text(-0.16, 1.1, letter, transform=ax.transAxes, fontsize=24, fontweight="bold", va="top")
    fig.suptitle("Real-cell morphometrics of the predictive phenotypes — NTC vs complex gene KOs",
                 fontsize=20, fontweight="bold", y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.955), h_pad=3.0, w_pad=3.0)
    save(fig, "violin_combined_primary")
    composite(blocks, tabs)
    for stem, blk in by_stem.items():                           # Δmedian for every candidate feature
        d = tabs[stem]
        for feat, _ in SPECS[stem]["feats"]:
            med = lambda g: np.nanmedian(d.loc[(d["gene"] == g) & (d["feature"] == feat), "value"])
            nmed = med("NTC")
            deltas = ", ".join(f"{g} {(med(g) - nmed) / (abs(nmed) or 1e-9) * 100:+.0f}%" for g in blk["genes"])
            star = " *primary" if feat == SPECS[stem]["primary"] else ""
            print(f"{stem:22s} {feat:32s} NTC {nmed:9.3g} | {deltas}{star}")


if __name__ == "__main__":
    main(refresh="--refresh" in sys.argv)
