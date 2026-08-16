"""EBI-complex SHAP grids from the multi_rank screen — per complex, a 3-cells x 3-geneKO block of the
top-SHAP KO cells (right) beside the same-shape block of top-SHAP NTC cells (left).

Cells come from Alex's EBI multi_rank screen (shap_screen_ebi_{phase,fluor}_all.csv: per-cell SHAP under
the complex classifier, `gene` = complex member gene). Complex membership is read from the EBI yaml, and
for complexes with >3 members the 3 genes with the strongest top cell are used. U7 snRNP has only 2
members (SNRPD3, SNRPG) -> its block is 2 rows, not 3.

Blocks: U7 snRNP + EMC on Phase2D; EMC on lipid-droplet BODIPY + Dynein-1 on ER/Golgi COPE.
NTC is the top-SHAP NTC of the same channel, so both phase blocks show the identical NTC cells, and
intensity windows are per channel (KO+NTC pooled) so KO vs NTC brightness is comparable.

Run: python figure_multirank_ebi_grid.py          # combined + per-complex figures
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from _setacc_common import CROP_SIZE, _materialize, composite, seg_crop

MR = "/hpc/projects/icd.fast.ops/models/alex_lin_attention/v5/multi_rank"
EBI_YAML = "/hpc/projects/icd.fast.ops/configs/gene_clusters/EBI_complexes_v1_old_gene_names.yaml"
OUT = "/hpc/projects/icd.fast.ops/analysis/figure4_multirank_ebi"
CACHE = f"{OUT}/_cache"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]   # Arial first: no Illustrator substitution

N_CELLS = 3          # cells per gene row
N_GENES = 3          # gene rows per block (fewer if the complex has fewer members)
WIN = 6              # extra ranks materialized per row, so dropped stores don't shrink the row

U7 = "U7 small nuclear ribonucleoprotein complex"
EMC = "Endoplasmic reticulum membrane complex, EMC8 variant"
DYNEIN = "Dynein-1 complex, variant 2"

# ch_name = multi_rank `channel_name`; mc = DirConfig marker_channel (None -> phase); ch = raw zarr channel
BLOCKS = [
    dict(cx=U7, label="U7 snRNP complex", modality="phase", ch_name="Phase2D", mc=None, ch="Phase2D",
         marker_label="label-free phase", stem="phase_U7_snRNP"),
    dict(cx=EMC, label="EMC complex", modality="phase", ch_name="Phase2D", mc=None, ch="Phase2D",
         marker_label="label-free phase", stem="phase_EMC"),
    dict(cx=EMC, label="EMC complex", modality="fluor", ch_name="lipid droplet_BODIPY live cell dye",
         mc="lipid droplet_BODIPY live cell dye", ch="GFP", marker_label="lipid droplet (BODIPY)",
         stem="fluor_EMC_BODIPY"),
    dict(cx=DYNEIN, label="Dynein-1 complex", modality="fluor", ch_name="ER/Golgi_COPE",
         mc="ER/Golgi_COPE", ch="GFP", marker_label="ER/Golgi (COPE)", stem="fluor_Dynein1_COPE"),
]
# combined-figure order (2 per row): the two EMC blocks pair up (fluorescence left of phase), then Dynein-1, U7
COMBINED_ORDER = [2, 1, 3, 0]
COLS = ["gene", "channel_name", "rank", "shap", "experiment", "well", "x_pheno", "y_pheno", "segmentation_id"]


def members(cx):
    y = yaml.safe_load(open(EBI_YAML))
    for v in y.values():
        if v.get("name") == cx:
            return list(v["genes"])
    raise KeyError(f"{cx!r} not in {EBI_YAML}")


def ebi_rows(modality):
    """Cached SHAP rows for every gene/channel the BLOCKS of this modality need (the source CSVs are 2-11GB)."""
    genes = {g for b in BLOCKS if b["modality"] == modality for g in members(b["cx"])} | {"NTC"}
    chans = {b["ch_name"] for b in BLOCKS if b["modality"] == modality}
    p = f"{CACHE}/ebi_{modality}.parquet"
    if os.path.exists(p):
        df = pd.read_parquet(p)
        if genes <= set(df["gene"]) and chans <= set(df["channel_name"]):
            return df
    os.makedirs(CACHE, exist_ok=True)
    csv = f"{MR}/shap_screen_ebi_{modality}_all.csv"
    print(f"[cache] scanning {csv} for {len(genes)} genes x {len(chans)} channels ...", flush=True)
    keep = [c[c["gene"].isin(genes) & c["channel_name"].isin(chans)]
            for c in pd.read_csv(csv, usecols=COLS, chunksize=2_000_000)]
    df = pd.concat(keep, ignore_index=True)
    df.to_parquet(p)
    print(f"[cache] wrote {p}  ({len(df):,} rows)", flush=True)
    return df


def top_rows(df, gene, ch_name, n):
    """Top-n SHAP cells of one (gene, channel), rank-ordered, in _materialize's column contract."""
    d = df[(df["gene"] == gene) & (df["channel_name"] == ch_name)].sort_values("rank").head(n).copy()
    d["score"] = d["shap"]
    d["segmentation"] = d["segmentation_id"]
    return d


def pick_genes(df, b):
    """The N_GENES complex members with the strongest top cell in this channel (all members if fewer),
    displayed in EBI member order (EMC1/EMC2/EMC3, not SHAP order)."""
    mem = members(b["cx"])
    d = df[df["gene"].isin(mem) & (df["channel_name"] == b["ch_name"])]
    best = d.groupby("gene")["shap"].max().sort_values(ascending=False)
    missing = [g for g in mem if g not in best.index]
    if missing:
        print(f"  [{b['label']} / {b['ch_name']}] members absent from the screen: {missing}")
    keep = set(best.index[:N_GENES])
    return [g for g in mem if g in keep]


def crop_row(df, b, gene, n):
    """Crop the top-n surviving cells of one gene (rank order preserved)."""
    raw, recs = _materialize(top_rows(df, gene, b["ch_name"], WIN + n), b["mc"], b["ch"], gene)
    o = np.argsort(recs["rank"].values)[:n]
    if len(o) < n:
        raise RuntimeError(f"{gene} ({b['ch_name']}): only {len(o)}/{n} cells survived cropping")
    return raw[o], recs.iloc[o].reset_index(drop=True)


def build_block(b):
    """(genes, ko_raw[G,N,1,H,W], ko_recs list, ntc_raw[G,N,...], ntc_recs) for one complex x channel."""
    df = ebi_rows(b["modality"])
    genes = pick_genes(df, b)
    ko = [crop_row(df, b, g, N_CELLS) for g in genes]
    ntc_raw, ntc_recs = crop_row(df, b, "NTC", len(genes) * N_CELLS)
    print(f"  [{b['label']} / {b['ch_name']}] rows {genes}  "
          + "  ".join(f"{g}:#{'/#'.join(str(int(r)) for r in rec['rank'])}" for g, (_, rec) in zip(genes, ko))
          + f"  NTC:#{'/#'.join(str(int(r)) for r in ntc_recs['rank'])}", flush=True)
    return dict(b=b, genes=genes, ko=ko, ntc=(ntc_raw, ntc_recs))


def tile(raw, rec, lo, hi):
    half = CROP_SIZE // 2
    gray = np.clip((raw - lo) / (hi - lo), 0, 1) * 255
    return composite(gray, seg_crop(rec["experiment"], rec["well"], rec["x_pheno"], rec["y_pheno"], half), half)


# block geometry, inches — tiles are placed by hand (not gridspec) so they stay square with no
# aspect padding, which is where the dead space came from.
T = 1.5              # tile edge
GAP = 0.03           # gap between tiles
MID = 0.42           # gap between the NTC half and the KO half
TITLE = 0.86         # title + subheader band above the tiles
BGX, BGY = 0.34, 0.3  # gaps between blocks (combined figure)
SUP, FOOT = 0.62, 0.5   # suptitle / footer bands
FS_TITLE, FS_SUB, FS_GENE, FS_BADGE, FS_SUP, FS_LET = 21, 16, 17, 12, 26, 26
SHOW_BADGES = False  # rank badges on each tile (cells are rank-ordered regardless)
BADGE_A = 0.3        # rank-badge alpha when shown
HALF = N_CELLS * T + (N_CELLS - 1) * GAP           # width of one 3-tile half
CAPTION = ("Cells ordered by per-cell SHAP from the EBI-complex multi_rank screen (most predictive first, left to "
           "right); NTC = top-SHAP non-targeting cells of the same channel, shared intensity window per channel.")


def block_h(nrows):
    return TITLE + nrows * T + (nrows - 1) * GAP


def block_w(genes):
    """Block width incl. the gene-label gutter, sized to the longest gene name (FS_GENE bold)."""
    return 2 * HALF + MID + 0.18 + 0.115 * max(len(g) for g in genes)


def _put(fig, W, H, x, y, im, badge):
    """One square tile at (x, y) inches from the figure's top-left."""
    ax = fig.add_axes([x / W, 1 - (y + T) / H, T / W, T / H])
    ax.imshow(im)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor("#888"); s.set_linewidth(0.5)
    if SHOW_BADGES:
        ax.text(0.04, 0.96, badge, transform=ax.transAxes, fontsize=FS_BADGE, fontweight="bold", color="white",
                alpha=0.72, va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.12", fc="#c1272d", ec="none", alpha=BADGE_A))


def draw_block(fig, blk, x0, y0, lo, hi, W, H, letter=None):
    """NTC 3xN (left) | KO 3xN (right, one row per gene) at inch offset (x0, y0) from the top-left."""
    b, genes = blk["b"], blk["genes"]
    nrows = len(genes)
    xk = x0 + HALF + MID
    ytop = y0 + TITLE
    ntc_raw, ntc_recs = blk["ntc"]
    for i, g in enumerate(genes):
        ko_raw, ko_recs = blk["ko"][i]
        y = ytop + i * (T + GAP)
        for j in range(N_CELLS):
            k = i * N_CELLS + j
            _put(fig, W, H, x0 + j * (T + GAP), y, tile(ntc_raw[k, 0], ntc_recs.iloc[k], lo, hi),
                 f"#{int(ntc_recs.iloc[k]['rank'])}")
            _put(fig, W, H, xk + j * (T + GAP), y, tile(ko_raw[j, 0], ko_recs.iloc[j], lo, hi),
                 f"#{int(ko_recs.iloc[j]['rank'])}")
        fig.text((xk + HALF + 0.1) / W, 1 - (y + T / 2) / H, g, fontsize=FS_GENE, fontweight="bold",
                 color="#c1272d", va="center", ha="left")
    fig.text((x0 + (2 * HALF + MID) / 2) / W, 1 - (y0 + 0.3) / H,
             f"{b['label']}  —  {b['marker_label']}", fontsize=FS_TITLE, fontweight="bold",
             va="center", ha="center")
    if letter:
        fig.text((x0 - 0.3) / W, 1 - (y0 + 0.26) / H, letter, fontsize=FS_LET, fontweight="bold",
                 va="center", ha="left")
    fig.text((x0 + HALF / 2) / W, 1 - (ytop - 0.13) / H, "control (NTC)", fontsize=FS_SUB,
             va="bottom", ha="center", color="#444")
    fig.text((xk + HALF / 2) / W, 1 - (ytop - 0.13) / H,
             f"{b['label'].replace(' complex', '')} gene KOs", fontsize=FS_SUB, fontweight="bold",
             va="bottom", ha="center", color="#c1272d")
    xd = (x0 + HALF + MID / 2) / W                                  # divider between the halves
    fig.add_artist(plt.Line2D([xd, xd], [1 - (ytop + nrows * T + (nrows - 1) * GAP) / H, 1 - (ytop - 0.06) / H],
                              color="#bbb", lw=1.0, transform=fig.transFigure))


def windows(blocks):
    """1-99 pct intensity window per channel over that channel's KO+NTC crops (comparable brightness)."""
    pool = {}
    for blk in blocks:
        raws = [r for r, _ in blk["ko"]] + [blk["ntc"][0]]
        pool.setdefault(blk["b"]["ch_name"], []).extend(x.ravel() for x in raws)
    out = {}
    for ch, vals in pool.items():
        lo, hi = np.percentile(np.concatenate(vals), (1, 99))
        out[ch] = (lo, hi if hi - lo > 1e-6 else lo + 1)
    return out


def save(fig, stem):
    os.makedirs(OUT, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(f"{OUT}/{stem}.{ext}", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {OUT}/{stem}.png/.svg", flush=True)


def build_blocks(refresh=False):
    """All blocks, with the cropped tiles cached so layout iterations don't re-read the zarrs."""
    p = f"{CACHE}/blocks_{N_GENES}x{N_CELLS}.pkl"
    if os.path.exists(p) and not refresh:
        return pd.read_pickle(p)
    blocks = [build_block(b) for b in BLOCKS]
    os.makedirs(CACHE, exist_ok=True)
    pd.to_pickle(blocks, p)
    return blocks


def main(ncol=2, refresh=False):
    blocks = build_blocks(refresh)
    win = windows(blocks)

    for blk in blocks:                                   # per-complex figures
        W, H = block_w(blk["genes"]) + 0.16, block_h(len(blk["genes"])) + 0.08
        fig = plt.figure(figsize=(W, H), facecolor="white")
        draw_block(fig, blk, 0.08, 0.04, *win[blk["b"]["ch_name"]], W, H)
        save(fig, f"ebi_shap_grid_{blk['b']['stem']}")

    order = [blocks[i] for i in COMBINED_ORDER]          # combined figure
    rows = [order[i:i + ncol] for i in range(0, len(order), ncol)]
    rowh = [max(block_h(len(blk["genes"])) for blk in r) for r in rows]
    colw = [max(block_w(r[c]["genes"]) for r in rows if c < len(r)) for c in range(ncol)]
    x0 = 0.45                                            # left margin holds the panel letters
    W = x0 + sum(colw) + (ncol - 1) * BGX + 0.12
    H = SUP + sum(rowh) + BGY * (len(rows) - 1) + FOOT
    fig = plt.figure(figsize=(W, H), facecolor="white")
    y = SUP
    for r, row in enumerate(rows):
        for c, blk in enumerate(row):
            draw_block(fig, blk, x0 + sum(colw[:c]) + c * BGX, y, *win[blk["b"]["ch_name"]], W, H,
                       letter="ABCDEFGH"[r * ncol + c])
        y += rowh[r] + BGY
    fig.text(0.5, 1 - 0.3 / H, "Top Predictive cells per protein complex",
             fontsize=FS_SUP, fontweight="bold", ha="center", va="center")
    fig.text(0.5, (FOOT - 0.28) / H, CAPTION, fontsize=13, style="italic", color="#333",
             ha="center", va="center")
    save(fig, "ebi_shap_grid_combined")


if __name__ == "__main__":
    main(refresh="--refresh" in sys.argv)
