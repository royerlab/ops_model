"""Project generated geneKO cells into the REAL phase gene embedding (UMAP/PHATE).

The real gene embedding (gene_embedding_pca_optimized.h5ad) is 1052 genes in a 101-d phase-CellDINO PCA space,
built from per-exp z-scored raw 1024-d CellDINO features. Our generated cells live in the SAME raw CellDINO space
(embed_crops, cached in gen_real_map_cache). So we can pass each class's 30 generated cells straight through the
exact PCA (z-score vs the α=0 generated-NTC baseline → subtract pca_mean → project onto pca_components), aggregate
to a "generated geneKO dot", and land it in the real layout by kNN landmark projection (k=8, cosine — the UMAP's
own n_neighbors/metric). We highlight the EBI complexes whose generated dots land tightest on their true gene.

Validation (rank of true gene among 1052, α=3): generated centroids match real held-out centroids
(gen top-20 64% vs real 58%) — generated cells land where real cells land.
"""
import os, glob, json
import numpy as np
import pandas as pd
import anndata as ad
from scipy.spatial.distance import cdist

D = "/hpc/projects/icd.fast.ops/organelle_attribution/pca_optimized_v0.3/cell_dino/zscore_per_exp/paper_v2/phase_only/fixed_80%/cosine"
CACHE = "/hpc/projects/icd.fast.ops/analysis/figure4_traversals/gen_real_map_cache/geneKO"
OUT = "/hpc/projects/icd.fast.ops/analysis/figure4_embedding/gen_passthrough"
K = 8                      # kNN landmarks = UMAP n_neighbors
A0 = 8                     # α=0 index in the cached alpha grid


def _load_embedding():
    a = ad.read_h5ad(f"{D}/gene_embedding_pca_optimized.h5ad")
    g = ad.read_h5ad(f"{D}/per_signal/Phase_gene.h5ad")
    comp = np.asarray(g.uns["pca_components"], np.float64)   # 101 x 1024
    mean = np.asarray(g.uns["pca_mean"], np.float64)         # 1024
    return a, comp, mean


def _cache_files():
    return sorted(glob.glob(f"{CACHE}/*.npz"))


# Standardization mode for placing generated cells into the real embedding:
#   "self" — z-score generated vs the pooled α=0 generated-NTC (cancels the DiffAE→CellDINO domain offset;
#            forces α=0 to the origin, so the α=0 start point is pinned/derived, not where CellDINO puts it)
#   "real" — z-score generated vs the real population (same treatment real cells get) → FAITHFUL TO CELLDINO:
#            α=0 lands wherever CellDINO actually encodes it (honest; exposes the domain gap, no pinning)
STD_MODE = "real"
PTAG = "" if STD_MODE == "self" else f"_{STD_MODE}"


def _pq(ai):
    return f"{OUT}/proj{PTAG}_a{ai}.parquet"


def _gen_baseline(files):
    """Pooled α=0 generated-NTC mean/std over all classes (the generated-domain standardization)."""
    a0 = [np.asarray(np.load(f, allow_pickle=True)["gen"][A0], np.float64)
          for f in files if np.load(f, allow_pickle=True)["gen"][A0] is not None]
    x = np.concatenate([v for v in a0 if len(v)])
    return x.mean(0), x.std(0) + 1e-6


def _real_baseline(files):
    """Pooled real-cell mean/std over all classes (real-population standardization — faithful to CellDINO)."""
    x = np.concatenate([np.asarray(np.load(f, allow_pickle=True)["real"], np.float64) for f in files])
    return x.mean(0), x.std(0) + 1e-6


def _baseline(files):
    return _real_baseline(files) if STD_MODE == "real" else _gen_baseline(files)


def _project(x, mu, sd, comp, mean):
    """raw 1024-d cells -> 101-d PCs via the real pipeline's z-score + PCA."""
    return (((x - mu) / sd) - mean) @ comp.T


def _ntc_mask(a):
    return a.obs["perturbation"].astype(str).str.startswith("NTC").values    # real NTC_grp* rows in the embedding


def _ntc_flip(a, comp=None, mean=None):
    """Real NTC points from the actual embedding. Per-layout (fx, fy, ntc_coords[N,2]) flips (PHATE only) that
    put the real-NTC centroid in the bottom-left corner (relative to the real-gene median)."""
    m = _ntc_mask(a)
    out = {}
    for layout in ("umap", "phate"):
        C = np.asarray(a.obsm[f"X_{layout}"], np.float64)
        ntc = C[m]
        cx, cy = np.median(C, 0); nx, ny = ntc.mean(0)
        fx = (-1.0 if nx > cx else 1.0) if layout == "phate" else 1.0     # only PHATE flips
        fy = (-1.0 if ny > cy else 1.0) if layout == "phate" else 1.0
        out[layout] = (fx, fy, ntc * np.array([fx, fy]))
    return out


def _bg_colors(a, col="leiden_r4"):
    """Uniform categorical color per Leiden cluster for the real-embedding background (tab20 family, cycled)."""
    import matplotlib.cm as cm
    from matplotlib.colors import to_rgba
    lab = a.obs[col].astype(object)
    cats = sorted(lab.dropna().unique(), key=lambda x: int(x) if str(x).isdigit() else str(x))
    base = [to_rgba(c) for c in (list(cm.get_cmap("tab20").colors)
                                 + list(cm.get_cmap("tab20b").colors)
                                 + list(cm.get_cmap("tab20c").colors))]
    return lab, {c: base[i % len(base)] for i, c in enumerate(cats)}


def _draw_bg(ax, a, layout, fx, fy, lab, cmap):
    """Real genes colored by Leiden cluster — bigger, uniform dots."""
    C = np.asarray(a.obsm[f"X_{layout}"], np.float64) * np.array([fx, fy])
    ax.scatter(C[:, 0], C[:, 1], s=42, c=[cmap[v] for v in lab], lw=0, alpha=0.35, zorder=2)


def _draw_ntc(ax, ntc):
    ax.scatter(ntc[:, 0], ntc[:, 1], s=90, marker="X", color="#8b0000", edgecolor="k", lw=0.6, zorder=9)


def _landmark(pc, Xpca, coords):
    """Place a query PC vector in the 2-D layout at the cosine-weighted mean of its K nearest real genes."""
    d = cdist(pc[None], Xpca, metric="cosine")[0]
    nn = np.argsort(d)[:K]
    w = 1.0 / (d[nn] + 1e-6); w /= w.sum()
    return w @ coords[nn]


def compute(ai=14):
    """Project every generated geneKO centroid at alpha-index `ai` into UMAP+PHATE. Cache to OUT/proj_a{ai}.npz."""
    os.makedirs(OUT, exist_ok=True)
    a, comp, mean = _load_embedding()
    files = _cache_files()
    idx = {n: i for i, n in enumerate(a.obs_names)}
    Xpca = np.asarray(a.obsm["X_pca"], np.float64)
    Uc, Pc = np.asarray(a.obsm["X_umap"]), np.asarray(a.obsm["X_phate"])
    mu, sd = _baseline(files)
    alpha = None
    rows = []
    for f in files:
        d = np.load(f, allow_pickle=True); gene = str(d["gene"]); alpha = float(d["alphas"][ai])
        if gene not in idx:
            continue
        gv = d["gen"][ai]
        if gv is None or not len(gv):
            continue
        pc = _project(np.asarray(gv, np.float64), mu, sd, comp, mean).mean(0)   # generated dot in PC space
        gi = idx[gene]
        rank = int(np.where(np.argsort(cdist(pc[None], Xpca, "cosine")[0]) == gi)[0][0]) + 1
        gu, gp = _landmark(pc, Xpca, Uc), _landmark(pc, Xpca, Pc)
        rows.append((gene, gu[0], gu[1], gp[0], gp[1], Uc[gi, 0], Uc[gi, 1], Pc[gi, 0], Pc[gi, 1], rank))
    df = pd.DataFrame(rows, columns=["gene", "gu0", "gu1", "gp0", "gp1", "ru0", "ru1", "rp0", "rp1", "rank"])
    df.to_parquet(_pq(ai))
    print(f"[compute] ai={ai} α={alpha:+.1f}: projected {len(df)} generated geneKO dots  "
          f"(rank-to-true median {df['rank'].median():.0f}, top20 {(df['rank']<=20).mean():.0%})")
    return alpha


POS_AIS = [8, 9, 10, 11, 12, 13, 14, 15, 16]      # α = 0, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5


def compute_many(ais=POS_AIS):
    """Project generated centroids for several alpha indices in one pass (embedding loaded once)."""
    os.makedirs(OUT, exist_ok=True)
    a, comp, mean = _load_embedding()
    files = _cache_files()
    idx = {n: i for i, n in enumerate(a.obs_names)}
    Xpca = np.asarray(a.obsm["X_pca"], np.float64)
    Uc, Pc = np.asarray(a.obsm["X_umap"]), np.asarray(a.obsm["X_phate"])
    mu, sd = _baseline(files)
    for ai in ais:
        if os.path.exists(_pq(ai)):
            continue
        rows = []
        for f in files:
            d = np.load(f, allow_pickle=True); gene = str(d["gene"])
            if gene not in idx:
                continue
            gv = d["gen"][ai]
            if gv is None or not len(gv):
                continue
            pc = _project(np.asarray(gv, np.float64), mu, sd, comp, mean).mean(0)
            gi = idx[gene]
            rank = int(np.where(np.argsort(cdist(pc[None], Xpca, "cosine")[0]) == gi)[0][0]) + 1
            gu, gp = _landmark(pc, Xpca, Uc), _landmark(pc, Xpca, Pc)
            rows.append((gene, gu[0], gu[1], gp[0], gp[1], Uc[gi, 0], Uc[gi, 1], Pc[gi, 0], Pc[gi, 1], rank))
        pd.DataFrame(rows, columns=["gene", "gu0", "gu1", "gp0", "gp1", "ru0", "ru1", "rp0", "rp1", "rank"]
                     ).to_parquet(_pq(ai))
        print(f"[compute_many] a{ai} done ({len(rows)} dots)")


MARKERS = ["o", "s", "^", "D", "v", "P", "p", "h", "<", ">", "*", "8"]
CMAP_ALPHA = "viridis_r"     # light = low α, dark = high α


def plot_traversal(n_complex=10, ais=POS_AIS, cmap=CMAP_ALPHA):
    """One plot per layout: generated-dot trajectory across α=0→+5, line/dots colored by α (viridis),
    converging onto each true real gene from the NTC (✕). Marker SHAPE = EBI complex.
    Highlighted set = member genes of the top-`n_complex` closest complexes at α=5 (tightest median displacement)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize
    plt.rcParams["pdf.fonttype"] = 42

    a, comp, mean = _load_embedding()
    flips = _ntc_flip(a)
    ec, cxcmap = _bg_colors(a)
    alphas = np.array([float(np.load(_cache_files()[0], allow_pickle=True)["alphas"][ai]) for ai in ais])
    norm = Normalize(vmin=alphas.min(), vmax=alphas.max())
    P = {ai: pd.read_parquet(_pq(ai)).set_index("gene") for ai in ais}
    end = P[ais[-1]].reset_index()                                     # α=5 frame
    picked_cx, dfc = _pick_complexes(a, end, n=n_complex, min_members=2)   # top-N closest complexes @ α=5
    g2c = dfc.set_index("gene")["complex"].to_dict()
    members = [g for g in dfc[dfc["complex"].isin(picked_cx)]["gene"]]
    shape = {c: MARKERS[i % len(MARKERS)] for i, c in enumerate(picked_cx)}
    end = end.set_index("gene")
    # shared α=0 origin: DERIVED from the pooled generated-NTC baseline (all genes are NTC at α=0), not pinned.
    # This is the honest projection of the α=0 baseline — it lands ~near the real NTC cluster on its own.
    Xpca = np.asarray(a.obsm["X_pca"], np.float64)
    pc0 = (np.zeros(mean.shape[0]) - mean) @ comp.T
    origin = {lay: _landmark(pc0, Xpca, np.asarray(a.obsm[f"X_{lay}"], np.float64)) for lay in ("umap", "phate")}

    for layout, (gx, gy, rx, ry) in {"umap": ("gu0", "gu1", "ru0", "ru1"),
                                     "phate": ("gp0", "gp1", "rp0", "rp1")}.items():
        fx, fy, ntc = flips[layout]
        o = origin[layout] * np.array([fx, fy])    # derived α=0 baseline (not the NTC centroid)
        fig, ax = plt.subplots(figsize=(12, 10))
        _draw_bg(ax, a, layout, fx, fy, ec, cxcmap)
        _draw_ntc(ax, ntc)
        sm = None
        for g in members:
            mk = shape[g2c[g]]
            tx = np.array([o[0]] + [P[ai].loc[g, gx] * fx for ai in ais[1:]])   # α=0 snapped to real NTC
            ty = np.array([o[1]] + [P[ai].loc[g, gy] * fy for ai in ais[1:]])
            pts = np.column_stack([tx, ty]).reshape(-1, 1, 2)
            segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
            lc = LineCollection(segs, cmap=cmap, norm=norm, lw=2.0, zorder=3)
            lc.set_array((alphas[:-1] + alphas[1:]) / 2)               # line color = α (light→dark)
            sm = ax.add_collection(lc)
            rxv, ryv = end.loc[g, rx] * fx, end.loc[g, ry] * fy
            ax.plot([tx[-1], rxv], [ty[-1], ryv], ls=":", color="#444", lw=1.3, zorder=4)   # residual gap: endpoint→target
            ax.scatter([tx[-1]], [ty[-1]], s=70, c=[alphas[-1]], cmap=cmap, norm=norm,
                       marker="o", edgecolor="k", lw=0.6, zorder=5)    # generated α=5 endpoint (no complex symbol)
            ax.scatter(rxv, ryv, s=180, marker=mk, facecolor="none", edgecolor="k", lw=2.2, zorder=6)  # target: complex symbol
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(f"Generated geneKO trajectory α=0→+5 in real phase {layout.upper()}\n"
                     f"line color = α (light→dark)   ● α=5 endpoint  ···→ gap to target   "
                     f"symbol = true real gene   ✕ real NTC   (top-{n_complex} complexes)", fontsize=12)
        cb = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.02); cb.set_label("traversal α", fontsize=11)
        handles = [Line2D([0], [0], marker=shape[c], color="k", markerfacecolor="none", markersize=11,
                          lw=0, label=(c[:38] + "…") if len(c) > 39 else c) for c in picked_cx]
        handles.append(Line2D([0], [0], marker="X", color="w", markerfacecolor="#8b0000", markersize=13, label="NTC anchor"))
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.16, 1.0), fontsize=9, frameon=False,
                  title="EBI complex", title_fontsize=10)
        fig.tight_layout()
        for e in ("png", "svg"):
            fig.savefig(f"{OUT}/traversal_{layout}.{e}", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"saved traversal_{layout}: {len(members)} genes across {len(picked_cx)} complexes")
    print("complexes:", picked_cx)


def _traj(o, P, ais, alphas, g, gx, gy, fx, fy, tgt):
    """Trajectory for gene g, truncated at the α of CLOSEST approach to its target (drop α's that don't help).
    o is a fixed α=0 origin (self-std, pinned to shared baseline) or None to use the gene's own α=0 projection."""
    xs = np.array(([o[0]] if o is not None else [P[ais[0]].loc[g, gx] * fx])
                  + [P[ai].loc[g, gx] * fx for ai in ais[1:]])
    ys = np.array(([o[1]] if o is not None else [P[ais[0]].loc[g, gy] * fy])
                  + [P[ai].loc[g, gy] * fy for ai in ais[1:]])
    b = int(np.hypot(xs - tgt[0], ys - tgt[1]).argmin())
    return xs[:b + 1], ys[:b + 1], alphas[:b + 1]


def _fps(pts, k, seeds):
    """Farthest-point sampling of k indices over 2-D pts, seeded with the given index/indices."""
    order = list(seeds) if hasattr(seeds, "__iter__") else [seeds]
    while len(order) < min(k, len(pts)):
        d = cdist(pts, pts[order]).min(1); d[order] = -1
        order.append(int(d.argmax()))
    return order


MITO_MARKERS = {"TOMM20", "TOMM70", "TOMM40", "HSPD1", "HSPE1", "ATAD3A", "MRM1", "MRPL39",
                "SDHA", "SDHB", "NDUFA9", "NDUFS1", "TIMM23", "TIMM44", "OPA1", "MFN2", "VDAC1"}


def _best_mito(pool, g2c, rcol="reach"):
    """Best-reaching mitochondrial gene in `pool` (EBI complex names an organelle, or a known mito marker)."""
    mito = [g for g in pool.index if isinstance(g2c.get(g), str) and "mitochond" in g2c[g].lower()]
    if not mito:
        mito = [g for g in pool.index if g in MITO_MARKERS]
    return pool.loc[mito, rcol].idxmin() if mito else None


def plot_diverse(n=10, ais=POS_AIS, cmap=CMAP_ALPHA, rank_max=20, genes=None):
    """Clean version: a handful of INDIVIDUAL geneKO trajectories that (1) reach their target (rank ≤ rank_max),
    (2) travel a real distance from NTC (strong traversal), and (3) are spread across the embedding via
    farthest-point sampling — so picks land in different corners (mito, 60S, etc.)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize
    plt.rcParams["pdf.fonttype"] = 42

    a, comp, mean = _load_embedding()
    flips = _ntc_flip(a)
    ec, cxcmap = _bg_colors(a)
    alphas = np.array([float(np.load(_cache_files()[0], allow_pickle=True)["alphas"][ai]) for ai in ais])
    norm = Normalize(vmin=alphas.min(), vmax=alphas.max())
    P = {ai: pd.read_parquet(_pq(ai)).set_index("gene") for ai in ais}
    end = P[ais[-1]].copy()
    Xpca = np.asarray(a.obsm["X_pca"], np.float64)
    pc0 = (np.zeros(mean.shape[0]) - mean) @ comp.T
    g2c = a.obs["ebi_complex"].dropna().to_dict()

    for layout, (gx, gy, rx, ry) in {"umap": ("gu0", "gu1", "ru0", "ru1"),
                                     "phate": ("gp0", "gp1", "rp0", "rp1")}.items():
        fx, fy, ntc = flips[layout]
        # self-std pins α=0 to the shared baseline origin; real-std uses each gene's own α=0 (faithful to CellDINO)
        o = _landmark(pc0, Xpca, np.asarray(a.obsm[f"X_{layout}"], np.float64)) * np.array([fx, fy]) \
            if STD_MODE == "self" else None
        ntc_c = ntc.mean(0)
        T = end[[rx, ry]].values * np.array([fx, fy])                            # targets (flipped), aligned to end
        # reach = CLOSEST approach to target over the whole traversal (all α, not just α=5)
        D = np.full(len(end), np.inf)
        for ai in ais:
            Pa = P[ai].reindex(end.index)
            D = np.minimum(D, np.hypot(Pa[gx].values * fx - T[:, 0], Pa[gy].values * fy - T[:, 1]))
        if o is not None:
            D = np.minimum(D, np.hypot(o[0] - T[:, 0], o[1] - T[:, 1]))
        end["reach"] = D
        end["journey"] = np.hypot(T[:, 0] - ntc_c[0], T[:, 1] - ntc_c[1])        # target distance from real NTC
        if genes:
            picked = [g for g in genes if g in end.index]
        else:
            pool = end[end["journey"] >= end["journey"].median()]              # traveled a real distance
            cand = pool[(pool["reach"] <= pool["reach"].quantile(0.20))        # lands close in 2-D
                        & (pool["rank"] <= rank_max)].copy()                   # AND ranks well (clean, not coincidental)
            seeds = [cand["reach"].idxmin()]
            if layout == "umap":                                               # PHATE already has a mito example
                extra = [_best_mito(pool, g2c)]                                # force a mito example
                tl = end[(end[rx] < end[rx].quantile(0.20))                    # far-left region
                         & (end["rank"] <= rank_max) & (end["reach"] <= 1.0)]  # clean reacher
                extra.append((tl[ry] - tl[rx]).idxmax() if len(tl) else None)  # most top-left (high y, low x)
                for g in extra:
                    if g is not None:
                        if g not in cand.index:
                            cand = pd.concat([cand, end.loc[[g]]])
                        seeds.append(g)
                seeds = list(dict.fromkeys(seeds[1:] + [seeds[0]]))            # forced seeds first, then min-reach
            gl = list(cand.index)
            pts = cand[[rx, ry]].values * np.array([fx, fy])
            si = list(dict.fromkeys(gl.index(g) for g in seeds))
            picked = [gl[i] for i in _fps(pts, n, si)]
        fig, ax = plt.subplots(figsize=(12, 10))
        _draw_bg(ax, a, layout, fx, fy, ec, cxcmap)
        _draw_ntc(ax, ntc)
        for g in picked:
            tgt = np.array([end.loc[g, rx] * fx, end.loc[g, ry] * fy])
            tx, ty, al = _traj(o, P, ais, alphas, g, gx, gy, fx, fy, tgt)       # stop at closest approach
            if len(tx) >= 2:
                pts = np.column_stack([tx, ty]).reshape(-1, 1, 2)
                segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
                colr = plt.get_cmap(cmap)(norm((al[:-1] + al[1:]) / 2))         # color = α (light→dark)
                colr[:, 3] = np.linspace(0.55, 1.0, len(colr))                  # opacity ramps up toward the end
                ax.add_collection(LineCollection(segs, colors=colr, lw=2.6, zorder=3))
            ax.plot([tx[-1], tgt[0]], [ty[-1], tgt[1]], ls=":", color="#444", lw=1.3, zorder=4)  # residual gap
            ax.scatter([tx[-1]], [ty[-1]], s=85, c=[al[-1]], cmap=cmap, norm=norm,
                       marker="o", edgecolor="k", lw=0.6, zorder=5)                              # closest-approach endpoint
            ax.scatter(tgt[0], tgt[1], s=200, marker="*", facecolor="none", edgecolor="k", lw=2.2, zorder=6)  # target
            ax.annotate(g, (tgt[0], tgt[1]), textcoords="offset points", xytext=(7, 5),
                        fontsize=11, fontweight="bold", color="k", zorder=9)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(f"Best geneKO trajectories reaching diverse targets — real phase {layout.upper()}\n"
                     f"line color = α (light→dark), fades in toward the end   ● closest-approach α  ···→ residual gap   "
                     f"★ true real gene   ✕ real NTC", fontsize=11.5)
        from matplotlib.cm import ScalarMappable
        smap = ScalarMappable(norm=norm, cmap=cmap); smap.set_array([])
        cb = fig.colorbar(smap, ax=ax, fraction=0.035, pad=0.02); cb.set_label("traversal α", fontsize=11)
        fig.tight_layout()
        for e in ("png", "svg"):
            fig.savefig(f"{OUT}/traversal_best_{layout}{PTAG}.{e}", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"saved traversal_best_{layout}{PTAG}: {[(g, int(end.loc[g, 'rank']), round(end.loc[g, 'reach'], 1)) for g in picked]}")


def _pick_complexes(a, df, n=12, min_members=3):
    """EBI complexes with >=min_members generated dots, ranked by tightest median gen->true UMAP displacement."""
    ec = a.obs["ebi_complex"]
    g2c = ec.dropna().to_dict()
    df = df.copy(); df["complex"] = df["gene"].map(g2c)
    df = df.dropna(subset=["complex"])
    df["disp"] = np.hypot(df["gu0"] - df["ru0"], df["gu1"] - df["ru1"])
    agg = df.groupby("complex").agg(n=("gene", "size"), disp=("disp", "median"))
    agg = agg[agg["n"] >= min_members].sort_values("disp")
    return list(agg.index[:n]), df


def plot(ai=14, n_complexes=12, names=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.lines import Line2D
    plt.rcParams["pdf.fonttype"] = 42

    a, comp, mean = _load_embedding()
    flips = _ntc_flip(a)
    df = pd.read_parquet(_pq(ai))
    alpha = float(np.load(_cache_files()[0], allow_pickle=True)["alphas"][ai])
    picked, dfc = _pick_complexes(a, df, n=n_complexes)
    if names:
        picked = [c for c in dfc["complex"].unique() if any(nm.lower() in c.lower() for nm in names)]
    cols = cm.get_cmap("gist_rainbow")(np.linspace(0, 1, len(picked), endpoint=False))
    cmap = dict(zip(picked, cols))

    for layout, (gx, gy, rx, ry) in {"umap": ("gu0", "gu1", "ru0", "ru1"),
                                     "phate": ("gp0", "gp1", "rp0", "rp1")}.items():
        fx, fy, ntc = flips[layout]
        Xr = np.asarray(a.obsm[f"X_{layout}"]) * np.array([fx, fy])
        fig, ax = plt.subplots(figsize=(11, 10))
        ax.scatter(Xr[:, 0], Xr[:, 1], s=10, c="#dddddd", lw=0, zorder=1)          # all real genes (grey)
        _draw_ntc(ax, ntc)
        for cx in picked:
            sub = dfc[dfc["complex"] == cx]; col = cmap[cx]
            for _, r in sub.iterrows():
                ax.plot([r[rx] * fx, r[gx] * fx], [r[ry] * fy, r[gy] * fy], "-", color=col, lw=0.8, alpha=0.5, zorder=2)
            ax.scatter(sub[rx] * fx, sub[ry] * fy, s=95, marker="o", facecolor="none",
                       edgecolor=col, lw=2.0, zorder=4)                              # real member gene
            ax.scatter(sub[gx] * fx, sub[gy] * fy, s=150, marker="*", color=col,
                       edgecolor="k", lw=0.6, zorder=5)                              # generated dot
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(f"Generated geneKO cells projected into real phase {layout.upper()}  (α = {alpha:+.1f})\n"
                     f"★ generated dot   ◯ true real gene   — same complex", fontsize=13)
        handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=cmap[c], markersize=10,
                          label=(c[:42] + "…") if len(c) > 43 else c) for c in picked]
        ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8.5, frameon=False)
        fig.tight_layout()
        for e in ("png", "svg"):
            fig.savefig(f"{OUT}/passthrough_{layout}_a{ai}.{e}", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"saved passthrough_{layout}_a{ai}  ({len(picked)} complexes highlighted)")
    json.dump(picked, open(f"{OUT}/picked_a{ai}.json", "w"))
    print("picked complexes:", picked)


def plot_top_genes(ai, n=5):
    """Highlight the n individual geneKOs whose generated dot lands closest to its true real gene."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.lines import Line2D
    plt.rcParams["pdf.fonttype"] = 42

    a, comp, mean = _load_embedding()
    flips = _ntc_flip(a)
    ec, cxcmap = _bg_colors(a)
    df = pd.read_parquet(_pq(ai)).copy()
    alpha = float(np.load(_cache_files()[0], allow_pickle=True)["alphas"][ai])
    df["disp"] = np.hypot(df["gu0"] - df["ru0"], df["gu1"] - df["ru1"])
    top = df.nsmallest(n, "disp").reset_index(drop=True)
    cols = cm.get_cmap("gist_rainbow")(np.linspace(0, 1, n, endpoint=False))

    for layout, (gx, gy, rx, ry) in {"umap": ("gu0", "gu1", "ru0", "ru1"),
                                     "phate": ("gp0", "gp1", "rp0", "rp1")}.items():
        fx, fy, ntc = flips[layout]
        fig, ax = plt.subplots(figsize=(11, 10))
        _draw_bg(ax, a, layout, fx, fy, ec, cxcmap)
        _draw_ntc(ax, ntc)
        for i, r in top.iterrows():
            col = cols[i]
            ax.plot([r[rx] * fx, r[gx] * fx], [r[ry] * fy, r[gy] * fy], "-", color=col, lw=1.0, alpha=0.6, zorder=2)
            ax.scatter(r[rx] * fx, r[ry] * fy, s=110, marker="o", facecolor="none", edgecolor=col, lw=2.2, zorder=4)
            ax.scatter(r[gx] * fx, r[gy] * fy, s=170, marker="*", color=col, edgecolor="k", lw=0.6, zorder=5)
            ax.annotate(r["gene"], (r[gx] * fx, r[gy] * fy), textcoords="offset points", xytext=(6, 6),
                        fontsize=10, fontweight="bold", color=col, zorder=6)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(f"Top-{n} closest generated→true geneKOs in real phase {layout.upper()}  (α = {alpha:+.1f})\n"
                     f"★ generated dot   ◯ true real gene", fontsize=13)
        handles = [Line2D([0], [0], marker="*", color="w", markerfacecolor=cols[i], markersize=13,
                          label=f"{r['gene']}  (rank {int(r['rank'])})") for i, r in top.iterrows()]
        handles.append(Line2D([0], [0], marker="X", color="w", markerfacecolor="#8b0000", markersize=13, label="NTC anchor"))
        ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=10, frameon=False)
        fig.tight_layout()
        for e in ("png", "svg"):
            fig.savefig(f"{OUT}/topgenes_{layout}_a{ai}.{e}", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"saved topgenes_{layout}_a{ai}: {list(top['gene'])}")


if __name__ == "__main__":
    import sys
    ai = int(sys.argv[sys.argv.index("--ai") + 1]) if "--ai" in sys.argv else 14
    if "--diverse" in sys.argv:
        compute_many()
        plot_diverse(n=10)
    elif "--traversal" in sys.argv:
        compute_many()
        plot_traversal(n_complex=10)
    elif "--topgenes" in sys.argv:
        if not os.path.exists(_pq(ai)):
            compute(ai=ai)
        plot_top_genes(ai=ai, n=5)
    elif "--plot" in sys.argv:
        plot(ai=ai)
    else:
        compute(ai=ai)
        plot(ai=ai)
