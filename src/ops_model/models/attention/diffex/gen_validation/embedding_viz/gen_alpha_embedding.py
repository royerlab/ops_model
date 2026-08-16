"""Per-α UMAP animation: top-K real cells + generated cells at every α in ONE shared standardized embedding.
Generated cells start piled at the center (α=0, no phenotype) and migrate into their real-class territories as α
rises — the visual analog of the α→distinctiveness/mAP curve. Reuses the gen_real_map_cache embeddings (real +
gen, same embed_crops space); per-domain standardization (real vs population, gen vs α0) cancels the DiffAE offset.

Real + generated are BOTH colored by class (real faint, gen bright) so you can watch gen-X home in on real-X.
"""
import os, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
plt.rcParams["pdf.fonttype"] = 42

CACHE = "/hpc/projects/icd.fast.ops/analysis/figure4_traversals/gen_real_map_cache"
CENT = "/hpc/projects/icd.fast.ops/analysis/figure4_traversals/gen_real_centroid"   # faithful centroids for the color metric
OUT = "/hpc/projects/icd.fast.ops/analysis/figure4_embedding/gen_alpha_frames"
K = 20
SEED = 0


def load(grain, keep=None):
    real, rlab, gx, gl, alphas = [], [], {}, {}, None
    for c in sorted(glob.glob(f"{CACHE}/{grain}/*.npz")):
        d = np.load(c, allow_pickle=True); g = str(d["gene"]); alphas = list(d["alphas"])
        if keep is not None and g not in keep:
            continue
        r = np.asarray(d["real"], np.float32)[:K]; real.append(r); rlab += [g] * len(r)
        for ai in range(len(alphas)):
            gv = d["gen"][ai]
            if gv is None or not len(gv):
                continue
            gv = np.asarray(gv, np.float32)[:K]
            gx.setdefault(ai, []).append(gv); gl.setdefault(ai, []).extend([g] * len(gv))
    return (np.concatenate(real), np.array(rlab),
            {ai: np.concatenate(v) for ai, v in gx.items()}, {ai: np.array(gl[ai]) for ai in gl}, alphas)


def run(grain, gpu=False, include_real=False, fast=False, only=None, tag=None):
    tag = tag or grain
    real, rlab, gx, gl, alphas = load(grain, keep=only)
    a0 = len(alphas) // 2
    mu_g, sd_g = gx[a0].mean(0), gx[a0].std(0) + 1e-6           # gen baseline = α0 (NTC reconstruction)
    ais = sorted(gx)
    gz = {ai: (gx[ai] - mu_g) / sd_g for ai in ais}
    blocks = [gz[ai] for ai in ais]
    if include_real:
        mu_r, sd_r = real.mean(0), real.std(0) + 1e-6; rz = (real - mu_r) / sd_r; blocks = [rz] + blocks
    X = np.concatenate(blocks).astype(np.float32)
    print(f"[{grain}] UMAP fit on {len(X):,} pts (generated only over {len(ais)} α{', + real' if include_real else ''})")
    if gpu:
        from cuml.manifold import UMAP as U
        XY = U(n_neighbors=15, min_dist=0.3, random_state=SEED).fit_transform(X)
    else:
        import umap
        kw = dict(n_neighbors=15, min_dist=0.3, metric="euclidean")
        kw["n_jobs"] = -1 if fast else 1                                             # fast: all cores (drops exact seed reproducibility)
        if not fast:
            kw["random_state"] = SEED
        XY = umap.UMAP(**kw).fit_transform(X)
    XY = np.asarray(XY)
    off = 0; rxy = None
    if include_real:
        rxy = XY[:len(real)]; off = len(real)
    gxy = {}
    for ai in ais:
        gxy[ai] = XY[off:off + len(gz[ai])]; off += len(gz[ai])
    os.makedirs(OUT, exist_ok=True)
    np.savez(f"{OUT}/{tag}_coords.npz", XY=XY.astype(np.float32), alphas=np.array(alphas),
             ais=np.array(ais), splits=np.array([len(gz[ai]) for ai in ais]),
             labels=np.concatenate([gl[ai] for ai in ais]), off0=len(real) if include_real else 0)
    plot_frames(tag)


def plot_frames(tag, highlight=None, out=None, legend=False):
    """Per-α sweep on one shared UMAP: shape = cell (base-cell idx), color = class. If `highlight` (a set of
    classes) is given, those are drawn bright and all others faded (same size, low opacity) — all cells stay in
    frame. legend=True adds an EBI-complex legend of the highlighted set."""
    import colorcet as cc, matplotlib.patches as mp
    grain = tag; out = out or tag
    d = np.load(f"{OUT}/{tag}_coords.npz", allow_pickle=True)
    XY = d["XY"]; alphas = list(d["alphas"]); ais = list(d["ais"]); splits = list(d["splits"]); labels = d["labels"]
    off = int(d["off0"]); gxy, gl = {}, {}
    for ai, n in zip(ais, splits):
        gxy[ai] = XY[off:off + n]; gl[ai] = labels[off - int(d["off0"]):off - int(d["off0"]) + n]; off += n
    xlim = (XY[:, 0].min() - 1, XY[:, 0].max() + 1); ylim = (XY[:, 1].min() - 1, XY[:, 1].max() + 1)
    K_ = splits[0] // max(len(set(labels)), 1)                                        # cells per class (base-cell index)
    MARKERS = ["o", "s", "^", "v", "D", "P", "*", "X", "<", ">", "p", "h", "d"]
    from matplotlib.colors import to_rgba
    classes = sorted(set(labels)); c2c = {c: to_rgba(cc.glasbey[i % len(cc.glasbey)]) for i, c in enumerate(classes)}
    active = sorted(highlight) if highlight else classes                             # rainbow the colored set (picks, or all)
    hp = cm.get_cmap("gist_rainbow")(np.linspace(0, 1, len(active), endpoint=False))
    for i, c in enumerate(active):
        c2c[c] = tuple(hp[i])

    from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
    from scipy.spatial import cKDTree
    rad = 0.03 * max(xlim[1] - xlim[0], ylim[1] - ylim[0])

    def _dens(P, groups, r):                                                          # per-point # of same-group neighbors within r
        dn = np.zeros(len(P)); groups = np.asarray(groups)
        for g in np.unique(groups):
            idx = np.where(groups == g)[0]
            if len(idx) < 2:
                continue
            dn[idx] = np.asarray(cKDTree(P[idx]).query_ball_point(P[idx], r, return_length=True)) - 1
        return dn

    def panel(ax, ai):                                                               # shape=cell; color=class; density → saturation & size
        P = gxy[ai]; labs = gl[ai]; cellidx = np.arange(len(P)) % K_; mk_i = cellidx % len(MARKERS)
        hi = np.array([c in highlight for c in labs]) if highlight is not None else np.ones(len(labs), bool)
        cdn = np.zeros(len(P)); sdn = np.zeros(len(P))
        if hi.any():
            cd = _dens(P[hi], labs[hi], rad); sd = _dens(P[hi], mk_i[hi], rad)         # same-class / same-shape overlap
            cdn[hi] = cd / (cd.max() + 1e-9); sdn[hi] = sd / (sd.max() + 1e-9)
        base = np.array([c2c[c] for c in labs])
        hsv = rgb_to_hsv(base[:, :3]); hsv[:, 1] = hsv[:, 1] * np.clip(0.60 + 0.40 * cdn, 0, 1)   # saturation ↑ with same-class overlap (higher floor)
        col = np.concatenate([hsv_to_rgb(hsv), base[:, 3:4]], axis=1)
        lw = np.clip(3.5 * sdn ** 3, 0, 3.5)                                          # black border ↑ steeply (cubic) with same-shape overlap
        if highlight is not None and (~hi).any():                                     # faded rest (grey backdrop)
            for mi, mk in enumerate(MARKERS):
                m = (~hi) & (mk_i == mi)
                if m.any():
                    ax.scatter(P[m, 0], P[m, 1], s=60, c="0.8", marker=mk, lw=0, alpha=.07)
        for mi, mk in enumerate(MARKERS):
            m = hi & (mk_i == mi)
            if m.any():
                ax.scatter(P[m, 0], P[m, 1], s=80, c=col[m], marker=mk, edgecolors="black", linewidths=lw[m], alpha=.9)
        from matplotlib.lines import Line2D                                          # small legend: shape = cell number
        sh = [Line2D([0], [0], marker=MARKERS[i], color="0.35", lw=0, markersize=6, label=f"cell {i + 1}") for i in range(5)]
        sh.append(Line2D([0], [0], marker=r"$\cdots$", color="0.35", lw=0, markersize=8, label="…"))
        ax.legend(handles=sh, loc="upper left", fontsize=6, frameon=False, handletextpad=.2, labelspacing=.25, title="shape = cell", title_fontsize=6)
        ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_xticks([]); ax.set_yticks([])

    handles = [mp.Patch(color=c2c[c], label=c) for c in sorted(highlight)] if (legend and highlight) else None

    for ai in ais:                                                                   # per-α frame (for the GIF)
        fig, ax = plt.subplots(figsize=(11 if handles else 7, 7)); panel(ax, ai)
        sub = f"   ({len(highlight)} highlighted)" if highlight is not None else ""
        fig.suptitle(f"{grain}   α = {alphas[ai]:+.1f}{sub}\nshape = cell · color = class", fontsize=13, fontweight="bold")
        if handles:
            fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.62, 0.5), fontsize=7, frameon=False)
            fig.subplots_adjust(right=0.6)
        else:
            fig.tight_layout()
        fig.savefig(f"{OUT}/{out}_a{ai:02d}.png", dpi=110, bbox_inches="tight"); plt.close(fig)

    keys = [ai for ai in ais if alphas[ai] in (-5, -3, -2, -1, 0, 1, 2, 3, 5)]       # symmetric columns
    nc = len(keys); fig, axes = plt.subplots(1, nc, figsize=(3.0 * nc, 3.4))
    for ci, ai in enumerate(keys):
        panel(axes[ci], ai); axes[ci].set_title(f"α = {alphas[ai]:+.1f}", fontsize=11)
    fig.suptitle(f"{grain}: generated cells per α — shape = cell, color = class (one UMAP)", fontsize=14, fontweight="bold")
    if handles:
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.02), fontsize=7, ncol=min(6, len(handles)), frameon=False)
    fig.tight_layout()
    for e in ("png", "svg"):
        fig.savefig(f"{OUT}/{out}_montage.{e}", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}_montage + {len(ais)} frames")


def _tight_few_subset(grain, hi=5.0, member_max=4, tight_top=0.4):
    """Complexes that are well-grouped at α=hi (tight UMAP cluster) AND have few member genes."""
    import pandas as pd
    d = np.load(f"{OUT}/{grain}_coords.npz", allow_pickle=True)
    XY = d["XY"]; alphas = list(d["alphas"]); ais = list(d["ais"]); splits = list(d["splits"]); labels = d["labels"]
    off = int(d["off0"]); gxy, gl = {}, {}
    for ai, n in zip(ais, splits):
        gxy[ai] = XY[off:off + n]; gl[ai] = labels[off - int(d["off0"]):off - int(d["off0"]) + n]; off += n
    hi_ai = ais[int(np.argmin(np.abs(np.array(alphas) - hi)))]
    pc = pd.read_parquet("/hpc/projects/icd.fast.ops/models/diffex/viewer_assets_v5/_rankings/pma_v5_phase_complex.parquet",
                         columns=["predicted_class", "gene"])
    members = pc.groupby("predicted_class")["gene"].nunique().to_dict()
    classes = sorted(set(labels))
    tight = {c: (1.0 / (np.linalg.norm((p := gxy[hi_ai][gl[hi_ai] == c]) - p.mean(0), axis=1).mean() + 1e-9)
                 if (gl[hi_ai] == c).any() else 0) for c in classes}
    thr = np.quantile([tight[c] for c in classes], 1 - tight_top)
    sel = sorted(c for c in classes if members.get(c, 99) <= member_max and tight[c] >= thr)
    print(f"[{grain}] subset: {len(sel)}/{len(classes)} complexes (≤{member_max} members & tight@α{hi:g}):")
    for c in sel:
        print(f"    {c}  (members={members.get(c)})")
    return set(sel)


def legend_frame(grain, alpha=5.0):
    """Reference α frame where each complex has BOTH a distinct color AND shape (easier to tell apart), + a
    color+shape legend, so you can pick which to highlight. (The final highlight legend is color-only.)"""
    import colorcet as cc
    from matplotlib.lines import Line2D
    MARKERS = ["o", "s", "^", "v", "D", "P", "*", "X", "<", ">", "p", "h", "d", "8", "H"]
    d = np.load(f"{OUT}/{grain}_coords.npz", allow_pickle=True)
    XY = d["XY"]; alphas = list(d["alphas"]); ais = list(d["ais"]); splits = list(d["splits"]); labels = d["labels"]
    off = int(d["off0"]); gxy, gl = {}, {}
    for ai, n in zip(ais, splits):
        gxy[ai] = XY[off:off + n]; gl[ai] = labels[off - int(d["off0"]):off - int(d["off0"]) + n]; off += n
    ai = ais[int(np.argmin(np.abs(np.array(alphas) - alpha)))]
    classes = sorted(set(labels))
    c2c = {c: cc.glasbey[i % len(cc.glasbey)] for i, c in enumerate(classes)}
    c2m = {c: MARKERS[i % len(MARKERS)] for i, c in enumerate(classes)}               # distinct shape per complex
    fig, ax = plt.subplots(figsize=(15, 10))
    for c in classes:
        m = gl[ai] == c
        if m.any():
            ax.scatter(gxy[ai][m, 0], gxy[ai][m, 1], s=55, c=[c2c[c]], marker=c2m[c], lw=0, alpha=.9)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(f"{grain}  α = {alpha:g}  — pick complexes to highlight (color+shape)", fontsize=13, fontweight="bold")
    handles = [Line2D([0], [0], marker=c2m[c], color="w", markerfacecolor=c2c[c], markersize=8, label=c) for c in classes]
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.6, 0.5), fontsize=5.5, ncol=2, frameon=False, handlelength=1, columnspacing=1)
    fig.subplots_adjust(left=0.02, right=0.6)
    fig.savefig(f"{OUT}/{grain}_legend.png", dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"saved {grain}_legend.png ({len(classes)} complexes, color+shape)")


def highlight_sweep(grain, hi=5.0, member_max=4, tight_top=0.4, names=None):
    """α sweep on the FULL embedding with a subset drawn bright and everything else faded (same size, low opacity;
    all cells kept in frame). `names`: explicit complexes to highlight; else the tight/few-member auto-subset.
    Uses the existing {grain}_coords.npz — no refit. Saved as {grain}_highlight_*, with a legend."""
    sel = set(names) if names else _tight_few_subset(grain, hi, member_max, tight_top)
    plot_frames(grain, highlight=sel, out=f"{grain}_highlight", legend=True)
    make_gif(f"{grain}_highlight")


def make_gif(grain, ms=450, first=0, last=None, suffix="", hold_ais=(8, 16), hold_ms=1000):
    """Stitch per-α frames into a ping-pong GIF. first/last select a frame-index window (e.g. α0→+5 = first=8).
    hold_ais: α-frame indices (8=α0, 16=α5) held an extra hold_ms. suffix names {grain}_alpha{suffix}.gif."""
    from PIL import Image
    files = sorted(glob.glob(f"{OUT}/{grain}_a*.png"))[first:last]
    if not files:
        print("no frames"); return
    ais = [int(f.rsplit("_a", 1)[-1][:-4]) for f in files]
    imgs = [Image.open(f).convert("RGB") for f in files]
    w = min(i.width for i in imgs); h = min(i.height for i in imgs)
    imgs = [i.resize((w, h)) for i in imgs]
    n = len(imgs)
    seq = list(range(n)) + list(range(n - 2, 0, -1))                                 # ping-pong; endpoints shown once (no double-hold)
    dur = [ms + hold_ms if ais[i] in hold_ais else ms for i in seq]                  # hold on α0 / α5
    imgs[seq[0]].save(f"{OUT}/{grain}_alpha{suffix}.gif", save_all=True, append_images=[imgs[i] for i in seq[1:]],
                      duration=dur, loop=0, disposal=2)
    print(f"saved {grain}_alpha{suffix}.gif ({len(seq)} frames, hold α0/α5)")


if __name__ == "__main__":
    import sys
    g = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else "complex"
    if "--gif-only" in sys.argv:
        make_gif(g)
    else:
        run(g, gpu="--gpu" in sys.argv)
        make_gif(g)
