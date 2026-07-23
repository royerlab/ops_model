"""Faithful re-embedding of generated centroids into the real phase map (UMAP transform + joint PHATE).

Replaces the kNN-landmark placement (gen_phate_passthrough) with proper out-of-sample / joint embeddings —
NEW outputs only, nothing existing is overwritten (writes to gen_passthrough_refit/).

  UMAP  : fit umap-learn reducer on the 1052 real genes (X_pca), then reducer.transform() the generated
          centroids → real layout fixed, generated projected out-of-sample (can land off-manifold), class-blind.
          (umap-learn direct = the "gav" recipe in pca_optimization/embeddings.py; supports .transform(),
          unlike the scanpy "max" recipe — hence a slightly different, but faithful, layout.)
  PHATE : no out-of-sample transform exists → JOINT fit_transform on [real genes ; generated centroids],
          knn=8, decay=10 (the GRASSP-canonical PHATE params). Generated participate in the embedding
          (not snapped onto real genes); the real layout shifts slightly.

Generated = per-class centroid at each positive α (real-population standardization, faithful to CellDINO),
reusing gen_phate_passthrough's exact projection (z-score vs real pop → subtract pca_mean → pca_components).
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_phate_passthrough as gp

OUT = "/hpc/projects/icd.fast.ops/analysis/figure4_embedding/gen_passthrough_refit"
REP_AI = 14        # α=3 — one representative centroid per class (best-α phenotype); balanced ~1:1 vs real


def build(rep_ai=REP_AI):
    import umap, phate
    os.makedirs(OUT, exist_ok=True)
    a, comp, mean = gp._load_embedding()
    Xr = np.asarray(a.obsm["X_pca"], np.float64)                       # real genes (1052,101)
    real_names = list(a.obs_names)
    files = gp._cache_files()
    mu, sd = gp._baseline(files)                                       # real-population std (STD_MODE="real")
    alpha = float(np.load(files[0], allow_pickle=True)["alphas"][rep_ai])

    genes, G = [], []                                                  # ONE generated centroid per class (Nclass,101)
    idx = {n: i for i, n in enumerate(real_names)}
    for f in files:
        d = np.load(f, allow_pickle=True); g = str(d["gene"])
        gv = d["gen"][rep_ai]
        if g not in idx or gv is None or not len(gv):
            continue
        genes.append(g)
        G.append(gp._project(np.asarray(gv, np.float64), mu, sd, comp, mean).mean(0))
    G = np.stack(G)                                                    # (Nclass, 101)
    print(f"[refit] real {Xr.shape}  generated {G.shape} at α={alpha:+.1f}  (total {len(Xr)+len(G)})", flush=True)

    # ---- UMAP: fit on real, transform generated (out-of-sample; real layout fixed) ----
    ur = umap.UMAP(n_neighbors=8, min_dist=0.25, metric="cosine", random_state=1)
    real_umap = ur.fit_transform(Xr)
    gen_umap = ur.transform(G)
    print("[refit] UMAP done", flush=True)

    # ---- PHATE: joint fit_transform on [real ; generated] (knn=8, decay=10 canonical) ----
    pj = phate.PHATE(knn=8, decay=10, n_components=2, random_state=1, n_jobs=-1, verbose=False)
    emb = pj.fit_transform(np.vstack([Xr, G]))
    real_phate, gen_phate = emb[:len(Xr)], emb[len(Xr):]
    print("[refit] PHATE done", flush=True)

    np.savez(f"{OUT}/coords.npz", genes=np.array(genes), alpha=alpha, real_names=np.array(real_names),
             real_umap=real_umap, real_phate=real_phate, gen_umap=gen_umap, gen_phate=gen_phate)
    print(f"[refit] saved {OUT}/coords.npz", flush=True)
    _sanity()


def _sanity():
    """Does each generated centroid land near its true real gene in the NEW layouts? 2D rank among all real genes."""
    d = np.load(f"{OUT}/coords.npz", allow_pickle=True)
    genes = list(d["genes"]); real_names = list(d["real_names"])
    ridx = {n: i for i, n in enumerate(real_names)}
    for lay in ("umap", "phate"):
        R = d[f"real_{lay}"]; Gc = d[f"gen_{lay}"]
        ranks = []
        for k, g in enumerate(genes):
            if g not in ridx:
                continue
            order = np.argsort(np.linalg.norm(R - Gc[k], axis=1))
            ranks.append(int(np.where(order == ridx[g])[0][0]) + 1)
        ranks = np.array(ranks)
        print(f"[sanity] {lay} α={float(d['alpha']):+.1f}: 2D rank-to-true median {np.median(ranks):.0f}  "
              f"top1 {(ranks == 1).mean():.0%}  top20 {(ranks <= 20).mean():.0%}  (N={len(ranks)})")


TRAJ = f"{OUT}/coords_traj.npz"


def build_traj():
    """Per-α coords for the trajectory figure. UMAP: fit real + transform every (class,α) (real fixed).
    Also caches generated PC per (class,α) so PHATE can be joint-fit on real + plotted subset at plot time."""
    import umap
    os.makedirs(OUT, exist_ok=True)
    a, comp, mean = gp._load_embedding()
    Xr = np.asarray(a.obsm["X_pca"], np.float64); real_names = list(a.obs_names)
    files = gp._cache_files(); mu, sd = gp._baseline(files)
    alphas = np.load(files[0], allow_pickle=True)["alphas"][gp.POS_AIS].astype(float)
    idx = {n: i for i, n in enumerate(real_names)}
    genes, PC = [], []
    for f in files:
        d = np.load(f, allow_pickle=True); g = str(d["gene"])
        if g not in idx:
            continue
        rows, ok = [], True
        for ai in gp.POS_AIS:
            gv = d["gen"][ai]
            if gv is None or not len(gv):
                ok = False; break
            rows.append(gp._project(np.asarray(gv, np.float64), mu, sd, comp, mean).mean(0))
        if ok:
            genes.append(g); PC.append(np.stack(rows))
    PC = np.stack(PC); nC, nA = PC.shape[:2]
    ur = umap.UMAP(n_neighbors=8, min_dist=0.25, metric="cosine", random_state=1)
    real_umap = ur.fit_transform(Xr)
    gen_umap = ur.transform(PC.reshape(nC * nA, -1)).reshape(nC, nA, 2)
    np.savez(TRAJ, genes=np.array(genes), alphas=alphas, real_names=np.array(real_names),
             real_umap=real_umap, real_xpca=Xr, gen_umap=gen_umap, gen_pc=PC)
    print(f"[traj] saved {TRAJ}  real {Xr.shape}  gen {PC.shape}", flush=True)


def plot_traj(n=10, rank_max=20):
    """Reproduce the traversal_best figure using the proper-transform placement (UMAP transform; PHATE joint
    on real + the plotted genes only). NEW files: traversal_best_{umap,phate}_transform.*"""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    from scipy.spatial.distance import cdist
    plt.rcParams["pdf.fonttype"] = 42
    a, _, _ = gp._load_embedding()
    ec, cxcmap = gp._bg_colors(a)
    flips = gp._ntc_flip(a)
    d = np.load(TRAJ, allow_pickle=True)
    genes = list(d["genes"]); real_names = list(d["real_names"]); alphas = d["alphas"]
    ridx = {nm: i for i, nm in enumerate(real_names)}
    tr = np.array([ridx[g] for g in genes])
    norm = Normalize(vmin=alphas.min(), vmax=alphas.max())
    g2c = a.obs["ebi_complex"].dropna().to_dict()
    ai3 = int(np.argmin(np.abs(alphas - 3.0)))
    Xr = d["real_xpca"]; PCg = d["gen_pc"]
    # PC-cosine rank gate (class-blind, in 101-D)
    pcrank = np.array([int(np.where(np.argsort(cdist(PCg[k, ai3][None], Xr, "cosine")[0]) == tr[k])[0][0]) + 1
                       for k in range(len(genes))])

    for layout in ("umap", "phate"):
        fx, fy, ntc = flips[layout]
        if layout == "umap":
            R = d["real_umap"]; G = d["gen_umap"]
        else:
            import phate
            # joint PHATE on real + generated selected later; first need a layout for selection → do full joint
            pj = phate.PHATE(knn=8, decay=10, n_components=2, random_state=1, n_jobs=-1, verbose=False)
            emb = pj.fit_transform(np.vstack([Xr, PCg.reshape(len(genes) * len(alphas), -1)]))
            R = emb[:len(Xr)]; G = emb[len(Xr):].reshape(len(genes), len(alphas), 2)
        Rf = R * np.array([fx, fy]); Gf = G * np.array([fx, fy])
        tgt = Rf[tr]                                                      # true-gene 2-D per class
        reach = np.min(np.linalg.norm(Gf - tgt[:, None, :], axis=2), axis=1)    # closest approach over α
        ntc_c = ntc.mean(0); journey = np.linalg.norm(tgt - ntc_c, axis=1)
        ok = (pcrank <= rank_max) & (journey >= np.median(journey))
        cand = np.where(ok & (reach <= np.quantile(reach[ok], 0.5)))[0]
        pts2 = Gf[cand, ai3]
        seeds = [int(cand[np.argmin(reach[cand])])]
        order = gp._fps(pts2, n, [list(cand).index(seeds[0])])
        pick = [int(cand[i]) for i in order]

        fig, ax = plt.subplots(figsize=(12, 10))
        gp._draw_bg(ax, a, layout, fx, fy, ec, cxcmap); gp._draw_ntc(ax, ntc)
        sm = ScalarMappable(norm=norm, cmap="viridis_r"); sm.set_array([])
        for k in pick:
            xy = Gf[k]; dd = np.linalg.norm(xy - tgt[k], axis=1); b = int(dd.argmin())
            xs, ys, al = xy[:b + 1, 0], xy[:b + 1, 1], alphas[:b + 1]
            if len(xs) >= 2:
                seg = np.concatenate([np.c_[xs, ys][:-1, None], np.c_[xs, ys][1:, None]], axis=1)
                col = plt.get_cmap("viridis_r")(norm((al[:-1] + al[1:]) / 2)); col[:, 3] = np.linspace(0.55, 1, len(col))
                ax.add_collection(LineCollection(seg, colors=col, lw=2.6, zorder=3))
            ax.plot([xs[-1], tgt[k, 0]], [ys[-1], tgt[k, 1]], ls=":", color="#444", lw=1.3, zorder=4)
            ax.scatter([xs[-1]], [ys[-1]], s=85, c=[al[-1]], cmap="viridis_r", norm=norm, marker="o", edgecolor="k", lw=.6, zorder=5)
            ax.scatter(tgt[k, 0], tgt[k, 1], s=200, marker="*", facecolor="none", edgecolor="k", lw=2.2, zorder=6)
            ax.annotate(genes[k], (tgt[k, 0], tgt[k, 1]), textcoords="offset points", xytext=(7, 5),
                        fontsize=11, fontweight="bold", zorder=9)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(f"Generated geneKO trajectories — {layout.upper()} via "
                     f"{'UMAP .transform' if layout=='umap' else 'joint PHATE'} (proper out-of-sample, real-pop std)\n"
                     f"line color = α   ● closest-approach  ···→ residual gap   ★ true gene   ✕ real NTC", fontsize=11.5)
        cb = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.02); cb.set_label("traversal α")
        fig.tight_layout()
        for e in ("png", "svg"):
            fig.savefig(f"{OUT}/traversal_best_{layout}_transform.{e}", dpi=150, bbox_inches="tight")
        plt.close(fig); print(f"saved traversal_best_{layout}_transform: {[genes[k] for k in pick]}")


def submit():
    from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs
    submit_parallel_jobs([{"name": "gen_embed_refit", "func": build, "kwargs": {}}],
                         experiment="gen_embed_refit",
                         slurm_params={"slurm_partition": "cpu", "cpus_per_task": 16, "mem_gb": 64,
                                       "timeout_min": 120}, log_dir="gen_embed_refit", wait_for_completion=False)


if __name__ == "__main__":
    if "--sanity" in sys.argv:
        _sanity()
    elif "--traj" in sys.argv:
        if not os.path.exists(TRAJ):
            build_traj()
        plot_traj()
    elif "--submit" in sys.argv:
        submit()
    else:
        build()
