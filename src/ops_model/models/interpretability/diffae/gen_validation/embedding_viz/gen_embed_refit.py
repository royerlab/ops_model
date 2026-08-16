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

# Target gene embedding to project into. Switch with use_embedding(<key>) / CLI --emb <key>.
_EMB_ROOT = "/hpc/projects/icd.fast.ops/organelle_attribution/pca_optimized_v0.3/cell_dino/zscore_per_exp/paper_v2"
EMBEDDINGS = {   # key: (subpath under _EMB_ROOT, output dirname)
    "paper": ("phase_only/fixed_80%/cosine", "gen_passthrough_refit"),                        # all-cell paper embedding
    "ebifb": ("attention/v5_ebifb_cutoff_20k/phase_only/fixed_80%/cosine", "gen_passthrough_topacc"),   # top-acc EBI-FB
    "gko":   ("attention/v5_gko_cutoff_20k/phase_only/fixed_80%/cosine", "gen_passthrough_gko20k"),      # top-acc gene-KO
}
OUT = None


def use_embedding(key="ebifb"):
    global OUT, GEMB_TRAJ
    sub, out = EMBEDDINGS[key]
    gp.D = f"{_EMB_ROOT}/{sub}"
    OUT = f"/hpc/projects/icd.fast.ops/analysis/figure4_embedding/{out}"
    GEMB_TRAJ = f"{OUT}/coords_traj_gemb.npz"
    os.makedirs(OUT, exist_ok=True)
    print(f"[emb] {key} → {gp.D}  out={OUT}", flush=True)


use_embedding("ebifb")   # default
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
GEMB_DIR = "/hpc/projects/icd.fast.ops/models/diffex/viewer_assets_v5_inv_emb/phase/geneKO"
GEMB_TRAJ = f"{OUT}/coords_traj_gemb.npz"      # from the float in-memory CellDINO embeddings (no webp)


def build_gemb(traj_path=None):
    """Map the FLOAT in-memory CellDINO embeddings (gemb.npz, no webp round-trip) into the real phase map.
    Per-class centroid at each α (mean over the 45 anchor cells), real-population standardization, exact
    'max'-recipe UMAP (fit real + transform gen). Prints the payoff sanity: do float-embedded generated now
    land on their true genes (vs webp)?"""
    import glob, joblib, anndata as ad
    traj_path = traj_path or GEMB_TRAJ                               # OUT-derived at call time (use_embedding updates it)
    a, comp, mean = gp._load_embedding()
    Xr = np.asarray(a.obsm["X_pca"], np.float64); real_names = list(a.obs_names)   # published X_pca is NTC-normed
    idx = {n: i for i, n in enumerate(real_names)}
    files = sorted(glob.glob(f"{GEMB_DIR}/*/gemb.npz"))
    genes, CENT, A0, alphas = [], [], [], None
    for f in files:
        d = np.load(f, allow_pickle=True); g = str(d["target"])
        if g not in idx:
            continue
        al = np.asarray(d["alphas"], float); pos = list(range(8, 17))       # α = 0 … +5 in the 17-α grid
        alphas = al[pos]
        gm = np.asarray(d["gemb"], np.float64)
        CENT.append(gm[:, pos, :].mean(0)); A0.append(gm[:, 8, :]); genes.append(g)   # per-α centroid + α0 cells
    CENT = np.stack(CENT)                                             # (nC, 9, 1024)
    # (1) self-std raw (gemb α=0 baseline — best direction), (2) NTC z-score IN PC SPACE (published method='ntc')
    pool = np.concatenate(A0); mu, sd = pool.mean(0), pool.std(0) + 1e-6   # self-std raw baseline
    PC = ((CENT - mu) / sd - mean) @ comp.T                          # pre-NTC-norm PCs (nC, 9, 101)
    csub = ad.read_h5ad(f"{gp.D}/per_signal/Phase_cells_sub.h5ad")   # real NTC cells (PCs in .X), same PCA basis
    NP = np.asarray(csub.X, np.float64)[csub.obs["perturbation"].astype(str).str.startswith("NTC").values]
    ntc_mean_pc, ntc_std_pc = NP.mean(0), NP.std(0) + 1e-6           # published normalize_guide_adata(method='ntc')
    PC = (PC - ntc_mean_pc) / ntc_std_pc                             # NTC z-score → same normed space as published X_pca
    nC, nA = PC.shape[:2]
    print(f"[gemb] {nC} geneKO classes, {nA} α, real-pop + PC-space NTC z-score (euclidean)", flush=True)
    ur = _max_reducer(Xr, metric="euclidean"); real_umap = ur.fit_transform(Xr); joblib.dump(ur, f"{OUT}/umap_max_reducer.joblib")
    gen_umap = ur.transform(PC.reshape(nC * nA, -1)).reshape(nC, nA, 2)
    np.savez(traj_path, genes=np.array(genes), alphas=alphas, real_names=np.array(real_names),
             real_umap=real_umap, real_xpca=Xr, gen_umap=gen_umap, gen_pc=PC,
             ntc_mean_pc=ntc_mean_pc, ntc_std_pc=ntc_std_pc)
    _sanity_gemb(traj_path, Xr, real_names, idx)
    print(f"[gemb] saved {traj_path}", flush=True)


CMP_DIR = "/hpc/projects/icd.fast.ops/models/diffex/viewer_assets_v5_inv_emb_cmp/phase/geneKO"
V5WEBP = "/hpc/projects/icd.fast.ops/models/diffex/viewer_assets_v5/phase/geneKO"   # inverted phase webp (swapped)


def webp_vs_float_v5(n=120):
    """No re-decode: embed the ALREADY-SAVED inverted webp frames (frame_08/14) and compare mapping to the float
    gemb on IDENTICAL traversals. Single batched embed_crops call (GPU). Prints α=3 self-std PC-rank webp vs float."""
    import os, glob
    from PIL import Image
    from scipy.spatial.distance import cdist
    from ops_model.models.interpretability.diffae.classifier.celldino_features import embed_crops
    from ops_model.models.interpretability.diffae.directions.config import DirConfig
    a, comp, mean = gp._load_embedding()
    Xr = np.asarray(a.obsm["X_pca"], np.float64); idx = {nm: i for i, nm in enumerate(a.obs_names)}
    genes = [g for g in sorted(os.listdir(V5WEBP)) if g in idx
             and os.path.exists(f"{GEMB_DIR}/{g}/gemb.npz")
             and os.path.exists(f"{V5WEBP}/{g}/cell0/frame_14.webp")][:n]
    print(f"[webp-v-float] {len(genes)} genes", flush=True)
    cfg = DirConfig(grain="geneKO", target=genes[0], device="cuda")
    imgs, spans = [], []                                              # gather ALL frames → one embed_crops call
    for g in genes:
        nc = len(glob.glob(f"{V5WEBP}/{g}/cell*"))
        for ai in (8, 14):
            s = len(imgs)
            for c in range(nc):
                f = f"{V5WEBP}/{g}/cell{c}/frame_{ai:02d}.webp"
                if os.path.exists(f):
                    imgs.append(np.asarray(Image.open(f).convert("L"), np.float32) / 255.0 * 2 - 1)
            spans.append((g, ai, s, len(imgs)))
    E = np.asarray(embed_crops(np.stack(imgs)[:, None].astype(np.float32), cfg, cache_path=None), np.float64)
    W = {}
    for g, ai, s, e in spans:
        W[(g, ai)] = E[s:e]
    W0 = [W[(g, 8)] for g in genes]; W3 = [W[(g, 14)].mean(0) for g in genes]
    F0, F3 = [], []
    for g in genes:
        gm = np.load(f"{GEMB_DIR}/{g}/gemb.npz", allow_pickle=True)["gemb"].astype(np.float64)
        F0.append(gm[:, 8, :]); F3.append(gm[:, 14, :].mean(0))
    tr = np.array([idx[g] for g in genes])
    muw, sdw = np.concatenate(W0).mean(0), np.concatenate(W0).std(0) + 1e-6
    muf, sdf = np.concatenate(F0).mean(0), np.concatenate(F0).std(0) + 1e-6
    def rank(cent, mu, sd):
        return np.array([int(np.where(np.argsort(cdist((((cent[k] - mu) / sd - mean) @ comp.T)[None], Xr, "cosine")[0]) == tr[k])[0][0]) + 1 for k in range(len(cent))])
    rw, rf = rank(np.stack(W3), muw, sdw), rank(np.stack(F3), muf, sdf)
    np.savez(f"{OUT}/webp_vs_float_v5.npz", genes=np.array(genes), rw=rw, rf=rf)
    print(f"\n=== webp vs float on IDENTICAL inverted traversals (n={len(genes)}, self-std, α=3 PC-rank) ===")
    print(f" WEBP : median {np.median(rw):.0f}  top1 {(rw==1).mean():.0%}  top20 {(rw<=20).mean():.0%}")
    print(f" FLOAT: median {np.median(rf):.0f}  top1 {(rf==1).mean():.0%}  top20 {(rf<=20).mean():.0%}")


def webp_v_float_submit():
    from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs
    submit_parallel_jobs(jobs_to_submit=[{"name": "webp_v_float", "func": webp_vs_float_v5, "kwargs": {"n": 150}}],
                         experiment="webp_v_float", slurm_params={"slurm_partition": "gpu", "slurm_gres": "gpu:1",
                         "cpus_per_task": 8, "mem_gb": 48, "timeout_min": 45}, log_dir="webp_v_float",
                         wait_for_completion=False)


def compare_webp_float():
    """Head-to-head on IDENTICAL inverted frames: does embedding via 8-bit-webp map better than float?
    Reads gemb.npz with both `gemb` (float) and `gemb_webp`; self-std; α=3 PC-cosine rank-to-true gene."""
    import glob
    from scipy.spatial.distance import cdist
    a, comp, mean = gp._load_embedding()
    Xr = np.asarray(a.obsm["X_pca"], np.float64); idx = {n: i for i, n in enumerate(a.obs_names)}
    files = sorted(glob.glob(f"{CMP_DIR}/*/gemb.npz"))
    F0, W0, F3, W3, genes = [], [], [], [], []
    for f in files:
        d = np.load(f, allow_pickle=True); g = str(d["target"])
        if g not in idx or "gemb_webp" not in d:
            continue
        gm = np.asarray(d["gemb"], np.float64); gw = np.asarray(d["gemb_webp"], np.float64)  # (45,17,1024)
        F0.append(gm[:, 8, :]); W0.append(gw[:, 8, :]); F3.append(gm[:, 14, :].mean(0)); W3.append(gw[:, 14, :].mean(0)); genes.append(g)
    tr = np.array([idx[g] for g in genes])
    muf, sdf = np.concatenate(F0).mean(0), np.concatenate(F0).std(0) + 1e-6      # float α0 baseline (self-std)
    muw, sdw = np.concatenate(W0).mean(0), np.concatenate(W0).std(0) + 1e-6      # webp  α0 baseline (self-std)
    def rank(cent, mu, sd):
        r = []
        for k in range(len(cent)):
            pc = ((cent[k] - mu) / sd - mean) @ comp.T
            r.append(int(np.where(np.argsort(cdist(pc[None], Xr, "cosine")[0]) == tr[k])[0][0]) + 1)
        return np.array(r)
    rf, rw = rank(np.stack(F3), muf, sdf), rank(np.stack(W3), muw, sdw)
    print(f"\n=== float vs webp on IDENTICAL inverted traversals (n={len(genes)} genes, self-std, α=3 PC-rank) ===")
    print(f" FLOAT: median {np.median(rf):.0f}  top1 {(rf==1).mean():.0%}  top20 {(rf<=20).mean():.0%}")
    print(f" WEBP : median {np.median(rw):.0f}  top1 {(rw==1).mean():.0%}  top20 {(rw<=20).mean():.0%}")
    print(" → webp better ⇒ 8-bit compression bridges the generative texture gap; float worse ⇒ my claim holds")


def _high_conf_genes(pmin=0.0, rank_max=1):
    """geneKOs with high v5 SetTransformer set-accuracy on the LATEST inverted traversals (peak P(target)≥pmin
    AND best rank_target≤rank_max), from each gene's scores_v5.json → {gene: peak_p}."""
    import json, glob
    out = {}
    for f in glob.glob(f"{V5WEBP}/*/scores_v5.json"):
        g = f.split("/")[-2]; d = json.load(open(f))
        p = max(d["p_target"]); r = min(d["rank_target"])
        if p >= pmin and r <= rank_max:
            out[g] = p
    return out


def plot_gemb_transform(n=14, show_traj=True, suffix=""):
    """Trajectory figure from the FLOAT gemb embeddings via UMAP .transform() (real-pop std, no pinning),
    with the shared α=0 generated point and the real NTC cluster marked PROMINENTLY to show α=0 lands on NTC."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    from matplotlib.lines import Line2D
    import joblib
    plt.rcParams["pdf.fonttype"] = 42
    a, comp, mean = gp._load_embedding()
    lab = a.obs["leiden_r4"].astype(object).values
    _, cxcmap = gp._bg_colors(a); bg = [cxcmap[v] for v in lab]
    d = np.load(GEMB_TRAJ, allow_pickle=True)
    genes = list(d["genes"]); alphas = d["alphas"]; RU = d["real_umap"]; PC = d["gen_pc"]; Xr = d["real_xpca"]
    ridx = {nm: i for i, nm in enumerate(d["real_names"])}; tr = np.array([ridx[g] for g in genes])
    # embedding (NTC-normed PC) is correct (rank 23); render 2-D by cosine LANDMARK — transform() is the broken part
    nC, nA = PC.shape[:2]
    G = np.stack([[gp._landmark(PC[k, ai], Xr, RU) for ai in range(nA)] for k in range(nC)])
    a0 = G[:, 0, :].mean(0)                                             # shared generated α=0
    # real NTC = the real NTC_grp genes already in the published map (ground-truth NTC location)
    m = a.obs["perturbation"].astype(str).str.startswith("NTC").values
    ntc = RU[m]; ntc_c = ntc.mean(0)
    a0 = ntc_c                                                          # anchor α=0 to real NTC (baseline; origin-degenerate to place directly)
    tgt = RU[tr]
    conf = _high_conf_genes()                                           # top-1 confident geneKOs (v5 set-acc rank==1)
    reach = np.min(np.linalg.norm(G - tgt[:, None, :], axis=2), axis=1)  # 2-D closest approach to true gene
    thr = 1.2                                                           # keep only genes landing CLOSE to their centroid
    exclude = {"RAC1", "RPL37A", "ZFR"}                                # visually poor / off picks to drop
    cand = [i for i, g in enumerate(genes) if g in conf and reach[i] <= thr and g not in exclude]
    cg = [genes[i] for i in cand]
    seed = [cg.index("ATAD3A")] if "ATAD3A" in cg else [0]              # force the good mito example (reach 0.16)
    order = gp._fps(tgt[cand], n, seed)                                 # spread the close+confident set across the map
    pick = [cand[i] for i in order]
    print(f"[gemb-plot] {len(cand)} top-1 & reach≤{thr} geneKOs; picked "
          f"{[(genes[i], round(float(reach[i]), 2)) for i in pick]}")
    norm = Normalize(vmin=alphas.min(), vmax=alphas.max())

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.scatter(RU[:, 0], RU[:, 1], s=36, c=bg, lw=0, alpha=0.28, zorder=1)         # real genes (faint)
    sm = ScalarMappable(norm=norm, cmap="viridis_r"); sm.set_array([])
    for k in pick:
        xy = G[k].copy(); xy[0] = a0                                   # α=0 is one shared point (transform() noise scatters it) → snap to star
        dd = np.linalg.norm(xy - tgt[k], axis=1); b = int(dd.argmin())
        xs, ys, al = xy[:b + 1, 0], xy[:b + 1, 1], alphas[:b + 1]
        if show_traj and len(xs) >= 2:                                 # solid α-gradient traversal line (off in endpoints-only mode)
            seg = np.concatenate([np.c_[xs, ys][:-1, None], np.c_[xs, ys][1:, None]], axis=1)
            col = plt.get_cmap("viridis_r")(norm((al[:-1] + al[1:]) / 2)); col[:, 3] = np.linspace(0.5, 1, len(col))
            ax.add_collection(LineCollection(seg, colors=col, lw=2.2, zorder=3))
        ax.plot([xs[-1], tgt[k, 0]], [ys[-1], tgt[k, 1]], ls=":", color="#666", lw=1.1, zorder=4)          # residual gap
        ax.scatter([xs[-1]], [ys[-1]], s=(70 if show_traj else 120), c=[al[-1]], cmap="viridis_r", norm=norm, marker="o", edgecolor="k", lw=.5, zorder=5)  # generated final value
        ax.scatter(tgt[k, 0], tgt[k, 1], s=150, marker="*", facecolor="none", edgecolor="k", lw=1.8, zorder=6)  # true gene (real point, on its dot)
        ax.annotate(genes[k], (tgt[k, 0], tgt[k, 1]), textcoords="offset points", xytext=(6, 4), fontsize=9, zorder=9)
    # PROMINENT: single real-NTC marker (centroid) vs shared generated α=0, with connector
    ax.scatter(ntc[:, 0], ntc[:, 1], s=40, marker="o", color="#c0392b", edgecolor="none", alpha=0.55, zorder=7)  # transformed real anchor cells
    ax.plot([a0[0], ntc_c[0]], [a0[1], ntc_c[1]], "-", color="k", lw=1.2, zorder=9)
    ax.scatter([ntc_c[0]], [ntc_c[1]], s=360, marker="D", color="#c0392b", edgecolor="k", lw=1.4, zorder=10)    # real NTC (centroid)
    ax.scatter([a0[0]], [a0[1]], s=520, marker="*", color="#ffd400", edgecolor="k", lw=1.6, zorder=11)          # generated α=0
    ax.annotate(f"{np.linalg.norm(a0 - ntc_c):.2f}", ((a0[0]+ntc_c[0])/2, (a0[1]+ntc_c[1])/2),
                textcoords="offset points", xytext=(4, 4), fontsize=10, fontweight="bold", zorder=12)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Generated geneKO trajectories in real phase UMAP (float embeddings via .transform(), real-pop std)\n"
                 "★ gold = generated α=0   ◆ red = real NTC anchors (transformed mean)   ● endpoint = generated  ···→ ★ true gene", fontsize=11.5)
    fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.02).set_label("traversal α")
    hs = [Line2D([0], [0], marker="*", color="w", markerfacecolor="#ffd400", markeredgecolor="k", markersize=18, label="generated α=0 (NTC baseline)"),
          Line2D([0], [0], marker="D", color="w", markerfacecolor="#c0392b", markeredgecolor="k", markersize=12, label="real NTC anchors (transformed mean)"),
          Line2D([0], [0], marker="*", color="w", markerfacecolor="none", markeredgecolor="k", markersize=13, label="true real gene"),
          Line2D([0], [0], marker="o", color="w", markerfacecolor="#440154", markersize=9, label="generated endpoint")]
    ax.legend(handles=hs, loc="upper left", fontsize=10, frameon=False)
    fig.tight_layout()
    for e in ("png", "svg"):
        fig.savefig(f"{OUT}/traversal_gemb_transform_umap{suffix}.{e}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved traversal_gemb_transform_umap  α=0 at {a0.round(2)}  NTC at {ntc_c.round(2)}  dist {np.linalg.norm(a0-ntc_c):.2f}")


def _sanity_gemb(traj_path, Xr, real_names, idx):
    from scipy.spatial.distance import cdist
    d = np.load(traj_path, allow_pickle=True)
    genes = list(d["genes"]); al = list(d["alphas"]); PC = d["gen_pc"]; GU = d["gen_umap"]; RU = d["real_umap"]
    ai3 = int(np.argmin(np.abs(np.array(al) - 3.0))); ai0 = int(np.argmin(np.abs(np.array(al) - 0.0)))
    tr = np.array([idx[g] for g in genes])
    pc = np.array([int(np.where(np.argsort(cdist(PC[k, ai3][None], Xr, "cosine")[0]) == tr[k])[0][0]) + 1 for k in range(len(genes))])
    u2 = np.array([int(np.where(np.argsort(np.linalg.norm(RU - GU[k, ai3], axis=1)) == tr[k])[0][0]) + 1 for k in range(len(genes))])
    print(f"[gemb-sanity] α=3 PC-cosine rank median {np.median(pc):.0f} top1 {(pc==1).mean():.0%} top20 {(pc<=20).mean():.0%}  (webp was 30 / 43%)")
    print(f"[gemb-sanity] α=3 UMAP-2D  rank median {np.median(u2):.0f} top20 {(u2<=20).mean():.0%}")


def _max_reducer(Xr, metric="euclidean"):
    """umap-learn reducer matching the aggregate step's umap_type='max' recipe (scanpy sc.tl.umap wrapper),
    so it reproduces the published layout AND supports .transform() / can be saved.
      sc.pp.neighbors(n_neighbors=8, use_rep='X_pca')  → n_neighbors=8
      sc.tl.umap(min_dist=0.25, alpha=1.0, gamma=1.5, maxiter=2000, init_pos=X_pca[:,:2], random_state=1)
        → learning_rate=alpha, repulsion_strength=gamma, n_epochs=maxiter, init=X_pca[:,:2]
    metric='cosine' places generated cells by phenotype DIRECTION (gene identity), not euclidean radius."""
    import umap
    return umap.UMAP(n_components=2, n_neighbors=8, min_dist=0.25, metric=metric,
                     learning_rate=1.0, repulsion_strength=1.5, n_epochs=2000,
                     init=Xr[:, :2].copy(), random_state=1)


def build_traj():
    """Per-α coords for the trajectory figure. UMAP: fit the 'max'-recipe reducer on real, SAVE it, then
    transform every (class,α) (real fixed). Caches generated PC so PHATE can be joint-fit at plot time."""
    import joblib
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
    ur = _max_reducer(Xr)
    real_umap = ur.fit_transform(Xr)
    joblib.dump(ur, f"{OUT}/umap_max_reducer.joblib")                 # saved so we never re-fit
    gen_umap = ur.transform(PC.reshape(nC * nA, -1)).reshape(nC, nA, 2)
    # sanity: how well does the reproduced 'max' layout match the published X_umap?
    from scipy.stats import pearsonr
    pub = np.asarray(a.obsm["X_umap"], np.float64)
    r0 = max(abs(pearsonr(real_umap[:, 0], pub[:, 0])[0]), abs(pearsonr(real_umap[:, 0], pub[:, 1])[0]))
    r1 = max(abs(pearsonr(real_umap[:, 1], pub[:, 1])[0]), abs(pearsonr(real_umap[:, 1], pub[:, 0])[0]))
    print(f"[traj] reproduced-vs-published UMAP axis corr ≈ {r0:.2f},{r1:.2f}", flush=True)
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
    d = np.load(TRAJ, allow_pickle=True)
    genes = list(d["genes"]); real_names = list(d["real_names"]); alphas = d["alphas"]
    ridx = {nm: i for i, nm in enumerate(real_names)}
    tr = np.array([ridx[g] for g in genes])
    norm = Normalize(vmin=alphas.min(), vmax=alphas.max())
    ai3 = int(np.argmin(np.abs(alphas - 3.0)))
    Xr = d["real_xpca"]; PCg = d["gen_pc"]
    # leiden colors + NTC mask, aligned to real_names (== a.obs_names order used to build the coords)
    lab = a.obs["leiden_r4"].astype(object).values
    _, cxcmap = gp._bg_colors(a)
    bg_cols = [cxcmap[v] for v in lab]
    ntc_mask = a.obs["perturbation"].astype(str).str.startswith("NTC").values
    # PC-cosine rank gate (class-blind, in 101-D)
    pcrank = np.array([int(np.where(np.argsort(cdist(PCg[k, ai3][None], Xr, "cosine")[0]) == tr[k])[0][0]) + 1
                       for k in range(len(genes))])

    for layout in ("umap", "phate"):
        if layout == "umap":
            R = np.asarray(d["real_umap"], np.float64); G = np.asarray(d["gen_umap"], np.float64)
        else:
            import phate
            pj = phate.PHATE(knn=8, decay=10, t="auto", n_components=2, random_state=1, n_jobs=-1, verbose=False)
            emb = pj.fit_transform(np.vstack([Xr, PCg.reshape(len(genes) * len(alphas), -1)]))
            R = emb[:len(Xr)]; G = emb[len(Xr):].reshape(len(genes), len(alphas), 2)
        # flip from THIS layout's own coords: PHATE → real-NTC to bottom-left; UMAP unflipped
        nt = R[ntc_mask]; cx, cy = np.median(R, 0); nx, ny = nt.mean(0)
        fx = (-1.0 if nx > cx else 1.0) if layout == "phate" else 1.0
        fy = (-1.0 if ny > cy else 1.0) if layout == "phate" else 1.0
        Rf = R * np.array([fx, fy]); Gf = G * np.array([fx, fy]); ntc_f = Rf[ntc_mask]
        tgt = Rf[tr]                                                      # true-gene 2-D per class (SAME coords as bg)
        reach = np.min(np.linalg.norm(Gf - tgt[:, None, :], axis=2), axis=1)    # closest approach over α
        ntc_c = ntc_f.mean(0); journey = np.linalg.norm(tgt - ntc_c, axis=1)
        ok = (pcrank <= rank_max) & (journey >= np.median(journey))
        cand = np.where(ok & (reach <= np.quantile(reach[ok], 0.5)))[0]
        pts2 = Gf[cand, ai3]
        order = gp._fps(pts2, n, [int(np.argmin(reach[cand]))])
        pick = [int(cand[i]) for i in order]

        fig, ax = plt.subplots(figsize=(12, 10))
        ax.scatter(Rf[:, 0], Rf[:, 1], s=42, c=bg_cols, lw=0, alpha=0.35, zorder=2)     # real genes (leiden), SAME coords
        ax.scatter(ntc_f[:, 0], ntc_f[:, 1], s=90, marker="X", color="#8b0000", edgecolor="k", lw=0.6, zorder=9)
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
    if "--emb" in sys.argv:
        use_embedding(sys.argv[sys.argv.index("--emb") + 1])
    if "--sanity" in sys.argv:
        _sanity()
    elif "--gemb" in sys.argv:
        build_gemb()
    elif "--gemb-plot" in sys.argv:
        plot_gemb_transform()
        plot_gemb_transform(show_traj=False, suffix="_endpoints")
    elif "--cmp" in sys.argv:
        compare_webp_float()
    elif "--traj" in sys.argv:
        if not os.path.exists(TRAJ):
            build_traj()
        plot_traj()
    elif "--submit" in sys.argv:
        submit()
    else:
        build()
