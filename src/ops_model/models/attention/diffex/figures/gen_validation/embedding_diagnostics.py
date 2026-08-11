"""Diagnostic plots comparing how generated centroids get PLACED in the real phase embedding, plus the
real-NTC vs inverse-α=0 domain-gap. Reads only already-computed artifacts; writes to figure4_embedding/diagnostics/.

Approaches compared (all at α≈3):
  - kNN landmark (self-std)   : cosine-NN interpolation onto the real manifold (gen_phate_passthrough default)
  - kNN landmark (real-std)   : same, real-population standardization
  - UMAP .transform (real-std): proper out-of-sample projection (gen_embed_refit)
  - PHATE joint (real-std)    : joint fit_transform (gen_embed_refit)
Metric: 2-D rank-to-true = rank of a class's true real gene among all 1052 genes by 2-D distance to its
generated dot (class-blind). Low = generated lands on its own gene; ~random (≈480) = lands in a blob.
"""
import os
import numpy as np
import pandas as pd

B = "/hpc/projects/icd.fast.ops/analysis/figure4_embedding"
GP = f"{B}/gen_passthrough"
RF = f"{B}/gen_passthrough_refit/coords.npz"
GAP = f"{B}/ntc_inverse_gap/emb_5xUPRE.npz"
OUT = f"{B}/diagnostics"
AI = 14        # α=3 parquet


def _ranks(gen_xy, real_xy, true_row):
    """2-D rank of each class's true gene among all real genes (Euclidean)."""
    from scipy.spatial.distance import cdist
    D = cdist(gen_xy, real_xy)
    order = np.argsort(D, axis=1)
    return np.array([int(np.where(order[k] == true_row[k])[0][0]) + 1 for k in range(len(gen_xy))])


def _landmark(parq):
    """From a proj parquet: gen 2-D (gu), real-gene 2-D set (unique ru per gene), and true-row index."""
    df = pd.read_parquet(parq)
    real_xy = df[["ru0", "ru1"]].values                       # one row per gene = its true coord
    gen_xy = df[["gu0", "gu1"]].values
    return gen_xy, real_xy, np.arange(len(df)), list(df["gene"])


def _refit(layout):
    d = np.load(RF, allow_pickle=True)
    genes = list(d["genes"]); real_names = list(d["real_names"]); ridx = {n: i for i, n in enumerate(real_names)}
    R = d[f"real_{layout}"]; G = d[f"gen_{layout}"]
    true_row = np.array([ridx[g] for g in genes])
    return G, R, true_row, genes


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams["pdf.fonttype"] = 42
    os.makedirs(OUT, exist_ok=True)

    approaches = []
    gx, rx, tr, _ = _landmark(f"{GP}/proj_a{AI}.parquet");        approaches.append(("kNN landmark (self-std)", gx, rx, tr))
    gx, rx, tr, _ = _landmark(f"{GP}/proj_real_a{AI}.parquet");   approaches.append(("kNN landmark (real-std)", gx, rx, tr))
    gx, rx, tr, _ = _refit("umap");                              approaches.append(("UMAP .transform (real-std)", gx, rx, tr))
    gx, rx, tr, _ = _refit("phate");                             approaches.append(("PHATE joint (real-std)", gx, rx, tr))

    # ---- Fig A: placement panels (real grey + generated colored by 2-D rank-to-true + connector to true gene) ----
    fig, axes = plt.subplots(1, 4, figsize=(22, 6))
    stats = []
    for ax, (name, G, R, tr) in zip(axes, approaches):
        rk = _ranks(G, R, tr)
        stats.append((name, rk))
        ax.scatter(R[:, 0], R[:, 1], s=6, c="#dddddd", lw=0, zorder=1)
        for k in range(len(G)):
            ax.plot([G[k, 0], R[tr[k], 0]], [G[k, 1], R[tr[k], 1]], "-", color="#888", lw=0.3, alpha=0.08, zorder=2)
        sc = ax.scatter(G[:, 0], G[:, 1], s=10, c=np.log10(rk), cmap="viridis_r", lw=0, zorder=3)
        ax.set_title(f"{name}\nmedian rank {np.median(rk):.0f} · top-20 {(rk<=20).mean():.0%}", fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    cb = fig.colorbar(sc, ax=axes, fraction=0.012, pad=0.01); cb.set_label("log10 rank-to-true (lower = on its own gene)")
    fig.suptitle("Where generated centroids (α≈3) land, by placement method — grey = real genes, lines = gap to true gene",
                 fontweight="bold", fontsize=13)
    for e in ("png", "svg"):
        fig.savefig(f"{OUT}/placement_methods.{e}", dpi=140, bbox_inches="tight")
    plt.close(fig); print("saved placement_methods")

    # ---- Fig B: rank-to-true ECDF (cumulative % of classes within rank X) ----
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for name, rk in stats:
        xs = np.sort(rk); ys = np.arange(1, len(xs) + 1) / len(xs) * 100
        ax.plot(xs, ys, lw=2.2, label=f"{name}  (med {np.median(rk):.0f})")
    ax.axvline(20, color="#c0392b", ls=":", lw=1.2, label="rank-20")
    ax.set_xscale("log"); ax.set_xlabel("2-D rank-to-true gene (log)"); ax.set_ylabel("% of classes ≤ rank")
    ax.set_title("How close generated dots land to their true gene (2-D)", fontweight="bold")
    ax.grid(alpha=.25); ax.legend(fontsize=9, loc="lower right")
    for e in ("png", "svg"):
        fig.savefig(f"{OUT}/rank_ecdf.{e}", dpi=150, bbox_inches="tight")
    plt.close(fig); print("saved rank_ecdf")

    _plot_gap()


def _plot_gap():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from numpy.linalg import norm
    if not os.path.exists(GAP):
        print("no NTC-gap npz; skipping"); return
    d = np.load(GAP, allow_pickle=True)
    R, G = d["R"].astype(np.float64), d["G"].astype(np.float64)
    ch = str(d["channel"]) if "channel" in d else "5xUPRE"
    cos = np.array([float(R[i] @ G[i] / (norm(R[i]) * norm(G[i]))) for i in range(len(R))])
    # PCA of pooled R+G (2-D) to show overlap
    X = np.vstack([R, G]); Xc = X - X.mean(0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False); P = Xc @ Vt[:2].T
    Pr, Pg = P[:len(R)], P[len(R):]

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].hist(cos, bins=20, color="#7fbf9a", edgecolor="k")
    ax[0].axvline(cos.mean(), color="k", lw=2, label=f"mean {cos.mean():.3f}")
    ax[0].set_xlabel("per-cell cosine(real, inverse-α=0)"); ax[0].set_ylabel("count")
    ax[0].set_title(f"Real NTC vs inverse-α=0 — same-cell fidelity ({ch})", fontweight="bold"); ax[0].legend()
    for i in range(len(Pr)):
        ax[1].plot([Pr[i, 0], Pg[i, 0]], [Pr[i, 1], Pg[i, 1]], "-", color="#bbb", lw=0.6, zorder=1)
    ax[1].scatter(Pr[:, 0], Pr[:, 1], s=45, c="#8fa9c9", edgecolor="k", lw=.4, label="real NTC", zorder=3)
    ax[1].scatter(Pg[:, 0], Pg[:, 1], s=45, c="#c0666b", edgecolor="k", lw=.4, label="inverse α=0", zorder=3)
    ax[1].set_title("CellDINO PCA (paired, same cell)", fontweight="bold"); ax[1].legend()
    ax[1].set_xticks([]); ax[1].set_yticks([])
    fig.tight_layout()
    for e in ("png", "svg"):
        fig.savefig(f"{OUT}/ntc_inverse_gap.{e}", dpi=150, bbox_inches="tight")
    plt.close(fig); print("saved ntc_inverse_gap")


if __name__ == "__main__":
    plot()
