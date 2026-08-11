"""Faithful real class centroids in the GENERATED (embed_crops CellDINO) space.

Re-embed the top-N accuracy real cells per class with the SAME encoder the generated cells use (embed_crops via
_gather_class) and average → a low-variance centroid (fixes v2's noisy 30-cell centroid). Then score the cached
generated cells (nearest faithful centroid) → top-1/top-5 vs α, with the v2 per-domain standardization (gen vs
pooled gen-α0, centroids vs the real population). The training .pt store can't substitute — it's z-scored, a
different space than embed_crops (empirically 5-11% nearest-centroid, no better than raw).
"""
import os, json, glob
import numpy as np

V5 = "/hpc/projects/icd.fast.ops/models/diffex/viewer_assets_v5"
RANK = {"geneKO": f"{V5}/_rankings/pma_v5_phase_geneKO.parquet",
        "complex": f"{V5}/_rankings/pma_v5_phase_complex.parquet"}
CACHE = os.environ.get("GRC_CACHE", "/hpc/projects/icd.fast.ops/analysis/figure4_traversals/gen_real_map_cache")
OUT = os.environ.get("GRC_OUT", "/hpc/projects/icd.fast.ops/analysis/figure4_traversals/gen_real_centroid")
N = 1000                 # top-N real cells per class → faithful centroid
PER_SHARD = 12           # classes per GPU shard (fine → high parallelism)


def _classes(grain):
    # read the ORIGINAL class name from each npz (filenames are slugified; complex names have spaces)
    return sorted(str(np.load(f, allow_pickle=True)["gene"]) for f in glob.glob(f"{CACHE}/{grain}/*.npz"))


def embed_centroids(grain, classes):
    from ops_model.models.interpretability.diffae.traversal.precompute import _gather_class
    from ops_model.models.interpretability.diffae.directions.config import DirConfig
    from ops_model.models.interpretability.diffae.classifier.config import slugify
    cfg = DirConfig(grain=grain, target=classes[0], device="cuda"); cfg.num_workers = 12
    cents, S, SS, n = {}, np.zeros(1024), np.zeros(1024), 0
    embs, lbl = [], []
    for c in classes:
        try:
            _, e = _gather_class(cfg, c, N, parquet=RANK[grain])
            e = np.asarray(e, np.float32)
            cents[c] = e.mean(0).astype(np.float32)
            embs.append(e); lbl += [c] * len(e)                        # keep individual embeddings for the proper-mAP gallery
            S += e.sum(0); SS += (e.astype(np.float64) ** 2).sum(0); n += len(e)
        except Exception as ex:
            print(f"skip {c}: {ex}")
    os.makedirs(f"{OUT}/shards", exist_ok=True)
    names = list(cents)
    np.savez(f"{OUT}/shards/{grain}_{slugify(classes[0])}.npz",
             names=np.array(names), cents=np.stack([cents[k] for k in names]) if names else np.zeros((0, 1024)),
             embs=np.concatenate(embs).astype(np.float32) if embs else np.zeros((0, 1024), np.float32),
             labels=np.array(lbl), S=S, SS=SS, n=n)
    return {"grain": grain, "n": len(cents), "cells": n}


def merge(grain):
    names, C, S, SS, n = [], [], np.zeros(1024), np.zeros(1024), 0
    for f in glob.glob(f"{OUT}/shards/{grain}_*.npz"):
        d = np.load(f, allow_pickle=True)
        if len(d["names"]):
            names += list(d["names"]); C.append(d["cents"])
        S += d["S"]; SS += d["SS"]; n += int(d["n"])
    C = np.concatenate(C); mu = S / n; sd = np.sqrt(np.clip(SS / n - mu ** 2, 1e-12, None)) + 1e-6
    np.savez(f"{OUT}/{grain}_centroids.npz", names=np.array(names), cents=C, mu=mu.astype(np.float32), sd=sd.astype(np.float32))
    return {"grain": grain, "classes": len(names), "cells": n}


def score(grain, cache=None, out=None, cap=None):
    from ops_model.models.interpretability.diffae.classifier.config import slugify
    CACHE_ = cache or CACHE; OUT_ = out or OUT                            # explicit args (SLURM-safe) override module globals
    cap = cap if cap is not None else (int(os.environ.get("GRC_GEN_CAP", "0")) or None)   # subsample gen bag to first `cap` cells/class
    d = np.load(f"{OUT_}/{grain}_centroids.npz", allow_pickle=True)
    names = list(d["names"]); cidx = {slugify(str(c)): i for i, c in enumerate(names)}   # slug-key: cache genes may be slugged
    mu_r, sd_r = d["mu"], d["sd"]
    cz = (d["cents"] - mu_r) / sd_r
    cz = cz / (np.linalg.norm(cz, axis=1, keepdims=True) + 1e-9)
    caches = sorted(glob.glob(f"{CACHE_}/{grain}/*.npz"))
    g0 = []
    for f in caches:
        dd = np.load(f, allow_pickle=True)
        a0i = int(np.argmin(np.abs(np.asarray(dd["alphas"], float))))     # α=0 baseline (grid may be one-sided, not symmetric)
        z0 = dd["gen"][a0i]
        if z0 is not None and len(z0):
            g0.append(np.asarray(z0, np.float32)[:cap])
    g0 = np.concatenate(g0)
    mu_g, sd_g = g0.mean(0), g0.std(0) + 1e-6
    by = {}
    for f in caches:
        dd = np.load(f, allow_pickle=True); g = str(dd["gene"]); al = dd["alphas"]
        if slugify(g) not in cidx:
            continue
        ti = cidx[slugify(g)]
        for ai in range(len(al)):
            gv = dd["gen"][ai]
            if gv is None or not len(gv):
                continue
            gz = (np.asarray(gv, np.float32)[:cap] - mu_g) / sd_g
            gz = gz / (np.linalg.norm(gz, axis=1, keepdims=True) + 1e-9)
            order = np.argsort(-(gz @ cz.T), axis=1)
            rank_true = np.where(order == ti)[1] + 1                 # 1-based rank of the true class centroid per cell
            a = float(al[ai]); by.setdefault(a, {"top1": {}, "top5": {}, "map": {}})
            by[a]["top1"][g] = float(np.mean(order[:, 0] == ti))
            by[a]["top5"][g] = float(np.mean([ti in r[:5] for r in order]))
            by[a]["map"][g] = float(np.mean(1.0 / rank_true))        # retrieval AP (1 positive = true centroid) → mean = mAP
    json.dump({"alphas": sorted(by), "by_alpha": by, "n_classes": len(cidx)},
              open(f"{OUT_}/{grain}_scored.json", "w"))
    return {"grain": grain, "n": len(cidx)}


DIST = "/hpc/projects/icd.fast.ops/analysis/figure4_traversals/gen_real_distinct"   # real distinctiveness@20 per class


def ceiling(grain):
    """Real cells (cached embed_crops) → nearest faithful centroid: per-class real mAP/top1/top5 (the ceiling)."""
    from ops_model.models.interpretability.diffae.classifier.config import slugify
    d = np.load(f"{OUT}/{grain}_centroids.npz", allow_pickle=True)
    names = list(d["names"]); cidx = {slugify(str(c)): i for i, c in enumerate(names)}
    cz = (d["cents"] - d["mu"]) / d["sd"]; cz = cz / (np.linalg.norm(cz, axis=1, keepdims=True) + 1e-9)
    out = {}
    for f in glob.glob(f"{CACHE}/{grain}/*.npz"):
        dd = np.load(f, allow_pickle=True); c = str(dd["gene"])
        if slugify(c) not in cidx:
            continue
        ti = cidx[slugify(c)]; rz = (np.asarray(dd["real"], np.float32) - d["mu"]) / d["sd"]
        rz = rz / (np.linalg.norm(rz, axis=1, keepdims=True) + 1e-9)
        order = np.argsort(-(rz @ cz.T), axis=1); rank = np.where(order == ti)[1] + 1
        out[c] = {"top1": float(np.mean(order[:, 0] == ti)), "top5": float(np.mean([ti in o[:5] for o in order])),
                  "map": float(np.mean(1.0 / rank))}
    json.dump(out, open(f"{OUT}/{grain}_ceiling.json", "w"))
    return {"grain": grain, "n": len(out)}


def _acc_keep(grain, acc_thr):
    """SetTransformer-distinguishable subset: real cells top1_acc > acc_thr @ bag20 (real_acc20.json), slug-keyed."""
    R = json.load(open("/hpc/projects/icd.fast.ops/models/diffex/viewer_assets_v5/real_acc20.json"))
    pre = f"phase/{grain}/"
    return {k[len(pre):] for k, v in R.items() if k.startswith(pre) and v > acc_thr}


def plot(min_dist=None, acc_thr=None, fname="centroid_topk", overlay=None, overlay_label="200-cell bag"):
    """3 scores (mAP, top-1, top-5) vs α from the faithful centroids. min_dist: restrict to classes whose real
    distinctiveness/EBI mAP@20 > min_dist. acc_thr: restrict to real top1_acc>acc_thr @bag20 (SetTransformer
    subset). overlay: {grain: scored_dir} → dashed second line (e.g. the 200-cell bag) on that grain's axis."""
    from ops_model.models.interpretability.diffae.classifier.config import slugify
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["pdf.fonttype"] = 42
    C1, C5, CM = "#c0392b", "#27ae60", "#2471a3"
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, (grain, lbl) in zip(axes, [("geneKO", "Gene-level"), ("complex", "Protein complex")]):
        d = json.load(open(f"{OUT}/{grain}_scored.json")); al = d["alphas"]; ba = d["by_alpha"]
        keep = None
        if acc_thr is not None:
            ks = _acc_keep(grain, acc_thr); keep = {"__slug__"}; keep = ks   # slug-keyed
        elif min_dist is not None:
            rd = json.load(open(f"{DIST}/{grain}_real.json"))
            keep = {c for c, v in rd.items() if v > min_dist}
        ink = lambda c: keep is None or c in keep or slugify(c) in keep
        def vals_of(ba_, a, key):
            return [v for c, v in ba_[str(a)][key].items() if ink(c)]
        vals = lambda a, key: vals_of(ba, a, key)
        n = len(vals(al[0], "top1"))
        sem = lambda a, key: np.std(vals(a, key), ddof=1) / np.sqrt(max(len(vals(a, key)), 1))
        ceil = json.load(open(f"{OUT}/{grain}_ceiling.json")) if os.path.exists(f"{OUT}/{grain}_ceiling.json") else None
        ov = None; n_ov = 0
        if overlay and grain in overlay:
            ov = json.load(open(f"{overlay[grain]}/{grain}_scored.json")); n_ov = len(vals_of(ov["by_alpha"], ov["alphas"][0], "top1"))
        gtag = "" if ov is None else " (45-cell)"
        for key, col, name in [("map", CM, "MRR"), ("top1", C1, "top-1"), ("top5", C5, "top-5")]:
            m = np.array([np.mean(vals(a, key)) for a in al]); se = np.array([sem(a, key) for a in al])
            ax.plot(al, m, "-", color=col, lw=2.6, label=f"{name} — generated{gtag}")
            ax.fill_between(al, m - se, m + se, color=col, alpha=.18, lw=0)   # mean ± SEM across classes
            if ov is not None:                                           # dashed = overlay bag (same centroids, more cells)
                alo = ov["alphas"]; mo = np.array([np.mean(vals_of(ov["by_alpha"], a, key)) for a in alo])
                ax.plot(alo, mo, "--", color=col, lw=2.2, label=f"{name} — {overlay_label}")
            if ceil:                                                     # dotted = real-cell ceiling (same centroids)
                cvs = [v[key] for c, v in ceil.items() if ink(c)]
                cv = np.mean(cvs); cse = np.std(cvs, ddof=1) / np.sqrt(max(len(cvs), 1))
                ax.axhline(cv, color=col, ls=":", lw=2.0, label=f"{name} — real ceiling")
                ax.axhspan(cv - cse, cv + cse, color=col, alpha=.07, lw=0)
        ax.axvline(0, color="#ccc", lw=1); ax.axvline(1, color="#bbb", lw=1, ls="--")
        nlab = f"n={n}" if ov is None else f"n={n} (45c) / {n_ov} (200c)"
        ax.set_title(f"{lbl}  ({nlab})"); ax.set_xlabel("traversal α"); ax.grid(alpha=.25)
    axes[0].set_ylabel(f"generated → nearest real centroid (top-{N} cells)"); axes[0].set_ylim(-0.02, 1.02); axes[0].legend(fontsize=9, ncol=2)
    gate = "" if acc_thr is None else f" · real top1_acc@20 > {acc_thr}"
    if min_dist is not None:
        gate = f" · real distinctiveness@20 > {min_dist}"
    fig.suptitle(f"Generated cells → faithful real centroids (top-{N} real cells/class){gate}", fontweight="bold")
    fig.tight_layout()
    for e in ("png", "svg"):
        fig.savefig(f"{OUT}/{fname}.{e}", dpi=150, bbox_inches="tight")
    plt.close(fig); print(f"saved {fname}")


def merge_embs(grain):
    feats, lbl = [], []
    for f in glob.glob(f"{OUT}/shards/{grain}_*.npz"):
        d = np.load(f, allow_pickle=True)
        if len(d["labels"]):
            feats.append(d["embs"]); lbl += [str(x) for x in d["labels"]]
    feats = np.concatenate(feats).astype(np.float32)
    np.savez(f"{OUT}/{grain}_embs.npz", feats=feats, labels=np.array(lbl))
    return {"grain": grain, "gallery_cells": len(feats), "classes": len(set(lbl))}


def proper_map_gpu(grain):
    """Proper retrieval mAP with the full 1000-cell/class real gallery, on GPU (no copairs, no permutations).
    gen queries → real gallery, AP over same-class positives; ceiling = held-out real queries (self masked)."""
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    E = np.load(f"{OUT}/{grain}_embs.npz", allow_pickle=True)
    cen = np.load(f"{OUT}/{grain}_centroids.npz", allow_pickle=True)
    mu, sd = cen["mu"], cen["sd"]
    labels = E["labels"]; cls = sorted(set(labels)); c2i = {c: i for i, c in enumerate(cls)}
    G = (E["feats"] - mu) / sd
    Gt = torch.tensor(G, device=dev, dtype=torch.float32); Gt = Gt / Gt.norm(dim=1, keepdim=True).clamp_min(1e-9)
    glab = torch.tensor([c2i[c] for c in labels], device=dev)

    def ap(Q, ti, self_idx=None):                        # Q:(b,D) normalized, ti:(b,) class idx; self_idx:(b,) gallery row to mask
        out = []
        for s in range(0, len(Q), 256):
            q = Q[s:s + 256]; sim = q @ Gt.T
            if self_idx is not None:
                sim[torch.arange(len(q)), self_idx[s:s + 256]] = -1e9
            order = torch.argsort(sim, dim=1, descending=True)
            pos = (glab[order] == ti[s:s + 256, None]).float()
            csum = torch.cumsum(pos, 1); ranks = torch.arange(1, pos.shape[1] + 1, device=dev).float()
            a = ((csum / ranks) * pos).sum(1) / pos.sum(1).clamp_min(1)
            out.append(a.cpu().numpy())
        return np.concatenate(out)

    caches = sorted(glob.glob(f"{CACHE}/{grain}/*.npz"))
    g0 = []
    for f in caches:
        dd = np.load(f, allow_pickle=True)
        a0i = int(np.argmin(np.abs(np.asarray(dd["alphas"], float))))     # α=0 baseline (grid may be one-sided, not symmetric)
        z0 = dd["gen"][a0i]
        if z0 is not None and len(z0):
            g0.append(np.asarray(z0, np.float32))
    g0 = np.concatenate(g0); mu_g, sd_g = g0.mean(0), g0.std(0) + 1e-6
    gen = {}
    for f in caches:
        dd = np.load(f, allow_pickle=True); g = str(dd["gene"]); al = dd["alphas"]
        if g not in c2i:
            continue
        ti0 = c2i[g]
        for ai in range(len(al)):
            gv = dd["gen"][ai]
            if gv is None or not len(gv):
                continue
            gz = (np.asarray(gv, np.float32) - mu_g) / sd_g
            Q = torch.tensor(gz, device=dev, dtype=torch.float32); Q = Q / Q.norm(dim=1, keepdim=True).clamp_min(1e-9)
            aps = ap(Q, torch.full((len(Q),), ti0, device=dev))
            a = float(al[ai]); gen.setdefault(a, {})[g] = float(aps.mean())
    # ceiling: sample 30 real cells/class as queries vs full gallery, mask self
    ceil = {}
    rng = np.random.default_rng(0)
    for c in cls:
        idx = np.where(labels == c)[0]
        qi = idx[:30]                                    # first 30 gallery cells of the class as held-in queries (self masked)
        Q = Gt[torch.tensor(qi, device=dev)]
        aps = ap(Q, torch.full((len(qi),), c2i[c], device=dev), self_idx=torch.tensor(qi, device=dev))
        ceil[c] = float(aps.mean())
    json.dump({"alphas": sorted(gen), "gen": gen, "ceiling": ceil, "n_classes": len(c2i),
               "gallery_cells": len(labels)}, open(f"{OUT}/{grain}_propermap1k.json", "w"))
    return {"grain": grain, "n": len(c2i), "gallery": len(labels)}


def plot_proper1k(min_dist=None, fname="mAP_proper1k"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["pdf.fonttype"] = 42
    CM = "#2471a3"
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, (grain, lbl) in zip(axes, [("geneKO", "Gene-level"), ("complex", "Protein complex")]):
        d = json.load(open(f"{OUT}/{grain}_propermap1k.json")); al = d["alphas"]; gen = d["gen"]; ceil = d["ceiling"]
        keep = None
        if min_dist is not None:
            rd = json.load(open(f"{DIST}/{grain}_real.json")); keep = {c for c, v in rd.items() if v > min_dist}
        def vals(a):
            return [v for c, v in gen[str(a)].items() if keep is None or c in keep]
        m = np.array([np.mean(vals(a)) for a in al]); se = np.array([np.std(vals(a), ddof=1) / np.sqrt(len(vals(a))) for a in al])
        ax.plot(al, m, "-", color=CM, lw=2.6, label="generated (mAP, 1000-cell gallery)")
        ax.fill_between(al, m - se, m + se, color=CM, alpha=.18, lw=0)
        cvs = [v for c, v in ceil.items() if keep is None or c in keep]
        cv = np.mean(cvs); cse = np.std(cvs, ddof=1) / np.sqrt(len(cvs))
        ax.axhline(cv, color=CM, ls=":", lw=2.0, label=f"real ceiling ({cv:.3f})"); ax.axhspan(cv - cse, cv + cse, color=CM, alpha=.07, lw=0)
        ax.axvline(0, color="#ccc", lw=1); ax.axvline(1, color="#bbb", lw=1, ls="--")
        n = len(vals(al[0]))
        ax.set_title(f"{lbl}  (n={n})"); ax.set_xlabel("traversal α"); ax.grid(alpha=.25)
    axes[0].set_ylabel("real↔generated retrieval mAP (copairs-style)")
    gate = "" if min_dist is None else f" · real distinctiveness@20 > {min_dist}"
    fig.suptitle(f"Proper retrieval mAP: generated → full real gallery (1000 cells/class){gate}", fontweight="bold")
    fig.tight_layout()
    for e in ("png", "svg"):
        fig.savefig(f"{OUT}/{fname}.{e}", dpi=150, bbox_inches="tight")
    plt.close(fig); print(f"saved {fname}")


V2 = "/hpc/projects/icd.fast.ops/analysis/figure4_traversals/gen_real_map_v2"   # proper copairs retrieval mAP (gen→real cells)


def plot_map_proper(min_dist=None, fname="mAP_proper"):
    """Proper copairs retrieval mAP (generated cells → real CELLS gallery, multiple positives) from the v2 data,
    same design as centroid_topk: solid generated (mean±SEM) + dotted real split-half ceiling. Own y-scale per grain."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    plt.rcParams["pdf.fonttype"] = 42
    CM = "#2471a3"
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, (grain, lbl) in zip(axes, [("geneKO", "Gene-level"), ("complex", "Protein complex")]):
        files = sorted(glob.glob(f"{V2}/{grain}_a*.json"), key=lambda p: int(p.split("_a")[-1][:-5]))
        al, cols = [], {}
        for f in files:
            d = json.load(open(f)); al.append(d["alpha"]); cols[d["alpha"]] = d["gen_map"]
        df = pd.DataFrame(cols)                                          # index=class, cols=alpha
        keep = None
        if min_dist is not None:
            rd = json.load(open(f"{DIST}/{grain}_real.json")); keep = {c for c, v in rd.items() if v > min_dist}
            df = df[df.index.map(lambda c: keep is not None and c in keep)]
        m = df[al].mean(0).values; se = df[al].std(0, ddof=1).values / np.sqrt(len(df))
        ax.plot(al, m, "-", color=CM, lw=2.6, label="generated (copairs mAP)")
        ax.fill_between(al, m - se, m + se, color=CM, alpha=.18, lw=0)
        ceil = json.load(open(f"{V2}/{grain}_ceiling.json"))["map"]
        cvs = [v for c, v in ceil.items() if keep is None or c in keep]
        cv = np.mean(cvs); cse = np.std(cvs, ddof=1) / np.sqrt(len(cvs))
        ax.axhline(cv, color=CM, ls=":", lw=2.0, label=f"real split-half ceiling ({cv:.3f})")
        ax.axhspan(cv - cse, cv + cse, color=CM, alpha=.07, lw=0)
        ax.axvline(0, color="#ccc", lw=1); ax.axvline(1, color="#bbb", lw=1, ls="--")
        ax.set_title(f"{lbl}  (n={len(df)})"); ax.set_xlabel("traversal α"); ax.grid(alpha=.25)
    axes[0].set_ylabel("real↔generated retrieval mAP (copairs)")
    gate = "" if min_dist is None else f" · real distinctiveness@20 > {min_dist}"
    fig.suptitle(f"Proper retrieval mAP: generated cells → real-cell gallery (30/class){gate}", fontweight="bold")
    fig.tight_layout()
    for e in ("png", "svg"):
        fig.savefig(f"{OUT}/{fname}.{e}", dpi=150, bbox_inches="tight")
    plt.close(fig); print(f"saved {fname}")


def main():
    from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs
    jobs = []
    for grain in ["geneKO", "complex"]:
        cls = _classes(grain)
        for i in range(0, len(cls), PER_SHARD):
            jobs.append({"name": f"grc_{grain}_{i}", "func": embed_centroids, "kwargs": {"grain": grain, "classes": cls[i:i + PER_SHARD]}})
    print(f"[gen-real-centroid] {len(jobs)} embed shards (N={N}, {PER_SHARD}/shard)")
    submit_parallel_jobs(jobs, experiment="gen_real_centroid",
                         slurm_params={"slurm_partition": "gpu", "slurm_gres": "gpu:1", "cpus_per_task": 12,
                                       "mem_gb": 64, "timeout_min": 300}, log_dir="gen_real_centroid", wait_for_completion=False)


if __name__ == "__main__":
    import sys
    if "--merge-score-plot" in sys.argv:
        for g in ["geneKO", "complex"]:
            print(merge(g)); print(score(g)); print(ceiling(g))
        plot()
        plot(min_dist=0.1, fname="centroid_topk_dist01")
    else:
        main()
