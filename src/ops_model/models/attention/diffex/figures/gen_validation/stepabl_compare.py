"""Step-ablation comparison on the fixed 15-gene set: centroid recovery + SetTransformer, at matched 45-cell bag.
50-step arm = existing valid200 w=1.5 cache (capped to 45); 100/200 = stepabl caches. Baseline = orig 30-cell w=2.
"""
import json, glob, os
import numpy as np

CV = "/hpc/projects/icd.fast.ops/analysis/figure4_traversals"
B = "/hpc/projects/icd.fast.ops/models/diffex"
cen = np.load(f"{CV}/gen_real_centroid/geneKO_centroids.npz", allow_pickle=True)
names = list(cen["names"]); cidx = {c: i for i, c in enumerate(names)}
cz = (cen["cents"] - cen["mu"]) / cen["sd"]; cz = cz / (np.linalg.norm(cz, axis=1, keepdims=True) + 1e-9)

GENES = set(json.load(open(f"{CV}/gen_real_map_cache_stepabl_s100/geneKO/AACS.npz", allow_pickle=True).files) if False else
            [os.path.basename(f)[:-4] for f in glob.glob(f"{CV}/gen_real_map_cache_stepabl_s100/geneKO/*.npz")])


def centroid_recovery(cache, cap=45):
    """per-α mean top1/top5/mAP over the 15 genes, α=0 baseline = argmin|α|, gen capped to `cap`."""
    caches = [f"{cache}/geneKO/{g}.npz" for g in GENES if os.path.exists(f"{cache}/geneKO/{g}.npz")]
    g0 = []
    for f in caches:
        d = np.load(f, allow_pickle=True); a0 = int(np.argmin(np.abs(np.asarray(d["alphas"], float)))); z = d["gen"][a0]
        if z is not None and len(z): g0.append(np.asarray(z, np.float32)[:cap])
    mu, sd = np.concatenate(g0).mean(0), np.concatenate(g0).std(0) + 1e-6
    by = {}
    for f in caches:
        d = np.load(f, allow_pickle=True); g = str(d["gene"]); al = list(d["alphas"])
        if g not in cidx: continue
        ti = cidx[g]
        for ai, a in enumerate(al):
            gv = d["gen"][ai]
            if gv is None or not len(gv): continue
            gz = (np.asarray(gv, np.float32)[:cap] - mu) / sd; gz = gz / (np.linalg.norm(gz, axis=1, keepdims=True) + 1e-9)
            order = np.argsort(-(gz @ cz.T), axis=1); rk = np.where(order == ti)[1] + 1
            by.setdefault(a, {"t1": [], "t5": [], "mp": []})
            by[a]["t1"].append(np.mean(order[:, 0] == ti)); by[a]["t5"].append(np.mean([ti in r[:5] for r in order]))
            by[a]["mp"].append(np.mean(1.0 / rk))
    al = sorted(by); mp = [np.mean(by[a]["mp"]) for a in al]; t1 = [np.mean(by[a]["t1"]) for a in al]; t5 = [np.mean(by[a]["t5"]) for a in al]
    k = int(np.argmax(mp)); return al[k], mp[k], t1[k], t5[k]


def settransformer(tree):
    """mean peak-α P(target), median rank, top5% over the 15 genes from scores_v5.json in `tree`."""
    P, RK, T5, al = [], [], [], None
    for g in GENES:
        f = f"{B}/{tree}/phase/geneKO/{g}/scores_v5.json"
        if not os.path.exists(f): continue
        s = json.load(open(f)); al = s["alphas"]; P.append(s["p_target"]); RK.append(s["rank_target"]); T5.append(s["top5_target"])
    if not P: return None
    P = np.array(P, float); RK = np.array(RK, float); T5 = np.array(T5, float)
    k = int(np.argmax(P.mean(0))); return al[k], P[:, k].mean(), np.median(RK[:, k]), T5[:, k].mean(), len(P)


if __name__ == "__main__":
    print(f"n genes = {len(GENES)}")
    print("\n=== CENTROID RECOVERY (peak α) — 15 genes, 45-cell bag ===")
    for lbl, c in [("baseline 30c w=2 (orig)", f"{CV}/gen_real_map_cache"),
                   ("50-step  w=1.5 (valid200)", f"{CV}/gen_real_map_cache_valid200"),
                   ("100-step w=1.5 INVERTED ", f"{CV}/gen_real_map_cache_stepabl_s100"),
                   ("100-step w=1.5 RANDOM-xT", f"{CV}/gen_real_map_cache_stepabl_s100_randxt"),
                   ("200-step w=1.5 INVERTED ", f"{CV}/gen_real_map_cache_stepabl_s200")]:
        try:
            a, mp, t1, t5 = centroid_recovery(c); print(f"  {lbl}: α={a:+.1f}  mAP={mp:.3f}  top1={t1:.1%}  top5={t5:.1%}")
        except Exception as e:
            print(f"  {lbl}: ERR {e}")
    print("\n=== SetTransformer (peak-α mean over 15 genes) ===")
    for lbl, t in [("50-step  w=1.5 (valid200, bag=200!)", "viewer_assets_valid200"),
                   ("100-step w=1.5 (stepabl, bag=45)", "viewer_assets_stepabl_s100"),
                   ("200-step w=1.5 (stepabl, bag=45)", "viewer_assets_stepabl_s200")]:
        r = settransformer(t)
        if r: a, p, rk, t5, n = r; print(f"  {lbl}: α={a:+.1f}  P(target)={p:.3f}  medRank={rk:.0f}  top5={t5:.0%}  (n={n})")
