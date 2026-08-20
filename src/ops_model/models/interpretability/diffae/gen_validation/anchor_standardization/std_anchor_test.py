"""Is the baseline>>valid200 gap a STANDARDIZATION-anchor artifact? Re-score both caches on the same genes under
several gen-standardization schemes. If the gap closes under 'real_pop'/'none' vs 'gen_a0', the per-domain
gen-α=0 anchor (which changed meaning under DDIM inversion) was mismeasuring — not a generation regression.
"""
import json, glob, os
import numpy as np

CV = "/hpc/projects/icd.fast.ops/analysis/figure4_traversals"
cen = np.load(f"{CV}/gen_real_centroid/geneKO_centroids.npz", allow_pickle=True)
names = list(cen["names"]); cidx = {c: i for i, c in enumerate(names)}
mu_r, sd_r = cen["mu"], cen["sd"]

# same 15 genes as the ablation
GENES = sorted(os.path.basename(f)[:-4] for f in glob.glob(f"{CV}/gen_real_map_cache_stepabl_s100/geneKO/*.npz"))


def load(cache, cap):
    """→ {gene: {alpha: (n,1024)}} + pooled α=0 for the 15 genes."""
    per, a0 = {}, []
    for g in GENES:
        f = f"{cache}/geneKO/{g}.npz"
        if not os.path.exists(f) or g not in cidx: continue
        d = np.load(f, allow_pickle=True); al = list(np.asarray(d["alphas"], float))
        i0 = int(np.argmin(np.abs(np.array(al))))
        per[g] = {a: (None if d["gen"][i] is None else np.asarray(d["gen"][i], np.float32)[:cap]) for i, a in enumerate(al)}
        if per[g][al[i0]] is not None: a0.append(per[g][al[i0]])
    return per, np.concatenate(a0)


def recover(per, mu_g, sd_g, scheme):
    czr = (cen["cents"] - mu_r) / sd_r; czr = czr / (np.linalg.norm(czr, axis=1, keepdims=True) + 1e-9)
    czn = cen["cents"] / (np.linalg.norm(cen["cents"], axis=1, keepdims=True) + 1e-9)
    best = None
    alphas = sorted({a for g in per for a in per[g]})
    for a in alphas:
        t1s = []
        for g in per:
            gv = per[g].get(a)
            if gv is None or not len(gv): continue
            if scheme == "gen_a0":   gz = (gv - mu_g) / sd_g; cz = czr
            elif scheme == "real_pop": gz = (gv - mu_r) / sd_r; cz = czr
            elif scheme == "none":     gz = gv;                 cz = czn
            gz = gz / (np.linalg.norm(gz, axis=1, keepdims=True) + 1e-9)
            order = np.argsort(-(gz @ cz.T), axis=1)
            t1s.append(np.mean(order[:, 0] == cidx[g]))
        m = np.mean(t1s)
        if best is None or m > best[1]: best = (a, m)
    return best


if __name__ == "__main__":
    print(f"n genes = {len(GENES)}  (cap=30, both caches have >=30 cells)\n")
    caches = [("baseline 30c w=2 (old frames)", f"{CV}/gen_real_map_cache"),
              ("valid200 w=1.5 INVERTED", f"{CV}/gen_real_map_cache_valid200")]
    for scheme in ("gen_a0", "real_pop", "none"):
        print(f"--- standardization = {scheme} ---")
        for lbl, c in caches:
            per, a0 = load(c, 30); mu_g, sd_g = a0.mean(0), a0.std(0) + 1e-6
            a, m = recover(per, mu_g, sd_g, scheme)
            print(f"   {lbl:34s}: peak α={a:+.1f}  top1={m:.1%}")
