"""Diagnostic: is the valid200 effect-size gap to v5 driven by bag COMPOSITION (extra generic cells drag the
mean) or by the generation itself? Recompute α=5 nearest-centroid top-1 for valid200 using first-{45,100,200}
cells vs the v5 (~45-cell) bag, with the CORRECT α=0 baseline (argmin|α|). Faithful centroids reused from v5.
"""
import numpy as np, glob

CV = "/hpc/projects/icd.fast.ops/analysis/figure4_traversals"
cen = np.load(f"{CV}/gen_real_centroid/geneKO_centroids.npz", allow_pickle=True)
names = list(cen["names"]); cidx = {c: i for i, c in enumerate(names)}
cz = (cen["cents"] - cen["mu"]) / cen["sd"]; cz = cz / (np.linalg.norm(cz, axis=1, keepdims=True) + 1e-9)


def collect(cache):
    A0, A5, GN = [], [], []
    for f in sorted(glob.glob(f"{cache}/geneKO/*.npz")):
        d = np.load(f, allow_pickle=True); g = str(d["gene"]); al = list(d["alphas"])
        if g not in cidx:
            continue
        i0 = int(np.argmin(np.abs(np.array(al)))); i5 = int(np.argmin(np.abs(np.array(al) - 5.0)))
        z0, z5 = d["gen"][i0], d["gen"][i5]
        if z0 is None or z5 is None or not len(z0) or not len(z5):
            continue
        A0.append(np.asarray(z0, np.float32)); A5.append(np.asarray(z5, np.float32)); GN.append(g)
    return A0, A5, GN


def top1(A0, A5, GN, cap=None):
    mu = np.concatenate([a[:cap] for a in A0]).mean(0); sd = np.concatenate([a[:cap] for a in A0]).std(0) + 1e-6
    t = []
    for z5, g in zip(A5, GN):
        gz = (z5[:cap] - mu) / sd; gz = gz / (np.linalg.norm(gz, axis=1, keepdims=True) + 1e-9)
        order = np.argsort(-(gz @ cz.T), axis=1); t.append(np.mean(order[:, 0] == cidx[g]))
    return np.mean(t), len(t)


if __name__ == "__main__":
    A0, A5, GN = collect(f"{CV}/gen_real_map_cache_valid200")
    for cap in (45, 100, 200):
        m, n = top1(A0, A5, GN, cap); print(f"valid200 first-{cap:3d} @a=5 top1={m:.1%} (n={n})")
    A0v, A5v, GNv = collect(f"{CV}/gen_real_map_cache")
    m, n = top1(A0v, A5v, GNv); print(f"v5 (~45-cell)      @a=5 top1={m:.1%} (n={n})")
