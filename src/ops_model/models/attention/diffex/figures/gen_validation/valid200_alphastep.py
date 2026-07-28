"""Is the valid200 rightward-α shift a real generation property (compressed α-steps) or an analysis bug?
Independently of scores_v5 and of any standardization, measure the raw CellDINO displacement of each α-frame
from the α=0 frame, per gene, averaged — for v5 vs valid200 at the α values present in BOTH grids. If valid200's
per-α displacement is compressed (smaller step per α), the phenotype emerges later in α and EVERY downstream
measure inherits the shift → generation, not analysis.
"""
import numpy as np, glob

CV = "/hpc/projects/icd.fast.ops/analysis/figure4_traversals"
SHARED = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0]   # α present in both grids


def profile(cache):
    disp = {a: [] for a in SHARED}
    for f in sorted(glob.glob(f"{cache}/geneKO/*.npz")):
        d = np.load(f, allow_pickle=True); al = list(np.asarray(d["alphas"], float))
        i0 = int(np.argmin(np.abs(np.array(al))))
        z0 = d["gen"][i0]
        if z0 is None or not len(z0):
            continue
        c0 = np.asarray(z0, np.float32).mean(0)                    # gene's α=0 centroid in raw CellDINO
        for a in SHARED:
            if a not in al:
                continue
            za = d["gen"][al.index(a)]
            if za is None or not len(za):
                continue
            disp[a].append(np.linalg.norm(np.asarray(za, np.float32).mean(0) - c0))
    return {a: (np.mean(v) if v else np.nan) for a, v in disp.items()}


if __name__ == "__main__":
    v5 = profile(f"{CV}/gen_real_map_cache")
    v2 = profile(f"{CV}/gen_real_map_cache_valid200")
    print(f"{'alpha':>6} {'v5 disp':>10} {'v200 disp':>10} {'ratio v200/v5':>14}")
    for a in SHARED:
        r = v2[a] / v5[a] if v5[a] else np.nan
        print(f"{a:6.1f} {v5[a]:10.2f} {v2[a]:10.2f} {r:14.2f}")
