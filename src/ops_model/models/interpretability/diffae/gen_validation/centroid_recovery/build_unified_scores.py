"""Unify the 3 metrics into one metadata index, keyed (perturbation × channel). Reuses existing outputs
(no recompute): A = global-std bag centroid recovery; B = LOO percentile/rank; C = per-domain per-cell centroid
recovery. Each of A/B/C is filled for BOTH phase and fluor channels. Per-cell LOO arrays stay on disk (referenced).

  unified_scores.json:  {perturbation: {channel: {grain, A:{f,top1,alphas}, B:{f,pct,rank,per_cell_npz}, C:{f,top1}}}}
"""
import glob
import json
import os

import numpy as np

FT = "/hpc/projects/icd.fast.ops/analysis/figure4_traversals"
A_PHASE = f"{FT}/centroid_pooled_bagsweep_v5new"      # A (phase)
A_FLUOR = f"{FT}/metricA_fluor"                       # A (fluor, new)
C_PHASE = f"{FT}/metricC_phase"                       # C (phase, new)
C_FLUOR = f"{FT}/f_centroid_recovery/fluor"           # C (fluor = fcr per-domain per-cell)
LOO = f"{FT}/loo_cache"                               # B (both)
OUT = f"{FT}/unified_scores.json"
BAG = "200"


def _slug(s):
    from ops_model.models.interpretability.diffae.classifier.config import slugify
    return slugify(str(s))


def _ipk(al, y):
    al = np.asarray(al, float); y = np.array([np.nan if v is None else v for v in y], float)
    pos = al > 0; a, v = al[pos], y[pos]; ok = ~np.isnan(v); a, v = a[ok], v[ok]
    if len(a) < 2:
        return None
    i = int(np.argmax(v))
    if i == 0 or i == len(a) - 1:
        return round(float(a[i]), 3)
    x3, y3 = a[i - 1:i + 2], v[i - 1:i + 2]; c = np.polyfit(x3, y3, 2)
    return round(float(a[i]) if c[0] >= 0 else float(np.clip(-c[1] / (2 * c[0]), x3[0], x3[2])), 3)


def _set(U, pert, chan, grain, key, val):
    U.setdefault(pert, {}).setdefault(chan, {"grain": grain})[key] = val


def build():
    U = {}
    # ---- A phase (pooled json: gen[bag][α]["top1"][gene]) ----
    for grain in ("geneKO", "complex"):
        p = f"{A_PHASE}/{grain}_pooled.json"
        if not os.path.exists(p):
            continue
        bb = json.load(open(p))["gen"][BAG]; al = sorted(float(a) for a in bb)
        genes = set().union(*[set(bb[k]["top1"]) for k in bb])
        for g in genes:
            t1 = [bb[str(a) if str(a) in bb else a]["top1"].get(g) for a in al]
            _set(U, g, "phase", grain, "A", {"f": _ipk(al, t1), "top1": t1, "alphas": al})
    # ---- A fluor (new) ----
    for fp in glob.glob(f"{A_FLUOR}/*__*.json"):
        mod, grain = os.path.basename(fp)[:-5].rsplit("__", 1); d = json.load(open(fp))
        bb = d["gen"][BAG]; al = d["alphas"]
        for cls, f in d["f"].items():
            t1 = [bb.get(str(a), {"top1": {}})["top1"].get(cls) for a in al]
            _set(U, cls, mod, grain, "A", {"f": f, "top1": t1, "alphas": al})
    # ---- C phase (new) ----
    for fp in glob.glob(f"{C_PHASE}/*.json"):
        for g, rec in json.load(open(fp)).items():
            _set(U, g, "phase", fp.split("/")[-1].split("_")[0], "C", {"f": rec["f"], "top1": rec["top1"], "alphas": rec["alphas"]})
    # ---- C fluor (fcr) ----
    for fp in glob.glob(f"{C_FLUOR}/*__*.json"):
        mod, grain = os.path.basename(fp)[:-5].rsplit("__", 1)
        for cls, rec in json.load(open(fp)).items():
            _set(U, _slug(cls), mod, grain, "C", {"f": rec["f"], "top1": rec["top1"], "alphas": rec["alphas"]})
    # ---- B (LOO) both: loo_cache/{channel}/{grain}/{class}.npz ----
    for fp in glob.glob(f"{LOO}/*/*/*.npz"):
        chan, grain, cls = fp.split("/")[-3:]; cls = cls[:-4]
        z = np.load(fp, allow_pickle=True); al = [float(a) for a in z["alphas"]]
        _set(U, cls, chan, grain, "B", {"f": float(z["f"]), "pct": [None if v is None else round(float(v), 4) for v in z["curve"]],
                                         "rank": [round(float(np.nanmean(z["real_rank"][:, i])), 1) for i in range(len(al))] if "real_rank" in z else None,
                                         "n_real": int(z["n_real"]) if "n_real" in z else None, "alphas": al, "per_cell_npz": fp})
    json.dump(U, open(OUT, "w"))
    npert = len(U); nch = sum(len(v) for v in U.values())
    cov = {m: sum(1 for v in U.values() for c in v.values() if m in c) for m in ("A", "B", "C")}
    print(f"unified: {npert} perturbations, {nch} (pert×channel) records → {OUT}")
    print(f"metric coverage across records: {cov}")
    return U


if __name__ == "__main__":
    build()
