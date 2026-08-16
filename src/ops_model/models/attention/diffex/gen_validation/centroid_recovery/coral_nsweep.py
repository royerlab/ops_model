"""Real-centroid depth sweep (N=100/500/1000) using Alex's cached per-cell embeddings (paper_v2, NO re-embed).

Real cells: paper_v2_{phase,fluor} .pt (SetTransformer bags → flatten to per-cell emb + segmentation_id, z-scored
classifier space), ranked by shap_screen (join on seg_id) → top-N centroid. Generated cells stay FIXED (existing
embcache; embed_crops). CORAL bridges the embed_crops↔classifier space + the real↔gen gap. Stores per (N, method)
the full per-α curve + peak + f, so any variant can feed the viewer without recompute.

  python coral_nsweep.py submit   # shard per (mod, grain)
  python coral_nsweep.py merge    # per-N table (f / peak / %f=1), phase vs fluor
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

from ops_model.models.attention.diffex.gen_validation.centroid_recovery.coral_degap_f import ALPHAS, EMB, THR, _loadf, _norm, _slug, _sympow, ipk, _curve

FT = "/hpc/projects/icd.fast.ops/analysis/figure4_traversals"
ALX = "/hpc/projects/icd.fast.ops/models/alex_lin_attention/v5"
PV2 = {"phase": f"{ALX}/paper_v2_phase/val", "fluor": f"{ALX}/paper_v2_fluor_cp_4i_cellstrat/val"}
SHAP_PHASE = f"{ALX}/multi_rank/shap_screen_phase_all.parquet"
FLUOR_SHAP = "/hpc/projects/icd.fast.ops/models/diffex/viewer_assets_v5/_rankings/fluor_shap"   # per-channel: gene,rank,segmentation
NS = [100, 500, 1000]
DOM_GENES, DOM_CAP = 80, 15000
OUT = f"{FT}/coral_Nsweep"


def _flatten_pt(p):
    """paper_v2 .pt -> (seg_ids, embeddings, channel_ids) per cell (bags flattened)."""
    d = torch.load(p, map_location="cpu")
    E = d["embeddings"].numpy().astype(np.float32)
    segs = np.array([int(s) for bag in d["cell_metadata"]["segmentation_id"] for s in bag], dtype=np.int64)
    ch = d["channel_ids"].numpy()
    return segs, E, ch


def _ranked_real(dom, gene, rank_map):
    """Top-ranked real embeddings for a class, ordered by shap rank (top-1000). For fluor, the marker's
    paper_v2 channel is inferred empirically = the modal channel among the rank-matched cells (the shap
    zarr_channel_index does NOT equal paper_v2's channel_id)."""
    p = f"{PV2[dom]}/{gene}.pt"
    if not os.path.exists(p) or gene not in rank_map:
        return None
    segs, E, ch = _flatten_pt(p)
    r = rank_map[gene]                                        # {seg_id: rank}
    keep = np.array([s in r for s in segs])
    if keep.sum() < 20:
        return None
    if dom == "fluor":                                       # disambiguate the marker's channel
        mch = np.bincount(ch[keep]).argmax()
        keep &= (ch == mch)
        if keep.sum() < 20:
            return None
    segs, E = segs[keep], E[keep]
    order = np.argsort([r[s] for s in segs])                 # ascending rank = most predictive first
    return E[order][:1000]


def _phase_ranks():
    df = pd.read_parquet(SHAP_PHASE, columns=["gene", "segmentation_id", "rank"])
    return {g: dict(zip(d.segmentation_id.astype(int), d["rank"])) for g, d in df.groupby("gene")}


def _fluor_ranks(mod, grain):
    p = f"{FLUOR_SHAP}/{grain}/{mod}.parquet"
    if not os.path.exists(p):
        return {}
    df = pd.read_parquet(p)
    seg = "segmentation" if "segmentation" in df.columns else "segmentation_id"
    return {g: dict(zip(d[seg].astype(int), d["rank"])) for g, d in df.groupby("gene")}


def shard(mod, grain, genes=None, tag=None):
    os.makedirs(OUT, exist_ok=True)
    dom = "phase" if mod == "phase" else "fluor"
    ranks = _phase_ranks() if mod == "phase" else _fluor_ranks(mod, grain)
    names = sorted(ranks.keys())
    score_genes = genes if genes is not None else names
    # ranked real per class (cache in-shard)
    real = {}
    for g in names:
        rr = _ranked_real(dom, g, ranks)
        if rr is not None:
            real[g] = rr
    if len(real) < 3:
        return {"mod": mod, "grain": grain, "skip": "real<3"}
    # CORAL fit: real domain cov from pooled top-500 of the domain classes; gen domain from embcache
    dom_names = list(real)[:DOM_GENES]
    R = np.concatenate([real[g][:500] for g in dom_names])
    gd = []
    for g in dom_names:
        gd += _loadf(sorted(glob.glob(f"{EMB}/gen/{mod}/{grain}/{g}/a*.npz")))
    Gd = np.concatenate(gd)
    if len(Gd) > DOM_CAP:
        Gd = Gd[np.random.default_rng(0).choice(len(Gd), DOM_CAP, replace=False)]
    mu_r, sd_r = R.mean(0), R.std(0) + 1e-6; mu_g = Gd.mean(0)
    W = _sympow(np.cov(Gd.T), -0.5) @ _sympow(np.cov(R.T), 0.5)
    coral = lambda gv: _norm(((gv - mu_g) @ W) / sd_r)
    # nearest-centroid competes against the WHOLE bank -> build the bank at each N, then score every gene
    for n in NS:
        bnames = list(real); bank = np.stack([real[g][:n].mean(0) for g in bnames])
        czb = _norm((bank - mu_r) / sd_r); bidx = {g: i for i, g in enumerate(bnames)}
        out = {}
        for g in score_genes:
            if g not in bidx:
                continue
            fs = sorted(glob.glob(f"{EMB}/gen/{mod}/{grain}/{g}/a*.npz"), key=lambda p: int(os.path.basename(p)[1:-4]))
            if not fs:
                continue
            ais = [int(os.path.basename(p)[1:-4]) for p in fs]; al = [ALPHAS[a] for a in ais]
            gens = _loadf(fs); a0 = int(np.argmin(np.abs(np.array(al)))); g0 = gens[a0]
            mu_p, sd_p = g0.mean(0), g0.std(0) + 1e-6
            zscore = lambda gv, mu_p=mu_p, sd_p=sd_p: _norm((gv - mu_p) / sd_p)
            ti = bidx[g]
            out[f"{mod}/{grain}/{g}"] = {"alphas": al,
                                        "z": _curve(gens, al, ti, czb, zscore),
                                        "coral": _curve(gens, al, ti, czb, coral)}
        os.makedirs(f"{OUT}/N{n}", exist_ok=True)
        suffix = f"__{tag}" if tag else ""
        json.dump(out, open(f"{OUT}/N{n}/{mod}__{grain}{suffix}.json", "w"))
    return {"mod": mod, "grain": grain, "n": len(real)}


EMB100 = f"{OUT}/gal100"   # {mod}/{grain}/{gene}.npz = top-100 paper_v2 real embeddings (built once per gene)


def _all_fluor_rankmaps():
    maps = {}
    for grain in ("geneKO", "complex"):
        for p in glob.glob(f"{FLUOR_SHAP}/{grain}/*.parquet"):
            maps[(os.path.basename(p)[:-8], grain)] = _fluor_ranks(os.path.basename(p)[:-8], grain)
    return maps


CP_CSV = f"{ALX}/fluorescence/misc/gene_marker_1K_CP.csv"   # fixed (CellProfiler-seg) markers: seg matches paper_v2


def _cp_rankmaps():
    df = pd.read_csv(CP_CSV, usecols=["channel_name", "gene", "segmentation_id", "rank"])
    return {(_slug(ch), "geneKO"): {g: dict(zip(dd.segmentation_id.astype(int), dd["rank"]))
                                    for g, dd in d.groupby("gene")} for ch, d in df.groupby("channel_name")}


def build_gal100_cp(genes, topn=100):
    """gal100 for the FIXED markers, ranked/seg-matched via gene_marker_1K_CP.csv (CP seg = paper_v2 seg)."""
    maps = _cp_rankmaps()
    done = 0
    for gene in genes:
        p = f"{PV2['fluor']}/{gene}.pt"
        if not os.path.exists(p):
            continue
        segs, E, ch = _flatten_pt(p)
        for (mod, grain), rm in maps.items():
            r = rm.get(gene)
            if not r:
                continue
            keep = np.array([s in r for s in segs])
            if keep.sum() < 20:
                continue
            keep &= (ch == np.bincount(ch[keep]).argmax())
            if keep.sum() < 20:
                continue
            ss, EE = segs[keep], E[keep]
            order = np.argsort([r[s] for s in ss])[:topn]
            d = f"{EMB100}/{mod}/{grain}"; os.makedirs(d, exist_ok=True)
            np.savez(f"{d}/{gene}.npz", features=EE[order].astype(np.float32))
        done += 1
    return {"genes": done}


V4H5 = "/hpc/projects/icd.fast.ops/models/alex_lin_attention/v4/expansion_v1/per_experiment_v4_fluor"
SHAP20K = f"{ALX}/multi_rank/shap_screen_fluor_full_20k_all.csv"
FIXED_MARKERS = ["Mitochondria_TOMM20", "Nucleoli_NPM1", "Endoplasmic Reticulum_Concanavalin A",
                 "F-actin_Phalloidin", "Microtubules_Tubulin", "Plasma Membrane_Wheat Germ Agglutinin",
                 "b-catenin_b-catenin (mouse-488)", "c-Myc_c-Myc (mouse-488)", "p21_p21 (rabbit-647)",
                 "p53_p53 (mouse-488)", "pRb_pRb (rabbit-647)", "pS6_pS6 (rabbit-647)"]   # Hoechst excluded
FIX_PARTS = f"{OUT}/fixed_parts"


def _normch(s):
    return str(s).replace(", ", "_").replace(" ", "_")


def _fixed_ranks(topk=1000):
    """Per-marker fluor_shap parquets for the 12 fixed markers → (mod=slug, chan_norm, gene, experiment, well,
    segmentation_id, rank). mod = gal100/gen dir slug; chan_norm = _normch(label) to join the h5ad channel_name."""
    dfs = []
    for mc in FIXED_MARKERS:
        p = f"{FLUOR_SHAP}/geneKO/{_slug(mc)}.parquet"
        if not os.path.exists(p):
            continue
        d = pd.read_parquet(p)
        seg = "segmentation" if "segmentation" in d.columns else "segmentation_id"
        d = d[d["rank"] <= topk][["gene", "experiment", "well", seg, "rank"]].rename(columns={seg: "segmentation_id"})
        d["mod"] = _slug(mc); d["chan_norm"] = _normch(mc)
        dfs.append(d)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def harvest_fixed(experiment, topk=1000):
    """CPU: pull cached CellDINO features for the fixed markers from the experiment's v4 h5ad (recipe join)."""
    import anndata as ad
    import collections
    os.makedirs(FIX_PARTS, exist_ok=True)
    t = _fixed_ranks(topk); t = t[t.experiment == experiment]
    if not len(t):
        return {"experiment": experiment, "skip": "no fixed rows"}
    fs = glob.glob(f"{V4H5}/{experiment}*.h5ad")
    if not fs:
        return {"experiment": experiment, "skip": "no h5ad"}
    a = ad.read_h5ad(fs[0], backed="r"); obs = a.obs
    cn = obs["channel_name"].astype(str).map(_normch).values
    well = obs["well"].astype(str).values; seg = obs["segmentation_id"].astype("int64").values
    row_of = {(well[i], int(seg[i]), cn[i]): i for i in range(len(cn))}                # h5ad key -> row
    idx, meta = [], []
    for r in t.itertuples():
        i = row_of.get((str(r.well), int(r.segmentation_id), r.chan_norm))
        if i is not None:
            idx.append(i); meta.append((r.mod, r.gene, int(r.rank)))
    if not idx:
        return {"experiment": experiment, "matched": 0}
    order = np.argsort(idx); idx_s = list(np.array(idx)[order])                        # backed load wants sorted rows
    X = a[idx_s].X; X = np.asarray(X.todense()) if hasattr(X, "todense") else np.asarray(X)
    meta_s = [meta[k] for k in order]
    part = collections.defaultdict(lambda: {"rank": [], "feat": []})
    for (mk, g, rk), x in zip(meta_s, X):
        part[(mk, g)]["rank"].append(rk); part[(mk, g)]["feat"].append(x.astype(np.float32))
    np.savez(f"{FIX_PARTS}/{experiment}.npz",
             keys=np.array(["|".join(k) for k in part], dtype=object),
             **{f"r::{'|'.join(k)}": np.array(v["rank"]) for k, v in part.items()},
             **{f"f::{'|'.join(k)}": np.stack(v["feat"]) for k, v in part.items()})
    return {"experiment": experiment, "matched": len(idx)}


def submit_harvest_fixed():
    from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs
    exps = sorted(_fixed_ranks().experiment.unique())
    jobs = [{"name": f"hfx_{e}", "func": harvest_fixed, "kwargs": {"experiment": e}} for e in exps]
    print(f"[harvest fixed] {len(exps)} experiments: {exps}")
    submit_parallel_jobs(jobs, experiment="harvest_fixed",
                         slurm_params={"slurm_partition": "cpu", "cpus_per_task": 8, "mem_gb": 64, "timeout_min": 60},
                         log_dir="harvest_fixed", wait_for_completion=False)


def merge_fixed(topn=100):
    """Assemble per (marker, gene) top-N centroid features from the harvested partials → gal100."""
    import collections
    agg = collections.defaultdict(lambda: {"rank": [], "feat": []})
    for p in glob.glob(f"{FIX_PARTS}/*.npz"):
        z = np.load(p, allow_pickle=True)
        for k in z["keys"]:
            agg[k]["rank"].append(z[f"r::{k}"]); agg[k]["feat"].append(z[f"f::{k}"])
    n = 0
    for k, v in agg.items():
        mk, g = k.split("|"); rank = np.concatenate(v["rank"]); feat = np.concatenate(v["feat"])
        order = np.argsort(rank)[:topn]
        d = f"{EMB100}/{mk}/geneKO"; os.makedirs(d, exist_ok=True)
        np.savez(f"{d}/{g}.npz", features=feat[order].astype(np.float32)); n += 1
    print(f"merge_fixed: wrote {n} (marker,gene) gal100 files")
    return {"n": n}


def submit_build_cp():
    from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs
    genes = sorted(os.path.basename(f)[:-3] for f in glob.glob(f"{PV2['fluor']}/*.pt"))
    jobs = [{"name": f"g100cp_{i}", "func": build_gal100_cp, "kwargs": {"genes": genes[i:i + 40]}}
            for i in range(0, len(genes), 40)]
    print(f"[gal100 CP build] {len(jobs)} shards")
    submit_parallel_jobs(jobs, experiment="gal100_cp",
                         slurm_params={"slurm_partition": "cpu", "cpus_per_task": 8, "mem_gb": 48, "timeout_min": 60},
                         log_dir="gal100_cp", wait_for_completion=False)


def build_gal100(genes, topn=100):
    """Load each fluor gene .pt ONCE; write its top-N real embeddings for every marker×grain it appears in."""
    maps = _all_fluor_rankmaps()
    done = 0
    for gene in genes:
        p = f"{PV2['fluor']}/{gene}.pt"
        if not os.path.exists(p):
            continue
        segs, E, ch = _flatten_pt(p)
        for (mod, grain), rm in maps.items():
            r = rm.get(gene)
            if not r:
                continue
            keep = np.array([s in r for s in segs])
            if keep.sum() < 20:
                continue
            keep &= (ch == np.bincount(ch[keep]).argmax())          # marker's channel
            if keep.sum() < 20:
                continue
            ss, EE = segs[keep], E[keep]
            order = np.argsort([r[s] for s in ss])[:topn]
            d = f"{EMB100}/{mod}/{grain}"; os.makedirs(d, exist_ok=True)
            np.savez(f"{d}/{gene}.npz", features=EE[order].astype(np.float32))
        done += 1
    return {"genes": done}


def score_n100(mod, grain):
    """Assemble the top-100 real bank from the gal100 cache + score gen (fixed) with z-score AND CORAL."""
    gal = sorted(glob.glob(f"{EMB100}/{mod}/{grain}/*.npz"))
    names = [os.path.basename(f)[:-4] for f in gal]
    if len(names) < 3:
        return {"mod": mod, "grain": grain, "skip": "<3"}
    E = [np.load(f)["features"] for f in gal]
    cents = np.stack([e.mean(0) for e in E]); R = np.concatenate(E)
    dom = names[:DOM_GENES]; gd = []
    for g in dom:
        gd += _loadf(sorted(glob.glob(f"{EMB}/gen/{mod}/{grain}/{g}/a*.npz")))
    Gd = np.concatenate(gd)
    if len(Gd) > DOM_CAP:
        Gd = Gd[np.random.default_rng(0).choice(len(Gd), DOM_CAP, replace=False)]
    mu_r, sd_r = R.mean(0), R.std(0) + 1e-6; mu_g = Gd.mean(0)
    W = _sympow(np.cov(Gd.T), -0.5) @ _sympow(np.cov(R.T), 0.5)
    coral = lambda gv: _norm(((gv - mu_g) @ W) / sd_r)
    czb = _norm((cents - mu_r) / sd_r); bidx = {g: i for i, g in enumerate(names)}
    out = {}
    for g in names:
        fs = sorted(glob.glob(f"{EMB}/gen/{mod}/{grain}/{g}/a*.npz"), key=lambda p: int(os.path.basename(p)[1:-4]))
        if not fs:
            continue
        ais = [int(os.path.basename(p)[1:-4]) for p in fs]; al = [ALPHAS[a] for a in ais]
        gens = _loadf(fs); a0 = int(np.argmin(np.abs(np.array(al)))); g0 = gens[a0]
        mu_p, sd_p = g0.mean(0), g0.std(0) + 1e-6
        zscore = lambda gv, mu_p=mu_p, sd_p=sd_p: _norm((gv - mu_p) / sd_p)
        out[f"{mod}/{grain}/{g}"] = {"alphas": al, "z": _curve(gens, al, bidx[g], czb, zscore),
                                     "coral": _curve(gens, al, bidx[g], czb, coral)}
    os.makedirs(f"{OUT}/N100", exist_ok=True)
    json.dump(out, open(f"{OUT}/N100/{mod}__{grain}.json", "w"))
    return {"mod": mod, "grain": grain, "n": len(out)}


def submit_build():
    from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs
    genes = sorted(os.path.basename(f)[:-3] for f in glob.glob(f"{PV2['fluor']}/*.pt"))
    jobs = [{"name": f"g100_{i}", "func": build_gal100, "kwargs": {"genes": genes[i:i + 30]}}
            for i in range(0, len(genes), 30)]
    print(f"[gal100 build] {len(genes)} genes -> {len(jobs)} shards")
    submit_parallel_jobs(jobs, experiment="gal100_build",
                         slurm_params={"slurm_partition": "cpu", "cpus_per_task": 8, "mem_gb": 48, "timeout_min": 60},
                         log_dir="gal100_build", wait_for_completion=False)


def submit_score():
    from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs
    jobs = []
    for grain in ("geneKO", "complex"):
        for d in sorted(glob.glob(f"{EMB100}/*/{grain}")):
            mod = os.path.basename(os.path.dirname(d))
            if glob.glob(f"{d}/*.npz"):
                jobs.append({"name": f"s100_{mod[:16]}_{grain[0]}"[:40], "func": score_n100,
                             "kwargs": {"mod": mod, "grain": grain}})
    print(f"[score n100] {len(jobs)} marker shards")
    submit_parallel_jobs(jobs, experiment="score_n100",
                         slurm_params={"slurm_partition": "cpu", "cpus_per_task": 16, "mem_gb": 32, "timeout_min": 30},
                         log_dir="score_n100", wait_for_completion=False)


def build_gal_phase(genes, topn=1000):
    """Phase gal (top-1000 paper_v2_phase real per gene), built once per gene → EMB100/phase/geneKO."""
    ranks = _phase_ranks()
    d = f"{EMB100}/phase/geneKO"; os.makedirs(d, exist_ok=True)
    done = 0
    for g in genes:
        rr = _ranked_real("phase", g, ranks)
        if rr is not None:
            np.savez(f"{d}/{g}.npz", features=rr[:topn].astype(np.float32)); done += 1
    return {"genes": done}


def submit_build_phase():
    from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs
    genes = sorted(os.path.basename(f)[:-3] for f in glob.glob(f"{PV2['phase']}/*.pt"))
    jobs = [{"name": f"galp_{i}", "func": build_gal_phase, "kwargs": {"genes": genes[i:i + 50]}}
            for i in range(0, len(genes), 50)]
    print(f"[gal_phase] {len(genes)} genes -> {len(jobs)} shards")
    submit_parallel_jobs(jobs, experiment="gal_phase",
                         slurm_params={"slurm_partition": "cpu", "cpus_per_task": 8, "mem_gb": 48, "timeout_min": 120},
                         log_dir="gal_phase", wait_for_completion=False)


def score_nsweep(mod, grain):
    """Score gen (fixed) vs top-N centroids at N∈{100,500,1000}, reading the saved gal (top-1000) — no .pt reload."""
    gal = sorted(glob.glob(f"{EMB100}/{mod}/{grain}/*.npz"))
    names = [os.path.basename(f)[:-4] for f in gal]
    if len(names) < 3:
        return {"mod": mod, "grain": grain, "skip": "<3"}
    E = [np.load(f)["features"] for f in gal]
    dom = names[:DOM_GENES]; gd = []
    for g in dom:
        gd += _loadf(sorted(glob.glob(f"{EMB}/gen/{mod}/{grain}/{g}/a*.npz")))
    Gd = np.concatenate(gd)
    if len(Gd) > DOM_CAP:
        Gd = Gd[np.random.default_rng(0).choice(len(Gd), DOM_CAP, replace=False)]
    R = np.concatenate([e[:500] for e in E[:DOM_GENES]])
    mu_r, sd_r = R.mean(0), R.std(0) + 1e-6; mu_g = Gd.mean(0)
    W = _sympow(np.cov(Gd.T), -0.5) @ _sympow(np.cov(R.T), 0.5)
    coral = lambda gv: _norm(((gv - mu_g) @ W) / sd_r)
    Ei = {g: e for g, e in zip(names, E)}
    for n in NS:
        bank = np.stack([Ei[g][:n].mean(0) for g in names])
        czb = _norm((bank - mu_r) / sd_r); bidx = {g: i for i, g in enumerate(names)}
        out = {}
        for g in names:
            fs = sorted(glob.glob(f"{EMB}/gen/{mod}/{grain}/{g}/a*.npz"), key=lambda p: int(os.path.basename(p)[1:-4]))
            if not fs:
                continue
            ais = [int(os.path.basename(p)[1:-4]) for p in fs]; al = [ALPHAS[a] for a in ais]
            gens = _loadf(fs); a0 = int(np.argmin(np.abs(np.array(al)))); g0 = gens[a0]
            mu_p, sd_p = g0.mean(0), g0.std(0) + 1e-6
            zscore = lambda gv, mu_p=mu_p, sd_p=sd_p: _norm((gv - mu_p) / sd_p)
            out[f"{mod}/{grain}/{g}"] = {"alphas": al, "z": _curve(gens, al, bidx[g], czb, zscore),
                                        "coral": _curve(gens, al, bidx[g], czb, coral)}
        os.makedirs(f"{OUT}/N{n}", exist_ok=True)
        json.dump(out, open(f"{OUT}/N{n}/{mod}__{grain}.json", "w"))
    return {"mod": mod, "grain": grain, "n": len(names)}


def submit_score_nsweep(only=None):
    from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs
    jobs = []
    for grain in ("geneKO", "complex"):
        for d in sorted(glob.glob(f"{EMB100}/*/{grain}")):
            mod = os.path.basename(os.path.dirname(d))
            if (only is None or mod in only) and glob.glob(f"{d}/*.npz"):
                jobs.append({"name": f"sw_{mod[:15]}_{grain[0]}"[:40], "func": score_nsweep,
                             "kwargs": {"mod": mod, "grain": grain}})
    print(f"[score nsweep] {len(jobs)} shards")
    submit_parallel_jobs(jobs, experiment="score_nsweep",
                         slurm_params={"slurm_partition": "cpu", "cpus_per_task": 16, "mem_gb": 40, "timeout_min": 40},
                         log_dir="score_nsweep", wait_for_completion=False)


def _jobs():
    jobs = []
    for grain in ("geneKO", "complex"):
        if os.path.exists(f"{PV2['phase']}"):
            g = sorted(os.path.basename(f)[:-3] for f in glob.glob(f"{PV2['phase']}/*.pt"))
            for i in range(0, len(g), 200):
                jobs.append({"name": f"ns_phase_{grain}_{i}"[:40], "func": shard,
                             "kwargs": {"mod": "phase", "grain": grain, "genes": g[i:i + 200], "tag": str(i)}})
    for p in sorted(glob.glob(f"{FLUOR_SHAP}/geneKO/*.parquet")):
        mod = os.path.basename(p)[:-8]
        jobs.append({"name": f"ns_{mod[:16]}_g"[:40], "func": shard, "kwargs": {"mod": mod, "grain": "geneKO"}})
    for p in sorted(glob.glob(f"{FLUOR_SHAP}/complex/*.parquet")):
        mod = os.path.basename(p)[:-8]
        jobs.append({"name": f"ns_{mod[:16]}_c"[:40], "func": shard, "kwargs": {"mod": mod, "grain": "complex"}})
    return jobs


def submit():
    from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs
    jobs = _jobs()
    print(f"[coral-nsweep] {len(jobs)} shards")
    submit_parallel_jobs(jobs, experiment="coral_nsweep",
                         slurm_params={"slurm_partition": "cpu", "cpus_per_task": 16, "mem_gb": 64, "timeout_min": 60},
                         log_dir="coral_nsweep", wait_for_completion=False)


def merge():
    import collections
    print(f"{'N':>5} {'method':>7} {'group':>6} {'n':>6} {'f=1':>7} {'medPeak':>8}")
    for n in NS:
        M = {}
        for p in glob.glob(f"{OUT}/N{n}/*__*.json"):
            M.update(json.load(open(p)))
        json.dump(M, open(f"{OUT}/unified_N{n}.json", "w"))
        grp = collections.defaultdict(lambda: {"n": 0, "z1": 0, "c1": 0, "zp": [], "cp": []})
        for k, v in M.items():
            for key in (("phase" if k.split("/")[0] == "phase" else "fluor"), "ALL"):
                g = grp[key]; g["n"] += 1
                g["z1"] += v["z"]["f"] == 1.0; g["c1"] += v["coral"]["f"] == 1.0
                g["zp"].append(v["z"]["peak"]); g["cp"].append(v["coral"]["peak"])
        for key in ("phase", "fluor", "ALL"):
            g = grp[key]
            if not g["n"]:
                continue
            print(f"{n:>5} {'z-score':>7} {key:>6} {g['n']:>6} {100*g['z1']/g['n']:>6.1f}% {np.median(g['zp']):>8.3f}")
            print(f"{n:>5} {'CORAL':>7} {key:>6} {g['n']:>6} {100*g['c1']/g['n']:>6.1f}% {np.median(g['cp']):>8.3f}")
    print(f"\ncached: {OUT}/N{{100,500,1000}}/ + unified_N*.json")


if __name__ == "__main__":
    print({"submit": submit, "submit_build": submit_build, "submit_build_cp": submit_build_cp,
           "submit_harvest_fixed": submit_harvest_fixed, "merge_fixed": merge_fixed,
           "submit_build_phase": submit_build_phase, "submit_score_nsweep": submit_score_nsweep,
           "submit_score": submit_score, "merge": merge}[sys.argv[1] if sys.argv[1:] else "submit"]())
