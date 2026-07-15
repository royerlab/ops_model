"""Per-marker PC-strip build (Kyle-style) from paper_v2 cell_dino_features_v2.

For a viewer marker we refit PCA on that marker's per-cell CellDINO features (which carry the cell's
position, exactly like Kyle's .pt inputs), bin cells low->high along each PC, and crop that marker's own
fluor channel from phenotyping_v3.zarr with the blue negative cell-mask overlay. Gene chips/loadings come
from the same refit (gene-mean PC scores), so strips + chips are self-consistent.

Output is ADDITIVE and isolated (does not touch the phase pcs/ cache that is syncing to S3):
    viewer_assets/pcs/markers/<slug>/index.json   (same schema as the phase pcs/index.json)
    viewer_assets/pcs/markers/<slug>/crops/pc###_bin##_row#.png

  python -m ops_model.models.attention.diffex.viewer.build_pcs_marker --marker "autophagosome_MAP1LC3B"
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

from . import catalog as C
from . import marker_leaves as ML
from .build_pc_crops_masked import CROP_SIZE, _crop, _is_blank, _render, _zarr_patch

FOPS = "/hpc/projects/intracellular_dashboard/fast_ops"
PCS_OUT = f"{C.OUT}/viewer_assets/pcs/markers"
N_PCS, N_BINS, N_ROWS = 40, 15, 3          # PCs shown; strip bins; cells per bin
FIT_N, SEL_N = 120_000, 300_000            # cells subsampled per experiment for PCA fit / representative selection


def _exp_dir(exp):
    d = sorted(glob.glob(f"{FOPS}/{exp}_*"))
    return d[0] if d else None


def _features_dir(leaf):   # CP / 4i channel CellDINO features live in adjacent _cp / _4i folders
    return "cell_dino_features_v2_cp" if leaf.endswith("_cp") else "cell_dino_features_v2_4i" if leaf.endswith("_4i") else "cell_dino_features_v2"


def _marker_meta(marker_channel):
    """Resolve a viewer marker → (reporter, channel_name, [exp prefixes]) using its leaf + a features_processed file."""
    import h5py
    leaf = ML.resolve_leaf(marker_channel)
    if not leaf:
        raise SystemExit(f"no paper_v2 leaf for {marker_channel}")
    manifest = f"{ML.leaf_dir(leaf)}/downsampled_manifest.csv"
    fdir = _features_dir(leaf)
    import csv
    exps = []
    with open(manifest) as f:
        for row in csv.DictReader(f):
            exps = [e.strip() for e in row["experiments"].split(",")]
    # find the marker's features_processed_<REPORTER>.h5ad (non-Phase) in the first available experiment
    reporter = channel = None
    for e in exps:
        d = _exp_dir(e)
        if not d:
            continue
        cand = [f for f in glob.glob(f"{d}/3-assembly/{fdir}/anndata_objects/features_processed_*.h5ad")
                if "_Phase" not in f]
        for fp in cand:
            with h5py.File(fp, "r") as h:
                ch = h["uns"]["channel"][()]
                ch = ch.decode() if isinstance(ch, bytes) else ch
            reporter = os.path.basename(fp)[len("features_processed_"):-len(".h5ad")]
            channel = ch
            break
        if reporter:
            break
    if not reporter:
        raise SystemExit(f"no features_processed for {marker_channel} ({leaf})")
    return leaf, reporter, channel, exps, fdir


def _fp(exp, reporter, fdir="cell_dino_features_v2"):
    d = _exp_dir(exp)
    return f"{d}/3-assembly/{fdir}/anndata_objects/features_processed_{reporter}.h5ad" if d else None


def _channel_index(exp, channel_name):
    """Index of `channel_name` in this experiment's phenotyping_v3.zarr channel list."""
    import zarr
    g = zarr.open_group(f"{_exp_dir(exp)}/3-assembly/phenotyping_v3.zarr/A/1/0", mode="r")
    labels = [c["label"] for c in dict(g.attrs["ome"])["omero"]["channels"]]
    return labels.index(channel_name)


def _load(fp, cols, n=None, rng=None):
    """Read features X (subsampled) + obs columns from a features_processed h5ad."""
    import h5py
    with h5py.File(fp, "r") as h:
        N = h["X"].shape[0]
        idx = np.arange(N) if (n is None or n >= N) else np.sort(rng.choice(N, n, replace=False))
        X = h["X"][idx].astype(np.float64)
        out = {}
        for c in cols:
            g = h["obs"][c]
            if isinstance(g, h5py.Group):   # categorical
                cats = [x.decode() if isinstance(x, bytes) else x for x in g["categories"][:]]
                out[c] = np.array([cats[i] for i in g["codes"][idx]])
            else:
                out[c] = g[idx]
    return X, out


def build_marker(marker_channel, out=PCS_OUT, with_enrich=True):
    from PIL import Image
    import zarr
    _zarr_patch()
    leaf, reporter, channel, exps, fdir = _marker_meta(marker_channel)
    slug = ML._norm(marker_channel)
    print(f"[pcm] {marker_channel} → leaf {leaf}, reporter {reporter}, channel {channel}, {len(exps)} exps")
    rng = np.random.RandomState(0)

    # 1) fit PCA on z-scored features (per-experiment z-score), pooled subsample
    from sklearn.decomposition import PCA
    zparts, stats = [], {}
    for e in exps:
        fp = _fp(e, reporter, fdir)
        if not fp or not os.path.exists(fp):
            continue
        X, _ = _load(fp, [], FIT_N, rng)
        mu, sd = X.mean(0), X.std(0) + 1e-8
        stats[e] = (mu, sd)
        zparts.append((X - mu) / sd)
    pca = PCA(n_components=N_PCS, svd_solver="randomized", random_state=0).fit(np.vstack(zparts))
    ev = (pca.explained_variance_ratio_ * 100).round(3).tolist()
    print(f"[pcm] PCA fit; PC0-4 var {ev[:5]}")

    # 2) selection sample (with positions) across experiments → PC scores
    scores, well, xs, ys, pert, exp_of = [], [], [], [], [], []
    for e in exps:
        fp = _fp(e, reporter, fdir)
        if e not in stats or not fp:
            continue
        X, obs = _load(fp, ["well", "x_position", "y_position", "perturbation"], SEL_N, rng)
        mu, sd = stats[e]
        sc = ((X - mu) / sd - pca.mean_) @ pca.components_.T
        scores.append(sc); xs.append(obs["x_position"]); ys.append(obs["y_position"])
        pert.append(obs["perturbation"]); exp_of += [e] * len(sc)
        well.append(np.array([w.split("_")[0] for w in obs["well"]]))
    S = np.vstack(scores); X0 = np.concatenate(xs); Y0 = np.concatenate(ys)
    W = np.concatenate(well); P = np.concatenate(pert); E = np.array(exp_of)
    print(f"[pcm] selection sample {S.shape}")

    # 3) gene loadings = gene-mean PC scores (self-consistent with the strips)
    genes = sorted(set(P))
    gmean = {g: S[P == g].mean(0) for g in genes}
    geneData = {}
    for g in genes:
        prof = gmean[g]
        order = np.argsort(-np.abs(prof))[:15]
        geneData[g] = {"top_pcs": [{"pc": int(p) + 1, "score": round(float(prof[p]), 3)} for p in order],
                       "profile": [round(float(v), 3) for v in prof]}

    # 4) representatives per PC×bin + crop the marker channel
    crops_dir = f"{out}/{slug}/crops"; os.makedirs(crops_dir, exist_ok=True)
    half = CROP_SIZE // 2
    zc, ci = {}, {}     # zarr handle cache, channel-index cache
    pcData = {}; ok = 0
    for p in range(N_PCS):
        sc = S[:, p]
        tgt = np.percentile(sc, np.linspace(2, 98, N_BINS))
        strip = []
        high = sorted(genes, key=lambda g: -gmean[g][p])[:15]
        low = sorted(genes, key=lambda g: gmean[g][p])[:15]
        for bi, t in enumerate(tgt):
            near = np.argsort(np.abs(sc - t))[:N_ROWS * 4]   # candidates; take first N_ROWS that crop OK
            cells = []
            for j in near:
                if len(cells) >= N_ROWS:
                    break
                e = E[j]; w = W[j]; x, y = int(round(X0[j])), int(round(Y0[j]))
                r, col = w.split("/")[0], w.split("/")[1]
                key = (e, r, col)
                if key not in zc:
                    b = f"{_exp_dir(e)}/3-assembly/phenotyping_v3.zarr/{r}/{col}/0"
                    try:
                        zc[key] = (zarr.open(f"{b}/0", mode="r"), zarr.open(f"{b}/labels/cell_seg/0", mode="r"))
                        ci[e] = ci.get(e) or _channel_index(e, channel)
                    except Exception:
                        zc[key] = None
                if zc[key] is None:
                    continue
                img, seg = zc[key]
                try:
                    ph = _crop(img, ci[e], x, y, half)
                    if _is_blank(ph):
                        continue
                    fn = f"pc{p:03d}_bin{bi:02d}_row{len(cells)}.png"
                    Image.fromarray(_render(ph, _crop(seg, None, x, y, half), half)).save(f"{crops_dir}/{fn}")
                    cells.append({"gene": str(P[j]), "score": round(float(sc[j]), 2), "experiment": e,
                                  "well": w, "x": round(float(X0[j]), 1), "y": round(float(Y0[j]), 1),
                                  "has_crop": True, "img": fn})
                    ok += 1
                except Exception:
                    continue
            while len(cells) < N_ROWS:
                cells.append(None)
            strip.append({"cells": cells})
        pcData[str(p + 1)] = {"pc": p + 1, "explained_variance": ev[p], "strip": strip,
                              "high_genes": [{"gene": g, "score": round(float(gmean[g][p]), 3)} for g in high],
                              "low_genes": [{"gene": g, "score": round(float(gmean[g][p]), 3)} for g in low]}
        if (p + 1) % 10 == 0:
            print(f"[pcm]   {p + 1}/{N_PCS} PCs, {ok} crops")

    index = {"overview": {"n_genes": len(genes), "n_pcs": N_PCS, "n_bins": N_BINS, "n_rows": N_ROWS,
                          "crop_size": CROP_SIZE, "marker": marker_channel, "channel": channel,
                          "total_variance": round(sum(ev), 1), "explained_variance": ev},
             "pcData": pcData, "geneData": geneData, "geneNames": genes}
    with open(f"{out}/{slug}/index.json", "w") as f:
        json.dump(index, f)
    print(f"[pcm] {marker_channel}: {ok} crops, {len(genes)} genes -> {out}/{slug}/index.json")
    if with_enrich:
        from .build_pcs import build_enrichment
        build_enrichment(out=f"{out}/{slug}")          # ontology enrichment (speedrichr) + term sizes
        build_marker_features(slug, out)               # morphometric features
    return slug


GENE_FEATURE_MEANS = "/hpc/projects/icd.fast.ops/analysis/pc_feature_correlation/phase_only/gene_feature_means.h5ad"


def build_marker_features(slug, out=PCS_OUT):
    """Per-marker morphometric features: correlate the marker's gene PC loadings vs the OP/CP gene feature means.
    Strips + loadings share one PCA here, so no sign flip is needed (flips = +1, dirConf = True)."""
    import anndata as ad
    import pandas as pd
    from .build_pc_features import _load_helpers, write_features_json
    idx = json.load(open(f"{out}/{slug}/index.json"))
    n_pcs = idx["overview"]["n_pcs"]
    gd = idx["geneData"]
    genes = [g for g in idx["geneNames"] if g in gd]
    P = pd.DataFrame([gd[g]["profile"][:n_pcs] for g in genes], index=genes,
                     columns=[f"Phase_PC{i}" for i in range(n_pcs)])
    fm = ad.read_h5ad(GENE_FEATURE_MEANS)
    fm_genes = [str(g) for g in fm.obs_names]
    common = [g for g in genes if g in set(fm_genes)]
    P = P.loc[common]
    F = pd.DataFrame(np.asarray(fm[[g for g in common], :].X), index=common, columns=[str(v) for v in fm.var_names])
    H = _load_helpers()
    R = H.correlate(P, F)                       # n_pcs × feature signed Pearson r
    TF, _ = H.tfidf_distinctive(R)              # tf-idf distinctiveness
    R.index = [f"Phase_PC{i}" for i in range(n_pcs)]; TF.index = R.index
    flips = {p: 1.0 for p in range(1, n_pcs + 1)}; conf = {p: True for p in range(1, n_pcs + 1)}
    return write_features_json(R, TF, flips, conf, n_pcs, f"{out}/{slug}", H)


def build_marker_layouts(mont_dir=None):
    """Per-marker Live-mode layouts: reposition dots by each marker's own embedding (X_umap/X_phate), reusing
    the phase layout's rich gene annotations for color-by. Cheap (no tiles) → Live renders these per marker."""
    import anndata as ad
    from . import build_umap_montage as BM
    mont_dir = mont_dir or f"{C.OUT}/viewer_assets/_montage"
    phase = json.load(open(f"{mont_dir}/layout_umap.json"))
    ann_of = {g["g"]: {k: v for k, v in g.items() if k not in ("g", "nx", "ny")} for g in phase["genes"]}
    cfields = phase["color_fields"]
    import re
    jss = lambda s: re.sub(r"[^A-Za-z0-9]", "_", str(s)).strip("_")   # matches the app's jsSlug (montage modality)
    n = 0
    for mk in _all_markers():
        h5 = ML.embedding_h5ad(mk); slug = jss(mk)
        a = ad.read_h5ad(h5)
        for emb in ("umap", "phate"):
            c = BM._embed_coords(a, emb); lo = c.min(0); rng = c.max(0) - lo; rng[rng == 0] = 1
            genes = []
            for i, g in enumerate(a.obs["perturbation"]):
                g = str(g)
                rec = {"g": g, "nx": float((c[i, 0] - lo[0]) / rng[0]), "ny": float((c[i, 1] - lo[1]) / rng[1])}
                rec.update(ann_of.get(g, {}))
                genes.append(rec)
            with open(f"{mont_dir}/layout_{slug}_{emb}.json", "w") as f:
                json.dump({"embedding": emb, "color_fields": cfields, "genes": genes}, f)
        n += 1
    print(f"[pcm] wrote per-marker Live layouts for {n} markers → {mont_dir}/layout_<slug>_<emb>.json")


def build_marker_job(marker_channel, out=PCS_OUT):
    """SLURM job: PC strips + morphometric features for one marker (no external calls)."""
    slug = build_marker(marker_channel, out, with_enrich=False)
    build_marker_features(slug, out)
    return {"marker": marker_channel, "slug": slug}


def enrich_shard(markers, out=PCS_OUT):
    """SLURM job: ontology enrichment (speedrichr) for a handful of markers, serially (few shards = throttled)."""
    from .build_pcs import build_enrichment
    done = []
    for mk in markers:
        slug = ML._norm(mk)
        try:
            build_enrichment(out=f"{out}/{slug}"); done.append(slug)
        except Exception as e:
            print(f"[pcm] enrich failed {slug}: {e}")
    return {"done": done}


def _all_markers():
    m = json.load(open(f"{C.OUT}/viewer_assets/manifest.json"))
    mks = [x["marker_channel"] for x in m["markers"] if x.get("marker_channel") and ML.resolve_leaf(x["marker_channel"])]
    return sorted(set(mks))


def submit_strips(out=PCS_OUT):
    """Fan out PC strips + features for all 55 markers (parallel, no external calls)."""
    from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs
    mks = _all_markers()
    jobs = [{"name": f"pcm_{ML._norm(mk)[:18]}", "func": build_marker_job, "kwargs": {"marker_channel": mk, "out": out}} for mk in mks]
    print(f"[pcm] submitting {len(jobs)} marker strip+feature jobs")
    submit_parallel_jobs(jobs, experiment="pcs_markers",
                         slurm_params={"slurm_partition": "cpu", "cpus_per_task": 8, "mem_gb": 64, "timeout_min": 150},
                         log_dir="pcs_markers", wait_for_completion=True)


def submit_enrich(out=PCS_OUT, n_shards=4):
    """Throttled ontology enrichment for all markers (few shards so speedrichr isn't hammered)."""
    from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs
    mks = _all_markers()
    shards = [mks[i::n_shards] for i in range(n_shards)]
    jobs = [{"name": f"pcm_enrich_{i}", "func": enrich_shard, "kwargs": {"markers": s, "out": out}} for i, s in enumerate(shards) if s]
    print(f"[pcm] submitting {len(jobs)} throttled enrichment shards for {len(mks)} markers")
    submit_parallel_jobs(jobs, experiment="pcs_markers_enrich",
                         slurm_params={"slurm_partition": "cpu", "cpus_per_task": 4, "mem_gb": 16, "timeout_min": 300},
                         log_dir="pcs_markers_enrich", wait_for_completion=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--marker", help="viewer marker_channel → build one marker (strips+enrich+features)")
    ap.add_argument("--features-slug", help="build per-marker features.json for an already-built slug")
    ap.add_argument("--submit-strips", action="store_true", help="SLURM fan-out strips+features for all 55")
    ap.add_argument("--submit-enrich", action="store_true", help="SLURM throttled enrichment for all 55")
    ap.add_argument("--out", default=PCS_OUT)
    a = ap.parse_args()
    if a.submit_strips:
        submit_strips(a.out)
    elif a.submit_enrich:
        submit_enrich(a.out)
    elif a.features_slug:
        build_marker_features(a.features_slug, a.out)
    else:
        build_marker(a.marker, a.out)
