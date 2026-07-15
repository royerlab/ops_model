"""Build the DiffEx viewer cache — reproducible, version-controlled entrypoint (replaces the
one-off scratchpad drivers). All target selection comes from `catalog.py`.

  python -m ops_model.models.attention.diffex.viewer.submit seed              # per-marker NTC traversals
  python -m ops_model.models.attention.diffex.viewer.submit anchors --k 5     # A→B anchor pairs
  python -m ops_model.models.attention.diffex.viewer.submit manifest          # rebuild manifest.json (local)
  python -m ops_model.models.attention.diffex.viewer.submit montage --cell 0 --alpha 2   # harvest cache -> UMAP montage zarr
"""
from __future__ import annotations

import argparse

from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs

from ..classifier.config import slugify
from . import catalog as C
from .build_umap_montage import build_montage_grid, build_montage_web
from .precompute import build_manifest, precompute_marker, precompute_target

PHASE_CK = f"{C.DD}/phase_v1/diffae_best.pt"
UMAP_H5AD = ("/hpc/projects/icd.fast.ops/organelle_attribution/pca_optimized_v0.3/cell_dino/"
             "zscore_per_exp/paper_v2/phase_only/fixed_80%/cosine/gene_embedding_pca_optimized.h5ad")
PHASE_COMPLEXES = ["40S cytosolic small ribosomal subunit", "60S cytosolic large ribosomal subunit",
                   "DNA-directed RNA polymerase II complex", "Chaperonin-containing T-complex", "SF3B complex"]
FLUOR_EBI_CSV = "/hpc/projects/icd.fast.ops/models/alex_lin_attention/v4/pma_fluorescent_cells_ebi_all.csv"


def _gpu(**kw):
    return {"slurm_partition": "gpu", "gpus_per_node": 1, "cpus_per_task": 12, "mem_gb": 64,
            "slurm_constraint": "[a100_80|h100|h200|6000_blackwell]", **kw}


def _job(name, func, kwargs, stage):
    return {"name": name, "func": func, "kwargs": kwargs, "metadata": {"stage": stage}}


def cmd_seed(args):
    """Per-marker NTC traversals: every complete fluorescent marker (top-N genes) + phase geneKO + phase complex."""
    dist = C.dist_matrix(); jobs = []
    for d, mc, ch in C.complete_markers(min_ep=args.min_ep):
        rep = C.rep_of(dist, mc)
        if not rep or rep not in dist.columns:
            continue
        if args.map_thr is not None:                         # full buildout: ALL genes the marker distinguishes >= thr (0 → all ~1000)
            sc = dist[rep]
            tg = [g for g in sc.index[sc >= args.map_thr] if not str(g).startswith("NTC")]
        else:
            tg = C.top_genes(dist, rep, args.n)
        if tg:
            jobs.append(_job(f"pm_{slugify(mc)[:20]}", precompute_marker,
                             dict(grain="geneKO", targets=tg, marker_channel=mc, channel=ch,
                                  ckpt=f"{C.DD}/{d}/diffae_best.pt", out_root=C.OUT, load_workers=12,
                                  score=not args.no_score), "seed"))
    for d, mc, ch, rep in C.NO_PMA_MARKERS:                  # no-PMA markers: build from features_processed anndata
        r = C.rep_of(dist, mc)
        if not r or r not in dist.columns:
            continue
        if args.map_thr is not None:
            sc = dist[r]; tg = [g for g in sc.index[sc >= args.map_thr] if not str(g).startswith("NTC")]
        else:
            tg = C.top_genes(dist, r, args.n)
        if tg:
            jobs.append(_job(f"pm_nopma_{slugify(mc)[:16]}", precompute_marker,
                             dict(grain="geneKO", targets=tg, marker_channel=mc, channel=ch,
                                  ckpt=f"{C.DD}/{d}/diffae_best.pt", out_root=C.OUT, load_workers=12,
                                  fluor_rows_h5ad=C.NO_PMA_H5AD.format(rep=rep),
                                  score=not args.no_score), "seed"))
    if args.map_thr is None:                                 # phase already fully built — only (re)seed with top-N mode
        jobs.append(_job("pm_phase_geneKO", precompute_marker,
                         dict(grain="geneKO", targets=C.top_genes(dist, "Phase", args.n + 4),
                              ckpt=PHASE_CK, out_root=C.OUT, load_workers=12), "seed"))
        jobs.append(_job("pm_phase_complex", precompute_marker,
                         dict(grain="complex", targets=PHASE_COMPLEXES, ckpt=PHASE_CK, out_root=C.OUT, load_workers=12), "seed"))
    tgt = sum(len(j["kwargs"]["targets"]) for j in jobs)
    print(f"seed: {len(jobs)} per-marker jobs, {tgt} total targets"
          + (f" (mAP>={args.map_thr} filter)" if args.map_thr is not None else f" (top-{args.n})"))
    sync = getattr(args, "sync", False)                  # --sync: wait, then refresh manifest/attention/montages
    sp = _gpu(timeout_min=args.timeout)
    if args.parallel is not None:                        # default None = no concurrency cap (all markers at once)
        sp["slurm_array_parallelism"] = args.parallel
    submit_parallel_jobs(jobs_to_submit=jobs, experiment="diffex_gifs",
                         slurm_params=sp, log_dir="diffex_gifs", wait_for_completion=sync,
                         post_completion_callback=(lambda *_: run_full_sync()) if sync else None)


def cmd_anchors(args):
    """A→B anchor traversals: all ordered pairs among each marker's top-K classes (phase + fluor)."""
    dist = C.dist_matrix(); jobs = []
    markers = C.complete_markers()
    if args.markers:
        markers = [m for m in markers if m[1] in args.markers or slugify(m[1]) in args.markers]
    for d, mc, ch in markers:
        rep = C.rep_of(dist, mc)
        top = C.top_genes(dist, rep, args.k) if rep else []
        for a in top:
            for b in top:
                if a == b:
                    continue
                jobs.append(_job(f"ab_{slugify(mc)[:12]}_{a[:6]}_{b[:6]}", precompute_target,
                                 dict(grain="geneKO", target=b, control=a, marker_channel=mc, channel=ch,
                                      ckpt=f"{C.DD}/{d}/diffae_best.pt", out_root=C.OUT, load_workers=12), "anchors"))
    print(f"anchors: {len(jobs)} A→B pair jobs across {len(markers)} markers")
    submit_parallel_jobs(jobs_to_submit=jobs, experiment="diffex_gifs",
                         slurm_params=_gpu(timeout_min=45, slurm_array_parallelism=args.parallel),
                         log_dir="diffex_gifs", wait_for_completion=False)


def cmd_manifest(args):
    """Rebuild manifest.json in place (dist mAP for sorting + gene/complex descriptions). Local, no SLURM.
    Also writes gene_desc.json (ALL genes) so the viewer shows info for genes not yet cached as targets."""
    import json
    va = f"{C.OUT}/viewer_assets"
    dm = C.desc_map()
    build_manifest(C.OUT, dist_map=C.dist_map_for_assets(va), desc_map=dm)
    open(f"{va}/gene_desc.json", "w").write(json.dumps(dm))
    print(f"[viewer] gene_desc.json: {len(dm)} entries")


def cmd_montage(args):
    """Per-marker UMAP montage: place each gene's cached α-frame at its gene-UMAP coord. Layout is ALWAYS
    the shared phase gene embedding (UMAP_H5AD); only the images swap per marker (modality). ONE SLURM job
    per discrete montage (marker × emb × cell × α) for maximal concurrency; content-aware skip at submit
    time so only stale montages are even queued. No decode/re-embed — reads the traversal cache. CPU-only."""
    import glob
    import os
    import shutil
    import time
    from pathlib import Path
    va = f"{C.OUT}/viewer_assets"
    # sweep orphan montage zarrs (transient intermediates now live in _montage_zarr, outside viewer_assets;
    # each job deletes its own, but killed jobs leave them). Also sweep the legacy viewer_assets/_montage
    # location for old stragglers. Skip any <10 min old so a concurrently-running build isn't disturbed.
    now = time.time(); freed = 0
    for zdir in (f"{C.OUT}/_montage_zarr", f"{va}/_montage"):
        for z in glob.glob(f"{zdir}/*.zarr"):
            if now - os.path.getmtime(z) > 600:
                shutil.rmtree(z, ignore_errors=True); freed += 1
    if freed:
        print(f"[montage] swept {freed} orphan zarr intermediates")
    mods = ["phase"]                                     # phase + markers that have geneKO traversal frames
    for mdir in sorted(glob.glob(f"{va}/*/geneKO")):
        mod = Path(mdir).parent.name
        if mod != "phase" and any(e.is_dir() for e in os.scandir(mdir)):   # cheap: ≥1 gene dir (don't enumerate all frames)
            mods.append(mod)
    if args.markers:
        mods = [m for m in mods if m in args.markers or slugify(m) in args.markers]
    jobs = []                                            # one job PER (marker, emb, cell, α) = a discrete unit
    for mod in mods:
        gk = f"{va}/{mod}/geneKO"
        cache_mtime = os.path.getmtime(gk) if os.path.isdir(gk) else 0   # bumps when a new gene traversal lands
        for emb in args.embeddings:
            for cell in args.cells:
                for a in args.alphas:
                    oz = f"{va}/_montage/{mod}_geneKO_{emb}_cell{cell}_a{a:g}.zarr"
                    tj = f"{oz[:-5]}_tiles/tiles.json"
                    if not args.force and os.path.exists(tj) and os.path.getmtime(tj) >= cache_mtime:
                        continue                         # already reflects the current cache → don't even queue it
                    jobs.append(_job(f"mtg_{mod[:10]}_{emb[:2]}_c{cell}_a{a:g}", build_montage_web,
                                     dict(h5ad=UMAP_H5AD, out_zarr=oz, cell=cell, alpha=a, embedding=emb, modality=mod), "montage"))
    print(f"montage: {len(jobs)} discrete jobs across {len(mods)} markers "
          f"(≤{len(mods) * len(args.embeddings) * len(args.cells) * len(args.alphas)} combos; skipped up-to-date)")
    submit_parallel_jobs(jobs_to_submit=jobs, experiment="diffex_gifs",
                         slurm_params={"slurm_partition": "cpu", "cpus_per_task": 4, "mem_gb": 24, "timeout_min": 60,
                                       "slurm_array_parallelism": args.parallel},
                         log_dir="diffex_gifs", wait_for_completion=getattr(args, "wait", False))


def run_full_sync():
    """Refresh the entire viewer from the CURRENT cache (all incremental): manifest → attention render →
    per-marker montages (full cell×α×emb grid). Safe to re-run; only new/missing assets are built.
    Top-level (picklable) so it can be a submitit job func or a seed post-completion callback."""
    import argparse
    from .build_attention_heads import submit as attn_submit
    # content-aware montage skip handles freshness (rebuilds only montages older than the marker's cache)
    ns = argparse.Namespace(cells=list(range(20)), alphas=[1., 2., 3., 4., 5.],
                            embeddings=["umap", "phate"], markers=None, force=False, wait=True, parallel=100)
    print("[sync] 1/3 manifest"); cmd_manifest(ns)
    print("[sync] 2/3 attention-head render"); attn_submit(parallel=40)      # waits + rebuilds index.json
    print("[sync] 3/3 montages"); cmd_montage(ns)                            # waits
    print("[sync] viewer refreshed")
    return "sync complete"


def cmd_sync(args):
    """Make the viewer current. Inline by default; with --after <jobids>, submit a SLURM gate job that
    runs the refresh automatically once those (seed) jobs finish (afterany dependency)."""
    if getattr(args, "after", None):
        ids = ":".join(str(j) for j in args.after)
        print(f"[sync] gating on afterany:{ids} → will refresh when the seed build finishes")
        submit_parallel_jobs(
            jobs_to_submit=[{"name": "viewer_sync", "func": run_full_sync, "kwargs": {}, "metadata": {"stage": "sync"}}],
            experiment="diffex_sync",
            slurm_params={"slurm_partition": "cpu", "cpus_per_task": 4, "mem_gb": 16, "timeout_min": 600,
                          "slurm_additional_parameters": {"dependency": f"afterany:{ids}"}},
            log_dir="diffex_sync", wait_for_completion=False)
    else:
        run_full_sync()


def _chunks(lst, n):
    return [lst[i:i + n] for i in range(0, len(lst), n)]


def cmd_fluor_complex(args):
    """NTC-anchored complex traversals (all EBI complexes) for every complete fluorescent marker."""
    cx = C.ebi_complexes()
    markers = C.complete_markers()
    if args.markers:
        markers = [m for m in markers if m[1] in args.markers or slugify(m[1]) in args.markers]
    jobs = [_job(f"fcx_{slugify(mc)[:18]}", precompute_marker,
                 dict(grain="complex", targets=cx, marker_channel=mc, channel=ch, fluor_csv=C.EBI_FLUOR_CSV,
                      ckpt=f"{C.DD}/{d}/diffae_best.pt", out_root=C.OUT, load_workers=12, batch=args.batch), "fluor_complex")
            for d, mc, ch in markers]
    print(f"fluor-complex: {len(jobs)} markers × {len(cx)} complexes (batch={args.batch}, no constraint/cap)")
    sp = {"slurm_partition": "gpu", "gpus_per_node": 1, "cpus_per_task": 12, "mem_gb": 64, "timeout_min": 720}
    submit_parallel_jobs(jobs_to_submit=jobs, experiment="diffex_gifs", slurm_params=sp,
                         log_dir="diffex_gifs", wait_for_completion=False)


def cmd_phase_full(args):
    """Full phase NTC cache on v1: all ~1000 geneKOs + all EBI complexes, chunked across GPU jobs.
    No GPU-type constraint and no concurrency cap — batch is shrunk (default 24 → ~28GB peak) so it
    fits any GPU (incl. the plentiful 40GB a100 / 48GB l40s|a40|a6000), maximizing availability."""
    genes, cx = C.all_genes(), C.ebi_complexes()
    jobs = [_job(f"phg_{i}", precompute_marker,
                 dict(grain="geneKO", targets=ch, ckpt=PHASE_CK, out_root=C.OUT, load_workers=12, batch=args.batch), "phase_full")
            for i, ch in enumerate(_chunks(genes, args.chunk_size))]
    jobs += [_job(f"phc_{i}", precompute_marker,
                  dict(grain="complex", targets=ch, ckpt=PHASE_CK, out_root=C.OUT, load_workers=12, batch=args.batch), "phase_full")
             for i, ch in enumerate(_chunks(cx, args.chunk_size))]
    print(f"phase-full: {len(genes)} geneKO + {len(cx)} complex → {len(jobs)} chunked jobs (batch={args.batch}, no constraint/cap)")
    sp = {"slurm_partition": "gpu", "gpus_per_node": 1, "cpus_per_task": 12, "mem_gb": 64, "timeout_min": 720}
    submit_parallel_jobs(jobs_to_submit=jobs, experiment="diffex_gifs", slurm_params=sp,
                         log_dir="diffex_gifs", wait_for_completion=False)


def main():
    ap = argparse.ArgumentParser(description="Build the DiffEx viewer cache")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("seed"); s.add_argument("--n", type=int, default=8); s.add_argument("--map-thr", dest="map_thr", type=float, default=None); s.add_argument("--min-ep", dest="min_ep", type=int, default=0, help="min generator epoch to include; default 0 = no epoch gate (diffae_best.pt banks the peak regardless — epoch != quality)"); s.add_argument("--no-score", dest="no_score", action="store_true"); s.add_argument("--parallel", type=int, default=None, help="max concurrent SLURM tasks; default None = no cap"); s.add_argument("--timeout", type=int, default=180, help="per-marker SLURM timeout (min); bump for full ~1000-gene buildouts"); s.add_argument("--sync", action="store_true", help="on completion, auto-refresh manifest + attention + montages"); s.set_defaults(fn=cmd_seed)
    a = sub.add_parser("anchors"); a.add_argument("--k", type=int, default=5); a.add_argument("--markers", nargs="*"); a.add_argument("--parallel", type=int, default=12); a.set_defaults(fn=cmd_anchors)
    m = sub.add_parser("manifest"); m.set_defaults(fn=cmd_manifest)
    g = sub.add_parser("montage"); g.add_argument("--cells", type=int, nargs="+", default=list(range(20))); g.add_argument("--alphas", type=float, nargs="+", default=[1.0, 2.0, 3.0, 4.0, 5.0]); g.add_argument("--embeddings", nargs="+", default=["umap", "phate"]); g.add_argument("--markers", nargs="+", help="restrict to these markers (raw or slug); default all with geneKO traversals"); g.add_argument("--force", action="store_true", help="rebuild montages even if tiles already exist"); g.add_argument("--parallel", type=int, default=100, help="max concurrent SLURM tasks"); g.set_defaults(fn=cmd_montage)
    fc = sub.add_parser("fluor-complex"); fc.add_argument("--markers", nargs="*"); fc.add_argument("--batch", type=int, default=24); fc.set_defaults(fn=cmd_fluor_complex)
    pf = sub.add_parser("phase-full"); pf.add_argument("--chunk-size", type=int, default=50); pf.add_argument("--batch", type=int, default=24); pf.set_defaults(fn=cmd_phase_full)
    sy = sub.add_parser("sync", help="refresh manifest + attention + montages from the current cache"); sy.add_argument("--after", nargs="+", help="SLURM job IDs to gate on (afterany); refreshes when they finish"); sy.set_defaults(fn=cmd_sync)
    args = ap.parse_args(); args.fn(args)


if __name__ == "__main__":
    main()
