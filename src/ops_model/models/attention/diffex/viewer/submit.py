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
from .build_umap_montage import build_montage_web
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
    for d, mc, ch in C.complete_markers():
        rep = C.rep_of(dist, mc)
        if not rep or rep not in dist.columns:
            continue
        if args.map_thr:                                     # full buildout: ALL genes the marker distinguishes >= thr
            sc = dist[rep]
            tg = [g for g in sc.index[sc >= args.map_thr] if not str(g).startswith("NTC")]
        else:
            tg = C.top_genes(dist, rep, args.n)
        if tg:
            jobs.append(_job(f"pm_{slugify(mc)[:20]}", precompute_marker,
                             dict(grain="geneKO", targets=tg, marker_channel=mc, channel=ch,
                                  ckpt=f"{C.DD}/{d}/diffae_best.pt", out_root=C.OUT, load_workers=12), "seed"))
    if not args.map_thr:                                     # phase already fully built — only (re)seed with top-N mode
        jobs.append(_job("pm_phase_geneKO", precompute_marker,
                         dict(grain="geneKO", targets=C.top_genes(dist, "Phase", args.n + 4),
                              ckpt=PHASE_CK, out_root=C.OUT, load_workers=12), "seed"))
        jobs.append(_job("pm_phase_complex", precompute_marker,
                         dict(grain="complex", targets=PHASE_COMPLEXES, ckpt=PHASE_CK, out_root=C.OUT, load_workers=12), "seed"))
    tgt = sum(len(j["kwargs"]["targets"]) for j in jobs)
    print(f"seed: {len(jobs)} per-marker jobs, {tgt} total targets"
          + (f" (mAP>={args.map_thr} filter)" if args.map_thr else f" (top-{args.n})"))
    submit_parallel_jobs(jobs_to_submit=jobs, experiment="diffex_gifs",
                         slurm_params=_gpu(timeout_min=180, slurm_array_parallelism=args.parallel),
                         log_dir="diffex_gifs", wait_for_completion=False)


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
    """UMAP montage: harvest each gene's cached α-frame (correct top-attention direction) + place at
    its gene-UMAP coord. No decode/re-embed — reads the phase geneKO traversal cache. CPU-only, one job."""
    jobs = []
    for emb in args.embeddings:
        for cell in args.cells:
            for a in args.alphas:
                zarr = f"{C.OUT}/viewer_assets/_montage/phase_geneKO_{emb}_cell{cell}_a{a:g}.zarr"
                jobs.append(_job(f"montage_{emb}_c{cell}_a{a:g}", build_montage_web,
                                 dict(h5ad=UMAP_H5AD, out_zarr=zarr, cell=cell, alpha=a, embedding=emb), "montage"))
    print(f"montage: {len(jobs)} montages ({len(args.embeddings)} emb × {len(args.cells)} cell × {len(args.alphas)} α)")
    submit_parallel_jobs(jobs_to_submit=jobs, experiment="diffex_gifs",
                         slurm_params={"slurm_partition": "cpu", "cpus_per_task": 8, "mem_gb": 64, "timeout_min": 45,
                                       "slurm_array_parallelism": min(len(jobs), 30)},
                         log_dir="diffex_gifs", wait_for_completion=False)


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
    s = sub.add_parser("seed"); s.add_argument("--n", type=int, default=8); s.add_argument("--map-thr", dest="map_thr", type=float, default=None); s.add_argument("--parallel", type=int, default=10); s.set_defaults(fn=cmd_seed)
    a = sub.add_parser("anchors"); a.add_argument("--k", type=int, default=5); a.add_argument("--markers", nargs="*"); a.add_argument("--parallel", type=int, default=12); a.set_defaults(fn=cmd_anchors)
    m = sub.add_parser("manifest"); m.set_defaults(fn=cmd_manifest)
    g = sub.add_parser("montage"); g.add_argument("--cells", type=int, nargs="+", default=[0]); g.add_argument("--alphas", type=float, nargs="+", default=[2.0]); g.add_argument("--embeddings", nargs="+", default=["umap"]); g.set_defaults(fn=cmd_montage)
    fc = sub.add_parser("fluor-complex"); fc.add_argument("--markers", nargs="*"); fc.add_argument("--batch", type=int, default=24); fc.set_defaults(fn=cmd_fluor_complex)
    pf = sub.add_parser("phase-full"); pf.add_argument("--chunk-size", type=int, default=50); pf.add_argument("--batch", type=int, default=24); pf.set_defaults(fn=cmd_phase_full)
    args = ap.parse_args(); args.fn(args)


if __name__ == "__main__":
    main()
