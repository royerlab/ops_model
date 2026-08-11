"""Render Kevin's CellDINO attention-head pixel-attribution npz into static WebP tiles the viewer
overlays (inferno) on the real phenotype cells. Reproducible + SLURM-parallel (one job per shard).

Kevin's dump (`{SRC}`) has four pixel-attribution trees, all with the same npz schema:
    maps (n_cells, n_heads, 128, 128) f16 · crops (n_cells,128,128) f32 (z-scored)
    heads (n_heads,2) int32 = (layer,head) · patch_masks (n_cells,196) bool
  - phase geneKO   : pixel_attribution_cache/<GENE>.npz
  - phase complex  : complex_pixel_attribution/phase/pixel_attribution_cache/<COMPLEX>.npz
  - fluor geneKO   : fluorescence_pixel_attribution/<marker>/pixel_attribution_cache/<GENE>.npz
  - fluor complex  : complex_pixel_attribution/fluorescence/<marker>/pixel_attribution_cache/<COMPLEX>.npz
Ranking metrics (auroc/spec) only exist for phase geneKO (`head_rankings_per_gene.json`); the npz
`heads` array carries the (layer,head) pairs for the rest.

Overlay pipeline matches Ritvik: Gaussian-smooth each map (σ=2), mask to the CELL (patch_masks →
14×14 → 128 nearest, outside→0). We ship grayscale crop + mask + per-head maps; the webapp applies the
inferno LUT + clim + alpha live. Output (uniform, addressable by the viewer's marker×grain×target):
    {AH}/<modality>/<grain>/<key>/cell<c>/{crop,mask,head0..5}.webp + heads.json
    {AH}/index.json  {global_max, assets:{<modality>:{<grain>:[keys]}}}
where modality = "phase" | slugify(marker_channel), grain = geneKO|complex, key = gene | complex-slug.

  python -m ops_model.models.attention.diffex.viewer.build_attention_heads render            # SLURM (all trees)
  python -m ops_model.models.attention.diffex.viewer.build_attention_heads render --local    # serial, no SLURM
  python -m ops_model.models.attention.diffex.viewer.build_attention_heads render --dry-run
  python -m ops_model.models.attention.diffex.viewer.build_attention_heads index             # (re)aggregate index.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

from ..classifier.config import slugify
from . import catalog as C

AH_ROOT = f"{C.OUT}/viewer_assets/attention_heads"
SRC = f"{AH_ROOT}/celldino_attention_head_analysis"           # Kevin's persistent source dump
RANKINGS = f"{SRC}/head_rankings_per_gene.json"


def _save_gray(path, u8, upsize):
    im = Image.fromarray(u8)
    if upsize:
        im = im.resize((upsize, upsize), Image.BILINEAR)
    im.save(path, quality=90, method=6)


def _crop_u8(crop):
    """z-scored crop → uint8 via robust (1–99 pct) min-max, matching typical grayscale display."""
    lo, hi = np.percentile(crop, (1, 99))
    if hi <= lo:
        hi = lo + 1e-6
    return (np.clip((crop - lo) / (hi - lo), 0, 1) * 255).astype("uint8")


def sources():
    """(modality, grain, cache_dir) for every pixel-attribution tree present in Kevin's dump."""
    out = [("phase", "geneKO", f"{SRC}/pixel_attribution_cache")]
    pcx = f"{SRC}/complex_pixel_attribution"
    if os.path.isdir(f"{pcx}/phase/pixel_attribution_cache"):
        out.append(("phase", "complex", f"{pcx}/phase/pixel_attribution_cache"))
    for md in sorted(glob.glob(f"{SRC}/fluorescence_pixel_attribution/*/pixel_attribution_cache")):
        out.append((slugify(Path(md).parent.name), "geneKO", md))
    for md in sorted(glob.glob(f"{pcx}/fluorescence/*/pixel_attribution_cache")):
        out.append((slugify(Path(md).parent.name), "complex", md))
    return out


def _render_one(path, out_dir, rankings, upsize, n_workers):
    d = np.load(path, allow_pickle=True)
    maps, crops, heads, pmasks = d["maps"].astype(np.float32), d["crops"], d["heads"], d["patch_masks"]
    n_cells, n_heads = maps.shape[:2]
    H = maps.shape[-1]
    grid = int(round(pmasks.shape[1] ** 0.5))
    idx = np.minimum((np.arange(H) * grid // H), grid - 1)    # nearest upscale 14→128
    cell_masks = np.empty((n_cells, H, H), bool)
    pmaps = np.empty_like(maps)
    for c in range(n_cells):
        cm = pmasks[c].reshape(grid, grid)[np.ix_(idx, idx)]
        cell_masks[c] = cm
        for h in range(n_heads):
            pmaps[c, h] = np.where(cm, gaussian_filter(maps[c, h], sigma=2.0), 0.0)
    gene_max = float(pmaps.max())
    scale = 255.0 / gene_max if gene_max > 0 else 0.0
    pool = ThreadPoolExecutor(max_workers=n_workers)
    for c in range(n_cells):
        cdir = out_dir / f"cell{c}"
        cdir.mkdir(parents=True, exist_ok=True)
        pool.submit(_save_gray, cdir / "crop.webp", _crop_u8(crops[c]), upsize)
        pool.submit(_save_gray, cdir / "mask.webp", (cell_masks[c] * 255).astype("uint8"), upsize)
        for h in range(n_heads):
            pool.submit(_save_gray, cdir / f"head{h}.webp", np.clip(pmaps[c, h] * scale, 0, 255).astype("uint8"), upsize)
    pool.shutdown(wait=True)
    key = out_dir.name
    rk = {(r["layer"], r["head"]): r for r in rankings.get(key, [])}    # metrics by (layer,head), phase-geneKO only
    head_meta = []
    for h in range(n_heads):
        layer, head = int(heads[h][0]), int(heads[h][1])
        m = rk.get((layer, head), {})
        head_meta.append({"layer": layer, "head": head, "feature": m.get("feature"),
                          "spec_p10": m.get("spec_p10"), "spec_min": m.get("spec_min"),
                          "auroc_vs_ntc": m.get("auroc_vs_ntc")})
    (out_dir / "heads.json").write_text(json.dumps(
        {"gene": key, "n_cells": n_cells, "gene_max": gene_max, "heads": head_meta}))
    return gene_max


def render_shard(modality, grain, npz_paths, out_root=AH_ROOT, upsize=256, n_workers=8, use_rankings=False, force=False):
    """SLURM job unit: render a list of npz for one (modality, grain) into <out_root>/<modality>/<grain>/<key>/.
    Incremental by default (skips keys whose heads.json already exists); `force` re-renders. Skips
    unreadable npz loudly (corrupt-at-source). Returns a small summary."""
    rankings = json.load(open(RANKINGS)) if (use_rankings and os.path.exists(RANKINGS)) else {}
    base = Path(out_root) / modality / grain
    done, bad, skip = [], [], 0
    for p in npz_paths:
        key = Path(p).stem
        out_dir = base / key
        if not force and (out_dir / "heads.json").exists():
            skip += 1
            continue
        try:
            _render_one(p, out_dir, rankings, upsize, n_workers)
            done.append(key)
        except Exception as e:
            bad.append(key)
            print(f"[attn] SKIP {modality}/{grain}/{key}: {type(e).__name__}")
    print(f"[attn] {modality}/{grain}: {len(done)} rendered, {skip} already-present, {len(bad)} unreadable")
    return {"modality": modality, "grain": grain, "rendered": len(done), "skipped": skip, "bad": bad}


def build_index(out_root=AH_ROOT):
    """Aggregate every rendered heads.json into one index.json (availability + fixed-norm scale).
    Source of truth = the rendered dirs, so re-running any shard just re-scans the filesystem."""
    assets, gmax = {}, 0.0
    for hj in glob.glob(f"{out_root}/*/*/*/heads.json"):
        p = Path(hj)
        modality, grain, key = p.parents[2].name, p.parents[1].name, p.parent.name
        assets.setdefault(modality, {}).setdefault(grain, []).append(key)
        try:
            gmax = max(gmax, float(json.loads(p.read_text()).get("gene_max", 0.0)))
        except Exception:
            pass
    for m in assets:
        for g in assets[m]:
            assets[m][g] = sorted(assets[m][g])
    (Path(out_root) / "index.json").write_text(json.dumps({"global_max": gmax, "assets": assets}))
    n = sum(len(v) for m in assets.values() for v in m.values())
    print(f"[attn] index: {len(assets)} modalities, {n} keys, global_max={gmax:.4f} -> {out_root}/index.json")
    return f"{out_root}/index.json"


def submit(dry_run=False, parallel=40, chunk=150, local=False, upsize=256, force=False):
    """Fan out render_shard over all four trees (chunked), then aggregate index.json. Incremental by
    default — only new/unrendered npz are processed, so re-running picks up Kevin's newly-dumped markers."""
    jobs = []
    for modality, grain, cache in sources():
        paths = sorted(glob.glob(f"{cache}/*.npz"))
        use_r = modality == "phase" and grain == "geneKO"
        for i in range(0, len(paths), chunk):
            jobs.append({"name": f"ah_{modality[:12]}_{grain[:2]}_{i // chunk}",
                         "func": render_shard,
                         "kwargs": dict(modality=modality, grain=grain, npz_paths=paths[i:i + chunk],
                                        use_rankings=use_r, upsize=upsize, force=force),
                         "metadata": {"modality": modality, "grain": grain}})
    total = sum(len(glob.glob(f'{c}/*.npz')) for _, _, c in sources())
    print(f"[attn] {len(jobs)} shard jobs across {len(sources())} trees ({total} npz, chunk={chunk})")
    if local:
        for j in jobs:
            j["func"](**j["kwargs"])
        return build_index()
    from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs
    return submit_parallel_jobs(
        jobs_to_submit=jobs, experiment="diffex_attn_heads",
        slurm_params={"slurm_partition": "cpu", "cpus_per_task": 8, "mem_gb": 32, "timeout_min": 120,
                      "slurm_array_parallelism": parallel},
        log_dir="diffex_attn_heads", dry_run=dry_run,
        post_completion_callback=lambda *_: build_index())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    r = sub.add_parser("render", help="fan out render_shard on SLURM (default), then build index")
    r.add_argument("--local", action="store_true", help="run serially in this process, no SLURM")
    r.add_argument("--dry-run", action="store_true")
    r.add_argument("--parallel", type=int, default=40)
    r.add_argument("--chunk", type=int, default=150)
    r.add_argument("--upsize", type=int, default=256)
    r.add_argument("--force", action="store_true", help="re-render even if heads.json already exists")
    sub.add_parser("index", help="(re)aggregate index.json from rendered dirs")
    args = ap.parse_args()
    if args.cmd == "index":
        build_index()
    else:
        submit(dry_run=getattr(args, "dry_run", False), parallel=getattr(args, "parallel", 40),
               chunk=getattr(args, "chunk", 150), local=getattr(args, "local", False),
               upsize=getattr(args, "upsize", 256), force=getattr(args, "force", False))
