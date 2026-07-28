"""Combine phase counterfactual traversals with the multi-marker virtual-staining (VS) system.

Pipeline: a real phase cell's traversal frame at α (already built, viewer_assets_v5/phase/geneKO/<gene>/cell<c>/
frame_<ai>.webp) → CellDINO embed → multi-marker VS model → the SAME synthesized cell rendered in every one of
the 42 live markers. So one phase cell yields, per geneKO and α, the phase phenotype + all 42 marker phenotypes.

Stages:
  proto()      — sanity montage: a few geneKOs × 42 markers at α=5 (tests VS on GENERATED phase).
  stain_shard()— stain a chunk of geneKOs (all 42 markers) at (cell, α) → per-(marker,gene) webp on disk.
  submit_stain()— shard the ~1000 geneKOs across GPUs.
Then render_montage_scales.render_composed sources these stained tiles → one composed montage per channel.
"""
import glob
import json
import os

import numpy as np
import torch
from PIL import Image

from ..classifier.config import slugify

VS_OUT = "/hpc/projects/icd.fast.ops/analysis/virtual_staining/multi_marker"
V5 = "/hpc/projects/icd.fast.ops/models/diffex/viewer_assets_v5"
STAINED = "/hpc/projects/icd.fast.ops/analysis/figure4_embedding/phase_vs_combine/stained"   # <marker_slug>/<gene>_c<c>_a<a>.webp
AI_OF = {a: i for i, a in enumerate([-5, -4, -3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5])}


def load_vs(dev):
    from ..diffae.config import DiffAEConfig
    from ..diffae.model import DiffAE
    markers = json.load(open(f"{VS_OUT}/markers.json"))
    cfg = DiffAEConfig(spatial_cond=True, n_markers=len(markers), device="cuda", epochs=1)
    ema = DiffAE(cfg).to(dev).eval()
    st = torch.load(f"{VS_OUT}/train_state.pt", map_location=dev)
    ema.load_state_dict(st["ema"])
    return ema, markers, cfg, st.get("epoch")


def _load_phase(gene, cell, ai, H):
    p = f"{V5}/phase/geneKO/{gene}/cell{cell}/frame_{ai:02d}.webp"
    if not os.path.exists(p):
        return None
    im = Image.open(p).convert("L").resize((H, H))
    return (np.asarray(im, np.float32) / 255.0 * 2 - 1)[None, None]        # (1,1,H,H) in [-1,1]


@torch.no_grad()
def stain(ema, markers, cfg, dev, phase_np, seed=0):
    """phase_np (1,1,H,H) in [-1,1] → {marker_idx: pred (H,H)} for all markers (fixed xT seed)."""
    from ..diffae.virtstain_multi import _sample_marker
    from ..classifier.celldino_features import embed_crops
    H = cfg.crop_size
    emb = torch.as_tensor(embed_crops(phase_np, cfg), dtype=torch.float32, device=dev)
    ci = torch.as_tensor(phase_np, dtype=torch.float32, device=dev)
    out = {}
    for mid in range(len(markers)):
        g = torch.Generator(device=dev).manual_seed(seed)
        xT = torch.randn(1, 1, H, H, generator=g, device=dev)
        mk = torch.as_tensor([mid], dtype=torch.long, device=dev)
        out[mid] = _sample_marker(ema, xT, emb, ci, mk, cfg, dev).cpu().numpy()[0, 0]
    return out


def _save(path, arr):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray((np.clip((arr + 1) / 2, 0, 1) * 255).astype("uint8")).resize((256, 256)).save(path, quality=90, method=6)


def stain_shard(genes, cell=1, alphas=(1.0, 2.0, 3.0, 4.0, 5.0)):
    """Stain a chunk of geneKOs into all 42 markers at (cell, each α); write <marker_slug>/<gene>_c<c>_a<a>.webp.
    Model loaded once per shard; all requested alphas stained per gene (skip-guard resumes)."""
    dev = torch.device("cuda")
    ema, markers, cfg, ep = load_vs(dev)
    last = slugify(markers[-1]); done = skip = 0              # gene×α complete once the LAST marker webp exists
    for g in genes:
        for a in alphas:
            an = f"a{a:g}"
            if os.path.exists(f"{STAINED}/{last}/{slugify(g)}_c{cell}_{an}.webp"):
                skip += 1; continue                           # resume: already stained
            ph = _load_phase(g, cell, AI_OF[a], cfg.crop_size)
            if ph is None:
                continue
            preds = stain(ema, markers, cfg, dev, ph)
            for mid, name in enumerate(markers):
                _save(f"{STAINED}/{slugify(name)}/{slugify(g)}_c{cell}_{an}.webp", preds[mid])
            done += 1
    print(f"[phasevs] stained {done} gene×α ({skip} already done) × {len(markers)} markers (VS ep{ep}) -> {STAINED}")
    return {"done": done, "skipped": skip, "markers": len(markers)}


def submit_stain(cell=1, alphas=(1.0, 2.0, 3.0, 4.0, 5.0), chunk=8, parallel=64):
    """Shard the ~1000 geneKOs across GPUs; each shard stains its genes into all 42 markers at (cell, each α).
    Resumable (skip-guard) → re-run to fill gaps. Small chunks + high parallelism for short wall time."""
    from . import catalog as C
    from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs
    genes = C.all_genes()
    ch = lambda l, n: [l[i:i + n] for i in range(0, len(l), n)]
    jobs = [{"name": f"pvs_c{cell}_{i}", "func": stain_shard,
             "kwargs": {"genes": s, "cell": cell, "alphas": list(alphas)}} for i, s in enumerate(ch(genes, chunk))]
    print(f"[phasevs] {len(genes)} geneKOs → {len(jobs)} stain shards (chunk {chunk}, parallel {parallel}, cell {cell}, α={list(alphas)})")
    submit_parallel_jobs(
        jobs_to_submit=jobs, experiment="diffex_phasevs",
        slurm_params={"slurm_partition": "gpu", "gpus_per_node": 1, "cpus_per_task": 8, "mem_gb": 64,
                      "timeout_min": 90, "slurm_constraint": "[a100_80|h100|h200|6000_blackwell]",
                      "slurm_array_parallelism": parallel},
        log_dir="diffex_phasevs", wait_for_completion=False)


COMPOSED = "/hpc/projects/icd.fast.ops/analysis/figure4_embedding/phase_vs_combine/composed"
STAINED_ALL = "/hpc/projects/icd.fast.ops/analysis/figure4_embedding/phase_vs_combine/stained_all"   # RGB 42-marker merge


def marker_colors(n):
    """n distinct hues (unique color per marker), full saturation/value → (n,3) RGB in [0,1]."""
    import matplotlib.pyplot as plt
    return plt.get_cmap("hsv")(np.linspace(0, 1, n, endpoint=False))[:, :3]


def compose_all_shard(genes, cell=1, alpha=5.0):
    """Merge the 42 stained marker tiles per gene into ONE RGB image: unique hue per marker, fluorescence-style
    false-color on BLACK. Each stain sits at a high flat baseline (~0.49), so a naive additive sum of 42
    channels clips to white everywhere — instead background-subtract + stretch each marker, then max/lighten
    blend (each pixel = brightest marker's color). → stained_all/<gene>_c<cell>_a<alpha>.webp (+ __NTC)."""
    markers = json.load(open(f"{VS_OUT}/markers.json"))
    slugs = [slugify(m) for m in markers]
    cols = marker_colors(len(markers))                       # (42,3)
    a5 = f"a{alpha:g}"; done = 0
    for g in list(genes) + ["__NTC"]:
        imgs, ok = [], True
        for s in slugs:
            p = f"{STAINED}/{s}/__NTC_c{cell}.webp" if g == "__NTC" else f"{STAINED}/{s}/{slugify(g)}_c{cell}_{a5}.webp"
            if not os.path.exists(p):
                ok = False; break
            imgs.append(np.asarray(Image.open(p).convert("L"), np.float32) / 255.0)
        if not ok:
            continue
        stack = np.stack(imgs, 0)                                                     # (M,H,W) in [0,1]
        lo = np.percentile(stack, 70, axis=(1, 2), keepdims=True)                     # per-marker background
        hi = np.percentile(stack, 99.5, axis=(1, 2), keepdims=True)
        xs = np.clip((stack - lo) / np.clip(hi - lo, 1e-6, None), 0, 1)               # isolate bright structures → black bg
        rgb = (xs[..., None] * cols[:, None, None, :]).max(axis=0)                    # (H,W,3) max/lighten blend
        out = f"{STAINED_ALL}/{'__NTC_c%d' % cell if g == '__NTC' else slugify(g) + '_c%d_%s' % (cell, a5)}.webp"
        os.makedirs(os.path.dirname(out), exist_ok=True)
        Image.fromarray((rgb * 255).astype("uint8")).resize((256, 256)).save(out, quality=90, method=6)
        done += 1
    print(f"[phasevs] merged {done} all-marker RGB tiles → {STAINED_ALL}")
    return {"done": done}


def submit_compose_all(cell=1, alphas=(1.0, 2.0, 3.0, 4.0, 5.0), chunk=60):
    from . import catalog as C
    from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs
    genes = C.all_genes()
    ch = lambda l, n: [l[i:i + n] for i in range(0, len(l), n)]
    jobs = [{"name": f"pvsall_a{a:g}_{i}", "func": compose_all_shard, "kwargs": {"genes": s, "cell": cell, "alpha": a}}
            for a in alphas for i, s in enumerate(ch(genes, chunk))]
    print(f"[phasevs] {len(genes)} genes × {len(alphas)} α → {len(jobs)} all-marker merge shards")
    submit_parallel_jobs(jobs_to_submit=jobs, experiment="diffex_phasevs",
                         slurm_params={"slurm_partition": "cpu", "cpus_per_task": 4, "mem_gb": 32, "timeout_min": 40,
                                       "slurm_array_parallelism": 32},
                         log_dir="diffex_phasevs", wait_for_completion=False)


def render_channel(channel, cell=1, alpha=5.0, level=4):
    """Render ONE composed montage for `channel` ("phase" or a marker slug) on the shared phase embedding,
    matching the no-marks reference (phate, L4). Phase tiles from viewer_assets_v5; marker tiles from STAINED."""
    from .render_montage_scales import render_composed
    render_composed([alpha], cell=cell, level=level, out_dir=f"{COMPOSED}/{channel}", marks=False, bg="white",
                    channel=channel, stained_dir=STAINED, assets="viewer_assets_v5")   # white canvas; fluor tiles magma-composited
    return {"channel": channel}


def submit_montages(cell=1, alpha=5.0, level=4):
    """Fan out the phase + 42 marker composed montages (all on the phase embedding)."""
    from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs
    markers = json.load(open(f"{VS_OUT}/markers.json"))
    chans = ["phase"] + [slugify(m) for m in markers]
    jobs = [{"name": f"pvsmtg_{c[:12]}", "func": render_channel,
             "kwargs": {"channel": c, "cell": cell, "alpha": alpha, "level": level}} for c in chans]
    print(f"[phasevs] {len(jobs)} composed montages (phase + {len(markers)} markers) → {COMPOSED}/<channel>")
    submit_parallel_jobs(
        jobs_to_submit=jobs, experiment="diffex_phasevs",
        slurm_params={"slurm_partition": "cpu", "cpus_per_task": 8, "mem_gb": 64, "timeout_min": 60,
                      "slurm_array_parallelism": 43},
        log_dir="diffex_phasevs", wait_for_completion=False)


def render_all(cell=1, alpha=5.0, level=4):
    """Render the ALL-MARKERS composed figure (42-marker RGB merge) on the shared phase embedding, white bg."""
    from .render_montage_scales import render_composed
    render_composed([alpha], cell=cell, level=level, out_dir=f"{COMPOSED}/__allmarkers__", marks=False, bg="black",
                    channel="__allmarkers__", stained_dir=STAINED_ALL, assets="viewer_assets_v5")
    return {"channel": "__allmarkers__"}


def submit_all_figure(cell=1, alphas=(1.0, 2.0, 3.0, 4.0, 5.0), level=4):
    """Fan out the all-markers composed figure per α."""
    from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs
    jobs = [{"name": f"pvsallfig_a{a:g}", "func": render_all, "kwargs": {"cell": cell, "alpha": a, "level": level}}
            for a in alphas]
    print(f"[phasevs] {len(jobs)} all-marker composed figures → {COMPOSED}/__allmarkers__")
    submit_parallel_jobs(
        jobs_to_submit=jobs, experiment="diffex_phasevs",
        slurm_params={"slurm_partition": "cpu", "cpus_per_task": 8, "mem_gb": 64, "timeout_min": 60,
                      "slurm_array_parallelism": 5},
        log_dir="diffex_phasevs", wait_for_completion=False)


def montage_vs_tiles(marker_slug, cell=1, alpha=5.0, embedding="phate", tile=256, ppu=5600):
    """Interactive VS montage tiles for the viewer: place each geneKO's STAINED tile at its PHASE-embedding
    coordinate → OME-zarr → PNG tile pyramid at viewer_assets_v5/_montage_vs/<marker>_<emb>_cell<c>_a<a>_tiles/.
    Same layout as the phase montage (NTC nodes use the stained α0 anchor)."""
    import shutil
    import anndata as ad
    from latent_lens import MontageConfig, build_montage
    from .build_umap_montage import _embed_coords, montage_to_tiles, ZARR_SCRATCH, OUT as MOUT
    from .render_montage_scales import UMAP_H5AD
    a5 = f"a{alpha:g}"
    ann = ad.read_h5ad(UMAP_H5AD)
    coords_all = _embed_coords(ann, embedding)
    gc = {str(g): coords_all[i] for i, g in enumerate(ann.obs["perturbation"])}
    genes, coords, srcs = [], [], []
    for g, xy in gc.items():                                  # real genes with a stained tile (same set as phase)
        if str(g).startswith("NTC"):
            continue
        if os.path.exists(f"{STAINED}/{marker_slug}/{slugify(g)}_c{cell}_{a5}.webp"):
            genes.append(g); coords.append(xy); srcs.append(slugify(g))
    for g, xy in gc.items():                                  # NTC nodes → stained α0 anchor
        if str(g).startswith("NTC"):
            genes.append(g); coords.append(xy); srcs.append("__NTC")
    coords = np.asarray(coords, np.float32)

    def crops(i):
        p = (f"{STAINED}/{marker_slug}/__NTC_c{cell}.webp" if srcs[i] == "__NTC"
             else f"{STAINED}/{marker_slug}/{srcs[i]}_c{cell}_{a5}.webp")
        return np.asarray(Image.open(p).convert("L"))

    os.makedirs(ZARR_SCRATCH, exist_ok=True)
    oz = f"{ZARR_SCRATCH}/vs_{marker_slug}_{embedding}_c{cell}_{a5}.zarr"
    build_montage(umap_coords=coords, crops=crops, categories=np.array(["m"] * len(genes)),
                  category_colors={"m": (1.0, 1.0, 1.0)}, output_path=oz, labels=np.array(genes),
                  config=MontageConfig(crop_size=tile, px_per_umap=ppu, border_width=max(4, tile // 40)))
    tiles = f"{MOUT}/viewer_assets_v5/_montage_vs/{marker_slug}_{embedding}_cell{cell}_{a5}_tiles"
    montage_to_tiles(oz, UMAP_H5AD, out_dir=tiles, placed=set(genes), embedding=embedding)
    shutil.rmtree(oz, ignore_errors=True)
    print(f"[phasevs] VS tiles {marker_slug} {embedding} → {tiles}")
    return {"marker": marker_slug, "genes": len(genes)}


def submit_vs_tiles(cell=1, alphas=(1.0, 2.0, 3.0, 4.0, 5.0)):
    """Build interactive VS montage tiles for all 42 markers × {umap, phate} × each α → viewer_assets_v5/_montage_vs/."""
    from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs
    markers = json.load(open(f"{VS_OUT}/markers.json"))
    jobs = [{"name": f"vstile_{slugify(m)[:10]}_{e[:2]}_a{a:g}", "func": montage_vs_tiles,
             "kwargs": {"marker_slug": slugify(m), "cell": cell, "alpha": a, "embedding": e}}
            for m in markers for e in ("umap", "phate") for a in alphas]
    print(f"[phasevs] {len(jobs)} VS montage-tile jobs (42 markers × 2 emb × {len(alphas)} α)")
    submit_parallel_jobs(
        jobs_to_submit=jobs, experiment="diffex_phasevs",
        slurm_params={"slurm_partition": "cpu", "cpus_per_task": 6, "mem_gb": 48, "timeout_min": 60,
                      "slurm_array_parallelism": 48},
        log_dir="diffex_phasevs", wait_for_completion=False)


def montage_all_tiles(cell=1, alpha=5.0, embedding="phate", tile=256, ppu=5600):
    """Interactive ALL-MARKERS montage tiles: place each geneKO's pre-composited 42-marker RGB tile at its
    phase-embedding coord → OSD pyramid at _montage_vs/__allmarkers___<emb>_cell<c>_a<a>_tiles/.
    build_montage only tints grayscale by one color, so its two crop helpers are monkeypatched to pass RGB through."""
    import shutil
    import anndata as ad
    from latent_lens import MontageConfig, build_montage
    from latent_lens import montage as _M
    from .build_umap_montage import _embed_coords, montage_to_tiles, ZARR_SCRATCH, OUT as MOUT
    from .render_montage_scales import UMAP_H5AD
    an = f"a{alpha:g}"
    ann = ad.read_h5ad(UMAP_H5AD)
    coords_all = _embed_coords(ann, embedding)
    gc = {str(g): coords_all[i] for i, g in enumerate(ann.obs["perturbation"])}
    genes, coords, srcs = [], [], []
    for g, xy in gc.items():                                  # real genes with a composited all-marker tile
        if str(g).startswith("NTC"):
            continue
        if os.path.exists(f"{STAINED_ALL}/{slugify(g)}_c{cell}_{an}.webp"):
            genes.append(g); coords.append(xy); srcs.append(slugify(g))
    for g, xy in gc.items():                                  # NTC nodes → composited α0 anchor
        if str(g).startswith("NTC"):
            genes.append(g); coords.append(xy); srcs.append("__NTC")
    coords = np.asarray(coords, np.float32)

    def crops(i):
        p = (f"{STAINED_ALL}/__NTC_c{cell}.webp" if srcs[i] == "__NTC"
             else f"{STAINED_ALL}/{srcs[i]}_c{cell}_{an}.webp")
        return np.asarray(Image.open(p).convert("RGB"), np.float32) / 255.0    # (H,W,3) in [0,1]

    def _norm_rgb(crop, cs):                                  # RGB-aware pad/normalize (grayscale falls back to orig)
        if crop.ndim == 2:
            return _norm_orig(crop, cs)
        if crop.shape[0] != cs or crop.shape[1] != cs:
            pad = np.zeros((cs, cs, 3), np.float32); h = min(crop.shape[0], cs); w = min(crop.shape[1], cs)
            pad[:h, :w] = crop[:h, :w]; crop = pad
        return crop.astype(np.float32)

    def _tint_rgb(crop, color):                              # already colored → (3,H,W), skip single-color tint
        return np.transpose(crop, (2, 0, 1)) if crop.ndim == 3 else _tint_orig(crop, color)

    _norm_orig, _tint_orig = _M._normalize_crop, _M.tint_crop
    _M._normalize_crop, _M.tint_crop = _norm_rgb, _tint_rgb
    try:
        os.makedirs(ZARR_SCRATCH, exist_ok=True)
        oz = f"{ZARR_SCRATCH}/vsall_{embedding}_c{cell}_{an}.zarr"
        build_montage(umap_coords=coords, crops=crops, categories=np.array(["m"] * len(genes)),
                      category_colors={"m": (1.0, 1.0, 1.0)}, output_path=oz, labels=np.array(genes),
                      config=MontageConfig(crop_size=tile, px_per_umap=ppu, border_width=max(4, tile // 40)))
    finally:
        _M._normalize_crop, _M.tint_crop = _norm_orig, _tint_orig
    tiles = f"{MOUT}/viewer_assets_v5/_montage_vs/__allmarkers___{embedding}_cell{cell}_{an}_tiles"
    montage_to_tiles(oz, UMAP_H5AD, out_dir=tiles, placed=set(genes), embedding=embedding)
    shutil.rmtree(oz, ignore_errors=True)
    print(f"[phasevs] ALL-marker tiles {embedding} a{alpha:g} → {tiles}")
    return {"genes": len(genes)}


def submit_all_tiles(cell=1, alphas=(1.0, 2.0, 3.0, 4.0, 5.0)):
    """Build interactive ALL-MARKERS montage tiles for {umap, phate} × each α → viewer_assets_v5/_montage_vs/."""
    from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs
    jobs = [{"name": f"vsalltile_{e[:2]}_a{a:g}", "func": montage_all_tiles,
             "kwargs": {"cell": cell, "alpha": a, "embedding": e}}
            for e in ("umap", "phate") for a in alphas]
    print(f"[phasevs] {len(jobs)} all-marker montage-tile jobs (2 emb × {len(alphas)} α)")
    submit_parallel_jobs(
        jobs_to_submit=jobs, experiment="diffex_phasevs",
        slurm_params={"slurm_partition": "cpu", "cpus_per_task": 6, "mem_gb": 48, "timeout_min": 60,
                      "slurm_array_parallelism": 10},
        log_dir="diffex_phasevs", wait_for_completion=False)


def stain_ntc(cell=1):
    """Stain the α=0 anchor (control) cell → all 42 markers, for the composed montage's NTC nodes.
    Saved as <marker_slug>/__NTC_c<cell>.webp (α=0 content, shared by every NTC grid node)."""
    dev = torch.device("cuda")
    ema, markers, cfg, ep = load_vs(dev)
    a0 = AI_OF[0.0]
    g0 = next(os.path.basename(os.path.dirname(os.path.dirname(p)))
              for p in sorted(glob.glob(f"{V5}/phase/geneKO/*/cell{cell}/frame_{a0:02d}.webp")))
    preds = stain(ema, markers, cfg, dev, _load_phase(g0, cell, a0, cfg.crop_size))
    for mid, name in enumerate(markers):
        _save(f"{STAINED}/{slugify(name)}/__NTC_c{cell}.webp", preds[mid])
    print(f"[phasevs] stained NTC anchor (α0 from {g0}) → {len(markers)} markers")
    return {"markers": len(markers)}


def proto():
    """Sanity montage: a few geneKOs × 42 markers at α=5. Writes phase_vs_combine/proto.png."""
    dev = torch.device("cuda")
    ema, markers, cfg, ep = load_vs(dev); H = cfg.crop_size
    cand = ["TP53", "KRAS", "TUBB", "ACTB", "POLR1B", "KIF23", "MYC", "CTNNB1"]
    genes = [g for g in cand if os.path.exists(f"{V5}/phase/geneKO/{g}/cell1/frame_16.webp")]
    if len(genes) < 4:
        genes = [os.path.basename(os.path.dirname(os.path.dirname(p)))
                 for p in sorted(glob.glob(f"{V5}/phase/geneKO/*/cell1/frame_16.webp"))[:8]]
    rows = [(g, _load_phase(g, 1, 16, H)[0, 0], stain(ema, markers, cfg, dev, _load_phase(g, 1, 16, H))) for g in genes]
    import matplotlib
    matplotlib.use("Agg"); matplotlib.rcParams["pdf.fonttype"] = 42
    import matplotlib.pyplot as plt
    ncol = 1 + len(markers)
    fig, ax = plt.subplots(len(rows), ncol, figsize=(ncol * 0.85, len(rows) * 0.95), squeeze=False)
    for r_ in range(len(rows)):
        for c_ in range(ncol):
            ax[r_][c_].set_xticks([]); ax[r_][c_].set_yticks([])
    for r, (g, ph, preds) in enumerate(rows):
        ax[r][0].imshow(ph, cmap="gray", vmin=-1, vmax=1, aspect="auto"); ax[r][0].set_ylabel(g, fontsize=7)
        for m in range(len(markers)):
            ax[r][1 + m].imshow(preds[m], cmap="magma", vmin=-1, vmax=1, aspect="auto")
    ax[0][0].set_title("phase α5", fontsize=6)
    for m, name in enumerate(markers):
        ax[0][1 + m].set_title(name.split("_")[0][:10], fontsize=5, rotation=90, va="bottom")
    fig.suptitle(f"Phase traversal (α=5) virtually stained → all {len(markers)} markers · VS ep{ep} · {len(rows)} geneKOs", fontsize=10)
    fig.subplots_adjust(wspace=0.03, hspace=0.05, top=0.9)
    out = os.path.dirname(STAINED); os.makedirs(out, exist_ok=True)
    fig.savefig(f"{out}/proto.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"[phasevs] proto -> {out}/proto.png ({genes})")
    return {"genes": genes}
