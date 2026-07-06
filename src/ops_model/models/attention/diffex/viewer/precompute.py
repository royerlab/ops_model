"""Precompute per-(marker, target, cell) traversal frames + manifest for the DiffEx viewer.

A shareable MOPS-style static viewer can't run GPU diffusion live, so we precompute the
α-frame sequence of every traversal and let the frontend scrub it. w is FIXED (default 2.0,
the validated default); α is the scrub axis; marker / geneKO|complex / cell are routing.

Each traversal emits raw decoded frames (no label overlay — the viewer draws its own α axis
and −KO/NTC/+KO cues) at:
    <out_root>/viewer_assets/<modality>/<grain>/<slug>/cell<c>/frame_<i>.webp
plus a per-target meta.json. `build_manifest` aggregates all meta.json into one manifest.json
(marker -> grain -> targets -> cells + α list) that the static S3 viewer reads.

The α seed noise xT is fixed per cell, so identity is anchored and only the phenotype shifts
across frames — smooth to scrub. α=0 (center) is the true NTC; +α = toward the KO phenotype,
−α = pushed to the opposite extreme.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from concurrent.futures import ThreadPoolExecutor

from ..classifier.celldino_features import embed_crops
from ..classifier.config import slugify
from ..diffae.data import normalize
from ..directions.make_gifs import _pair_slug, _setup, _sample_guided
from ..directions.rank import supervised_direction

# scrub axis; α in units of the control→KD gap (α=±1 ≈ full traversal). Dense in ±3 where the
# phenotype resolves, reaching ±5 for the subtle markers where extreme α still adds signal.
VIEWER_ALPHAS = (-5.0, -4.0, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0,
                 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0)


def _save_webp(path, arr, upsize):
    im = Image.fromarray((np.clip((arr + 1) / 2, 0, 1) * 255).astype("uint8"))
    if upsize:
        im = im.resize((upsize, upsize), Image.BILINEAR)
    im.save(path, quality=90, method=6)


@torch.no_grad()
def precompute_target(grain, target, ckpt, out_root, marker_channel=None, channel=None,
                      fluor_csv=None, control=None, n_cells=20, w=2.0, alphas=VIEWER_ALPHAS,
                      device="cuda", upsize=256, score=True, batch=48, n_workers=8,
                      load_workers=10, keep_crops=False):
    """Decode + save the α-frame sequence for the first n_cells control cells of one
    (marker, target). Batched GPU decode, batched re-encode → per-image classifier
    confidence (sigmoid of the control→KD LR logit), threaded WebP save. Writes frames +
    meta.json + scores.json. control=None → NTC-anchored; else an A→B anchor class.
    load_workers: parallel zarr crop-read workers (the gather dominates runtime)."""
    ctx = _setup(grain, target, out_root, device, ckpt=ckpt, marker_channel=marker_channel,
                 channel=channel, fluor_csv=fluor_csv, control=control, num_workers=load_workers,
                 return_images=True)
    dev, cfg, slug, _out, embs, labels, fixed_dir, gap, diffae, null_base, real_imgs = ctx
    ci = np.flatnonzero(labels == 0)
    ncell = min(n_cells, len(ci))
    H, al, A = cfg.crop_size, sorted(alphas), len(sorted(alphas))
    modality = slugify(marker_channel) if marker_channel else "phase"
    adir = Path(out_root) / "viewer_assets" / modality / grain / slug

    # per-cell fixed noise (identity anchor); assemble all (cell, α) latents for batched decode
    conds, xts, keys = [], [], []
    for cell in range(ncell):
        z0 = torch.as_tensor(embs[ci[cell]:ci[cell] + 1], dtype=torch.float32, device=dev)
        xT = torch.randn(1, 1, H, H, generator=torch.Generator(device=dev).manual_seed(1234 + cell), device=dev)
        for ai, a in enumerate(al):
            conds.append(z0 + (a * gap) * fixed_dir); xts.append(xT); keys.append((cell, ai))

    gen = np.empty((ncell, A, H, H), dtype=np.float32)
    for i0 in range(0, len(conds), batch):
        cb = torch.cat(conds[i0:i0 + batch], 0)
        xb = torch.cat(xts[i0:i0 + batch], 0)
        nb = null_base.expand(cb.shape[0], -1)
        out = _sample_guided(diffae, xb, cb, nb, w, cfg).cpu().numpy()[:, 0]
        for j, (cell, ai) in enumerate(keys[i0:i0 + batch]):
            gen[cell, ai] = out[j]

    scores = None
    if score:                                    # re-encode every frame → LR class confidence
        _, lr_w, lr_b, _ = supervised_direction(embs, labels, cfg)
        gemb = embed_crops(gen.reshape(-1, 1, H, H).astype(np.float32), cfg, cache_path=None)
        logits = gemb @ lr_w + lr_b
        scores = (1.0 / (1.0 + np.exp(-logits))).reshape(ncell, A)

    real = normalize(real_imgs[ci[:ncell]])          # actual source-cell crops, [-1,1] for display
    pool = ThreadPoolExecutor(max_workers=n_workers)
    for cell in range(ncell):
        cdir = adir / f"cell{cell}"; cdir.mkdir(parents=True, exist_ok=True)
        pool.submit(_save_webp, cdir / "real.webp", real[cell, 0], upsize)   # static real cell (α=0 is a recon)
        for ai in range(A):
            pool.submit(_save_webp, cdir / f"frame_{ai:02d}.webp", gen[cell, ai], upsize)
    pool.shutdown(wait=True)

    if scores is not None:
        (adir / "scores.json").write_text(json.dumps({"alphas": al, "scores": np.round(scores, 3).tolist()}))
    meta = {"grain": grain, "target": target, "modality": modality, "control": control,
            "marker_channel": marker_channel, "channel": channel, "slug": slug,
            "w": w, "alphas": al, "gap": float(gap), "n_cells": ncell, "has_scores": scores is not None,
            "has_real": True, "asset_dir": f"{modality}/{grain}/{slug}"}
    (adir / "meta.json").write_text(json.dumps(meta))

    if not keep_crops:      # drop the ~195MB materialized-crop cache the viewer never needs
        cp = Path(_out) / "cache" / f"crops_{slug}_{cfg.crop_size}.npz"
        if cp.exists():
            cp.unlink()
    print(f"[viewer] {modality}/{grain}/{slug}: {ncell}×{A} frames" + (" +scores" if score else ""))
    return meta


def build_manifest(out_root, dist_map=None, desc_map=None):
    """Aggregate every viewer_assets/*/*/*/meta.json into one manifest.json the frontend reads.
    dist_map: optional {(modality, grain, slug): mAP} to attach for sorting targets.
    desc_map: optional {target_name: description} (gene function / complex members)."""
    root = Path(out_root) / "viewer_assets"
    markers = {}
    for mj in sorted(root.glob("*/*/*/meta.json")):
        m = json.loads(mj.read_text())
        mod = m["modality"]
        mk = markers.setdefault(mod, {"modality": mod, "marker_channel": m["marker_channel"],
                                      "channel": m["channel"], "targets": []})
        key = (mod, m["grain"], m["slug"])
        adir = m["asset_dir"]
        if adir.startswith("viewer_assets/"):        # normalize pre-fix-era meta.json
            adir = adir[len("viewer_assets/"):]
        mk["targets"].append({"grain": m["grain"], "target": m["target"], "slug": m["slug"],
                              "control": m.get("control"),   # None = NTC-anchored; else A→B anchor class
                              "has_real": m.get("has_real", False),
                              "n_cells": m["n_cells"], "asset_dir": adir, "alphas": m["alphas"],
                              "dist_map": (dist_map or {}).get(key),
                              "desc": (desc_map or {}).get(m["target"])})
    for mk in markers.values():
        mk["targets"].sort(key=lambda t: (-(t["dist_map"] or -1), t["target"]))
    manifest = {"alphas": list(VIEWER_ALPHAS), "w": 2.0,
                "markers": sorted(markers.values(), key=lambda x: x["marker_channel"] or "")}
    out = root / "manifest.json"
    out.write_text(json.dumps(manifest, indent=2))
    n_t = sum(len(mk["targets"]) for mk in markers.values())
    print(f"[viewer] manifest: {len(markers)} markers, {n_t} targets -> {out}")
    return str(out)
