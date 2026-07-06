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

from ..classifier.config import slugify
from ..directions.make_gifs import _setup, _sample_guided

# scrub axis; α in units of the control→KD gap (α=±1 ≈ full traversal). Dense in ±3 where the
# phenotype resolves, reaching ±5 for the subtle markers where extreme α still adds signal.
VIEWER_ALPHAS = (-5.0, -4.0, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0,
                 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0)


@torch.no_grad()
def precompute_target(grain, target, ckpt, out_root, marker_channel=None, channel=None,
                      fluor_csv=None, n_cells=20, w=2.0, alphas=VIEWER_ALPHAS,
                      device="cuda", upsize=256):
    """Decode + save the α-frame sequence for the first n_cells control cells of one
    (marker, target). Returns the per-target meta dict (also written to meta.json)."""
    ctx = _setup(grain, target, out_root, device, ckpt=ckpt,
                 marker_channel=marker_channel, channel=channel, fluor_csv=fluor_csv)
    dev, cfg, slug, _out, embs, labels, fixed_dir, gap, diffae, null_base = ctx
    ci = np.flatnonzero(labels == 0)
    ncell = min(n_cells, len(ci))
    H = cfg.crop_size
    al = sorted(alphas)
    modality = slugify(marker_channel) if marker_channel else "phase"
    adir = Path(out_root) / "viewer_assets" / modality / grain / slug
    for cell in range(ncell):
        z0 = torch.as_tensor(embs[ci[cell]:ci[cell] + 1], dtype=torch.float32).to(dev)
        ge = torch.Generator(device=dev).manual_seed(1234 + cell)
        xT = torch.randn(1, 1, H, H, generator=ge, device=dev)
        cdir = adir / f"cell{cell}"
        cdir.mkdir(parents=True, exist_ok=True)
        for i, a in enumerate(al):
            img = _sample_guided(diffae, xT.clone(), z0 + (a * gap) * fixed_dir, null_base, w, cfg)
            arr = np.clip((img.cpu().numpy()[0, 0] + 1) / 2, 0, 1)
            im = Image.fromarray((arr * 255).astype("uint8"))
            if upsize:
                im = im.resize((upsize, upsize), Image.BILINEAR)
            im.save(cdir / f"frame_{i:02d}.webp", quality=90, method=6)
    # asset_dir is RELATIVE to viewer_assets/ (where manifest.json lives) so the static
    # frontend resolves it against the manifest URL without double-prefixing.
    meta = {"grain": grain, "target": target, "modality": modality,
            "marker_channel": marker_channel, "channel": channel, "slug": slug,
            "w": w, "alphas": al, "gap": float(gap), "n_cells": ncell,
            "asset_dir": f"{modality}/{grain}/{slug}"}
    (adir / "meta.json").write_text(json.dumps(meta))
    print(f"[viewer] {modality}/{grain}/{slug}: {ncell} cells × {len(al)} α-frames")
    return meta


def build_manifest(out_root, dist_map=None):
    """Aggregate every viewer_assets/*/*/*/meta.json into one manifest.json the frontend reads.
    dist_map: optional {(modality, grain, slug): mAP} to attach for sorting targets."""
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
                              "n_cells": m["n_cells"], "asset_dir": adir, "alphas": m["alphas"],
                              "dist_map": (dist_map or {}).get(key)})
    for mk in markers.values():
        mk["targets"].sort(key=lambda t: (-(t["dist_map"] or -1), t["target"]))
    manifest = {"alphas": list(VIEWER_ALPHAS), "w": 2.0,
                "markers": sorted(markers.values(), key=lambda x: x["marker_channel"] or "")}
    out = root / "manifest.json"
    out.write_text(json.dumps(manifest, indent=2))
    n_t = sum(len(mk["targets"]) for mk in markers.values())
    print(f"[viewer] manifest: {len(markers)} markers, {n_t} targets -> {out}")
    return str(out)
