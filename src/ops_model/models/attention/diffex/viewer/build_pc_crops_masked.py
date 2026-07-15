"""Re-render the PC-strip crops with the attention_atlas blue cell-mask overlay.

Kyle's compute_pc_strips.py baked plain 96px Phase2D crops. This re-crops the SAME
representative cells (positions already in pcs/index.json) at a larger size and paints
everything OUTSIDE the target cell translucent blue — matching attention_atlas.py's
"negative overlay" (RGB 0.30/0.40/0.85, alpha 0.55, mask dilated 15px) so the cell of
interest reads in natural phase-gray against a blue surround.

Source store (per reference_phenotyping_v3_store):
  {BASE}/{exp}/3-assembly/phenotyping_v3.zarr/{row}/{col}/0/0            image [1,C,1,Y,X], Phase2D=ch0
  {BASE}/{exp}/3-assembly/phenotyping_v3.zarr/{row}/{col}/0/labels/cell_seg/0   int32 labels

  python -m ops_model.models.attention.diffex.viewer.build_pc_crops_masked --sample 24   # preview
  python -m ops_model.models.attention.diffex.viewer.build_pc_crops_masked               # full (overwrites crops/)
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from . import catalog as C

BASE = "/hpc/projects/intracellular_dashboard/fast_ops"
PCS_OUT = f"{C.OUT}/viewer_assets/pcs"
CROP_SIZE = 160                      # native px re-crop (was 96); crisper at the 150px display + shows surround
PHASE_CHANNEL = 0
MASK_DILATION = 15                   # matches attention_atlas.MASK_DILATION
OVERLAY_RGB = (0.30, 0.40, 0.85)     # attention_atlas negative-overlay blue (#4D66D9)
OVERLAY_ALPHA = 0.55


def _zarr_patch():
    try:
        import zarr
        from zarr.core.metadata.v3 import ArrayV3Metadata
        _o = ArrayV3Metadata.from_dict.__func__

        @classmethod
        def _p(cls, data):
            if isinstance(data, dict):
                data.pop("storage_transformers", None)
            return _o(cls, data)
        ArrayV3Metadata.from_dict = _p
    except Exception:
        pass


def _cells(idx):
    """Flatten index.json pcData → ALL representative cells with a position + filename (incl. the ones Kyle
    left has_crop=false; the zarr is reachable now so most re-crop fine), dedup by output filename."""
    out, seen = [], set()
    for d in idx["pcData"].values():
        for b in d["strip"]:
            for c in b["cells"]:
                if c and c.get("img") and c.get("experiment") and c["img"] not in seen:
                    seen.add(c["img"]); out.append(c)
    return out


def _is_blank(phase):
    """A crop is blank if it lands in empty/edge space: mostly exact-zeros (stitch gaps / edge pad) or flat."""
    return float((phase == 0).mean()) > 0.4 or float(phase.std()) < 0.02


def _crop(arr, ch, x, y, half):
    """Centered crop (channel ch) with zero-pad at the edges → (2*half, 2*half)."""
    _, _, _, H, W = arr.shape
    y0, y1 = max(0, y - half), min(H, y + half)
    x0, x1 = max(0, x - half), min(W, x + half)
    raw = np.array(arr[0, ch, 0, y0:y1, x0:x1]) if ch is not None else np.array(arr[0, 0, 0, y0:y1, x0:x1])
    n = 2 * half
    if raw.shape != (n, n):
        pad = np.zeros((n, n), dtype=raw.dtype)
        py, px = (n - raw.shape[0]) // 2, (n - raw.shape[1]) // 2
        pad[py:py + raw.shape[0], px:px + raw.shape[1]] = raw
        raw = pad
    return raw


def _render(phase, seg, half):
    """Gray phase (1–99 pct) with blue overlay outside the dilated center cell → uint8 RGB."""
    from scipy.ndimage import binary_dilation
    lo, hi = np.percentile(phase, (1, 99))
    if hi - lo < 1e-6:
        hi = lo + 1
    g = np.clip((phase - lo) / (hi - lo), 0, 1)
    rgb = np.stack([g, g, g], axis=-1)

    center = seg[half, half]
    if center == 0:   # center on background → use most common label in the central 24px box
        c = seg[half - 12:half + 12, half - 12:half + 12]
        nz = c[c > 0]
        center = np.bincount(nz).argmax() if nz.size else 0
    if center != 0:
        inv = ~binary_dilation(seg == center, iterations=MASK_DILATION)
        for k in range(3):
            rgb[..., k][inv] = rgb[..., k][inv] * (1 - OVERLAY_ALPHA) + OVERLAY_RGB[k] * OVERLAY_ALPHA
    return (rgb * 255).astype(np.uint8)


def build(sample=0, out=PCS_OUT):
    import zarr
    from PIL import Image
    _zarr_patch()
    idx = json.load(open(f"{out}/index.json"))
    cells = _cells(idx)
    if sample:
        cells = cells[:sample]
        crops_dir = f"{out}/_crops_sample"
    else:
        crops_dir = f"{out}/crops"
    os.makedirs(crops_dir, exist_ok=True)
    half = CROP_SIZE // 2
    cache, ok, blank, fail, valid = {}, 0, 0, 0, set()
    for i, c in enumerate(cells):
        exp, well = c["experiment"], c["well"]
        wr, wc = well[0], well[1:]
        key = (exp, well)
        if key not in cache:
            pos = f"{BASE}/{exp}/3-assembly/phenotyping_v3.zarr/{wr}/{wc}/0"
            try:
                cache[key] = (zarr.open(f"{pos}/0", mode="r"), zarr.open(f"{pos}/labels/cell_seg/0", mode="r"))
            except Exception as e:
                cache[key] = None
                print(f"[crops] open failed {exp}/{well}: {e}")
        if cache[key] is None:
            fail += 1; continue
        img, seg = cache[key]
        x, y = int(round(c["x"])), int(round(c["y"]))
        try:
            phase = _crop(img, PHASE_CHANNEL, x, y, half)
            if _is_blank(phase):   # empty/edge region → leave slot as placeholder (no valid crop to show)
                blank += 1; continue
            segc = _crop(seg, None, x, y, half)
            Image.fromarray(_render(phase, segc, half)).save(f"{crops_dir}/{c['img']}")
            ok += 1; valid.add(c["img"])
        except Exception as e:
            fail += 1
            if fail <= 5:
                print(f"[crops] crop failed {c['img']} ({exp}/{well} {x},{y}): {e}")
        if (i + 1) % 500 == 0:
            print(f"[crops] {i + 1}/{len(cells)}  ok={ok} blank={blank} fail={fail}")
    print(f"[crops] done: {ok} written, {blank} blank(placeholder), {fail} failed -> {crops_dir}")
    if not sample:
        _sync_has_crop(idx, valid, out)
    return crops_dir


def _sync_has_crop(idx, valid, out):
    """Update index.json has_crop to reflect what actually rendered (recovers Kyle's false-blanks that now
    crop; demotes any that came back blank), so the app shows real crops and only truly-empty slots stay blank."""
    recovered = demoted = 0
    for d in idx["pcData"].values():
        for b in d["strip"]:
            for c in b["cells"]:
                if not (c and c.get("img")):
                    continue
                now = c["img"] in valid
                if now and not c.get("has_crop"):
                    recovered += 1
                elif not now and c.get("has_crop"):
                    demoted += 1
                c["has_crop"] = now
    with open(f"{out}/index.json", "w") as f:
        json.dump(idx, f)
    print(f"[crops] index.json has_crop synced: +{recovered} recovered, -{demoted} demoted blank")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=0, help="only render first N crops to _crops_sample/ (preview)")
    ap.add_argument("--out", default=PCS_OUT)
    build(ap.parse_args().sample, ap.parse_args().out)
