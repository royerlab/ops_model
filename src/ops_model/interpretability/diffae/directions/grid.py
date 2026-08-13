"""Composite existing per-target NTC→KO GIFs into a synchronized grid-canvas GIF.

Each tile is one target's already-rendered animation (same cell, same w); all tiles share
the identical frame schedule, so we stack them frame-by-frame into one canvas that morphs
every perturbation at once. Per-frame durations are read from the source GIFs so the
end-of-traversal settle/hold is preserved. Pure image compositing — no GPU.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageSequence

from ..classifier.config import slugify


def _font(sz):
    try:
        import matplotlib.font_manager as fm
        return ImageFont.truetype(fm.findfont("DejaVu Sans"), sz)
    except Exception:
        return ImageFont.load_default()


def _frames_dur(gif_path):
    im = Image.open(gif_path)
    frames, durs = [], []
    for f in ImageSequence.Iterator(im):
        frames.append(f.convert("RGB").copy())
        durs.append(int(f.info.get("duration", 180)))
    return frames, durs


def make_labeled_grid(grid_paths, out_path, row_labels, col_labels, title=None, tile_w=260):
    """Composite a synchronized R×C matrix of GIFs with row/col axis labels (a disentanglement
    figure). grid_paths: list of rows, each a list of C gif paths. Preserves frame durations."""
    seqs = [[_frames_dur(p) for p in row] for row in grid_paths]
    F = min(len(fr) for row in seqs for fr, _ in row)
    durs = seqs[0][0][1][:F]
    tw0, th0 = seqs[0][0][0][0].size
    tw, th = tile_w, round(th0 * tile_w / tw0)
    R, C = len(grid_paths), len(grid_paths[0])
    gap, lm, tm = 8, 92, (58 if title else 30)       # left margin (row labels), top margin
    W = lm + C * tw + (C + 1) * gap
    H = tm + 24 + R * th + (R + 1) * gap              # +24 for col-label row
    tfont, lfont = _font(30), _font(20)

    frames = []
    for k in range(F):
        cv = Image.new("RGB", (W, H), (0, 0, 0)); d = ImageDraw.Draw(cv)
        if title:
            d.text(((W - d.textlength(title, font=tfont)) / 2, 12), title, font=tfont, fill=(235, 235, 235))
        for c, cl in enumerate(col_labels):          # column headers
            x0 = lm + gap + c * (tw + gap)
            d.text((x0 + (tw - d.textlength(cl, font=lfont)) / 2, tm), cl, font=lfont, fill=(0, 200, 255))
        for r, rl in enumerate(row_labels):          # row labels (left, rotated-ish: just left-aligned)
            y0 = tm + 24 + gap + r * (th + gap)
            d.text((8, y0 + th / 2 - 10), rl, font=lfont, fill=(255, 170, 40))
        for r in range(R):
            for c in range(C):
                tile = seqs[r][c][0][k].resize((tw, th), Image.BILINEAR)
                cv.paste(tile, (lm + gap + c * (tw + gap), tm + 24 + gap + r * (th + gap)))
        frames.append(cv)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out_path, save_all=True, append_images=frames[1:], duration=durs, loop=0)
    print(f"wrote {out_path}  ({R}x{C}, {F} frames, {W}x{H})")
    return str(out_path)


def build_tiles(grain, cell, out_root, csv_dir=None, names=None,
                n_genes=50, n_complex=20, w=5.0, suffix="", modality="phase"):
    """suffix: '' | '_axis' | '_half'. names overrides the CSV top-N list.
    modality: 'phase' or a marker slug (directions/<modality>/<grain>/<slug>)."""
    if names is None:
        if grain == "geneKO":
            df = pd.read_csv(f"{csv_dir}/k10_ranked_all_geneKOs.csv").sort_values("rank_by_K10_mAP").head(n_genes)
            names = df["geneKO"].tolist()
        else:
            df = pd.read_csv(f"{csv_dir}/k10_ranked_all_complexes.csv").sort_values("rank_by_K10_mAP").head(n_complex)
            names = df["complex_name"].tolist()
    tiles, missing = [], []
    for nm in names:
        s = slugify(nm)
        p = f"{out_root}/directions/{modality}/{grain}/{s}/strips/{s}_w{w:g}_cell{cell}{suffix}.gif"
        (tiles if os.path.exists(p) else missing).append(p)
    if missing:
        print(f"  [{grain} cell{cell}{suffix}] {len(missing)} tiles missing (skipped)")
    return tiles


def make_grid(tiles, out_path, ncols=10, gap=6, bg=(0, 0, 0), title=None, tile_w=240):
    """tiles: list of GIF paths (same schedule). Preserves per-frame durations."""
    ref_frames, durs = _frames_dur(tiles[0])
    F = len(ref_frames)
    seqs = [ref_frames]
    for t in tiles[1:]:
        fr, _ = _frames_dur(t)
        seqs.append(fr)
    F = min(F, min(len(s) for s in seqs))
    durs = durs[:F]

    tw0, th0 = seqs[0][0].size
    if tile_w:
        tw, th = tile_w, round(th0 * tile_w / tw0)
    else:
        tw, th = tw0, th0
    n = len(seqs)
    nrows = (n + ncols - 1) // ncols
    hdr = 40 if title else 0
    W = ncols * tw + (ncols + 1) * gap
    H = hdr + nrows * th + (nrows + 1) * gap
    tfont = _font(26)

    out_frames = []
    for k in range(F):
        canvas = Image.new("RGB", (W, H), bg)
        if title:
            d = ImageDraw.Draw(canvas)
            d.text(((W - d.textlength(title, font=tfont)) / 2, 8), title, font=tfont, fill=(235, 235, 235))
        for i, s in enumerate(seqs):
            r, c = divmod(i, ncols)
            tile = s[k].resize((tw, th), Image.BILINEAR) if tile_w else s[k]
            canvas.paste(tile, (gap + c * (tw + gap), hdr + gap + r * (th + gap)))
        out_frames.append(canvas)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out_frames[0].save(out_path, save_all=True, append_images=out_frames[1:],
                       duration=durs, loop=0)
    print(f"wrote {out_path}  ({n} tiles, {F} frames, {W}x{H})")
    return str(out_path)
