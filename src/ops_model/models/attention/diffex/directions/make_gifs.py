"""Render NTC→KO animation GIFs for specific (target, cell) traversals.

Reproduces the exact frames of a per-cell strip (same seed 1234+cell, deterministic
mean_diff direction, same CFG guidance/null baseline as traverse) and writes an
animated GIF ordered most-NTC-like → most-KO-like (ping-pong loop). Each frame gets a
FIXED label "NTC → {target}" whose two ends brighten (cyan↔red) with a progress bar to
show position — constant text, no flicker. GPU.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from ..classifier.config import DEFAULT_OUT_ROOT, GRAINS, slugify
from .config import DirConfig
from .data import gather
from .rank import supervised_direction
from .traverse import _sample_guided, load_diffae

# (grain, target, cell index into control cells, guidance w, short label)
SPECS = [
    ("geneKO", "TIMM23", 6, 5.0, "TIMM23"),
    ("complex", "Chaperonin-containing T-complex", 6, 5.0, "CCT"),
    ("complex", "Actin-related protein 2/3 complex, ARPC1A-ACTR3B-ARPC5 variant", 5, 5.0, "Arp2/3"),
    ("geneKO", "POLR1B", 7, 5.0, "POLR1B"),
    ("geneKO", "HSPA5", 6, 5.0, "HSPA5"),
    ("geneKO", "POLR2C", 7, 5.0, "POLR2C"),
]

_NTC = (0, 200, 255)     # cyan (NTC end)
_KO = (255, 90, 60)      # red  (KO end)
_DIM = (110, 110, 110)


def _font(sz):
    try:
        import matplotlib.font_manager as fm
        return ImageFont.truetype(fm.findfont("DejaVu Sans"), sz)
    except Exception:
        return ImageFont.load_default()


def _lerp(a, b, t):
    t = float(np.clip(t, 0, 1))
    return tuple(int(round(a[i] + (b[i] - a[i]) * t)) for i in range(3))


def _labeled(cell_u8, pos, label, W=340, hdr=54):
    """cell_u8 (H,W) uint8; pos in [0,1] (0=NTC, 1=KO)."""
    canvas = Image.new("RGB", (W, hdr + W), (0, 0, 0))
    cell = Image.fromarray(cell_u8).resize((W, W), Image.NEAREST).convert("RGB")
    canvas.paste(cell, (0, hdr))
    d = ImageDraw.Draw(canvas)
    f = _font(22)
    parts = [("NTC", _lerp(_DIM, _NTC, 1 - pos)),
             ("  →  ", (225, 225, 225)),
             (label, _lerp(_DIM, _KO, pos))]
    widths = [d.textlength(t, font=f) for t, _ in parts]
    x = (W - sum(widths)) / 2
    for (t, c), wdt in zip(parts, widths):
        d.text((x, 12), t, font=f, fill=c)
        x += wdt
    bx0, bx1, by = 14, W - 14, hdr - 12
    d.rectangle([bx0, by, bx1, by + 6], fill=(55, 55, 55))
    d.rectangle([bx0, by, bx0 + (bx1 - bx0) * pos, by + 6], fill=_lerp(_NTC, _KO, pos))
    return canvas


_ANTI = (255, 170, 40)   # amber — anti-KO extreme (−label)


def _labeled3(cell_u8, pos, label, W=340, hdr=58):
    """Full-axis header: −label ← NTC(center) → label. pos in [0,1], 0.5 = NTC (α=0)."""
    canvas = Image.new("RGB", (W, hdr + W), (0, 0, 0))
    cell = Image.fromarray(cell_u8).resize((W, W), Image.NEAREST).convert("RGB")
    canvas.paste(cell, (0, hdr))
    d = ImageDraw.Draw(canvas)
    # auto-shrink font so long labels (−label / label) don't collide with centered NTC
    left = f"−{label}"
    sz = 19
    while sz > 11:
        f = _font(sz)
        side = max(d.textlength(left, font=f), d.textlength(label, font=f))
        if side <= W / 2 - d.textlength("NTC", font=f) / 2 - 8:
            break
        sz -= 1
    f = _font(sz)
    aL = max(0.0, 1 - abs(pos - 0.0) / 0.5)
    aM = max(0.0, 1 - abs(pos - 0.5) / 0.5)
    aR = max(0.0, 1 - abs(pos - 1.0) / 0.5)
    d.text((10, 11), f"−{label}", font=f, fill=_lerp(_DIM, _ANTI, aL))
    wm = d.textlength("NTC", font=f)
    d.text(((W - wm) / 2, 11), "NTC", font=f, fill=_lerp(_DIM, _NTC, aM))
    wr = d.textlength(label, font=f)
    d.text((W - 10 - wr, 11), label, font=f, fill=_lerp(_DIM, _KO, aR))
    bx0, bx1, by = 14, W - 14, hdr - 14
    d.rectangle([bx0, by, bx1, by + 6], fill=(55, 55, 55))
    cx = (bx0 + bx1) / 2
    d.rectangle([cx - 1, by - 3, cx + 1, by + 9], fill=_NTC)   # NTC center tick
    mx = bx0 + (bx1 - bx0) * pos
    d.ellipse([mx - 5, by - 3, mx + 5, by + 9], fill=(_KO if pos > 0.5 else _ANTI))
    return canvas


def _strip(frames, gap=4):
    w, h = frames[0].size
    n = len(frames)
    c = Image.new("RGB", (n * w + (n + 1) * gap, h + 2 * gap), (0, 0, 0))
    for i, fr in enumerate(frames):
        c.paste(fr, (gap + i * (w + gap), gap))
    return c


@torch.no_grad()
def _render_review(ctx, cell, w, label):
    """Two review styles from one set of generated frames: full axis + NTC→KO half."""
    dev, cfg, slug, out, embs, labels, fixed_dir, gap, diffae, null_base = ctx
    ci = np.flatnonzero(labels == 0)
    z0 = torch.as_tensor(embs[ci[cell]:ci[cell] + 1], dtype=torch.float32).to(dev)
    H = cfg.crop_size
    ge = torch.Generator(device=dev).manual_seed(1234 + cell)
    xT = torch.randn(1, 1, H, H, generator=ge, device=dev)
    alphas = sorted(cfg.alphas); amin, amax = alphas[0], alphas[-1]
    raw = []
    for a in alphas:
        img = _sample_guided(diffae, xT.clone(), z0 + (a * gap) * fixed_dir, null_base, w, cfg)
        raw.append(np.clip((img.cpu().numpy()[0, 0] + 1) / 2, 0, 1))
    sd = out / "strips"; sd.mkdir(parents=True, exist_ok=True)

    # FULL axis: −label ← NTC(center) → label. Start at NTC → +KO → back → −KO → back.
    # Long settle at the two extremes, SHORT pause at NTC.
    full = [_labeled3((u * 255).astype("uint8"), (a - amin) / (amax - amin), label)
            for a, u in zip(alphas, raw)]
    m = len(full) // 2; n = len(full); He, Hm = 5, 2
    idx = ([m] * Hm + list(range(m + 1, n)) + [n - 1] * He           # NTC → +label, hold
           + list(range(n - 2, m - 1, -1)) + [m] * Hm                # back to NTC (short)
           + list(range(m - 1, -1, -1)) + [0] * He                   # NTC → −label, hold
           + list(range(1, m + 1)) + [m] * Hm)                       # back to NTC (short)
    seq = [full[i] for i in idx]
    seq[0].save(sd / f"{slug}_w{w:g}_cell{cell}_axis.gif", save_all=True,
                append_images=seq[1:], duration=180, loop=0)
    _strip(full).save(sd / f"{slug}_w{w:g}_cell{cell}_axis_strip.png")

    # HALF: true NTC (α=0) → label; pause at both ends
    pa = [(a, u) for a, u in zip(alphas, raw) if a >= 0]
    amx = pa[-1][0]
    half = [_labeled((u * 255).astype("uint8"), a / amx, label) for a, u in pa]
    sh = [half[0]] * 5 + half + [half[-1]] * 6 + half[-2:0:-1] + [half[0]] * 2
    sh[0].save(sd / f"{slug}_w{w:g}_cell{cell}_half.gif", save_all=True,
               append_images=sh[1:], duration=180, loop=0)
    _strip(half).save(sd / f"{slug}_w{w:g}_cell{cell}_half_strip.png")
    print(f"review {slug} cell{cell}: axis+half gif/strip written")
    return slug


def run_review(specs=None, device="cuda"):
    specs = specs or [("geneKO", "GBF1", 2, 5.0, "GBF1"),
                      ("complex", "mTORC1 complex", 2, 5.0, "mTORC1")]
    return [_render_review(_setup(g, t, DEFAULT_OUT_ROOT, device), c, w, lab)
            for g, t, c, w, lab in specs]


def render_all_review(grain, target, label, w=5.0, cells=None, device="cuda"):
    """Both styles (3-way axis + 2-way half), GIF + panel PNG, for every traversed cell."""
    ctx = _setup(grain, target, DEFAULT_OUT_ROOT, device)
    n_ctrl = int((ctx[5] == 0).sum())                 # ctx[5] = labels
    cells = list(cells) if cells is not None else list(range(min(ctx[1].n_traverse, n_ctrl)))
    return [_render_review(ctx, c, w, label) for c in cells]


def _setup(grain, target, out_root, device):
    """Expensive per-target setup shared by all cells: gather + direction + model."""
    dev = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    cfg = DirConfig(grain=grain, target=target, device=device)
    slug = slugify(target)
    out = Path(out_root) / "directions" / grain / slug
    cache = out / "cache"
    tag = f"{slug}_{cfg.crop_size}"
    _, embs, labels = gather(
        cfg, str(cache / f"crops_{tag}.npz"), str(cache / f"celldino_{tag}.npz"))
    d, _, _, _ = supervised_direction(embs, labels, cfg)
    gap = float(np.linalg.norm(embs[labels == 1].mean(0) - embs[labels == 0].mean(0)))
    fixed_dir = torch.as_tensor(d, dtype=torch.float32, device=dev)[None]
    diffae = load_diffae(cfg, dev)
    null_base = diffae.null_emb.detach()[None].to(dev)
    return dev, cfg, slug, out, embs, labels, fixed_dir, gap, diffae, null_base


@torch.no_grad()
def _render_cell(ctx, cell, w, label):
    dev, cfg, slug, out, embs, labels, fixed_dir, gap, diffae, null_base = ctx
    ctrl_idx = np.flatnonzero(labels == 0)
    z0 = torch.as_tensor(embs[ctrl_idx[cell]:ctrl_idx[cell] + 1], dtype=torch.float32).to(dev)
    H = cfg.crop_size
    ge = torch.Generator(device=dev).manual_seed(1234 + cell)
    xT = torch.randn(1, 1, H, H, generator=ge, device=dev)

    alphas = sorted(cfg.alphas)                       # ascending = NTC → KO
    amin, amax = alphas[0], alphas[-1]
    frames = []
    for a in alphas:
        img = _sample_guided(diffae, xT.clone(), z0 + (a * gap) * fixed_dir, null_base, w, cfg)
        u = np.clip((img.cpu().numpy()[0, 0] + 1) / 2, 0, 1)
        pos = (a - amin) / (amax - amin)
        frames.append(_labeled((u * 255).astype("uint8"), pos, label))

    # ping-pong with a hold at each extreme; slower playback
    seq = [frames[0]] * 3 + frames + [frames[-1]] * 4 + frames[-2:0:-1] + [frames[0]] * 2
    gif = out / "strips" / f"{slug}_w{w:g}_cell{cell}.gif"
    seq[0].save(gif, save_all=True, append_images=seq[1:], duration=180, loop=0)
    print(f"wrote {gif}  ({len(alphas)} frames, gap={gap:.2f})")
    return str(gif)


def make_gif(grain, target, cell, w, label, out_root=DEFAULT_OUT_ROOT, device="cuda"):
    ctx = _setup(grain, target, out_root, device)
    return _render_cell(ctx, cell, w, label)


def make_all_gifs(grain, target, label, w=5.0, cells=None, out_root=DEFAULT_OUT_ROOT, device="cuda"):
    """Render GIFs for every traversed cell of one target (setup done once)."""
    ctx = _setup(grain, target, out_root, device)
    n_ctrl = int((ctx[5] == 0).sum())                 # ctx[5] = labels
    cells = list(cells) if cells is not None else list(range(min(ctx[1].n_traverse, n_ctrl)))
    return [_render_cell(ctx, c, w, label) for c in cells]


def run_all(specs=SPECS, device="cuda"):
    return [make_gif(g, t, c, w, lab, device=device) for g, t, c, w, lab in specs]


if __name__ == "__main__":
    from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs
    submit_parallel_jobs(
        jobs_to_submit=[{"name": "diffex_gifs", "func": run_all, "kwargs": {},
                         "metadata": {"stage": "gifs"}}],
        experiment="diffex_gifs",
        slurm_params={"slurm_partition": "gpu", "gpus_per_node": 1, "cpus_per_task": 8,
                      "mem_gb": 64, "timeout_min": 60,
                      "slurm_constraint": "[a100_80|h100|h200|6000_blackwell]"},
        log_dir="diffex_gifs", wait_for_completion=False,
    )
