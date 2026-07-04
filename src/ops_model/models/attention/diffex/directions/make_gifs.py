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
def _render_review(ctx, cell, w, label, tag="", styles=("axis",)):
    """Render traversal styles from one set of frames. DEFAULT = three-way ('axis',
    −label ← NTC(center) → label). Pass styles=('axis','half') or ('half',) for the
    two-way NTC→KO view. `tag` is inserted into filenames (e.g. '_v2')."""
    dev, cfg, slug, out, embs, labels, fixed_dir, gap, diffae, null_base = ctx
    ci = np.flatnonzero(labels == 0)
    z0 = torch.as_tensor(embs[ci[cell]:ci[cell] + 1], dtype=torch.float32).to(dev)
    H = cfg.crop_size
    ge = torch.Generator(device=dev).manual_seed(1234 + cell)
    xT = torch.randn(1, 1, H, H, generator=ge, device=dev)
    all_alphas = sorted(cfg.alphas); amin, amax = all_alphas[0], all_alphas[-1]
    # only decode the α needed: axis needs the full range, half only α≥0
    alphas = all_alphas if "axis" in styles else [a for a in all_alphas if a >= 0]
    raw = {}
    for a in alphas:
        img = _sample_guided(diffae, xT.clone(), z0 + (a * gap) * fixed_dir, null_base, w, cfg)
        raw[a] = np.clip((img.cpu().numpy()[0, 0] + 1) / 2, 0, 1)
    sd = out / "strips"; sd.mkdir(parents=True, exist_ok=True)

    if "axis" in styles:  # FULL 3-way: −label ← NTC(center) → label; NTC→+→−→ back
        full = [_labeled3((raw[a] * 255).astype("uint8"), (a - amin) / (amax - amin), label)
                for a in all_alphas]
        m = len(full) // 2; n = len(full); He, Hm = 5, 2
        idx = ([m] * Hm + list(range(m + 1, n)) + [n - 1] * He
               + list(range(n - 2, m - 1, -1)) + [m] * Hm
               + list(range(m - 1, -1, -1)) + [0] * He
               + list(range(1, m + 1)) + [m] * Hm)
        seq = [full[i] for i in idx]
        seq[0].save(sd / f"{slug}_w{w:g}_cell{cell}{tag}_axis.gif", save_all=True,
                    append_images=seq[1:], duration=180, loop=0)
        _strip(full).save(sd / f"{slug}_w{w:g}_cell{cell}{tag}_axis_strip.png")

    if "half" in styles:  # TWO-WAY (default): true NTC (α=0) → label; pause at both ends
        pa = [a for a in all_alphas if a >= 0]; amx = pa[-1]
        half = [_labeled((raw[a] * 255).astype("uint8"), a / amx, label) for a in pa]
        sh = [half[0]] * 5 + half + [half[-1]] * 6 + half[-2:0:-1] + [half[0]] * 2
        sh[0].save(sd / f"{slug}_w{w:g}_cell{cell}{tag}_half.gif", save_all=True,
                   append_images=sh[1:], duration=180, loop=0)
        _strip(half).save(sd / f"{slug}_w{w:g}_cell{cell}{tag}_half_strip.png")
    print(f"review {slug} cell{cell}: {'+'.join(styles)} written")
    return slug


def run_review(specs=None, device="cuda"):
    specs = specs or [("geneKO", "GBF1", 2, 5.0, "GBF1"),
                      ("complex", "mTORC1 complex", 2, 5.0, "mTORC1")]
    return [_render_review(_setup(g, t, DEFAULT_OUT_ROOT, device), c, w, lab)
            for g, t, c, w, lab in specs]


def run_v2_review(device="cuda"):
    """Compare the phase_v2_aug (dihedral) model on GBF1 + mTORC1, cell2, at w=5 and w=8."""
    ckpt = f"{DEFAULT_OUT_ROOT}/diffae/phase_v2_aug/diffae_best.pt"
    specs = [("geneKO", "GBF1", 2, "GBF1"), ("complex", "mTORC1 complex", 2, "mTORC1")]
    outs = []
    for g, t, cell, lab in specs:
        ctx = _setup(g, t, DEFAULT_OUT_ROOT, device, ckpt=ckpt)
        for w in (5.0, 8.0):
            outs.append(_render_review(ctx, cell, w, lab, tag="_v2"))
    return outs


def render_flow(grain, target, label, cells=(0, 2, 3, 5), w=5.0, n_record=10,
                device="cuda", ckpt=None, two_way=False, overshoot=1.0):
    """OPTIONAL traversal via CellFlow-style conditional flow matching (see flow.py).
    Learns a control→KD velocity field in CellDINO space, integrates the ODE from each
    control cell, decodes each step with the frozen DiffAE. DEFAULT 3-way (−label ← NTC →
    label; anti arm is a backward extrapolation). two_way=True → forward-only NTC→KO.
    overshoot = ODE end time t_max: 1.0 lands at the KD manifold; >1 overshoots past it
    (overshoot≈3 ≈ mean-diff's α=3 drama). Filenames get '_o{overshoot}' when ≠1."""
    from .flow import train_flow, integrate_flow, integrate_flow_bidir
    ctx = _setup(grain, target, DEFAULT_OUT_ROOT, device, ckpt=ckpt)
    dev, cfg, slug, out, embs, labels, fixed_dir, gap, diffae, null_base = ctx
    net = train_flow(embs, labels, dev, seed=cfg.seed)
    sd = out / "strips"; sd.mkdir(parents=True, exist_ok=True)
    ci = np.flatnonzero(labels == 0)
    H = cfg.crop_size
    otag = "" if overshoot == 1.0 else f"_o{overshoot:g}"
    outs = []
    for cell in cells:
        z0 = torch.as_tensor(embs[ci[cell]:ci[cell] + 1], dtype=torch.float32).to(dev)
        ge = torch.Generator(device=dev).manual_seed(1234 + cell)
        xT = torch.randn(1, 1, H, H, generator=ge, device=dev)
        if two_way:
            traj = integrate_flow(net, z0, dev, n_record=n_record, t_max=overshoot)
            n = traj.shape[0]
            frames = [_labeled(
                (np.clip((_sample_guided(diffae, xT.clone(), traj[k:k + 1], null_base, w, cfg)
                          .cpu().numpy()[0, 0] + 1) / 2, 0, 1) * 255).astype("uint8"),
                k / (n - 1), label) for k in range(n)]
            sq = [frames[0]] * 5 + frames + [frames[-1]] * 6 + frames[-2:0:-1] + [frames[0]] * 2
            suffix = "_flow2way"
        else:
            traj = integrate_flow_bidir(net, z0, dev, n_record=n_record, t_max=overshoot)
            n = traj.shape[0]
            frames = [_labeled3(
                (np.clip((_sample_guided(diffae, xT.clone(), traj[k:k + 1], null_base, w, cfg)
                          .cpu().numpy()[0, 0] + 1) / 2, 0, 1) * 255).astype("uint8"),
                k / (n - 1), label) for k in range(n)]
            m = n // 2; He, Hm = 5, 2
            idx = ([m] * Hm + list(range(m + 1, n)) + [n - 1] * He
                   + list(range(n - 2, m - 1, -1)) + [m] * Hm
                   + list(range(m - 1, -1, -1)) + [0] * He
                   + list(range(1, m + 1)) + [m] * Hm)
            sq = [frames[i] for i in idx]
            suffix = "_flow"
        gif = sd / f"{slug}{suffix}{otag}_cell{cell}.gif"
        sq[0].save(gif, save_all=True, append_images=sq[1:], duration=180, loop=0)
        _strip(frames).save(sd / f"{slug}{suffix}{otag}_cell{cell}_strip.png")
        print(f"wrote {gif}  ({n} steps)")
        outs.append(str(gif))
    return outs


def compare_ckpts(grain, target, cell, w, label, ckpts, device="cuda"):
    """Render the same (target, cell, w) under multiple checkpoints, tagged, for A/B compare.
    ckpts: {tag: ckpt_path} e.g. {'_v1': '.../phase_v1/diffae_best.pt', '_v2': '.../phase_v2_aug/...'}"""
    out = []
    for tag, ckpt in ckpts.items():
        ctx = _setup(grain, target, DEFAULT_OUT_ROOT, device, ckpt=ckpt)
        out.append(_render_review(ctx, cell, w, label, tag=tag))
    return out


def render_all_review(grain, target, label, w=5.0, cells=None, device="cuda", ckpt=None, tag="",
                      marker_channel=None, channel=None):
    """Both styles (3-way axis + 2-way half), GIF + panel PNG, for the given cells.
    ckpt overrides the DiffAE checkpoint; tag suffixes filenames. For fluor pass
    marker_channel (fluor-CSV channel) + channel (raw GFP/mCherry) + the marker's DiffAE ckpt."""
    ctx = _setup(grain, target, DEFAULT_OUT_ROOT, device, ckpt=ckpt,
                 marker_channel=marker_channel, channel=channel)
    n_ctrl = int((ctx[5] == 0).sum())                 # ctx[5] = labels
    cells = list(cells) if cells is not None else list(range(min(ctx[1].n_traverse, n_ctrl)))
    return [_render_review(ctx, c, w, label, tag=tag) for c in cells]


def _setup(grain, target, out_root, device, ckpt=None, marker_channel=None, channel=None):
    """Expensive per-target setup shared by all cells: gather + direction + model.
    Fluor: pass marker_channel (fluor-CSV channel) + channel (raw GFP/mCherry) + the marker's
    DiffAE via ckpt. Fluor gets its own out dir (<slug>__<channel>) + cache so it never
    collides with the phase pipeline."""
    dev = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    cfg = DirConfig(grain=grain, target=target, device=device)
    if ckpt:
        cfg.diffae_ckpt = ckpt
    if marker_channel:
        cfg.marker_channel = marker_channel
    if channel:
        cfg.channel = channel
    slug = slugify(target)
    # modality-first layout: directions/<phase|marker>/<grain>/<slug> — keeps each modality's
    # per-target listing separate (phase not overwhelmed by per-marker copies).
    modality = slugify(cfg.marker_channel) if cfg.marker_channel else "phase"
    out = Path(out_root) / "directions" / modality / grain / slug
    cache = out / "cache"
    cache.mkdir(parents=True, exist_ok=True)          # brand-new targets have no cache dir yet
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

    # Default = THREE-WAY: −label ← NTC(center) → label. Start at NTC → +KO → back → −KO → back.
    alphas = sorted(cfg.alphas); amin, amax = alphas[0], alphas[-1]
    frames = []
    for a in alphas:
        img = _sample_guided(diffae, xT.clone(), z0 + (a * gap) * fixed_dir, null_base, w, cfg)
        u = np.clip((img.cpu().numpy()[0, 0] + 1) / 2, 0, 1)
        frames.append(_labeled3((u * 255).astype("uint8"), (a - amin) / (amax - amin), label))

    m = len(frames) // 2; n = len(frames); He, Hm = 5, 2
    idx = ([m] * Hm + list(range(m + 1, n)) + [n - 1] * He
           + list(range(n - 2, m - 1, -1)) + [m] * Hm
           + list(range(m - 1, -1, -1)) + [0] * He
           + list(range(1, m + 1)) + [m] * Hm)
    seq = [frames[i] for i in idx]
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
