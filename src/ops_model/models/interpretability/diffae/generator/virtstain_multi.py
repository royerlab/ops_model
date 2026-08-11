"""Multi-marker virtual staining: ONE spatial DiffAE predicts every fluor marker from phase,
switched by a learned marker-id embedding. Pools paired (phase, marker) crops across all markers
(each marker's own exps); the phase image is concatenated into the UNet (registered stain) and the
marker id selects which channel to render.

    python -m ops_model.models.interpretability.diffae.generator.virtstain_multi --submit --cap 2500 --epochs 120
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from diffusers import DDIMScheduler, DDPMScheduler
from torch.utils.data import DataLoader, TensorDataset

from ..classifier.config import slugify
from .config import DiffAEConfig
from .data import load_diffae_crops
from .model import DiffAE
from .train import _pearson

OUT = "/hpc/projects/icd.fast.ops/analysis/virtual_staining/multi_marker"


def markers_list():
    """LIVE fluor markers only: (dir, marker_channel, raw_channel), deduped. 4i/CP markers (raw channel
    CP1_/CP2_/4i_R*) are stained AFTER the live phase acquisition, so their cells have moved/changed and
    the phase→marker registration is broken — that misalignment poisons the spatial conditioning, so they
    are excluded. Live channels (GFP/mCherry/Cy5/farred) are imaged concurrently with phase → registered."""
    from ..traversal import catalog as C
    seen, out = set(), []
    for d, mc, ch in C.complete_markers():
        if ch.startswith(("CP", "4i")):
            continue
        if mc not in seen:
            seen.add(mc); out.append((d, mc, ch))
    return out


# Every scored cell per marker (concat of Alex's 6 per-cell shards, incl NTC) — NOT the acc>0.5 qualifying
# subset. Virtual staining needs a representative pool of paired (phase, marker) crops, so no distinctiveness
# filtering: use all cells (74k–1.3M/marker) and sample ≤cap per marker. Built by scratchpad/build_allcells.py.
ALL_CELLS = ("/hpc/projects/icd.fast.ops/models/alex_lin_attention/v5/fluorescence/"
             "misc/all_cells_bychannel.parquet")


def load_multi(markers, cap, cache_root):
    """Per marker: gather ≤cap paired (phase, marker) crops from ALL_CELLS (every cell, no accuracy filter),
    materialize marker (target) + phase (cond) crops, CellDINO-embed phase; cache per marker."""
    Path(cache_root).mkdir(parents=True, exist_ok=True)
    pre = pd.read_parquet(ALL_CELLS, columns=["channel", "gene", "experiment", "well",
                                              "segmentation", "x_pheno", "y_pheno"])
    Xs, Es, Ps, Ms, kept = [], [], [], [], []
    for d, mc, ch in markers:
        sl = slugify(mc)
        rows = pre[pre["channel"] == mc]
        if rows.empty:
            print(f"[skip] {mc}: no rows in v5 qualifying"); continue
        try:
            cfg = DiffAEConfig(marker_channel=mc, channel=ch, cond_channel="Phase2D",
                               spatial_cond=True, n_crops=cap, seed=len(kept))
            cfg._fluor_rows = rows
            x, e, p = load_diffae_crops(
                cfg, crops_cache=f"{cache_root}/{sl}_marker.npz",
                emb_cache=f"{cache_root}/{sl}_emb.npz", cond_cache=f"{cache_root}/{sl}_phase.npz",
                return_cond_images=True)
        except Exception as exc:                                  # noqa: BLE001 — skip finicky markers, keep going
            print(f"[skip] {mc}: {exc}"); continue
        mid = len(kept); kept.append((d, mc, ch))
        Xs.append(x); Es.append(e); Ps.append(p); Ms.append(np.full(len(x), mid, np.int64))
        print(f"[marker {mid}] {mc}: {len(x)} crops")
    X = np.concatenate(Xs); E = np.concatenate(Es); P = np.concatenate(Ps); M = np.concatenate(Ms)
    print(f"[multi] {len(kept)} markers, {len(X)} total crops")
    return X, E, P, M, kept


@torch.no_grad()
def _sample_marker(model, xT, emb, cond_img, marker_id, cfg, dev):
    fwd = DDIMScheduler(num_train_timesteps=cfg.train_timesteps); fwd.set_timesteps(cfg.ddim_steps)
    c = model.cond(emb, marker_id); x = xT
    for t in fwd.timesteps:
        x = fwd.step(model.denoise(x, t, c, cond_img), t, x).prev_sample
    return x


def train_multi(X, E, P, M, cfg, out_dir, epochs, batch, device):
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    model = DiffAE(cfg).to(dev)
    ema = copy.deepcopy(model).eval()
    for pr in ema.parameters():
        pr.requires_grad_(False)
    sched = DDPMScheduler(num_train_timesteps=cfg.train_timesteps)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    crit = nn.MSELoss()
    n_probe = min(64, len(X) // 20)
    loader = DataLoader(TensorDataset(torch.as_tensor(X[:-n_probe]), torch.as_tensor(E[:-n_probe]),
                                      torch.as_tensor(P[:-n_probe]), torch.as_tensor(M[:-n_probe])),
                        batch_size=batch, shuffle=True, drop_last=True)
    scaler = torch.cuda.amp.GradScaler(enabled=dev.type == "cuda")
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    state = out / "train_state.pt"; start = 0

    def ema_up():
        for e_, p_ in zip(ema.parameters(), model.parameters()):
            e_.mul_(cfg.ema_decay).add_(p_.detach(), alpha=1 - cfg.ema_decay)
        for eb, pb in zip(ema.buffers(), model.buffers()):
            eb.copy_(pb)

    if state.exists():
        st = torch.load(state, map_location=dev)
        model.load_state_dict(st["model"]); ema.load_state_dict(st["ema"]); opt.load_state_dict(st["opt"])
        start = st["epoch"] + 1; print(f"[resume] from epoch {start}")
    for ep in range(start, epochs):
        model.train(); tot = 0.0
        for x, e, p, m in loader:
            x, e, p, m = x.to(dev), e.to(dev), p.to(dev), m.to(dev)
            if cfg.cond_dropout > 0:                               # drop the CellDINO emb (keep marker id)
                drop = torch.rand(e.shape[0], device=dev) < cfg.cond_dropout
                if drop.any():
                    e = torch.where(drop[:, None], model.null_emb[None].to(e.dtype), e)
            noise = torch.randn_like(x)
            t = torch.randint(0, cfg.train_timesteps, (x.shape[0],), device=dev).long()
            noisy = sched.add_noise(x, noise, t); opt.zero_grad()
            with torch.autocast("cuda", enabled=dev.type == "cuda"):
                loss = crit(model(noisy, t, e, p, m), noise)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); ema_up()
            tot += float(loss) * x.shape[0]
        tot /= len(loader.dataset)
        torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(), "epoch": ep, "loss": tot}, state)
        if (ep + 1) % 10 == 0 or ep == epochs - 1:
            torch.save(ema.state_dict(), out / "diffae_best.pt")
        print(f"epoch {ep:03d}: loss={tot:.4f}", flush=True)
    torch.save(ema.state_dict(), out / "diffae_best.pt")
    return ema


def _plot_montage(rowspecs, n_kept, epoch, loss, ncell, out, eval_name):
    """Render the per-marker phase/pred/real montage from cached rowspecs (no GPU). Each marker is a
    phase/pred/real column; markers pack across PER-per-section, wrapping into stacked sections. Nested
    gridspec keeps phase/pred/real tight, with a big gap BETWEEN sections for the 3-line title."""
    import matplotlib
    matplotlib.use("Agg"); matplotlib.rcParams["pdf.fonttype"] = 42
    import matplotlib.pyplot as plt
    r_by = {name: r for name, _, _, _, r in rowspecs}
    mean_r = float(np.mean(list(r_by.values()))) if r_by else 0.0
    n = len(rowspecs)
    NSEC = 3 if n > 16 else 1                                     # ~3 long horizontal sections for the full panel
    PER = max(1, -(-n // NSEC))                                   # markers/section (ceil) → long rows
    nb = -(-n // PER)
    figH = nb * 3 * 1.28
    fig = plt.figure(figsize=(PER * 1.05, figH))
    top_frac = 1 - 0.85 / figH                                    # reserve a fixed band at the top for the 2-line suptitle
    # Nested gridspec: big gap BETWEEN sections (room for the 3-line title), tight WITHIN each phase/pred/real triplet.
    outer = fig.add_gridspec(nb, PER, hspace=0.12, wspace=0.04, top=top_frac, bottom=0.015)
    axmap = {}
    for k, (name, ph, pr, re, rr) in enumerate(rowspecs):
        band, col = divmod(k, PER)
        inner = outer[band, col].subgridspec(3, 1, hspace=0.03)  # phase/pred/real stay tight together
        for j, (img, cm) in enumerate([(ph, "gray"), (pr, "magma"), (re, "magma")]):
            a = fig.add_subplot(inner[j])
            a.imshow(img, cmap=cm, vmin=-1, vmax=1, aspect="auto"); a.set_xticks([]); a.set_yticks([])
            axmap[(band, col, j)] = a
        org, _, prot = name.rpartition("_")                      # 3-line title: organelle / protein / Pearson
        if not org:
            org, prot = prot, ""
        axmap[(band, col, 0)].set_title(f"{org}\n{prot}\nr={rr:.2f}", fontsize=5.5, pad=3, linespacing=1.25)
    for band in range(nb):                                        # phase/pred/real labels on the left of each section
        for j, lbl in enumerate(["phase", "pred", "real"]):
            a = axmap.get((band, 0, j))
            if a is not None:
                a.set_ylabel(lbl, fontsize=7, rotation=90, labelpad=1)
    ep = f"{epoch}" if epoch is not None else "?"
    ls = f"{loss:.4f}" if loss is not None else "—"
    fig.suptitle(f"Multi-marker virtual staining (phase → fluorescent marker) — {n_kept} live markers, one model\n"
                 f"epoch {ep}  ·  train loss {ls}  ·  {ncell:,} cells trained  ·  overall Pearson r = {mean_r:.3f}",
                 fontsize=11, y=1 - 0.28 / figH)
    fig.savefig(out / eval_name / "multi_montage.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    (out / eval_name / "multi_metrics.json").write_text(json.dumps(
        {"n_markers": n_kept, "epoch": epoch, "loss": loss, "n_train": ncell,
         "mean_pearson": round(mean_r, 3), "per_marker_pearson": {k: round(v, 3) for k, v in r_by.items()}}, indent=2))
    print(f"[eval] {n_kept} markers, epoch {ep}, mean Pearson {mean_r:.3f} -> {out/eval_name/'multi_montage.png'}")


def replot_eval(subdir="eval", out_dir=None):
    """Re-render the montage from the cached arrays (montage_cache.npz) — NO GPU, NO model. Use this to
    iterate on montage layout/spacing without re-running the eval sampling."""
    out = Path(out_dir or OUT)
    d = np.load(out / subdir / "montage_cache.npz", allow_pickle=True)
    rowspecs = list(zip(d["names"].tolist(), d["phase"], d["pred"], d["real"], d["r"].tolist()))
    loss = d["loss"].item()
    if loss is not None and loss != loss:                        # nan → treat as missing
        loss = None
    _plot_montage(rowspecs, int(d["n_kept"]), d["epoch"].item(), loss, int(d["ncell"]), out, subdir)


@torch.no_grad()
def eval_multi(ema, X, E, P, M, kept, cfg, out_dir, dev, epoch=None, loss=None, n_train=None, per_marker=1, eval_name="eval"):
    """Sample each kept marker from phase + marker-id, cache the (phase, pred, real, Pearson) arrays to
    montage_cache.npz (so the montage can be re-rendered later with no GPU via replot_eval), then plot."""
    out = Path(out_dir); (out / eval_name).mkdir(parents=True, exist_ok=True)
    H = cfg.crop_size
    rowspecs = []
    for mid, (_, mc, _) in enumerate(kept):
        idx = np.where(M == mid)[0][-per_marker:]                 # tail cells (least likely in early SGD)
        for i in idx:
            g = torch.Generator(device=dev).manual_seed(100 + int(i))
            xT = torch.randn(1, 1, H, H, generator=g, device=dev)
            e = torch.as_tensor(E[i:i + 1], dtype=torch.float32, device=dev)
            ci = torch.as_tensor(P[i:i + 1], dtype=torch.float32, device=dev)
            mk = torch.as_tensor([mid], dtype=torch.long, device=dev)
            pred = _sample_marker(ema, xT, e, ci, mk, cfg, dev).cpu().numpy()[0, 0]
            rowspecs.append((mc, P[i, 0], pred, X[i, 0], _pearson(pred, X[i, 0])))   # FULL marker name (protein) — no collisions
    ncell = int(n_train) if n_train is not None else int(len(X))
    np.savez_compressed(out / eval_name / "montage_cache.npz",   # everything the montage needs → replot with no GPU
                        names=np.array([s[0] for s in rowspecs]),
                        phase=np.stack([s[1] for s in rowspecs]).astype(np.float32),
                        pred=np.stack([s[2] for s in rowspecs]).astype(np.float32),
                        real=np.stack([s[3] for s in rowspecs]).astype(np.float32),
                        r=np.array([s[4] for s in rowspecs], np.float32),
                        n_kept=len(kept), epoch=epoch, loss=loss if loss is not None else np.nan, ncell=ncell)
    _plot_montage(rowspecs, len(kept), epoch, loss, ncell, out, eval_name)


def run(cap=2500, epochs=120, batch=48, device="cuda"):
    markers = markers_list()
    print(f"[multi] {len(markers)} candidate markers")
    X, E, P, M, kept = load_multi(markers, cap, f"{OUT}/cache")
    cfg = DiffAEConfig(spatial_cond=True, n_markers=len(kept), device=device, epochs=epochs, batch_size=batch)
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    ema = train_multi(X, E, P, M, cfg, OUT, epochs, batch, device)
    st = torch.load(Path(OUT) / "train_state.pt", map_location="cpu")
    (Path(OUT) / "markers.json").write_text(json.dumps([mc for _, mc, _ in kept], indent=2))
    eval_multi(ema, X, E, P, M, kept, cfg, OUT, dev, epoch=st.get("epoch"), loss=st.get("loss"), n_train=int(len(X)))
    return {"n_markers": len(kept), "n_crops": int(len(X))}


def eval_only(cap=2500, batch=48, device="cuda", subdir="eval"):
    """Produce the eval montage/metrics from the CURRENT checkpoint (train_state.pt EMA, newest epoch) without
    finishing training — caches make the load fast. Numbers are a floor (model still undertrained)."""
    X, E, P, M, kept = load_multi(markers_list(), cap, f"{OUT}/cache")
    cfg = DiffAEConfig(spatial_cond=True, n_markers=len(kept), device=device, epochs=1, batch_size=batch)
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    ema = DiffAE(cfg).to(dev).eval()
    st = torch.load(Path(OUT) / "train_state.pt", map_location=dev)
    ema.load_state_dict(st["ema"]); print(f"[eval-only] loaded EMA @ epoch {st['epoch']}")
    loss = st.get("loss")
    if loss is None or (isinstance(loss, float) and loss != loss):   # missing/NaN → read latest from the train log
        import glob, os, re
        for f in sorted(glob.glob("/hpc/mydata/gav.sturm/ops_mono/slurm_logs/diffae/*/*.out"), key=os.path.getmtime)[::-1][:8]:
            m = re.findall(r"epoch \d+: loss=([\d.]+)", open(f).read())
            if m: loss = float(m[-1]); break
    (Path(OUT) / "markers.json").write_text(json.dumps([mc for _, mc, _ in kept], indent=2))
    eval_multi(ema, X, E, P, M, kept, cfg, OUT, dev, epoch=st["epoch"], loss=loss, n_train=int(len(X)), eval_name=subdir)
    return {"epoch": st["epoch"], "n_markers": len(kept)}


_SP = {"slurm_partition": "gpu", "gpus_per_node": 1, "cpus_per_task": 12, "mem_gb": 300,
       "timeout_min": 720, "slurm_constraint": "[a100_80|h100|h200|6000_blackwell]",
       "slurm_setup": ["export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"]}


def submit_chain(n_links=4, cap=2500, epochs=120, batch=48):
    """Auto-relaunch scheme: chain n_links resume-jobs with afterany dependencies so training auto-continues
    from train_state.pt across the 12h wall-clock limit until it reaches `epochs`. Each link resumes
    automatically (train_multi loads train_state.pt); the link that reaches `epochs` runs eval + writes the
    montage. Links that start already at `epochs` are cheap no-ops (load → skip loop → re-eval). afterany =
    the next link runs regardless of how the previous ended (timeout saves train_state every epoch)."""
    from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs
    prev = None
    for i in range(n_links):
        sp = dict(_SP)
        if prev:
            sp["slurm_additional_parameters"] = {"dependency": f"afterany:{prev}"}
        r = submit_parallel_jobs(jobs_to_submit=[{"name": f"vstain_multi_link{i}", "func": run,
                                 "kwargs": {"cap": cap, "epochs": epochs, "batch": batch}}],
                                 experiment="diffae", slurm_params=sp, log_dir="diffae", wait_for_completion=False)
        prev = r["base_job_id"]
        print(f"[chain] link {i}: job {prev}" + (f" (afterany prev)" if i else " (head, starts now)"))
    print(f"[chain] {n_links} links → resumes to epoch {epochs}, final link evals")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=2500, help="cells per marker")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--eval", action="store_true", help="eval the current checkpoint only (no training)")
    ap.add_argument("--chain", type=int, default=0, help="submit N afterany-chained resume jobs to reach --epochs")
    ap.add_argument("--replot", metavar="SUBDIR", help="re-render the montage from a cached eval (no GPU)")
    args = ap.parse_args()
    if args.replot:
        replot_eval(subdir=args.replot)
        return
    if args.chain:
        submit_chain(n_links=args.chain, cap=args.cap, epochs=args.epochs, batch=args.batch)
        return
    func, name = (eval_only, "virtstain_multi_eval") if args.eval else (run, "virtstain_multi")
    kw = {"cap": args.cap, "batch": args.batch} if args.eval else {"cap": args.cap, "epochs": args.epochs, "batch": args.batch}
    if args.submit:
        from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs
        submit_parallel_jobs(jobs_to_submit=[{"name": name, "func": func, "kwargs": kw}],
            experiment="diffae", slurm_params={"slurm_partition": "gpu", "gpus_per_node": 1,
            "cpus_per_task": 12, "mem_gb": 300, "timeout_min": 720,
            "slurm_constraint": "[a100_80|h100|h200|6000_blackwell]",
            "slurm_setup": ["export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"]},
            log_dir="diffae", wait_for_completion=False)
    elif args.eval:
        eval_only(cap=args.cap, batch=args.batch)
    else:
        run(cap=args.cap, epochs=args.epochs, batch=args.batch)


if __name__ == "__main__":
    main()
