"""Multi-marker virtual staining: ONE spatial DiffAE predicts every fluor marker from phase,
switched by a learned marker-id embedding. Pools paired (phase, marker) crops across all markers
(each marker's own exps); the phase image is concatenated into the UNet (registered stain) and the
marker id selects which channel to render.

    python -m ops_model.models.attention.diffex.diffae.virtstain_multi --submit --cap 2500 --epochs 120
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
    from ..viewer import catalog as C
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
        torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(), "epoch": ep}, state)
        if (ep + 1) % 10 == 0 or ep == epochs - 1:
            torch.save(ema.state_dict(), out / "diffae_best.pt")
        print(f"epoch {ep:03d}: loss={tot:.4f}", flush=True)
    torch.save(ema.state_dict(), out / "diffae_best.pt")
    return ema


@torch.no_grad()
def eval_multi(ema, X, E, P, M, kept, cfg, out_dir, dev, per_marker=1):
    """Per-marker held-out-ish montage: for each kept marker, sample the marker from phase + marker-id,
    Pearson(pred, real). Rows = markers, cols = phase | pred | real."""
    out = Path(out_dir); (out / "eval").mkdir(parents=True, exist_ok=True)
    H = cfg.crop_size
    import matplotlib
    matplotlib.use("Agg"); matplotlib.rcParams["pdf.fonttype"] = 42
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    rowspecs, metrics = [], {}
    for mid, (_, mc, _) in enumerate(kept):
        idx = np.where(M == mid)[0][-per_marker:]                 # tail cells (least likely in early SGD)
        for i in idx:
            g = torch.Generator(device=dev).manual_seed(100 + int(i))
            xT = torch.randn(1, 1, H, H, generator=g, device=dev)
            e = torch.as_tensor(E[i:i + 1], dtype=torch.float32, device=dev)
            ci = torch.as_tensor(P[i:i + 1], dtype=torch.float32, device=dev)
            mk = torch.as_tensor([mid], dtype=torch.long, device=dev)
            pred = _sample_marker(ema, xT, e, ci, mk, cfg, dev).cpu().numpy()[0, 0]
            r = _pearson(pred, X[i, 0])
            rowspecs.append((mc.split("_")[0][:14], P[i, 0], pred, X[i, 0], r))
        metrics.setdefault(mc, []).append(None)
    r_by = {}
    fig, ax = plt.subplots(len(rowspecs), 3, figsize=(4.6, 1.5 * len(rowspecs)), squeeze=False)
    for row, (name, ph, pr, re, r) in enumerate(rowspecs):
        for c, (img, cm) in enumerate([(ph, "gray"), (pr, "magma"), (re, "magma")]):
            ax[row, c].imshow(img, cmap=cm, vmin=-1, vmax=1); ax[row, c].set_xticks([]); ax[row, c].set_yticks([])
        ax[row, 0].set_ylabel(name, fontsize=7)
        ax[row, 1].text(0.04, 0.96, f"r={r:.2f}", transform=ax[row, 1].transAxes, fontsize=7, color="white",
                        va="top", path_effects=[pe.withStroke(linewidth=1.5, foreground="black")])
        r_by[name] = r
    for c, t in enumerate(["phase", "predicted", "real"]):
        ax[0, c].set_title(t, fontsize=9)
    fig.suptitle(f"multi-marker virtual staining — {len(kept)} markers, one model", fontsize=10)
    fig.tight_layout(); fig.savefig(out / "eval" / "multi_montage.png", dpi=140, bbox_inches="tight"); plt.close(fig)
    (out / "eval" / "multi_metrics.json").write_text(json.dumps(
        {"n_markers": len(kept), "mean_pearson": round(float(np.mean(list(r_by.values()))), 3),
         "per_marker_pearson": {k: round(v, 3) for k, v in r_by.items()}}, indent=2))
    print(f"[eval] {len(kept)} markers, mean Pearson {np.mean(list(r_by.values())):.3f} -> {out/'eval'/'multi_montage.png'}")


def run(cap=2500, epochs=120, batch=48, device="cuda"):
    markers = markers_list()
    print(f"[multi] {len(markers)} candidate markers")
    X, E, P, M, kept = load_multi(markers, cap, f"{OUT}/cache")
    cfg = DiffAEConfig(spatial_cond=True, n_markers=len(kept), device=device, epochs=epochs, batch_size=batch)
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    ema = train_multi(X, E, P, M, cfg, OUT, epochs, batch, device)
    (Path(OUT) / "markers.json").write_text(json.dumps([mc for _, mc, _ in kept], indent=2))
    eval_multi(ema, X, E, P, M, kept, cfg, OUT, dev)
    return {"n_markers": len(kept), "n_crops": int(len(X))}


def eval_only(cap=2500, batch=48, device="cuda"):
    """Produce the eval montage/metrics from the CURRENT checkpoint (train_state.pt EMA, newest epoch) without
    finishing training — caches make the load fast. Numbers are a floor (model still undertrained)."""
    X, E, P, M, kept = load_multi(markers_list(), cap, f"{OUT}/cache")
    cfg = DiffAEConfig(spatial_cond=True, n_markers=len(kept), device=device, epochs=1, batch_size=batch)
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    ema = DiffAE(cfg).to(dev).eval()
    st = torch.load(Path(OUT) / "train_state.pt", map_location=dev)
    ema.load_state_dict(st["ema"]); print(f"[eval-only] loaded EMA @ epoch {st['epoch']}")
    (Path(OUT) / "markers.json").write_text(json.dumps([mc for _, mc, _ in kept], indent=2))
    eval_multi(ema, X, E, P, M, kept, cfg, OUT, dev)
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
    args = ap.parse_args()
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
