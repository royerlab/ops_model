"""Prototype: DDIM-INVERTED anchors for phase geneKO traversals.

The production traversal seeds a FIXED RANDOM xT per cell, so α=0 is a generic DDPM recon of z0,
not the real anchor cell (see traverse.py). Here we instead DDIM-INVERT each anchor cell to its own
xT (conditioned on z0), then sweep α with the SAME direction. Claim to prove:
  (1) α=0 with the inverted xT reconstructs the REAL anchor cell (vs the generic random-xT α=0), and
  (2) the α-sweep morph toward the KO is preserved.

Same NTC anchor cells v5 uses (top-rank NTC), KIF23 + POLR1B, α 0->5. Everything reused from the
existing traversal stack; the only new step is _ddim(..., inverse=True) to get xT.

    python -m ops_model.models.interpretability.diffae.directions.proto_ddim_anchors --submit
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..classifier.config import slugify
from ..generator.data import normalize
from .config import DirConfig
from .rank import supervised_direction
from .traverse import _ddim_guided, _sample_guided, load_diffae
from ops_model.paths import BASE_PATH

ANALYSIS = f"{BASE_PATH}/analysis"
DD = f"{BASE_PATH}/models/diffex/diffae"
DIR_CACHE = f"{BASE_PATH}/models/diffex/viewer_assets/_directions"

# (label, marker_channel|None, raw_channel, diffae_ckpt, gene). marker_channel=None → phase.
FLUOR_SPECS = [
    ("FastAct_CAPZB", "actin filament_FastAct_SPY555 Live Cell Dye", "mCherry", f"{DD}/fluor_FastAct/diffae_best.pt", "CAPZB"),
    ("TOMM20_TOMM20", "Mitochondria_TOMM20", "CP1_mitochondria_TOMM20", f"{DD}/fluor_Mitochondria_TOMM20/diffae_best.pt", "TOMM20"),
    ("NucleoLive_KIF23", "nucleus_NucleoLIVE Live Cell dye", "mCherry", f"{DD}/fluor_NucleoLive/diffae_best.pt", "KIF23"),
]
PHASE_SPECS = [("phase_KIF23", None, "Phase2D", f"{DD}/phase_v1/diffae_best.pt", "KIF23"),
               ("phase_POLR1B", None, "Phase2D", f"{DD}/phase_v1/diffae_best.pt", "POLR1B")]


def _pearson(a, b) -> float:
    a, b = a.ravel(), b.ravel()
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-12))


@torch.no_grad()
def _direction(cfg, gene, modality, ctrl_embs, mu_ctrl, gather, dev):
    """Load the cached control→KO direction, else compute it. Returns (d_vec, gap, lr_w, lr_b) —
    the LR probe scores re-encoded morphs to measure phenotype strength across α."""
    dcache = Path(DIR_CACHE) / modality / "geneKO" / f"{slugify(gene)}.npz"
    if dcache.exists():
        z = np.load(dcache); return z["d_vec"], float(z["gap"]), z["lr_w"], float(z["lr_b"])
    _, kd_embs = gather(cfg, gene, 1000)
    embs = np.concatenate([kd_embs, ctrl_embs], 0)
    labels = np.concatenate([np.ones(len(kd_embs)), np.zeros(len(ctrl_embs))]).astype(int)
    d_vec, lr_w, lr_b, _ = supervised_direction(embs, labels, cfg)
    return d_vec, float(np.linalg.norm(kd_embs.mean(0) - mu_ctrl)), lr_w, float(lr_b)


@torch.no_grad()
def run(specs, n_cells=4, alphas=(0, 1, 2, 3, 4, 5), ws=(1.0, 1.5, 2.0, 3.0), out_name="ddim_anchors", device="cuda"):
    from ..traversal.precompute import _gather_class      # gather NTC anchors (imgs + CellDINO embs)
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    out = Path(ANALYSIS) / out_name; out.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg"); matplotlib.rcParams["pdf.fonttype"] = 42
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    metrics = {}
    for label, mc, ch, ckpt, gene in specs:
        cfg = DirConfig(grain="geneKO", target=gene, control="NTC", device=device)
        cfg.channel = ch; cfg.diffae_ckpt = ckpt
        modality = slugify(mc) if mc else "phase"
        if mc:                                                   # fluor: read the marker's cells once
            cfg.marker_channel = mc
            _cols = {"gene", "channel", "experiment", "well", "segmentation", "x_pheno", "y_pheno", "rank_type", "rank"}
            _all = pd.read_csv(cfg.fluor_csv, usecols=lambda c: c in _cols)
            cfg._fluor_rows = _all[(_all["channel"] == mc) & (_all["rank_type"] == "top")]
        diffae = load_diffae(cfg, dev); null = diffae.null_emb.detach()[None].to(dev); H = cfg.crop_size
        # NTC control cells: 1000 for direction, first n_cells as anchors
        ntc_imgs, ntc_embs = _gather_class(cfg, "NTC", 1000)
        mu_ctrl = ntc_embs.mean(0)
        x0 = normalize(ntc_imgs[:n_cells]); x0t = torch.as_tensor(x0, dtype=torch.float32, device=dev)
        z0 = torch.as_tensor(ntc_embs[:n_cells], dtype=torch.float32, device=dev)
        d_vec, gap, lr_w, lr_b = _direction(cfg, gene, modality, ntc_embs, mu_ctrl, _gather_class, dev)
        d = torch.as_tensor(d_vec, dtype=torch.float32, device=dev)[None]
        tgt = label
        from ..classifier.celldino_features import embed_crops
        xT_rand = torch.cat([torch.randn(1, 1, H, H, generator=torch.Generator(device=dev).manual_seed(1234 + c), device=dev)
                             for c in range(n_cells)], 0)
        for w in ws:                                             # sweep w: gather/direction reused, only xT+decode redo
            xT_inv = torch.cat([_ddim_guided(diffae, x0t[c:c + 1], z0[c:c + 1], null, w, cfg, inverse=True)
                                for c in range(n_cells)], 0)
            gen_inv = np.empty((n_cells, len(alphas), H, H), np.float32); gen_rand = np.empty_like(gen_inv)
            for c in range(n_cells):
                for ai, a in enumerate(alphas):
                    cond = z0[c:c + 1] + (a * gap) * d
                    gen_inv[c, ai] = _sample_guided(diffae, xT_inv[c:c + 1].clone(), cond, null, w, cfg).cpu().numpy()[0, 0]
                    gen_rand[c, ai] = _sample_guided(diffae, xT_rand[c:c + 1].clone(), cond, null, w, cfg).cpu().numpy()[0, 0]
            r_inv = float(np.mean([_pearson(gen_inv[c, 0], x0[c, 0]) for c in range(n_cells)]))
            r_rand = float(np.mean([_pearson(gen_rand[c, 0], x0[c, 0]) for c in range(n_cells)]))
            # morph strength: re-encode the inverted sweep, score with the direction's LR probe (logit α=max − α=0)
            gemb = embed_crops(gen_inv.reshape(-1, 1, H, H).astype(np.float32), cfg, cache_path=None)
            logits = (gemb @ lr_w + lr_b).reshape(n_cells, len(alphas))
            a0, amax = (alphas.index(0) if 0 in alphas else 0), int(np.argmax(alphas))
            morph_shift = float(np.mean(logits[:, amax] - logits[:, a0]))
            metrics[f"{tgt}_w{w:g}"] = {"w": w, "alpha0_pearson_inverted": round(r_inv, 3),
                                        "alpha0_pearson_random": round(r_rand, 3),
                                        "morph_logit_shift_a0_to_amax": round(morph_shift, 3), "gap": round(gap, 3)}
            print(f"[{tgt} w={w:g}] alpha0 Pearson inv={r_inv:.3f} rand={r_rand:.3f}  morph_shift={morph_shift:.2f}")
            ncols = 1 + len(alphas)
            fig, ax = plt.subplots(2 * n_cells, ncols, figsize=(1.5 * ncols, 1.5 * 2 * n_cells), squeeze=False)
            for c in range(n_cells):
                for row, gen, seed_lbl in [(2 * c, gen_rand, "random xT"), (2 * c + 1, gen_inv, "inverted xT")]:
                    ax[row, 0].imshow(x0[c, 0], cmap="gray", vmin=-1, vmax=1)
                    ax[row, 0].set_ylabel(f"cell{c}\n{seed_lbl}", fontsize=7)
                    for ai, a in enumerate(alphas):
                        axi = ax[row, ai + 1]; axi.imshow(gen[c, ai], cmap="gray", vmin=-1, vmax=1)
                        if a == 0:
                            axi.text(0.04, 0.96, f"r={_pearson(gen[c, ai], x0[c, 0]):.2f}", transform=axi.transAxes,
                                     fontsize=7, color="white", va="top",
                                     path_effects=[pe.withStroke(linewidth=1.5, foreground="black")])
                    for a_ax in ax[row]:
                        a_ax.set_xticks([]); a_ax.set_yticks([])
            for j, t in enumerate(["REAL"] + [f"α={a}" for a in alphas]):
                ax[0, j].set_title(t, fontsize=8)
            fig.suptitle(f"{tgt} w={w:g} — inverted vs random  "
                         f"(α=0 Pearson inv {r_inv:.2f} vs rand {r_rand:.2f}; morph {morph_shift:.1f})", fontsize=9)
            fig.tight_layout()
            fig.savefig(out / f"ddim_anchor_{slugify(tgt)}_w{w:g}.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    import json
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["fluor", "phase"], default="fluor")
    ap.add_argument("--n-cells", type=int, default=4)
    ap.add_argument("--ws", type=float, nargs="+", default=[1.0, 1.5, 2.0, 3.0], help="w values to sweep in ONE job")
    ap.add_argument("--submit", action="store_true")
    args = ap.parse_args()
    specs = FLUOR_SPECS if args.set == "fluor" else PHASE_SPECS
    out_name = f"ddim_anchors_{args.set}_wsweep"
    if args.submit:
        from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs
        submit_parallel_jobs(jobs_to_submit=[{"name": f"ddim_wsweep_{args.set}", "func": run,
            "kwargs": {"specs": specs, "n_cells": args.n_cells, "ws": tuple(args.ws), "out_name": out_name}}],
            experiment="diffae", slurm_params={"slurm_partition": "gpu", "gpus_per_node": 1,
            "cpus_per_task": 8, "mem_gb": 96, "timeout_min": 120,
            "slurm_constraint": "[a100_80|h100|h200|6000_blackwell]"}, log_dir="diffae",
            wait_for_completion=False)
    else:
        run(specs, n_cells=args.n_cells, ws=tuple(args.ws), out_name=out_name)


if __name__ == "__main__":
    main()
