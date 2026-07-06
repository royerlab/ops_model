"""Batch DiffEx: strips + NTC→KO GIF for the top-ranked geneKOs and EBI complexes.

Reads the k10-ranked CSVs from the attention-selection page (top-N by rank_by_K10_mAP),
submits one GPU job per target. Each job: run_directions at w=5 only → per-cell strips +
scores, then a GIF for the auto-picked best-Δscore cell.

    python -m ops_model.models.attention.diffex.directions.batch \
        --genes-csv <k10_ranked_all_geneKOs.csv> --complex-csv <k10_ranked_all_complexes.csv> \
        --n-genes 50 --n-complex 20
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs

from ..classifier.config import DEFAULT_OUT_ROOT, slugify
from .config import DirConfig
from .make_gifs import make_all_gifs, make_gif, render_all_review
from .run import run_directions


# ≤10-char header labels so complex names don't overflow the grid tiles.
COMPLEX_ABBR = {
    "UTP-B complex": "UTP-B",
    "mTORC1 complex": "mTORC1",
    "Chaperonin-containing T-complex": "CCT",
    "Box C/D snoRNA-Guided RNP methyltransferase complex, FBLL1 variant": "Box C/D",
    "COPI vesicle coat complex, COPG1-COPZ1 variant": "COPI",
    "SF3B complex": "SF3B",
    "COP9 signalosome variant 1": "COP9",
    "Nucleolar exosome complex, EXOSC10 variant": "Exosome",
    "LSM2-8 complex": "LSM2-8",
    "DNA polymerase alpha:primase complex": "Pol α-prim",
    "40S cytosolic small ribosomal subunit": "40S ribo",
    "19S proteasome regulatory complex": "19S prot",
    "DNA polymerase epsilon complex": "DNA Pol ε",
    "DNA-directed RNA polymerase III complex, POLR3G variant": "Pol III",
    "DNA-directed RNA polymerase II complex": "Pol II",
    "Core mediator complex": "Core mediator",
    "Sm complex": "Sm core",
    "Eukaryotic translation initiation factor 3 complex": "eIF3",
    "ESCRT-III complex": "ESCRT-III",
    "60S cytosolic large ribosomal subunit": "60S ribo",
    "Actin-related protein 2/3 complex, ARPC1A-ACTR3B-ARPC5 variant": "Arp2/3",
}


def _short(name: str, n: int = 10) -> str:
    """≤n-char GIF header label for a complex name (genes pass through unchanged)."""
    if name in COMPLEX_ABBR:
        return COMPLEX_ABBR[name]
    s = name
    for suf in (" complex", " subunit", " variant"):
        s = s.replace(suf, "")
    s = s.strip().strip(",")
    return s if len(s) <= n else s[: n - 1] + "…"


def run_target(grain: str, target: str, label: str, w: float = 5.0) -> dict:
    cfg = DirConfig(grain=grain, target=target, device="cuda")
    cfg.guidance_scales = (w,)                      # w=5 only (batch)
    out = f"{DEFAULT_OUT_ROOT}/directions/phase/{grain}/{slugify(target)}"
    run_directions(cfg, out)
    sc = np.load(f"{out}/scores_w{w:g}.npy")        # (n_cells, n_alphas)
    delta = sc[:, -1] - sc[:, 0]
    best = int(np.argmax(delta))                    # cell that moves most toward the phenotype
    make_gif(grain, target, best, w, label)
    return {"target": target, "best_cell": best, "delta": float(delta[best])}


def all_gifs_target(grain: str, target: str, label: str, w: float = 5.0) -> list:
    """Render GIFs for every traversed cell of one target (strips must already exist)."""
    return make_all_gifs(grain, target, label, w=w)


def review_all_target(grain: str, target: str, label: str, w: float = 5.0) -> list:
    """Both styles (3-way axis + 2-way half) GIF + panel PNG for every traversed cell."""
    return render_all_review(grain, target, label, w=w)


# cells×α grid α-levels (each a full −max→+max sweep at w). See DiffEx defaults: w=2, α 2–4;
# ±5 included for the most subtle phenotypes where extreme α still adds signal.
_ALPHA_LEVELS = {
    "a2": [-2, -1.6, -1.2, -0.8, -0.4, 0, 0.4, 0.8, 1.2, 1.6, 2],
    "a3": [-3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3],
    "a4": [-4, -3.2, -2.4, -1.6, -0.8, 0, 0.8, 1.6, 2.4, 3.2, 4],
    "a5": [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
}


def marker_grid(marker_channel: str = None, channel: str = None, target: str = None,
                ckpt: str = None, label: str = None, cells=(0, 1, 2), w: float = 2.0,
                grain: str = "geneKO", control: str = None, device: str = "cuda") -> str:
    """cells×α grid for one (marker, target): render every α-level (sharing one
    gather+decoder) then composite a labeled rows=cells × cols=α grid. Returns the
    grid gif path under directions/_grids/.

    marker_channel=None → phase mode (grain parquet + Phase2D crops; pass ckpt=phase_v1).
    grain='complex' → EBI complexes. control=None → NTC-anchored; set control to another
    class for an A→B traversal (α=0 shows the control/anchor class, +α → target)."""
    from .grid import make_labeled_grid
    from .make_gifs import _pair_slug
    label = label or target
    for ak, al in _ALPHA_LEVELS.items():
        render_all_review(grain, target, label, w=w, cells=list(cells), device=device,
                          ckpt=ckpt, tag=f"_{ak}", marker_channel=marker_channel,
                          channel=channel, alphas=al, control=control)
    modality = slugify(marker_channel) if marker_channel else "phase"
    slug = _pair_slug(target, control)
    sd = f"{DEFAULT_OUT_ROOT}/directions/{modality}/{grain}/{slug}/strips"
    grid = [[f"{sd}/{slug}_w{w:g}_cell{c}_{ak}_axis.gif" for ak in _ALPHA_LEVELS] for c in cells]
    prefix = "fluor_" if marker_channel else ""
    out = f"{DEFAULT_OUT_ROOT}/directions/_grids/{prefix}{modality}_{slug}_cellsxalpha.gif"
    col_labels = [f"α=±{ak[1:]}" for ak in _ALPHA_LEVELS]  # stays in sync with _ALPHA_LEVELS
    anchor = f"{control} → " if control and control != "NTC" else ""
    make_labeled_grid(grid, out, row_labels=[f"cell {c}" for c in cells],
                      col_labels=col_labels,
                      title=f"{marker_channel or 'Phase'} — {anchor}{target} (w={w:g})", tile_w=280)
    return out


def build_targets(genes_csv: str, complex_csv: str, n_genes: int, n_complex: int):
    g = pd.read_csv(genes_csv).sort_values("rank_by_K10_mAP").head(n_genes)
    c = pd.read_csv(complex_csv).sort_values("rank_by_K10_mAP").head(n_complex)
    jobs = [("geneKO", t, t) for t in g["geneKO"].tolist()]
    jobs += [("complex", t, _short(t)) for t in c["complex_name"].tolist()]
    return jobs


def main():
    ap = argparse.ArgumentParser(description="Batch DiffEx strips+GIFs for top-ranked targets")
    ap.add_argument("--genes-csv", required=True)
    ap.add_argument("--complex-csv", required=True)
    ap.add_argument("--n-genes", type=int, default=50)
    ap.add_argument("--n-complex", type=int, default=20)
    ap.add_argument("--w", type=float, default=5.0)
    ap.add_argument("--gifs-only", action="store_true",
                    help="strips already exist: only render GIFs for all traversed cells")
    ap.add_argument("--review", action="store_true",
                    help="render both 3-way axis + 2-way half GIF+panel for all cells")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    targets = build_targets(args.genes_csv, args.complex_csv, args.n_genes, args.n_complex)
    mode = "review, all cells, 3-way+2-way" if args.review else ("gifs-only, all cells" if args.gifs_only else "full")
    print(f"{len(targets)} targets: {args.n_genes} geneKO + {args.n_complex} complex  [{mode}]")
    func = review_all_target if args.review else (all_gifs_target if args.gifs_only else run_target)
    jobs = [{
        "name": f"dx_{grain}_{slugify(target)[:24]}",
        "func": func,
        "kwargs": {"grain": grain, "target": target, "label": label, "w": args.w},
        "metadata": {"stage": "batch_review" if args.review else ("batch_gifs" if args.gifs_only else "batch_directions"),
                     "grain": grain, "target": target},
    } for grain, target, label in targets]

    submit_parallel_jobs(
        jobs_to_submit=jobs, experiment="diffex_batch",
        slurm_params={"slurm_partition": "gpu", "gpus_per_node": 1, "cpus_per_task": 8,
                      "mem_gb": 64, "timeout_min": 90,
                      "slurm_constraint": "[a100_80|h100|h200|6000_blackwell]",
                      "slurm_setup": ["export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"]},
        log_dir="diffex_batch", dry_run=args.dry_run, wait_for_completion=False,
    )


if __name__ == "__main__":
    main()
