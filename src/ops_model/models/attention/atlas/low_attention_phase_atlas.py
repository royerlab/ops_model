"""Render the lowest-attention phase cells across the v4 atlas as multi-page PDFs.

Per-perturbation organization: each gene (and NTC) gets its OWN page in the
PDF, with its bottom-K cells filling a 6×5 grid. This makes it visually
obvious which geneKO / complex each low-attention cell came from. NTC is
always included as the final page so KO behaviour can be compared against
the model's "no perturbation" baseline.

Three outputs (one PDF + companion CSV + per-page PNGs each):

  low_attention_geneko.pdf            top-N geneKOs (by max geneko_rank)
                                       + NTC, each gene's bottom-K cells
  low_attention_ebi.pdf               same, picking by ebi_rank
  low_attention_intersection_pNN.pdf  cells in the bottom-P% of BOTH heads
                                       per gene, ranked by combined
                                       normalized rank

"Lowest attention" = largest rank value within the head's tagged cells.
Per-gene threshold for the intersection PDF mirrors the Sweep-B logic from
``map_attention_expansion_v4.py``: threshold = max_rank × (1 − P/100).

Default budget: 20 geneKOs + 1 NTC = 21 pages × 30 cells/page ≈ 630 cells per
PDF. Tile size matches attention_atlas.py default crop (200 px = 65 µm at
0.325 µm/px). No NTC pairing, no fluor — phase only.

Always runs under SLURM — one job per PDF, in parallel, blocking until done.
Tile rendering pulls hundreds of zarr crops per PDF, so login-node runs are
slow; the script just submits the work.

Usage:
    python low_attention_phase_atlas.py
    python low_attention_phase_atlas.py --cells-per-gene 12 --top-perturbations 5
    python low_attention_phase_atlas.py --skip ebi intersection
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Optional, Sequence

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

# Reuse the segmentation-aware loading + rendering pipeline from
# attention_atlas.py (sibling file). This gives us the exact same
# cell-mask overlay (translucent blue outside the cell) and optional
# nuclear contour styling.
_ATLAS_DIR = Path(__file__).resolve().parent
if str(_ATLAS_DIR) not in sys.path:
    sys.path.insert(0, str(_ATLAS_DIR))
from attention_atlas import (  # noqa: E402  (sys.path mutation above)
    NUCLEAR_MASK_PREFERENCE_PHASE,
    PHASE_CHANNEL_CANDIDATES,
    StoreCache,
    _bbox_center_crop,
    _build_base_dataset,
    _load_nuc_mask,
    _render_cell,
)

logger = logging.getLogger(__name__)

PCA_DIR = Path(
    "/hpc/projects/icd.fast.ops/models/alex_lin_attention/v4/expansion_v1/per_experiment_v4_pca"
)
DEFAULT_OUTPUT_DIR = Path(
    "/home/gav.sturm/linked_folders/mydata/ops_mono/coding_exps/low_attention_atlas"
)
PHASE_CHANNEL_CANDIDATES = ("Phase2D", "Phase")
CROP_SIZE = 200
CELLS_PER_GENE = 30          # 6×5 grid fits one page per perturbation
TOP_PERTURBATIONS = 20       # + NTC always included → ~21 pages per PDF
INTERSECTION_P = 10          # bottom-P% threshold per head, per gene
GRID_COLS, GRID_ROWS = 6, 5

# attention_atlas's BaseDataset loads at a wider window so the display crop
# can be re-centered on the cell's segmentation bbox. 1.5× matches the
# attention_atlas.py default.
LOAD_PAD_FACTOR = 1.5


def load_global_obs(pca_dir: Path = PCA_DIR) -> pd.DataFrame:
    """Concat every per-experiment v4 PCA h5ad's obs (no X) into one frame."""
    paths = sorted(pca_dir.glob("*.h5ad"))
    if not paths:
        raise FileNotFoundError(f"no PCA h5ads in {pca_dir}")
    cols = ["experiment", "well", "well_flat", "segmentation_id",
            "x_position", "y_position",
            "perturbation", "gene_name", "sgRNA",
            "ebi_rank", "chad_rank", "geneko_rank",
            "ebi_rank_type", "chad_rank_type", "geneko_rank_type"]
    blocks = []
    for p in paths:
        a = ad.read_h5ad(p, backed="r")
        blocks.append(a.obs[cols].copy())
        a.file.close()
    df = pd.concat(blocks, ignore_index=True)
    logger.info("loaded obs: %d cells across %d experiments", len(df), len(paths))
    return df


# ---------------------------------------------------------------------------
# Selection — per-perturbation stratified bottom-K
# ---------------------------------------------------------------------------

def _filter_top_perturbations(sub: pd.DataFrame, score_col: str,
                                top_n: Optional[int]) -> pd.DataFrame:
    """Keep the top-N perturbations by max-score, always keeping NTC.

    Picking by `max` of the score within each perturbation favours genes
    whose worst cell is the most extreme — those are the most diagnostic
    failure modes of the classifier. Setting `top_n=None` keeps everything.
    """
    if top_n is None or top_n <= 0:
        return sub
    worst_per_pert = sub.groupby("perturbation", observed=True)[score_col].max().sort_values(ascending=False)
    worst_no_ntc = worst_per_pert[worst_per_pert.index.astype(str) != "NTC"]
    keep = set(worst_no_ntc.head(top_n).index)
    if (sub["perturbation"].astype(str) == "NTC").any():
        keep.add("NTC")
    return sub[sub["perturbation"].isin(keep)].copy()


def _sort_for_pages(sub: pd.DataFrame, score_col: str) -> pd.DataFrame:
    """Order rows so each perturbation's cells are contiguous (a single page
    or block of pages), real genes alphabetical, NTC last, and within a
    perturbation the lowest-attention cell (largest score) is first.
    """
    is_ntc = sub["perturbation"].astype(str).eq("NTC")
    sort_key = np.where(is_ntc, "z_NTC", "a_" + sub["perturbation"].astype(str))
    out = sub.assign(__sort_key=sort_key).sort_values(
        ["__sort_key", score_col], ascending=[True, False]
    )
    return out.drop(columns="__sort_key").reset_index(drop=True)


def select_low_per_perturbation(obs: pd.DataFrame, rank_col: str,
                                  cells_per_gene: int = CELLS_PER_GENE,
                                  top_perturbations: Optional[int] = TOP_PERTURBATIONS,
                                  ) -> pd.DataFrame:
    """For each perturbation: take `cells_per_gene` cells with the LARGEST
    rank value under `rank_col`. Then optionally restrict to the top-N
    perturbations by max rank. NTC is always kept (if present in `obs`).
    """
    sub = obs[obs[rank_col].notna()].copy()
    if sub.empty:
        return sub
    # The h5ad obs typed `perturbation` as Categorical with every registered
    # gene name. groupby on a Categorical yields a group for EVERY level
    # (most empty) — which then renders as a thousand blank PDF pages.
    # Coerce to string so groupby only sees the perturbations actually
    # present in this slice.
    sub["perturbation"] = sub["perturbation"].astype(str)
    sub = sub.sort_values(rank_col, ascending=False)
    sub = sub.groupby("perturbation", sort=False, observed=True).head(cells_per_gene)
    sub = _filter_top_perturbations(sub, rank_col, top_perturbations)
    return _sort_for_pages(sub, rank_col)


def select_intersection_per_perturbation(
    obs: pd.DataFrame, p: int,
    cells_per_gene: int = CELLS_PER_GENE,
    top_perturbations: Optional[int] = TOP_PERTURBATIONS,
) -> pd.DataFrame:
    """Per-perturbation bottom-`cells_per_gene` cells from the EBI ∩ geneKO
    bottom-P% intersection, ranked by combined normalized score. Mirrors
    ``_bottom_p_set_positions`` from map_attention_expansion_v4.py.
    """
    sub = obs[obs["ebi_rank"].notna() & obs["geneko_rank"].notna()].copy()
    if sub.empty:
        return sub
    # Drop Categorical to avoid blank-group artifacts (see same note in
    # select_low_per_perturbation).
    sub["perturbation"] = sub["perturbation"].astype(str)
    ebi_max = sub.groupby("perturbation", observed=True)["ebi_rank"].transform("max")
    geneko_max = sub.groupby("perturbation", observed=True)["geneko_rank"].transform("max")
    keep = (
        (sub["ebi_rank"] > ebi_max * (1.0 - p / 100.0))
        & (sub["geneko_rank"] > geneko_max * (1.0 - p / 100.0))
    )
    sub = sub.loc[keep].copy()
    if sub.empty:
        return sub
    sub["combined_low_score"] = (
        sub["ebi_rank"] / ebi_max.loc[sub.index]
        + sub["geneko_rank"] / geneko_max.loc[sub.index]
    )
    sub = sub.sort_values("combined_low_score", ascending=False)
    sub = sub.groupby("perturbation", sort=False, observed=True).head(cells_per_gene)
    sub = _filter_top_perturbations(sub, "combined_low_score", top_perturbations)
    return _sort_for_pages(sub, "combined_low_score")


# ---------------------------------------------------------------------------
# Rendering — one page per perturbation
# ---------------------------------------------------------------------------

def _cell_title(row: pd.Series, rank_col: str) -> str:
    """Compact tile title. For the intersection PDF (rank_col == combined…)
    show both head ranks; otherwise show just the head's rank value.
    """
    exp_short = str(row["experiment"]).split("_")[0]
    if rank_col == "combined_low_score":
        ebi = row.get("ebi_rank"); gk = row.get("geneko_rank")
        bits = []
        if pd.notna(ebi): bits.append(f"ebi={int(ebi)}")
        if pd.notna(gk):  bits.append(f"gk={int(gk)}")
        rank_line = "  ".join(bits) if bits else ""
    else:
        rv = row.get(rank_col)
        rank_line = f"rank={rv:.0f}" if pd.notna(rv) else "rank=?"
    return f"{exp_short}\n{rank_line}"


def _safe_filename(name: str) -> str:
    """Filesystem-safe perturbation name for PNG filenames."""
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", str(name))
    return s or "unknown"


def _resolve_phase_idx(store) -> Optional[int]:
    """Index of the phase channel inside a `store`'s channel axis, or None
    when the store lacks any of the candidate names.
    """
    if store is None:
        return None
    ch_names = list(store.channel_names)
    for cand in PHASE_CHANNEL_CANDIDATES:
        if cand in ch_names:
            return ch_names.index(cand)
    return None


def _load_page_crops(cells_in_page: list[dict], store_cache: StoreCache,
                       crop_size: int, load_pad_factor: float,
                       nuclear_overlay: str,
                       nuc_label_cache: dict,
                       ) -> list[tuple]:
    """Run one page's cells through attention_atlas's BaseDataset pipeline.

    Returns a list aligned with ``cells_in_page``: each entry is
    (phase_crop, cell_mask, nuc_mask) or (None, None, None) on load failure.
    """
    ds, input_indices = _build_base_dataset(
        cells_in_page, store_cache, crop_size, load_pad_factor=load_pad_factor,
    )
    out: list[tuple] = [(None, None, None)] * len(cells_in_page)
    if ds is None:
        return out

    for ds_idx, in_idx in enumerate(input_indices):
        try:
            batch = ds[ds_idx]
            data = batch["data"].numpy()       # (C, Y, X) at LOAD size
            mask = batch["mask"].numpy()[0].astype(bool)
            rec = ds.labels_df.iloc[ds_idx]

            nuc_mask = None
            if nuclear_overlay == "seg":
                try:
                    nuc_mask = _load_nuc_mask(
                        ds, rec["store_key"], rec["well"], rec["bbox"],
                        mask, NUCLEAR_MASK_PREFERENCE_PHASE,
                        label_cache=nuc_label_cache,
                    )
                except Exception:
                    nuc_mask = None

            # Slice everything down to the display crop_size, centered on
            # the cell-mask bbox so the cell stays middle-of-tile even
            # when load_pad_factor > 1.
            data_c, mask_c, nuc_c, _ = _bbox_center_crop(
                data, mask, crop_size, nuc_mask=nuc_mask,
            )

            store = store_cache.get(cells_in_page[in_idx]["experiment"])
            pidx = _resolve_phase_idx(store)
            if pidx is None or pidx >= data_c.shape[0]:
                out[in_idx] = (None, mask_c, nuc_c)
                continue
            phase_crop = data_c[pidx]
            out[in_idx] = (phase_crop, mask_c, nuc_c)
        except Exception as e:
            cell = cells_in_page[in_idx]
            logger.warning("cell load failed (exp=%s well=%s seg=%s): %s",
                            cell.get("experiment"), cell.get("well"),
                            cell.get("segmentation"), e)
    return out


def render_pdf(cells: pd.DataFrame, cache: StoreCache, out_pdf: Path,
                rank_col: str, crop_size: int = CROP_SIZE,
                grid_cols: int = GRID_COLS, grid_rows: int = GRID_ROWS,
                also_png: bool = True, png_dpi: int = 150,
                nuclear_overlay: str = "seg",
                load_pad_factor: float = 1.5) -> None:
    """Render `cells` grouped by perturbation. Each gene's cells fill its
    own page; cells are overlaid with the segmentation mask (translucent
    blue outside the cell) using ``attention_atlas._render_cell``. When
    ``nuclear_overlay='seg'`` a thin blue nuclear contour is also drawn.
    """
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    per_page = grid_cols * grid_rows

    pages = []  # list of (pert, page_i, n_pages_for_pert, chunk_df)
    for pert, df_g in cells.groupby("perturbation", sort=False, observed=True):
        n_pg = max(1, (len(df_g) + per_page - 1) // per_page)
        for pg in range(n_pg):
            chunk = df_g.iloc[pg * per_page:(pg + 1) * per_page]
            pages.append((str(pert), pg, n_pg, chunk))
    if not pages:
        logger.warning("[%s] no pages to render", out_pdf.name)
        return
    n_total = len(pages)
    n_perts = cells["perturbation"].nunique()
    logger.info("→ %s   (%d cells / %d perturbations / %d pages)",
                out_pdf.name, len(cells), n_perts, n_total)

    png_dir = out_pdf.parent / f"{out_pdf.stem}_pages" if also_png else None
    if png_dir is not None:
        png_dir.mkdir(parents=True, exist_ok=True)

    nuc_label_cache: dict = {}
    pad = max(2, len(str(n_total)))
    with PdfPages(out_pdf) as pdf:
        for global_pg, (pert, local_pg, n_pg_for_pert, chunk) in enumerate(pages, start=1):
            # Translate this page's rows into the cell-row dicts that
            # _build_base_dataset expects (matches attention_atlas
            # render_gene_page schema).
            cells_in_page = [
                {
                    "experiment": str(r["experiment"]),
                    "well": str(r["well"]),
                    "x_pheno": float(r["x_position"]),
                    "y_pheno": float(r["y_position"]),
                    "segmentation": r["segmentation_id"],
                    "gene": str(r["perturbation"]),
                    "kind": "phase",
                }
                for _, r in chunk.iterrows()
            ]
            crops = _load_page_crops(
                cells_in_page, cache, crop_size,
                load_pad_factor=load_pad_factor,
                nuclear_overlay=nuclear_overlay,
                nuc_label_cache=nuc_label_cache,
            )

            fig, axes = plt.subplots(
                grid_rows, grid_cols,
                figsize=(grid_cols * 2.4, grid_rows * 2.7),
            )
            axes = np.atleast_2d(axes).reshape(grid_rows, grid_cols)
            for i, ax in enumerate(axes.flat):
                if i >= len(chunk):
                    ax.axis("off")
                    continue
                row = chunk.iloc[i]
                phase_crop, mask, nuc_mask = crops[i]
                title = _cell_title(row, rank_col)
                _render_cell(ax, phase_crop, mask, title,
                              nuc_mask=nuc_mask)

            suffix = f"  •  page {local_pg + 1}/{n_pg_for_pert}" if n_pg_for_pert > 1 else ""
            header = (
                f"{pert}  —  {len(chunk)} cell{'s' if len(chunk) != 1 else ''}"
                f"{suffix}    ({global_pg}/{n_total})"
            )
            fig.suptitle(header, fontsize=12, fontweight="bold", y=0.997)
            fig.tight_layout(rect=(0, 0, 1, 0.978))
            pdf.savefig(fig, bbox_inches="tight", dpi=120)
            if png_dir is not None:
                stem = _safe_filename(pert)
                if n_pg_for_pert > 1:
                    stem = f"{stem}_{local_pg + 1:02d}"
                png_path = png_dir / f"{global_pg:0{pad}d}_{stem}.png"
                fig.savefig(png_path, dpi=png_dpi, bbox_inches="tight")
            plt.close(fig)
            logger.info("   page %d/%d  %s  (%d cells)",
                          global_pg, n_total, pert, len(chunk))


def render_one_pdf(pdf_tag: str, output_dir: Path,
                     cells_per_gene: int = CELLS_PER_GENE,
                     top_perturbations: Optional[int] = TOP_PERTURBATIONS,
                     intersection_p: int = INTERSECTION_P,
                     crop_size: int = CROP_SIZE,
                     grid_cols: int = GRID_COLS,
                     grid_rows: int = GRID_ROWS,
                     also_png: bool = True,
                     png_dpi: int = 150,
                     nuclear_overlay: str = "seg",
                     load_pad_factor: float = LOAD_PAD_FACTOR) -> dict:
    """End-to-end render of ONE PDF (geneko | ebi | intersection).

    Module-level so it can be pickled and dispatched by ``submit_parallel_jobs``.
    Returns a small dict with the result path + cell count (useful for the
    SLURM manifest aggregation).
    """
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(message)s", force=False)
    output_dir = Path(output_dir)
    obs = load_global_obs()
    cache = StoreCache()

    if pdf_tag == "geneko":
        sel = select_low_per_perturbation(
            obs, "geneko_rank",
            cells_per_gene=cells_per_gene, top_perturbations=top_perturbations,
        )
        rank_col = "geneko_rank"
        tag = "geneko"
    elif pdf_tag == "ebi":
        sel = select_low_per_perturbation(
            obs, "ebi_rank",
            cells_per_gene=cells_per_gene, top_perturbations=top_perturbations,
        )
        rank_col = "ebi_rank"
        tag = "ebi"
    elif pdf_tag == "intersection":
        sel = select_intersection_per_perturbation(
            obs, intersection_p,
            cells_per_gene=cells_per_gene, top_perturbations=top_perturbations,
        )
        rank_col = "combined_low_score"
        tag = f"intersection_p{intersection_p}"
    else:
        raise ValueError(f"unknown pdf_tag: {pdf_tag!r}")

    if sel.empty:
        logger.warning("[%s] no cells selected", pdf_tag)
        return {"pdf_tag": pdf_tag, "status": "no_cells"}

    pdf_path = output_dir / f"low_attention_{tag}.pdf"
    csv_path = output_dir / f"low_attention_{tag}_cells.csv"
    render_pdf(sel, cache, pdf_path, rank_col=rank_col,
                crop_size=crop_size, grid_cols=grid_cols, grid_rows=grid_rows,
                also_png=also_png, png_dpi=png_dpi,
                nuclear_overlay=nuclear_overlay,
                load_pad_factor=load_pad_factor)
    write_cells_csv(sel, csv_path)
    return {"pdf_tag": pdf_tag, "status": "done",
            "pdf": str(pdf_path), "csv": str(csv_path),
            "n_cells": int(len(sel)),
            "n_perturbations": int(sel["perturbation"].nunique())}


def write_cells_csv(cells: pd.DataFrame, out_csv: Path) -> None:
    """Companion CSV — same row set as the PDF, kept side-by-side for downstream."""
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    cols = ["experiment", "well", "well_flat", "x_position", "y_position",
            "perturbation", "gene_name", "sgRNA",
            "ebi_rank", "chad_rank", "geneko_rank"]
    keep = [c for c in cols if c in cells.columns]
    if "combined_low_score" in cells.columns:
        keep = keep + ["combined_low_score"]
    cells[keep].to_csv(out_csv, index=False)
    logger.info("   wrote %s (%d rows)", out_csv.name, len(cells))


def _ops_mono_root() -> Path:
    """Walk up from this file to the ``ops_mono`` repo root."""
    p = Path(__file__).resolve()
    for cand in [p] + list(p.parents):
        if cand.name == "ops_mono":
            return cand
    raise RuntimeError(
        f"could not find 'ops_mono' in any parent of {Path(__file__).resolve()}"
    )


def submit_slurm_jobs(args) -> None:
    """Submit one SLURM job per requested PDF. Waits for completion.

    Each job runs :func:`render_one_pdf` independently — selection + render
    happens inside each job, so they share no in-memory state. ~30 GB peak
    per job (loading the global obs frame across 77 experiments).
    """
    from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs

    pdf_tags = [t for t in ("geneko", "ebi", "intersection") if t not in args.skip]
    if not pdf_tags:
        logger.warning("nothing to submit — all PDFs skipped")
        return

    ops_mono = _ops_mono_root()
    atlas_dir = Path(__file__).resolve().parent  # contains attention_atlas.py
    common_kwargs = dict(
        output_dir=args.output_dir,
        cells_per_gene=args.cells_per_gene,
        top_perturbations=(args.top_perturbations if args.top_perturbations > 0 else None),
        intersection_p=args.intersection_p,
        crop_size=args.crop_size,
        grid_cols=args.grid_cols,
        grid_rows=args.grid_rows,
        also_png=not args.no_png,
        png_dpi=args.png_dpi,
        nuclear_overlay=args.nuclear_overlay,
        load_pad_factor=args.load_pad_factor,
    )
    jobs = [
        {
            "name": f"lowattn_{tag}",
            "func": render_one_pdf,
            "kwargs": {"pdf_tag": tag, **common_kwargs},
            "metadata": {"type": "low_attention_atlas", "pdf_tag": tag},
        }
        for tag in pdf_tags
    ]
    slurm_params = {
        "timeout_min": 120,
        "mem": "48GB",
        "cpus_per_task": 4,
        "slurm_partition": "cpu",
        "slurm_setup": [
            # Both paths are needed: ops_mono for ops_utils / ops_model
            # packages, atlas_dir so `import attention_atlas` and
            # `import low_attention_phase_atlas` resolve when submitit
            # unpickles the job on the worker (BEFORE our top-level
            # sys.path.insert in low_attention_phase_atlas runs).
            f"export PYTHONPATH={atlas_dir}:{ops_mono}:$PYTHONPATH",
            "export OMP_NUM_THREADS=1",
        ],
    }
    logger.info("SLURM: submitting %d job(s) (%s)", len(jobs), ", ".join(pdf_tags))
    submit_parallel_jobs(
        jobs_to_submit=jobs,
        experiment="low_attention_atlas",
        slurm_params=slurm_params,
        log_dir="low_attention_atlas",
        manifest_prefix="low_attention_atlas",
        wait_for_completion=True,
        verbose=True,
    )

    # Report what each job should have written (same path mapping
    # render_one_pdf uses). Print regardless of success so missing files
    # are visible too — a stat() on each shows status at a glance.
    out_dir = Path(args.output_dir)
    print("\nOutputs:")
    for tag in pdf_tags:
        file_tag = (
            f"intersection_p{args.intersection_p}" if tag == "intersection" else tag
        )
        pdf_path = out_dir / f"low_attention_{file_tag}.pdf"
        csv_path = out_dir / f"low_attention_{file_tag}_cells.csv"
        png_dir = out_dir / f"low_attention_{file_tag}_pages"
        status = "✓" if pdf_path.exists() else "✗ MISSING"
        print(f"  [{tag}] {status}")
        print(f"    PDF : {pdf_path}")
        print(f"    CSV : {csv_path}")
        if not args.no_png:
            n_pngs = len(list(png_dir.glob("*.png"))) if png_dir.exists() else 0
            print(f"    PNGs: {png_dir}  ({n_pngs} pages)")


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                    help=f"Default: {DEFAULT_OUTPUT_DIR} (auto-created).")
    p.add_argument("--cells-per-gene", type=int, default=CELLS_PER_GENE,
                    help=f"Bottom-K cells per perturbation (default {CELLS_PER_GENE}, "
                         "= one full 6×5 grid per gene).")
    p.add_argument("--top-perturbations", type=int, default=TOP_PERTURBATIONS,
                    help=f"Keep the top-N perturbations by max rank value "
                         f"(default {TOP_PERTURBATIONS}); NTC is always added. "
                         "Pass 0 or a negative number to keep ALL perturbations "
                         "(beware: produces 1000+ pages).")
    p.add_argument("--crop-size", type=int, default=CROP_SIZE,
                    help=f"Phase tile size in px (default {CROP_SIZE} = "
                         f"65 um at 0.325 um/px).")
    p.add_argument("--intersection-p", type=int, default=INTERSECTION_P,
                    help=f"Bottom-P%% threshold for intersection PDF "
                         f"(default {INTERSECTION_P}).")
    p.add_argument("--grid-cols", type=int, default=GRID_COLS)
    p.add_argument("--grid-rows", type=int, default=GRID_ROWS)
    p.add_argument("--nuclear-overlay", choices=("none", "seg"), default="seg",
                    help="Overlay a thin nuclear contour on each tile "
                         "(default 'seg'; pass 'none' to disable).")
    p.add_argument("--load-pad-factor", type=float, default=LOAD_PAD_FACTOR,
                    help=f"BaseDataset load size = crop_size × factor "
                         f"(default {LOAD_PAD_FACTOR}); enables bbox-center "
                         "re-cropping of each tile.")
    p.add_argument("--no-png", action="store_true",
                    help="Skip per-page PNG export. By default each page is "
                         "saved as PNG into <stem>_pages/ alongside the PDF, "
                         "named <NN>_<perturbation>.png.")
    p.add_argument("--png-dpi", type=int, default=150,
                    help="DPI for per-page PNGs (default 150).")
    p.add_argument("--skip", nargs="*",
                    choices=("geneko", "ebi", "intersection"), default=[],
                    help="Skip one or more PDFs.")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    submit_slurm_jobs(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
