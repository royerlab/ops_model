#!/usr/bin/env python3
"""Animate gene-level UMAP + PHATE colored by attention-model top-1 accuracy.

For every ``n_cells`` bin present in the cdino eval CSVs (default bins
[10, 20, 50, 100, 200, 500, 1000, 2000, 5000]), render the gene-level
embedding (UMAP and PHATE) colored by the model's top-1 accuracy at
that bin — one frame per bin — for both modalities (phase + fluor),
and combine the frames into GIFs.

The accuracy CSVs live next to one another in
``v3/attention_v3/cdino/`` — ``cdino_eval_phase_50.csv`` and
``cdino_eval_fluorescent_50.csv``. Each is keyed by ``gene_name`` with
rows for every ``n_cells`` bin. The embeddings come from the
``gene_embedding_pca_optimized.h5ad`` written by Phase 2 of the
pca_optimization pipeline (``obsm["X_umap"]`` / ``obsm["X_phate"]``).

Multiple embedding sources can be passed via ``--embedding-h5ad`` (one
entry per UMAP recipe — the "max" recipe lives next to a "gav" recipe
in different output trees because pca_optimization writes to the same
obsm key regardless of recipe). Each input produces its own quartet of
GIFs labeled by the recipe; the label is read from
``uns["umap"]["params"]["umap_type"]`` when not explicitly provided.

Per (embedding, recipe-label, modality) pair the script writes::

    {embedding}_{label}_{modality}_accuracy.gif

e.g. ``umap_max_phase_accuracy.gif``, ``phate_gav_fluor_accuracy.gif``.

Usage:

    # Default — one h5ad (the "max"-recipe path)
    uv run python ops_process/ops_analysis/napari/attention_accuracy_umap_animation.py

    # Two h5ads, explicit labels (preferred when uns doesn't record umap_type):
    uv run python ops_process/ops_analysis/napari/attention_accuracy_umap_animation.py \\
        --embedding-h5ad max=/path/to/max/gene_embedding_pca_optimized.h5ad \\
                          gav=/path/to/gav/gene_embedding_pca_optimized.h5ad

    # Custom range of bins (still capped to what's in the CSV):
    uv run python ops_process/ops_analysis/napari/attention_accuracy_umap_animation.py \\
        --min-bin 20 --max-bin 1000
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# Defaults mirror gene_best_marker_assignment.py / attention_atlas.py.
_PCA_DIR = Path(
    "/home/gav.sturm/linked_folders/icd.fast.ops/organelle_attribution/"
    "pca_optimized_v0.3/cell_dino/zscore_per_exp/paper_v1/with_cp/with_4i/"
    "all_livecell/fixed_80%/cosine"
)
DEFAULT_EMBED_H5AD = (
    _PCA_DIR / "second_pca_consensus" / "gene_embedding_pca_optimized.h5ad"
)
DEFAULT_OUT_DIR = (
    _PCA_DIR / "second_pca_consensus" / "plots" / "marker_overlay"
    / "attention_accuracy_animation"
)
DEFAULT_PHASE_CSV = Path(
    "/home/gav.sturm/linked_folders/icd.fast.ops/models/alex_lin_attention/"
    "v3/attention_v3/cdino/cdino_eval_phase_50.csv"
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _resolve_fluor_csv(phase_csv: Path) -> Path:
    """Sibling fluor CSV: replace first 'phase' → 'fluorescent' in filename.

    Matches ``attention_atlas._load_eval_accuracy``.
    """
    return phase_csv.with_name(phase_csv.name.replace("phase", "fluorescent", 1))


def _load_acc_matrix(csv_path: Path) -> pd.DataFrame:
    """Return DataFrame indexed by gene_name with columns = n_cells bins."""
    df = pd.read_csv(csv_path)
    key = "gene_name" if "gene_name" in df.columns else "class_name"
    sub = df[[key, "n_cells", "top1_acc"]].copy()
    sub[key] = sub[key].astype(str).str.strip()
    mat = sub.pivot_table(
        index=key, columns="n_cells", values="top1_acc", aggfunc="mean"
    )
    mat.index.name = "gene"
    mat.columns = [int(c) for c in mat.columns]
    return mat


def _load_embedding(
    embed_h5ad: Path, embedding: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (coords[N,2], gene_names[N], is_ntc[N]) for the requested
    embedding ('umap' or 'phate'). Pulls coords from
    ``obsm[X_{embedding}]`` written by pca_optimization Phase 2.
    """
    adata = ad.read_h5ad(embed_h5ad)
    obsm_key = f"X_{embedding}"
    if obsm_key not in adata.obsm:
        raise KeyError(
            f"{obsm_key!r} not in {embed_h5ad}.obsm "
            f"(have: {sorted(adata.obsm.keys())})"
        )
    gene_col = (
        "geneKO_name" if "geneKO_name" in adata.obs.columns else "perturbation"
    )
    genes = adata.obs[gene_col].astype(str).values
    coords = np.asarray(adata.obsm[obsm_key], dtype=np.float32)
    is_ntc = np.array([g.startswith("NTC") for g in genes])
    return coords, genes, is_ntc


def _infer_recipe_label(embed_h5ad: Path) -> str:
    """Read ``uns["umap"]["params"]["umap_type"]`` if present, else 'default'.

    pca_optimization stamps the recipe name into uns when it writes the
    h5ad, so the same on-disk file is self-describing — no need to pass
    a label on the command line if the user trusts the stamp.
    """
    try:
        adata = ad.read_h5ad(embed_h5ad, backed="r")
        label = (adata.uns.get("umap") or {}).get("params", {}).get("umap_type")
        try:
            adata.file.close()
        except Exception:
            pass
        if isinstance(label, str) and label.strip():
            return label.strip()
    except Exception as e:
        logger.warning(f"  could not infer label from {embed_h5ad}: {e!r}")
    return "default"


def _parse_embedding_spec(spec: str) -> Tuple[str, Path]:
    """Parse ``label=path`` or bare ``path`` (label inferred from uns)."""
    if "=" in spec:
        label, _, path_str = spec.partition("=")
        return label.strip() or "default", Path(path_str).expanduser()
    path = Path(spec).expanduser()
    return _infer_recipe_label(path), path


def _parse_phate_recipe(spec: str) -> Tuple[int, int]:
    """Parse ``knn,decay`` (e.g. '5,10') into a (knn, decay) tuple."""
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    if len(parts) != 2:
        raise ValueError(
            f"Expected --extra-phate as 'knn,decay' (e.g. '5,10'); got {spec!r}"
        )
    try:
        return int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise ValueError(
            f"--extra-phate values must be integers; got {spec!r}"
        ) from exc


def _parse_umap_recipe(spec: str) -> Tuple[int, float]:
    """Parse ``n_neighbors,min_dist`` (e.g. '10,0.25') into a tuple.

    Matches the legacy 'gav' UMAP recipe from pca_optimization.embeddings:
    ``UMAP(n_neighbors=min(10, n-1), min_dist=0.25, random_state=seed)``.
    """
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    if len(parts) != 2:
        raise ValueError(
            "Expected --extra-umap as 'n_neighbors,min_dist' "
            f"(e.g. '10,0.25'); got {spec!r}"
        )
    try:
        return int(parts[0]), float(parts[1])
    except ValueError as exc:
        raise ValueError(
            f"--extra-umap values must be int,float; got {spec!r}"
        ) from exc


def _load_pca_matrix(
    embed_h5ad: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pull (X_pca-or-X, gene_names, is_ntc) from an embedding h5ad.

    Shared by ``_compute_phate`` / ``_compute_umap`` so neither has to
    re-open the h5ad on its own.
    """
    adata = ad.read_h5ad(embed_h5ad)
    if "X_pca" in adata.obsm:
        X = np.asarray(adata.obsm["X_pca"], dtype=np.float32)
    else:
        X = np.asarray(adata.X, dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    gene_col = (
        "geneKO_name" if "geneKO_name" in adata.obs.columns else "perturbation"
    )
    genes = adata.obs[gene_col].astype(str).values
    is_ntc = np.array([g.startswith("NTC") for g in genes])
    return X, genes, is_ntc


def _compute_umap(
    embed_h5ad: Path,
    n_neighbors: int,
    min_dist: float,
    random_seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit a fresh UMAP from X_pca with caller-set n_neighbors / min_dist.

    Mirrors the legacy 'gav' recipe in pca_optimization.embeddings (umap-learn
    direct, no scanpy neighbor graph), so ``--extra-umap 10,0.25`` reproduces
    what ``--umap-type gav`` would have written to ``obsm["X_umap"]``.
    """
    from umap import UMAP

    X, genes, is_ntc = _load_pca_matrix(embed_h5ad)
    n_obs = X.shape[0]
    nn_eff = min(n_neighbors, max(n_obs - 1, 2))
    if nn_eff < 2:
        raise ValueError(
            f"Too few obs ({n_obs}) for UMAP n_neighbors={n_neighbors} "
            f"in {embed_h5ad}"
        )
    logger.info(
        f"  fitting UMAP (n_neighbors={nn_eff}, min_dist={min_dist}) on "
        f"{n_obs}x{X.shape[1]} matrix from {embed_h5ad.name}"
    )
    coords = UMAP(
        n_components=2,
        n_neighbors=nn_eff,
        min_dist=min_dist,
        random_state=random_seed,
    ).fit_transform(X)
    return np.asarray(coords, dtype=np.float32), genes, is_ntc


def _compute_phate(
    embed_h5ad: Path, knn: int, decay: int, random_seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit a fresh PHATE with custom params on the h5ad's X_pca.

    Mirrors the call signature used by pca_optimization.embeddings (t='auto',
    n_jobs=-1, n_components=2), but with caller-controlled knn + decay so
    the user can sweep recipes without re-running Phase 2.
    """
    import phate

    X, genes, is_ntc = _load_pca_matrix(embed_h5ad)
    n_obs = X.shape[0]
    knn_eff = min(knn, max(n_obs - 1, 2))
    if knn_eff < 2:
        raise ValueError(
            f"Too few obs ({n_obs}) for PHATE knn={knn} in {embed_h5ad}"
        )
    logger.info(
        f"  fitting PHATE (knn={knn_eff}, decay={decay}) on "
        f"{n_obs}x{X.shape[1]} matrix from {embed_h5ad.name}"
    )
    coords = phate.PHATE(
        n_components=2,
        knn=knn_eff,
        decay=decay,
        t="auto",
        n_jobs=-1,
        random_state=random_seed,
        verbose=0,
    ).fit_transform(X)
    return np.asarray(coords, dtype=np.float32), genes, is_ntc


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _frame_path(
    out_dir: Path, embedding: str, label: str, modality: str, n_cells: int
) -> Path:
    return (
        out_dir / "frames"
        / f"{embedding}_{label}_{modality}_n{n_cells:05d}.png"
    )


def render_frame(
    coords: np.ndarray,
    genes: np.ndarray,
    is_ntc: np.ndarray,
    acc_matrix: pd.DataFrame,
    n_cells: int,
    embedding: str,
    label: str,
    modality: str,
    out_path: Path,
    n_genes_total: int,
) -> None:
    """Save one PNG: scatter colored by top1_acc at this n_cells bin."""
    if n_cells not in acc_matrix.columns:
        raise KeyError(f"n_cells={n_cells} not in {modality} acc matrix")
    lookup = acc_matrix[n_cells].to_dict()
    acc = np.array([lookup.get(str(g), np.nan) for g in genes], dtype=np.float64)
    has_acc = ~is_ntc & ~np.isnan(acc)

    fig, ax = plt.subplots(figsize=(13, 11))
    # Grey background: genes with no accuracy entry at this bin.
    bg = ~is_ntc & ~has_acc
    if bg.any():
        ax.scatter(
            coords[bg, 0], coords[bg, 1],
            c="#d8d8d8", s=18, alpha=0.35,
            edgecolors="none", label=f"no entry  n={int(bg.sum())}",
        )
    # Colored layer: top1_acc on viridis [0, 1].
    sc = ax.scatter(
        coords[has_acc, 0], coords[has_acc, 1],
        c=acc[has_acc], cmap="viridis", vmin=0.0, vmax=1.0,
        s=55, alpha=0.92, edgecolors="white", linewidths=0.25,
    )
    if is_ntc.any():
        ax.scatter(
            coords[is_ntc, 0], coords[is_ntc, 1],
            c="#e08080", marker="X", s=110, alpha=0.45,
            edgecolors="#b05050", linewidths=0.3,
            label=f"NTC  n={int(is_ntc.sum())}", zorder=10,
        )

    mean_acc = float(np.nanmean(acc[has_acc])) if has_acc.any() else float("nan")
    n_with = int(has_acc.sum())
    cbar = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("top-1 accuracy", fontsize=12)

    ax.set_title(
        f"Attention-model top-1 accuracy on {embedding.upper()} "
        f"[{label}]  ({modality})  —  n_cells = {n_cells}\n"
        f"{n_with}/{n_genes_total} genes covered  ·  mean acc = {mean_acc:.1%}",
        fontsize=14,
    )
    ax.set_xlabel(f"{embedding.upper()} 1")
    ax.set_ylabel(f"{embedding.upper()} 2")
    if ax.get_legend_handles_labels()[1]:
        ax.legend(loc="upper right", fontsize=10, frameon=False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  saved {out_path.name}  (mean acc {mean_acc:.1%})")


# ---------------------------------------------------------------------------
# Animation
# ---------------------------------------------------------------------------


def _build_gif(frame_paths: Sequence[Path], out_gif: Path, fps: float = 1.5) -> None:
    """Combine PNG frames into a looping GIF.

    Uses imageio when available (faster, better palette); falls back to
    Pillow's ``save(..., save_all=True)`` otherwise.
    """
    out_gif.parent.mkdir(parents=True, exist_ok=True)
    try:
        import imageio.v3 as iio  # type: ignore

        frames = [iio.imread(p) for p in frame_paths]
        # imageio takes duration in ms per frame.
        iio.imwrite(out_gif, frames, duration=int(1000 / fps), loop=0)
        logger.info(f"Wrote {out_gif} ({len(frames)} frames @ {fps:g} fps)")
        return
    except Exception as e:
        logger.warning(f"imageio path failed ({e!r}); falling back to PIL")
    from PIL import Image

    pil_frames = [Image.open(p).convert("RGBA") for p in frame_paths]
    pil_frames[0].save(
        out_gif,
        save_all=True,
        append_images=pil_frames[1:],
        duration=int(1000 / fps),
        loop=0,
        optimize=False,
        disposal=2,
    )
    logger.info(f"Wrote {out_gif} ({len(pil_frames)} frames, PIL fallback)")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _select_bins(
    available: Iterable[int],
    min_bin: int,
    max_bin: int,
    bins_override: Optional[Sequence[int]],
) -> List[int]:
    """Pick which n_cells bins to render."""
    available = sorted({int(b) for b in available})
    if bins_override:
        chosen = [b for b in sorted(set(bins_override)) if b in available]
    else:
        chosen = [b for b in available if min_bin <= b <= max_bin]
    if not chosen:
        raise ValueError(
            f"No usable n_cells bins (have {available}, requested "
            f"[{min_bin}, {max_bin}] / override {bins_override})"
        )
    return chosen


def run(
    embedding_sources: Sequence[Tuple[str, Path]],
    phase_csv: Path,
    fluor_csv: Path,
    out_dir: Path,
    embeddings: Sequence[str] = ("umap", "phate"),
    min_bin: int = 10,
    max_bin: int = 5000,
    bins_override: Optional[Sequence[int]] = None,
    fps: float = 1.5,
    extra_phate_recipes: Sequence[Tuple[int, int]] = (),
    extra_umap_recipes: Sequence[Tuple[int, float]] = (),
    embedding_seed: int = 42,
) -> None:
    logger.info(f"Phase CSV      : {phase_csv}")
    logger.info(f"Fluor CSV      : {fluor_csv}")
    logger.info(f"Output dir     : {out_dir}")
    for label, path in embedding_sources:
        logger.info(f"Source [{label}] : {path}")

    phase_mat = _load_acc_matrix(phase_csv)
    fluor_mat = _load_acc_matrix(fluor_csv)
    acc_matrices = {"phase": phase_mat, "fluor": fluor_mat}
    logger.info(
        f"Loaded accuracy matrices: phase {phase_mat.shape}  fluor {fluor_mat.shape}"
    )

    available_bins = sorted(set(phase_mat.columns) & set(fluor_mat.columns))
    bins = _select_bins(available_bins, min_bin, max_bin, bins_override)
    logger.info(f"Rendering bins: {bins}")

    # De-duplicate labels so two sources don't clobber each other's outputs.
    seen_labels: dict = {}
    for label, embed_h5ad in embedding_sources:
        # Disambiguate duplicate labels with _2, _3, ... suffixes.
        base = label
        i = 2
        while label in seen_labels:
            label = f"{base}_{i}"
            i += 1
        seen_labels[label] = embed_h5ad

        def _render_set(
            embedding_tag: str,
            coords: np.ndarray,
            genes: np.ndarray,
            is_ntc: np.ndarray,
            full_label: str,
        ) -> None:
            n_genes_total = int((~is_ntc).sum())
            logger.info(
                f"[{full_label}/{embedding_tag}] {len(genes)} obs "
                f"({n_genes_total} non-NTC, {int(is_ntc.sum())} NTC)"
            )
            for modality, acc_mat in acc_matrices.items():
                frame_paths: List[Path] = []
                for n_cells in bins:
                    if n_cells not in acc_mat.columns:
                        logger.warning(
                            f"  skip {full_label}/{embedding_tag}/{modality}/"
                            f"n={n_cells}: missing from accuracy matrix"
                        )
                        continue
                    fp = _frame_path(
                        out_dir, embedding_tag, full_label, modality, n_cells
                    )
                    render_frame(
                        coords, genes, is_ntc, acc_mat, n_cells,
                        embedding_tag, full_label, modality, fp, n_genes_total,
                    )
                    frame_paths.append(fp)
                if len(frame_paths) < 2:
                    logger.warning(
                        f"  not enough frames for {full_label}/"
                        f"{embedding_tag}/{modality} GIF "
                        f"({len(frame_paths)} frame(s)); skipping GIF"
                    )
                    continue
                gif_path = (
                    out_dir
                    / f"{embedding_tag}_{full_label}_{modality}_accuracy.gif"
                )
                _build_gif(frame_paths, gif_path, fps=fps)

        # Stock embeddings (read coords directly out of obsm).
        for embedding in embeddings:
            try:
                coords, genes, is_ntc = _load_embedding(embed_h5ad, embedding)
            except KeyError as e:
                logger.warning(f"[{label}/{embedding}] skipping — {e}")
                continue
            _render_set(embedding, coords, genes, is_ntc, label)

        # Extra UMAP recipes (fit fresh from X_pca with caller-set
        # n_neighbors / min_dist — mirrors the 'gav' recipe).
        for nn, md in extra_umap_recipes:
            md_tag = f"{md:.2f}".rstrip("0").rstrip(".") or "0"
            recipe_label = f"{label}-umap-nn{nn}-md{md_tag}"
            try:
                coords, genes, is_ntc = _compute_umap(
                    embed_h5ad,
                    n_neighbors=nn,
                    min_dist=md,
                    random_seed=embedding_seed,
                )
            except Exception as e:
                logger.warning(
                    f"[{recipe_label}] UMAP compute failed: {e!r}; skipping"
                )
                continue
            _render_set("umap", coords, genes, is_ntc, recipe_label)

        # Extra PHATE recipes (fit fresh from X_pca with caller-set knn/decay).
        for knn, decay in extra_phate_recipes:
            recipe_label = f"{label}-phate-knn{knn}-dec{decay}"
            try:
                coords, genes, is_ntc = _compute_phate(
                    embed_h5ad, knn=knn, decay=decay, random_seed=embedding_seed
                )
            except Exception as e:
                logger.warning(
                    f"[{recipe_label}] PHATE compute failed: {e!r}; skipping"
                )
                continue
            _render_set("phate", coords, genes, is_ntc, recipe_label)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--embedding-h5ad", nargs="+", default=[str(DEFAULT_EMBED_H5AD)],
                    help="One or more gene_embedding_pca_optimized.h5ad files "
                         "(each has X_umap + X_phate). Pass either a bare "
                         "path (label inferred from uns['umap']['params']"
                         "['umap_type'], else 'default') or 'label=path' for "
                         "an explicit label, e.g.: "
                         "--embedding-h5ad max=/p/max.h5ad gav=/p/gav.h5ad")
    ap.add_argument("--phase-csv", type=Path, default=DEFAULT_PHASE_CSV,
                    help="cdino_eval_phase_50.csv. Fluor sibling is "
                         "auto-discovered ('phase' → 'fluorescent').")
    ap.add_argument("--fluor-csv", type=Path, default=None,
                    help="Override fluor CSV path (default: sibling of "
                         "--phase-csv with 'phase' → 'fluorescent').")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                    help="Output directory for PNG frames + GIFs.")
    ap.add_argument("--embeddings", nargs="+", default=["umap", "phate"],
                    choices=["umap", "phate"],
                    help="Which precomputed embeddings to animate "
                         "(default: both).")
    ap.add_argument("--extra-umap", action="append", default=[],
                    metavar="N_NEIGHBORS,MIN_DIST",
                    help="Additional UMAP recipe to fit fresh from X_pca on "
                         "each --embedding-h5ad input. Format: "
                         "'n_neighbors,min_dist' (e.g. --extra-umap 10,0.25 "
                         "reproduces the legacy 'gav' UMAP). Repeatable. "
                         "Outputs are labeled '<source>-umap-nn{N}-md{D}'.")
    ap.add_argument("--extra-phate", action="append", default=[],
                    metavar="KNN,DECAY",
                    help="Additional PHATE recipe to fit fresh from X_pca on "
                         "each --embedding-h5ad input. Format: 'knn,decay' "
                         "(e.g. --extra-phate 5,10). Repeatable. Outputs are "
                         "labeled '<source>-phate-knn{K}-dec{D}'.")
    ap.add_argument("--embedding-seed", "--phate-seed", type=int, default=42,
                    dest="embedding_seed",
                    help="random_state for fresh --extra-umap / --extra-phate "
                         "fits (default 42). --phase-seed kept as an alias.")
    ap.add_argument("--min-bin", type=int, default=10,
                    help="Smallest n_cells bin to include (default 10).")
    ap.add_argument("--max-bin", type=int, default=5000,
                    help="Largest n_cells bin to include (default 5000).")
    ap.add_argument("--bins", type=int, nargs="+", default=None,
                    help="Explicit list of n_cells bins (overrides "
                         "--min-bin/--max-bin); intersected with CSV.")
    ap.add_argument("--fps", type=float, default=1.5,
                    help="GIF frame rate (default 1.5).")
    args = ap.parse_args()

    embedding_sources = [_parse_embedding_spec(s) for s in args.embedding_h5ad]
    extra_umap_recipes = [_parse_umap_recipe(s) for s in args.extra_umap]
    extra_phate_recipes = [_parse_phate_recipe(s) for s in args.extra_phate]

    fluor_csv = args.fluor_csv or _resolve_fluor_csv(args.phase_csv)
    run(
        embedding_sources=embedding_sources,
        phase_csv=args.phase_csv,
        fluor_csv=fluor_csv,
        out_dir=args.out_dir,
        embeddings=tuple(args.embeddings),
        min_bin=args.min_bin,
        max_bin=args.max_bin,
        bins_override=args.bins,
        fps=args.fps,
        extra_umap_recipes=extra_umap_recipes,
        extra_phate_recipes=extra_phate_recipes,
        embedding_seed=args.embedding_seed,
    )


if __name__ == "__main__":
    main()
