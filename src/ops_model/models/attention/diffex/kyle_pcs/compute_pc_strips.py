#!/usr/bin/env python3
"""Compute PCA strip artifacts from cDINO embeddings and zarr image stores.

For each of N principal components, selects representative cells spanning
low-to-high along that axis and extracts Phase2D crop images. Outputs a
directory of artifacts consumed by build_static_explorer.py.

Outputs (in --output-dir):
  representatives.json   — selected cells with PC, bin, gene, score, position
  crops_png/             — 96×96 Phase2D crops named pc{NNN}_bin{NN}_row{N}.png
  gene_names.json        — ordered list of gene names
  gene_pc_scores.npy     — (n_genes, n_pcs) mean PC score per gene

Data format:
  --emb-dir: per-gene .pt files.
    Each .pt: {embeddings: tensor[N,D], cell_metadata}
    cell_metadata: {experiment, well, x_pheno, y_pheno, segmentation_id}
    (each as list-of-lists, one inner list per FOV group)

  --zarr-base: experiment dirs, each with:
    3-assembly/phenotyping_v3.zarr/{row}/{col}/0/0  (zarr v3 array)
    Shape [1, C, 1, Y, X]. Phase2D = channel 0 by default.

Examples:
  # Fit PCA from scratch and save model
  python3 compute_pc_strips.py \\
      --emb-dir /path/to/embeddings \\
      --zarr-base /path/to/zarrs \\
      --output-dir /path/to/artifacts \\
      --n-components 97 \\
      --save-model /path/to/pca_model

  # Reuse a saved PCA model (skip 40-min fit)
  python3 compute_pc_strips.py \\
      --emb-dir /path/to/embeddings \\
      --zarr-base /path/to/zarrs \\
      --output-dir /path/to/artifacts \\
      --load-model /path/to/pca_model
"""

import argparse
import heapq
import json
import os
import time

import numpy as np
import torch

# ── Zarr v3 compatibility patch ──
try:
    import zarr
    from zarr.core.metadata.v3 import ArrayV3Metadata
    _orig_from_dict = ArrayV3Metadata.from_dict.__func__
    @classmethod
    def _patched_from_dict(cls, data):
        if isinstance(data, dict):
            data.pop("storage_transformers", None)
        return _orig_from_dict(cls, data)
    ArrayV3Metadata.from_dict = _patched_from_dict
except Exception:
    pass


def list_gene_files(emb_dir):
    return sorted([
        f for f in os.listdir(emb_dir)
        if f.endswith(".pt") and f != "metadata.pt"
    ])


def load_gene_embeddings(path):
    d = torch.load(path, weights_only=False)
    return d["embeddings"].numpy()


def load_gene_metadata(path):
    d = torch.load(path, weights_only=False)
    cm = d["cell_metadata"]
    n = d["embeddings"].shape[0]
    flat = {k: [] for k in ("experiment", "well", "x_pheno", "y_pheno")}
    for g in range(len(cm["experiment"])):
        for k in flat:
            flat[k].extend(cm[k][g])
    assert len(flat["experiment"]) == n
    return flat


# ── PCA ──

def load_or_fit_pca(emb_dir, n_components, batch_size, save_model=None, load_model=None):
    from sklearn.decomposition import IncrementalPCA

    gene_files = list_gene_files(emb_dir)

    if load_model:
        print(f"  Loading PCA model from {load_model}")
        components = np.load(os.path.join(load_model, "components.npy"))
        mean = np.load(os.path.join(load_model, "mean.npy"))
        ev = np.load(os.path.join(load_model, "explained_variance.npy"))
        n_components = components.shape[0]
    else:
        pca = IncrementalPCA(n_components=n_components)
        print(f"  PCA fit: {len(gene_files)} genes, n_components={n_components}")
        total = 0
        t0 = time.time()
        for gi, gf in enumerate(gene_files):
            emb = load_gene_embeddings(os.path.join(emb_dir, gf))
            for start in range(0, emb.shape[0], batch_size):
                batch = emb[start:start + batch_size]
                if batch.shape[0] < n_components:
                    continue
                pca.partial_fit(batch)
            total += emb.shape[0]
            if (gi + 1) % 200 == 0:
                print(f"    [{gi+1}/{len(gene_files)}] {total:,} cells, {time.time()-t0:.0f}s")
        print(f"    Fit complete: {total:,} cells, {time.time()-t0:.0f}s")
        print(f"    Explained variance: {pca.explained_variance_ratio_.sum()*100:.1f}%")
        components = pca.components_
        mean = pca.mean_
        ev = pca.explained_variance_ratio_

        if save_model:
            os.makedirs(save_model, exist_ok=True)
            np.save(os.path.join(save_model, "components.npy"), components)
            np.save(os.path.join(save_model, "mean.npy"), mean)
            np.save(os.path.join(save_model, "explained_variance.npy"), ev)
            print(f"    Saved PCA model to {save_model}")

    def scorer(embeddings):
        return (embeddings - mean) @ components.T

    return scorer, n_components, ev.tolist()


# ── Representative selection ──

def select_representatives(emb_dir, scorer, n_strips, n_bins, n_rows, batch_size):
    gene_files = list_gene_files(emb_dir)

    print("  Pass 1/3: counting cells...")
    total_cells = 0
    for gf in gene_files:
        total_cells += load_gene_embeddings(os.path.join(emb_dir, gf)).shape[0]
    print(f"    {total_cells:,} cells across {len(gene_files)} genes")

    print("  Pass 2/3: computing percentile boundaries (200K subsample)...")
    rng = np.random.RandomState(42)
    subsample_scores = []
    for gf in gene_files:
        emb = load_gene_embeddings(os.path.join(emb_dir, gf))
        mask = rng.random(emb.shape[0]) < min(1.0, 200_000 / max(total_cells, 1))
        if mask.sum() > 0:
            subsample_scores.append(scorer(emb[mask]))
    subsample_scores = np.concatenate(subsample_scores, axis=0)
    print(f"    Subsample: {subsample_scores.shape[0]:,} cells")

    percentiles = np.linspace(0, 100, n_bins)
    bin_targets = np.zeros((n_strips, n_bins))
    for s in range(n_strips):
        bin_targets[s] = np.percentile(subsample_scores[:, s], percentiles)
    del subsample_scores

    print(f"  Pass 3/3: selecting representatives ({n_strips} x {n_bins} x {n_rows})...")
    heaps = [[[] for _ in range(n_bins)] for _ in range(n_strips)]
    gene_names = []
    gene_mean_scores = []
    t0 = time.time()

    for gi, gf in enumerate(gene_files):
        gene_name = gf.replace(".pt", "")
        emb = load_gene_embeddings(os.path.join(emb_dir, gf))
        gene_names.append(gene_name)
        gene_score_sum = np.zeros(n_strips)
        gene_score_count = 0

        for start in range(0, emb.shape[0], batch_size):
            batch = emb[start:start + batch_size]
            scores = scorer(batch)
            gene_score_sum += scores.sum(axis=0)
            gene_score_count += scores.shape[0]

            for s in range(n_strips):
                s_scores = scores[:, s]
                dists_all = np.abs(s_scores[:, None] - bin_targets[s][None, :])
                nearest_bin = np.argmin(dists_all, axis=1)
                for bi in range(n_bins):
                    mask = (nearest_bin == bi)
                    if not mask.any():
                        continue
                    indices = np.where(mask)[0]
                    dists = dists_all[indices, bi]
                    k = min(n_rows, len(indices))
                    if k >= len(indices):
                        top_k = range(len(indices))
                    else:
                        top_k = np.argpartition(dists, k)[:k]
                    for ti in top_k:
                        idx = int(indices[ti])
                        d = float(dists[ti])
                        local_i = start + idx
                        entry = (-d, gene_name, local_i, float(scores[idx, s]))
                        heap = heaps[s][bi]
                        if len(heap) < n_rows:
                            heapq.heappush(heap, entry)
                        elif -d > heap[0][0]:
                            heapq.heapreplace(heap, entry)

        gene_mean_scores.append(gene_score_sum / max(gene_score_count, 1))

        if (gi + 1) % 200 == 0:
            print(f"    [{gi+1}/{len(gene_files)}] {time.time()-t0:.0f}s")

    reps = []
    for s in range(n_strips):
        for bi in range(n_bins):
            entries = sorted(heaps[s][bi], key=lambda x: x[3])
            for row, (neg_d, gene, local_i, score) in enumerate(entries):
                reps.append((s, bi, row, gene, local_i, score))

    print(f"    Selected {len(reps)} representatives in {time.time()-t0:.0f}s")
    return reps, total_cells, gene_names, np.array(gene_mean_scores)


# ── Crop extraction ──

def extract_crops(emb_dir, zarr_base, reps, crop_size, phase_channel):
    print(f"  Loading metadata for {len(reps)} representatives...")

    needed = {}
    for s, bi, row, gene, local_i, score in reps:
        needed.setdefault(gene, set()).add(local_i)

    cell_info = {}
    for gene, indices in needed.items():
        meta = load_gene_metadata(os.path.join(emb_dir, gene + ".pt"))
        for idx in indices:
            cell_info[(gene, idx)] = {
                "experiment": meta["experiment"][idx],
                "well": meta["well"][idx],
                "x": meta["x_pheno"][idx],
                "y": meta["y_pheno"][idx],
            }

    # Deduplicate by spatial position within each strip
    seen_per_strip = {}
    deduped_reps = []
    for r in reps:
        s, bi, row, gene, local_i, score = r
        info = cell_info[(gene, local_i)]
        pos_key = (info["experiment"], info["well"],
                   int(round(info["x"])), int(round(info["y"])))
        seen = seen_per_strip.setdefault(s, set())
        if pos_key in seen:
            continue
        seen.add(pos_key)
        deduped_reps.append(r)
    if len(deduped_reps) < len(reps):
        print(f"    Removed {len(reps) - len(deduped_reps)} spatial duplicates")
    reps = deduped_reps

    print(f"  Extracting {len(reps)} crops...")
    crop_half = crop_size // 2
    zarr_cache = {}
    crops = {}
    failures = 0
    t0 = time.time()

    for ri, (s, bi, row, gene, local_i, score) in enumerate(reps):
        info = cell_info[(gene, local_i)]
        x = int(round(info["x"]))
        y = int(round(info["y"]))
        exp = info["experiment"]
        well = info["well"]
        well_row, well_col = well[0], well[1:]

        key = (exp, well)
        if key not in zarr_cache:
            zpath = os.path.join(
                zarr_base, exp,
                "3-assembly", "phenotyping_v3.zarr",
                well_row, well_col, "0", "0"
            )
            try:
                zarr_cache[key] = zarr.open(zpath, mode="r")
            except Exception:
                zarr_cache[key] = None

        arr = zarr_cache[key]
        crop = None
        if arr is not None:
            try:
                _, _, _, H, W = arr.shape
                y0, y1 = max(0, y - crop_half), min(H, y + crop_half)
                x0, x1 = max(0, x - crop_half), min(W, x + crop_half)
                raw = np.array(arr[0, phase_channel, 0, y0:y1, x0:x1])
                if raw.shape != (crop_size, crop_size):
                    padded = np.zeros((crop_size, crop_size), dtype=raw.dtype)
                    py = (crop_size - raw.shape[0]) // 2
                    px = (crop_size - raw.shape[1]) // 2
                    padded[py:py+raw.shape[0], px:px+raw.shape[1]] = raw
                    raw = padded
                crop = raw
            except Exception:
                failures += 1

        crops[(s, bi, row)] = crop
        if (ri + 1) % 1000 == 0:
            print(f"    [{ri+1}/{len(reps)}] {failures} failures, {time.time()-t0:.0f}s")

    ok = sum(1 for v in crops.values() if v is not None)
    print(f"    Done: {ok} crops, {failures} failures")
    return reps, crops, cell_info


# ── Save artifacts ──

def save_artifacts(output_dir, reps, crops, cell_info, gene_names,
                   gene_mean_scores, explained_variance, n_bins, n_rows, crop_size):
    from PIL import Image

    os.makedirs(output_dir, exist_ok=True)
    crops_dir = os.path.join(output_dir, "crops_png")
    os.makedirs(crops_dir, exist_ok=True)

    print(f"  Saving crop PNGs to {crops_dir}")
    saved = 0
    for (s, bi, row), crop in sorted(crops.items()):
        if crop is None:
            continue
        lo, hi = np.percentile(crop, (1, 99))
        if hi - lo < 1e-6:
            hi = lo + 1
        normed = np.clip((crop - lo) / (hi - lo), 0, 1)
        img = Image.fromarray((normed * 255).astype(np.uint8), mode="L")
        img.save(os.path.join(crops_dir, f"pc{s:03d}_bin{bi:02d}_row{row}.png"))
        saved += 1
    print(f"    {saved} PNGs saved")

    rep_list = []
    for s, bi, row, gene, local_i, score in reps:
        info = cell_info.get((gene, local_i), {})
        crop = crops.get((s, bi, row))
        rep_list.append({
            "pc": s,
            "bin": bi,
            "row": row,
            "gene": gene,
            "local_i": local_i,
            "score": round(float(score), 4),
            "experiment": info.get("experiment", ""),
            "well": info.get("well", ""),
            "x": round(float(info.get("x", 0)), 1),
            "y": round(float(info.get("y", 0)), 1),
            "has_crop": crop is not None,
        })
    with open(os.path.join(output_dir, "representatives.json"), "w") as f:
        json.dump({
            "representatives": rep_list,
            "cells_per_row": n_bins,
            "n_rows": n_rows,
            "crop_size": crop_size,
        }, f)
    print(f"    representatives.json: {len(rep_list)} entries")

    with open(os.path.join(output_dir, "gene_names.json"), "w") as f:
        json.dump(gene_names, f)
    np.save(os.path.join(output_dir, "gene_pc_scores.npy"), gene_mean_scores)
    print(f"    gene_names.json: {len(gene_names)} genes")
    print(f"    gene_pc_scores.npy: {gene_mean_scores.shape}")

    with open(os.path.join(output_dir, "gene_pc_analysis.json"), "w") as f:
        json.dump({"explained_variance": explained_variance}, f)
    print(f"    gene_pc_analysis.json saved")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--emb-dir", required=True,
                        help="Per-gene .pt files with embeddings + cell_metadata")
    parser.add_argument("--zarr-base", required=True,
                        help="Base directory of experiment zarr stores")
    parser.add_argument("--output-dir", required=True,
                        help="Directory for output artifacts")

    pca = parser.add_argument_group("PCA model")
    pca.add_argument("--n-components", type=int, default=97,
                     help="Number of PCA components (default: 97)")
    pca.add_argument("--save-model",
                     help="Save fitted PCA model to this directory")
    pca.add_argument("--load-model",
                     help="Load a previously saved PCA model (skip fitting)")

    vis = parser.add_argument_group("strip layout")
    vis.add_argument("--cells-per-row", type=int, default=15,
                     help="Bins along each strip (default: 15)")
    vis.add_argument("--n-rows", type=int, default=3,
                     help="Cells per bin (default: 3)")
    vis.add_argument("--crop-size", type=int, default=96,
                     help="Crop size in pixels (default: 96)")
    vis.add_argument("--phase-channel", type=int, default=0,
                     help="Zarr channel index for Phase2D (default: 0)")
    vis.add_argument("--batch-size", type=int, default=50000,
                     help="Batch size for streaming (default: 50000)")

    args = parser.parse_args()

    gene_files = list_gene_files(args.emb_dir)
    print(f"Found {len(gene_files)} gene files in {args.emb_dir}")

    print(f"\n=== PCA ({args.n_components} components) ===")
    scorer, n_strips, ev = load_or_fit_pca(
        args.emb_dir, args.n_components, args.batch_size,
        save_model=args.save_model, load_model=args.load_model
    )

    print(f"\n=== Selecting representatives ===")
    reps, total_cells, gene_names, gene_mean_scores = select_representatives(
        args.emb_dir, scorer, n_strips,
        args.cells_per_row, args.n_rows, args.batch_size
    )

    print(f"\n=== Extracting crops ===")
    reps, crops, cell_info = extract_crops(
        args.emb_dir, args.zarr_base, reps, args.crop_size,
        phase_channel=args.phase_channel
    )

    print(f"\n=== Saving artifacts to {args.output_dir} ===")
    save_artifacts(args.output_dir, reps, crops, cell_info,
                   gene_names, gene_mean_scores, ev,
                   args.cells_per_row, args.n_rows, args.crop_size)

    print(f"\nDone. {total_cells:,} cells, {len(gene_names)} genes, {n_strips} PCs.")
    print(f"Next: python3 build_static_explorer.py --artifacts-dir {args.output_dir}")


if __name__ == "__main__":
    main()
