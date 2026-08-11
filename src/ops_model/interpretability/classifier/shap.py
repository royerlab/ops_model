#!/usr/bin/env python
r"""Rank cells of (gene, marker) pairs by a SHAP-like attribution from the set classifier.

For each cell the attribution is the leave-one-out marginal contribution to
``P(true class)`` — ``P(class | bag) - P(class | bag without the cell)`` — averaged over
random bags and then averaged uniformly over a grid of bag sizes (the uniform-coalition-size
Shapley value). Bag size 1 is the deterministic single-cell probability ``P(class | cell)``.

This complements the existing rankings:
  - ``rank_gene_marker.py`` scores by class_prob (mean P(class) over bags) — an *aggregate*.
  - ``export_pma_attention.py`` scores by the PMA attention weight.
  - this script scores by each cell's *marginal* contribution across bag sizes.

Bags are single-channel (all cells of one marker). One CSV row per cell carries the SHAP
value, the per-bag-size marginals, and the cell's coordinates (so montages/viewers can render
straight from the ranking without a lossy re-merge to the dump).

Example:
-------
    python shap_rank.py \\
        --checkpoint best_set_classifier_paper_v2_phase_e200.pt \\
        --dump_dir .../paper_v2_phase/train .../paper_v2_phase/val \\
        --channel Phase2D --genes KIF23 HSPA5 AURKB \\
        --bag_sizes 1 2 5 10 20 50 100 200 500 --reps 50 \\
        --out_csv shap_phase.csv
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from .train import build_model


def flatten_meta(cm: dict) -> dict:
    """Flatten a per-cell metadata dict whose values may be lists-of-lists (one inner
    list per experiment chunk) into flat per-cell lists.
    """
    return {
        k: ([x for seg in v for x in seg] if v and isinstance(v[0], list) else list(v))
        for k, v in cm.items()
    }


@torch.no_grad()
def single_cell_prob(model, emb: torch.Tensor, ch: torch.Tensor, tl: int) -> np.ndarray:
    """Deterministic bag-size-1 attribution: ``P(true class | cell alone)`` per cell."""
    n = emb.shape[0]
    out = torch.empty(n, dtype=torch.float64)
    for i in range(0, n, 4096):
        e = emb[i : i + 4096][:, None, :]
        c = ch[i : i + 4096][:, None]
        m = torch.ones_like(c, dtype=torch.bool)
        out[i : i + 4096] = torch.softmax(model(e, c, m), -1)[:, tl].double().cpu()
    return out.numpy()


@torch.no_grad()
def marginal(
    model,
    emb: torch.Tensor,
    ch: torch.Tensor,
    tl: int,
    bag: int,
    n_reps: int,
    seed: int,
    device: torch.device,
) -> np.ndarray:
    """Mean leave-one-out marginal of each cell at a given bag size.

    Over ``n_reps`` random partitions of the cells into bags of size ``bag``, each cell lands
    in exactly one bag per rep. Within a bag we compute the full-bag prediction once and each
    leave-one-out prediction (``bag + 1`` forward passes per bag), so a cell's marginal is
    ``P(class | bag) - P(class | bag without cell)`` averaged over the ``n_reps`` bags it fell
    into. The leave-one-out set is materialized by masking, and bags are processed in blocks to
    bound peak activation memory (the block spans a ``(block*bag, bag, d)`` expansion).
    """
    n = emb.shape[0]
    marg_sum = torch.zeros(n, dtype=torch.float64, device=device)
    cnt = torch.zeros(n, dtype=torch.float64, device=device)
    nbpr = (n + bag - 1) // bag
    eye = torch.eye(bag, dtype=torch.bool, device=device)
    block = max(1, 300_000 // (bag * bag))
    for rep in range(n_reps):
        g = torch.Generator().manual_seed(seed * 1_000_003 + rep)
        perm = torch.randperm(n, generator=g)
        pad = nbpr * bag - n
        if pad:
            perm = torch.cat(
                [perm, perm.new_zeros(pad)]
            )  # pad with a real index; masked out below
        valid = torch.ones(nbpr * bag, dtype=torch.bool)
        valid[n:] = False
        bag_idx = perm.view(nbpr, bag).to(device)
        vm = valid.view(nbpr, bag).to(device)
        for b0 in range(0, nbpr, block):
            sub = bag_idx[b0 : b0 + block]
            vmb = vm[b0 : b0 + block]
            nb = sub.shape[0]
            e = emb[sub]
            c = ch[sub]
            p_full = torch.softmax(model(e, c, vmb), -1)[:, tl].double()
            mask_loo = vmb[:, None, :] & (~eye)[None, :, :]
            er = e[:, None, :, :].expand(nb, bag, bag, -1).reshape(nb * bag, bag, -1)
            cr = c[:, None, :].expand(nb, bag, bag).reshape(nb * bag, bag)
            mr = mask_loo.reshape(nb * bag, bag)
            p_loo = torch.softmax(model(er, cr, mr), -1)[:, tl].double().view(nb, bag)
            mg = p_full[:, None] - p_loo
            flat = sub[vmb]
            marg_sum.index_add_(0, flat, mg[vmb])
            cnt.index_add_(
                0, flat, torch.ones(int(vmb.sum()), dtype=torch.float64, device=device)
            )
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return (marg_sum / cnt.clamp(min=1)).cpu().numpy()


# Relative reps taper (reps_B ∝ σ_B/√B, from the measured per-sample marginal std on phase
# genes): fraction of the anchor reps to spend at each bag size. Off-grid bags use a power-law
# fallback. Bags land no lower than --min_reps.
_TAPER_FRAC = {
    2: 1.0,
    5: 0.6,
    10: 0.3,
    20: 0.1,
    50: 0.04,
    100: 0.02,
    200: 0.01,
    500: 0.002,
}


def reps_for_bag(bag: int, anchor: int, min_reps: int) -> int:
    """Reps to use at a given bag size: the anchor scaled by the variance-based taper, floored
    at min_reps. Bag 1 is deterministic (handled by the caller) so this is only for bag >= 2.
    """
    frac = _TAPER_FRAC.get(bag, (2.0 / bag) ** 0.85)
    return max(min_reps, round(anchor * frac))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument(
        "--dump_dir", required=True, nargs="+", help="dump dirs pooled (train val)"
    )
    ap.add_argument(
        "--channel", required=True, help="marker/channel name, e.g. Phase2D or 5xUPRE"
    )
    ap.add_argument("--genes", required=True, nargs="+", help="genes to rank")
    ap.add_argument(
        "--bag_sizes",
        type=int,
        nargs="+",
        default=[1, 2, 5, 10, 20, 50],
        help="bag sizes to average the marginal over (bag 1 = single-cell P(class))",
    )
    ap.add_argument(
        "--reps",
        type=int,
        default=100,
        help="anchor reps for the smallest bag; larger bags are tapered down toward "
        "--min_reps (the per-sample marginal variance collapses with bag size). "
        "Use --flat_reps to disable the taper.",
    )
    ap.add_argument(
        "--min_reps", type=int, default=10, help="floor on reps at any bag size"
    )
    ap.add_argument(
        "--flat_reps",
        action="store_true",
        help="use --reps uniformly for every bag size (no taper)",
    )
    ap.add_argument(
        "--reps_schedule",
        type=int,
        nargs="+",
        default=None,
        help="explicit reps per bag size (must match --bag_sizes length); overrides taper",
    )
    ap.add_argument(
        "--max_cells",
        type=int,
        default=65000,
        help="subsample a (gene, channel) pool larger than this (fixed seed); the "
        "full leave-one-out SHAP is infeasible for huge pools like the NTC control",
    )
    ap.add_argument("--out_csv", required=True)
    ap.add_argument(
        "--seed", type=int, default=0, help="seed for the random bag partitions"
    )
    ap.add_argument(
        "--subsample_seed",
        type=int,
        default=0,
        help="seed for the >--max_cells subsample; kept separate from --seed so the "
        "same cells are scored when only the partition seed changes (e.g. reruns)",
    )
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build_model(ckpt, device)
    gene_to_idx = ckpt["gene_to_idx"]
    cci = ckpt["channel_to_idx"][args.channel]
    label_remap = ckpt.get("label_remap")
    md = torch.load(
        Path(args.dump_dir[0]) / "metadata.pt", map_location="cpu", weights_only=False
    )
    dci = md["channel_to_idx"][args.channel]
    bags = sorted(set(args.bag_sizes))
    # reps per bag: explicit schedule > flat > variance-based taper (default)
    if args.reps_schedule is not None:
        if len(args.reps_schedule) != len(args.bag_sizes):
            raise ValueError("--reps_schedule length must match --bag_sizes length")
        reps_map = dict(zip(args.bag_sizes, args.reps_schedule))
    elif args.flat_reps:
        reps_map = dict.fromkeys(bags, args.reps)
    else:
        reps_map = {b: reps_for_bag(b, args.reps, args.min_reps) for b in bags}
    reps_map[1] = 1  # bag 1 is deterministic (single forward pass)
    print("reps per bag: " + ", ".join(f"{b}:{reps_map[b]}" for b in bags), flush=True)

    with open(args.out_csv, "w", newline="") as fo:
        w = csv.writer(fo)
        w.writerow(
            ["gene", "channel_name", "rank", "shap", "bag1"]
            + [f"marg_{b}" for b in bags]
            + [
                "split",
                "experiment",
                "well",
                "y_pheno",
                "x_pheno",
                "segmentation_id",
                "zarr_channel_index",
                "n_cells",
            ]
        )
        for gene in args.genes:
            if gene not in gene_to_idx:
                print(f"skip {gene}: not in gene_to_idx", flush=True)
                continue
            tl = (
                label_remap[gene_to_idx[gene]]
                if label_remap is not None
                else gene_to_idx[gene]
            )
            # pool the gene's cells for this one channel across dump dirs, tracking metadata + split
            embs_l, meta_l, split_l = [], [], []
            for dd in args.dump_dir:
                gp = Path(dd) / f"{gene}.pt"
                if not gp.exists():
                    continue
                d = torch.load(gp, map_location="cpu", weights_only=False)
                sel = np.flatnonzero(d["channel_ids"].long().numpy() == dci)
                if sel.size == 0:
                    continue
                embs_l.append(d["embeddings"].float().numpy()[sel])
                m = flatten_meta(d["cell_metadata"])
                meta_l.append(
                    {
                        k: [m[k][i] for i in sel]
                        for k in (
                            "experiment",
                            "well",
                            "y_pheno",
                            "x_pheno",
                            "segmentation_id",
                            "index",
                        )
                    }
                )
                split_l.extend([Path(dd).name] * sel.size)
            if not embs_l:
                print(f"skip {gene}: no {args.channel} cells", flush=True)
                continue
            emb_np = np.concatenate(embs_l, 0)
            meta = {k: [v for m in meta_l for v in m[k]] for k in meta_l[0]}
            splits = np.array(split_l)
            n_full = emb_np.shape[0]
            keep = np.arange(n_full)
            if n_full > args.max_cells:
                rng = np.random.default_rng(args.subsample_seed)
                keep = np.sort(rng.choice(n_full, size=args.max_cells, replace=False))
                emb_np = emb_np[keep]
                splits = splits[keep]

            emb = torch.tensor(emb_np, device=device)
            n = emb.shape[0]
            ch = torch.full((n,), cci, dtype=torch.long, device=device)
            margs = {
                b: (
                    single_cell_prob(model, emb, ch, tl)
                    if b == 1
                    else marginal(model, emb, ch, tl, b, reps_map[b], args.seed, device)
                )
                for b in bags
            }
            shap = np.mean(np.stack([margs[b] for b in bags], 0), 0)
            bag1 = margs[1] if 1 in margs else np.full(n, np.nan)
            order = np.argsort(-shap)
            for rk, ci in enumerate(order):
                gi = int(keep[ci])
                w.writerow(
                    [gene, args.channel, rk + 1, f"{shap[ci]:.6f}", f"{bag1[ci]:.6f}"]
                    + [f"{margs[b][ci]:.6f}" for b in bags]
                    + [
                        splits[ci],
                        meta["experiment"][gi],
                        meta["well"][gi],
                        meta["y_pheno"][gi],
                        meta["x_pheno"][gi],
                        meta["segmentation_id"][gi],
                        meta["index"][gi],
                        n,
                    ]
                )
            print(
                f"{gene} x {args.channel}: n={n} ranked (shap mean={shap.mean():+.5f})",
                flush=True,
            )
    print(f"done -> {args.out_csv}", flush=True)


if __name__ == "__main__":
    main()
