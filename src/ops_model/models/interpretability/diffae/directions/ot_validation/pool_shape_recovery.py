"""Decode-validated version of the 2026-08-20 bimodal-shape-recovery finding
(FLOW_OT_CELLSTATE_PROGRAM.md build log / bimodal_shape_test.json): that result compared raw
pre-decode CellDINO embeddings (mean_diff's output KMeans-separation == the untouched NTC null,
flow-OT's matches real KD truth). This closes the loop through the actual generation pipeline —
push a RANDOM POOL of real NTC cells (not 2 hand-picked extremes, a real population statistic)
through both methods to the SAME alpha, decode each endpoint with the frozen DiffAE, RE-EMBED
the decoded image via CellDINO, and check whether the same separation pattern survives contact
with real image generation (a known risk: the documented generated-vs-real domain gap could
smear out a signal that's clean in raw embedding space).

    python -m ops_model.models.interpretability.diffae.directions.pool_shape_recovery \
        --target TUBGCP6 --flow-field-dir /path/to/flow_fields/geneKO/TUBGCP6 --out-dir /path/to/out
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from ..classifier.celldino_features import embed_crops
from .config import DirConfig
from .flow import integrate_flow
from .flow_field_compute import load_net
from .traverse import _sample_guided, load_diffae


def _split_score(X: np.ndarray, seed: int = 0) -> dict | None:
    """2-cluster separation stat (bimodality_scan.py / bimodal_shape_test.py convention). Kept
    for continuity with the original bar chart, but NOT the trustworthy metric here — it
    re-clusters a population against ITSELF, so a population that's half stuck-at-NTC and half
    moved-toward-one-real-cluster is ALSO internally bimodal and passes this stat without
    meaning what the number implies (see the 2026-08-20 build-log correction). Use
    `_classify_nearest` for the real claim."""
    if len(X) < 20:
        return None
    km = KMeans(n_clusters=2, n_init=4, random_state=seed).fit(X)
    n0, n1 = (km.labels_ == 0).sum(), (km.labels_ == 1).sum()
    if n0 == 0 or n1 == 0:
        return None
    c0, c1 = X[km.labels_ == 0].mean(0), X[km.labels_ == 1].mean(0)
    sep = float(np.linalg.norm(c0 - c1) / (np.sqrt(0.5 * (X[km.labels_ == 0].var(0).mean() +
                                                          X[km.labels_ == 1].var(0).mean())) + 1e-6))
    return {"sep": sep, "balance": float(min(n0, n1) / max(n0, n1))}


def _classify_nearest(pts: np.ndarray, c_ntc: np.ndarray, c_a: np.ndarray, c_b: np.ndarray) -> dict:
    """The actual claim being tested: does each point land nearest the REAL NTC centroid, real
    cluster-A centroid, or real cluster-B centroid — not whether the population re-clusters
    against itself. Fixes the blind spot in `_split_score`."""
    d_ntc = np.linalg.norm(pts - c_ntc[None], axis=1)
    d_a = np.linalg.norm(pts - c_a[None], axis=1)
    d_b = np.linalg.norm(pts - c_b[None], axis=1)
    lab = np.argmin(np.stack([d_ntc, d_a, d_b], axis=1), axis=1)
    n = len(lab)
    return {"NTC": float((lab == 0).sum() / n), "A": float((lab == 1).sum() / n), "B": float((lab == 2).sum() / n)}


def _decode_and_reencode(diffae, null_base, embs: np.ndarray, xT_list, w: float, cfg) -> np.ndarray:
    """Decode each row of `embs` with its OWN fixed noise, re-embed the decoded image via
    CellDINO — the actual generation pipeline the raw-embedding bar chart bypassed entirely."""
    dev = xT_list[0].device
    imgs = []
    for i in range(len(embs)):
        z = torch.as_tensor(embs[i:i + 1], dtype=torch.float32, device=dev)
        img = _sample_guided(diffae, xT_list[i].clone(), z, null_base, w, cfg)
        imgs.append(img.cpu().numpy()[0, 0])
    flat = np.stack(imgs)[:, None].astype(np.float32)
    return embed_crops(flat, cfg, cache_path=None)


def run_pool_shape_recovery(cfg: DirConfig, flow_field_dir: str, out_dir: str,
                            n_pool: int = 100, alpha: float = 3.0, w: float = 1.0, seed: int = 0) -> dict:
    dev = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    ff = Path(flow_field_dir); cache = ff / "cache"
    feats = np.load(cache / "celldino.npz")["features"].astype(np.float32)
    labels = np.load(cache / "crops.npz")["labels"]
    kd, ntc = feats[labels == 1], feats[labels == 0]
    km_real = KMeans(n_clusters=2, n_init=4, random_state=0).fit(kd)
    c_ntc, c_a, c_b = ntc.mean(0), kd[km_real.labels_ == 0].mean(0), kd[km_real.labels_ == 1].mean(0)

    mdz = np.load(ff / "mean_diff.npz")
    d_vec, gap = mdz["d_vec"], float(mdz["gap"])
    net = load_net(ff).to(dev)

    rng = np.random.default_rng(seed)
    ntc_idx = np.flatnonzero(labels == 0)
    pool_idx = rng.choice(ntc_idx, size=min(n_pool, len(ntc_idx)), replace=False)
    z0 = feats[pool_idx]

    diffae = load_diffae(cfg, dev)
    null_base = diffae.null_emb.detach()[None].to(dev)
    H = cfg.crop_size
    xT_list = [torch.randn(1, 1, H, H, generator=torch.Generator(device=dev).manual_seed(1234 + i),
                           device=dev) for i in range(len(pool_idx))]

    md_end = z0 + alpha * gap * d_vec[None, :]
    z0_t = torch.as_tensor(z0, dtype=torch.float32, device=dev)
    # integrate_flow concatenates trajectory steps along dim=0, which only yields the right
    # shape for a batch-size-1 call (see traverse.py/bimodal_probe_montage.py's per-cell usage)
    # -- feeding it the whole pool at once silently flattens (n_record+1)*n_pool rows together.
    fo_end = np.stack([integrate_flow(net, z0_t[i:i + 1], dev, n_record=1, t_max=alpha)[-1].cpu().numpy()
                       for i in range(len(pool_idx))])

    decoded_ntc = _decode_and_reencode(diffae, null_base, z0, xT_list, w, cfg)      # alpha=0 decode-noise null
    decoded_md = _decode_and_reencode(diffae, null_base, md_end, xT_list, w, cfg)
    decoded_fo = _decode_and_reencode(diffae, null_base, fo_end, xT_list, w, cfg)

    result = {
        "target": cfg.target, "n_pool": int(len(pool_idx)), "alpha": alpha,
        # kept for continuity with the earlier bar chart -- NOT the trustworthy metric, see docstring
        "kd_truth": _split_score(kd),
        "decoded_ntc_null": _split_score(decoded_ntc),
        "decoded_mean_diff": _split_score(decoded_md),
        "decoded_flow_ot": _split_score(decoded_fo),
        # the actual claim: nearest real-centroid classification
        "classify": {
            "sanity_real_A": _classify_nearest(kd[km_real.labels_ == 0], c_ntc, c_a, c_b),
            "sanity_real_B": _classify_nearest(kd[km_real.labels_ == 1], c_ntc, c_a, c_b),
            "sanity_real_NTC_pool": _classify_nearest(ntc, c_ntc, c_a, c_b),
            "decoded_ntc_null": _classify_nearest(decoded_ntc, c_ntc, c_a, c_b),
            "decoded_mean_diff": _classify_nearest(decoded_md, c_ntc, c_a, c_b),
            "decoded_flow_ot": _classify_nearest(decoded_fo, c_ntc, c_a, c_b),
        },
    }
    print(json.dumps(result, indent=2))

    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    stem = f"pool_shape_recovery_{cfg.target}_n{len(pool_idx)}_a{alpha:g}"
    (out / f"{stem}.json").write_text(json.dumps(result, indent=2))
    # cache raw arrays so plot-only style tweaks (colors, markers, ...) don't need to redo the
    # GPU decode -- see replot_pool_shape_recovery, CPU-only / no DiffAE reload required.
    np.savez(out / f"{stem}.npz", ntc=ntc, kd=kd, kd_cluster=km_real.labels_,
            decoded_ntc=decoded_ntc, decoded_md=decoded_md, decoded_fo=decoded_fo)

    p = _render(result, ntc, kd, km_real.labels_, decoded_md, decoded_fo, cfg.target,
               len(pool_idx), alpha, out / f"{stem}.png")
    print(f"wrote {p}")
    result["fig_path"] = str(p)
    return result


def _render(result: dict, ntc, kd, kd_cluster, decoded_md, decoded_fo, target: str,
           n_pool: int, alpha: float, out_path: Path,
           md_color: str = "blue", fo_color: str = "magenta", ntc_color: str = "dimgray") -> str:
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["pdf.fonttype"] = 42
    import matplotlib.pyplot as plt

    pca = PCA(n_components=2).fit(np.vstack([ntc, kd]))
    c_ntc, c_a, c_b = ntc.mean(0), kd[kd_cluster == 0].mean(0), kd[kd_cluster == 1].mean(0)
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(16, 6.5))
    ntc_2d, kd_2d = pca.transform(ntc), pca.transform(kd)
    ax_l.scatter(ntc_2d[:, 0], ntc_2d[:, 1], s=10, c=ntc_color, alpha=0.6, label="real NTC")
    ax_l.scatter(kd_2d[kd_cluster == 0, 0], kd_2d[kd_cluster == 0, 1], s=10, c="salmon", alpha=0.6, label="real KD (cluster A)")
    ax_l.scatter(kd_2d[kd_cluster == 1, 0], kd_2d[kd_cluster == 1, 1], s=10, c="orange", alpha=0.6, label="real KD (cluster B)")
    md_2d, fo_2d = pca.transform(decoded_md), pca.transform(decoded_fo)
    ax_l.scatter(md_2d[:, 0], md_2d[:, 1], s=45, c=md_color, marker="o", edgecolor="black", linewidth=0.6,
                label=f"decoded mean_diff (n={len(md_2d)})")
    ax_l.scatter(fo_2d[:, 0], fo_2d[:, 1], s=70, c=fo_color, marker="*", edgecolor="black", linewidth=0.6,
                label=f"decoded flow-OT (n={len(fo_2d)})")
    for c, name in [(c_ntc, "NTC"), (c_a, "A"), (c_b, "B")]:
        c2d = pca.transform(c[None])[0]
        ax_l.scatter(*c2d, s=180, c="black", marker="X", edgecolor="white", linewidth=1.2, zorder=5)
        ax_l.annotate(f"centroid-{name}", c2d, fontsize=8, fontweight="bold", ha="center", va="bottom",
                     xytext=(0, 8), textcoords="offset points")
    ax_l.set_xlabel("PC1"); ax_l.set_ylabel("PC2"); ax_l.legend(loc="best", fontsize=8, frameon=False)
    ax_l.set_title(f"{target}: decoded+re-embedded endpoints (α={alpha:g}×gap, n={n_pool} real NTC pool)\n"
                  f"black X = the 3 centroids nearest-classification is measured against", fontsize=10)

    c = result["classify"]
    rows = ["sanity_real_A", "sanity_real_B", "sanity_real_NTC_pool", "decoded_ntc_null", "decoded_mean_diff", "decoded_flow_ot"]
    row_labels = ["sanity: real A\ncells ->", "sanity: real B\ncells ->", "sanity: real NTC\npool ->",
                 "decoded NTC\n(α=0 null)", "decoded\nmean_diff", "decoded\nflow-OT"]
    seg_colors = {"NTC": ntc_color, "A": "salmon", "B": "orange"}
    y = np.arange(len(rows))
    left = np.zeros(len(rows))
    for seg in ["NTC", "A", "B"]:
        vals = np.array([c[r][seg] for r in rows])
        ax_r.barh(y, vals, left=left, color=seg_colors[seg], label=f"classifies as {seg}", edgecolor="black", linewidth=0.4)
        for yi, (v, l) in enumerate(zip(vals, left)):
            if v > 0.04:
                ax_r.text(l + v / 2, yi, f"{v:.0%}", ha="center", va="center", fontsize=8)
        left += vals
    ax_r.set_yticks(y); ax_r.set_yticklabels(row_labels, fontsize=9)
    ax_r.invert_yaxis()
    ax_r.set_xlabel("fraction classified nearest each real centroid")
    ax_r.set_xlim(0, 1)
    ax_r.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False, fontsize=9)
    ax_r.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"{target} (α={alpha:g}): nearest-real-centroid classification of decoded endpoints "
                f"— the corrected test (see 2026-08-20 build-log correction)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return str(out_path)


def replot_pool_shape_recovery(npz_path: str, json_path: str, out_path: str,
                               md_color: str = "blue", fo_color: str = "magenta",
                               ntc_color: str = "dimgray") -> str:
    """Re-render the figure from cached arrays (see the np.savez call above) -- CPU-only, no
    DiffAE/GPU reload, for style-only iteration (colors, markers, ...)."""
    d = np.load(npz_path)
    result = json.loads(Path(json_path).read_text())
    return _render(result, d["ntc"], d["kd"], d["kd_cluster"], d["decoded_md"], d["decoded_fo"],
                  result["target"], result["n_pool"], result["alpha"], Path(out_path),
                  md_color=md_color, fo_color=fo_color, ntc_color=ntc_color)


def main() -> None:
    ap = argparse.ArgumentParser(description="Decode-validated pool-level bimodal shape recovery test")
    ap.add_argument("--target", required=True)
    ap.add_argument("--flow-field-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-pool", type=int, default=100)
    ap.add_argument("--alpha", type=float, default=3.0)
    ap.add_argument("--w", type=float, default=1.0)
    args = ap.parse_args()
    cfg = DirConfig(grain="geneKO", target=args.target, device="cuda")
    run_pool_shape_recovery(cfg, args.flow_field_dir, args.out_dir, n_pool=args.n_pool, alpha=args.alpha, w=args.w)


if __name__ == "__main__":
    main()
