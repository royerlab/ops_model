"""Visual counterfactual for the 2026-08-20 bimodal-shape-recovery finding (see
FLOW_OT_CELLSTATE_PROGRAM.md build log): mean_diff cannot manufacture bimodal structure a
translation didn't already have; flow-OT can. That result was embedding-space-only (a KMeans
separation statistic); this renders the actual DECODED IMAGES behind it, for exactly the 2 real
NTC cells whose flow-OT endpoint lands closest to each real KD sub-cluster — the pair the
trained field itself says diverges most, not an arbitrary/random pair. Reuses the already-built
`flow_ot.pt` / `mean_diff.npz` artifacts from `flow_field.build_flow_field` (no retraining, no
LR refit) and the montage rendering machinery from `ot_cfm_test.py`.

    python -m ops_model.models.interpretability.diffae.directions.bimodal_probe_montage \
        --target KIF11 --flow-field-dir /path/to/flow_fields/geneKO/KIF11 --out-dir /path/to/out
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import KMeans

from ..generator.data import normalize
from .config import DirConfig
from .flow import integrate_flow
from .flow_field_compute import integrate_path, load_net
from .ot_cfm_test import _mean_diff_trajectory, _score_trajectory
from .traverse import load_diffae


def _pick_divergent_probes(feats, labels, net, n_sub: int = 30, t_max: float = 1.0, seed: int = 0):
    """Among ALL real NTC cells, find the one whose flow-OT endpoint lands closest to each real
    KD sub-cluster centroid — the pair the trained field itself routes furthest apart, matching
    the exact selection criterion (same KMeans seed, same n_sub/t_max) behind the sep-recovery
    numbers in bimodal_shape_test.py."""
    kd, ntc = feats[labels == 1], feats[labels == 0]
    km = KMeans(n_clusters=2, n_init=4, random_state=seed).fit(kd)
    cA, cB = kd[km.labels_ == 0].mean(0), kd[km.labels_ == 1].mean(0)

    ntc_idx = np.flatnonzero(labels == 0)
    endpoints = integrate_path(net, torch.as_tensor(ntc, dtype=torch.float32), n_sub=n_sub, t_max=t_max)[-1]
    dA = np.linalg.norm(endpoints - cA[None], axis=1)
    dB = np.linalg.norm(endpoints - cB[None], axis=1)
    probe_a, probe_b = int(ntc_idx[np.argmin(dA)]), int(ntc_idx[np.argmin(dB)])

    kd_idx = np.flatnonzero(labels == 1)
    real_a_idx = int(kd_idx[km.labels_ == 0][0])
    real_b_idx = int(kd_idx[km.labels_ == 1][0])
    return probe_a, probe_b, real_a_idx, real_b_idx


def build_bimodal_probe_montage(cfg: DirConfig, flow_field_dir: str, out_dir: str,
                                w: float = 1.0, alphas=(0.0, 1.0, 2.0), dpi: int = 300) -> str:
    """KO-arm only (alphas>=0) — the anti-KO extrapolation half isn't part of the claim being
    shown (see ot_cfm_test.py's half-axis rationale) and dropping it doubles the size of the
    panels that actually matter. `alphas[-1]` also sets flow-OT's integration t_max, so the two
    methods land on the exact same evenly-spaced grid (same apples-to-apples convention as
    run_sweep/render_montage)."""
    dev = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    ff = Path(flow_field_dir)
    cache = ff / "cache"
    feats = np.load(cache / "celldino.npz")["features"].astype(np.float32)
    crops = np.load(cache / "crops.npz")
    images, labels = crops["images"], crops["labels"]

    net_cpu = load_net(ff)
    probe_a, probe_b, real_a_idx, real_b_idx = _pick_divergent_probes(feats, labels, net_cpu)
    print(f"[{cfg.target}] probe_a(->clusterA)={probe_a} probe_b(->clusterB)={probe_b}")

    mdz = np.load(ff / "mean_diff.npz")
    d_vec, lr_w, lr_b, gap = mdz["d_vec"], mdz["lr_w"], mdz["lr_b"], float(mdz["gap"])

    diffae = load_diffae(cfg, dev)
    null_base = diffae.null_emb.detach()[None].to(dev)
    net = net_cpu.to(dev)
    H = cfg.crop_size
    alphas = list(alphas)

    probes = [probe_a, probe_b]
    src_embs = feats[probes]
    xT_list = [torch.randn(1, 1, H, H, generator=torch.Generator(device=dev).manual_seed(1234 + i),
                           device=dev) for i in range(len(probes))]

    mean_diff_traj = [torch.as_tensor(_mean_diff_trajectory(src_embs[i], d_vec, gap, alphas))
                      for i in range(len(probes))]
    gen_md, _, _ = _score_trajectory(diffae, null_base, mean_diff_traj, xT_list, w, cfg, lr_w, lr_b)

    flow_traj = [integrate_flow(net, torch.as_tensor(src_embs[i:i + 1], dtype=torch.float32),
                                dev, n_record=len(alphas) - 1, t_max=alphas[-1]) for i in range(len(probes))]
    gen_fo, _, _ = _score_trajectory(diffae, null_base, flow_traj, xT_list, w, cfg, lr_w, lr_b)

    ctrl_imgs = normalize(images[probes])
    ref_imgs = normalize(images[[real_a_idx, real_b_idx]])

    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["pdf.fonttype"] = 42
    import matplotlib.pyplot as plt

    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    ncols = len(alphas) + 2
    row_specs = [("mean_diff", 0, gen_md), ("flow-OT", 0, gen_fo), ("mean_diff", 1, gen_md), ("flow-OT", 1, gen_fo)]
    block_labels = {0: "probe -> clusterA", 1: "probe -> clusterB"}

    fig, axes = plt.subplots(4, ncols, figsize=(3.0 * ncols, 3.2 * 4), squeeze=False,
                             gridspec_kw={"wspace": 0.02, "hspace": 0.08})
    for row, (method, b, gen) in enumerate(row_specs):
        axrow = axes[row]
        axrow[0].imshow(ctrl_imgs[b, 0], cmap="gray", vmin=-1, vmax=1, interpolation="bicubic")
        axrow[0].set_title("REAL ctrl (probe)", fontsize=11)
        for j, a in enumerate(alphas):
            axrow[j + 1].imshow(gen[b, j], cmap="gray", vmin=-1, vmax=1, interpolation="bicubic")
            axrow[j + 1].set_title("NTC (α=0)" if a == 0 else f"α={a:+.1f}×gap", fontsize=11)
        axrow[-1].imshow(ref_imgs[b, 0], cmap="gray", vmin=-1, vmax=1, interpolation="bicubic")
        axrow[-1].set_title(f"REAL KD ({block_labels[b].split('-> ')[-1]})", fontsize=11)
        for ax in axrow:
            ax.axis("off")
        axrow[0].text(-0.06, 0.5, f"{method}\n({block_labels[b]})", transform=axrow[0].transAxes,
                     fontsize=13, fontweight="bold", ha="right", va="center", rotation=90)
    fig.suptitle(
        f"{cfg.target} (w={w:g}, KO-arm only): same 2 real control cells per block, only the "
        f"direction/field differs — does flow-OT's endpoint visually track the real KD "
        f"sub-cluster it was routed to, while mean_diff doesn't?",
        fontsize=13, y=1.01)
    p = out / f"bimodal_visual_{cfg.target}_w{w:g}_hires.png"
    fig.savefig(p, dpi=dpi, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"wrote {p}")
    return str(p)


def main() -> None:
    ap = argparse.ArgumentParser(description="Decoded-image demo of the mean_diff-vs-flow-OT bimodal-shape finding")
    ap.add_argument("--target", required=True)
    ap.add_argument("--flow-field-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--w", type=float, default=1.0)
    args = ap.parse_args()
    cfg = DirConfig(grain="geneKO", target=args.target, device="cuda")
    build_bimodal_probe_montage(cfg, args.flow_field_dir, args.out_dir, w=args.w)


if __name__ == "__main__":
    main()
