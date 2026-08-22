"""OT-CFM vs. independent-CFM vs. mean_diff — direction-method comparison, multi-metric.

Workstream A of interpretability/FLOW_OT_CELLSTATE_PROGRAM.md. v1 of this test used ONE
metric pair (full-axis monotonicity + endpoint score-delta) at ONE guidance scale and found
it degenerate (monotonicity 0.0 for all three methods, including production mean_diff,
because w=5 is known — 2026-06-18 build log — to trade monotonicity for score magnitude).
v2 (this module) fixes that and adds checks a single classifier-logit metric can't provide:

  - half-axis metrics, split at NTC: only the control->KO arm is the one DiffEx actually
    claims to explain; the anti-KO extrapolation arm is a bonus, not the interpretability
    target, so conflating them into one monotonicity number (v1's mistake) hides the signal.
  - overshoot ratio on the KO arm: v1's comparison PLOT showed independent-coupling flow
    peaking mid-traversal then declining — the literal "noisy negative extreme" symptom, but
    v1 had no NUMBER for it. overshoot = (peak - endpoint) / (peak - start), so a method that
    keeps rising to its endpoint scores ~0 and one that peaks-then-decays scores higher.
  - faithfulness-to-real-population distance: the LR logit alone can be gamed by a direction
    that increases the classifier score without the embedding actually approaching the real
    KD cell cloud. Distance from the generated (re-encoded) endpoint to the REAL KD/NTC
    centroids is a classifier-independent sanity check.
  - pixel-localization proxy: a known failure mode (PLAN.md 2026-06-18) is edits landing on
    crop borders/background instead of the cell body. No segmentation mask is used by
    default (cfg.mask_cell=False), so this is an approximation — energy inside a centered
    disk vs. the outer ring of the Delta-pixel map — not a substitute for a real mask-based
    check, but a cheap first filter for "is this edit even on the cell."
  - reproducibility across training seeds: mean_diff is deterministic by construction; a
    flow net is fit by SGD from a random init with random minibatch draws, so seed-to-seed
    consistency is a real risk this codebase has hit before (the unsupervised InfoNCE
    direction bank was demoted for exactly this — run-to-run instability). v1 tested ONE
    seed per method and so could not see this at all.
  - swept over multiple guidance scales `w` (not just the most aggressive production value),
    since w trades monotonicity for magnitude — comparing methods at only one (aggressive) w
    conflates "is this method good" with "is this w too aggressive for anything."

Also fixes a v1 rigor gap: mean_diff's traversal grid (`cfg.alphas`) is NOT evenly spaced
(denser near 0), while the flow arms' Euler-integration steps ARE evenly spaced — so v1's
x-axis (plotted by raw index) silently compared methods at slightly different actual
alpha/t values at the same index. v2 puts mean_diff on the SAME evenly-spaced grid the flow
arms use, so every index is a true apples-to-apples point across all three methods.

Reads the production phase_v1 DiffAE checkpoint read-only. All outputs go under the
caller's --out-dir (default: a scratch dir under coding_exps/diffex/ot_cfm_test/) — never
the production diffae/directions/... tree.

    python -m ops_model.models.interpretability.diffae.directions.ot_cfm_test --target HSPA5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ...classifier.celldino_features import embed_crops
from ...classifier.config import GRAINS
from ...classifier.data import make_labels_df, materialize_crops
from ...generator.data import normalize
from ..config import DirConfig
from ..data import _top_cells, gather
from ..flow_matching.flow import integrate_flow_bidir, train_flow
from ..rank import supervised_direction
from ..traverse import _sample_guided, load_diffae

METHODS = ("mean_diff", "flow_independent", "flow_ot")


def _gather_pooled_complex(cfg: DirConfig, crops_cache: str, emb_cache: str):
    """Like `data.gather`, but for a pooled/multimodal EBI-complex target: the complex
    ranking parquet (GRAINS['complex'], `pma_shap_phase_complex.parquet`) carries NO NTC
    rows at all — it's a different (SHAP-based) ranking source than the geneKO attention
    parquet, so `gather()`'s single-parquet assumption (target + control from the same
    file) doesn't hold for grain='complex'. Pulls the target's top cells from the complex
    parquet and the NTC control from the geneKO parquet instead — both share the same
    `_BASE_COLS` schema (experiment/well/segmentation/x_pheno/y_pheno/pma_attention/rank),
    so this is a safe cross-source join, not a hack around incompatible data."""
    g_complex, g_gene = GRAINS["complex"], GRAINS["geneKO"]
    pos = _top_cells(g_complex["parquet"], g_complex["class_col"], cfg.target, cfg.n_per_class)
    pos["label"] = 1
    ctl = _top_cells(g_gene["parquet"], g_gene["class_col"], cfg.control, cfg.n_per_class)
    ctl["label"] = 0
    print(f"gather (pooled complex): {len(pos)} {cfg.target} + {len(ctl)} {cfg.control}")
    df = pd.concat([pos, ctl], ignore_index=True)
    labels_df = make_labels_df(df, cfg)
    images, labels, _ = materialize_crops(labels_df, cfg, cache_path=crops_cache)
    embs = embed_crops(images, cfg, cache_path=emb_cache)
    return images, embs, labels


def _gather_any(cfg: DirConfig, crops_cache: str, emb_cache: str):
    """Dispatches to the pooled cross-source gather for grain='complex' (see
    `_gather_pooled_complex`), else the standard single-parquet `data.gather`."""
    if cfg.grain == "complex":
        return _gather_pooled_complex(cfg, crops_cache, emb_cache)
    return gather(cfg, crops_cache, emb_cache)


def _score_trajectory(diffae, null_base, traj_list, xT_list, w, cfg, lr_w, lr_b):
    """traj_list[i]: (n_steps, D) embedding trajectory for control cell i. xT_list[i]:
    matching FIXED noise for that cell. Returns (gen (M,A,H,W), scores (M,A), gen_embs (M,A,D))."""
    dev = xT_list[0].device
    gen = []
    for traj, xT in zip(traj_list, xT_list):
        row = []
        for k in range(traj.shape[0]):
            z = traj[k:k + 1].to(dev)
            img = _sample_guided(diffae, xT.clone(), z, null_base, w, cfg)
            row.append(img.cpu().numpy()[0, 0])
        gen.append(row)
    gen = np.array(gen)  # (M,A,H,W)
    flat = gen.reshape(-1, 1, gen.shape[-2], gen.shape[-1]).astype(np.float32)
    gen_embs = embed_crops(flat, cfg, cache_path=None)  # (M*A, D)
    scores = (gen_embs @ lr_w + lr_b).reshape(gen.shape[0], gen.shape[1])
    gen_embs = gen_embs.reshape(gen.shape[0], gen.shape[1], -1)
    return gen, scores, gen_embs


def _mean_diff_trajectory(z0: np.ndarray, d_vec: np.ndarray, gap: float, alphas) -> np.ndarray:
    """(len(alphas), D) — same shape/ordering convention as integrate_flow_bidir's output,
    on the SAME evenly-spaced alpha grid the flow arms use (fixes the v1 x-axis mismatch)."""
    return np.stack([z0 + a * gap * d_vec for a in alphas], axis=0).astype(np.float32)


def _half_axis_metrics(scores: np.ndarray, center: int) -> dict:
    """Split at the NTC index. ko_arm = control->KO (the arm DiffEx actually explains);
    anti_ko_arm = the extrapolated opposite direction (bonus, not the interpretability
    target) — v1 conflated the two into one full-axis number."""
    ko = scores[:, center:]
    anti = scores[:, :center + 1][:, ::-1]

    def _half(h: np.ndarray) -> dict:
        mono = float(np.mean([np.all(np.diff(s) >= -1e-6) for s in h]))
        peak, end, start = h.max(1), h[:, -1], h[:, 0]
        rise = np.clip(peak - start, 1e-6, None)
        overshoot = float(np.mean(np.clip(peak - end, 0, None) / rise))
        return {"frac_nondecreasing": mono, "overshoot_ratio": overshoot,
                "delta": float((end - start).mean())}

    return {"ko_arm": _half(ko), "anti_ko_arm": _half(anti)}


def _faithfulness(gen_embs: np.ndarray, real_kd_embs: np.ndarray, real_ntc_embs: np.ndarray) -> dict:
    """Distance from the generated (re-encoded) trajectory endpoints to the REAL KD/NTC
    embedding clouds — independent of the LR classifier, so it can't be gamed by a
    direction that moves the logit without the embedding actually approaching real cells."""
    kd_c, ntc_c = real_kd_embs.mean(0), real_ntc_embs.mean(0)
    start, end = gen_embs[:, 0], gen_embs[:, -1]
    return {
        "start_dist_to_real_NTC": float(np.linalg.norm(start - ntc_c, axis=1).mean()),
        "start_dist_to_real_KD": float(np.linalg.norm(start - kd_c, axis=1).mean()),
        "end_dist_to_real_KD": float(np.linalg.norm(end - kd_c, axis=1).mean()),
        "end_dist_to_real_NTC": float(np.linalg.norm(end - ntc_c, axis=1).mean()),
    }


def _pixel_localization(gen: np.ndarray, center: int) -> dict:
    """Approximate on-cell vs. off-cell energy of the Delta-pixel map: fraction of
    |Delta| inside a centered disk (proxy for 'the cell body', since crops are built
    around a segmentation centroid) vs. the outer ring. NOT a substitute for a real
    mask-based check (cfg.mask_cell=False by default) — catches the gross border/
    background-artifact failure mode logged 2026-06-18, nothing subtler."""
    H, W = gen.shape[-2], gen.shape[-1]
    yy, xx = np.mgrid[0:H, 0:W]
    r = np.sqrt((yy - H / 2) ** 2 + (xx - W / 2) ** 2) / (min(H, W) / 2)
    inner = r <= 0.5
    diffs = gen - gen[:, center:center + 1]
    energy = np.abs(diffs)
    tot = energy.reshape(*energy.shape[:2], -1).sum(-1)
    frac_inner = energy[..., inner].sum(-1) / (tot + 1e-9)
    return {"mean_frac_energy_inside_cell_proxy": float(frac_inner.mean())}


def _reproducibility(score_list: list) -> dict:
    """Across independent training seeds of the SAME method/target/w. mean_diff needs no
    such check (deterministic by construction) — this is specifically for the flow variants."""
    deltas = [float((s[:, -1] - s[:, 0]).mean()) for s in score_list]
    curves = np.stack([s.mean(0) for s in score_list])
    corr = float(np.corrcoef(curves)[0, 1]) if len(score_list) > 1 else 1.0
    return {"delta_per_seed": deltas, "delta_range": float(max(deltas) - min(deltas)),
            "mean_curve_pairwise_corr": corr}


def _bundle(gen, scores, gen_embs, center, real_kd_embs, real_ntc_embs) -> dict:
    out = {"full_axis_delta": float((scores[:, -1] - scores[:, 0]).mean())}
    out.update(_half_axis_metrics(scores, center))
    out["faithfulness"] = _faithfulness(gen_embs, real_kd_embs, real_ntc_embs)
    out["pixel_localization"] = _pixel_localization(gen, center)
    return out


def _plot_sweep(w: float, mean_sc, flow_sc: dict, cfg, out: Path) -> None:
    """flow_sc: {(coupling, seed): scores}. One line per seed (thin) + per-coupling mean (bold)."""
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["pdf.fonttype"] = 42
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(mean_sc.shape[1])
    for s in mean_sc:
        ax.plot(x, s, color="black", alpha=0.15)
    ax.plot(x, mean_sc.mean(0), color="black", lw=2.5, label="mean_diff (production)")
    colors = {"independent": "tab:orange", "ot": "tab:blue"}
    for coupling, color in colors.items():
        seed_curves = [sc for (c, _seed), sc in flow_sc.items() if c == coupling]
        for sc in seed_curves:
            for s in sc:
                ax.plot(x, s, color=color, alpha=0.08)
        mean_of_seeds = np.stack([sc.mean(0) for sc in seed_curves]).mean(0)
        label = "flow, independent coupling" if coupling == "independent" else "flow, OT coupling"
        ax.plot(x, mean_of_seeds, color=color, lw=2.5, label=f"{label} ({len(seed_curves)} seeds)")
    ax.set_xlabel("trajectory step (anti-KO extreme -> NTC -> KO extreme, evenly spaced)")
    ax.set_ylabel("classifier logit (re-encoded generated image)")
    ax.set_title(f"{cfg.target}: direction-method comparison (w={w:g})")
    ax.axhline(0, color="gray", ls=":")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / f"comparison_{cfg.target}_w{w:g}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_sweep(cfg: DirConfig, out_dir: str, ws=(1.0, 3.0, 5.0), seeds=(0, 1),
             flow_steps: int = 2000, t_max: float = 3.0, n_record: int = 5) -> dict:
    dev = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    out = Path(out_dir); cache = out / "cache"; cache.mkdir(parents=True, exist_ok=True)

    images, embs, labels = _gather_any(cfg, str(cache / "crops.npz"), str(cache / "celldino.npz"))
    real_kd_embs, real_ntc_embs = embs[labels == 1], embs[labels == 0]

    d_vec, lr_w, lr_b, lr_acc = supervised_direction(embs, labels, cfg)
    gap = float(np.linalg.norm(real_kd_embs.mean(0) - real_ntc_embs.mean(0)))
    print(f"[shared] LR acc={lr_acc:.3f}  gap={gap:.2f}")

    ctrl_idx = np.flatnonzero(labels == 0)[: cfg.n_traverse]
    src_embs = embs[ctrl_idx]
    diffae = load_diffae(cfg, dev)
    null_base = diffae.null_emb.detach()[None].to(dev)
    H = cfg.crop_size
    n_steps = 2 * n_record + 1
    center = n_record
    alphas = list(np.linspace(-t_max, t_max, n_steps))  # SAME evenly-spaced grid as the flow arms

    xT_list = [torch.randn(1, 1, H, H, generator=torch.Generator(device=dev).manual_seed(1234 + i),
                           device=dev) for i in range(len(ctrl_idx))]

    mean_diff_traj = [torch.as_tensor(_mean_diff_trajectory(src_embs[i], d_vec, gap, alphas))
                      for i in range(len(ctrl_idx))]

    # Train each flow variant ONCE per seed (training is w-independent; only decode uses w).
    flow_traj = {}
    for coupling in ("independent", "ot"):
        for seed in seeds:
            net = train_flow(embs, labels, dev, steps=flow_steps, seed=seed, coupling=coupling)
            flow_traj[(coupling, seed)] = [
                integrate_flow_bidir(net, torch.as_tensor(src_embs[i:i + 1], dtype=torch.float32),
                                    dev, n_record=n_record, t_max=t_max)
                for i in range(len(ctrl_idx))]

    results: dict = {}
    for w in ws:
        gen, sc_mean, ge_mean = _score_trajectory(diffae, null_base, mean_diff_traj, xT_list, w, cfg, lr_w, lr_b)
        results[f"mean_diff|w{w:g}"] = _bundle(gen, sc_mean, ge_mean, center, real_kd_embs, real_ntc_embs)

        flow_sc_this_w = {}
        for coupling in ("independent", "ot"):
            per_seed_sc = []
            for seed in seeds:
                gen, sc, ge = _score_trajectory(diffae, null_base, flow_traj[(coupling, seed)],
                                                xT_list, w, cfg, lr_w, lr_b)
                tag = f"flow_{coupling}|seed{seed}|w{w:g}"
                results[tag] = _bundle(gen, sc, ge, center, real_kd_embs, real_ntc_embs)
                per_seed_sc.append(sc)
                flow_sc_this_w[(coupling, seed)] = sc
            results[f"flow_{coupling}|repro|w{w:g}"] = _reproducibility(per_seed_sc)

        _plot_sweep(w, sc_mean, flow_sc_this_w, cfg, out)
        print(f"[w={w:g}] done")

    summary = {"target": cfg.target, "grain": cfg.grain, "lr_acc": lr_acc, "gap": gap,
              "ws": list(ws), "seeds": list(seeds), "n_traverse": int(len(ctrl_idx)),
              "results": results}
    (out / "ot_cfm_sweep_metrics.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


def _render_row(axrow, gen_cell: np.ndarray, real_ctrl_img, real_kd_img, center: int, method_label: str,
               alphas, show_titles: bool) -> None:
    """gen_cell: (A,H,W). One row = REAL ctrl | raw generated sweep (no heatmap overlay — just
    the image, easier to actually look at) | REAL KD. Alpha/t value labels only on the top row
    (show_titles) to avoid repeating them 3x; NTC (center) is labeled explicitly."""
    A = gen_cell.shape[0]
    axrow[0].imshow(real_ctrl_img, cmap="gray", vmin=-1, vmax=1, interpolation="nearest")
    axrow[-1].imshow(real_kd_img, cmap="gray", vmin=-1, vmax=1, interpolation="nearest")
    for j in range(A):
        axrow[j + 1].imshow(gen_cell[j], cmap="gray", vmin=-1, vmax=1, interpolation="nearest")
    for ax in axrow:
        ax.axis("off")
    axrow[0].text(-0.08, 0.5, method_label, transform=axrow[0].transAxes, fontsize=9,
                 fontweight="bold", ha="right", va="center", rotation=90)
    if show_titles:
        axrow[0].set_title("REAL ctrl", fontsize=7)
        axrow[-1].set_title("REAL KD", fontsize=7)
        for j in range(A):
            lab = "NTC (α=0)" if j == center else f"{alphas[j]:+.1f}×gap"
            axrow[j + 1].set_title(lab, fontsize=6.5)


def render_montage(cfg: DirConfig, out_dir: str, w: float = 1.0, n_cells: int = 3,
                   flow_steps: int = 2000, seed: int = 0, t_max: float = 3.0, n_record: int = 5) -> list:
    """Actual DECODED IMAGES for mean_diff / flow_independent / flow_ot side by side, per cell —
    everything in run_sweep is classifier-embedding-space evidence; this is the missing visual
    check (DiffEx's actual deliverable is a visual counterfactual, not a logit curve). Reuses the
    SAME cell indices/noise seeds as run_sweep (same `1234+i` convention), so these images are the
    literal pictures behind that run's score curves for cell 0..n_cells-1, IF out_dir's cache
    already exists from a prior run_sweep call on this target (reused read-only, not recomputed).

    w defaults to 1.0, NOT the production "visible GIF" convention of 5.0: the sweep already
    showed w=5 is where monotonicity/overshoot are WORST for every method (the classic w trades
    monotonicity-for-magnitude tradeoff) — using the noisiest guidance setting for a "does this
    look real" sanity check undermines the check. w=1 is where the numbers looked cleanest, so
    it's the fairer setting to actually look at."""
    dev = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    out = Path(out_dir); cache = out / "cache"; cache.mkdir(parents=True, exist_ok=True)

    images, embs, labels = _gather_any(cfg, str(cache / "crops.npz"), str(cache / "celldino.npz"))
    real_kd_embs, real_ntc_embs = embs[labels == 1], embs[labels == 0]
    d_vec, lr_w, lr_b, lr_acc = supervised_direction(embs, labels, cfg)
    gap = float(np.linalg.norm(real_kd_embs.mean(0) - real_ntc_embs.mean(0)))

    ctrl_idx = np.flatnonzero(labels == 0)[:n_cells]
    kd_idx = np.flatnonzero(labels == 1)[:n_cells]
    src_embs = embs[ctrl_idx]
    ctrl_imgs = normalize(images[ctrl_idx])
    kd_imgs = normalize(images[kd_idx]) if len(kd_idx) else ctrl_imgs

    diffae = load_diffae(cfg, dev)
    null_base = diffae.null_emb.detach()[None].to(dev)
    H = cfg.crop_size
    n_steps = 2 * n_record + 1
    center = n_record
    alphas = list(np.linspace(-t_max, t_max, n_steps))

    xT_list = [torch.randn(1, 1, H, H, generator=torch.Generator(device=dev).manual_seed(1234 + i),
                           device=dev) for i in range(len(ctrl_idx))]
    mean_diff_traj = [torch.as_tensor(_mean_diff_trajectory(src_embs[i], d_vec, gap, alphas))
                      for i in range(len(ctrl_idx))]

    gens = {}
    gen, _, _ = _score_trajectory(diffae, null_base, mean_diff_traj, xT_list, w, cfg, lr_w, lr_b)
    gens["mean_diff"] = gen
    for coupling in ("independent", "ot"):
        net = train_flow(embs, labels, dev, steps=flow_steps, seed=seed, coupling=coupling)
        traj = [integrate_flow_bidir(net, torch.as_tensor(src_embs[i:i + 1], dtype=torch.float32),
                                     dev, n_record=n_record, t_max=t_max) for i in range(len(ctrl_idx))]
        gen, _, _ = _score_trajectory(diffae, null_base, traj, xT_list, w, cfg, lr_w, lr_b)
        gens[f"flow_{coupling}"] = gen

    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["pdf.fonttype"] = 42
    import matplotlib.pyplot as plt

    montage_dir = out / "montage"; montage_dir.mkdir(exist_ok=True)
    method_order = ["mean_diff", "flow_independent", "flow_ot"]
    method_labels = {"mean_diff": "mean_diff", "flow_independent": "flow (independent)", "flow_ot": "flow (OT)"}
    ncols = n_steps + 2
    paths = []
    for i in range(len(ctrl_idx)):
        fig, axes = plt.subplots(len(method_order), ncols, figsize=(1.35 * ncols, 1.55 * len(method_order)),
                                 squeeze=False, gridspec_kw={"wspace": 0.0, "hspace": 0.015})
        for row, method in enumerate(method_order):
            _render_row(axes[row], gens[method][i], ctrl_imgs[i, 0], kd_imgs[i % len(kd_imgs), 0],
                       center, method_labels[method], alphas, show_titles=(row == 0))
        fig.suptitle(f"{cfg.target} cell {i} (w={w:g}): mean_diff vs. flow-independent vs. flow-OT — "
                    f"same real cell, same fixed noise, only the direction/trajectory method differs",
                    fontsize=9, y=1.0)
        p = montage_dir / f"montage_{cfg.target}_w{w:g}_cell{i}.png"
        fig.savefig(p, dpi=170, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)
        paths.append(str(p))
        print(f"wrote {p}")
    return paths


def run_v5_score(cfg: DirConfig, out_dir: str, w: float = 1.0, flow_steps: int = 2000,
                 t_max: float = 3.0, n_record: int = 5, seed: int = 0) -> dict:
    """Score mean_diff / flow-independent / flow-OT trajectories with the REAL production
    classifier — Alex Lin's v5 (no-mask) cellstate-set-classifier SetTransformer — instead of
    the ad-hoc per-test LR run_sweep/render_montage use. That LR is fit fresh per comparison
    and is itself linear (same family as mean_diff), so it can't be fully trusted as neutral;
    P(target gene) from the actual classifier this whole project treats as authoritative is a
    real, independent check, not a proxy. Bag = all cfg.n_traverse cells at a given alpha,
    standardized on the alpha=0 (NTC) bag's own mean/std — the documented recipe for scoring
    generated frames (removes the DiffAE domain offset; see diffae/viewer/score_generated.py)."""
    from ..viewer.set_classifier import V5_CKPT_ROOT, V5_RUNS, load_set_classifier, score_bags

    dev = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    out = Path(out_dir); cache = out / "cache"; cache.mkdir(parents=True, exist_ok=True)

    images, embs, labels = _gather_any(cfg, str(cache / "crops.npz"), str(cache / "celldino.npz"))
    real_kd_embs, real_ntc_embs = embs[labels == 1], embs[labels == 0]

    d_vec, lr_w, lr_b, lr_acc = supervised_direction(embs, labels, cfg)
    gap = float(np.linalg.norm(real_kd_embs.mean(0) - real_ntc_embs.mean(0)))

    ctrl_idx = np.flatnonzero(labels == 0)[: cfg.n_traverse]
    src_embs = embs[ctrl_idx]
    diffae = load_diffae(cfg, dev)
    null_base = diffae.null_emb.detach()[None].to(dev)
    H = cfg.crop_size
    n_steps = 2 * n_record + 1
    center = n_record
    alphas = list(np.linspace(-t_max, t_max, n_steps))

    xT_list = [torch.randn(1, 1, H, H, generator=torch.Generator(device=dev).manual_seed(1234 + i),
                           device=dev) for i in range(len(ctrl_idx))]

    run = V5_RUNS[("phase", "geneKO")]
    model, g2i, c2i = load_set_classifier(run=run, device=dev, root=V5_CKPT_ROOT)
    ci = c2i.get("Phase2D", 0)
    if cfg.target not in g2i:
        raise ValueError(f"{cfg.target!r} not in the v5 SetTransformer's 1001-way class map")
    gi = g2i[cfg.target]

    def _v5_curve(gen_embs: np.ndarray) -> list:
        """gen_embs: (M,A,D) -> P(target) at each alpha, bagging all M cells together."""
        embs_by_alpha = [gen_embs[:, a, :] for a in range(gen_embs.shape[1])]
        mu = embs_by_alpha[center].mean(0); sd = embs_by_alpha[center].std(0) + 1e-6
        return [float(score_bags(model, ((e - mu) / sd)[None], channel_idx=ci, device=dev)[0][gi])
               for e in embs_by_alpha]

    results: dict = {}
    mean_diff_traj = [torch.as_tensor(_mean_diff_trajectory(src_embs[i], d_vec, gap, alphas))
                      for i in range(len(ctrl_idx))]
    _, _, ge_md = _score_trajectory(diffae, null_base, mean_diff_traj, xT_list, w, cfg, lr_w, lr_b)
    results["mean_diff"] = _v5_curve(ge_md)

    for coupling in ("independent", "ot"):
        net = train_flow(embs, labels, dev, steps=flow_steps, seed=seed, coupling=coupling)
        traj = [integrate_flow_bidir(net, torch.as_tensor(src_embs[i:i + 1], dtype=torch.float32),
                                     dev, n_record=n_record, t_max=t_max) for i in range(len(ctrl_idx))]
        _, _, ge_f = _score_trajectory(diffae, null_base, traj, xT_list, w, cfg, lr_w, lr_b)
        results[f"flow_{coupling}"] = _v5_curve(ge_f)

    summary = {"target": cfg.target, "w": w, "alphas": alphas, "center_idx": center,
              "n_traverse": int(len(ctrl_idx)), "results": results}
    (out / "v5_score_metrics.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


def run_real_cell_diagnostic(cfg: DirConfig, out_dir: str, n_cells: int = 100) -> dict:
    """Isolates "is it generation quality, or our scoring pipeline?" using ONLY real cells —
    no DiffAE, no direction method, nothing this test has been validating. If OUR pipeline
    scores real cells about as well as Alex's own validation embeddings, the near-zero
    scores we see on GENERATED cells are a generation/domain-gap issue, not a bug in how we
    assemble/standardize bags. If OUR real-cell score is ALSO much lower than Alex's, the
    bag-assembly/standardization step itself needs fixing before generation quality is even
    the right thing to be debugging. Mirrors diffae/viewer/score_generated.py::diagnose(),
    but writes to the caller's out_dir instead of that function's hardcoded shared path."""
    from ..viewer.precompute import _gather_class
    from ..viewer.set_classifier import V5_CKPT_ROOT, V5_RUNS, load_set_classifier, score_bags

    dev = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    run = V5_RUNS[("phase", "geneKO")]
    model, g2i, c2i = load_set_classifier(run=run, device=dev, root=V5_CKPT_ROOT)
    ci = c2i.get("Phase2D", 0)
    if cfg.target not in g2i:
        raise ValueError(f"{cfg.target!r} not in the v5 SetTransformer's class map")
    gi = g2i[cfg.target]

    # Alex's own real validation embeddings — the positive-control ceiling
    av = torch.load(f"/hpc/projects/icd.fast.ops/models/alex_lin_attention/v5/paper_v2_phase/val/{cfg.target}.pt",
                    map_location="cpu")["embeddings"].numpy()
    p_alex = float(score_bags(model, av[:n_cells][None], channel_idx=ci, device=dev)[0][gi])

    # OUR pipeline, on the SAME kind of real cells (top-attention, our materialize+embed_crops)
    _, real_kd = _gather_class(cfg, cfg.target, n_cells)
    p_raw = float(score_bags(model, real_kd[None], channel_idx=ci, device=dev)[0][gi])

    z = np.load("/hpc/projects/icd.fast.ops/models/diffex/viewer_assets_v5/phase/_anchors/NTC/ctrl.npz")
    mu, sd = z["ctrl_embs"].mean(0), z["ctrl_embs"].std(0) + 1e-6
    p_zstd = float(score_bags(model, ((real_kd - mu) / sd)[None], channel_idx=ci, device=dev)[0][gi])

    result = {"target": cfg.target, "n_cells": n_cells, "p_alex": p_alex, "p_our_raw": p_raw,
             "p_our_zstd_on_real_ntc": p_zstd,
             "our_emb_mean": float(real_kd.mean()), "our_emb_std": float(real_kd.std()),
             "alex_emb_mean": float(av.mean()), "alex_emb_std": float(av.std())}
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "real_cell_diagnostic.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="OT-CFM vs independent-CFM vs mean_diff, multi-metric sweep")
    ap.add_argument("--grain", choices=["geneKO", "complex"], default="geneKO")
    ap.add_argument("--target", default="HSPA5")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--ws", type=float, nargs="+", default=[1.0, 3.0, 5.0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--flow-steps", type=int, default=2000)
    ap.add_argument("--montage", action="store_true", help="render visual montage instead of the sweep")
    ap.add_argument("--montage-w", type=float, default=1.0)
    ap.add_argument("--montage-cells", type=int, default=3)
    args = ap.parse_args()

    cfg = DirConfig(grain=args.grain, target=args.target, device="cuda")
    if args.montage:
        render_montage(cfg, args.out_dir, w=args.montage_w, n_cells=args.montage_cells,
                      flow_steps=args.flow_steps)
    else:
        run_sweep(cfg, args.out_dir, ws=tuple(args.ws), seeds=tuple(args.seeds), flow_steps=args.flow_steps)


if __name__ == "__main__":
    main()
