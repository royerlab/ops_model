"""Build & save a validated OT-coupled flow-matching field for one target (gene or EBI
complex) — the production build-out of Workstream A after the multi-metric + pooled-complex
validation in `ot_cfm_test.py` (see FLOW_OT_CELLSTATE_PROGRAM.md build log). Deliberately
lighter than the validation sweep: ONE seed, ONE guidance scale (w=1.0, the setting where
monotonicity/overshoot were cleanest for every method in the sweep), and only the OT-coupled
flow variant — independent coupling is the already-rejected baseline, not rebuilt at scale.

Per target, saves:
  - flow_ot.pt      trained FlowNet state_dict (the field itself)
  - mean_diff.npz   deterministic direction vector + LR weights/bias (comparison baseline,
                    ~free to keep since every target already needs the LR fit for scoring)
  - metrics.json    ko-arm delta/monotonicity/overshoot + faithfulness for BOTH methods, plus
                    `flow_advantage_ko_delta_ratio` (flow-OT's ko-arm delta / mean_diff's) — a
                    screening signal: near/above 1.0 is where a target's KO response looks
                    genuinely multimodal/nonlinear rather than the single-clean-cluster case
                    mean_diff already handles optimally (see the pooled-complex result, where
                    this ratio was 0.90-0.95 and flow-OT matched/beat mean_diff on structure and
                    faithfulness, vs. 0.80-0.92 on clean single genes HSPA5/TIMM23).

    python -m ops_model.models.interpretability.diffae.directions.flow_field \
        --grain geneKO --target HSPA5 --out-dir /path/to/out
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ..config import DirConfig
from .flow import integrate_flow_bidir, train_flow
from ..ot_validation.ot_cfm_test import _faithfulness, _gather_any, _half_axis_metrics, _mean_diff_trajectory, _score_trajectory
from ..rank import supervised_direction
from ..traverse import load_diffae


def build_flow_field(cfg: DirConfig, out_dir: str, seed: int = 0, w: float = 1.0,
                     flow_steps: int = 2000, t_max: float = 3.0, n_record: int = 5) -> dict:
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

    def _bundle(scores, gen_embs):
        b = {"full_axis_delta": float((scores[:, -1] - scores[:, 0]).mean())}
        b.update(_half_axis_metrics(scores, center))
        b["faithfulness"] = _faithfulness(gen_embs, real_kd_embs, real_ntc_embs)
        return b

    # mean_diff — the deterministic comparison baseline every field is judged against
    mean_diff_traj = [torch.as_tensor(_mean_diff_trajectory(src_embs[i], d_vec, gap, alphas))
                      for i in range(len(ctrl_idx))]
    _, sc_md, ge_md = _score_trajectory(diffae, null_base, mean_diff_traj, xT_list, w, cfg, lr_w, lr_b)
    m_mean_diff = _bundle(sc_md, ge_md)

    # the field itself: OT-coupled flow matching (the validated method)
    net = train_flow(embs, labels, dev, steps=flow_steps, seed=seed, coupling="ot")
    traj = [integrate_flow_bidir(net, torch.as_tensor(src_embs[i:i + 1], dtype=torch.float32),
                                 dev, n_record=n_record, t_max=t_max) for i in range(len(ctrl_idx))]
    _, sc_fo, ge_fo = _score_trajectory(diffae, null_base, traj, xT_list, w, cfg, lr_w, lr_b)
    m_flow_ot = _bundle(sc_fo, ge_fo)

    torch.save(net.state_dict(), out / "flow_ot.pt")
    np.savez(out / "mean_diff.npz", d_vec=d_vec, lr_w=lr_w, lr_b=lr_b, gap=gap)

    md_delta, fo_delta = m_mean_diff["ko_arm"]["delta"], m_flow_ot["ko_arm"]["delta"]
    flow_advantage = float(fo_delta / md_delta) if abs(md_delta) > 1e-6 else None

    summary = {
        "target": cfg.target, "grain": cfg.grain, "lr_acc": lr_acc, "gap": gap, "w": w,
        "seed": seed, "n_traverse": int(len(ctrl_idx)),
        "mean_diff": m_mean_diff, "flow_ot": m_flow_ot,
        "flow_advantage_ko_delta_ratio": flow_advantage,
    }
    (out / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(f"[{cfg.grain}/{cfg.target}] flow_advantage_ko_delta_ratio={flow_advantage}")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a validated OT-coupled flow field for one target")
    ap.add_argument("--grain", choices=["geneKO", "complex"], required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--w", type=float, default=1.0)
    ap.add_argument("--flow-steps", type=int, default=2000)
    args = ap.parse_args()
    cfg = DirConfig(grain=args.grain, target=args.target, device="cuda")
    print(json.dumps(build_flow_field(cfg, args.out_dir, seed=args.seed, w=args.w,
                                      flow_steps=args.flow_steps), indent=2))


if __name__ == "__main__":
    main()
