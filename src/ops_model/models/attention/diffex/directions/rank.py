"""Stage 2b: rank the K directions by control-vs-target classifier score shift.

Fit a logistic regression (control=0, target=1) on the embeddings (post-hoc, never
used during 2a). For each trained direction, shift embeddings ±alpha and measure the
mean signed change in classifier logit. The direction with the largest |shift| is the
target axis. Returns (best_k, shifts, lr_weight, lr_bias)."""
from __future__ import annotations

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression


def supervised_direction(embs: np.ndarray, labels: np.ndarray, cfg):
    """Deterministic control→KD direction (plan C primary). No training, no seeds:
    reproducible by construction, and +α → toward KD by construction.

      - 'mean_diff'  : normalize(mean(KD emb) − mean(control emb))  [centroid vector]
      - 'lr_weight'  : normalize(logistic-regression weight)         [covariance-aware]

    Returns (d_unit (D,), lr_weight (D,), lr_bias, lr_acc). The LR is kept for the
    re-encoded score verification only.
    """
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(embs, labels)
    lr_w = clf.coef_[0].astype(np.float32)
    acc = float(clf.score(embs, labels))
    if cfg.direction_method == "lr_weight":
        d = lr_w.copy()
    else:  # mean_diff (default)
        d = (embs[labels == 1].mean(0) - embs[labels == 0].mean(0)).astype(np.float32)
    d = d / (np.linalg.norm(d) + 1e-9)
    print(f"[direction] supervised '{cfg.direction_method}' (deterministic); LR acc={acc:.3f}")
    return d.astype(np.float32), lr_w, float(clf.intercept_[0]), acc


def rank_directions(bank, embs: np.ndarray, labels: np.ndarray, cfg, dev):
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(embs, labels)
    w = torch.as_tensor(clf.coef_[0], dtype=torch.float32, device=dev)
    acc = float(clf.score(embs, labels))

    bank = bank.to(dev).eval()
    n = min(1024, len(embs))
    z = torch.as_tensor(embs[:n], dtype=torch.float32, device=dev)
    a = cfg.rank_alpha
    shifts = []
    with torch.no_grad():
        for k in range(bank.K):
            d = bank.direction(z, k)               # (n,D)
            shift = (((z + a * d) @ w) - ((z - a * d) @ w)).mean().item()
            shifts.append(shift)
    best_k = int(max(range(bank.K), key=lambda k: abs(shifts[k])))
    # orient so +α always increases the KD score (the MLP's sign is arbitrary):
    # +α → more KO-like, −α → more control-like, consistently across targets.
    sign = 1.0 if shifts[best_k] >= 0 else -1.0
    print(f"[rank] LR train acc={acc:.3f}; per-direction score shift: "
          + ", ".join(f"{k}:{s:+.3f}" for k, s in enumerate(shifts)))
    print(f"[rank] selected direction {best_k} (|shift|={abs(shifts[best_k]):.3f}), "
          f"orient sign={sign:+.0f} (+α = toward KO)")
    return best_k, shifts, clf.coef_[0].astype(np.float32), float(clf.intercept_[0]), acc, sign
