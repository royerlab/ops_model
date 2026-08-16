"""Generic train/eval loop shared by options B and C.

Success criterion (PoC): held-out AUROC clearly > 0.5. Because the split is
grouped by experiment, a high AUROC means the classifier separates the gene on
biology that generalizes across experiments, not on a per-plate confound.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset


def _resolve_device(name: str) -> torch.device:
    if name.startswith("cuda") and not torch.cuda.is_available():
        print("[device] cuda requested but unavailable -> cpu")
        return torch.device("cpu")
    return torch.device(name)


@torch.no_grad()
def evaluate(model, X, y, batch_size, dev) -> float:
    model.eval()
    probs = []
    for i in range(0, len(X), batch_size):
        xb = torch.as_tensor(X[i:i + batch_size]).to(dev)
        probs.append(torch.softmax(model(xb), -1)[:, 1].cpu().numpy())
    return float(roc_auc_score(y, np.concatenate(probs)))


def train_classifier(model, Xtr, ytr, Xva, yva, Xte, yte, cfg) -> dict:
    dev = _resolve_device(cfg.device)
    model = model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    crit = nn.CrossEntropyLoss()

    tr = DataLoader(
        TensorDataset(torch.as_tensor(Xtr), torch.as_tensor(ytr)),
        batch_size=cfg.batch_size, shuffle=True,
    )
    best = {"val_auroc": -1.0, "train_auroc": -1.0, "epoch": -1, "state": None}
    history = []
    for ep in range(cfg.epochs):
        model.train()
        ep_loss = 0.0
        for xb, yb in tr:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            ep_loss += float(loss) * len(xb)
        ep_loss /= len(Xtr)
        # train + val AUROC each epoch -> watch the gap (overfit vs underfit)
        tr_auroc = evaluate(model, Xtr, ytr, cfg.batch_size, dev)
        va_auroc = evaluate(model, Xva, yva, cfg.batch_size, dev)
        history.append({"epoch": ep, "train_loss": ep_loss,
                        "train_auroc": tr_auroc, "val_auroc": va_auroc})
        if va_auroc > best["val_auroc"]:
            best = {
                "val_auroc": va_auroc, "train_auroc": tr_auroc, "epoch": ep,
                "state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            }
        print(f"epoch {ep:02d}: loss={ep_loss:.4f}  train_auroc={tr_auroc:.4f}  "
              f"val_auroc={va_auroc:.4f}  (best val {best['val_auroc']:.4f})")
    best["history"] = history

    # clean held-out number: reload the val-selected best, score test ONCE
    if best["state"] is not None:
        model.load_state_dict(best["state"])
    best["test_auroc"] = evaluate(model, Xte, yte, cfg.batch_size, dev)
    print(f"selected epoch {best['epoch']}: val={best['val_auroc']:.4f}  "
          f"test={best['test_auroc']:.4f}")
    return best
