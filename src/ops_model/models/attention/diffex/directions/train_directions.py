"""Stage 2a: train the DirectionBank UNSUPERVISED on CellDINO embeddings."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .losses import decorrelation, infonce_directions


def train_directions(bank, embs: np.ndarray, cfg, dev) -> dict:
    bank = bank.to(dev)
    opt = torch.optim.AdamW(bank.parameters(), lr=cfg.dir_lr)
    loader = DataLoader(
        TensorDataset(torch.as_tensor(embs, dtype=torch.float32)),
        batch_size=cfg.batch_size, shuffle=True, drop_last=True,
    )
    history = []
    for ep in range(cfg.dir_epochs):
        bank.train()
        tot = nce = dec = 0.0
        for (z,) in loader:
            z = z.to(dev)
            dirs = bank(z)
            l_nce = infonce_directions(dirs, cfg.tau)
            l_dec = decorrelation(dirs)
            loss = l_nce + cfg.decorr_weight * l_dec
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss); nce += float(l_nce); dec += float(l_dec)
        nb = len(loader)
        history.append({"epoch": ep, "loss": tot / nb, "infonce": nce / nb, "decorr": dec / nb})
        if ep % 10 == 0 or ep == cfg.dir_epochs - 1:
            print(f"dir epoch {ep:03d}: loss={tot/nb:.4f}  infonce={nce/nb:.4f}  decorr={dec/nb:.4f}")
    return {"history": history}
