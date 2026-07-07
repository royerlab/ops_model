"""Alex Lin's cellstate-set-classifier (SetTransformer) — reconstructed from the checkpoint state_dict
+ config (the training code lives in a private repo). A bag of per-cell CellDINO features → class logits.

Used for the DiffEx viewer's honest 1-of-N score: feed a bag of the N generated cells at a given α →
softmax → P(target class). Report it against the REAL-cell bag score (the achievable ceiling, since
val_acc is only ~0.50 even on real phase cells).

Input space = MASKED CellDINO ViT-L/16, z-standardized on control (Alex's train_ops_zstdcontrol_cdino).
Checkpoints: /hpc/projects/icd.fast.ops/models/alex_lin_attention/v4/wandb/cellstate_set_classifier/<run>/
  miwkg1cy=1K phase geneKO, epzvv0m1=EBI phase, hx6q8byj/ggdfggsn/ciw91el9=fluor.
"""
from __future__ import annotations

import glob

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CKPT_ROOT = "/hpc/projects/icd.fast.ops/models/alex_lin_attention/v4/wandb/cellstate_set_classifier"


class MAB(nn.Module):
    """Pre-norm multihead cross-attention block (query attends to kv) + feed-forward, both residual."""
    def __init__(self, d, heads, d_ff):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.norm_q = nn.LayerNorm(d)
        self.norm_ff = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, d_ff), nn.GELU(), nn.Dropout(0.0), nn.Linear(d_ff, d))

    def forward(self, q, kv):
        a, _ = self.attn(self.norm_q(q), kv, kv, need_weights=False)
        h = q + a
        return h + self.ff(self.norm_ff(h))


class ISAB(nn.Module):
    """Induced set-attention block: inducing points attend to X (cross1), then X attends to that (cross2)."""
    def __init__(self, d, heads, m, d_ff):
        super().__init__()
        self.inducing = nn.Parameter(torch.zeros(1, m, d))
        self.cross1 = MAB(d, heads, d_ff)
        self.cross2 = MAB(d, heads, d_ff)

    def forward(self, x):
        h = self.cross1(self.inducing.expand(x.size(0), -1, -1), x)
        return self.cross2(x, h)


class PMA(nn.Module):
    """Pooling by multihead attention: k seed(s) attend to the set → k pooled vectors."""
    def __init__(self, d, heads, k, d_ff):
        super().__init__()
        self.seeds = nn.Parameter(torch.zeros(1, k, d))
        self.cross = MAB(d, heads, d_ff)

    def forward(self, x):
        return self.cross(self.seeds.expand(x.size(0), -1, -1), x)


class Encoder(nn.Module):
    def __init__(self, d, heads, m, n_layers, d_ff):
        super().__init__()
        self.layers = nn.ModuleList([ISAB(d, heads, m, d_ff) for _ in range(n_layers)])
        self.pool = PMA(d, heads, 1, d_ff)
        self.final_norm = nn.LayerNorm(d)

    def forward(self, x):
        for lyr in self.layers:
            x = lyr(x)
        return self.final_norm(self.pool(x).squeeze(1))


class CosineHead(nn.Module):
    """Cosine classifier: temperature-scaled cosine similarity between the pooled vector and class prototypes."""
    def __init__(self, d, n):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(n, d))
        self.log_scale = nn.Parameter(torch.zeros(()))

    def forward(self, x):
        return self.log_scale.exp() * (F.normalize(x, dim=-1) @ F.normalize(self.weight, dim=-1).t())


class SetClassifier(nn.Module):
    def __init__(self, d=512, heads=4, m=32, n_layers=2, n_classes=1001, n_channels=1, d_ff=2048):
        super().__init__()
        self.input_proj = nn.Linear(1024, d)
        self.channel_embeddings = nn.Embedding(n_channels, d)
        self.concat_proj = nn.Linear(2 * d, d)
        self.encoder = Encoder(d, heads, m, n_layers, d_ff)
        self.head = nn.Sequential(nn.Identity(), CosineHead(d, n_classes))

    def forward(self, feats, channel_idx):
        x = self.input_proj(feats)                                   # (B,N,d)
        ce = self.channel_embeddings(channel_idx)[:, None, :].expand(-1, x.size(1), -1)
        x = self.concat_proj(torch.cat([x, ce], -1))                 # concat channel conditioning
        return self.head(self.encoder(x))                            # (B, n_classes)


def load_set_classifier(run="miwkg1cy", device="cpu"):
    """Load a checkpoint → (model, gene_to_idx, channel_to_idx). Architecture read from the bundled config."""
    ckpt = torch.load(glob.glob(f"{CKPT_ROOT}/{run}/*.pt")[0], map_location=device)
    mc = ckpt["config"]["model"] if "model" in ckpt.get("config", {}) else ckpt["config"]
    m = SetClassifier(d=mc["d_model"], heads=mc["n_heads"], m=mc["n_inducing_cell"],
                      n_layers=mc["n_layers_cell"], n_classes=len(ckpt["gene_to_idx"]),
                      n_channels=len(ckpt["channel_to_idx"]), d_ff=mc.get("d_ff") or 4 * mc["d_model"])
    m.load_state_dict(ckpt["model_state_dict"])
    m.eval().to(device)
    return m, ckpt["gene_to_idx"], ckpt["channel_to_idx"]


@torch.no_grad()
def score_bags(model, feats, channel_idx=0, device="cpu"):
    """feats: (B, N, 1024) bags → softmax probabilities (B, n_classes)."""
    f = torch.as_tensor(np.asarray(feats), dtype=torch.float32, device=device)
    ci = torch.full((f.size(0),), int(channel_idx), dtype=torch.long, device=device)
    return F.softmax(model(f, ci), dim=-1).cpu().numpy()
