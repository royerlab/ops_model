"""Option C features: embed the SAME phase crops with the local CellDINO encoder.

Uses `ops_model.models.cell_dino.CellDinoModel` (channel-adaptive DINO ViT-L/16,
resize 224 + per-image z-score, in_channels=1) — the same encoder that produced
Alex's feature dumps, so option C lives in the classifier's true feature space.
Requires a GPU (CellDinoModel runs on cuda).
"""
from __future__ import annotations

import os

import numpy as np
import torch


def embed_crops(images: np.ndarray, cfg, cache_path=None) -> np.ndarray:
    """images (N,1,H,W) float32 -> CellDINO features (N, D). Cached to .npz."""
    if cache_path and os.path.exists(cache_path):
        feats = np.load(cache_path)["features"]
        print(f"[cache] celldino features <- {cache_path}  {feats.shape}")
        return feats

    from ops_model.models.cell_dino import CellDinoModel

    model = CellDinoModel(z_score=True)  # loads checkpoint + moves to cuda
    feats = []
    with torch.inference_mode():
        for i in range(0, len(images), cfg.batch_size):
            x = torch.as_tensor(images[i:i + cfg.batch_size])  # (B,1,H,W)
            out = model.extract_features({"data": x})  # preprocess + forward (cuda)
            feats.append(out.float().cpu().numpy())
    features = np.concatenate(feats).astype(np.float32)
    if features.ndim != 2:
        features = features.reshape(features.shape[0], -1)

    if cache_path:
        np.savez(cache_path, features=features)
        print(f"[cache] celldino features -> {cache_path}  {features.shape}")
    return features
