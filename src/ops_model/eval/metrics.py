"""Shared metric helpers for the evaluation suite."""

from __future__ import annotations

import numpy as np
import anndata as ad
from sklearn.metrics.pairwise import cosine_similarity


def mean_cosine_sim_within_groups(
    adata: ad.AnnData, groups: list[list[int]]
) -> float:
    """Compute mean pairwise cosine similarity within each group of embedding indices.

    Parameters
    ----------
    adata : AnnData
        AnnData object with embedding matrix in .X.
    groups : list of list of int
        Groups of integer indices into adata.X. Groups with fewer than 2 members
        are skipped.

    Returns
    -------
    float
        Mean pairwise cosine similarity averaged over all groups. Returns NaN
        if no group has 2 or more members.
    """
    X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    per_group_sims: list[float] = []
    for indices in groups:
        if len(indices) < 2:
            continue
        embeddings = X[indices]
        sim_matrix = cosine_similarity(embeddings)
        upper = sim_matrix[np.triu_indices(len(indices), k=1)]
        per_group_sims.append(float(upper.mean()))
    if not per_group_sims:
        return float("nan")
    return float(np.mean(per_group_sims))
