from tqdm import tqdm
from pathlib import Path
import yaml

import torch
import numpy as np
import pandas as pd
import anndata as ad
import torch.nn.functional as F

from ops_model.data.paths import OpsPaths


def mean_similarity(
    adata,
    n_samples=10_000_000,
    batch_size=10000,
):
    """
    Compute mean and std of pairwise cosine similarities.

    Args:
        adata: AnnData object with embeddings in .X
        use_sampling: If True, sample random pairs instead of computing all pairs
        n_samples: Number of pairs to sample (only used if use_sampling=True)
        batch_size: Batch size for processing (only used if use_sampling=False)
    """
    embeddings = torch.tensor(adata.X).cuda()
    x = F.normalize(embeddings, dim=1)
    n = x.shape[0]

    # Sampling approach: much faster for large datasets
    # Sample random pairs and compute their similarities
    n_samples = min(n_samples, n * (n - 1) // 2)  # Don't sample more than total pairs

    # Generate random pairs
    idx_i = torch.randint(0, n, (n_samples,), device="cuda")
    idx_j = torch.randint(0, n, (n_samples,), device="cuda")

    # Ensure i != j
    mask = idx_i == idx_j
    idx_j[mask] = (idx_j[mask] + 1) % n

    # Compute similarities for sampled pairs in batches
    similarities = []
    for start in tqdm(range(0, n_samples, batch_size)):
        end = min(start + batch_size, n_samples)
        batch_i = x[idx_i[start:end]]
        batch_j = x[idx_j[start:end]]
        sim = (batch_i * batch_j).sum(dim=1)
        similarities.append(sim.cpu())

    similarities = torch.cat(similarities)
    mean_similarity = similarities.mean().item()
    std_similarity = similarities.std().item()

    return mean_similarity, std_similarity


def alignment_and_uniformity(adata, n_uniformity_samples=1_000_000, batch_size=10000):
    """
    Compute alignment and uniformity metrics for embeddings.

    Code adapted from:
    title={Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere},
    author={Wang, Tongzhou and Isola, Phillip},
    booktitle={International Conference on Machine Learning},
    organization={PMLR},
    pages={9929--9939},
    year={2020}

    Args:
        adata: AnnData object with embeddings and gene_int labels
        n_uniformity_samples: Number of random pairs to sample for uniformity
        batch_size: Batch size for processing
    """
    gene_int_list = adata.obs["label_int"].unique().tolist()
    x = []  # positive pair i
    y = []  # positive pair j
    for i in tqdm(gene_int_list):
        single_gene_embs = adata.X[adata.obs["label_int"] == i]
        x += [single_gene_embs[j] for j in range(len(single_gene_embs) - 1)]
        y += [
            single_gene_embs[z]
            for z in np.random.permutation(np.arange(len(single_gene_embs) - 1))
        ]

    x = torch.tensor(np.asarray(x)).cuda()
    y = torch.tensor(np.asarray(y)).cuda()

    # Compute alignment (all positive pairs)
    alignment = (x - y).norm(p=2, dim=1).pow(2).mean().item()

    # Compute uniformity using sampling to avoid OOM
    n = x.shape[0]
    n_samples = min(n_uniformity_samples, n * (n - 1) // 2)

    # Sample random pairs for uniformity
    idx_i = torch.randint(0, n, (n_samples,), device="cuda")
    idx_j = torch.randint(0, n, (n_samples,), device="cuda")

    # Ensure i != j
    mask = idx_i == idx_j
    idx_j[mask] = (idx_j[mask] + 1) % n

    # Compute pairwise distances for sampled pairs in batches
    uniformity_vals = []
    for start in tqdm(range(0, n_samples, batch_size)):
        end = min(start + batch_size, n_samples)
        batch_i = x[idx_i[start:end]]
        batch_j = x[idx_j[start:end]]
        # Compute squared L2 distance
        dist_sq = (batch_i - batch_j).norm(p=2, dim=1).pow(2)
        uniformity_vals.append(dist_sq.cpu())

    uniformity_vals = torch.cat(uniformity_vals)
    uniformity = uniformity_vals.mul(-2).exp().mean().log().item()

    return alignment, uniformity


if __name__ == "__main__":
    pass
