# %%
from tqdm import tqdm
from pathlib import Path

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd

from ops_model.data.embeddings.utils import load_adata


def embedding_spread(adata, label, plot=True):
    """
    Calculates the mean cosine similarity from each embedding to the centroid of the class
    for all embeddings with the specified label in adata.obs['label_str'].

    Returns:
        - the average cosine similarity to the centroid
        - a histogram of the cosine similarities

    """

    # Filter observations with the specified label
    mask = adata.obs["label_str"] == label
    embeddings = torch.tensor(adata.X[mask]).cuda()

    if embeddings.shape[0] < 2:
        print(f"Not enough samples for label '{label}': {embeddings.shape[0]}")
        return None, None

    # Normalize embeddings for cosine similarity computation
    x = F.normalize(embeddings, dim=1)

    # Compute the centroid (mean of normalized embeddings, then re-normalize)
    centroid = x.mean(dim=0, keepdim=True)
    centroid = F.normalize(centroid, dim=1)

    # Compute cosine similarity from each embedding to the centroid
    cosine_similarities = (x * centroid).sum(dim=1).cpu()
    mean_similarity = cosine_similarities.mean().item()
    std_similarity = cosine_similarities.std().item()

    if plot:
        # Create histogram
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(cosine_similarities.numpy(), bins=50, edgecolor="black", alpha=0.7)
        ax.set_xlabel("Cosine Similarity to Centroid")
        ax.set_ylabel("Frequency")
        ax.set_title(f"Cosine Similarity to Centroid for Label: {label}")
        ax.axvline(
            mean_similarity,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {mean_similarity:.4f}",
        )
        ax.legend()
        plt.tight_layout()
        plt.close(fig)

        return mean_similarity, std_similarity, fig

    return mean_similarity, std_similarity, None


def embedding_spread_all_labels(adata, min_samples=2):
    """
    Calculates the embedding spread (mean cosine similarity to centroid) for all labels
    in the adata object.

    Args:
        adata: AnnData object with embeddings in .X
        min_samples: Minimum number of samples required to compute spread (default: 2)

    Returns:
        results_dict: Dictionary mapping label names to (mean_similarity, std_similarity) tuples
        sorted_results: List of (label, mean_similarity, std_similarity) tuples sorted by mean similarity
        fig: Matplotlib figure with histogram of mean similarities across all labels
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Get all unique labels
    unique_labels = adata.obs["label_str"].unique()

    results_dict = {}

    for label in tqdm(unique_labels):
        # Check if label has enough samples
        n_samples = (adata.obs["label_str"] == label).sum()

        if n_samples < min_samples:
            print(f"Skipping label '{label}': only {n_samples} samples")
            continue

        # Compute embedding spread for this label
        mean_sim, std_sim, _ = embedding_spread(adata, label, plot=False)

        if mean_sim is not None:
            results_dict[label] = (mean_sim, std_sim)

    # Sort by mean similarity (highest similarity = most compact clusters first)
    sorted_results = sorted(
        [(label, mean, std) for label, (mean, std) in results_dict.items()],
        key=lambda x: x[1],
        reverse=True,
    )

    # Create histogram of mean similarities
    mean_similarities = [mean for _, mean, _ in sorted_results]
    overall_mean = np.mean(mean_similarities)

    top_10_tightest = sorted_results[:10]
    top_10_diffuse = sorted_results[-10:]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(mean_similarities, bins=30, edgecolor="black", alpha=0.7)
    ax.set_xlabel("Mean Cosine Similarity to Centroid")
    ax.set_ylabel("Number of Labels")
    ax.set_title("Distribution of Embedding Spread Across Labels")
    ax.axvline(
        overall_mean,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {overall_mean:.4f}",
    )
    ax.legend()
    plt.tight_layout()

    return top_10_tightest, top_10_diffuse, sorted_results, fig


def cosine_similarity_to_reference(adata, reference_label):
    """
    Calculates the cosine similarity from each label's embedding to a reference label's embedding.
    Assumes each label has exactly one embedding in the adata object.

    Args:
        adata: AnnData object with embeddings in .X (one embedding per label)
        reference_label: The reference label to compare against

    Returns:
        similarities_dict: Dictionary mapping label names to cosine similarities from reference
        sorted_labels: List of (label, similarity) tuples sorted by similarity (most similar first)
        fig: Matplotlib figure with histogram of similarities
    """

    # Get reference embedding
    ref_mask = adata.obs["label_str"] == reference_label
    if ref_mask.sum() == 0:
        raise ValueError(f"Reference label '{reference_label}' not found in adata")
    if ref_mask.sum() > 1:
        raise ValueError(
            f"Reference label '{reference_label}' has multiple embeddings, expected 1"
        )

    ref_embedding = torch.tensor(adata.X[ref_mask]).cuda()
    ref_x = F.normalize(ref_embedding, dim=1)

    # Get all embeddings and labels
    all_embeddings = torch.tensor(adata.X).cuda()
    all_x = F.normalize(all_embeddings, dim=1)

    # Compute cosine similarity between reference and all embeddings
    cosine_similarities = (ref_x @ all_x.T).squeeze().cpu()

    # Create dictionary mapping labels to similarities
    similarities_dict = {}
    for idx, label in enumerate(adata.obs["label_str"]):
        similarities_dict[label] = cosine_similarities[idx].item()

    # Sort labels by similarity (highest first)
    sorted_labels = sorted(similarities_dict.items(), key=lambda x: x[1], reverse=True)

    # Calculate mean similarity
    mean_similarity = cosine_similarities.mean().item()

    top_10_closest = sorted_labels[1:11]
    top_10_furthest = sorted_labels[-10:]

    # Create histogram
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(cosine_similarities.numpy(), bins=50, edgecolor="black", alpha=0.7)
    ax.set_xlabel("Cosine Similarity")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Cosine Similarity Distribution to Reference: {reference_label}")
    ax.axvline(
        mean_similarity,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {mean_similarity:.4f}",
    )
    ax.legend()
    plt.tight_layout()

    return top_10_closest, top_10_furthest, sorted_labels, fig


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

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(similarities, bins=100, edgecolor="black", alpha=0.7)
    ax.set_xlabel("Cosine Similarity")
    ax.set_ylabel("Number of Labels")
    ax.set_title("Cosine Similarity Distribution Across Labels")
    ax.axvline(
        mean_similarity,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {mean_similarity:.4f}",
    )
    ax.legend()
    plt.tight_layout()
    plt.close(fig)

    return mean_similarity, std_similarity, fig


def mean_similarity_within_labels(
    adata,
    n_samples_per_label=10000,
    batch_size=10000,
):
    """
    Compute mean and std of pairwise cosine similarities for pairs within the same label.
    Only computes similarities between embeddings that share the same label in adata.obs['label_str'].

    Args:
        adata: AnnData object with embeddings in .X
        n_samples_per_label: Number of pairs to sample per label
        batch_size: Batch size for processing

    Returns:
        mean_similarity: Mean cosine similarity across all within-label pairs
        std_similarity: Standard deviation of cosine similarities
    """
    import numpy as np

    embeddings = torch.tensor(adata.X).cuda()
    x = F.normalize(embeddings, dim=1)

    # Get unique labels
    unique_labels = adata.obs["label_str"].unique()

    all_similarities = []

    for label in tqdm(unique_labels, desc="Processing labels"):
        # Get indices for this label
        label_mask = adata.obs["label_str"] == label
        label_indices = np.where(label_mask)[0]
        n_label = len(label_indices)

        # Skip labels with only one sample
        if n_label < 2:
            continue

        # Determine number of pairs to sample for this label
        n_possible_pairs = n_label * (n_label - 1) // 2
        n_samples = min(n_samples_per_label, n_possible_pairs)

        # Convert label indices to tensor
        label_indices_tensor = torch.tensor(label_indices, device="cuda")

        # Generate random pairs within this label
        idx_i = torch.randint(0, n_label, (n_samples,), device="cuda")
        idx_j = torch.randint(0, n_label, (n_samples,), device="cuda")

        # Ensure i != j
        mask = idx_i == idx_j
        idx_j[mask] = (idx_j[mask] + 1) % n_label

        # Map to actual indices in the full dataset
        actual_idx_i = label_indices_tensor[idx_i]
        actual_idx_j = label_indices_tensor[idx_j]

        # Compute similarities for sampled pairs in batches
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch_i = x[actual_idx_i[start:end]]
            batch_j = x[actual_idx_j[start:end]]
            sim = (batch_i * batch_j).sum(dim=1)
            all_similarities.append(sim.cpu())

    # Combine all similarities
    all_similarities = torch.cat(all_similarities)
    mean_similarity = all_similarities.mean().item()
    std_similarity = all_similarities.std().item()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(all_similarities, bins=100, edgecolor="black", alpha=0.7)
    ax.set_xlabel("Cosine Similarity")
    ax.set_ylabel("Number of Labels")
    ax.set_title("Cosine Similarity Distribution Within Labels")
    ax.axvline(
        mean_similarity,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {mean_similarity:.4f}",
    )
    ax.legend()
    plt.tight_layout()
    plt.close(fig)

    return mean_similarity, std_similarity, fig


if __name__ == "__main__":
    adata_path = "/hpc/projects/intracellular_dashboard/ops/ops0031_20250424/3-assembly/dynaclr_features"
    adata_cells, adata_guides, adata_genes = load_adata(adata_path)

    mean_similarity_all, std_similarity_all, across_labels_fig = mean_similarity(
        adata_cells, n_samples=1_000_000
    )
    mean_similarity_within, std_similarity_within, within_labels_fig = (
        mean_similarity_within_labels(adata_cells, n_samples_per_label=10_000)
    )

    top_10_closest, top_10_furthest, sorted_labels, fig = (
        cosine_similarity_to_reference(adata_genes, reference_label="NTC")
    )
    top_10_tightest, top_10_diffuse, spread_sorted_labels, spread_fig = (
        embedding_spread_all_labels(adata_cells, min_samples=2)
    )

    print("Mean Cosine Similarity (All Cells):", f"{mean_similarity_all:.4f}")
    print("Std Dev Cosine Similarity (All Cells):", f"{std_similarity_all:.4f}")
    print("\nTop 10 Closest Labels to NTC:")
    for label, sim in top_10_closest:
        print(f"  {label}: {sim:.4f}")
    print("\nTop 10 Furthest Labels from NTC:")
    for label, sim in top_10_furthest:
        print(f"  {label}: {sim:.4f}")
    print("\nTop 10 Tightest Embedding Spreads:")
    for label, mean_sim, std_sim in top_10_tightest:
        print(f"  {label}: Mean Similarity = {mean_sim:.4f}, Std Dev = {std_sim:.4f}")
    print("\nTop 10 Most Diffuse Embedding Spreads:")
    for label, mean_sim, std_sim in top_10_diffuse:
        print(f"  {label}: Mean Similarity = {mean_sim:.4f}, Std Dev = {std_sim:.4f}")


# ==============================================================================
# Report Directory Integration - Save metrics to centralized reports
# ==============================================================================


def save_embedding_spread_metrics(
    adata: "anndata.AnnData",
    report_dir: str,
    min_samples: int = 2,
    top_n: int = 10,
) -> tuple:
    """
    Compute embedding spread for all labels and save top/bottom tables to report directory.

    Generates two CSV files:
    - embedding_spread_top10.csv: 10 tightest clusters (highest mean similarity)
    - embedding_spread_bottom10.csv: 10 most diffuse clusters (lowest mean similarity)

    Args:
        adata: AnnData object with embeddings
        report_dir: Path to report directory
        min_samples: Minimum samples required per label
        top_n: Number of top/bottom entries to save (default: 10)

    Returns:
        Tuple of (top_10_df, bottom_10_df) DataFrames
    """
    print("Computing embedding spread for all labels...")
    top_10_tightest, top_10_diffuse, sorted_results, fig = embedding_spread_all_labels(
        adata, min_samples=min_samples
    )

    # Get cell counts
    label_counts = adata.obs["label_str"].value_counts()

    # Create DataFrame for top 10 tightest (highest similarity = most compact)
    top_10_df = pd.DataFrame(
        top_10_tightest[:top_n],
        columns=["label_str", "mean_similarity_to_centroid", "std_similarity"],
    )
    top_10_df["n_cells"] = top_10_df["label_str"].map(label_counts)

    # Create DataFrame for bottom 10 most diffuse (lowest similarity = most spread out)
    bottom_10_df = pd.DataFrame(
        top_10_diffuse[:top_n],
        columns=["label_str", "mean_similarity_to_centroid", "std_similarity"],
    )
    bottom_10_df["n_cells"] = bottom_10_df["label_str"].map(label_counts)

    # Save to CSVs
    report_dir = Path(report_dir)

    top_path = report_dir / "metrics" / "embedding_spread_top10.csv"
    top_10_df.to_csv(top_path, index=False)
    print(f"Saved top 10 tightest clusters to: {top_path}")

    bottom_path = report_dir / "metrics" / "embedding_spread_bottom10.csv"
    bottom_10_df.to_csv(bottom_path, index=False)
    print(f"Saved bottom 10 most diffuse clusters to: {bottom_path}")

    # Save figure
    if fig is not None:
        fig_path = report_dir / "plots" / "embedding_spread_distribution.png"
        fig.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved embedding spread plot to: {fig_path}")

    return top_10_df, bottom_10_df


def save_similarity_to_reference_metrics(
    adata: "anndata.AnnData",
    reference_label: str,
    report_dir: str,
    top_n: int = 10,
) -> tuple:
    """
    Compute cosine similarity to reference label and save top/bottom tables to report directory.

    Generates two CSV files:
    - similarity_to_{reference}_top10.csv: 10 most similar genes (highest similarity)
    - similarity_to_{reference}_bottom10.csv: 10 least similar genes (lowest similarity)

    Args:
        adata: AnnData object with gene-level embeddings
        reference_label: Reference label to compare against (e.g., "NTC")
        report_dir: Path to report directory
        top_n: Number of top/bottom entries to save (default: 10)

    Returns:
        Tuple of (top_10_df, bottom_10_df) DataFrames
    """
    print(f"Computing cosine similarity to reference label: {reference_label}...")
    top_10_closest, top_10_furthest, sorted_labels, fig = (
        cosine_similarity_to_reference(adata, reference_label)
    )

    # Create DataFrame for top 10 most similar (highest similarity = closest to reference)
    top_10_df = pd.DataFrame(
        top_10_closest[:top_n],
        columns=["label_str", f"similarity_to_{reference_label}"],
    )

    # Create DataFrame for bottom 10 least similar (lowest similarity = furthest from reference)
    bottom_10_df = pd.DataFrame(
        top_10_furthest[:top_n],
        columns=["label_str", f"similarity_to_{reference_label}"],
    )

    # Save to CSVs
    report_dir = Path(report_dir)

    top_path = report_dir / "metrics" / f"similarity_to_{reference_label}_top10.csv"
    top_10_df.to_csv(top_path, index=False)
    print(f"Saved top 10 most similar to {reference_label} to: {top_path}")

    bottom_path = (
        report_dir / "metrics" / f"similarity_to_{reference_label}_bottom10.csv"
    )
    bottom_10_df.to_csv(bottom_path, index=False)
    print(f"Saved bottom 10 least similar to {reference_label} to: {bottom_path}")

    # Save figure
    if fig is not None:
        fig_path = report_dir / "plots" / f"similarity_to_{reference_label}.png"
        fig.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved similarity plot to: {fig_path}")

    return top_10_df, bottom_10_df


"""
Notes
    - Cosine Similarity across labels should be a broad distribution centered around 0
    - Cosine Similarity within labels should be skewed towards 1, indicating that embeddings
      sharing the same label are more similar to each other

    - TODO: how do we test the within / across labels with bulking?
"""
