"""
UMAP and Leiden clustering analysis for gene-level OPS embeddings.

Takes a gene-level AnnData object (PCA-optimized dino embeddings) and:
  1. Computes a 2D UMAP, storing coordinates and the UMAP neighbor graph
     (so Leiden clustering uses the same graph structure).
  2. Runs Leiden clustering at 10 resolutions (0.5–9.0), computing per-cell
     and global cosine silhouette scores at each resolution.
  3. Saves the annotated AnnData to disk.
  4. Plots a UMAP colored by cluster assignment for each resolution.
  5. Writes a text file per resolution listing which genes fall in each cluster
     along with per-cluster silhouette scores.
  6. Generates per-cluster highlight UMAPs at resolution 8.0, with each gene
     colored individually against a grey background.

Input:  gene_embedding_pca_optimized.h5ad  (gene × embedding)
Output: gene_embedding_pca_optimized_leiden.h5ad
        figures/leiden_<res>_clusters.txt  (one per resolution)
        figures/umap_leiden8.0_cluster<N>.png  (one per cluster)
"""

#%%
import scanpy as sc
import anndata as ad
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from pathlib import Path
from umap import UMAP
from sklearn.metrics import silhouette_score, silhouette_samples

INPUT_PATH = "/hpc/projects/icd.fast.ops/organelle_attribution/pca_optimized_v2/dino/all/gene_embedding_pca_optimized.h5ad"
OUTPUT_PATH = "/hpc/projects/icd.fast.ops/organelle_attribution/pca_optimized_v2/dino/all/gene_embedding_pca_optimized_leiden.h5ad"
FIGURE_PATH = '/hpc/mydata/alexander.hillsley/ops/ops_monorepo/experiments/scratch/figures'
RESOLUTIONS = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]


def compute_umap_and_neighbors(adata: ad.AnnData, n_neighbors: int = 15, random_state: int = 42) -> ad.AnnData:
    """
    Compute UMAP using umap-learn (matching pca_optimization.py) and store the neighbor
    graph used by UMAP into adata so that Leiden clustering uses the same graph.

    Stores:
        adata.obsm["X_umap"]         — 2D UMAP coordinates
        adata.obsp["connectivities"] — fuzzy simplicial set from umap-learn
        adata.uns["neighbors"]       — metadata for scanpy compatibility
    """
    X = np.asarray(adata.X, dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    nn = min(n_neighbors, adata.n_obs - 1)
    print(f"Computing UMAP (n_neighbors={nn})...")
    model = UMAP(n_components=2, n_neighbors=nn, random_state=random_state)
    coords = model.fit_transform(X)

    adata.obsm["X_umap"] = coords

    # Store the neighbor graph umap-learn built so Leiden uses the same structure
    connectivities = model.graph_.astype(np.float32)
    adata.obsp["connectivities"] = connectivities
    adata.obsp["distances"] = sp.csr_matrix(connectivities.shape)  # Leiden only needs connectivities
    adata.uns["neighbors"] = {
        "connectivities_key": "connectivities",
        "distances_key": "distances",
        "params": {"n_neighbors": nn, "metric": "euclidean", "method": "umap", "random_state": random_state},
    }

    return adata


#%%
adata = ad.read_h5ad(INPUT_PATH)

adata = compute_umap_and_neighbors(adata)

X_embed = np.asarray(adata.X, dtype=np.float32)
silhouette_per_cell = {}   # key → per-cell silhouette array
silhouette_global = {}     # key → scalar

print(f"\nRunning Leiden clustering at {len(RESOLUTIONS)} resolution(s)...")
for res in RESOLUTIONS:
    key = f"leiden_{res}"
    print(f"  Resolution {res}...")
    sc.tl.leiden(adata, resolution=res, key_added=key, random_state=42, flavor="igraph", n_iterations=2, directed=False)
    n_clusters = adata.obs[key].nunique()
    labels = adata.obs[key].astype(int).values
    if n_clusters > 1:
        per_cell = silhouette_samples(X_embed, labels, metric="cosine")
        silhouette_per_cell[key] = per_cell
        silhouette_global[key] = per_cell.mean()
        per_cluster_mean = {
            c: per_cell[labels == c].mean()
            for c in np.unique(labels)
        }
        print(f"    → {n_clusters} clusters | silhouette: mean={silhouette_global[key]:.3f}, "
              f"min={min(per_cluster_mean.values()):.3f}, max={max(per_cluster_mean.values()):.3f}")
    else:
        silhouette_per_cell[key] = np.zeros(adata.n_obs)
        silhouette_global[key] = float("nan")
        print(f"    → {n_clusters} cluster (silhouette undefined)")

adata.write_h5ad(OUTPUT_PATH)

#%%
# UMAP plots colored by leiden cluster
for res in RESOLUTIONS:
    key = f"leiden_{res}"
    sil = silhouette_global[key]
    title = f"Leiden {res}" if not np.isnan(sil) else f"Leiden {res}"
    sc.pl.umap(adata, color=key, title=title, save=f"_leiden_{res}.png")

#%%
# Genes per cluster

for res in RESOLUTIONS:
    key = f"leiden_{res}"
    per_cell = silhouette_per_cell[key]
    labels = adata.obs[key].astype(int).values
    lines = [f"Global silhouette score: {silhouette_global[key]:.4f}", ""]
    for cluster, group in adata.obs.groupby(key, observed=True):
        c = int(cluster)
        cluster_sil = per_cell[labels == c].mean()
        genes = sorted(group["perturbation"].tolist())
        header = f"Cluster {cluster} ({len(genes)} genes, silhouette={cluster_sil:.3f})"
        lines.append(header)
        lines.extend(f"  {g}" for g in genes)
        lines.append("")

    txt = "\n".join(lines)
    print(f"\n=== Leiden {res} ===")
    print(txt)

    out_path = Path(FIGURE_PATH) / f"leiden_{res}_clusters.txt"
    out_path.write_text(txt)
    print(f"Saved to {out_path}")

#%%
# Per-cluster highlight UMAPs for leiden 8.0
umap_coords = adata.obsm["X_umap"]
cluster_key = "leiden_8.0"
figures_dir = Path(FIGURE_PATH)
figures_dir.mkdir(exist_ok=True)

per_cell_8 = silhouette_per_cell[cluster_key]
labels_8 = adata.obs[cluster_key].astype(int).values

for cluster, group in adata.obs.groupby(cluster_key, observed=True):
    mask = adata.obs[cluster_key] == cluster
    perturbations = sorted(group["perturbation"].tolist())
    cluster_sil = per_cell_8[labels_8 == int(cluster)].mean()

    fig, ax = plt.subplots(figsize=(8, 6))

    # Background: all points in grey
    ax.scatter(umap_coords[~mask, 0], umap_coords[~mask, 1], c="lightgrey", s=10, linewidths=0)

    # Foreground: cluster points, one per perturbation for legend
    colors = plt.cm.tab20(np.linspace(0, 1, len(perturbations)))
    for pert, color in zip(perturbations, colors):
        pt_mask = mask & (adata.obs["perturbation"] == pert)
        ax.scatter(umap_coords[pt_mask, 0], umap_coords[pt_mask, 1], c=[color], s=20, linewidths=0, label=pert)

    ax.set_title(f"Leiden 8.0 — Cluster {cluster} ({len(perturbations)} genes)")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=7, markerscale=1.5, frameon=False)

    print(f"\nCluster {cluster}: {perturbations}")

    out_path = figures_dir / f"umap_leiden8.0_cluster{cluster}.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved to {out_path}")

#%%
