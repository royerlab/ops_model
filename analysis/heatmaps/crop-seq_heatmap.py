"""
Goal: generate a clustered heatmap of the OPS-image embeddings
1. run heirarchical clustering on the image embeddings
2. generate a heatmap of the OPS, ordered by the hierarchical clustering
    - color of the ehatmap should be cosine similarity between embeddings
    - colorbar should be cosine similarity
3. plot the results

Notes:
- Look at how cosine similarity is efficiently calculated in /hpc/mydata/alexander.hillsley/ops/ops_model/experiments/paper_ready/batch_effects/batch_effect_analysis.py
using L2 norm and dot product
"""

#%%
import anndata as ad
import numpy as np
import seaborn as sns
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
from pathlib import Path

#%%
# INPUT_PATH = "/hpc/projects/icd.omics/CropSeq_June2025/sequencing/Processed_Data_FULL/DataObjects/CropSeq_June2025_dimensionalReduction.h5ad"
INPUT_PATH = '/hpc/mydata/giovanni.palla/repos/ops_multi_modal/results/contrastive_vi_cellxstate/perturbed_adata_raw.h5ad'

adata = ad.read_h5ad(INPUT_PATH)
X = np.asarray(adata.obsm["salient_rep"], dtype=np.float32)
# X = np.asarray(adata.X.todense(), dtype=np.float32)  
guide_perturbations = adata.obs['perturbation'].values

# Aggregate guides -> perturbations by mean
perturbation_labels = np.array(sorted(set(guide_perturbations)))
X = np.stack([X[guide_perturbations == p].mean(axis=0) for p in perturbation_labels])
gene_names = perturbation_labels

ntc_mask = np.asarray([a.startswith('Nontargeting') for a in gene_names])

print(f"Loaded {X.shape[0]} perturbations x {X.shape[1]} features ({ntc_mask.sum()} NTCs)")

#%%
# L2-normalise rows once so dot products equal cosine similarity
norms  = np.linalg.norm(X, axis=1, keepdims=True)
X_norm = X / (norms + 1e-10)
sim    = X_norm @ X_norm.T   # (n_genes, n_genes) cosine similarity, all entries

# Non-NTC only
X_pert        = X_norm[~ntc_mask]
X_pert_centered = X_pert - X_pert.mean(axis=0, keepdims=True)
X_pert_centered_norm = X_pert_centered / (np.linalg.norm(X_pert_centered, axis=1, keepdims=True) + 1e-10)
sim_non_ntc   = X_pert_centered_norm @ X_pert_centered_norm.T

gene_names_non_ntc = gene_names[~ntc_mask]
triu_mask  = np.triu(np.ones(sim_non_ntc.shape, dtype=bool), k=1)
sim_vals   = sim_non_ntc[triu_mask]
mean_sim  = sim_vals.mean()
print(f"Mean pairwise cosine similarity (non-NTC only): {mean_sim:.4f}")

#%%
# Hierarchical clustering on cosine distance (non-NTC only)
dist_matrix = 1.0 - sim_non_ntc
np.fill_diagonal(dist_matrix, 0.0)  # guard against float32 rounding
condensed   = squareform(dist_matrix, checks=False)
Z           = linkage(condensed, method='average')

#%%
g = sns.clustermap(
    sim_non_ntc,
    row_linkage=Z,
    col_linkage=Z,
    cmap='RdBu_r',
    vmin=-1, vmax=1,
    xticklabels=False,
    yticklabels=False,
    figsize=(14, 14),
    cbar_kws={'label': 'Cosine similarity'},
)
g.ax_heatmap.set_xlabel('Genes')
g.ax_heatmap.set_ylabel('Genes')
g.fig.suptitle('OPS gene embeddings — pairwise cosine similarity', y=1.01)

# %%
# ── Interactive HTML heatmap ───────────────────────────────────────────────────
import plotly.graph_objects as go

order = g.dendrogram_row.reordered_ind
gene_names_ord = gene_names_non_ntc[order].tolist()
sim_ord = sim_non_ntc[np.ix_(order, order)]

trace = go.Heatmap(
    z=sim_ord,
    x=gene_names_ord,
    y=gene_names_ord,
    colorscale='RdBu_r',
    zmin=-1, zmax=1,
    colorbar=dict(title='Cosine similarity'),
    hovertemplate='rna: %{x}<br>rna: %{y}<br>sim: %{z:.3f}<extra></extra>',
)
fig_interactive = go.Figure(data=[trace])
fig_interactive.update_layout(
    title='RNA gene embeddings — pairwise cosine similarity',
    width=900, height=900,
    xaxis=dict(showticklabels=False),
    yaxis=dict(showticklabels=False, autorange='reversed'),
)

out_html = Path('/hpc/mydata/alexander.hillsley/ops/ops_monorepo/figures/interactive_heatmaps/crop-seq_heatmap.html')
out_html.parent.mkdir(parents=True, exist_ok=True)
fig_interactive.write_html(str(out_html), auto_open=False)
print(f"Saved interactive plot to {out_html}")
# %%
