"""
Goal: produce a joint heatmap of the crop-Seq and OPS image embeddings,
1. Load the OPS and CROP-Seq image embeddings
2. Concatenate them into a single embedding
3. Run hierarchical clustering on the combined embedding
4. Generate a heatmap of OPS embeddings, ordered by the hierarchical clustering of the combined embedding
5. Generate a heatmap of CROP-Seq embeddings, ordered by the hierarchical clustering of the combined embedding
6. Plot the results, I would like to below diagonal to be the OPS heatmap and above diagonal to be the CROP-Seq heatmap, with shared row/column ordering
    - color of the heatmap should be cosine similarity between embeddings
    - color for OPS half should be white-red and CROP-Seq half should be white-blue

Notes:
- see /hpc/mydata/alexander.hillsley/ops/ops_monorepo/experiments/scratch/20260325_rna_heatmap.py for CROP-Seq heatmap
- see /hpc/mydata/alexander.hillsley/ops/ops_monorepo/experiments/scratch/20260325_heatmap.py for OPS heatmap
"""

#%%
import anndata as ad
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
from sklearn.decomposition import PCA
from pathlib import Path

#%%
OPS_PATH     = "/hpc/projects/icd.fast.ops/organelle_attribution/pca_optimized_v2/dino/all/gene_embedding_pca_optimized.h5ad"
RNA_PATH = '/hpc/mydata/giovanni.palla/repos/ops_multi_modal/results/contrastive_vi_cellxstate/perturbed_adata_raw.h5ad'
RNA_OBSM_KEY = "salient_rep"  # adjust if needed — print(adata_rna.obsm.keys()) to check
FIGURES_DIR  = Path('/hpc/mydata/alexander.hillsley/ops/ops_monorepo/figures/interactive_heatmaps')

#%%
# ── Load OPS embeddings ────────────────────────────────────────────────────────
adata_ops = ad.read_h5ad(OPS_PATH)
X_ops_all  = np.asarray(adata_ops.X, dtype=np.float32)
ops_names  = adata_ops.obs['perturbation'].values
ops_ntc_mask = np.asarray([a.startswith('NTC') for a in ops_names])
print(f"OPS:  {X_ops_all.shape[0]} perturbations x {X_ops_all.shape[1]} features  ({ops_ntc_mask.sum()} NTCs)")
print(f"OPS sample names (first 5): {ops_names[:5]}")

#%%
# ── Load RNA embeddings ────────────────────────────────────────────────────────
adata_rna = ad.read_h5ad(RNA_PATH)
print(f"RNA obsm keys: {list(adata_rna.obsm.keys())}")

X_rna_guides       = np.asarray(adata_rna.obsm[RNA_OBSM_KEY], dtype=np.float32)
guide_perturbations = adata_rna.obs['perturbation'].values

# Aggregate guides -> perturbations by mean
rna_pert_labels = np.array(sorted(set(guide_perturbations)))
X_rna_all       = np.stack([X_rna_guides[guide_perturbations == p].mean(axis=0) for p in rna_pert_labels])
rna_ntc_mask    = np.asarray([a.startswith('Nontargeting') for a in rna_pert_labels])
print(f"RNA:  {X_rna_all.shape[0]} perturbations x {X_rna_all.shape[1]} features  ({rna_ntc_mask.sum()} NTCs)")
print(f"RNA sample names (first 5): {rna_pert_labels[:5]}")

#%%
# ── Intersect non-NTC perturbations ───────────────────────────────────────────
ops_genes = ops_names[~ops_ntc_mask]
rna_genes = rna_pert_labels[~rna_ntc_mask]

shared = sorted(set(ops_genes) & set(rna_genes))
ops_only = sorted(set(ops_genes) - set(rna_genes))
rna_only = sorted(set(rna_genes) - set(ops_genes))
print(f"Shared non-NTC perturbations: {len(shared)}  (OPS: {len(ops_genes)}, RNA: {len(rna_genes)})")
print(f"OPS only ({len(ops_only)}): {ops_only}")
print(f"RNA only ({len(rna_only)}): {rna_only}")

ops_idx = [np.where(ops_names == g)[0][0] for g in shared]
rna_idx = [np.where(rna_pert_labels == g)[0][0] for g in shared]

X_ops = X_ops_all[ops_idx]   # (n_shared, d_ops)
X_rna = X_rna_all[rna_idx]   # (n_shared, d_rna)

#%%
# ── Helper functions ───────────────────────────────────────────────────────────
def l2_norm(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-10)

def zscore_cols(X):
    std = X.std(axis=0)
    std[std == 0] = 1.0
    return (X - X.mean(axis=0)) / std

def cluster_order(X):
    X_norm = l2_norm(X)
    dist = 1.0 - (X_norm @ X_norm.T)
    np.fill_diagonal(dist, 0.0)
    Z = linkage(squareform(dist, checks=False), method='average')
    return leaves_list(Z)

def plot_heatmap(sim_ops, sim_rna, order, gene_names, title, out_path):
    n = len(gene_names)
    sim_ops_ord = sim_ops[np.ix_(order, order)]
    sim_rna_ord = sim_rna[np.ix_(order, order)]
    gene_names_ord = [gene_names[i] for i in order]

    lower = np.tril(np.ones((n, n), dtype=bool))
    upper = np.triu(np.ones((n, n), dtype=bool), k=1)
    ops_data = np.where(lower, sim_ops_ord, np.nan)
    rna_data = np.where(upper, sim_rna_ord, np.nan)

    # ── Static PNG ──
    fig, ax = plt.subplots(figsize=(14, 14))
    im_ops = ax.imshow(ops_data, cmap='seismic', vmin=-0.5, vmax=0.5, aspect='auto', interpolation='nearest')
    im_rna = ax.imshow(rna_data, cmap='PiYG', vmin=-1, vmax=1, aspect='auto', interpolation='nearest')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('Genes')
    ax.set_ylabel('Genes')
    cb_ops = fig.colorbar(im_ops, ax=ax, fraction=0.03, pad=0.01, location='right')
    cb_ops.set_label('OPS cosine similarity')
    cb_rna = fig.colorbar(im_rna, ax=ax, fraction=0.03, pad=0.06, location='right')
    cb_rna.set_label('RNA cosine similarity')
    ax.set_title(title, pad=12)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved to {out_path}")

    # ── Interactive HTML ──
    trace_ops = go.Heatmap(
        z=ops_data, x=gene_names_ord, y=gene_names_ord,
        colorscale='RdBu_r', zmin=-0.5, zmax=0.5,
        colorbar=dict(title='OPS cosine sim', x=1.02),
        hovertemplate='x: %{x}<br>y: %{y}<br>OPS sim: %{z:.3f}<extra></extra>',
        name='OPS',
    )
    trace_rna = go.Heatmap(
        z=rna_data, x=gene_names_ord, y=gene_names_ord,
        colorscale='PiYG', zmin=-1, zmax=1,
        colorbar=dict(title='RNA cosine sim', x=1.12),
        hovertemplate='x: %{x}<br>y: %{y}<br>RNA sim: %{z:.3f}<extra></extra>',
        name='RNA',
    )
    fig_interactive = go.Figure(data=[trace_ops, trace_rna])
    fig_interactive.update_layout(
        title=title, width=900, height=900,
        xaxis=dict(showticklabels=False),
        yaxis=dict(showticklabels=False, autorange='reversed'),
    )
    out_html = out_path.with_suffix('.html')
    fig_interactive.write_html(str(out_html), auto_open=False)
    print(f"Saved interactive plot to {out_html}")

#%%
# ── Normalize embeddings and compute similarity matrices ───────────────────────

# Center OPS embeddings by subtracting the mean perturbation embedding before
# re-normalizing. This removes the shared "generic perturbation direction" so
# cosine similarity reflects gene-specific phenotypic differences.
# We do not scale by std — higher-variance dims carry more signal and should
# contribute more to the angle between embeddings.
X_ops_l2 = l2_norm(X_ops)
X_ops_centered = X_ops_l2 - X_ops_l2.mean(axis=0, keepdims=True)
X_ops_norm = l2_norm(X_ops_centered)

X_rna_l2 = l2_norm(X_rna)
X_rna_centered = X_rna_l2 - X_rna_l2.mean(axis=0, keepdims=True)
X_rna_norm = l2_norm(X_rna_centered)

sim_ops = X_ops_norm @ X_ops_norm.T   # (n, n)
sim_rna = X_rna_norm @ X_rna_norm.T   # (n, n)

#%%
# ── PCA reduction of OPS for joint clustering ──────────────────────────────────
N_PCS = 10
pca_ops = PCA(n_components=N_PCS)
X_ops_pca = pca_ops.fit_transform(X_ops_centered)
print(f"OPS variance explained by {N_PCS} PCs: {pca_ops.explained_variance_ratio_.sum():.1%}  "
      f"(cumulative: {pca_ops.explained_variance_ratio_.cumsum().tolist()})")

#%%
# ── Generate heatmaps ──────────────────────────────────────────────────────────
n = len(shared)

order_joint = cluster_order(np.concatenate([X_ops_pca, X_rna_centered], axis=1))
plot_heatmap(sim_ops, sim_rna, order_joint, shared,
    f'Joint clustering  ({n} perturbations)\nLower triangle = OPS    |   Upper triangle = RNA ',
    FIGURES_DIR / 'joint_heatmap.png')

order_ops = cluster_order(X_ops_centered)
plot_heatmap(sim_ops, sim_rna, order_ops, shared,
    f'OPS clustering  ({n} perturbations)\nLower triangle = OPS    |   Upper triangle = RNA ',
    FIGURES_DIR / 'ops_heatmap.png')

order_rna = cluster_order(X_rna_centered)
plot_heatmap(sim_ops, sim_rna, order_rna, shared,
    f'RNA clustering  ({n} perturbations)\nLower triangle = OPS    |   Upper triangle = RNA ',
    FIGURES_DIR / 'rna_heatmap.png')

# %%