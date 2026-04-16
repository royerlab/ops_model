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
INPUT_PATH = "/hpc/projects/icd.fast.ops/organelle_attribution/pca_optimized_v2/dino/all/gene_embedding_pca_optimized.h5ad"

adata = ad.read_h5ad(INPUT_PATH)
X = np.asarray(adata.X, dtype=np.float32)
gene_names = adata.obs['perturbation'].values
ntc_mask   = np.asarray([a.startswith('NTC') for a in gene_names])

print(f"Loaded {X.shape[0]} genes x {X.shape[1]} features ({ntc_mask.sum()} NTCs)")

#%%
# L2-normalise rows once so dot products equal cosine similarity
norms  = np.linalg.norm(X, axis=1, keepdims=True)
X_norm = X / (norms + 1e-10)
sim    = X_norm @ X_norm.T   # (n_genes, n_genes) cosine similarity, all entries

# Non-NTC only
X_non_ntc        = X_norm[~ntc_mask]
gene_names_non_ntc = gene_names[~ntc_mask]
sim_non_ntc      = X_non_ntc @ X_non_ntc.T

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
    cmap='OrRd',
    vmin=0, vmax=1,
    xticklabels=False,
    yticklabels=False,
    figsize=(14, 14),
    cbar_kws={'label': 'Cosine similarity'},
)
g.ax_heatmap.set_xlabel('Genes')
g.ax_heatmap.set_ylabel('Genes')
g.fig.suptitle('OPS gene embeddings — pairwise cosine similarity', y=1.01)


#%%
# ── Sanity check: recompute cosine similarity with sklearn ─────────────────────
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity

sim_sklearn = sklearn_cosine_similarity(X)

print(f"Max absolute difference (L2+dot vs sklearn): {np.abs(sim - sim_sklearn).max():.6f}")

triu_mask         = np.triu(np.ones(sim.shape, dtype=bool), k=1)
triu_mask_non_ntc = np.triu(np.ones(sim_non_ntc.shape, dtype=bool), k=1)
print(f"Off-diagonal cosine similarity (L2+dot):  mean={sim[triu_mask].mean():.4f}  min={sim[triu_mask].min():.4f}  max={sim[triu_mask].max():.4f}")
print(f"Off-diagonal cosine similarity (sklearn):  mean={sim_sklearn[triu_mask].mean():.4f}  min={sim_sklearn[triu_mask].min():.4f}  max={sim_sklearn[triu_mask].max():.4f}")

#%%
# ── Distribution of off-diagonal cosine similarities ──────────────────────────
import matplotlib.pyplot as plt

off_diag_vals     = sim[triu_mask]
non_ntc_vals      = sim_non_ntc[triu_mask_non_ntc]

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(off_diag_vals, bins=100, alpha=0.5, color='steelblue',     edgecolor='none', density=True, label='all pairs')
ax.hist(non_ntc_vals,  bins=100, alpha=0.5, color='mediumpurple',  edgecolor='none', density=True, label='perturbations only')
ax.axvline(off_diag_vals.mean(), color='steelblue',    linestyle='--', linewidth=1.2, label=f'all mean = {off_diag_vals.mean():.3f}')
ax.axvline(non_ntc_vals.mean(),  color='mediumpurple', linestyle='--', linewidth=1.2, label=f'pert mean = {non_ntc_vals.mean():.3f}')
ax.set_xlabel('Cosine similarity')
ax.set_ylabel('Density')
ax.set_title('Distribution of pairwise cosine similarities (off-diagonal)')
ax.legend()
plt.tight_layout()

#%%
# ── NTC similarity characterization ───────────────────────────────────────────
X_ntc = X_norm[ntc_mask]

sim_ntc_ntc     = X_ntc @ X_ntc.T      # NTC vs NTC
sim_ntc_non_ntc = X_ntc @ X_non_ntc.T  # NTC vs perturbations

ntc_triu_mask = np.triu(np.ones(sim_ntc_ntc.shape, dtype=bool), k=1)
ntc_ntc_vals     = sim_ntc_ntc[ntc_triu_mask]
ntc_non_ntc_vals = sim_ntc_non_ntc.flatten()

for label, vals in [('NTC vs NTC', ntc_ntc_vals), ('NTC vs perturbations', ntc_non_ntc_vals)]:
    print(f"{label}: mean={vals.mean():.4f}  median={np.median(vals):.4f}  min={vals.min():.4f}  max={vals.max():.4f}")

#%%
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(non_ntc_vals,      bins=100, alpha=0.5, color='mediumpurple',    edgecolor='none', density=True, label='pert vs pert')
ax.hist(ntc_ntc_vals,      bins=100, alpha=0.5, color='tomato',          edgecolor='none', density=True, label='NTC vs NTC')
ax.hist(ntc_non_ntc_vals,  bins=100, alpha=0.5, color='mediumseagreen',  edgecolor='none', density=True, label='NTC vs perturbations')
ax.axvline(non_ntc_vals.mean(),      color='mediumpurple',   linestyle='--', linewidth=1.2, label=f'pert mean = {non_ntc_vals.mean():.3f}')
ax.axvline(ntc_ntc_vals.mean(),      color='tomato',         linestyle='--', linewidth=1.2, label=f'NTC vs NTC mean = {ntc_ntc_vals.mean():.3f}')
ax.axvline(ntc_non_ntc_vals.mean(),  color='mediumseagreen', linestyle='--', linewidth=1.2, label=f'NTC vs pert mean = {ntc_non_ntc_vals.mean():.3f}')
ax.set_xlabel('Cosine similarity')
ax.set_ylabel('Density')
ax.set_title('Cosine similarity distributions')
ax.legend()
plt.tight_layout()

#%%
# ── Per-dimension variance: NTC vs perturbations ───────────────────────────────
# Dimensions where perturbations have high variance but NTCs have low variance
# are "perturbation-specific" axes not captured by the NTC baseline.
import matplotlib.pyplot as plt

var_ntc  = X_ntc.var(axis=0)       # (n_dims,)
var_pert = X_non_ntc.var(axis=0)   # (n_dims,)

# Sort by perturbation variance descending
sort_idx = np.argsort(var_pert)[::-1]
var_ntc_sorted  = var_ntc[sort_idx]
var_pert_sorted = var_pert[sort_idx]

print(f"Top 10 pert-specific dims (pert var / NTC var):")
for i in range(10):
    ratio = var_pert_sorted[i] / (var_ntc_sorted[i] + 1e-10)
    print(f"  dim {sort_idx[i]:4d}: pert_var={var_pert_sorted[i]:.4f}  ntc_var={var_ntc_sorted[i]:.4f}  ratio={ratio:.2f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 4))

# Left: variance per dimension, sorted by pert variance
ax = axes[0]
x = np.arange(len(var_pert_sorted))
ax.plot(x, var_pert_sorted, color='mediumpurple', linewidth=0.8, label='perturbations')
ax.plot(x, var_ntc_sorted,  color='tomato',       linewidth=0.8, label='NTC')
ax.set_xlabel('Dimension (sorted by pert variance)')
ax.set_ylabel('Variance')
ax.set_title('Per-dimension variance: NTC vs perturbations')
ax.legend()

# Right: pert variance vs NTC variance scatter (one point per dim)
ax = axes[1]
ax.scatter(var_ntc, var_pert, s=4, alpha=0.4, color='steelblue')
lim = max(var_ntc.max(), var_pert.max()) * 1.05
ax.plot([0, lim], [0, lim], color='gray', linestyle='--', linewidth=1)
ax.set_xlabel('NTC variance')
ax.set_ylabel('Perturbation variance')
ax.set_title('NTC vs pert variance per dimension')

plt.tight_layout()

#%%
# ── Sweep over TOP_N_DIMS: mean cosine similarity vs N dims ───────────────────
dim_sweep = range(5, 1901, 10)
sweep_means   = []
sweep_medians = []

for n in dim_sweep:
    X_f     = X_non_ntc[:, sort_idx[:n]]
    norms_f = np.linalg.norm(X_f, axis=1, keepdims=True)
    X_f     = X_f / (norms_f + 1e-10)
    sim_f   = X_f @ X_f.T
    vals    = sim_f[np.triu(np.ones(sim_f.shape, dtype=bool), k=1)]
    sweep_means.append(vals.mean())
    sweep_medians.append(np.median(vals))

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(list(dim_sweep), sweep_means,   color='darkorange',   linewidth=1.5, label='mean')
ax.plot(list(dim_sweep), sweep_medians, color='steelblue',    linewidth=1.5, label='median')
ax.axhline(non_ntc_vals.mean(),   color='mediumpurple', linestyle='--', linewidth=1, label=f'all {X_non_ntc.shape[1]} dims mean')
ax.set_xlabel('Top N dimensions (by pert variance)')
ax.set_ylabel('Cosine similarity')
ax.set_title('Pert-pert mean cosine similarity vs number of dimensions')
ax.legend()
plt.tight_layout()

#%%
# ── Filtered similarity using top-N pert-specific dimensions ──────────────────
TOP_N_DIMS = 100  # ← change this

top_dims = sort_idx[:TOP_N_DIMS]
X_non_ntc_filtered = X_non_ntc[:, top_dims]

# Re-normalise after projection
norms_f       = np.linalg.norm(X_non_ntc_filtered, axis=1, keepdims=True)
X_non_ntc_f   = X_non_ntc_filtered / (norms_f + 1e-10)
sim_filtered  = X_non_ntc_f @ X_non_ntc_f.T

triu_mask_f   = np.triu(np.ones(sim_filtered.shape, dtype=bool), k=1)
filtered_vals = sim_filtered[triu_mask_f]

print(f"Pert-pert similarity (top {TOP_N_DIMS} dims): mean={filtered_vals.mean():.4f}  median={np.median(filtered_vals):.4f}  min={filtered_vals.min():.4f}  max={filtered_vals.max():.4f}")

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(non_ntc_vals,   bins=100, alpha=0.5, color='mediumpurple', edgecolor='none', density=True, label=f'all {X_non_ntc.shape[1]} dims')
ax.hist(filtered_vals,  bins=100, alpha=0.5, color='darkorange',   edgecolor='none', density=True, label=f'top {TOP_N_DIMS} dims')
ax.axvline(non_ntc_vals.mean(),  color='mediumpurple', linestyle='--', linewidth=1.2, label=f'all dims mean = {non_ntc_vals.mean():.3f}')
ax.axvline(filtered_vals.mean(), color='darkorange',   linestyle='--', linewidth=1.2, label=f'top {TOP_N_DIMS} mean = {filtered_vals.mean():.3f}')
ax.set_xlabel('Cosine similarity')
ax.set_ylabel('Density')
ax.set_title(f'Pert-pert cosine similarity: all dims vs top {TOP_N_DIMS} pert-specific dims')
ax.legend()
plt.tight_layout()

#%%
# Hierarchical clustering on cosine distance (non-NTC only)
dist_matrix = 1.0 - sim_filtered
np.fill_diagonal(dist_matrix, 0.0)  # guard against float32 rounding
condensed   = squareform(dist_matrix, checks=False)
Z           = linkage(condensed, method='average')

#%%
g = sns.clustermap(
    sim_filtered,
    row_linkage=Z,
    col_linkage=Z,
    cmap='OrRd',
    vmin=0, vmax=1,
    xticklabels=False,
    yticklabels=False,
    figsize=(14, 14),
    cbar_kws={'label': 'Cosine similarity'},
)
g.ax_heatmap.set_xlabel('Genes')
g.ax_heatmap.set_ylabel('Genes')
g.fig.suptitle('OPS gene embeddings — pairwise cosine similarity', y=1.01)
#%%
out_path = Path('/hpc/mydata/alexander.hillsley/ops/ops_monorepo/figures') / 'gene_embedding_cosine_similarity_heatmap.png'
out_path.parent.mkdir(parents=True, exist_ok=True)
g.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"Saved to {out_path}")

#%%
order = g.dendrogram_row.reordered_ind

ROW_START, ROW_END = 0,   100  # ← row interval
COL_START, COL_END = 100, 200  # ← col interval (set equal to row for diagonal block)

row_idx    = np.array(order)[ROW_START:ROW_END]
col_idx    = np.array(order)[COL_START:COL_END]
row_labels = gene_names_non_ntc[row_idx]
col_labels = gene_names_non_ntc[col_idx]
sim_sub    = sim_non_ntc[np.ix_(row_idx, col_idx)]

fig, ax = plt.subplots(figsize=(10, 10))
im = ax.imshow(sim_sub, cmap='OrRd', vmin=0, vmax=1, aspect='auto', interpolation='nearest')
ax.set_xticks(range(len(col_idx)))
ax.set_xticklabels(col_labels, rotation=90, fontsize=7)
ax.set_yticks(range(len(row_idx)))
ax.set_yticklabels(row_labels, fontsize=7)
fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01, label='Cosine similarity')
ax.set_title(f'OPS  rows [{ROW_START}:{ROW_END}] vs cols [{COL_START}:{COL_END}]')
plt.tight_layout()

#%%
# ── Re-center by mean perturbation embedding ──────────────────────────────────
# Removes the shared "generic perturbation direction", leaving gene-specific
# deviations from the average perturbed phenotype.
#
# Note: we subtract the mean but do NOT scale by std dev. X_non_ntc is already
# L2-normalized, so dimensions with higher variance across genes are the ones
# carrying more perturbation-specific signal (we identified ~100 such dims above).
# Scaling by std would equalize all dimensions, down-weighting those 100 signal
# dims to match the ~1800 noise dims — the opposite of what we want.
pert_mean     = X_non_ntc.mean(axis=0, keepdims=True)   # (1, n_dims)
pert_std = X_non_ntc.std(axis=0, keepdims=True)     # (1, n_dims)
X_pert_centered = X_non_ntc - pert_mean

norms_pc      = np.linalg.norm(X_pert_centered, axis=1, keepdims=True)
X_pert_c      = X_pert_centered / (norms_pc + 1e-10)
sim_pert_c    = X_pert_c @ X_pert_c.T

triu_mask_pc  = np.triu(np.ones(sim_pert_c.shape, dtype=bool), k=1)
pert_c_vals   = sim_pert_c[triu_mask_pc]

print(f"Pert-pert similarity after pert-mean centering: mean={pert_c_vals.mean():.4f}  median={np.median(pert_c_vals):.4f}  min={pert_c_vals.min():.4f}  max={pert_c_vals.max():.4f}")

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(non_ntc_vals,  bins=100, alpha=0.5, color='mediumpurple', edgecolor='none', density=True, label='original')
ax.hist(pert_c_vals,   bins=100, alpha=0.5, color='darkorange',   edgecolor='none', density=True, label='pert-mean centered')
ax.axvline(non_ntc_vals.mean(), color='mediumpurple', linestyle='--', linewidth=1.2, label=f'original mean = {non_ntc_vals.mean():.3f}')
ax.axvline(pert_c_vals.mean(),  color='darkorange',   linestyle='--', linewidth=1.2, label=f'centered mean = {pert_c_vals.mean():.3f}')
ax.set_xlabel('Cosine similarity')
ax.set_ylabel('Density')
ax.set_title('Pert-pert cosine similarity: before vs after pert-mean centering')
ax.legend()
plt.tight_layout()

#%%
dist_pc   = 1.0 - sim_pert_c
np.fill_diagonal(dist_pc, 0.0)
Z_pc      = linkage(squareform(dist_pc, checks=False), method='average')

g_pc = sns.clustermap(
    sim_pert_c,
    row_linkage=Z_pc,
    col_linkage=Z_pc,
    cmap='RdBu_r',
    vmin=-0.4, vmax=0.4,
    xticklabels=False,
    yticklabels=False,
    figsize=(14, 14),
    cbar_kws={'label': 'Cosine similarity'},
)
g_pc.ax_heatmap.set_xlabel('Genes')
g_pc.ax_heatmap.set_ylabel('Genes')
g_pc.fig.suptitle('OPS gene embeddings — pert-mean centered cosine similarity', y=1.01)

# %%
# ── Interactive HTML heatmap (pert-mean centered, full matrix) ─────────────────
import plotly.graph_objects as go

order = g_pc.dendrogram_row.reordered_ind
gene_names_ord = gene_names_non_ntc[order].tolist()
sim_ord = sim_pert_c[np.ix_(order, order)]

trace = go.Heatmap(
    z=sim_ord,
    x=gene_names_ord,
    y=gene_names_ord,
    colorscale='RdBu_r',
    zmin=-0.4, zmax=0.4,
    colorbar=dict(title='Cosine similarity'),
    hovertemplate='ops: %{x}<br>ops: %{y}<br>sim: %{z:.3f}<extra></extra>',
)
fig_interactive = go.Figure(data=[trace])
fig_interactive.update_layout(
    title='OPS gene embeddings — pert-mean centered cosine similarity',
    width=900, height=900,
    xaxis=dict(showticklabels=False),
    yaxis=dict(showticklabels=False, autorange='reversed'),
)

out_html = Path('/hpc/mydata/alexander.hillsley/ops/ops_monorepo/figures/interactive_heatmaps/OPS_heatmap.html')
out_html.parent.mkdir(parents=True, exist_ok=True)
fig_interactive.write_html(str(out_html), auto_open=False)
print(f"Saved interactive plot to {out_html}")
# %%
