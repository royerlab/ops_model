#%%
import anndata as ad
import numpy as np
from scipy import sparse
from tqdm import tqdm
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
#%%

def zscore_adata(adata: ad.AnnData) -> ad.AnnData:
    """Z-score each feature to zero mean and unit variance using global statistics.

    Applied per-experiment before concatenation so that each experiment's
    embedding distribution is centered and scaled independently, removing
    experiment-level offset and scale batch effects prior to any analysis.
    """
    X = (adata.X.toarray() if sparse.issparse(adata.X) else np.asarray(adata.X)).astype(np.float64)
    means = X.mean(axis=0)
    stds = X.std(axis=0, ddof=1)
    stds[stds == 0] = 1.0
    adata = adata.copy()
    adata.X = ((X - means) / stds).astype(np.float32)
    return adata


# Load guide-level Phase embeddings for all experiments, z-scoring each independently
# before concatenation so batch offsets don't inflate cross-experiment similarity.
EXPERIMENTS = [
    'ops0031_20250424', 'ops0032_20250428', 'ops0035_20250501', 'ops0036_20250505', 'ops0037_20250506',
    'ops0038_20250514', 'ops0041_20250519', 'ops0042_20250520', 'ops0043_20250605', 'ops0045_20250603',
    'ops0047_20250612', 'ops0048_20250616', 'ops0049_20250626', 'ops0051_20250623', 'ops0052_20250702', 'ops0053_20250709',
    'ops0054_20250710', 'ops0055_20250715', 'ops0056_20250721', 'ops0057_20250722', 'ops0058_20250805',
    'ops0059_20250804', 'ops0062_20250729', 'ops0063_20250731', 'ops0064_20250811', 'ops0065_20250812', 'ops0066_20250820', 'ops0067_20250826', 'ops0068_20250901',
    'ops0069_20250902', 'ops0070_20250908', 'ops0071_20250828', 'ops0072_20250904', 'ops0076_20250917', 'ops0078_20250923',
    'ops0081_20250924', 'ops0084_20251022', 'ops0085_20251118', 'ops0086_20250922', 'ops0089_20251119',
    'ops0090_20251120', 'ops0091_20251117', 'ops0092_20251027', 'ops0094_20251217', 'ops0097_20251023',
    'ops0100_20251218', 'ops0101_20251211', 'ops0102_20251210', 'ops0103_20251216', 'ops0104_20251215',
    'ops0105_20260106', 'ops0106_20251204', 'ops0107_20251208', 'ops0110_20260108', 'ops0113_20251219',
    'ops0114_20260112', 'ops0116_20260120', 'ops0117_20260128', 'ops0118_20260129', 'ops0119_20260203',
    'ops0120_20260204', 'ops0121_20260210', 'ops0122_20260211', 'ops0124_20260218', 'ops0125_20260219',
    'ops0126_20260224', 'ops0128_20260225', 'ops0129_20260303', 'ops0130_20260304', 'ops0131_20260310',
    'ops0132_20260316', 'ops0134_20260317', 'ops0135_20260318', 'ops0137_20260323', 'ops0139_20260325',
    'ops0140_20260331', 'ops0142_20260401', 'ops0143_20260407', 'ops0144_20260406', 'ops0146_20260402',
]
print(f"Loading {len(EXPERIMENTS)} experiments...")
adatas = [
    zscore_adata(ad.read_h5ad(f'/hpc/projects/icd.fast.ops/{exp}/3-assembly/dino_features/anndata_objects/guide_bulked_Phase.h5ad'))
    for exp in tqdm(EXPERIMENTS, desc='Loading')
]
adata = ad.concat(adatas, axis=0)
X = (adata.X.toarray() if sparse.issparse(adata.X) else np.asarray(adata.X)).astype(np.float32)

perturbations        = adata.obs['perturbation'].values
sgRNAs               = adata.obs['sgRNA'].values
experiments          = adata.obs['experiment'].values
experiments_unique   = sorted(adata.obs['experiment'].unique())
perturbations_unique = adata.obs['perturbation'].unique()
n_exp                = len(experiments_unique)

# L2-normalise rows so all dot products equal cosine similarity
norms  = np.linalg.norm(X, axis=1, keepdims=True)
X_norm = X / (norms + 1e-10)

print(f"Observations : {len(adata)}")
print(f"Experiments  : {experiments_unique}")
print(f"Perturbations: {len(perturbations_unique)}")
print(f"Features     : {X.shape[1]}")
print(f"NTC rows     : {(perturbations == 'NTC').sum()}")

# %% Within-experiment pairwise similarity — all guides
# For each experiment, compute the full pairwise cosine similarity matrix across all
# guide embeddings and extract the off-diagonal values. High mean similarity suggests
# that embeddings are not spread across the feature space, which could indicate that
# perturbation signal is weak or that batch effects dominate the variance.
records = []
for exp in tqdm(experiments_unique):
    mask = experiments == exp
    X_exp = X_norm[mask]
    n = X_exp.shape[0]

    sim = X_exp @ X_exp.T
    off_diag = sim[~np.eye(n, dtype=bool)]

    records.append({
        'experiment':   exp,
        'n_guides':     n,
        'mean_sim':     float(off_diag.mean()),
        'std_sim':      float(off_diag.std()),
        'min_sim':      float(off_diag.min()),
        'max_sim':      float(off_diag.max()),
    })

results = pd.DataFrame(records).set_index('experiment')
print("\nMean pairwise cosine similarity within each experiment:")
print(results.to_string(float_format=lambda x: f"{x:.4f}"))

fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(results['mean_sim'], bins=50, range=(-1, 1), color='skyblue', edgecolor='black')
ax.set_title('Mean cosine similarity: all guides within-experiment')
ax.set_xlabel('Mean cosine similarity')
ax.set_xlim(-1, 1)
fig.tight_layout()

# %% Within-experiment NTC pairwise similarity
# Restrict to NTC (non-targeting control) guides only and repeat the pairwise analysis.
# NTCs share no perturbation, so their similarity reflects the baseline spread of
# unperturbed cell embeddings within each experiment. A tight NTC cluster (high
# within-experiment NTC similarity) suggests experiment-level structure dominates.
ntc_mask = perturbations == 'NTC'

records_ntc = []
records_ntc_raw = []
for exp in tqdm(experiments_unique):
    mask = ntc_mask & (experiments == exp)
    X_exp = X_norm[mask]
    n = X_exp.shape[0]

    if n < 2:
        print(f"{exp}: fewer than 2 NTCs, skipping")
        continue

    sim = X_exp @ X_exp.T
    off_diag = sim[~np.eye(n, dtype=bool)]

    records_ntc.append({
        'experiment': exp,
        'n_ntc':      n,
        'mean_sim':   float(off_diag.mean()),
        'std_sim':    float(off_diag.std()),
        'min_sim':    float(off_diag.min()),
        'max_sim':    float(off_diag.max()),
    })
    records_ntc_raw.append(off_diag)

results_ntc = pd.DataFrame(records_ntc).set_index('experiment')
print("\nMean pairwise cosine similarity within each experiment (NTCs only):")
print(results_ntc.to_string(float_format=lambda x: f"{x:.4f}"))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(results_ntc['mean_sim'], bins=50, range=(-1, 1), color='skyblue', edgecolor='black')
axes[0].set_title('Per-experiment mean NTC similarity (within)')
axes[0].set_xlabel('Mean cosine similarity')
axes[0].set_xlim(-1, 1)
axes[1].hist(np.concatenate(records_ntc_raw), bins=100, range=(-1, 1), color='lightcoral', edgecolor='black')
axes[1].set_title('All pairwise NTC similarities (within-experiment)')
axes[1].set_xlabel('Cosine similarity')
axes[1].set_xlim(-1, 1)
fig.tight_layout()

# %% Cross-experiment NTC similarity + overlay
# Compare NTC embeddings of each experiment against NTCs from all other experiments.
# If experiments share a common biological baseline, cross-experiment NTC similarity
# should be comparable to within-experiment NTC similarity. A large gap between the
# two distributions is a strong indicator of residual batch effects after z-scoring.
records_cross = []
records_cross_raw = []
for exp in experiments_unique:
    mask_in  = ntc_mask & (experiments == exp)
    mask_out = ntc_mask & (experiments != exp)

    X_in  = X_norm[mask_in]
    X_out = X_norm[mask_out]

    if X_in.shape[0] < 1 or X_out.shape[0] < 1:
        print(f"{exp}: not enough NTCs for cross-experiment comparison, skipping")
        continue

    cross_sim = X_in @ X_out.T

    records_cross.append({
        'experiment':  exp,
        'n_ntc_in':    X_in.shape[0],
        'n_ntc_out':   X_out.shape[0],
        'mean_sim':    float(cross_sim.mean()),
        'std_sim':     float(cross_sim.std()),
        'min_sim':     float(cross_sim.min()),
        'max_sim':     float(cross_sim.max()),
    })
    records_cross_raw.append(cross_sim.flatten())

results_cross = pd.DataFrame(records_cross).set_index('experiment')
print("\nMean cosine similarity: NTCs of each experiment vs NTCs of all other experiments:")
print(results_cross.to_string(float_format=lambda x: f"{x:.4f}"))

# Overlay within- vs cross-experiment NTC distributions to quantify the batch gap
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(np.concatenate(records_cross_raw), bins=100, range=(-1, 1),
        color='steelblue', edgecolor='black', alpha=0.5, label='NTC cross-experiment', density=True)
ax.hist(np.concatenate(records_ntc_raw), bins=100, range=(-1, 1),
        color='lightcoral', edgecolor='black', alpha=0.5, label='NTC within-experiment', density=True)
ax.axvline(np.concatenate(records_cross_raw).mean(), color='steelblue', linestyle='--')
ax.axvline(np.concatenate(records_ntc_raw).mean(), color='lightcoral', linestyle='--')
ax.set_title('Cosine similarity distribution: NTCs within vs cross-experiment')
ax.set_xlabel('Cosine similarity')
ax.set_ylabel('Density')
ax.set_xlim(-1, 1)
ax.legend()
fig.tight_layout()

# %% Experiment-experiment mean embedding similarity heatmap
# Compute the mean L2-normalised embedding vector for each experiment (across all guides),
# then compute pairwise cosine similarity between those mean vectors. Experiments with
# similar mean embeddings cluster together on the heatmap — strong off-diagonal similarity
# indicates that the experiment-level mean dominates individual perturbation signals,
# i.e. residual batch structure not removed by per-feature z-scoring.
exp_means = np.stack([X_norm[experiments == exp].mean(axis=0) for exp in experiments_unique])
exp_means /= np.linalg.norm(exp_means, axis=1, keepdims=True) + 1e-10
sim_matrix = pd.DataFrame(exp_means @ exp_means.T, index=experiments_unique, columns=experiments_unique)

fig, ax = plt.subplots(figsize=(20, 15))
sns.heatmap(sim_matrix, vmin=0, vmax=1, cmap='Reds', annot=False, fmt='.2f', ax=ax)
ax.set_title('Mean embedding cosine similarity between experiments')
fig.tight_layout()
# %% UMAP of NTC guides coloured by experiment
# Project all NTC guide embeddings into 2D with UMAP to reveal whether any
# experiment's NTCs occupy a distinct region — a sign of residual batch structure
# that per-feature z-scoring did not remove.
import umap

X_ntc = X_norm[ntc_mask]
exp_ntc = experiments[ntc_mask]

reducer = umap.UMAP(n_components=2, random_state=42, metric='cosine')
embedding = reducer.fit_transform(X_ntc)

palette = sns.color_palette('husl', n_colors=n_exp)
exp_to_idx = {e: i for i, e in enumerate(experiments_unique)}

fig, ax = plt.subplots(figsize=(14, 10))
for exp in experiments_unique:
    mask_exp = exp_ntc == exp
    ax.scatter(embedding[mask_exp, 0], embedding[mask_exp, 1],
               s=10, alpha=0.6, label=exp, color=palette[exp_to_idx[exp]])
ax.set_title('UMAP of NTC guides coloured by experiment')
ax.set_xlabel('UMAP 1')
ax.set_ylabel('UMAP 2')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=6, ncol=2, markerscale=2)
fig.tight_layout()
# %%
