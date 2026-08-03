#%%
# %% [markdown]
# Compare evaluation metrics across cell_dino, cellprofiler, and dino models.

# %%
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

plt.rcParams["svg.fonttype"] = "none"

FIGURES_DIR = Path("/hpc/mydata/alexander.hillsley/ops/ops_monorepo/ops_model/analysis/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

METRICS_DIRS = {
    "cell_dino":    "/hpc/projects/icd.fast.ops/organelle_attribution/pca_optimized_v0.3/cell_dino/zscore_per_exp/paper_v1/all_livecell/fixed_80%/cosine/second_pca_consensus/metrics",
    "dino":         "/hpc/projects/icd.fast.ops/organelle_attribution/pca_optimized_v0.3/dino/zscore_per_exp/all/fixed_80%/cosine/metrics",
    "cell_profiler":"/hpc/projects/icd.fast.ops/experiments/evaluations/cell_profiler/all/final_eval",
    "dynaclr":      "/hpc/projects/icd.fast.ops/experiments/evaluations/dynaclr/all/final_eval",
    "subcell":      "/hpc/projects/icd.fast.ops/experiments/evaluations/subcell/all/final_eval",
}
MODELS = list(METRICS_DIRS.keys())

records = {}
for model, metrics_dir in METRICS_DIRS.items():
    activity        = pd.read_csv(f"{metrics_dir}/phenotypic_activity.csv")
    distinctiveness = pd.read_csv(f"{metrics_dir}/phenotypic_distinctiveness.csv")
    manual          = pd.read_csv(f"{metrics_dir}/phenotypic_consistency_manual.csv")
    corum           = pd.read_csv(f"{metrics_dir}/phenotypic_consistency_corum.csv")

    records[model] = {
        "pct_perturbations_active":         activity["below_corrected_p"].mean(),
        "mean_map_active":                  activity["mean_average_precision"].mean(),
        "pct_perturbations_distinct":       distinctiveness["below_corrected_p"].mean(),
        "mean_map_distinct":                distinctiveness["mean_average_precision"].mean(),
        "pct_complexes_significant_manual": manual["below_corrected_p"].mean(),
        "mean_map_complexes_manual":        manual["mean_average_precision"].mean(),
        "pct_complexes_significant_corum":  corum["below_corrected_p"].mean(),
        "mean_map_complexes_corum":         corum["mean_average_precision"].mean(),
    }

df = pd.DataFrame(records).T

# %%
GROUPS = {
    "Perturbation activity": [
        "pct_perturbations_active",
        "mean_map_active",
    ],
    "Distinct perturbations": [
        "pct_perturbations_distinct",
        "mean_map_distinct",
    ],
    "Complexes (manual)": [
        "pct_complexes_significant_manual",
        "mean_map_complexes_manual",
    ],
    "Complexes (CORUM)": [
        "pct_complexes_significant_corum",
        "mean_map_complexes_corum",
    ],
}

METRIC_LABELS = {
    "pct_perturbations_active":         "% active",
    "mean_map_active":                  "active",
    "pct_perturbations_distinct":       "% distinct",
    "mean_map_distinct":                "distinct",
    "pct_complexes_significant_manual": "% significant",
    "mean_map_complexes_manual":        "CHAD",
    "pct_complexes_significant_corum":  "% significant",
    "mean_map_complexes_corum":         "CORUM",
}

# %%
n_groups = len(GROUPS)
fig, axes = plt.subplots(1, n_groups, figsize=(4 * n_groups, 5))

n_models = len(MODELS)
colors = plt.cm.tab10(np.linspace(0, 0.9, n_models))
bar_w = 0.8 / n_models

for ax, (group_name, metrics) in zip(axes, GROUPS.items()):
    x_base = np.arange(len(metrics))
    for i, (model, color) in enumerate(zip(MODELS, colors)):
        vals = [df.loc[model, m] if m in df.columns else np.nan for m in metrics]
        offset = (i - (n_models - 1) / 2) * bar_w
        ax.bar(x_base + offset, vals, width=bar_w * 0.9, label=model, color=color)

    ax.set_title(group_name, fontsize=10, fontweight="bold")
    ax.set_xticks(x_base)
    ax.set_xticklabels([METRIC_LABELS[m] for m in metrics], fontsize=8)
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(axis="y", alpha=0.3)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=n_models,
           bbox_to_anchor=(0.5, -0.05), fontsize=9, frameon=False)

fig.suptitle("Model evaluation comparison — cell_dino / dino / cell_profiler / dynaclr / subcell",
             fontsize=13, fontweight="bold", y=1.02)
fig.tight_layout()

# out = "experiments/scratch/2026-04-24_eval_comparison.png"
# fig.savefig(out, dpi=150, bbox_inches="tight")
# print(f"Saved → {out}")
plt.show()

# %%
fig, ax = plt.subplots(figsize=(6, 5))

metrics_2 = ["mean_map_active", "mean_map_distinct", "mean_map_complexes_manual", "mean_map_complexes_corum"]
x = np.arange(len(metrics_2))
bar_w = 0.8 / n_models

for i, (model, color) in enumerate(zip(MODELS, colors)):
    offset = (i - (n_models - 1) / 2) * bar_w
    vals = [df.loc[model, m] for m in metrics_2]
    ax.bar(x + offset, vals, width=bar_w * 0.9, label=model, color=color)

ax.set_xticks(x)
ax.set_xticklabels([METRIC_LABELS[m] for m in metrics_2], fontsize=10)
ax.set_ylabel("mAP", fontsize=10)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
ax.grid(axis="y", alpha=0.3)
ax.legend(fontsize=9, frameon=False, bbox_to_anchor=(0.75, 0.99), loc="upper left")
# ax.set_title("mAP active vs distinct by model", fontsize=12, fontweight="bold")

fig.tight_layout()
fig.savefig(FIGURES_DIR / "model_comparison_map.svg", bbox_inches="tight")
plt.show()

# %%
EMBEDDING_DIRS = {
    "all_combined": "/hpc/projects/icd.fast.ops/organelle_attribution/pca_optimized_v0.3/cell_dino/zscore_per_exp/paper_v1/all_with_autofluorescence/fixed_80%/cosine/second_pca_consensus/metrics",
    "live-cell all":    "/hpc/projects/icd.fast.ops/organelle_attribution/pca_optimized_v0.3/cell_dino/zscore_per_exp/all/fixed_80%/cosine/metrics",
    "phase_only":       "/hpc/projects/icd.fast.ops/organelle_attribution/pca_optimized_v0.3/cell_dino/zscore_per_exp/paper_v1/phase_only/fixed_80%/cosine/second_pca_consensus/metrics",
    "live-fluorescence":"/hpc/projects/icd.fast.ops/organelle_attribution/pca_optimized_v0.3/cell_dino/zscore_per_exp/paper_v1/no_phase/fixed_80%/cosine/second_pca_consensus/metrics",
    "cell_painting":"/hpc/projects/icd.fast.ops/organelle_attribution/pca_optimized_v0.3/cell_dino/zscore_per_exp/only_cp/all/fixed_80%/cosine/second_pca_consensus/metrics",
    "4i":           "/hpc/projects/icd.fast.ops/organelle_attribution/pca_optimized_v0.3/cell_dino/zscore_per_exp/only_4i/all/fixed_80%/cosine/second_pca_consensus/metrics",
}
EMBEDDINGS = list(EMBEDDING_DIRS.keys())

emb_records = {}
for emb, metrics_dir in EMBEDDING_DIRS.items():
    activity        = pd.read_csv(f"{metrics_dir}/phenotypic_activity.csv")
    distinctiveness = pd.read_csv(f"{metrics_dir}/phenotypic_distinctiveness.csv")
    manual          = pd.read_csv(f"{metrics_dir}/phenotypic_consistency_manual.csv")
    corum           = pd.read_csv(f"{metrics_dir}/phenotypic_consistency_corum.csv")
    emb_records[emb] = {
        "mean_map_active":           activity["mean_average_precision"].mean(),
        "mean_map_distinct":         distinctiveness["mean_average_precision"].mean(),
        "mean_map_complexes_manual": manual["mean_average_precision"].mean(),
        "mean_map_complexes_corum":  corum["mean_average_precision"].mean(),
    }

emb_df = pd.DataFrame(emb_records).T

metrics_emb = ["mean_map_active", "mean_map_distinct", "mean_map_complexes_manual", "mean_map_complexes_corum"]
x_emb = np.arange(len(metrics_emb))

SUBSETS = [
    ("Cell-painting vs 4i",          ["all_combined", "cell_painting", "4i"]),
    ("Live-cell channel subsets",    ["all_combined", "live-cell all", "phase_only", "live-fluorescence"]),
]

for title, subset in SUBSETS:
    n_emb = len(subset)
    emb_colors = plt.cm.tab10(np.linspace(0, 0.9, n_emb))
    emb_bar_w = 0.8 / n_emb

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (emb, color) in enumerate(zip(subset, emb_colors)):
        offset = (i - (n_emb - 1) / 2) * emb_bar_w
        vals = [emb_df.loc[emb, m] for m in metrics_emb]
        ax.bar(x_emb + offset, vals, width=emb_bar_w * 0.9, label=emb, color=color)

    ax.set_xticks(x_emb)
    ax.set_xticklabels([METRIC_LABELS[m] for m in metrics_emb], fontsize=10)
    ax.set_ylabel("mAP", fontsize=10)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=9, frameon=False, bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.set_title(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    plt.show()

# %%
