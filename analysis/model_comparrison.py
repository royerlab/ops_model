#%%
# %% [markdown]
# Compare evaluation metrics across cell_dino, cellprofiler, and dino models.

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

EVAL_ROOT = "/hpc/projects/icd.fast.ops/organelle_attribution/pca_optimized_v0.3"
MODELS = ["cell_dino", "dino"]

records = {}
for model in MODELS:
    metrics_dir = f"{EVAL_ROOT}/{model}/zscore_per_exp/all/fixed_80%/cosine/metrics"
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
    "mean_map_active":                  "mAP active",
    "pct_perturbations_distinct":       "% distinct",
    "mean_map_distinct":                "mAP distinct",
    "pct_complexes_significant_manual": "% significant",
    "mean_map_complexes_manual":        "mAP",
    "pct_complexes_significant_corum":  "% significant",
    "mean_map_complexes_corum":         "mAP",
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

fig.suptitle("Model evaluation comparison — cell_dino / dino",
             fontsize=13, fontweight="bold", y=1.02)
fig.tight_layout()

# out = "experiments/scratch/2026-04-24_eval_comparison.png"
# fig.savefig(out, dpi=150, bbox_inches="tight")
# print(f"Saved → {out}")
plt.show()

# %%
