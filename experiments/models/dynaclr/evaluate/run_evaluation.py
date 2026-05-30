"""
Run mAP evaluation + silhouette scores + UMAP plots for DynaCLR predictions.
"""

# %%
import scanpy as sc
from pathlib import Path
from evaluate_map import (
    load_and_prepare,
    aggregate,
    aggregate_by_perturbation_reporter,
    marker_response_profile,
    within_complex_marker_consistency,
    plot_marker_response_heatmap,
    plot_umap_perturbation_reporter,
    reporter_silhouette_within_perturbation,
    phenotypic_activity,
    phenotypic_distinctiveness,
    compute_silhouette_by_complex,
    compute_batch_silhouette,
    plot_umap_by_complex,
    plot_map_volcano,
)

ROOT_DIR = Path(
    "/hpc/projects/intracellular_dashboard/ops/models/logs/dynaclr/ops_phase_only_self_256proj_10cplx_n_NTC/version_18/predict"
)

ZARR_PATH = (
    ROOT_DIR
    / "dynaclr_embeddings_ops_phase_only_self_256proj_10cplx_n_NTC_phase2d_ckpt008.zarr"
)
LABELS_PATH = "/home/eduardo.hirata/repos/ops_model/experiments/models/dynaclr/test/labels_testset_10complex_n_NTC_v2_filtered.parquet"

SAVE_DIR = Path(ROOT_DIR / "evaluate")

SAVE_DIR.mkdir(parents=True, exist_ok=True)

# 1) Load zarr + join sgRNA from labels
adata = load_and_prepare(ZARR_PATH, LABELS_PATH)

# 2) Batch effects diagnostic (cell-level)
print("\n--- Batch effects diagnostic ---")
n_pca = min(128, adata.shape[0] - 1, adata.shape[1] - 1)
sc.pp.pca(adata, n_comps=n_pca)
batch_sil = compute_batch_silhouette(adata, embedding_key="X_pca")

# 3) Aggregate
print("\n--- Aggregating to guide level ---")
adata_guide = aggregate(adata, level="guide")
print(f"Guide-level: {adata_guide.shape}")

print("\n--- Aggregating to gene level ---")
adata_gene = aggregate(adata, level="gene")
print(f"Gene-level: {adata_gene.shape}")

# 4) Silhouette by complex (gene-level)
print("\n--- Complex silhouette ---")
complex_sil = compute_silhouette_by_complex(adata_gene, embedding_key="X_pca")

# 5) UMAP per complex
print("\n--- UMAP per complex ---")
plot_umap_by_complex(adata_gene, save_path=str(SAVE_DIR / "umap_by_complex.png"))

# ---------- Marker-level analysis ----------

# 5a) Aggregate to (perturbation x reporter) level
print("\n--- Marker-level analysis ---")
adata_pr = aggregate_by_perturbation_reporter(adata)
print(f"(perturbation x reporter)-level: {adata_pr.shape}")

# 5a-ii) PCA + UMAP on (perturbation x reporter) embeddings
print("\n--- UMAP of (perturbation x reporter) embeddings ---")
n_pr_pca = min(128, adata_pr.shape[0] - 1, adata_pr.shape[1] - 1)
sc.pp.pca(adata_pr, n_comps=n_pr_pca)
sc.pp.neighbors(adata_pr, n_pcs=min(50, n_pr_pca), n_neighbors=15, metric="cosine")
sc.tl.umap(adata_pr)
plot_umap_perturbation_reporter(adata_pr, save_dir=SAVE_DIR)

# 5b) Marker response profiles (cosine distance to NTC per reporter)
print("\n--- Marker response profiles ---")
df_pivot, df_long = marker_response_profile(adata_pr)
df_pivot.to_csv(SAVE_DIR / "marker_response_pivot.csv")
df_long.to_csv(SAVE_DIR / "marker_response_long.csv", index=False)
print(df_pivot.to_markdown())

# 5c) Heatmap
plot_marker_response_heatmap(
    df_pivot, save_path=str(SAVE_DIR / "marker_response_heatmap.png")
)

# 5d) Within-complex marker consistency
print("\n--- Within-complex marker consistency ---")
complex_consistency = within_complex_marker_consistency(df_pivot)
complex_consistency.to_csv(
    SAVE_DIR / "within_complex_marker_consistency.csv", index=False
)
print(
    complex_consistency[
        ["complex", "n_members", "mean_pairwise_corr", "member_genes"]
    ].to_markdown(index=False)
)

# 5e) Reporter silhouette within perturbation
print("\n--- Reporter silhouette within perturbation ---")
reporter_sil = reporter_silhouette_within_perturbation(adata)
reporter_sil.to_csv(
    SAVE_DIR / "reporter_silhouette_within_perturbation.csv", index=False
)
if len(reporter_sil) > 0:
    print(
        reporter_sil[
            ["perturbation", "complex", "reporter_silhouette", "n_reporters"]
        ].to_markdown(index=False)
    )
else:
    print("Skipped: not enough reporters per perturbation")

# ------------------------------------------

# 6) Phenotypic activity
print("\n--- Phenotypic activity ---")
activity_map = phenotypic_activity(adata_guide)
active_ratio = activity_map["below_corrected_p"].mean()
print(f"Active ratio: {100 * active_ratio:.1f}%")
print(
    activity_map[
        [
            "perturbation",
            "mean_average_precision",
            "-log10(p-value)",
            "below_corrected_p",
        ]
    ].to_markdown(index=False)
)
activity_map.to_csv(SAVE_DIR / "phenotypic_activity.csv", index=False)
plot_map_volcano(
    activity_map,
    "Phenotypic Activity",
    save_path=str(SAVE_DIR / "phenotypic_activity.png"),
)

# 7) Phenotypic distinctiveness
print("\n--- Phenotypic distinctiveness ---")
dist_map = phenotypic_distinctiveness(adata_guide, activity_map)
dist_ratio = dist_map["below_corrected_p"].mean()
print(f"Distinctive ratio: {100 * dist_ratio:.1f}%")
print(
    dist_map[
        [
            "perturbation",
            "mean_average_precision",
            "-log10(p-value)",
            "below_corrected_p",
        ]
    ].to_markdown(index=False)
)
dist_map.to_csv(SAVE_DIR / "phenotypic_distinctiveness.csv", index=False)
plot_map_volcano(
    dist_map,
    "Phenotypic Distinctiveness",
    save_path=str(SAVE_DIR / "phenotypic_distinctiveness.png"),
)

# Summary
print("\n## Summary")
print("| Metric | Value |")
print("|--------|-------|")
print(f"| Total cells | {adata.shape[0]:,} |")
print(f"| Perturbations | {adata.obs['perturbation'].nunique()} |")
print(f"| sgRNAs | {adata.obs['sgRNA'].nunique()} |")
print(f"| Reporters | {adata.obs['reporter'].nunique()} |")
print(f"| Batch silhouette (experiment) | {batch_sil:.3f} |")
print(f"| Complex silhouette (gene-level) | {complex_sil['overall']:.3f} |")
mean_rep_sil = (
    reporter_sil["reporter_silhouette"].mean()
    if len(reporter_sil) > 0
    else float("nan")
)
mean_cpx_corr = (
    complex_consistency["mean_pairwise_corr"].mean()
    if len(complex_consistency) > 0
    else float("nan")
)
print(f"| Mean reporter silhouette (within-pert) | {mean_rep_sil:.3f} |")
print(f"| Mean within-complex marker corr | {mean_cpx_corr:.3f} |")
print(f"| Phenotypically active | {100 * active_ratio:.1f}% |")
print(f"| Phenotypically distinctive | {100 * dist_ratio:.1f}% |")

if complex_sil["per_complex"]:
    print("\n### Per-complex silhouette")
    print("| Complex | Silhouette |")
    print("|---------|------------|")
    for cpx, score in sorted(complex_sil["per_complex"].items()):
        print(f"| {cpx} | {score:.3f} |")

print(f"\nResults saved to {SAVE_DIR}")

# %%
