"""
Run mAP evaluation + silhouette scores + UMAP plots for DynaCLR predictions.
"""

# %%
import scanpy as sc
from pathlib import Path
from evaluate_map import (
    load_and_prepare,
    aggregate,
    phenotypic_activity,
    phenotypic_distinctiveness,
    compute_silhouette_by_complex,
    compute_batch_silhouette,
    plot_umap_by_complex,
    plot_map_volcano,
)

ZARR_PATH = "/hpc/projects/intracellular_dashboard/ops/models/logs/dynaclr/ops_phase_only_self_256proj_v1/version_5/predict/dynaclr_embeddings_ops_phase_only_self_256proj_v5_ckpt109_2.zarr"
LABELS_PATH = "/home/eduardo.hirata/repos/ops_model/experiments/models/dynaclr/phase_only/labels_testset_filtered_phase2d.csv"
SAVE_DIR = Path(
    "/hpc/projects/intracellular_dashboard/ops/models/logs/dynaclr/ops_phase_only_self_256proj_v1/version_5/predict/dynaclr_embeddings_ops_phase_only_self_256proj_v5_ckpt109_2"
)

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
print(f"| Batch silhouette (experiment) | {batch_sil:.3f} |")
print(f"| Complex silhouette (gene-level) | {complex_sil['overall']:.3f} |")
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
