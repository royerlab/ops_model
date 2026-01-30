"""
Example script demonstrating how to use the centralized report generation system.

This script shows the complete workflow:
1. Create a report directory
2. Generate UMAP plots
3. Generate Funk cluster plots
4. Compute embedding quality metrics
5. Save metadata
6. Generate PDF report

Usage:
    python example_generate_report.py --feature_dir /path/to/features --feature_type dinov3
"""

import argparse
from pathlib import Path
import anndata as ad

from ops_model.data.embeddings.report_manager import ReportManager
from ops_model.data.embeddings import umap_plots, funk_clusters, pca
from ops_model.data.embeddings.embedding_metrics import (
    save_alignment_uniformity_metrics,
)
from ops_model.data.embeddings.cosine_similarity import (
    save_embedding_spread_metrics,
    save_similarity_to_reference_metrics,
)
from ops_model.data.embeddings.generate_report import generate_report
from ops_model.data.embeddings.funk_clusters import funk_clusters as FUNK_CLUSTERS


def generate_full_report_example(
    feature_dir: str,
    feature_type: str = "dinov3",
    experiment_name: str = None,
    channel: str = "Phase2D",
    channels: list = None,
):
    """
    Complete example of generating a report with all visualizations and metrics.

    Args:
        feature_dir: Path to feature directory containing anndata_objects/
                     e.g., /path/to/ops0031/3-assembly/dino_features
        feature_type: Type of features (dinov3, cellprofiler, etc.)
        experiment_name: Optional experiment name
        channel: Channel to load (e.g., "Phase2D", "GFP", "mCherry")
        channels: Optional list of imaging channels for metadata
    """
    feature_dir = Path(feature_dir)

    # Extract experiment name from path if not provided
    if experiment_name is None:
        # Typical path: .../ops0031_20251119/3-assembly/dino_features
        try:
            experiment_name = feature_dir.parents[1].name
        except IndexError:
            experiment_name = "unknown_experiment"

    print("=" * 80)
    print("EMBEDDING ANALYSIS REPORT GENERATION EXAMPLE")
    print("=" * 80)
    print(f"\nFeature directory: {feature_dir}")
    print(f"Feature type: {feature_type}")
    print(f"Experiment: {experiment_name}")
    print(f"Channel: {channel}")

    # ===========================================================================
    # STEP 1: Create Report Directory
    # ===========================================================================
    print("\n" + "-" * 80)
    print("STEP 1: Creating report directory")
    print("-" * 80)

    mgr = ReportManager()
    report_dir = mgr.create_report_dir(
        feature_type=feature_type,
        experiments=[experiment_name],
    )
    print(f"✓ Report directory created: {report_dir}")

    # ===========================================================================
    # STEP 2: Load AnnData Objects
    # ===========================================================================
    print("\n" + "-" * 80)
    print(f"STEP 2: Loading AnnData objects")
    print("-" * 80)

    anndata_dir = feature_dir / "anndata_objects"
    if not anndata_dir.exists():
        raise FileNotFoundError(f"AnnData directory not found: {anndata_dir}")

    # Load cell-level data (channel-specific)
    cell_path = anndata_dir / f"features_processed_{channel}.h5ad"
    if cell_path.exists():
        print(f"Loading cell-level data: {cell_path}")
        adata_cells = ad.read_h5ad(cell_path)
        print(
            f"  ✓ Loaded {adata_cells.shape[0]} cells × {adata_cells.shape[1]} features"
        )
    else:
        print(f"⚠ Cell-level data not found: {cell_path}")
        adata_cells = None

    # Load guide-level data (channel-specific)
    guide_path = anndata_dir / f"guide_bulked_umap_{channel}.h5ad"
    if guide_path.exists():
        print(f"Loading guide-level data: {guide_path}")
        adata_guides = ad.read_h5ad(guide_path)
        print(f"  ✓ Loaded {adata_guides.shape[0]} guides")
    else:
        print(f"⚠ Guide-level data not found: {guide_path}")
        adata_guides = None

    # Load gene-level data (channel-specific)
    gene_path = anndata_dir / f"gene_bulked_umap_{channel}.h5ad"
    if gene_path.exists():
        print(f"Loading gene-level data: {gene_path}")
        adata_genes = ad.read_h5ad(gene_path)
        print(f"  ✓ Loaded {adata_genes.shape[0]} genes")
    else:
        print(f"⚠ Gene-level data not found: {gene_path}")
        adata_genes = None

    # ===========================================================================
    # STEP 3: Generate UMAP Plots
    # ===========================================================================
    print("\n" + "-" * 80)
    print("STEP 3: Generating UMAP visualizations")
    print("-" * 80)

    if adata_cells and adata_guides and adata_genes:
        # Generate NTC control plots
        print("Generating NTC control plots...")
        umap_plots.report_umap_plot_1(
            feature_dir=str(feature_dir),
            adata_cells=adata_cells,
            adata_guides=adata_guides,
            adata_genes=adata_genes,
            report_dir=str(report_dir),
        )
        print("  ✓ NTC plots saved")

        # Generate protein complex plots
        print("Generating protein complex plots...")
        umap_plots.report_umap_plot_2(
            feature_dir=str(feature_dir),
            adata_cells=adata_cells,
            adata_guides=adata_guides,
            adata_genes=adata_genes,
            report_dir=str(report_dir),
        )
        print("  ✓ Protein complex plots saved")
    else:
        print("⚠ Skipping UMAP plots (missing required AnnData objects)")

    # ===========================================================================
    # STEP 4: Generate Funk Cluster Plots
    # ===========================================================================
    print("\n" + "-" * 80)
    print("STEP 4: Generating Funk functional cluster plots")
    print("-" * 80)

    if adata_genes and "X_umap" in adata_genes.obsm:
        print("Generating Funk clusters plot...")
        funk_clusters.plot_funk_clusters(
            adata=adata_genes,
            funk_clusters=FUNK_CLUSTERS,
            report_dir=str(report_dir),
            filename="funk_clusters.png",
        )
        print("  ✓ Funk clusters plot saved")
    else:
        print("⚠ Skipping Funk clusters (gene-level data or UMAP not available)")

    # ===========================================================================
    # STEP 5: Generate PCA Plots
    # ===========================================================================
    print("\n" + "-" * 80)
    print("STEP 5: Generating PCA variance plots")
    print("-" * 80)

    if adata_cells and "X_pca" in adata_cells.obsm:
        print("Generating PCA variance plot...")
        pca.plot_pca(
            adata=adata_cells,
            report_dir=str(report_dir),
            filename="pca_variance_ratio.png",
            description="PCA Variance Explained",
        )
        print("  ✓ PCA plot saved")
    elif adata_guides and "X_pca" in adata_guides.obsm:
        print("Generating PCA variance plot (from guide-level data)...")
        pca.plot_pca(
            adata=adata_guides,
            report_dir=str(report_dir),
            filename="pca_variance_ratio.png",
            description="PCA Variance Explained (Guide-level)",
        )
        print("  ✓ PCA plot saved")
    else:
        print("⚠ Skipping PCA plot (no PCA embeddings available)")

    # ===========================================================================
    # STEP 6: Compute Embedding Quality Metrics
    # ===========================================================================
    print("\n" + "-" * 80)
    print("STEP 6: Computing embedding quality metrics")
    print("-" * 80)

    # Use cell-level data for metrics (if available)
    metrics_adata = adata_cells if adata_cells is not None else adata_genes

    if metrics_adata is not None:
        # Alignment and uniformity metrics
        try:
            print("Computing alignment and uniformity metrics...")
            save_alignment_uniformity_metrics(
                adata=metrics_adata,
                report_dir=str(report_dir),
                n_uniformity_samples=100_000,  # Reduced for faster computation
                batch_size=5000,
            )
            print("  ✓ Alignment and uniformity metrics saved")
        except Exception as e:
            print(f"  ⚠ Could not compute alignment/uniformity: {e}")

        # Embedding spread metrics
        try:
            print("Computing embedding spread metrics...")
            save_embedding_spread_metrics(
                adata=metrics_adata,
                report_dir=str(report_dir),
                min_samples=2,
            )
            print("  ✓ Embedding spread metrics saved")
        except Exception as e:
            print(f"  ⚠ Could not compute embedding spread: {e}")

        # Similarity to NTC reference
        if adata_genes is not None:
            try:
                print("Computing similarity to NTC reference...")
                save_similarity_to_reference_metrics(
                    adata=adata_genes,
                    reference_label="NTC",
                    report_dir=str(report_dir),
                )
                print("  ✓ Similarity to NTC metrics saved")
            except Exception as e:
                print(f"  ⚠ Could not compute similarity to NTC: {e}")
    else:
        print("⚠ Skipping metrics computation (no AnnData objects available)")

    # ===========================================================================
    # STEP 7: Update Metadata
    # ===========================================================================
    print("\n" + "-" * 80)
    print("STEP 7: Updating report metadata")
    print("-" * 80)

    # Collect information about generated files
    plots_dir = report_dir / "plots"
    metrics_dir = report_dir / "metrics"

    plot_files = [f.name for f in plots_dir.glob("*.png")] if plots_dir.exists() else []
    metric_files = (
        [f.name for f in metrics_dir.glob("*.csv")] if metrics_dir.exists() else []
    )

    # Detect CSV source path
    csv_pattern = feature_dir / f"{feature_type}_features*.csv"
    csv_files = list(feature_dir.glob(f"{feature_type}_features*.csv"))
    data_source = str(csv_files[0]) if csv_files else str(feature_dir / "features.csv")

    # Update metadata with complete information
    metadata_updates = {
        "feature_type": feature_type,
        "experiments": [experiment_name],
        "channels": channels if channels else ["Phase2D"],
        "data_sources": [data_source],
        "anndata_sources": {
            "cell_level": str(cell_path) if cell_path.exists() else None,
            "guide_level": str(guide_path) if guide_path.exists() else None,
            "gene_level": str(gene_path) if gene_path.exists() else None,
        },
        "processing": {
            "normalize_embeddings": False,
            "n_pca_components": 128,
            "n_umap_neighbors": 15,
            "aggregation_method": "mean",
        },
        "output_files": {
            "plots": [f"plots/{f}" for f in plot_files],
            "metrics": [f"metrics/{f}" for f in metric_files],
            "report_pdf": "embedding_report.pdf",
        },
    }

    mgr.update_metadata(report_dir, metadata_updates)
    print(
        f"  ✓ Metadata updated with {len(plot_files)} plots and {len(metric_files)} metrics"
    )

    # ===========================================================================
    # STEP 8: Generate PDF Report
    # ===========================================================================
    print("\n" + "-" * 80)
    print("STEP 8: Generating PDF report")
    print("-" * 80)

    pdf_path = generate_report(
        report_dir=report_dir,
        output_filename="embedding_report.pdf",
    )
    print(f"  ✓ PDF report generated: {pdf_path}")

    # ===========================================================================
    # SUMMARY
    # ===========================================================================
    print("\n" + "=" * 80)
    print("REPORT GENERATION COMPLETE")
    print("=" * 80)
    print(f"\nReport directory: {report_dir}")
    print(f"  - Plots: {len(plot_files)}")
    print(f"  - Metrics: {len(metric_files)}")
    print(f"  - PDF: {pdf_path}")
    print(f"\nView report with:")
    print(f"  open {pdf_path}  # macOS")
    print(f"  xdg-open {pdf_path}  # Linux")
    print("=" * 80)

    return report_dir


def main():
    """Command-line interface for example script."""
    parser = argparse.ArgumentParser(
        description="Example: Generate complete embedding analysis report"
    )
    parser.add_argument(
        "--feature_dir",
        type=str,
        required=True,
        help="Path to feature directory containing anndata_objects/ subdirectory",
    )
    parser.add_argument(
        "--feature_type",
        type=str,
        choices=["dinov3", "cellprofiler", "cytoself", "dynaclr"],
        required=True,
        help="Type of features",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="Experiment name (default: auto-detect from path)",
    )
    parser.add_argument(
        "--channel",
        type=str,
        default="Phase2D",
        help="Channel to load (default: Phase2D)",
    )
    parser.add_argument(
        "--channels",
        nargs="+",
        default=None,
        help="Imaging channels for metadata (e.g., Phase2D GFP mCherry)",
    )

    args = parser.parse_args()

    try:
        report_dir = generate_full_report_example(
            feature_dir=args.feature_dir,
            feature_type=args.feature_type,
            experiment_name=args.experiment,
            channel=args.channel,
            channels=args.channels,
        )
        print(f"\n✓ Success! Report available at: {report_dir}")
    except Exception as e:
        print(f"\n✗ Error generating report: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
