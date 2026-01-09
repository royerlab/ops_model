"""
Generate PDF report for single-cell embedding verification.

Takes a report directory (with plots/, metrics/, and report_metadata.yml)
and compiles them into a single PDF report with organized sections.

Updated to work with centralized report directory structure:
    report_dir/
        plots/
        metrics/
        report_metadata.yml
        embedding_report.pdf  ← output
"""

import argparse
from pathlib import Path
from typing import Dict, Optional, Any, Union
import pandas as pd
import yaml
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from PIL import Image
from datetime import datetime


# Expected plot patterns for matching files
PLOT_PATTERNS = {
    "umap_cells": ["umap_all_cells", "umap_cell", "umap_cell_ntc"],
    "umap_guide_bulked": ["umap_guide_bulked", "guide_bulked", "umap_guide_ntc"],
    "umap_protein_complex": ["umap_protein_complex", "protein_complex", "umap_complex"],
    "pca_variance_ratio": [
        "pca_variance_ratio",
        "pca_variance",
        "variance_ratio",
        "scree",
    ],
}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate embedding analysis report from report directory"
    )
    parser.add_argument(
        "--report_dir",
        type=str,
        required=True,
        help="Report directory containing plots/, metrics/, and report_metadata.yml",
    )
    parser.add_argument(
        "--output_filename",
        type=str,
        default="embedding_report.pdf",
        help="Output filename for PDF report (saved in report_dir)",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Title for the report (defaults to using metadata)",
    )
    return parser.parse_args()


def load_plots(
    report_dir: Union[str, Path], plots_subdir: str = "plots"
) -> Dict[str, Path]:
    """
    Load all plot files from report directory.

    Args:
        report_dir: Path to report directory
        plots_subdir: Subdirectory containing plots (default: "plots")

    Expected plot types:
    - umap_cell_ntc.png: UMAP of cells with NTC highlighted
    - umap_guide_ntc.png: UMAP of guides with NTC
    - umap_gene_ntc.png: UMAP of genes with NTC
    - umap_rpl_genes.png: UMAP of ribosomal protein genes
    - umap_nup_genes.png: UMAP of nuclear pore complex genes
    - funk_clusters.png: Funk functional cluster analysis
    - pca_variance_ratio.png: PCA variance explained plot

    Returns:
        Dictionary mapping plot types to file paths
    """
    report_dir = Path(report_dir)
    plots_dir = report_dir / plots_subdir

    if not plots_dir.exists():
        print(f"Warning: Plots directory not found: {plots_dir}")
        return {}

    plots = {}

    # Supported image formats
    image_extensions = [".png", ".pdf", ".jpg", ".jpeg"]

    # Find all image files in plots directory
    all_plots = []
    for ext in image_extensions:
        all_plots.extend(plots_dir.glob(f"*{ext}"))

    # Categorize plots
    for plot_file in all_plots:
        filename_lower = plot_file.stem.lower()

        # Categorize by type
        if "ntc" in filename_lower or "nontarget" in filename_lower:
            if "cell" in filename_lower:
                plots["umap_cell_ntc"] = plot_file
            elif "guide" in filename_lower:
                plots["umap_guide_ntc"] = plot_file
            elif "gene" in filename_lower:
                plots["umap_gene_ntc"] = plot_file
            else:
                plots["umap_ntc"] = plot_file

        elif "rpl" in filename_lower or "ribosom" in filename_lower:
            plots["umap_rpl"] = plot_file

        elif "nup" in filename_lower or "nuclear_pore" in filename_lower:
            plots["umap_nup"] = plot_file

        elif "trappc" in filename_lower:
            plots["umap_trappc"] = plot_file

        elif "krt" in filename_lower or "keratin" in filename_lower:
            plots["umap_krt"] = plot_file

        elif "funk" in filename_lower or "cluster" in filename_lower:
            plots["funk_clusters"] = plot_file

        elif "pca" in filename_lower and "variance" in filename_lower:
            plots["pca_variance"] = plot_file

        elif "spread" in filename_lower:
            plots["embedding_spread"] = plot_file

        elif "similarity" in filename_lower and "reference" in filename_lower:
            plots["similarity_to_reference"] = plot_file

        else:
            # Add as "other" with filename as key
            plots[f"other_{plot_file.stem}"] = plot_file

    # Log what was found
    print(f"Found {len(plots)} plot file(s):")
    for key, path in plots.items():
        print(f"  - {key}: {path.name}")

    return plots


def load_metrics(
    report_dir: Union[str, Path], metrics_subdir: str = "metrics"
) -> Dict[str, Any]:
    """
    Load metric files (YAML or CSV) from report directory.

    Args:
        report_dir: Path to report directory
        metrics_subdir: Subdirectory containing metrics (default: "metrics")

    Expected metrics files:
    - alignment_uniformity.csv: Alignment and uniformity scores
    - embedding_spread.csv: Embedding spread metrics per label
    - similarity_to_reference.csv: Similarity to reference label (e.g., NTC)

    Returns:
        Dictionary mapping metric types to DataFrames or dicts
    """
    report_dir = Path(report_dir)
    metrics_dir = report_dir / metrics_subdir

    if not metrics_dir.exists():
        print(f"Warning: Metrics directory not found: {metrics_dir}")
        return {}

    metrics = {}

    # Load YAML metrics files (legacy support)
    yaml_files = list(metrics_dir.glob("*.yaml")) + list(metrics_dir.glob("*.yml"))
    for yaml_file in yaml_files:
        try:
            with open(yaml_file, "r") as f:
                data = yaml.safe_load(f)
                # Convert to DataFrame for easier display
                if isinstance(data, dict):
                    df = pd.DataFrame([data]).T
                    df.columns = ["Value"]
                    metrics[yaml_file.stem] = df
                print(f"  Loaded YAML metrics: {yaml_file.name}")
        except Exception as e:
            print(f"  Warning: Could not load {yaml_file.name}: {e}")

    # Load CSV metrics files
    csv_files = list(metrics_dir.glob("*.csv"))
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            metrics[csv_file.stem] = df
            print(f"  Loaded CSV metrics: {csv_file.name}")
        except Exception as e:
            print(f"  Warning: Could not load {csv_file.name}: {e}")

    print(f"Found {len(metrics)} metric file(s)")
    return metrics


def load_report_metadata(report_dir: Union[str, Path]) -> Optional[Dict]:
    """
    Load report metadata from report_metadata.yml file.

    Args:
        report_dir: Path to report directory

    Returns:
        Dictionary with metadata, or None if file doesn't exist
    """
    report_dir = Path(report_dir)
    metadata_file = report_dir / "report_metadata.yml"

    if not metadata_file.exists():
        print(f"Warning: Metadata file not found: {metadata_file}")
        return None

    try:
        with open(metadata_file, "r") as f:
            metadata = yaml.safe_load(f)
        print(f"Loaded report metadata: {metadata_file}")
        return metadata
    except Exception as e:
        print(f"Warning: Could not load metadata: {e}")
        return None


def create_title_page(pdf: PdfPages, title: str, metadata: Optional[Dict] = None):
    """
    Create title page with report metadata.

    Args:
        pdf: PdfPages object
        title: Report title
        metadata: Optional metadata dict from report_metadata.yml
    """
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")

    # Report title
    ax.text(
        0.5,
        0.75,
        title,
        ha="center",
        va="center",
        fontsize=24,
        fontweight="bold",
        transform=ax.transAxes,
    )

    # Add metadata if provided
    if metadata:
        y_pos = 0.60
        line_spacing = 0.04

        # Report name
        if "report_name" in metadata:
            ax.text(
                0.5,
                y_pos,
                f"Report: {metadata['report_name']}",
                ha="center",
                va="top",
                fontsize=12,
                transform=ax.transAxes,
            )
            y_pos -= line_spacing

        # Creation date
        if "created_date" in metadata:
            date_str = metadata["created_date"]
            if isinstance(date_str, str):
                try:
                    # Try to parse ISO format
                    dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                    date_str = dt.strftime("%Y-%m-%d %H:%M:%S")
                except:
                    pass
            ax.text(
                0.5,
                y_pos,
                f"Generated: {date_str}",
                ha="center",
                va="top",
                fontsize=11,
                transform=ax.transAxes,
            )
            y_pos -= line_spacing * 1.5

        # Feature type
        if "feature_type" in metadata:
            ax.text(
                0.5,
                y_pos,
                f"Feature Type: {metadata['feature_type'].upper()}",
                ha="center",
                va="top",
                fontsize=12,
                fontweight="bold",
                transform=ax.transAxes,
            )
            y_pos -= line_spacing

        # Experiments
        if "experiments" in metadata:
            experiments = metadata["experiments"]
            if isinstance(experiments, list):
                exp_str = ", ".join(experiments)
            else:
                exp_str = str(experiments)
            ax.text(
                0.5,
                y_pos,
                f"Experiments: {exp_str}",
                ha="center",
                va="top",
                fontsize=10,
                transform=ax.transAxes,
            )
            y_pos -= line_spacing

        # Channels
        if "channels" in metadata:
            channels = metadata["channels"]
            if isinstance(channels, list):
                ch_str = ", ".join(channels)
            else:
                ch_str = str(channels)
            ax.text(
                0.5,
                y_pos,
                f"Channels: {ch_str}",
                ha="center",
                va="top",
                fontsize=10,
                transform=ax.transAxes,
            )
            y_pos -= line_spacing * 2

        # Processing parameters summary
        if "processing" in metadata:
            ax.text(
                0.5,
                y_pos,
                "Processing Parameters:",
                ha="center",
                va="top",
                fontsize=10,
                fontweight="bold",
                transform=ax.transAxes,
            )
            y_pos -= line_spacing

            proc = metadata["processing"]
            if isinstance(proc, dict):
                for key, value in list(proc.items())[:5]:  # Show first 5 params
                    ax.text(
                        0.5,
                        y_pos,
                        f"  {key}: {value}",
                        ha="center",
                        va="top",
                        fontsize=9,
                        transform=ax.transAxes,
                    )
                    y_pos -= line_spacing * 0.8

    # Footer
    ax.text(
        0.5,
        0.05,
        "Generated with ops_model embedding analysis pipeline",
        ha="center",
        va="bottom",
        fontsize=8,
        style="italic",
        transform=ax.transAxes,
    )

    pdf.savefig(fig, bbox_inches="tight", dpi=300)
    plt.close(fig)


def add_section_header(pdf: PdfPages, section_title: str, description: str = ""):
    """
    Add a section header page to the report.

    Args:
        pdf: PdfPages object
        section_title: Title of the section
        description: Optional description text
    """
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")

    # Section title
    ax.text(
        0.5,
        0.5,
        section_title,
        ha="center",
        va="center",
        fontsize=20,
        fontweight="bold",
        transform=ax.transAxes,
    )

    # Description
    if description:
        ax.text(
            0.5,
            0.42,
            description,
            ha="center",
            va="center",
            fontsize=12,
            transform=ax.transAxes,
            wrap=True,
        )

    pdf.savefig(fig, bbox_inches="tight", dpi=300)
    plt.close(fig)


def add_plot_page(pdf: PdfPages, plot_path: Path, title: str, caption: str = ""):
    """
    Add a plot to the report with title and optional caption.

    Args:
        pdf: PdfPages object
        plot_path: Path to plot image file
        title: Plot title
        caption: Optional caption/description
    """
    # Silently skip if file doesn't exist
    if not plot_path.exists():
        return

    try:
        # Create a new figure
        fig, ax = plt.subplots(figsize=(11, 8.5))
        fig.suptitle(title, fontsize=16, fontweight="bold", y=0.98)

        # Load and display the image
        if plot_path.suffix.lower() == ".pdf":
            # For PDF files, we'll add a note that it's embedded
            ax.text(
                0.5,
                0.5,
                f"PDF plot embedded: {plot_path.name}",
                ha="center",
                va="center",
                fontsize=12,
            )
            ax.axis("off")
        else:
            # Load image using PIL
            img = Image.open(plot_path)
            ax.imshow(img)
            ax.axis("off")

        # Add caption if provided
        if caption:
            fig.text(
                0.5, 0.02, caption, ha="center", fontsize=10, style="italic", wrap=True
            )

        # Adjust layout to prevent title/caption cutoff
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])

        # Save to PDF
        pdf.savefig(fig, bbox_inches="tight", dpi=300)
        plt.close(fig)

    except Exception as e:
        # Silently skip on any error (corrupt image, unsupported format, etc.)
        print(f"Warning: Skipping plot {plot_path.name}: {e}")
        return


def add_metrics_table(pdf: PdfPages, df: pd.DataFrame, title: str):
    """
    Add a metrics table to the report.

    Args:
        pdf: PdfPages object
        df: DataFrame containing metrics
        title: Table title
    """
    if df is None or df.empty:
        return

    # Create figure (half-page size to match other plots)
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.axis("off")

    # Add title
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.95)

    # Format DataFrame for display
    # Round numeric columns to 4 decimal places
    df_display = df.copy()
    for col in df_display.columns:
        if df_display[col].dtype in ["float64", "float32"]:
            df_display[col] = df_display[col].round(4)

    # Create table
    table = ax.table(
        cellText=df_display.values,
        colLabels=df_display.columns,
        rowLabels=df_display.index,
        cellLoc="center",
        loc="center",
        colWidths=[0.3] * len(df_display.columns),
    )

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)  # Make rows taller

    # Style header row
    for i in range(len(df_display.columns)):
        cell = table[(0, i)]
        cell.set_facecolor("#4472C4")
        cell.set_text_props(weight="bold", color="white")

    # Style row labels
    for i in range(1, len(df_display) + 1):
        cell = table[(i, -1)]  # Row label column
        cell.set_facecolor("#D9E1F2")
        cell.set_text_props(weight="bold")

    # Alternate row colors
    for i in range(1, len(df_display) + 1):
        for j in range(len(df_display.columns)):
            cell = table[(i, j)]
            if i % 2 == 0:
                cell.set_facecolor("#F2F2F2")
            else:
                cell.set_facecolor("white")

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    pdf.savefig(fig, bbox_inches="tight", dpi=300)
    plt.close(fig)


def generate_umap_section(pdf: PdfPages, plots: Dict[str, Path]):
    """
    Generate UMAP visualization section.

    Includes only NTC (non-targeting control) UMAP plots:
    - Cell-level NTC
    - Guide-level NTC
    - Gene-level NTC
    """
    # Collect available NTC UMAP plots
    umap_plots = []

    # Define order for NTC plots
    ntc_plot_order = [
        ("umap_cell_ntc", "UMAP: Cell-level (NTC highlighted)"),
        ("umap_guide_ntc", "UMAP: Guide-level (NTC highlighted)"),
        ("umap_gene_ntc", "UMAP: Gene-level (NTC highlighted)"),
        ("umap_ntc", "UMAP: NTC Control"),  # Generic NTC plot
    ]

    # Collect plots in order if they exist
    for key, title in ntc_plot_order:
        if key in plots and plots[key].exists():
            umap_plots.append((title, plots[key]))

    # If no plots found, return early
    if not umap_plots:
        print("Warning: No NTC UMAP plots found to add to report")
        return

    # Add plots 2 per page
    for i in range(0, len(umap_plots), 2):
        fig, axes = plt.subplots(2, 1, figsize=(11, 8.5))

        for j in range(2):
            idx = i + j
            if idx < len(umap_plots):
                title, plot_path = umap_plots[idx]
                ax = axes[j] if len(umap_plots[i : i + 2]) == 2 else axes

                try:
                    img = Image.open(plot_path)
                    ax.imshow(img)
                    ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
                    ax.axis("off")
                except Exception as e:
                    print(f"Warning: Skipping plot {plot_path.name}: {e}")
                    ax.axis("off")
            else:
                axes[j].axis("off")

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight", dpi=300)
        plt.close(fig)


def generate_pca_section(
    pdf: PdfPages, plots: Dict[str, Path], metrics: Dict[str, pd.DataFrame]
):
    """
    Generate PCA analysis section.

    Includes:
    - Variance ratio plot
    - Summary statistics
    """
    # Add PCA variance ratio plot (half page to match UMAP sizing)
    if "pca_variance_ratio" in plots and plots["pca_variance_ratio"].exists():
        fig, ax = plt.subplots(1, 1, figsize=(11, 4))

        try:
            img = Image.open(plots["pca_variance_ratio"])
            ax.imshow(img)
            ax.set_title("PCA Variance Ratio", fontsize=14, fontweight="bold", pad=10)
            ax.axis("off")
        except Exception as e:
            print(f"Warning: Skipping PCA plot: {e}")
            ax.axis("off")

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight", dpi=300)
        plt.close(fig)


def generate_characterization_section(pdf: PdfPages, metrics: Dict[str, pd.DataFrame]):
    """
    Generate embedding characterization section.

    Includes:
    - Alignment and uniformity metrics
    - Similarity statistics (mean, std)
    - Class similarity analysis
    """
    pass


def generate_report(
    report_dir: Union[str, Path],
    output_filename: str = "embedding_report.pdf",
    title: Optional[str] = None,
):
    """
    Main function to generate the complete PDF report from a report directory.

    Args:
        report_dir: Path to report directory containing plots/, metrics/, and report_metadata.yml
        output_filename: Filename for the PDF report (saved in report_dir)
        title: Optional report title (defaults to using metadata)
    """
    report_dir = Path(report_dir)

    if not report_dir.exists():
        raise FileNotFoundError(f"Report directory not found: {report_dir}")

    # Load all data
    print(f"\nGenerating report from: {report_dir}")
    metadata = load_report_metadata(report_dir)
    plots = load_plots(report_dir)
    metrics = load_metrics(report_dir)

    # Determine title
    if title is None:
        if metadata and "feature_type" in metadata:
            feature_type = metadata["feature_type"].upper()
            title = f"{feature_type} Embedding Analysis Report"
        else:
            title = "Embedding Analysis Report"

    # Output path
    output_path = report_dir / output_filename

    print(f"\nGenerating PDF report: {output_path}")

    # Generate PDF report
    with PdfPages(output_path) as pdf:
        # Title page with metadata
        create_title_page(pdf, title, metadata)

        # Section 1: UMAP Visualizations
        if plots:
            add_section_header(
                pdf, "UMAP Visualizations", "2D projections of embedding space"
            )
            generate_umap_section(pdf, plots)

        # Section 2: Functional Cluster Analysis
        if "funk_clusters" in plots:
            add_section_header(
                pdf, "Functional Cluster Analysis", "Funk functional clusters on UMAP"
            )
            add_plot_page(
                pdf,
                plots["funk_clusters"],
                "Funk Functional Clusters",
                "28 functional clusters colored on UMAP space",
            )

        # Section 3: PCA Analysis
        if "pca_variance" in plots or any("pca" in k for k in metrics.keys()):
            add_section_header(
                pdf,
                "PCA Analysis",
                "Principal component analysis and variance explained",
            )
            generate_pca_section(pdf, plots, metrics)

        # Section 4: Embedding Quality Metrics
        if metrics:
            add_section_header(
                pdf,
                "Embedding Quality Metrics",
                "Alignment, uniformity, and similarity analysis",
            )
            generate_characterization_section(pdf, metrics)

            # Add all metrics tables
            for metric_name, df in metrics.items():
                add_metrics_table(pdf, df, metric_name.replace("_", " ").title())

    print(f"✓ Report generated successfully: {output_path}")
    return output_path


def main():
    """Main entry point for command-line usage."""
    args = parse_args()
    output_path = generate_report(
        report_dir=args.report_dir,
        output_filename=args.output_filename,
        title=args.title,
    )
    print(f"\n✓ PDF report available at: {output_path}")


if __name__ == "__main__":
    main()
