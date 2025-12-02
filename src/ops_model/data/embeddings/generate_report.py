"""
Generate PDF report for single-cell embedding verification.

Takes a directory of plots (PNG/PDF) and CSV files and compiles them
into a single PDF report with organized sections.
"""

import argparse
from pathlib import Path
from typing import Dict, Optional, Any
import pandas as pd
import yaml
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from PIL import Image


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
        description="Generate embedding verification report from plots and CSVs"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing plots and CSV files",
    )
    parser.add_argument(
        "--output_path", type=str, required=True, help="Output path for PDF report"
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Embedding Verification Report",
        help="Title for the report",
    )
    return parser.parse_args()


def load_plots(input_dir: Path) -> Dict[str, Path]:
    """
    Load all plot files from input directory.

    Expected plot types:
    - umap_all_cells.png: UMAP of all cells with NTC labels
    - umap_guide_averaged.png: UMAP of guide-averaged embeddings
    - umap_protein_complex.png: UMAP colored by protein complex
    - pca_variance_ratio.png: PCA variance explained plot

    Returns:
        Dictionary mapping plot types to file paths
    """
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    plots = {}

    # Supported image formats
    image_extensions = [".png", ".pdf", ".jpg", ".jpeg"]

    # Find all image files in directory
    all_plots = []
    for ext in image_extensions:
        all_plots.extend(input_dir.glob(f"*{ext}"))

    # Match files to expected plot types
    for plot_file in all_plots:
        filename_lower = plot_file.stem.lower()
        matched = False

        # Try to match to known plot types
        for plot_key, patterns in PLOT_PATTERNS.items():
            if any(pattern in filename_lower for pattern in patterns):
                plots[plot_key] = plot_file
                matched = True
                break

        # If not matched, add as "other" with filename as key
        if not matched:
            other_key = f"other_{plot_file.stem}"
            plots[other_key] = plot_file

    # Log what was found
    print(f"Found {len(plots)} plot file(s):")
    for key, path in plots.items():
        print(f"  - {key}: {path.name}")

    return plots


def load_metrics(input_dir: Path) -> Dict[str, Any]:
    """
    Load metric files (YAML or CSV) from input directory.

    Expected metrics files:
    - metrics.yaml: YAML file with embedding metrics
    - alignment_uniformity.csv: Alignment and uniformity scores
    - similarity_stats.csv: Mean & std similarity of embeddings

    Returns:
        Dictionary mapping metric types to DataFrames or dicts
    """
    if not input_dir.exists():
        print(f"Warning: Input directory not found: {input_dir}")
        return {}

    metrics = {}

    # Load YAML metrics files
    yaml_files = list(input_dir.glob("*.yaml")) + list(input_dir.glob("*.yml"))
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
    csv_files = list(input_dir.glob("*.csv"))
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            metrics[csv_file.stem] = df
            print(f"  Loaded CSV metrics: {csv_file.name}")
        except Exception as e:
            print(f"  Warning: Could not load {csv_file.name}: {e}")

    print(f"Found {len(metrics)} metric file(s)")
    return metrics


def create_title_page(pdf: PdfPages, title: str, metadata: Optional[Dict] = None):
    """
    Create title page with report metadata.

    Args:
        pdf: PdfPages object
        title: Report title
        metadata: Optional metadata dict (date, embedding dim, num cells, etc.)
    """
    pass


def add_section_header(pdf: PdfPages, section_title: str, description: str = ""):
    """
    Add a section header page to the report.

    Args:
        pdf: PdfPages object
        section_title: Title of the section
        description: Optional description text
    """
    pass


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
        pdf.savefig(fig, bbox_inches="tight")
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
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def generate_umap_section(pdf: PdfPages, plots: Dict[str, Path]):
    """
    Generate UMAP visualization section.

    Includes:
    - All cells with NTC labels
    - Guide-averaged embeddings
    - Protein complex colored
    """
    # Collect available UMAP plots
    umap_plots = []

    if "umap_cells" in plots and plots["umap_cells"].exists():
        umap_plots.append(("UMAP: All Cells", plots["umap_cells"]))

    if "umap_guide_bulked" in plots and plots["umap_guide_bulked"].exists():
        umap_plots.append(("UMAP: Guide-Averaged", plots["umap_guide_bulked"]))

    if "umap_protein_complex" in plots and plots["umap_protein_complex"].exists():
        umap_plots.append(("UMAP: Protein Complex", plots["umap_protein_complex"]))

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
        pdf.savefig(fig, bbox_inches="tight")
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
        pdf.savefig(fig, bbox_inches="tight")
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


def generate_report(input_dir: str, output_path: str, title: str):
    """
    Main function to generate the complete PDF report.

    Args:
        input_dir: Directory containing plots and CSVs
        output_path: Output path for PDF report
        title: Report title
    """
    input_path = Path(input_dir)

    # Load all data
    plots = load_plots(input_path)
    metrics = load_metrics(input_path)

    # Generate PDF report
    with PdfPages(output_path) as pdf:
        # Title page
        create_title_page(pdf, title)

        # Section 1: UMAP Visualizations
        add_section_header(
            pdf, "UMAP Visualizations", "2D projections of embedding space"
        )
        generate_umap_section(pdf, plots)

        # Section 2: PCA Analysis
        add_section_header(
            pdf, "PCA Analysis", "Principal component analysis and variance explained"
        )
        generate_pca_section(pdf, plots, metrics)

        # Section 3: Embedding Characterization
        add_section_header(
            pdf,
            "Embedding Characterization",
            "Quality metrics: alignment, uniformity, and similarity",
        )
        generate_characterization_section(pdf, metrics)

        add_section_header(pdf, "Metrics")
        # Add all metrics tables
        for metric_name, df in metrics.items():
            add_metrics_table(pdf, df, metric_name.replace("_", " ").title())


def main():
    """Main entry point."""
    args = parse_args()
    generate_report(
        input_dir=args.input_dir, output_path=args.output_path, title=args.title
    )
    print(f"Report generated: {args.output_path}")


if __name__ == "__main__":
    main()
