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

## Architecture: Registry-Based Plot System

This module uses a registry pattern to make it easy to add new plot types:

1. **Plot Generator Functions**: Standardized signature for all plot generators:
   ```python
   def generate_<plot_type>(
       adata: ad.AnnData,
       config: Dict[str, Any],
       data_source: Optional[str] = None
   ) -> plt.Figure:
       ...
       return fig
   ```

2. **Plot Registry (PLOT_GENERATORS)**: Dictionary mapping plot types to metadata:
   - function: The generator function
   - description: Human-readable description
   - required_data: Which data levels it needs (cell/guide/gene)
   - section_title: Title for PDF section
   - section_description: Description text
   - supports_levels: Whether it supports multiple levels

3. **Adding New Plot Types**: Simply:
   a) Write a generator function following the standard signature
   b) Add one entry to PLOT_GENERATORS dictionary
   c) Done! Config validation and PDF generation handle it automatically

4. **Helper Functions**: Utilities for accessing comprehensive_metadata from
   AnnData objects (has_comprehensive_metadata, get_biological_groups, etc.)

Example: Adding a comprehensive_metadata plot:
    ```python
    def generate_biological_groups_summary(adata, config, data_source=None):
        groups = get_biological_groups(adata)
        # ... create plot using comprehensive_metadata ...
        return fig

    PLOT_GENERATORS['biological_groups_summary'] = {
        'function': generate_biological_groups_summary,
        'description': 'Cell count summary by biological group',
        'required_data': ['gene', 'guide'],
        'section_title': 'Biological Groups Summary',
        'section_description': 'Overview of biological signals',
        'supports_levels': False,
    }
    ```
"""

import argparse
from pathlib import Path
from typing import Dict, Optional, Any, Union, List
import pandas as pd
import yaml
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from PIL import Image
from datetime import datetime
import anndata as ad
import numpy as np
import scanpy as sc
import tempfile
import shutil
from ops_model.data.embeddings.funk_clusters import funk_clusters, plot_funk_clusters


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


# ============================================================================
# Helper Functions for Comprehensive Metadata
# ============================================================================


def has_comprehensive_metadata(adata: ad.AnnData) -> bool:
    """
    Check if AnnData has comprehensive_metadata.

    Args:
        adata: AnnData object to check

    Returns:
        True if comprehensive_metadata exists, False otherwise
    """
    return "comprehensive_metadata" in adata.uns


def get_comprehensive_metadata(adata: ad.AnnData) -> Dict[str, Any]:
    """
    Extract comprehensive_metadata from AnnData.

    Args:
        adata: AnnData object

    Returns:
        Comprehensive metadata dictionary

    Raises:
        ValueError: If comprehensive_metadata not found
    """
    if not has_comprehensive_metadata(adata):
        raise ValueError("AnnData missing comprehensive_metadata in .uns")
    return adata.uns["comprehensive_metadata"]


def get_biological_groups(adata: ad.AnnData) -> Dict[str, Any]:
    """
    Extract biological groups from comprehensive_metadata.

    Args:
        adata: AnnData object with comprehensive_metadata

    Returns:
        Dictionary mapping biological signal to group info

    Raises:
        ValueError: If comprehensive_metadata not found
    """
    meta = get_comprehensive_metadata(adata)
    return meta.get("biological_groups", {})


def get_aggregation_types(adata: ad.AnnData) -> Dict[str, str]:
    """
    Get aggregation type (vertical/horizontal) for each biological signal.

    Args:
        adata: AnnData object with comprehensive_metadata

    Returns:
        Dictionary mapping biological signal to aggregation type
    """
    groups = get_biological_groups(adata)
    return {signal: info["aggregation_type"] for signal, info in groups.items()}


def get_feature_slices(adata: ad.AnnData) -> Dict[str, Dict[str, Any]]:
    """
    Get feature slice information from comprehensive_metadata.

    Args:
        adata: AnnData object with comprehensive_metadata

    Returns:
        Dictionary with feature slice information per biological signal
    """
    meta = get_comprehensive_metadata(adata)
    return meta.get("feature_slices", {})


# ============================================================================
# Helper Functions for PNG-based PDF Generation
# ============================================================================


def save_fig_as_png_and_add_to_pdf(
    fig: plt.Figure, pdf: PdfPages, temp_dir: Path, plot_name: str, dpi: int = 300
) -> None:
    """
    Save matplotlib figure as PNG to temp directory, then add to PDF.

    This improves rendering performance by using raster images instead of
    vector graphics directly in the PDF.

    Args:
        fig: Matplotlib figure to save
        pdf: PdfPages object to add the plot to
        temp_dir: Temporary directory for PNG storage
        plot_name: Name for the PNG file (without extension)
        dpi: Resolution for PNG (default 300)
    """
    # Save figure as PNG to temp directory with high DPI
    png_path = temp_dir / f"{plot_name}.png"
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", format="png", facecolor="white")
    plt.close(fig)

    # Create a new figure with exact page size to hold the PNG
    fig_container = plt.figure(figsize=(8.5, 11))
    ax = fig_container.add_axes([0, 0, 1, 1])  # Fill entire figure
    ax.axis("off")

    # Load and display the PNG
    img = Image.open(png_path)
    ax.imshow(img, aspect="auto", interpolation="lanczos")

    # Save container figure to PDF without bbox_inches='tight' to maintain page size
    pdf.savefig(fig_container, dpi=100, facecolor="white")
    plt.close(fig_container)


# ============================================================================
# Plot Generator Functions (Standardized Signatures)
# ============================================================================


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate embedding analysis report from config file"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file specifying datasets and plots",
    )
    return parser.parse_args()


def load_config(config_path: Union[str, Path]) -> Dict:
    """
    Load and parse YAML config file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Dictionary with config contents
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        print(f"Loaded config from: {config_path}")
        return config
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML syntax in config file: {e}")


def validate_config(config: Dict) -> None:
    """
    Validate config structure and required fields.

    Args:
        config: Config dictionary

    Raises:
        ValueError: If config is invalid
    """
    # Check output section
    if "output" not in config:
        raise ValueError("Config must have 'output' section")
    if "filename" not in config["output"]:
        raise ValueError("Config output section must have 'filename' field")

    # Check datasets section
    if "datasets" not in config:
        raise ValueError("Config must have 'datasets' section")
    if not isinstance(config["datasets"], list):
        raise ValueError("Config 'datasets' must be a list")
    if len(config["datasets"]) == 0:
        raise ValueError("Config must have at least one dataset")

    # Validate each dataset
    # Get valid plot types from registry
    valid_plot_types = get_available_plot_types()
    valid_levels = ["cell_level", "guide_level", "gene_level"]

    for i, dataset in enumerate(config["datasets"]):
        ds_name = dataset.get("name", f"dataset_{i}")

        # Check data section
        if "data" not in dataset:
            raise ValueError(f"Dataset '{ds_name}' must have 'data' section")

        # Check at least one AnnData file specified
        data_section = dataset["data"]
        adata_keys = ["adata_cell", "adata_guide", "adata_gene"]
        has_adata = any(key in data_section for key in adata_keys)
        if not has_adata:
            raise ValueError(
                f"Dataset '{ds_name}' must specify at least one AnnData file "
                f"(adata_cell, adata_guide, or adata_gene)"
            )

        # Validate file paths exist
        for key in adata_keys:
            if key in data_section:
                adata_path = Path(data_section[key])
                if not adata_path.exists():
                    raise FileNotFoundError(
                        f"Dataset '{ds_name}': AnnData file not found: {adata_path}"
                    )

        # Check plots section
        if "plots" not in dataset:
            raise ValueError(f"Dataset '{ds_name}' must have 'plots' section")

        plots = dataset["plots"]
        if not any(plots.values() if isinstance(plots, dict) else plots):
            raise ValueError(f"Dataset '{ds_name}' must specify at least one plot type")

        # Validate plot types using registry
        for plot_type, plot_value in plots.items():
            if plot_type not in valid_plot_types:
                raise ValueError(
                    f"Dataset '{ds_name}': Unknown plot type '{plot_type}'. "
                    f"Valid types: {', '.join(valid_plot_types)}"
                )

            # Validate ntc_controls levels
            if plot_type == "ntc_controls" and isinstance(plot_value, list):
                for level in plot_value:
                    if level not in valid_levels:
                        raise ValueError(
                            f"Dataset '{ds_name}': Unknown level '{level}' for ntc_controls. "
                            f"Valid levels: {', '.join(valid_levels)}"
                        )

    print("Config validation passed")


def generate_ntc_umap(
    adata: ad.AnnData, config: Dict[str, Any], data_source: Optional[str] = None
) -> plt.Figure:
    """
    Generate UMAP plot with reference label (e.g., NTC) highlighted.

    Args:
        adata: AnnData object with UMAP coordinates in obsm['X_umap']
        config: Configuration dict with:
            - reference_label: Label to highlight (e.g., "NTC")
            - level_name: Optional level name for title (e.g., "cell", "guide", "gene")
        data_source: Optional path to data source file for subtitle

    Returns:
        matplotlib Figure object
    """
    # Extract config parameters
    reference_label = config.get("reference_label", "NTC")
    level_name = config.get("level_name", "")
    # Use standard letter size
    fig, ax = plt.subplots(figsize=(8.5, 11))

    # Check if UMAP exists
    if "X_umap" not in adata.obsm:
        # Compute UMAP if not present
        print(f"Computing UMAP for {level_name} level...")
        sc.pp.neighbors(adata, n_neighbors=15, use_rep="X")
        sc.tl.umap(adata)

    umap_coords = adata.obsm["X_umap"]
    labels = adata.obs["label_str"].values

    # Plot all cells in gray
    ax.scatter(
        umap_coords[:, 0],
        umap_coords[:, 1],
        c="lightgray",
        s=10,
        alpha=0.5,
        label="Other",
    )

    # Highlight reference label
    ref_mask = labels == reference_label
    if ref_mask.sum() > 0:
        ax.scatter(
            umap_coords[ref_mask, 0],
            umap_coords[ref_mask, 1],
            c="red",
            s=30,
            alpha=0.8,
            label=reference_label,
        )

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    title = f"UMAP: {level_name} level" if level_name else "UMAP"
    ax.set_title(
        f"{title} ({reference_label} highlighted)", fontsize=14, fontweight="bold"
    )
    ax.legend()
    ax.set_aspect("equal")

    # Add data source subtitle
    if data_source:
        fig.text(
            0.5,
            0.02,
            f"Data source: {data_source}",
            ha="center",
            fontsize=8,
            style="italic",
            color="gray",
        )

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    return fig


def generate_protein_complex_umap(
    adata: ad.AnnData, config: Dict[str, Any], data_source: Optional[str] = None
) -> plt.Figure:
    """
    Generate UMAP plot highlighting a specific protein complex.

    Args:
        adata: AnnData object (gene-level)
        config: Configuration dict with:
            - complex_name: Name of protein complex (e.g., "RPL", "NUP")
        data_source: Optional path to data source file for subtitle

    Returns:
        matplotlib Figure object
    """
    # Extract config parameters
    complex_name = config.get("complex_name", "RPL")
    # Define protein complex gene lists
    protein_complexes = {
        "RPL": [
            "RPL3",
            "RPL4",
            "RPL5",
            "RPL6",
            "RPL7",
            "RPL7A",
            "RPL8",
            "RPL9",
            "RPL10",
            "RPL10A",
            "RPL11",
            "RPL12",
            "RPL13",
            "RPL13A",
            "RPL14",
            "RPL15",
            "RPL17",
            "RPL18",
            "RPL18A",
            "RPL19",
            "RPL21",
            "RPL22",
            "RPL23",
            "RPL23A",
            "RPL24",
            "RPL26",
            "RPL27",
            "RPL27A",
            "RPL28",
            "RPL29",
            "RPL30",
            "RPL31",
            "RPL32",
            "RPL34",
            "RPL35",
            "RPL35A",
            "RPL36",
            "RPL36A",
            "RPL37",
            "RPL37A",
            "RPL38",
            "RPL39",
            "RPL40",
            "RPL41",
        ],
        "RPS": [
            "RPS2",
            "RPS3",
            "RPS3A",
            "RPS4X",
            "RPS4Y1",
            "RPS5",
            "RPS6",
            "RPS7",
            "RPS8",
            "RPS9",
            "RPS10",
            "RPS11",
            "RPS12",
            "RPS13",
            "RPS14",
            "RPS15",
            "RPS15A",
            "RPS16",
            "RPS17",
            "RPS18",
            "RPS19",
            "RPS20",
            "RPS21",
            "RPS23",
            "RPS24",
            "RPS25",
            "RPS26",
            "RPS27",
            "RPS27A",
            "RPS28",
            "RPS29",
        ],
        "NUP": [
            "NUP35",
            "NUP37",
            "NUP43",
            "NUP50",
            "NUP54",
            "NUP58",
            "NUP62",
            "NUP85",
            "NUP88",
            "NUP93",
            "NUP98",
            "NUP107",
            "NUP133",
            "NUP153",
            "NUP155",
            "NUP160",
            "NUP188",
            "NUP205",
            "NUP210",
            "NUP214",
            "NUP358",
        ],
        "TRAPPC": [
            "TRAPPC1",
            "TRAPPC2",
            "TRAPPC2L",
            "TRAPPC3",
            "TRAPPC4",
            "TRAPPC5",
            "TRAPPC6A",
            "TRAPPC6B",
            "TRAPPC8",
            "TRAPPC9",
            "TRAPPC10",
            "TRAPPC11",
            "TRAPPC12",
            "TRAPPC13",
        ],
        "KRT": [
            "KRT1",
            "KRT2",
            "KRT3",
            "KRT4",
            "KRT5",
            "KRT6A",
            "KRT6B",
            "KRT6C",
            "KRT7",
            "KRT8",
            "KRT9",
            "KRT10",
            "KRT12",
            "KRT13",
            "KRT14",
            "KRT15",
            "KRT16",
            "KRT17",
            "KRT18",
            "KRT19",
            "KRT20",
        ],
        "COPI": [
            "ARCN1",
            "COPA",
            "COPB1",
            "COPB2",
            "COPG1",
            "COPG2",
            "COPE",
            "COPZ1",
            "COPZ2",
        ],
        "COPII": [
            "SEC23A",
            "SEC23B",
            "SEC24A",
            "SEC24B",
            "SEC24C",
            "SEC24D",
            "SEC31A",
            "SEC31B",
            "SAR1A",
            "SAR1B",
        ],
    }

    if complex_name not in protein_complexes:
        raise ValueError(
            f"Unknown protein complex: {complex_name}. "
            f"Valid: {', '.join(protein_complexes.keys())}"
        )

    # Use standard letter size
    fig, ax = plt.subplots(figsize=(8.5, 11))

    # Check if UMAP exists
    if "X_umap" not in adata.obsm:
        print(f"Computing UMAP for protein complex analysis...")
        sc.pp.neighbors(adata, n_neighbors=15, use_rep="X")
        sc.tl.umap(adata)

    umap_coords = adata.obsm["X_umap"]
    labels = adata.obs["label_str"].values

    # Get complex genes present in data
    complex_genes = protein_complexes[complex_name]
    complex_mask = np.isin(labels, complex_genes)

    # Plot all cells in gray
    ax.scatter(
        umap_coords[:, 0],
        umap_coords[:, 1],
        c="lightgray",
        s=10,
        alpha=0.5,
        label="Other genes",
    )

    # Highlight complex members
    if complex_mask.sum() > 0:
        ax.scatter(
            umap_coords[complex_mask, 0],
            umap_coords[complex_mask, 1],
            c="blue",
            s=50,
            alpha=0.8,
            label=f"{complex_name} complex",
        )

        # Annotate gene names for complex members
        for i, (x, y, label) in enumerate(
            zip(
                umap_coords[complex_mask, 0],
                umap_coords[complex_mask, 1],
                labels[complex_mask],
            )
        ):
            ax.annotate(label, (x, y), fontsize=6, alpha=0.7)

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(f"{complex_name} Protein Complex", fontsize=14, fontweight="bold")
    ax.legend()
    ax.set_aspect("equal")

    # Add data source subtitle
    if data_source:
        fig.text(
            0.5,
            0.02,
            f"Data source: {data_source}",
            ha="center",
            fontsize=8,
            style="italic",
            color="gray",
        )

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    return fig


def generate_pca_variance(
    adata: ad.AnnData, config: Dict[str, Any], data_source: Optional[str] = None
) -> plt.Figure:
    """
    Generate PCA variance explained plot.

    Args:
        adata: AnnData object
        config: Configuration dict (currently unused, for consistency)
        data_source: Optional path to data source file for subtitle

    Returns:
        matplotlib Figure object
    """
    # Use standard letter size width, moderate height
    fig, ax = plt.subplots(figsize=(8.5, 6.5))

    # Compute PCA if not present
    if "X_pca" not in adata.obsm or "pca" not in adata.uns:
        print("Computing PCA...")
        sc.tl.pca(adata, n_comps=min(50, adata.shape[0] - 1, adata.shape[1]))

    variance_ratio = adata.uns["pca"]["variance_ratio"]
    n_components = len(variance_ratio)

    # Plot variance ratio
    ax.plot(range(1, n_components + 1), variance_ratio, "bo-", markersize=4)
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Variance Explained Ratio")
    ax.set_title("PCA Variance Explained", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)

    # Add cumulative variance
    cumsum = np.cumsum(variance_ratio)
    ax2 = ax.twinx()
    ax2.plot(range(1, n_components + 1), cumsum, "r--", alpha=0.6, label="Cumulative")
    ax2.set_ylabel("Cumulative Variance Explained", color="r")
    ax2.tick_params(axis="y", labelcolor="r")

    # Add data source subtitle
    if data_source:
        fig.text(
            0.5,
            0.02,
            f"Data source: {data_source}",
            ha="center",
            fontsize=8,
            style="italic",
            color="gray",
        )

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    return fig


def generate_funk_clusters(
    adata: ad.AnnData, config: Dict[str, Any], data_source: Optional[str] = None
) -> plt.Figure:
    """
    Generate UMAP colored by Funk functional clusters.

    This function wraps the actual plot_funk_clusters() function from funk_clusters.py
    which creates a 7x4 grid showing all 27 functional clusters.

    Args:
        adata: AnnData object (gene-level) with UMAP coordinates and gene labels
        config: Configuration dict (currently unused, for consistency)
        data_source: Optional path to data source file for subtitle

    Returns:
        matplotlib Figure object with 7x4 subplot grid
    """
    # Check if UMAP exists
    if "X_umap" not in adata.obsm:
        print("Computing UMAP for Funk cluster analysis...")
        sc.pp.neighbors(adata, n_neighbors=15, use_rep="X")
        sc.tl.umap(adata)

    # Call the real plot_funk_clusters function without saving
    # It creates the figure internally but doesn't return it
    plot_funk_clusters(adata, funk_clusters, save_path=None, report_dir=None)

    # Capture the figure that was just created
    fig = plt.gcf()

    # Add data source subtitle
    if data_source:
        fig.text(
            0.5,
            0.01,
            f"Data source: {data_source}",
            ha="center",
            fontsize=8,
            style="italic",
            color="gray",
        )

    return fig


def generate_biological_groups_summary(
    adata: ad.AnnData, config: Dict[str, Any], data_source: Optional[str] = None
) -> plt.Figure:
    """
    Generate comprehensive summary of biological groups from comprehensive_metadata.

    Creates a combined figure with:
    1. Summary box listing biological signals and experiments
    2. Bar chart showing cell counts per biological group
    3. Table showing biological group, contributing experiments, and total cells

    Args:
        adata: AnnData object with comprehensive_metadata
        config: Configuration dict (unused, for consistency)
        data_source: Optional path to data source file for subtitle

    Returns:
        matplotlib Figure object with 3-part visualization
    """
    # Extract comprehensive metadata
    if not has_comprehensive_metadata(adata):
        raise ValueError(
            "AnnData missing comprehensive_metadata. This plot requires data combined with concatenate_experiments_comprehensive()."
        )

    meta = get_comprehensive_metadata(adata)
    groups = get_biological_groups(adata)

    # Create figure with 3 subplots (summary, bar chart, table)
    # Use standard letter size (8.5 x 11 inches)
    fig = plt.figure(figsize=(8.5, 11))
    gs = fig.add_gridspec(3, 1, height_ratios=[0.8, 2.2, 4], hspace=0.35)

    # ========================================================================
    # Section 1: Summary Box (top)
    # ========================================================================
    ax_summary = fig.add_subplot(gs[0])
    ax_summary.axis("off")

    # Collect unique biological signals and experiments
    biological_signals = sorted(groups.keys())
    all_experiments = set()
    for info in groups.values():
        all_experiments.update(info["experiments"])
    experiments = sorted(all_experiments)

    # Format summary text
    summary_text = "Biological Signals:\n"
    summary_text += ", ".join(biological_signals)
    summary_text += "\n\nExperiments:\n"
    summary_text += ", ".join(experiments)

    # Add summary box
    ax_summary.text(
        0.5,
        0.5,
        summary_text,
        ha="center",
        va="center",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.3),
        transform=ax_summary.transAxes,
    )

    # ========================================================================
    # Section 2: Bar Chart (middle)
    # ========================================================================
    ax_bar = fig.add_subplot(gs[1])

    # Extract data for bar chart
    labels = []
    cell_counts = []

    # Sort by cell count (descending) for better visualization
    sorted_groups = sorted(
        groups.items(), key=lambda x: x[1]["n_cells_total"], reverse=True
    )

    for signal, info in sorted_groups:
        labels.append(info["short_label"])
        cell_counts.append(info["n_cells_total"])

    # Create horizontal bar chart with single color
    y_pos = np.arange(len(labels))
    bars = ax_bar.barh(y_pos, cell_counts, color="#4472C4")

    # Add value labels on bars
    for i, (bar, count) in enumerate(zip(bars, cell_counts)):
        width = bar.get_width()
        ax_bar.text(
            width,
            bar.get_y() + bar.get_height() / 2,
            f" {count:,}",
            ha="left",
            va="center",
            fontsize=9,
        )

    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(labels)
    ax_bar.set_xlabel("Total Cells", fontsize=11, fontweight="bold")
    ax_bar.set_title(
        "Cell Counts by Biological Group", fontsize=12, fontweight="bold", pad=15
    )
    ax_bar.grid(axis="x", alpha=0.3)

    # ========================================================================
    # Section 3: Detailed Table (bottom)
    # ========================================================================
    ax_table = fig.add_subplot(gs[2])
    ax_table.axis("off")

    # Build table data - one row per biological group
    table_data = []
    row_heights = []  # Track number of lines per row for height calculation
    for signal, info in sorted_groups:
        bio_signal = info["short_label"]
        total_cells = info["n_cells_total"]

        # Get list of contributing experiments
        experiments = info["experiments"]

        # Format experiments with line wrapping (max ~3-4 experiments per line)
        # This prevents overflow into adjacent columns
        max_per_line = 3
        if len(experiments) <= max_per_line:
            experiments_str = ", ".join(experiments)
            n_lines = 1
        else:
            # Split into multiple lines
            lines = []
            for i in range(0, len(experiments), max_per_line):
                chunk = experiments[i : i + max_per_line]
                lines.append(", ".join(chunk))
            experiments_str = "\n".join(lines)
            n_lines = len(lines)

        # Add single row for this biological group
        table_data.append([bio_signal, experiments_str, f"{total_cells:,}"])
        row_heights.append(n_lines)

    # Create table first
    if table_data:
        table = ax_table.table(
            cellText=table_data,
            colLabels=["Biological Group", "Experiments", "Total Cells"],
            cellLoc="center",
            loc="upper center",
            colWidths=[0.3, 0.5, 0.2],
        )

        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(9)

        # Calculate max lines across all rows to determine overall scale
        max_lines = max(row_heights) if row_heights else 1
        # Use adaptive scaling: base scale + extra for multi-line rows
        scale_factor = 2.0 + (max_lines - 1) * 0.5
        table.scale(1, scale_factor)

        # Style header row
        for i in range(3):
            cell = table[(0, i)]
            cell.set_facecolor("#4472C4")
            cell.set_text_props(weight="bold", color="white")

        # Alternate row colors and set text alignment
        for i in range(1, len(table_data) + 1):
            for j in range(3):
                cell = table[(i, j)]

                # Set colors
                if i % 2 == 0:
                    cell.set_facecolor("#F2F2F2")
                else:
                    cell.set_facecolor("white")

                # Set text alignment
                cell.set_text_props(
                    horizontalalignment="center",
                    verticalalignment="center",
                    wrap=True,  # Enable wrapping for all cells
                )

    # Add title above table (after table is created and positioned)
    ax_table.text(
        0.5,
        1.0,
        "Biological Groups and Experiments",
        ha="center",
        va="bottom",
        fontsize=12,
        fontweight="bold",
        transform=ax_table.transAxes,
    )

    # Add data source subtitle
    if data_source:
        fig.text(
            0.5,
            0.02,
            f"Data source: {data_source}",
            ha="center",
            fontsize=8,
            style="italic",
            color="gray",
        )

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    return fig


# ============================================================================
# Plot Registry System
# ============================================================================

# Registry mapping plot type to generator function and metadata
PLOT_GENERATORS = {
    "ntc_controls": {
        "function": generate_ntc_umap,
        "description": "Non-targeting controls highlighted on UMAP",
        "required_data": ["cell", "guide", "gene"],
        "section_title": "NTC Control Analysis",
        "section_description": "Non-targeting controls should cluster together",
        "supports_levels": True,  # This plot type supports multiple levels
    },
    "protein_complexes": {
        "function": generate_protein_complex_umap,
        "description": "Known protein complexes highlighted on UMAP",
        "required_data": ["gene"],
        "section_title": "Protein Complex Validation",
        "section_description": "Known protein complexes should cluster together",
        "supports_levels": False,  # Only gene-level
    },
    "pca_variance": {
        "function": generate_pca_variance,
        "description": "PCA variance explained ratio plot",
        "required_data": ["cell", "guide", "gene"],
        "section_title": "PCA Analysis",
        "section_description": "Principal component analysis and variance explained",
        "supports_levels": False,  # Uses first available
    },
    "funk_clusters": {
        "function": generate_funk_clusters,
        "description": "Funk functional clusters colored on UMAP",
        "required_data": ["gene"],
        "section_title": "Functional Cluster Analysis",
        "section_description": "28 functional gene sets colored on UMAP",
        "supports_levels": False,  # Only gene-level
    },
    "biological_groups_summary": {
        "function": generate_biological_groups_summary,
        "description": "Summary of biological groups with cell counts and experimental contributions",
        "required_data": ["gene", "guide"],
        "section_title": "Biological Groups Summary",
        "section_description": "Overview of biological signals, cell counts, and experimental contributions",
        "supports_levels": False,  # Uses comprehensive_metadata
    },
}


def get_available_plot_types() -> List[str]:
    """
    Get list of available plot types from registry.

    Returns:
        List of plot type names
    """
    return list(PLOT_GENERATORS.keys())


def get_plot_info(plot_type: str) -> Optional[Dict[str, Any]]:
    """
    Get information about a specific plot type.

    Args:
        plot_type: Name of plot type

    Returns:
        Plot info dictionary or None if not found
    """
    return PLOT_GENERATORS.get(plot_type)


def compute_alignment_uniformity(adata: ad.AnnData) -> pd.DataFrame:
    """
    Compute alignment and uniformity metrics from embeddings.

    Alignment measures how well samples from the same class cluster together.
    Uniformity measures how evenly samples are distributed in embedding space.

    Args:
        adata: AnnData object with embeddings in X or obsm

    Returns:
        DataFrame with metrics
    """
    # Use X_umap if available, otherwise use X
    if "X_umap" in adata.obsm:
        embeddings = adata.obsm["X_umap"]
    elif "X_pca" in adata.obsm:
        embeddings = adata.obsm["X_pca"]
    else:
        embeddings = adata.X

    if isinstance(embeddings, np.ndarray):
        pass
    else:
        embeddings = (
            embeddings.toarray()
            if hasattr(embeddings, "toarray")
            else np.array(embeddings)
        )

    # Simple alignment metric: within-class vs between-class distance ratio
    labels = adata.obs["label_str"].values
    unique_labels = np.unique(labels)

    # Compute pairwise distances
    from scipy.spatial.distance import pdist, squareform

    distances = squareform(pdist(embeddings))

    # Within-class distances
    within_class_dists = []
    for label in unique_labels:
        mask = labels == label
        if mask.sum() > 1:
            label_dists = distances[np.ix_(mask, mask)]
            within_class_dists.append(np.mean(label_dists[label_dists > 0]))

    # Between-class distances
    between_class_dists = []
    for i, label1 in enumerate(unique_labels):
        for label2 in unique_labels[i + 1 :]:
            mask1 = labels == label1
            mask2 = labels == label2
            between_dists = distances[np.ix_(mask1, mask2)]
            between_class_dists.append(np.mean(between_dists))

    alignment = np.mean(between_class_dists) / (np.mean(within_class_dists) + 1e-8)

    # Uniformity: measure how evenly distributed embeddings are
    # Higher uniformity = more evenly spread
    norms = np.linalg.norm(embeddings, axis=1)
    uniformity = np.std(norms) / (np.mean(norms) + 1e-8)

    metrics_df = pd.DataFrame(
        {
            "metric": [
                "alignment",
                "uniformity",
                "within_class_dist",
                "between_class_dist",
            ],
            "value": [
                alignment,
                uniformity,
                np.mean(within_class_dists) if within_class_dists else 0,
                np.mean(between_class_dists) if between_class_dists else 0,
            ],
        }
    )

    return metrics_df


def compute_similarity_to_reference(
    adata: ad.AnnData, reference_label: str
) -> pd.DataFrame:
    """
    Compute similarity of each label to a reference label (e.g., NTC).

    Args:
        adata: AnnData object
        reference_label: Reference label to compute distances from

    Returns:
        DataFrame with per-label similarities
    """
    # Use X_umap if available, otherwise use X
    if "X_umap" in adata.obsm:
        embeddings = adata.obsm["X_umap"]
    elif "X_pca" in adata.obsm:
        embeddings = adata.obsm["X_pca"]
    else:
        embeddings = adata.X

    if not isinstance(embeddings, np.ndarray):
        embeddings = (
            embeddings.toarray()
            if hasattr(embeddings, "toarray")
            else np.array(embeddings)
        )

    labels = adata.obs["label_str"].values

    # Get reference embeddings
    ref_mask = labels == reference_label
    if ref_mask.sum() == 0:
        print(f"Warning: Reference label '{reference_label}' not found in data")
        return pd.DataFrame()

    ref_embeddings = embeddings[ref_mask]
    ref_centroid = np.mean(ref_embeddings, axis=0)

    # Compute distance from each label's centroid to reference
    unique_labels = np.unique(labels)
    similarities = []

    for label in unique_labels:
        if label == reference_label:
            continue

        label_mask = labels == label
        label_embeddings = embeddings[label_mask]
        label_centroid = np.mean(label_embeddings, axis=0)

        # Euclidean distance
        distance = np.linalg.norm(label_centroid - ref_centroid)

        similarities.append(
            {
                "label": label,
                "distance_to_reference": distance,
                "n_samples": label_mask.sum(),
            }
        )

    similarity_df = pd.DataFrame(similarities)
    if not similarity_df.empty:
        similarity_df = similarity_df.sort_values("distance_to_reference")

    return similarity_df


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


def create_title_page(
    pdf: PdfPages,
    title: str,
    metadata: Optional[Dict] = None,
    temp_dir: Optional[Path] = None,
):
    """
    Create title page with report metadata.

    Args:
        pdf: PdfPages object
        title: Report title
        metadata: Optional metadata dict from report_metadata.yml
        temp_dir: Optional temporary directory for PNG storage
    """
    # Use standard letter size portrait
    fig, ax = plt.subplots(figsize=(8.5, 11))
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

    # Save as PNG then add to PDF if temp_dir provided
    if temp_dir:
        save_fig_as_png_and_add_to_pdf(fig, pdf, temp_dir, "title_page")
    else:
        pdf.savefig(fig, bbox_inches="tight", dpi=300)
        plt.close(fig)


def add_section_header(
    pdf: PdfPages,
    section_title: str,
    description: str = "",
    temp_dir: Optional[Path] = None,
):
    """
    Add a section header page to the report.

    Args:
        pdf: PdfPages object
        section_title: Title of the section
        description: Optional description text
        temp_dir: Optional temporary directory for PNG storage
    """
    # Use standard letter size portrait
    fig, ax = plt.subplots(figsize=(8.5, 11))
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

    # Save as PNG then add to PDF if temp_dir provided
    if temp_dir:
        # Use section title as part of filename (sanitize it)
        safe_title = section_title.replace(" ", "_").replace("/", "_")[:50]
        save_fig_as_png_and_add_to_pdf(fig, pdf, temp_dir, f"section_{safe_title}")
    else:
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
        # Create a new figure using standard letter size portrait
        fig, ax = plt.subplots(figsize=(8.5, 11))
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

    # Create figure using standard letter size width
    fig, ax = plt.subplots(figsize=(8.5, 5))
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
        # Use standard letter size portrait
        fig, axes = plt.subplots(2, 1, figsize=(8.5, 11))

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
    # Add PCA variance ratio plot
    if "pca_variance_ratio" in plots and plots["pca_variance_ratio"].exists():
        # Use standard letter size width
        fig, ax = plt.subplots(1, 1, figsize=(8.5, 5))

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


def generate_report_from_config(config_path: Union[str, Path]) -> Path:
    """
    Generate PDF report from config file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Path to generated PDF
    """
    # Load and validate config
    config = load_config(config_path)
    validate_config(config)

    # Get output filename
    output_filename = config["output"]["filename"]
    output_path = Path(output_filename)

    print(f"\nGenerating PDF report: {output_path}")

    # Create temporary directory for PNG storage
    temp_dir = Path(tempfile.mkdtemp(prefix="report_plots_"))
    print(f"Using temporary directory: {temp_dir}")

    try:
        # Generate PDF report
        with PdfPages(output_path) as pdf:
            # Create title page
            title = "Embedding Analysis Report"
            metadata = {
                "created_date": datetime.now().isoformat(),
                "config_file": str(config_path),
                "n_datasets": len(config["datasets"]),
            }
            create_title_page(pdf, title, metadata, temp_dir)

        # Process each dataset
        for dataset_idx, dataset_config in enumerate(config["datasets"]):
            dataset_name = dataset_config.get("name", f"Dataset {dataset_idx + 1}")
            print(f"\nProcessing dataset: {dataset_name}")

            # Add dataset section header if multiple datasets
            if len(config["datasets"]) > 1:
                add_section_header(pdf, f"Dataset: {dataset_name}", "", temp_dir)

            # Load AnnData files
            data_section = dataset_config["data"]
            adata_dict = {}

            if "adata_cell" in data_section:
                print(f"  Loading cell-level data...")
                adata_dict["cell"] = ad.read_h5ad(data_section["adata_cell"])

            if "adata_guide" in data_section:
                print(f"  Loading guide-level data...")
                adata_dict["guide"] = ad.read_h5ad(data_section["adata_guide"])

            if "adata_gene" in data_section:
                print(f"  Loading gene-level data...")
                adata_dict["gene"] = ad.read_h5ad(data_section["adata_gene"])

            reference_label = data_section.get("reference_label", "NTC")

            # Generate plots based on config using registry
            plots_config = dataset_config["plots"]

            # Process each plot type using registry
            for plot_type, plot_config_value in plots_config.items():
                # Skip if plot is disabled
                if not plot_config_value:
                    continue

                # Get plot info from registry
                plot_info = get_plot_info(plot_type)
                if plot_info is None:
                    print(f"    Warning: Unknown plot type '{plot_type}', skipping")
                    continue

                # Add section header
                add_section_header(
                    pdf,
                    plot_info["section_title"],
                    plot_info["section_description"],
                    temp_dir,
                )

                # Handle special cases for different plot types
                if plot_type == "ntc_controls":
                    # NTC controls can be generated at multiple levels
                    levels_to_plot = plot_config_value
                    if not isinstance(levels_to_plot, list):
                        levels_to_plot = ["cell_level", "guide_level", "gene_level"]

                    for level in levels_to_plot:
                        level_key = level.replace("_level", "")
                        if level_key in adata_dict:
                            print(f"    Generating NTC UMAP for {level_key} level...")
                            # Get the data source path for this level
                            data_source_key = f"adata_{level_key}"
                            data_source = data_section.get(data_source_key, None)

                            # Create config dict for plot generator
                            plot_gen_config = {
                                "reference_label": reference_label,
                                "level_name": level_key,
                            }

                            # Generate plot
                            fig = plot_info["function"](
                                adata_dict[level_key],
                                plot_gen_config,
                                data_source=data_source,
                            )
                            # Save as PNG then add to PDF
                            plot_name = f"ntc_{level_key}_{dataset_idx}"
                            save_fig_as_png_and_add_to_pdf(
                                fig, pdf, temp_dir, plot_name
                            )

                elif plot_type == "protein_complexes":
                    # Protein complexes - iterate over requested complexes
                    complexes = plot_config_value
                    # Use gene-level data for protein complexes (default to first available)
                    complex_adata = (
                        adata_dict.get("gene")
                        or adata_dict.get("guide")
                        or adata_dict.get("cell")
                    )

                    # Get the corresponding data source path
                    if "gene" in adata_dict:
                        complex_data_source = data_section.get("adata_gene")
                    elif "guide" in adata_dict:
                        complex_data_source = data_section.get("adata_guide")
                    else:
                        complex_data_source = data_section.get("adata_cell")

                    for complex_name in complexes:
                        print(
                            f"    Generating protein complex plot for {complex_name}..."
                        )

                        # Create config dict for plot generator
                        plot_gen_config = {"complex_name": complex_name}

                        # Generate plot
                        fig = plot_info["function"](
                            complex_adata,
                            plot_gen_config,
                            data_source=complex_data_source,
                        )
                        # Save as PNG then add to PDF
                        plot_name = f"complex_{complex_name}_{dataset_idx}"
                        save_fig_as_png_and_add_to_pdf(fig, pdf, temp_dir, plot_name)

                else:
                    # Generic plot types (pca_variance, funk_clusters, etc.)
                    # Use first available adata
                    plot_adata = (
                        adata_dict.get("cell")
                        or adata_dict.get("guide")
                        or adata_dict.get("gene")
                    )

                    # Get the corresponding data source path
                    if "cell" in adata_dict:
                        plot_data_source = data_section.get("adata_cell")
                    elif "guide" in adata_dict:
                        plot_data_source = data_section.get("adata_guide")
                    else:
                        plot_data_source = data_section.get("adata_gene")

                    print(f"    Generating {plot_type} plot...")

                    # Create config dict for plot generator (empty for now)
                    plot_gen_config = {}

                    # Generate plot
                    fig = plot_info["function"](
                        plot_adata, plot_gen_config, data_source=plot_data_source
                    )
                    # Save as PNG then add to PDF
                    plot_name = f"{plot_type}_{dataset_idx}"
                    save_fig_as_png_and_add_to_pdf(fig, pdf, temp_dir, plot_name)

        print(f"\n✓ Report generated successfully: {output_path}")

    finally:
        # Clean up temporary directory
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            print(f"Cleaned up temporary directory: {temp_dir}")

    return output_path


def main():
    """Main entry point for command-line usage."""
    args = parse_args()
    output_path = generate_report_from_config(config_path=args.config)
    print(f"\n✓ PDF report available at: {output_path}")


if __name__ == "__main__":
    main()
