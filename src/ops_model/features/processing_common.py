"""Unified CSV → AnnData processing pipeline for CellProfiler and embedding features.

A single entry point, ``process_features_csv``, branches once on the config's
feature type:

- ``cellprofiler``: build the cell-level AnnData with ``create_adata_object`` and
  split it into per-reporter subsets (``split_adata_by_reporter``), saving one set
  of outputs per reporter.
- any embedding type (``dinov3``/``cell_dino``/``subcell``): build with
  ``create_adata_object_embedding`` (one channel per CSV) and save a single set of
  outputs, the reporter name coming from ``FeatureMetadata``.

Both paths share validation, guide/gene aggregation, and saving.
"""

import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ops_model.features.anndata_utils import create_aggregated_embeddings
from ops_model.features.evaluate_cp import (
    create_adata_object,
    split_adata_by_reporter,
)
from ops_model.features.evaluate_embeddings import create_adata_object_embedding
from ops_model.post_process.anndata_processing.anndata_validator import AnndataValidator


# ---------------------------------------------------------------------------
# Embedding-config helpers (shared by both feature types)
# ---------------------------------------------------------------------------


@dataclass
class EmbeddingConfig:
    """Resolved embedding settings for one aggregation level."""

    compute_embeddings: bool
    n_pca_components: int
    n_neighbors: int
    compute_pca: bool
    compute_umap: bool
    compute_phate: bool


def extract_embedding_config(level_cfg: dict) -> EmbeddingConfig:
    """Resolve a ``guide_level``/``gene_level`` config block into embedding settings.

    Precedence:
    - ``compute_embeddings`` (default True) is the master switch; when False, PCA,
      UMAP and PHATE are all forced off regardless of their individual flags.
    - Otherwise each of ``pca``/``umap``/``phate`` defaults to True.
    - UMAP requires PCA: if UMAP is on but PCA is off, PCA is enabled.
    """
    emb = level_cfg.get("embeddings", {})
    compute_embeddings = level_cfg.get("compute_embeddings", True)

    compute_pca = emb.get("pca", True) if compute_embeddings else False
    compute_umap = emb.get("umap", True) if compute_embeddings else False
    compute_phate = emb.get("phate", True) if compute_embeddings else False
    if compute_umap and not compute_pca:
        compute_pca = True

    return EmbeddingConfig(
        compute_embeddings=compute_embeddings,
        n_pca_components=emb.get("n_pca_components", 128),
        n_neighbors=emb.get("n_neighbors", 15),
        compute_pca=compute_pca,
        compute_umap=compute_umap,
        compute_phate=compute_phate,
    )


def aggregate_level(cell_adata, level: str, level_cfg: dict):
    """Aggregate a cell-level AnnData to ``level`` ("guide"/"gene") with embeddings.

    Returns the aggregated AnnData, or ``None`` if the level is disabled
    (``enabled: false``).
    """
    if not level_cfg.get("enabled", True):
        return None
    cfg = extract_embedding_config(level_cfg)
    return create_aggregated_embeddings(
        cell_adata,
        level=level,
        n_pca_components=cfg.n_pca_components,
        n_neighbors=cfg.n_neighbors,
        compute_pca=cfg.compute_pca,
        compute_umap=cfg.compute_umap,
        compute_phate=cfg.compute_phate,
    )


def format_embeddings_list(level_cfg: dict, sep: str = ", ") -> str:
    """Human-readable list of which embeddings will run for a level (e.g. "PCA, UMAP")."""
    cfg = extract_embedding_config(level_cfg)
    if not cfg.compute_embeddings:
        return "none"
    names = []
    if cfg.compute_pca:
        names.append("PCA")
    if cfg.compute_umap:
        names.append("UMAP")
    if cfg.compute_phate:
        names.append("PHATE")
    return sep.join(names) if names else "none"


# ---------------------------------------------------------------------------
# Validation + saving
# ---------------------------------------------------------------------------


def validate_and_save(adata, path: Path, level: str) -> None:
    """Validate an AnnData object against the schema and save it to ``path``.

    Validation is enforced with hard constraints — any errors raise.

    Args:
        adata: AnnData object to validate and save.
        path: Destination .h5ad path.
        level: Schema level ("cell", "guide", "gene").
    """
    print(f"\nValidating {level}-level AnnData before saving...")
    report = AnndataValidator().validate(adata, level=level)
    errors = report.errors
    warnings = report.warnings

    if report.is_valid:
        print(f"✓ Validation passed: {level}-level AnnData is compliant")
    else:
        print(f"Validation found {len(errors)} errors, {len(warnings)} warnings")
        if errors:
            print("\nERROR-level issues:")
            for issue in errors[:10]:
                print(f"  - {issue.field}: {issue.message}")
            if len(errors) > 10:
                print(f"  ... and {len(errors) - 10} more errors")
        if warnings:
            print("\nWARNING-level issues:")
            for issue in warnings[:5]:
                print(f"  - {issue.field}: {issue.message}")
            if len(warnings) > 5:
                print(f"  ... and {len(warnings) - 5} more warnings")

    if errors:
        error_summary = f"{len(errors)} validation error(s) found in {level}-level AnnData"
        print(f"\n✗ Validation FAILED: {error_summary}")
        print(f"Fix these issues before saving. First error: {errors[0].message}")
        raise ValueError(error_summary)

    print(f"Saving {level}-level AnnData to {path}")
    adata.write_h5ad(path)
    print(f"✓ Saved successfully: {path}")


def _save_level_outputs(cell_adata, suffix, save_dir: Path, guide_cfg: dict, gene_cfg: dict):
    """Save cell-level, then aggregate to guide/gene level and save each.

    ``suffix`` is the reporter name (appended as ``_{suffix}``); pass ``None`` for
    unsuffixed combined outputs. Guide/gene levels are skipped when disabled
    (``aggregate_level`` returns None) or when ``save_output`` is False.
    """
    tag = f"_{suffix}" if suffix else ""

    validate_and_save(cell_adata, save_dir / f"features_processed{tag}.h5ad", "cell")

    adata_guide = aggregate_level(cell_adata, "guide", guide_cfg)
    if adata_guide is not None and guide_cfg.get("save_output", True):
        validate_and_save(adata_guide, save_dir / f"guide_bulked{tag}.h5ad", "guide")
        print(f"  guide-bulked embeddings: {format_embeddings_list(guide_cfg)}")

    adata_gene = aggregate_level(cell_adata, "gene", gene_cfg)
    if adata_gene is not None and gene_cfg.get("save_output", True):
        validate_and_save(adata_gene, save_dir / f"gene_bulked{tag}.h5ad", "gene")
        print(f"  gene-bulked embeddings: {format_embeddings_list(gene_cfg)}")


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------


def _detect_channel_and_experiment(save_path: Path):
    """For embedding CSVs: channel from the filename ``{model}_features_{channel}.csv``
    and experiment from the CSV's first row."""
    stem = save_path.stem
    marker = "_features_"
    channel = stem[stem.index(marker) + len(marker) :] if marker in stem else "unknown"

    df_sample = pd.read_csv(save_path, nrows=1)
    experiment = (
        df_sample["experiment"].iloc[0] if "experiment" in df_sample.columns else None
    )
    return channel, experiment


def process_features_csv(save_path: str, config_path: str = None):
    """Process a feature/embedding CSV into cell/guide/gene AnnData objects.

    Branches on ``config['model_type']`` (falling back to ``embedding_type``, then
    ``cellprofiler``): the ``cellprofiler`` path splits by reporter; embedding paths
    save a single reporter-named set of outputs. Returns the cell-level AnnData.

    Args:
        save_path: Path to the features/embeddings CSV.
        config_path: Path to the YAML config (must set ``cell_type``).
    """
    config = {}
    if config_path is not None:
        import yaml

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        print(f"Loaded configuration from {config_path}")

    cell_type = config.get("cell_type")
    if not cell_type:
        raise ValueError(
            "cell_type must be specified in config "
            "(e.g. cell_type: 'A549') for validator compliance."
        )

    feature_type = (
        config.get("model_type") or config.get("embedding_type") or "cellprofiler"
    )

    save_path = Path(save_path)
    save_dir = save_path.parent / "anndata_objects"
    save_dir.mkdir(parents=True, exist_ok=True)

    agg_config = config.get("aggregation", {})
    guide_cfg = agg_config.get("guide_level", {})
    gene_cfg = agg_config.get("gene_level", {})

    print("\n" + "=" * 60)
    print(f"Processing {feature_type} features: {save_path.name}")
    print("=" * 60)
    t_start = time.time()

    if feature_type == "cellprofiler":
        cell_adata = create_adata_object(
            str(save_path),
            config=config,
            cell_type=cell_type,
            embedding_type=feature_type,
        )
        # CellProfiler CSVs interleave multiple channels' features; split per reporter.
        if (
            "channel_mapping" in cell_adata.uns
            and len(cell_adata.uns["channel_mapping"]) >= 1
        ):
            reporter_adatas = split_adata_by_reporter(cell_adata, verbose=True)
            for reporter, adata_cell in reporter_adatas.items():
                print(f"\n--- reporter: {reporter} (channel: {adata_cell.uns['channel']}) ---")
                _save_level_outputs(adata_cell, reporter, save_dir, guide_cfg, gene_cfg)
        else:
            print("\n(No channel_mapping found - saving combined file)")
            _save_level_outputs(cell_adata, None, save_dir, guide_cfg, gene_cfg)
    else:
        channel, experiment = _detect_channel_and_experiment(save_path)
        from ops_utils.data.feature_metadata import FeatureMetadata

        reporter = FeatureMetadata().get_biological_signal(experiment, channel)
        print(f"channel={channel}  experiment={experiment}  reporter={reporter}")
        cell_adata = create_adata_object_embedding(
            str(save_path),
            config=config,
            channel=channel,
            experiment=experiment,
            cell_type=cell_type,
            embedding_type=feature_type,
        )
        _save_level_outputs(cell_adata, reporter, save_dir, guide_cfg, gene_cfg)

    elapsed = time.time() - t_start
    print("\n" + "=" * 60)
    print(f"Pipeline completed in {elapsed:.2f}s ({elapsed/60:.2f} min)")
    print("=" * 60 + "\n")
    return cell_adata


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Process a feature/embedding CSV into AnnData objects."
    )
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--config_path", type=str, default=None)
    args = parser.parse_args()
    process_features_csv(args.save_path, config_path=args.config_path)


if __name__ == "__main__":
    main()
