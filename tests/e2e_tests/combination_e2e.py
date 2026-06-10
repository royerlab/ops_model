"""End-to-end test for the embedding combination pipeline (pca_optimization).

Self-contained script (run directly, not via pytest). It exercises the full
multi-experiment combination path for a minimal example:

    1. discover 5 experiments that share the cell_dino reporter set
       (Phase / 5xUPRE / SEC61B), pull ALL channels (reporters) for each
    2. subset every per-experiment, per-reporter cell-level h5ad to NTC + 31
       genes (<=50 cells per gene), saved to a tmp dir
    3. write a config pointing `signal_paths` at the subsetted h5ads (one pooled
       list of the 5 experiments per reporter) and run the pipeline in-process
       (slurm: false) via run_from_config
    4. verify the canonical combined outputs are produced

The combiner pools cells across experiments sharing a signal, fits PCA, and
aggregates to guide/gene level (Phase 2 also NTC-normalizes + scores metrics).
On a 32-gene subset the metric values are not meaningful, so this test only
asserts the pipeline completes and produces sane-shaped outputs. All outputs go
to a fresh tmp dir (printed at the end).

Run with:
    uv run python tests/e2e_tests/combination_e2e.py
"""

import tempfile
from pathlib import Path

import numpy as np
import anndata as ad
import yaml

from ops_model.post_process.combination.pca_optimization import run_from_config

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ROOT = Path("/hpc/projects/icd.fast.ops")
FEATURE_DIR = "cell_dino_features"  # feature mode: cell_dino
REPORTERS = ["Phase", "5xUPRE", "SEC61B"]  # "all channels"
N_EXPERIMENTS = 5
N_GENES = 32  # NTC + 31 perturbations
CELLS_PER_GENE = 50  # cap per (gene, experiment) for speed
CHAD_ANNOTATION = (
    "/hpc/projects/icd.fast.ops/configs/gene_clusters/"
    "val_library_chad_positive_controls_v1.yml"
)


def anndata_path(experiment: str, reporter: str) -> Path:
    return (
        ROOT
        / experiment
        / "3-assembly"
        / FEATURE_DIR
        / "anndata_objects"
        / f"features_processed_{reporter}.h5ad"
    )


def find_experiments(n: int) -> list[str]:
    """First n experiments (sorted) with cell_dino h5ads for all REPORTERS."""
    found = []
    for d in sorted(ROOT.glob(f"ops0*/3-assembly/{FEATURE_DIR}/anndata_objects")):
        experiment = d.relative_to(ROOT).parts[0]
        if all(anndata_path(experiment, r).exists() for r in REPORTERS):
            found.append(experiment)
        if len(found) == n:
            break
    assert len(found) == n, f"only found {len(found)} experiments with all reporters"
    return found


def pick_genes(experiment: str) -> list[str]:
    """NTC + the 31 most-common non-NTC perturbations in one reference h5ad."""
    a = ad.read_h5ad(anndata_path(experiment, REPORTERS[0]), backed="r")
    counts = a.obs["perturbation"].astype(str).value_counts()
    non_ntc = [g for g in counts.index if g != "NTC"][: N_GENES - 1]
    genes = ["NTC"] + non_ntc
    assert len(genes) == N_GENES
    return genes


def subset_h5ad(src: Path, genes: list[str], out: Path) -> int:
    """Subset src to `genes`, capping cells per gene; write to out. Returns n_obs."""
    a = ad.read_h5ad(src, backed="r")
    pert = a.obs["perturbation"].astype(str).values
    keep = []
    for g in genes:
        idx = np.where(pert == g)[0]
        keep.extend(idx[:CELLS_PER_GENE].tolist())
    keep = sorted(keep)
    sub = a[keep].to_memory()
    sub.write_h5ad(out)
    return sub.n_obs


def build_config(signal_paths: dict, output_dir: Path) -> dict:
    return {
        "cell_dino": True,
        "signal_paths": signal_paths,  # {reporter: [pooled per-experiment h5ads]}
        "output_dir": str(output_dir),
        "run_tag": "e2e",
        "slurm": False,  # run Phase 1 + Phase 2 in-process
        "fixed_threshold": 0.80,  # single PCA cutoff (skip the consensus sweep)
        "norm_method": "ntc",
        "zscore_per_experiment": True,
        "second_pca": False,  # skip the second-pass PCA consensus
        "chad_annotation": CHAD_ANNOTATION,
    }


def verify_outputs(output_dir: Path) -> None:
    gene = list(output_dir.rglob("gene_embedding_pca_optimized.h5ad"))
    guide = list(output_dir.rglob("guide_pca_optimized.h5ad"))
    report = list(output_dir.rglob("pca_report.csv"))
    assert gene, f"no gene_embedding_pca_optimized.h5ad under {output_dir}"
    assert guide, f"no guide_pca_optimized.h5ad under {output_dir}"
    assert report, f"no pca_report.csv under {output_dir}"

    adata_gene = ad.read_h5ad(gene[0])
    adata_guide = ad.read_h5ad(guide[0])
    assert adata_gene.n_obs > 0 and adata_gene.n_vars > 0, "empty gene-level output"
    assert adata_guide.n_obs > 0 and adata_guide.n_vars > 0, "empty guide-level output"
    print(f"[4] Combined outputs OK:")
    print(f"      guide: {adata_guide.n_obs} x {adata_guide.n_vars} -> {guide[0]}")
    print(f"      gene:  {adata_gene.n_obs} x {adata_gene.n_vars} -> {gene[0]}")
    print(f"      report: {report[0]}")


def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="combination_e2e_"))
    inputs_dir = tmp / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    output_dir = tmp / "combined"

    print(f"Working dir: {tmp}\n")

    experiments = find_experiments(N_EXPERIMENTS)
    genes = pick_genes(experiments[0])
    print(f"[0] Experiments: {experiments}")
    print(f"[0] Genes ({len(genes)}): NTC + {len(genes) - 1} perturbations\n")

    # 1 + 2. subset every (experiment, reporter) h5ad -> tmp, build signal_paths
    signal_paths: dict[str, list[str]] = {r: [] for r in REPORTERS}
    for experiment in experiments:
        for reporter in REPORTERS:
            out = inputs_dir / f"{experiment}_{reporter}.h5ad"
            n = subset_h5ad(anndata_path(experiment, reporter), genes, out)
            signal_paths[reporter].append(str(out))
            print(f"[1] {experiment}/{reporter}: {n} cells -> {out.name}")

    config = build_config(signal_paths, output_dir)
    config_path = tmp / "config.yml"
    config_path.write_text(yaml.safe_dump(config))
    print(f"\n[2] Wrote config -> {config_path}")

    # 3. run the combination pipeline in-process
    run_from_config(str(config_path))

    # 4. verify
    verify_outputs(output_dir)

    print(f"\n✓ Combination e2e PASSED. Outputs under: {tmp}")


if __name__ == "__main__":
    main()
