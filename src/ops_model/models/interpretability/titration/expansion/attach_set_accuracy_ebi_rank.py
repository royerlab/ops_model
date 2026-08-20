"""Attach set_accuracy_ebi_rank / set_accuracy_ebi_rank_type + Alex's
authoritative gene name for cells he ranked.

The CSV/h5ad gene-name mismatch (NTC cells reassigned, barcode-mapping
drift) makes joining by (well, seg) alone unsafe if we then group by
obs.perturbation. We attach Alex's `gene_name` column as
`set_accuracy_ebi_gene` and let the sweep use that grouping key for
set_accuracy_ebi selections. All ranked cells retained (~100%).
"""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

CSV = Path(
    "/hpc/projects/icd.fast.ops/models/alex_lin_attention/v4/"
    "accuracy_ranking/ebi_pergene_phase_cell_rankings.csv"
)
PCA_DIR = Path(
    "/hpc/projects/icd.fast.ops/models/alex_lin_attention/v4/"
    "expansion_v1/per_experiment_v4_pca"
)


def main() -> None:
    print(f"Loading CSV: {CSV}")
    df = pd.read_csv(
        CSV,
        usecols=["gene_name", "experiment", "well", "segmentation_id", "rank"],
    )
    print(f"  {len(df):,} rows")
    df["well"] = df["well"].astype(str)

    # Per-cell dedup: keep the (gene_name, min_rank) — cells that appear
    # under multiple complexes take the best rank + first gene alphabetically.
    dedup = (
        df.sort_values(["experiment", "well", "segmentation_id", "rank"])
          .groupby(["experiment", "well", "segmentation_id"], as_index=False)
          .first()[["experiment", "well", "segmentation_id", "gene_name", "rank"]]
    )
    print(f"  dedup: {len(dedup):,} unique (exp, well, seg) cells")

    covered_exps = set(dedup["experiment"].unique())
    all_exps = [p.stem for p in sorted(PCA_DIR.glob("*.h5ad"))]

    for exp in all_exps:
        p = PCA_DIR / f"{exp}.h5ad"
        a = ad.read_h5ad(p)
        obs = a.obs.copy()
        obs["well"] = obs["well"].astype(str)

        if exp in covered_exps:
            g = dedup[dedup["experiment"] == exp].rename(
                columns={"gene_name": "_alex_gene"})
            merged = obs.merge(
                g[["well", "segmentation_id", "_alex_gene", "rank"]],
                on=["well", "segmentation_id"],
                how="left",
                validate="m:1",
            )
            rank_vals = merged["rank"].to_numpy(dtype=np.float64)
            alex_gene = merged["_alex_gene"].astype(str).values
            # NaN gene → keep as empty string (cell not ranked)
            alex_gene = np.where(merged["_alex_gene"].isna().values, "", alex_gene)
        else:
            rank_vals = np.full(len(a), np.nan, dtype=np.float64)
            alex_gene = np.array([""] * len(a))

        n_matched = int(np.isfinite(rank_vals).sum())
        pct = n_matched / max(1, len(a)) * 100
        n_mismatch = int(np.sum(
            (rank_vals == rank_vals)  # notna
            & (alex_gene != obs["perturbation"].astype(str).values)
        ))
        print(f"  {exp}: matched {n_matched:,}/{len(a):,} ({pct:.1f}%)  "
              f"of which obs.pert != alex.gene: {n_mismatch:,}")

        a.obs["set_accuracy_ebi_rank"] = rank_vals
        rank_type = np.where(np.isfinite(rank_vals), "top", "")
        a.obs["set_accuracy_ebi_rank_type"] = pd.Categorical(rank_type)
        a.obs["set_accuracy_ebi_gene"] = pd.Categorical(alex_gene)
        a.write_h5ad(p)


if __name__ == "__main__":
    main()
