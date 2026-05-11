"""Cell-level filtering framework for the combination pipeline.

Filters are applied to individual cell-level AnnData objects immediately after
loading, before any PCA, aggregation, or normalization.

A CellFilter is any callable (AnnData) -> AnnData.  Two concrete filters are
provided:

  DudGuideFilter      — remove cells whose sgRNA is in a known dud-guide list.
  TopPhenotypeFilter  — keep only the top-N cells per gene by a pre-ranked
                        phenotype score, derived from a model attention score
                        on Phase features.  Cells not in the score file are
                        dropped; NTC/control cells are always retained.

Multiple filters can be composed via ComposedFilter, which applies them in
sequence.  build_cell_filter() constructs a filter from a list of config dicts
read from the combination YAML.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

import anndata as ad
import numpy as np
import pandas as pd

from ops_model.features.anndata_utils import _guide_col

logger = logging.getLogger(__name__)

CellFilter = Callable[[ad.AnnData], ad.AnnData]


class DudGuideFilter:
    """Remove cells whose per-construct identifier appears in a known dud list.

    Parameters
    ----------
    dud_guides:
        Identifier values to exclude. Matched against the column named by
        ``adata.uns["guide_col"]`` (default ``"sgRNA"``).
    """

    def __init__(self, dud_guides: list[str]) -> None:
        self.dud_guides = set(dud_guides)

    def __call__(self, adata: ad.AnnData) -> ad.AnnData:
        guide_col = _guide_col(adata)
        if guide_col not in adata.obs.columns:
            return adata
        mask = ~adata.obs[guide_col].isin(self.dud_guides)
        n_removed = int((~mask).sum())
        if n_removed > 0:
            n_dud_found = adata.obs.loc[~mask, guide_col].nunique()
            logger.info(
                f"  DudGuideFilter: removed {n_removed}/{adata.n_obs} cells "
                f"({n_removed / adata.n_obs:.1%}) matching {n_dud_found} dud guides"
            )
        return adata[mask.values].copy()


class TopPhenotypeFilter:
    """Keep the top-N cells per gene by a pre-ranked phenotype attention score.

    Cells are selected by joining against a pre-ranked CSV (one row per cell,
    ranked 1–5000 within each gene by descending pma_attention score).  Only
    cells with rank <= top_n are retained.

    NTC / control cells are always passed through regardless of whether they
    appear in the score file.

    **Cross-reporter note:** the score is derived from Phase channel features.
    When this filter is applied to non-Phase reporters (MAP4, FeRhoNox, etc.),
    cells are selected or dropped purely on the basis of their Phase attention
    score.  A cell with strong MAP4 signal but low Phase attention will be
    dropped.  This is intentional: Phase attention is used as a global cell-
    quality gate across all reporters, exploiting the fact that all reporters
    share the same segmentation and therefore the same spatial coordinates.

    **Downsampling interaction:** the pre-scan cell-count step in
    _process_signal_group reads unfiltered counts before this filter runs.
    Proportional downsampling fractions are therefore computed on stale totals.
    It is strongly recommended to set ``downsampling.enabled: false`` in the
    combination config when using TopPhenotypeFilter.

    Parameters
    ----------
    score_file:
        Path to the ranked CSV.  Expected columns: experiment, x_pheno,
        y_pheno, rank (and gene, but that is not used for the join).
    top_n:
        Keep cells with rank <= top_n.  The CSV contains at most 5000 rows
        per gene; values above 5000 trigger a warning.
    control_gene:
        Value of obs["perturbation"] that identifies non-targeting controls.
        Matching cells are always retained.  Defaults to "NTC".
    """

    _MAX_RANK = 5000

    def __init__(
        self,
        score_file: str,
        top_n: int,
        control_gene: str = "NTC",
    ) -> None:
        self.score_file = score_file
        self.top_n = top_n
        self.control_gene = control_gene
        self._allow_df: Optional[pd.DataFrame] = None

        if top_n > self._MAX_RANK:
            logger.warning(
                f"TopPhenotypeFilter: top_n={top_n} exceeds the maximum rank "
                f"in the score file ({self._MAX_RANK}). All scored cells will "
                f"be retained for genes with fewer than {top_n} entries."
            )

    # ------------------------------------------------------------------
    # Pickle support — exclude cache so workers always rebuild from file
    # ------------------------------------------------------------------

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_allow_df"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _load(self) -> None:
        logger.info(f"  TopPhenotypeFilter: loading score file {self.score_file}")
        df = pd.read_csv(
            self.score_file,
            usecols=["experiment", "x_pheno", "y_pheno", "rank"],
        )
        df = df[df["rank"] <= self.top_n][
            ["experiment", "x_pheno", "y_pheno"]
        ].rename(columns={"x_pheno": "x_position", "y_pheno": "y_position"})
        self._allow_df = df.drop_duplicates().reset_index(drop=True)
        logger.info(
            f"  TopPhenotypeFilter: allow-list built — "
            f"{len(self._allow_df):,} cells (top_n={self.top_n})"
        )

    # ------------------------------------------------------------------
    # Filter
    # ------------------------------------------------------------------

    def __call__(self, adata: ad.AnnData) -> ad.AnnData:
        if self._allow_df is None:
            self._load()

        n_before = adata.n_obs

        # Separate control cells — always retain
        if "perturbation" in adata.obs.columns:
            control_mask = adata.obs["perturbation"].values == self.control_gene
        else:
            control_mask = np.zeros(n_before, dtype=bool)

        n_control = int(control_mask.sum())
        non_control_pos = np.where(~control_mask)[0]

        if len(non_control_pos) == 0:
            return adata

        # Required obs columns for the join
        missing = [
            c for c in ("experiment", "x_position", "y_position")
            if c not in adata.obs.columns
        ]
        if missing:
            logger.warning(
                f"  TopPhenotypeFilter: obs is missing columns {missing} — "
                f"skipping filter, returning all cells unfiltered"
            )
            return adata

        # Merge non-control rows against the allow-list to find kept positions
        obs_nc = adata.obs.iloc[non_control_pos][
            ["experiment", "x_position", "y_position"]
        ].copy()
        obs_nc["_pos"] = non_control_pos

        merged = obs_nc.merge(
            self._allow_df,
            on=["experiment", "x_position", "y_position"],
            how="inner",
        )
        kept_non_control_pos = merged["_pos"].values

        # Build boolean keep mask
        keep = np.zeros(n_before, dtype=bool)
        keep[control_mask] = True
        if len(kept_non_control_pos) > 0:
            keep[kept_non_control_pos] = True

        n_after = int(keep.sum())
        n_removed = n_before - n_after
        n_ko_remaining = n_after - n_control
        logger.info(
            f"  TopPhenotypeFilter: {n_before} → {n_after} cells "
            f"({n_removed} removed; {n_control} controls, {n_ko_remaining} KO cells remaining)"
        )

        return adata[keep].copy()


class ComposedFilter:
    """Apply a sequence of CellFilters in order.

    Parameters
    ----------
    filters:
        One or more CellFilter callables to apply sequentially.
    """

    def __init__(self, *filters: CellFilter) -> None:
        self.filters = list(filters)

    def __call__(self, adata: ad.AnnData) -> ad.AnnData:
        for f in self.filters:
            adata = f(adata)
        if adata.n_obs == 0:
            logger.warning("  ComposedFilter: all cells removed after filtering")
        return adata


def build_cell_filter(configs: list[dict]) -> Optional[CellFilter]:
    """Build a CellFilter from a list of config dicts (parsed from YAML).

    Returns None if configs is empty.

    Supported filter types and their required config keys:

      dud_guides:
        guides: list[str]

      top_phenotype:
        score_file: str
        top_n: int
        control_gene: str  (optional, default "NTC")
    """
    if not configs:
        return None

    filters: list[CellFilter] = []
    for cfg in configs:
        filter_type = cfg.get("type")
        if filter_type == "dud_guides":
            filters.append(DudGuideFilter(dud_guides=cfg["guides"]))
        elif filter_type == "top_phenotype":
            filters.append(
                TopPhenotypeFilter(
                    score_file=cfg["score_file"],
                    top_n=int(cfg["top_n"]),
                    control_gene=cfg.get("control_gene", "NTC"),
                )
            )
        else:
            raise ValueError(f"Unknown cell filter type: {filter_type!r}")

    if len(filters) == 1:
        return filters[0]
    return ComposedFilter(*filters)
