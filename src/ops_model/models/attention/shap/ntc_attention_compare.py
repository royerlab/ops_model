"""Diagnostic: 3×10 PDF comparing NTC-sampling strategies.

For one PMA NTC CSV (default phase, --modality fluor for fluor):
  row 1 — top 10 by attention rank (rank_type=top, smallest `rank`)
  row 2 — random 10 from the rank_type=top pool
  row 3 — bottom 10 by attention (rank_type=bottom, largest `rank` =
          absolute lowest attention)

Reuses StoreCache + BaseDataset + _render_cell from attention_atlas
so tile loading matches the atlas exactly (same channel mapping,
masks, vmin/vmax).

Usage:
  python ntc_attention_compare.py --output /tmp/ntc_compare.pdf
  python ntc_attention_compare.py --modality fluor --channel "lysosome_LysoTracker live-cell dye"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Atlas helpers live in the sibling atlas/ subdir
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "atlas"))
from attention_atlas import (  # noqa: E402
    StoreCache,
    _build_base_dataset,
    _render_cell,
    _resolve_fluor_channel,
    PHASE_MASK_PREFERENCE,
    FLUOR_MASK_PREFERENCE,
)

DEFAULT_PHASE_CSV = Path(
    "/hpc/projects/icd.fast.ops/models/alex_lin_attention/v3/attention_v3/"
    "pma_top_phase_cells_chad_ntc_v3.csv"
)
DEFAULT_FLUOR_CSV = Path(
    "/hpc/projects/icd.fast.ops/models/alex_lin_attention/v3/attention_v3/"
    "pma_top_fluorescent_cells_chad_ntc_v3.csv"
)


def _pick_groups(df: pd.DataFrame, n: int, seed: int) -> dict[str, pd.DataFrame]:
    top = df[df["rank_type"] == "top"]
    bot = df[df["rank_type"] == "bottom"]
    rng = np.random.default_rng(seed)
    rand_idx = rng.choice(len(top), size=min(n, len(top)), replace=False)
    return {
        "top10":    top.nsmallest(n, "rank").reset_index(drop=True),
        "random10": top.iloc[rand_idx].reset_index(drop=True),
        "bottom10": bot.nlargest(n, "rank").reset_index(drop=True),
    }


def _to_cell_rows(rows: pd.DataFrame, kind: str) -> list[dict]:
    out = []
    for _, r in rows.iterrows():
        out.append({
            "experiment": str(r["experiment"]),
            "well":        str(r["well"]),
            "segmentation": int(r["segmentation"]),
            "x_pheno":     float(r["x_pheno"]),
            "y_pheno":     float(r["y_pheno"]),
            "gene":        "NTC",
            "kind":        kind,
            "rank":        int(r.get("rank", -1)),
            "pma_attention": float(r.get("pma_attention", float("nan"))),
            "channel":     str(r.get("channel", "")),
        })
    return out


def _fluor_channel_index(store, channel_map: dict, viz_channel: str) -> int | None:
    """Resolve a CSV viz_channel (marker name like 'lysosome_LysoTracker live-cell dye'
    or 'p21_p21 (rabbit-647)') to its zarr channel-axis index.

    Zarr stores list channels as FLUOROPHORES (gfp / mcherry / cy5 / etc.) or
    fixed-modality keys (CP1_*, CP2_*, 4i_*) — NOT as marker labels. The
    channel_map yaml carries the marker label per zarr channel. We reuse
    the atlas's `_resolve_fluor_channel` so this matches production
    rendering exactly.
    """
    if store is None:
        return None
    zarr_ch = _resolve_fluor_channel(store, channel_map, viz_channel)
    if zarr_ch is None:
        return None
    names = list(store.channel_names)
    return names.index(zarr_ch) if zarr_ch in names else None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--modality", choices=("phase", "fluor"), default="phase")
    p.add_argument("--csv", type=Path, default=None,
                    help="PMA NTC CSV; defaults to the chad phase/fluor CSV based on --modality.")
    p.add_argument("--channel", default=None,
                    help="(fluor only) viz_channel to render. Default: pick "
                         "the most common channel in the CSV.")
    p.add_argument("--n", type=int, default=10, help="Cells per row")
    p.add_argument("--crop-size", type=int, default=306,
                    help="Tile size in pixels (atlas default is 306).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--output", type=Path,
        default=Path(__file__).resolve().parents[3] /
                "outputs" / "ko_shap_analysis" /
                "ntc_attention_compare.pdf",
        help="Output PDF path. Default: <repo>/outputs/ko_shap_analysis/.",
    )
    args = p.parse_args()

    csv_path = args.csv or (DEFAULT_PHASE_CSV if args.modality == "phase"
                             else DEFAULT_FLUOR_CSV)
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df):,} rows from {csv_path.name}")

    if args.modality == "fluor":
        channel = args.channel or df["channel"].value_counts().index[0]
        df = df[df["channel"].astype(str) == channel].reset_index(drop=True)
        print(f"Filtered to channel={channel!r} → {len(df):,} rows")
        kind = "fluor"
    else:
        channel = "Phase"
        kind = "phase"

    groups = _pick_groups(df, args.n, args.seed)
    for name, sub in groups.items():
        print(f"  {name}: {len(sub)} cells "
              f"(rank {sub['rank'].min()}–{sub['rank'].max()}, "
              f"attn {sub['pma_attention'].min():.3g}–{sub['pma_attention'].max():.3g})")

    # Load all 3*n cells through one BaseDataset call.
    all_rows = []
    for name in ("top10", "random10", "bottom10"):
        all_rows.extend(_to_cell_rows(groups[name], kind))
    store_cache = StoreCache()
    for r in all_rows:
        store_cache.get(r["experiment"])

    ds, input_indices = _build_base_dataset(all_rows, store_cache, args.crop_size)
    crops_by_idx: dict[int, tuple[np.ndarray, np.ndarray | None]] = {}
    if ds is not None:
        for ds_i, in_idx in enumerate(input_indices):
            try:
                batch = ds[ds_i]
                data = batch["data"].numpy()
                mask = batch["mask"].numpy()[0].astype(bool)
            except Exception as e:
                print(f"  [load] cell {in_idx} failed: {e}")
                continue
            crops_by_idx[in_idx] = (data, mask)

    # 3 × N grid
    n = args.n
    fig, axes = plt.subplots(
        3, n, figsize=(n * 1.6, 3 * 1.7 + 0.5),
        squeeze=False,
    )
    for row_i, (name, sub) in enumerate(groups.items()):
        for c in range(n):
            ax = axes[row_i, c]
            in_idx = row_i * n + c
            if c >= len(sub):
                _render_cell(ax, None, None, "")
                continue
            packed = crops_by_idx.get(in_idx)
            cell = all_rows[in_idx]
            if packed is None:
                _render_cell(ax, None, None, "missing")
                continue
            data, mask = packed

            # Pick which channel axis to display.
            if args.modality == "phase":
                store = store_cache.stores.get(cell["experiment"])
                ch_idx = 0
                if store is not None:
                    names = [n.lower() for n in store.channel_names]
                    for cand in ("phase2d", "phase"):
                        if cand in names:
                            ch_idx = names.index(cand)
                            break
                img = data[ch_idx]
            else:
                store = store_cache.stores.get(cell["experiment"])
                cm = store_cache.channel_map(cell["experiment"])
                ch_idx = _fluor_channel_index(store, cm, cell["channel"])
                if ch_idx is None:
                    _render_cell(ax, None, None,
                                 f"r={cell['rank']}\n(no channel match)")
                    continue
                img = data[ch_idx]

            title = f"r={cell['rank']}\na={cell['pma_attention']:.2g}"
            _render_cell(ax, img, mask, title)

    # Row labels on the left
    for row_i, label in enumerate(
        ["TOP 10\n(rank=1..)", "RANDOM 10\n(from top)", "BOTTOM 10\n(lowest attn)"]
    ):
        axes[row_i, 0].set_ylabel(label, fontsize=10, rotation=0, ha="right",
                                   va="center", labelpad=30)
        axes[row_i, 0].set_axis_on()
        for s in axes[row_i, 0].spines.values():
            s.set_visible(False)
        axes[row_i, 0].set_xticks([]); axes[row_i, 0].set_yticks([])

    fig.suptitle(
        f"NTC attention sampling comparison — "
        f"{args.modality}{' (' + channel + ')' if args.modality == 'fluor' else ''}",
        fontsize=11, y=0.995,
    )
    fig.tight_layout(rect=[0.04, 0.0, 1.0, 0.97])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, format="pdf", dpi=200)
    plt.close(fig)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
