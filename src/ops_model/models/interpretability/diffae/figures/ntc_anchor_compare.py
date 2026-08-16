"""Quick comparison grid: top-50 NTC anchor cells from the OLD v5-accuracy ranking vs the NEW phase
multirank (shap_screen). Crops the phase channel from phenotyping_v3.zarr (same source as the viewer)."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg"); matplotlib.rcParams["pdf.fonttype"] = 42
import matplotlib.pyplot as plt
import zarr
from ..viewer.build_pc_crops_masked import BASE, CROP_SIZE, PHASE_CHANNEL, _crop, _render_gray, _zarr_patch

R = "/hpc/projects/icd.fast.ops/models/diffex/viewer_assets_v5/_rankings"
N = 50


def top_ntc(parquet, n=N):
    df = pd.read_parquet(parquet, columns=["gene", "experiment", "well", "x_pheno", "y_pheno", "rank"])
    return df[df["gene"].astype(str) == "NTC"].sort_values("rank").head(n).reset_index(drop=True)


def crops(rows):
    _zarr_patch(); half = CROP_SIZE // 2; cache = {}; out = []
    for r in rows.itertuples():
        key = (r.experiment, r.well)
        if key not in cache:
            pos = f"{BASE}/{r.experiment}/3-assembly/phenotyping_v3.zarr/{r.well[0]}/{r.well[1:]}/0"
            try:
                cache[key] = zarr.open(f"{pos}/0", mode="r")
            except Exception:
                cache[key] = None
        img = cache[key]
        if img is None:
            out.append(None); continue
        try:
            out.append(_render_gray(_crop(img, PHASE_CHANNEL, int(round(r.x_pheno)), int(round(r.y_pheno)), half)))
        except Exception:
            out.append(None)
    return out


def panel(ax_grid, imgs, title):
    for i in range(N):
        a = ax_grid[i]
        if i < len(imgs) and imgs[i] is not None:
            a.imshow(imgs[i]);
        a.set_xticks([]); a.set_yticks([])
        a.set_title(str(i + 1), fontsize=5, pad=1)


old = top_ntc(f"{R}/pma_v5_phase_geneKO.parquet")
new = top_ntc(f"{R}/pma_shap_phase_geneKO.parquet")
oi, ni = crops(old), crops(new)

fig = plt.figure(figsize=(20, 22))
outer = fig.add_gridspec(2, 1, hspace=0.12)
for row, (imgs, title) in enumerate([(oi, "OLD v5-accuracy ranking — top-50 NTC anchors"),
                                     (ni, "NEW phase multirank (shap_screen) — top-50 NTC anchors")]):
    inner = outer[row].subgridspec(5, 10, hspace=0.25, wspace=0.05)
    axs = [fig.add_subplot(inner[j]) for j in range(N)]
    panel(axs, imgs, title)
    fig.text(0.5, 0.905 - row * 0.485, title, ha="center", fontsize=15, fontweight="bold")
out = "/hpc/projects/icd.fast.ops/analysis/ntc_anchor_old_vs_multirank.png"
fig.savefig(out, dpi=110, bbox_inches="tight"); plt.close(fig)
print("wrote", out)
