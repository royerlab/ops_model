"""Fluorescence 'top-predictive cells' panel — variant B (gene-KO, multibag SHAP): TOMM20, POLR1H,
CFL1 (actin FastAct), mTOR (LysoTracker). NTC top / KO bottom, per-column KO+NTC intensity window.

Run: python figure4_setacc_panel_fluorB.py"""
from figure4_setacc_panel import make_panel
from _setacc_common import column_tiles

COLS = [
    dict(slug="Mitochondria_TOMM20", mc="Mitochondria_TOMM20", ch="CP1_mitochondria_TOMM20",
         block="genes", key="TOMM20", top_label="TOMM20", marker_label="Mitochondria\n(TOMM20)",
         ko_rank=1, ntc_rank=18),
    dict(slug="nucleolus_GC_NPM3", mc="nucleolus-GC_NPM3", ch="GFP",
         block="genes", key="ZNRD1", top_label="POLR1H", marker_label="Nucleoli\n(NPM3-GFP)",
         ko_rank=1, ntc_rank=5),
    dict(slug="actin_filament_FastAct_SPY555_Live_Cell_Dye", mc="actin filament_FastAct_SPY555 Live Cell Dye",
         ch="mCherry", block="genes", key="CFL1", top_label="CFL1", marker_label="Actin\n(FastAct SPY555)",
         ko_rank=4, ntc_rank=20),
    dict(slug="lysosome_LysoTracker_live_cell_dye", mc="lysosome_LysoTracker live-cell dye", ch="GFP",
         block="genes", key="MTOR", top_label="mTOR", marker_label="Lysosome\n(LysoTracker)",
         ko_rank=9, ntc_rank=2),
]

if __name__ == "__main__":
    make_panel(COLS, "Top-predictive cells (fluorescence)", "panelG_fluor_variantB", tiles_fn=column_tiles, title_in=0.66)
