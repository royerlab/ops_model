"""Panel-E-style figure — top set-accuracy cells in label-free 2D phase, KO vs NTC.
Groups: TIMM23 & TIPARP (gene-level), Arp2/3 & Core Mediator (complex). Picks set in
_setacc_phase.COLS_PHASE. Vector output (SVG + PNG)."""
from figure4_setacc_panel import make_panel
from _setacc_phase import COLS_PHASE, column_tiles_phase

if __name__ == "__main__":
    make_panel(COLS_PHASE, "Top set-accuracy cells (phase)", "panelE_phase_setacc",
               tiles_fn=column_tiles_phase, bottom_caption="Label-free 2D Phase")
