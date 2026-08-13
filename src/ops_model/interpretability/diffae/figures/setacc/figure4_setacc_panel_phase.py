"""Panel-E-style figure — top set-accuracy cells in label-free 2D phase, KO vs NTC.
Groups: TIMM23 & TIPARP (gene-level), Arp2/3 & Core Mediator (complex). Picks set in
_setacc_phase.COLS_PHASE. Vector output (SVG + PNG)."""
from ops_model.interpretability.diffae.figures.setacc.figure4_setacc_panel import make_panel
from ops_model.interpretability.diffae.figures._setacc_phase import COLS_PHASE, column_tiles_phase

if __name__ == "__main__":
    make_panel(COLS_PHASE, "Top-predictive cells (phase)", "panelE_phase_setacc",
               tiles_fn=column_tiles_phase, bottom_caption="Label-free 2D Phase",
               title_in=0.66)   # room for 2-line column titles + suptitle, snug
