"""Candidate strip for the panel-D Rab-slot (alternatives to COPI·cis-Golgi) — KO vs NTC at rank 1
(most distinctive) for a few trafficking/organelle complex+marker pairs, to pick the most obvious."""
from ops_model.interpretability.diffae.figures.setacc.figure4_setacc_panel import make_panel

CANDS = [
    dict(slug="cis_Golgi_mStayGold_CENPRaltORF", mc="cis-Golgi_mStayGold-CENPRaltORF", ch="GFP",
         key="COPI vesicle coat complex, COPG1-COPZ1 variant", top_label="COPI", marker_label="cis-Golgi\n(CENPR)"),
    dict(slug="trans_Golgi_VAMP3", mc="trans-Golgi_VAMP3", ch="GFP",
         key="COPI vesicle coat complex, COPG1-COPZ1 variant", top_label="COPI", marker_label="trans-Golgi\n(VAMP3)"),
    dict(slug="ER_Golgi_COPE", mc="ER/Golgi_COPE", ch="GFP",
         key="COPI vesicle coat complex, COPG1-COPZ1 variant", top_label="COPI", marker_label="ER/Golgi\n(COPE)"),
    dict(slug="late_endosome_RAB7A", mc="late endosome_RAB7A", ch="GFP",
         key="COPI vesicle coat complex, COPG1-COPZ1 variant", top_label="COPI", marker_label="Endosome\n(RAB7A)"),
    dict(slug="lysosome_LysoTracker_live_cell_dye", mc="lysosome_LysoTracker live-cell dye", ch="GFP",
         key="mTORC1 complex", top_label="mTORC1", marker_label="Lysosome\n(LysoTracker)"),
    dict(slug="ER_SEC61B", mc="ER_SEC61B", ch="mCherry",
         key="SEC61 protein-conducting channel complex, SEC1A1 variant", top_label="SEC61", marker_label="ER\n(SEC61B)"),
    dict(slug="Microtubules_Tubulin", mc="Microtubules_Tubulin", ch="CP2_microtubules_Tubulin",
         key="HAUS complex", top_label="HAUS", marker_label="Microtubules\n(Tubulin)"),
]
for c in CANDS:
    c.update(block="complexes", ko_rank=1, ntc_rank=1)

if __name__ == "__main__":
    make_panel(CANDS, "cis-Golgi alternatives — Rab-slot candidates (rank-1 set-accuracy)",
               "rab_slot_alternatives", tile=1.5)
