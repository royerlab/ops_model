"""Top-N phase montages (KO per group + a single phase NTC) for picking cells for panel E."""
from _setacc_common import _materialize, slugify
from _setacc_phase import COLS_PHASE, PHASE_CH, phase_df, phase_ntc
from debug_setacc_top100 import render_montage

for c in COLS_PHASE:
    raw, recs = _materialize(phase_df(c["block"], c["key"]).head(100), None, PHASE_CH, c["key"])
    render_montage(raw, recs, f"KO — {c['top_label'].replace(chr(10),' ')} (phase)   rank-ordered set-accuracy",
                   f"debug_phase_KO_{slugify(c['key'])[:30]}")

raw, recs = _materialize(phase_ntc().head(100), None, PHASE_CH, "NTC")
render_montage(raw, recs, "NTC — phase   rank-ordered set-accuracy", "debug_phase_NTC")
