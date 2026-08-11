"""Phase KO montages for a sample of very-strong set-accuracy geneKOs/complexes to pick panel-E
gene/complex columns from. NTC = shared debug_phase_NTC.png."""
from _setacc_common import _materialize, slugify
from _setacc_phase import PHASE_CH, phase_df
from debug_setacc_top100 import render_montage

SAMPLE = [
    ("genes", "MICOS13"), ("genes", "KIF23"), ("genes", "CAPZB"), ("genes", "SAMM50"),
    ("genes", "SON"), ("genes", "RAB7A"), ("genes", "SRSF3"), ("genes", "MTOR"),
    ("complexes", "DNA-directed RNA polymerase I complex"),
    ("complexes", "ESCRT-III complex"),
    ("complexes", "TRAPP II complex, TRAPPC2 variant"),
    ("complexes", "Nuclear pore complex"),
    ("complexes", "COP9 signalosome variant 1"),
]

for block, key in SAMPLE:
    try:
        raw, recs = _materialize(phase_df(block, key).head(100), None, PHASE_CH, key)
        render_montage(raw, recs, f"KO — {key} (phase)   rank-ordered set-accuracy",
                       f"debug_phase_KO_{slugify(key)[:34]}")
    except Exception as e:
        print(f"skip {key}: {e}")
