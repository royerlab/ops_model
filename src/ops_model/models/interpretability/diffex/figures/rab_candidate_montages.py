"""Top-N montages (KO complex + per-marker NTC) for each cis-Golgi/Rab-slot candidate, so specific
cells can be picked. Complexes only have top-30 cells."""
from cis_golgi_alternatives import CANDS
from debug_setacc_top100 import montage
from ops_model.models.interpretability.diffex.classifier.config import slugify

ntc_done = set()
for c in CANDS:
    try:
        montage(c["mc"], c["ch"], "complexes", c["key"],
                f"KO — {c['top_label']} · {c['marker_label'].replace(chr(10),' ')}   rank-ordered set-accuracy",
                f"rab_cand_KO_{c['slug']}_{slugify(c['key'])[:24]}")
    except Exception as e:
        print(f"skip KO {c['slug']}: {e}")
    if c["slug"] not in ntc_done:
        ntc_done.add(c["slug"])
        try:
            montage(c["mc"], c["ch"], "complexes", "NTC",
                    f"NTC — {c['marker_label'].replace(chr(10),' ')} marker   rank-ordered set-accuracy",
                    f"rab_cand_NTC_{c['slug']}")
        except Exception as e:
            print(f"skip NTC {c['slug']}: {e}")
