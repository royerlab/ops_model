"""Resolve representative (gene, channel) per hand-annotated cluster.

Reads ``hand_annotated_cluster.txt`` (preserving cluster headers + gene lists),
cross-checks each cluster's well-known nominee gene against the fluor / phase
attention CSVs, and rewrites the ``# rep_gene`` / ``# rep_channel`` /
``# ch_rank`` / ``# rep_modality`` metadata block.

Resolution priority for each cluster:
  1. WELL_KNOWN gene + a SENSIBLE channel that's in that gene's fluor top-3
  2. Any other cluster gene + a SENSIBLE channel from its fluor top-3
  3. WELL_KNOWN gene + Phase2D (always available — every gene has phase
     attention)

Edit ``SENSIBLE`` (or the per-cluster ``# well_known:`` line in the txt) and
re-run::

    python /hpc/mydata/gav.sturm/ops_mono/resolve_cluster_picks.py
"""
from pathlib import Path
import pandas as pd


TXT = Path(__file__).with_name("hand_annotated_cluster.txt")
FLUOR_CSV = Path(
    "/home/gav.sturm/linked_folders/icd.fast.ops/models/alex_lin_attention/"
    "pma_top_fluorescent_cells_v2.csv"
)
PHASE_CSV = Path(
    "/home/gav.sturm/linked_folders/icd.fast.ops/models/alex_lin_attention/"
    "pma_top_phase_cells_v2.csv"
)
PHASE_CHANNEL = "Phase2D"


# Initial "most well-known gene" nominee per cluster. Used only when the txt
# file has no ``# well_known:`` line for that cluster; otherwise the txt wins
# and you can edit nominees there directly.
WELL_KNOWN_INITIAL = {
    "Actin Organization":                    "DYNC1H1",
    "Cytoskeleton Organization":             "ERN1",
    "Mitochondria Inner Membrane OXPHOS":    "ATP5F1B",
    "Mitochondrial Import":                  "ATAD3A",
    "Galactose Metabolism":                  "GLB1",
    "Lipid Metabolism":                      "CERS6",
    "Hypoxia":                               "MTOR",
    "MTOR Signaling":                        "MTOR",
    "Tight Junctions":                       "CLDN15",
    "Potassium Channels":                    "KCNG1",
    "Sodium Channels":                       "SCN8A",
    "Cristae Formation":                     "SAMM50",
    "Calcium Signaling":                     "MYCL",
    "Calcium-dependent Exocytosis":          "NRAS",
    "Purine Metabolism":                     "IMPDH2",
    "Triglyceride Metabolism":               "APC",
    "ER membrane":                           "CERS6",
    "ER Tubular Network":                    "INPPL1",
    "ER-Golgi Transport":                    "SEC23B",
    "Golgi Organization":                    "HMGCR",
    "Exocytic Vesicles":                     "RAB10",
    "Exosomal Secretion":                    "RAB7A",
    "Lysosome":                              "FTH1",
    "Endocytosis":                           "WASL",
    "Peroxisome":                            "CRAT",
    "Vacuolar Acidification":                "ATP6V1B2",
    "Intermediate Filament Cytoskeleton":    "FOXK1",
    "Mitochondrial Outer Membrane":          "FBXL4",
    "DNA Methylation":                       "DNMT1",
    "snRNA":                                 "INTS1",
    "Nuclear Export":                        "KPNB1",
    "Transcription Initiation":              "RAD21",
    "DNA-templated Transcription":           "POLR2B",
    "Spliceosome Assembly":                  "SF3B3",
    "Splicing":                              "SRSF3",
    "Translation Initiation":                "EIF4G1",
    "Ribosomal Large Subunit":               "RPL5",
    "Ribosomal Small Subunit":               "RPS6",
    "Ribosomal Biogenesis":                  "NOP58",
    "RNA Degradation":                       "EXOSC3",
    "RNA Polymerase":                        "NCL",
    "Spindle Assembly":                      "AURKB",
    "DNA recombination":                     "RPA2",
    "DNA Replication":                       "MCM7",
    "Nuclear Deformation":                   "PUF60",
    "Microtubule Nucleation":                "BUB1B",
    "Actin Nucleation":                      "ARPC2",
    "Actin Projections":                     "TWF1",
}

# CP cell-painting channels (fixed-cell, multi-round) — excluded from
# representative-cell picks because their seg-IDs in the attention CSVs don't
# round-trip through the live-cell `cell_seg` mask used by the animation.
# Live-cell equivalents (lowercase in the channel naming convention) cover
# the same biology and play nicely with the standard mask preference order.
CP_CHANNELS = frozenset({
    "Mitochondria_TOMM20",
    "Microtubules_Tubulin",
    "F-actin_Phalloidin",
    "Plasma Membrane_Wheat Germ Agglutinin",
    "Endoplasmic Reticulum_Concanavalin A",
    "Nucleus_Hoechst",
    "Nucleoli_NPM1",
})


# Biologically-sensible channels per cluster, in priority order. Tighten this
# (drop tangential entries) to push more clusters to the Phase fallback.
# CP channels above are filtered out at resolve time so we don't have to
# scrub them from each cluster's list manually.
SENSIBLE = {
    "Actin Organization":                    ["Microtubules_Tubulin", "microtubules_MAP4", "actin filament_FastAct_SPY555 Live Cell Dye", "F-actin_Phalloidin"],
    "Cytoskeleton Organization":             ["F-actin_Phalloidin", "actin filament_FastAct_SPY555 Live Cell Dye", "Microtubules_Tubulin", "microtubules_MAP4", "laminin_LMNB1"],
    "Intermediate Filament Cytoskeleton":    ["laminin_LMNB1", "Microtubules_Tubulin"],
    "Actin Nucleation":                      ["F-actin_Phalloidin", "actin filament_FastAct_SPY555 Live Cell Dye", "recycling endosome_TFRC", "early endosome_EEA1"],
    "Actin Projections":                     ["F-actin_Phalloidin", "actin filament_FastAct_SPY555 Live Cell Dye", "Plasma Membrane_Wheat Germ Agglutinin", "plasma membrane_ATP1B3", "plasma membrane_SLC3A2"],
    "Microtubule Nucleation":                ["Microtubules_Tubulin", "microtubules_MAP4", "cell proliferation marker_MKI67", "actin filament_FastAct_SPY555 Live Cell Dye"],
    "Spindle Assembly":                      ["Microtubules_Tubulin", "microtubules_MAP4", "cell proliferation marker_MKI67"],

    "Mitochondria Inner Membrane OXPHOS":    ["Mitochondria_TOMM20", "mitochondria_TOMM70A", "mitochondria_ChromaLIVE 561 excitation"],
    "Mitochondrial Import":                  ["Mitochondria_TOMM20", "mitochondria_TOMM70A", "mitochondria_ChromaLIVE 561 excitation"],
    "Cristae Formation":                     ["Mitochondria_TOMM20", "mitochondria_TOMM70A", "mitochondria_ChromaLIVE 561 excitation"],
    "Mitochondrial Outer Membrane":          ["Mitochondria_TOMM20", "mitochondria_TOMM70A", "mitochondria_ChromaLIVE 561 excitation"],

    "Galactose Metabolism":                  ["lysosome_LAMP1", "lysosome_LysoTracker live-cell dye", "endocytic vesicle pH_pHrodo-dextran Live Cell Dye"],
    "Lipid Metabolism":                      ["lipid droplet_BODIPY live cell dye", "lipid droplet_PLIN2", "ER_SEC61B", "Endoplasmic Reticulum_Concanavalin A"],
    "Triglyceride Metabolism":               ["lipid droplet_PLIN2", "lipid droplet_BODIPY live cell dye", "ER_SEC61B"],
    "Purine Metabolism":                     ["ER_SEC61B", "Endoplasmic Reticulum_Concanavalin A", "ER_NCLN", "Nucleoli_NPM1", "nucleolus-DFC_FBL"],
    "Hypoxia":                               ["oxidative stress_CellROX live-cell dye", "Mitochondria_TOMM20", "mitochondria_TOMM70A"],
    "Calcium Signaling":                     ["ER_SEC61B", "ER_NCLN", "Endoplasmic Reticulum_Concanavalin A", "plasma membrane_ATP1B3", "Plasma Membrane_Wheat Germ Agglutinin"],
    "Calcium-dependent Exocytosis":          ["clathrin vesicles_CLTA", "trans-Golgi_VAMP3", "plasma membrane_ATP1B3", "Plasma Membrane_Wheat Germ Agglutinin", "plasma membrane_SLC3A2"],
    "MTOR Signaling":                        ["lysosome_LAMP1", "lysosome_LysoTracker live-cell dye"],
    "Tight Junctions":                       ["Plasma Membrane_Wheat Germ Agglutinin", "plasma membrane_ATP1B3", "plasma membrane_SLC3A2"],
    "Potassium Channels":                    ["Plasma Membrane_Wheat Germ Agglutinin", "plasma membrane_ATP1B3", "plasma membrane_SLC3A2"],
    "Sodium Channels":                       ["Plasma Membrane_Wheat Germ Agglutinin", "plasma membrane_ATP1B3", "plasma membrane_SLC3A2"],

    "Golgi Organization":                    ["cis-Golgi_mStayGold-CENPRaltORF", "trans-Golgi_VAMP3", "ER/Golgi_COPE", "ER/Golgi COP-II_SEC23A", "ER/golgi bridge_VAPA"],
    "Exocytic Vesicles":                     ["trans-Golgi_VAMP3", "clathrin vesicles_CLTA", "plasma membrane_ATP1B3"],
    "Vacuolar Acidification":                ["lysosome_LysoTracker live-cell dye", "lysosome_LAMP1", "endocytic vesicle pH_pHrodo-dextran Live Cell Dye"],
    "Exosomal Secretion":                    ["late endosome_RAB7A", "endosome_VPS35", "clathrin vesicles_CLTA"],
    "Endocytosis":                           ["clathrin vesicles_CLTA", "early endosome_EEA1", "late endosome_RAB7A", "endosome_VPS35", "recycling endosome_TFRC"],
    "ER membrane":                           ["ER_SEC61B", "ER_NCLN", "Endoplasmic Reticulum_Concanavalin A"],
    "ER Tubular Network":                    ["ER_SEC61B", "ER_NCLN", "Endoplasmic Reticulum_Concanavalin A", "microtubules_MAP4", "Microtubules_Tubulin"],
    "ER-Golgi Transport":                    ["ER/Golgi COP-II_SEC23A", "ER/Golgi_COPE", "ER/golgi bridge_VAPA"],
    "Lysosome":                              ["lysosome_LAMP1", "lysosome_LysoTracker live-cell dye"],
    "Peroxisome":                            ["peroxisome_Peroxi_SPY650 live cell dye", "lipid droplet_BODIPY live cell dye", "lipid droplet_PLIN2", "Mitochondria_TOMM20"],

    "snRNA":                                 ["nuclear speckles_SRRM2", "chromatin_H2BC21", "Nucleus_Hoechst"],
    "Transcription Initiation":              ["chromatin_H2BC21", "Nucleus_Hoechst", "nuclear speckles_SRRM2"],
    "DNA-templated Transcription":           ["chromatin_H2BC21", "nuclear speckles_SRRM2", "Nucleus_Hoechst"],
    "DNA Methylation":                       ["chromatin_H2BC21", "Nucleus_Hoechst"],
    "Spliceosome Assembly":                  ["nuclear speckles_SRRM2"],
    "Splicing":                              ["nuclear speckles_SRRM2", "chromatin_H2BC21"],
    "Nuclear Export":                        ["Nucleus_Hoechst", "nucleus_NucleoLIVE Live Cell dye", "chromatin_H2BC21"],

    "Translation Initiation":                ["stress granule_G3BP1", "chaperones_HSPA1B", "ER_SEC61B", "Endoplasmic Reticulum_Concanavalin A"],
    "Ribosomal Large Subunit":               ["ER_SEC61B", "Endoplasmic Reticulum_Concanavalin A", "nucleolus-DFC_FBL", "Nucleoli_NPM1"],
    "Ribosomal Small Subunit":               ["ER_SEC61B", "Endoplasmic Reticulum_Concanavalin A", "nucleolus-DFC_FBL", "Nucleoli_NPM1"],
    "Ribosomal Biogenesis":                  ["nucleolus-DFC_FBL", "Nucleoli_NPM1", "nucleolus-GC_NPM3"],
    "RNA Degradation":                       ["Nucleoli_NPM1", "nucleolus-GC_NPM3", "nucleolus-DFC_FBL", "nuclear speckles_SRRM2"],
    "RNA Polymerase":                        ["nucleolus-DFC_FBL", "Nucleoli_NPM1", "nucleolus-GC_NPM3"],

    "DNA recombination":                     ["chromatin_H2BC21", "Nucleus_Hoechst"],
    "DNA Replication":                       ["chromatin_H2BC21", "Nucleus_Hoechst"],
    "Nuclear Deformation":                   ["laminin_LMNB1", "Nucleus_Hoechst", "nucleus_NucleoLIVE Live Cell dye", "chromatin_H2BC21"],
}


def load_gene_to_fluor_channels():
    df = pd.read_csv(
        FLUOR_CSV, usecols=["gene", "viz_channel", "channel_rank", "rank_type"]
    )
    df = df[df["rank_type"] == "top"]
    df = df.groupby(["gene", "viz_channel"])["channel_rank"].min().reset_index()
    out = {}
    for g, sub in df.groupby("gene"):
        sub = sub.sort_values("channel_rank")
        out[g] = list(zip(sub["channel_rank"].tolist(), sub["viz_channel"].tolist()))
    return out


def load_phase_genes():
    df = pd.read_csv(PHASE_CSV, usecols=["gene", "rank"])
    return set(df[df["rank"] == 1]["gene"].astype(str).unique())


def parse_clusters(text):
    """Returns ordered list of dicts: {name, well_known, genes, line_block}."""
    clusters = []
    current = None
    for raw in text.splitlines():
        s = raw.strip()
        if not s:
            if current and current.get("genes"):
                clusters.append(current)
                current = None
            continue
        if s.endswith(":") and not s.startswith("#"):
            if current and current.get("genes"):
                clusters.append(current)
            current = {"name": s[:-1].strip(), "genes": [], "well_known": None}
        elif s.startswith("#"):
            if current is None:
                continue
            if "well_known:" in s:
                current["well_known"] = s.split(":", 1)[1].strip()
        else:
            if current is not None:
                current["genes"].append(s)
    if current and current.get("genes"):
        clusters.append(current)
    return clusters


def resolve(cluster, gene_to_fluor, phase_genes):
    name = cluster["name"]
    nominee = cluster["well_known"] or WELL_KNOWN_INITIAL.get(name)
    sensible = [s for s in SENSIBLE.get(name, []) if s not in CP_CHANNELS]

    def _try_gene(g):
        avail = gene_to_fluor.get(g, [])
        names = [v for _, v in avail]
        for sense in sensible:
            if sense in names:
                rank = next(r for r, v in avail if v == sense)
                return (g, sense, rank, "fluor")
        return None

    if nominee:
        hit = _try_gene(nominee)
        if hit:
            return hit, "kept" if hit[0] == nominee else ""

    for g in cluster["genes"]:
        if g == nominee:
            continue
        hit = _try_gene(g)
        if hit:
            return hit, f"swapped gene ({nominee} -> {g})"

    if nominee and nominee in phase_genes:
        return (nominee, PHASE_CHANNEL, 1, "phase"), "phase fallback"
    for g in cluster["genes"]:
        if g in phase_genes:
            return (g, PHASE_CHANNEL, 1, "phase"), f"phase fallback (gene {nominee} -> {g})"
    return None, "UNRESOLVED"


def render(clusters, picks):
    """Rebuild the txt file. Cluster headers + gene lists are kept verbatim;
    metadata block is regenerated."""
    text = TXT.read_text().splitlines()
    out = []
    i = 0
    pick_iter = iter(picks)
    while i < len(text):
        line = text[i]
        s = line.strip()
        is_header = s.endswith(":") and not s.startswith("#")
        if is_header:
            cluster_name = s[:-1].strip()
            out.append(line)
            i += 1
            super_line = None
            while i < len(text) and text[i].lstrip().startswith("#"):
                ls = text[i].strip()
                if "supercluster:" in ls:
                    super_line = text[i]
                i += 1
            cluster, pick, note = next(pick_iter)
            assert cluster["name"] == cluster_name, (cluster["name"], cluster_name)
            if super_line:
                out.append(super_line)
            out.append(f"# well_known: {cluster['well_known'] or WELL_KNOWN_INITIAL.get(cluster_name, '')}")
            if pick:
                g, c, r, mod = pick
                out.append(f"# rep_gene: {g}")
                out.append(f"# rep_channel: {c}")
                out.append(f"# ch_rank: {r}")
                out.append(f"# rep_modality: {mod}")
            else:
                out.append(f"# rep_gene: UNRESOLVED")
            continue
        out.append(line)
        i += 1
    TXT.write_text("\n".join(out) + "\n")


def main():
    print(f"Reading {TXT}")
    clusters = parse_clusters(TXT.read_text())
    print(f"  {len(clusters)} clusters")

    print(f"Loading {FLUOR_CSV.name}...")
    gene_to_fluor = load_gene_to_fluor_channels()
    print(f"  {len(gene_to_fluor):,} genes with fluor attention")

    print(f"Loading {PHASE_CSV.name}...")
    phase_genes = load_phase_genes()
    print(f"  {len(phase_genes):,} genes with phase attention")

    print()
    print(f"{'CLUSTER':<42} {'GENE':<10} {'CH#':<4} {'CHANNEL':<48} MOD     NOTE")
    print("-" * 150)

    picks = []
    n_phase = 0
    n_unresolved = 0
    for c in clusters:
        pick, note = resolve(c, gene_to_fluor, phase_genes)
        picks.append((c, pick, note))
        if pick is None:
            print(f"!! {c['name']:<42} UNRESOLVED")
            n_unresolved += 1
            continue
        g, ch, r, mod = pick
        if mod == "phase":
            n_phase += 1
        print(f"{c['name']:<42} {g:<10} {r:<4} {ch:<48} {mod:<7} {note}")

    print()
    print(f"Resolved: {len(picks) - n_unresolved}/{len(picks)} "
          f"(fluor: {len(picks) - n_unresolved - n_phase}, phase fallback: {n_phase}, "
          f"unresolved: {n_unresolved})")

    render(clusters, picks)
    print(f"\nWrote {TXT}")


if __name__ == "__main__":
    main()
