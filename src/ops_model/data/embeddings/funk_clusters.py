import pandas as pd
from pathlib import Path
import anndata as ad
import matplotlib.pyplot as plt
from ops_model.data.embeddings.utils import load_adata


# Define Funk clusters
path = Path(
    "/hpc/projects/intracellular_dashboard/ops/configs/annotated_gene_panel_July2025.csv"
)
df = pd.read_csv(path)
cluster_numbers = [
    26,
    148,
    14,
    106,
    138,
    29,
    184,
    201,
    13,
    199,
    200,
    52,
    155,
    82,
    3,
    46,
    66,
    212,
    136,
    110,
    214,
    21,
    15,
    23,
    203,
    213,
    216,
    179,
]
funk_clusters = {
    "26": {
        "genes": df[df["funk_cluster"] == "26"]["Gene.name"].tolist(),
        "desc": "26 DNA Replication",
    },
    "148": {
        "genes": df[df["funk_cluster"] == "148"]["Gene.name"].tolist(),
        "desc": "148 Cell Cycle and Cytokinesis",
    },
    "14": {
        "genes": df[df["funk_cluster"] == "14"]["Gene.name"].tolist(),
        "desc": "14 Translation Initiation ",
        #  'desc': '14 Translation Initiation and tRNA Ligases'
    },
    "106": {
        "genes": df[df["funk_cluster"] == "106"]["Gene.name"].tolist(),
        "desc": "106 Proteasome 19S Regulatory Particle",
        # 'desc': '106 Proteasome 19S Regulatory Particle ATPase Subunits & Ubiquitination Factors',
    },
    "138": {
        "genes": df[df["funk_cluster"] == "138"]["Gene.name"].tolist(),
        "desc": "138 Spliceosome",
    },
    "29": {
        "genes": df[df["funk_cluster"] == "29"]["Gene.name"].tolist(),
        "desc": "29 Adhesion intracellular transport & NSL complex",
    },
    "184": {
        "genes": df[df["funk_cluster"] == "184"]["Gene.name"].tolist(),
        "desc": "184 Actin Cytoskeletion & nuclear transport",
    },
    "201": {
        "genes": df[df["funk_cluster"] == "201"]["Gene.name"].tolist(),
        "desc": "201 Golgi-ER transport",
    },
    "13": {
        "genes": df[df["funk_cluster"] == "13"]["Gene.name"].tolist(),
        "desc": "13 DNA damage",
    },
    "199": {
        "genes": df[df["funk_cluster"] == "199"]["Gene.name"].tolist(),
        "desc": "199 RNA Polymerase II",
    },
    "200": {
        "genes": df[df["funk_cluster"] == "200"]["Gene.name"].tolist(),
        "desc": "200 COP9 signalosome",
    },
    "52": {
        "genes": df[df["funk_cluster"] == "52"]["Gene.name"].tolist(),
        "desc": "52 Spliceosome",
    },
    "155": {
        "genes": df[df["funk_cluster"] == "155"]["Gene.name"].tolist(),
        "desc": "155 RNA Polymerase I",
    },
    # '82': {
    #     'genes': df[df['funk_cluster'] == '82']['Gene.name'].tolist(),
    #     'desc': '82 Mitochondrial Ribosome',
    # },
    "3": {
        "genes": df[df["funk_cluster"] == "3"]["Gene.name"].tolist(),
        "desc": "3 DNA Damage",
    },
    "46": {
        "genes": df[df["funk_cluster"] == "46"]["Gene.name"].tolist(),
        "desc": "46 Cell Cycle",
    },
    "66": {
        "genes": df[df["funk_cluster"] == "66"]["Gene.name"].tolist(),
        "desc": "66 Ribosome 40S subunit",
    },
    "212": {
        "genes": df[df["funk_cluster"] == "212"]["Gene.name"].tolist(),
        "desc": "212 Chaperonin TCP-1",
    },
    "136": {
        "genes": df[df["funk_cluster"] == "136"]["Gene.name"].tolist(),
        "desc": "136 40S Ribosome Biogenesis",
    },
    "110": {
        "genes": df[df["funk_cluster"] == "110"]["Gene.name"].tolist(),
        "desc": "110 Spliceosome",
    },
    "214": {
        "genes": df[df["funk_cluster"] == "214"]["Gene.name"].tolist(),
        "desc": "214 Augmin complex",
    },
    "21": {
        "genes": df[df["funk_cluster"] == "21"]["Gene.name"].tolist(),
        "desc": "21 Ribosome Biogenesis",
    },
    "15": {
        "genes": df[df["funk_cluster"] == "15"]["Gene.name"].tolist(),
        "desc": "15 60S Ribosome Biogenesis",
    },
    "23": {
        "genes": df[df["funk_cluster"] == "23"]["Gene.name"].tolist(),
        "desc": "23 Ribosome 60S subunit",
    },
    "203": {
        "genes": df[df["funk_cluster"] == "203"]["Gene.name"].tolist(),
        "desc": "203 Ribosome Biogenesis",
    },
    "216": {
        "genes": df[df["funk_cluster"] == "216"]["Gene.name"].tolist(),
        "desc": "216 Ribosome Biogenesis",
    },
    "179": {
        "genes": df[df["funk_cluster"] == "179"]["Gene.name"].tolist(),
        "desc": "179 DNA Damage",
    },
}


def print_cluster_info(funk_clusters):
    for k, v in funk_clusters.items():
        print(f"Cluster {k} ({v['desc']}): {len(v['genes'])} genes")

    return


def plot_funk_clusters(adata, funk_clusters, save_path=None):
    fig, axs = plt.subplots(nrows=7, ncols=4, figsize=(20, 28))
    axs = axs.flatten()
    for i, (cluster_num, cluster_info) in enumerate(funk_clusters.items()):
        genes = cluster_info["genes"]
        desc = cluster_info["desc"]
        plt.sca(axs[i])
        umap = adata.obsm["X_umap"]
        plt.scatter(umap[:, 0], umap[:, 1], c="lightgrey", s=20, alpha=0.8, linewidth=0)
        for gene in genes:
            subset = adata[adata.obs["label_str"] == gene].obsm["X_umap"]
            plt.scatter(
                subset[:, 0], subset[:, 1], s=40, alpha=1, linewidth=0, label=gene
            )
        plt.title(desc)
        plt.xticks([])
        plt.yticks([])
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize="10")
        plt.tight_layout()
    if save_path is not None:
        print(f"Saving funk cluster UMAP plot to {save_path}")
        plt.savefig(save_path, dpi=300)

    return


if __name__ == "__main__":
    adata_path = "/hpc/projects/intracellular_dashboard/ops/ops0031_20250424/3-assembly/dynaclr_features"
    save_path = adata_path + "/report_plots/funk_clusters_umap.png"
    adata_genes = load_adata(adata_path)
    plot_funk_clusters(
        adata_genes,
        funk_clusters,
        save_path=adata_path + "/report_plots/funk_clusters_genes.png",
    )


"""
funk_clusters_manual = {
    '112': {
        'included': ['NOLC1','SPHK2','TMA16', 'ZNF574'],
        'missing': ['ABCF1', 'BRIX1', 'EBNA1BP2', 'MRTO4', 'NOC3L', 'PAK1IP1', 'PELO', 'SPATA5', 'TEX10', 'WDR55',
                    'BOP1', 'DDX51', 'ISG20L2', 'NOC2L','NOL12','NVL','PPAN-P2RY11','RSL1D1' ],
    },
    '21': {
        'included': ['AATF','RRP7A','UTP14A','DDX18','DYRK1A','EIF3M','RIOK2','BMS1','DDX47'
                    'EIF4A1','MOB4','NOL10','RAE1','ZCCHC9','DDX52', 'TSR1'],
        'missing': ['C1orf131','DIMT1','EIF3F','HK2','NGDN','PCDC11','RBIS','ABT1','INO80',
                    'NOC4L','PPRC1','SRFBP1','UTP20','EIF3D','RRP12','TRMT112','BYSL',
                    'EIF3E','ESF1','NEPRO','NOP14', 'RAN','RRP36',],
    },
    '203': {
        'included': ['NOL7', 'TBL3','WDR36',],
        'missing': ['DCAF13','DNTTIP2','MPHOSPH10','RRP9', 'UTP4','UTP18','WDR3','WDR46'],
    },
    '216': {
        'included': ['NOP56','NOP58',],
        'missing': ['DDX21','FBL','IMP3','PWP2','UTP6','WDR43'],
    },
    '136': {
        'included': ['IMP4','NOB1','RPS27A','LTV1','NOL6','RCL1','RPS10','RPS25'],
        'missing': ['BUD23','DHX37','EIF3G','PNO1','RPS3','RPS21','RPS26','UTP11','DDX10'
                    'EIF3CL','FCF1','RPS27','UTP3'],
    },
    '66': {
        'included': ['EIF3A','EIF3B','FAU','HSPA5','RPS14','RPS16','RPS28','RPS29',
                     'RPS3A','RPS4X','RPS5','RPS6','RPS8','RPS9','RPSA'],
        'missing': ['EIF3I','RPS11','RPS12','RPS13','RPS15','RPS15A','RPS17',
                    'RPS18','RPS19','RPS2','RPS20','RPS23','RPS24','RPS7'],
    },
    '15': {
        'included': ['EIF6','FTSJ3','GNL2','GNL3','GNL3L','GTPBP4','MDN1','NCL','NLE1',
                     'NOL8','NOP16','NSA2','PPAN','RPF2','RRS1','RSL24D1','URB2','WDR12'],
        'missing': ['CEBPZ','DDX24','DDX27','DDX54','DDX56','LSG1','MAK16','METAP2',
                    'NIFK','NIP7','NMD3','NOP2','NOP53','PES1','RPF1','RPL28','RPL36',
                    'RRP1','RRP15','SDAD1','SPOUT1','URB1','WDR74'],
    },
    '23': {
        'included': ['RPL5','RPL9','RPL15','RPL18','RPL23','RPL26','RPL27A','RPL30'
                     'RPL32','RPL34','RPL35','RPL37A','UBA52'],
        'missing': ['AAMP','RPL3','RPL4','RPL6','RPL7','RPL7A','RPL7L1', 'RPL8','RPL10',
                    'RPL10A','RPL11','RPL13','RPL13A','RPL14','RPL17','RPL18A','RPL19'
                    'RPL24','RPL27','RPL31','RPL35A','RPL37','RPL38','RPL39','RPLP0',
                    'RPL17-C18orf32'],
    },
    '14': {
        'included': ['CTU1','CTU2','DPH3','EIF2B2','EIF2B3','EIF2B4','EIF2B5','EIF2S1',
                     'EIF2S2','FARSB','GBF1','GRPEL1','HSPA9','PHB','PHB2','PPP1R15B'
                     'PRELID3B','TIMM44','TIMM23','URM1',],
        'missing': ['CCDC51','CNOT9','EEF1A1','EIF1AX','EIF2S3','EIF4E','ETF1','HARS1',
                    'IARS1','KARS1','LARS1','MAT2A','NARS1','PRELID1','QARS1','RPL12',
                    'SARS1','SLC20A1','VARS1','YARS1','ZPR1'],
    },
}
"""
