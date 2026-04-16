"""
Eval Metrics comparing the results of different embedding models

Models:
 - dinov3
    - combination config: /hpc/projects/icd.fast.ops/experiments/paper/model_comparison/dino/all_experiments.yml
 - cell-profiler
 - cell_dino
 - subcell
 - dynaclr
 - katamari

Outputs:
 - Bar charts comparing different models on different metrics
 - scatter plots showing the mAP scores of different GKOs from different models 
    - i.e. mAP activity dino vs mAP activity cell-profiler
"""

#%%
# Load all the results
dinov3_anndata_path = ''
cell_profiler_anndata_path = ''
cell_dino_anndata_path = ''
subcell_anndata_path = ''
dynaclr_anndata_path = ''
katamari_anndata_path = ''

