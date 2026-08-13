"""DiffEx Stage 3 — contrastive direction discovery + classifier ranking + traversal.

K direction MLPs (InfoNCE + decorrelation, unsupervised) on CellDINO embeddings →
rank by control-vs-target classifier score shift → DDIM-traverse the selected
direction and verify with re-encoded scores. See ../PLAN.md §3.
"""
