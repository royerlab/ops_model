"""Set-classifier interpretability subpackage.

Train / evaluate / explain a permutation-invariant *set classifier* that predicts a
gene knockout (or a protein-complex / pathway label) from a *set* of single-cell
embeddings. Runs standalone from an embedding parquet (+ an optional gene->label
metadata CSV); see the package README for usage.

Requires the optional dependencies: ``pip install ops_model[classifier]``.
"""
