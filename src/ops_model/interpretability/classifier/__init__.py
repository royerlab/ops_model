"""Set-classifier interpretability subpackage.

Train / evaluate / explain a permutation-invariant *set classifier* that predicts a
gene knockout (or a protein-complex / pathway label) from a *set* of single-cell
embeddings. Runs standalone from an embedding parquet (+ an optional gene->label
metadata CSV). Entry points: ``train.py`` (hydra ``--config-name``, see ``configs/``),
``eval.py`` (accuracy vs cells-per-set), and ``score.py`` (per-cell scores, argparse).

Requires the optional dependencies: ``pip install ops_model[classifier]``.
"""
