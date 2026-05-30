# DynaCLR Training Recipes

## Overview

DynaCLR learns single-cell representations via contrastive learning. The training behavior is controlled by three independent axes:

1. **Positive source** — what defines a positive pair
2. **Batch sampling** — how batches are composed
3. **Balanced sampling** — what gets equalized within a batch

These combine into different recipes depending on what invariance the model should learn.

---

## Positive Source (`positive_source`)

| Value | Positive pair definition | What model learns |
|---|---|---|
| `self` | Same cell, copied then independently augmented | Augmentation invariance only |
| `perturbation` | Different cell, same `gene_name` (any reporter) | Cross-reporter gene-level similarity |
| `perturbation_n_reporter` | Different cell, same `gene_name` AND same `reporter` | Within-reporter gene KO phenotype |

---

## Batch Sampling

| Mode | Config | Behavior |
|---|---|---|
| **Random** | `balanced_sampling: False`, `grouped_sampling: False` | Uniform random (default PyTorch) |
| **Balanced** | `balanced_sampling: True`, `balance_col: "reporter"` | Reweight so each reporter is equally likely per sample. Batch still mixes reporters |
| **Grouped** | `grouped_sampling: True`, `group_col: "reporter"` | Entire batch drawn from one reporter. Reporter changes between batches |

---

## Recipes

### Recipe 1: Augmentation Invariance (baseline)
```yaml
contrastive_kwargs:
  positive_source: "self"
balanced_sampling: True
balance_col: "reporter"
grouped_sampling: False
```
- **Task**: "same cell under different augmentations should map nearby"
- **Learns**: Low-level visual features invariant to rotation, flip, blur, noise
- **Risk**: Converges to theoretical minimum very fast with pretrained backbone — model solves the task without learning biology
- **Use case**: Sanity check, pretraining baseline

### Recipe 2: Gene KO Phenotype per Organelle (recommended)
```yaml
contrastive_kwargs:
  positive_source: "perturbation_n_reporter"
grouped_sampling: True
grouped_sampling_val: True
group_col: "reporter"
balance_col: "gene_name"
```
- **Task**: "two cells with the same gene KO, imaged with the same reporter, should map nearby"
- **Learns**: Gene KO-induced morphological changes specific to each organelle
- **Batch composition**: All cells in a batch share the same reporter → negatives differ only by gene KO → model cannot use reporter identity as a shortcut
- **Balancing**: `gene_name` is balanced within each reporter group so rare KOs are seen equally
- **Use case**: Primary training recipe for phenotype discovery

### Recipe 3: Cross-Reporter Gene Similarity
```yaml
contrastive_kwargs:
  positive_source: "perturbation"
balanced_sampling: True
balance_col: "gene_name"
grouped_sampling: False
```
- **Task**: "two cells with the same gene KO (any reporter) should map nearby"
- **Learns**: Gene-level morphological signature that generalizes across organelle markers
- **Risk**: Very hard task — COPB1+SEC61B and COPB1+LAMP1 look completely different at the pixel level. Model may need to rely on gross cell morphology (shape, size) since organelle-specific features are not comparable across reporters
- **Use case**: Only if you want a single "gene KO embedding" that ignores reporter. Unlikely to capture organelle-specific phenotypes

### Recipe 4: Augmentation Invariance with Grouped Batches
```yaml
contrastive_kwargs:
  positive_source: "self"
grouped_sampling: True
group_col: "reporter"
balance_col: "gene_name"
```
- **Task**: Same as Recipe 1, but batches are homogeneous by reporter
- **Learns**: Augmentation invariance, but negatives are within-reporter → loss landscape differs from Recipe 1
- **Use case**: Debugging / comparison with Recipe 2 to isolate the effect of positive source vs batch composition

---

## Config Reference

### `data_manager` section
```yaml
data_manager:
  # --- Positive pairing ---
  contrastive_kwargs:
    positive_source: "perturbation_n_reporter"  # or "self", "perturbation"
    use_negative: False
    cell_masks: False
    transforms: [...]

  # --- Batch composition ---
  grouped_sampling: True        # All samples in a batch share group_col
  grouped_sampling_val: True    # Same for validation
  group_col: "reporter"         # Column to group batches by

  # --- Balancing (within group if grouped, otherwise global) ---
  balanced_sampling: True       # Only used when grouped_sampling=False
  balanced_sampling_val: False
  balance_col: "gene_name"      # Column to balance by
```

### Interaction between `balanced_sampling` and `grouped_sampling`

- `grouped_sampling: True` takes precedence — it uses `GroupedBatchSampler` which internally balances by `balance_col` within each group
- `balanced_sampling` is only used when `grouped_sampling: False` — it creates a `WeightedRandomSampler` that reweights globally by `balance_col`
- Both can be False for plain random sampling

---

## Dataset Statistics

### Smaller dataset (`labels_testset_filtered.csv`)
- 573,932 cells | 12 genes | 40 reporters | 480 (gene, reporter) combos
- Min cells per combo: 113 | Median: 528

### Larger dataset (`labels_testset_10complex_n_NTC_v2_filtered.parquet`)
- 7,033,743 cells | 33 genes (incl. NTC) | 40 reporters | 1,320 combos
- Min cells per combo: 55 | Median: 573

---

## Theoretical Loss Bounds (NT-Xent)

For batch size `B` and temperature `T`:
- Random (uniform similarity): `log(2B - 1)`
- Perfect separation (pos_sim=1, neg_sim=0): `log(2B - 1) - 1/T`

| Batch size | Temp | Random loss | Perfect separation |
|---|---|---|---|
| 256 | 0.5 | 6.24 | 4.24 |
| 256 | 0.3 | 6.24 | 2.90 |
| 512 | 0.5 | 6.93 | 4.93 |

If train loss reaches the "perfect separation" floor quickly, the task is too easy — switch to a harder positive source or stronger augmentations.

---

## Verification

Use `experiments/models/dynaclr/test/inspect_grouped_sampling.py` to verify:
- All samples in a batch share the same reporter
- Anchor-positive pairs share gene and reporter
- Gene distribution is balanced within each batch
- Visual inspection of anchor vs positive pairs
