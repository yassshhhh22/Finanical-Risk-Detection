# Stage 4: Chronological Train / Validation / Test Splits

## Purpose

Splitting data randomly would give falsely optimistic results.
Credit data is temporal: a model trained on future data predicting the past is
useless in production, where the model always sees the past and must predict the future.

This stage creates **chronological** splits that faithfully simulate the deployment
scenario: train on old snapshots, validate and test on newer ones.

---

## Why NOT Random Splitting?

Consider a borrower with 36 monthly snapshots spanning 3 years.
A random split would put some of their snapshots in train and others in test.
The model would see the borrower's behaviour (e.g. payment pattern, income type)
during training and then "predict" for a snapshot from the same borrower.

This creates **data leakage**: the model learns borrower-specific patterns that
it would never have access to in real production (where a borrower's future
behaviour is, by definition, unknown at prediction time).

A chronological split ensures all training snapshots come before all validation
and test snapshots. A borrower who first appears after the training cutoff date
is entirely unseen during training — exactly as in production.

---

## Split Boundaries

Snapshots are sorted by `snapshot_day` (days relative to loan start, converted
to a global calendar date using `DAYS_DECISION`).

| Split | Portion | Approximate period |
|---|---|---|
| Train | 60% | Earliest snapshots |
| Validation | 20% | Middle period |
| Test | 20% | Most recent snapshots |

The exact cutoff dates are recorded in `stage4_snapshot_day_split_map.csv`
so the split is fully reproducible.

---

## Why 60 / 20 / 20?

**Training set (60%):**
Must be large enough for SMOTE/class_weight to find enough minority-class examples
and for the Random Forest to grow 300 diverse trees.
With ~6M total snapshots, 60% gives ~3.6M training rows — sufficient even for
the severe 0.78% imbalance of `future_dpd30` (≈28,000 positive training examples).

**Validation set (20%):**
Used for LightGBM early stopping and threshold selection in Stage 6.
Must be large enough to give stable threshold estimates — ~570,000 rows is more
than sufficient.

**Test set (20%):**
The final held-out evaluation. No model or hyperparameter decision is made
based on test set results until Stage 5 evaluation.
The test set is strictly future data relative to training.

---

## Class Imbalance in Each Split

| Target | Split | Positive rate |
|---|---|---|
| missed_upcoming_emi | Train | 4.91% |
| missed_upcoming_emi | Validation | 3.71% |
| missed_upcoming_emi | Test | 3.20% |
| future_dpd30 | Train | 0.78% |
| future_dpd30 | Validation | 0.54% |
| future_dpd30 | Test | 0.29% |

The positive rate drops in later splits because the most recent borrowers tend to
have shorter loan histories, meaning fewer realised defaults have been recorded yet
(right-censoring). This is expected and realistic.

---

## Column Manifest

`stage4_column_manifest.json` records:

```json
{
  "numeric_feature_columns": ["AMT_CREDIT", "hist_late_count", ...],
  "categorical_feature_columns": ["CODE_GENDER", "NAME_INCOME_TYPE", ...],
  "passthrough_columns": ["SK_ID_CURR", "snapshot_day", "label", "split"]
}
```

This manifest is read by Stages 5, 7, 8, and 9 to identify which columns are
features and which are metadata. It is the single source of truth for the
feature schema.

---

## Outputs

```
artifacts/processed/
    model_features_missed_upcoming_emi_train.parquet       (~134 MB)
    model_features_missed_upcoming_emi_validation.parquet
    model_features_missed_upcoming_emi_test.parquet
    model_features_future_dpd30_train.parquet              (~133 MB)
    model_features_future_dpd30_validation.parquet
    model_features_future_dpd30_test.parquet

artifacts/raw_checks/
    stage4_snapshot_day_split_map.csv
    stage4_split_summary.csv
    stage4_column_manifest.json
    stage4_manifest.json
```

---

## Key Design Decisions

**Why not stratified time-series cross-validation?**
K-fold time-series CV would give better variance estimates of model performance
but is impractical at 6M rows with 2 models × 2 targets.
A single chronological split is standard practice at this scale.

**Why not use `GroupKFold` on `SK_ID_CURR`?**
Group-aware splitting would ensure each borrower appears in exactly one fold.
However, because our snapshots already have a temporal structure, and because
the same borrower appears in consecutive snapshots, chronological splitting
already prevents borrower-level leakage.

---

## Connects to Stage 5

Stage 5 loads `*_train.parquet`, fits the preprocessing pipeline and models,
and evaluates on `*_validation.parquet` (for LightGBM early stopping) and
`*_test.parquet` (for final metrics).
