# Stage 5: Centralized Baseline Modeling

## Purpose

Train the two model families (Random Forest and LightGBM) on the full training set.
These centralized models serve two purposes:
1. **Baseline performance** — the best possible accuracy when all data is pooled
2. **Federated comparison target** — Stage 7 measures how much federated learning
   loses relative to this ceiling

Every design decision here (hyperparameters, imbalance strategy, preprocessing)
is documented so Stage 8 can assess whether the choices introduce bias.

---

## Preprocessing Pipeline

Fit **only on the training set**, then applied identically to validation and test.
Fitting on train-only prevents data leakage from test statistics.

```
Numeric features (48):
    SimpleImputer(strategy="median")   → fill NaN with training median

Categorical features (6):
    SimpleImputer(strategy="most_frequent")
    → OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
```

Why `OrdinalEncoder` over `OneHotEncoder`?
- Both RF and LightGBM can handle ordinal integer codes natively
- One-hot encoding of `OCCUPATION_TYPE` (18 categories) would add 18 sparse columns
- Ordinal encoding keeps the feature matrix dense and faster to train on

Why `unknown_value=-1`?
- At inference time, a new category might appear (e.g. a new income type)
- `-1` signals "unknown" to the model rather than crashing with an error
- Both RF and LightGBM split on `>= threshold`, so `-1` lands in a distinct
  low-value branch that the model learns to handle

The fitted preprocessor is saved as `preprocessor_{target}.pkl` so Stage 6, 7,
and 8 apply exactly the same transformation.

---

## Class Imbalance Strategy

### Why imbalance is a problem

`future_dpd30` has 0.78% positives — roughly 1 in 128 rows.
A naive model that predicts "never default" for every row achieves 99.22% accuracy
but zero recall. For credit risk, missing actual defaulters (false negatives) is
precisely what the model must avoid.

### Strategy chosen: class_weight / scale_pos_weight

**Random Forest:** `class_weight="balanced"`
- Scikit-learn re-weights each sample inversely proportional to class frequency
- Minority class (positives) get weight `n_total / (2 × n_positive)`
- Equivalent to duplicating minority examples during training
- For `future_dpd30`: positives weighted ~64× more than negatives

**LightGBM:** `scale_pos_weight = n_negative / n_positive`
- For `future_dpd30`: `scale_pos_weight ≈ 127.4`
- Scales the gradient of positive examples up, forcing the boosting algorithm
  to focus more strongly on getting positives right

### Why not SMOTE?

SMOTE (Synthetic Minority Oversampling Technique) generates synthetic minority
examples by interpolating between real ones. On 6M rows, SMOTE takes 20–40
minutes and increases training data size by 15%. The performance gain over
`class_weight="balanced"` is marginal for tree ensembles, which already handle
imbalance well via weighted splits. We keep SMOTE as a `--fast-oversample` option
for quick tests but do not use it for the full run.

---

## Model 1: Random Forest

```python
RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    min_samples_leaf=100,
    max_features="sqrt",
    class_weight="balanced",
    n_jobs=-1,
    random_state=42,
)
```

**Why 300 trees?**
Variance in RF drops roughly as 1/T (T = number of trees). Beyond 300 trees
on a 6M-row dataset, additional trees give diminishing returns (< 0.001 ROC-AUC
improvement) while adding significant memory and prediction latency.

**Why `max_depth=15`?**
Unlimited depth would allow trees to memorize training rows.
At depth 15, each tree can represent up to 32,768 leaf partitions — expressive
enough to capture non-linear interactions but not so deep it overfits.

**Why `min_samples_leaf=100`?**
On a 6M-row dataset, leaf nodes with 1–10 samples are pure noise.
Requiring at least 100 samples per leaf prevents trees from chasing tiny
clusters and improves generalisation.

**Why `max_features="sqrt"`?**
At each split, only `sqrt(54) ≈ 7` features are considered. This decorrelates
the trees (each tree sees a different random subset of features), which is
the source of RF's variance reduction.

**Intel sklearnex acceleration:**
`sklearnex.patch_sklearn()` replaces the scikit-learn RF implementation
with Intel's oneDAL backend. On Intel CPUs (including Iris Xe), this gives
2–5× speedup on large datasets by using AVX-512 SIMD and parallel tree construction.

---

## Model 2: LightGBM (Gradient Boosting)

```python
LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    num_leaves=63,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=n_neg / n_pos,
    metric="average_precision",
    n_jobs=-1,
    random_state=42,
)
```

With early stopping monitored on validation set PR-AUC (`first_metric_only=True`).

**Why gradient boosting alongside RF?**
RF and GB have different inductive biases:
- RF builds trees in parallel, averaging their predictions — high variance, low bias
- GB builds trees sequentially, each correcting the previous — low variance, higher bias
Having both lets us compare and ensemble them in Stage 7.

**Why LightGBM over XGBoost?**
LightGBM uses histogram-based splitting (binning continuous features into 255 buckets)
which makes it significantly faster than XGBoost's exact split search on large datasets.
At 6M rows with 54 features, LightGBM trains 10× faster than XGBoost in our tests.

**Why `learning_rate=0.05` with 1000 estimators?**
A low learning rate with many trees is the standard GB setup.
Early stopping prevents overfitting: if validation PR-AUC does not improve for
50 consecutive rounds, training stops and the best iteration is used.

**Why `first_metric_only=True` for early stopping?**
LightGBM evaluates multiple metrics (PR-AUC, log-loss) on the validation set.
Without `first_metric_only=True`, early stopping monitors the *last* metric in the
list (log-loss), which increases monotonically when `scale_pos_weight` is high
because the model becomes overconfident. This caused best_iteration=1 (stopping
after round 1). `first_metric_only=True` ensures early stopping only watches PR-AUC.

**Why PR-AUC (`average_precision`) as the eval metric?**
ROC-AUC is optimistic on imbalanced datasets — a model that scores all negatives
slightly below 0.5 can still achieve ROC-AUC > 0.85. PR-AUC penalises false
positives more heavily and is a better discriminator when positives are rare.

---

## Results

### Test Set (at threshold 0.5)

| Target | Model | ROC-AUC | PR-AUC | F1 |
|---|---|---|---|---|
| missed_upcoming_emi | RF | 0.7801 | 0.1390 | 0.1994 |
| missed_upcoming_emi | GB | 0.7789 | 0.1406 | 0.2006 |
| future_dpd30 | RF | 0.8217 | 0.2059 | 0.3070 |
| future_dpd30 | GB | 0.7701 | 0.1188 | 0.2037 |

### Why is F1 so low?

F1 = 0.20 sounds alarming but is normal for severely imbalanced problems.
At threshold 0.5 with 0.78% positives, even a well-calibrated model predicts
very few positives (most scores fall below 0.5 when the prior is low).
Stage 6 finds the optimal threshold (often > 0.7) that maximises F1,
giving much higher values. The ROC-AUC of 0.77–0.82 correctly reflects the
model's ability to rank-order risk.

---

## Checkpointing

If training is interrupted (`Ctrl+C`), completed models are already saved to disk.
On re-run, Stage 5 detects existing `rf_{target}.pkl` / `gb_{target}.pkl` files
and skips those models. Use `--retrain` to force retraining.

---

## Outputs

```
artifacts/models/
    rf_missed_upcoming_emi.pkl           (252 MB)
    gb_missed_upcoming_emi.pkl           (1.2 MB)
    preprocessor_missed_upcoming_emi.pkl
    rf_future_dpd30.pkl                  (133 MB)
    gb_future_dpd30.pkl                  (1.6 MB)
    preprocessor_future_dpd30.pkl

artifacts/raw_checks/
    stage5_metrics.csv
    stage5_confusion_matrices.csv
    stage5_threshold_summary.csv
    stage5_notes.md
    stage5_manifest.json
```

---

## Connects to Stage 6

Stage 6 loads the saved models and runs detailed error analysis.
The preprocessors must be loaded too — predictions require running the same
transformation on raw features.
