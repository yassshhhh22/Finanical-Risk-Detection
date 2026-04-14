# Stage 6: Error Analysis and Diagnostics

## Purpose

A trained model is not the end of the story. At default threshold (0.5), both
LightGBM models produced F1 = 0 on `future_dpd30` — they predicted zero positives.
This stage diagnoses why and fixes it through optimal threshold selection.

Beyond threshold fixing, Stage 6 produces the rich diagnostic outputs that
Stage 8 uses for the ethical evaluation: per-group performance, calibration,
and false positive/negative breakdown.

---

## Problem: Why Did LightGBM Predict Zero Positives?

With `scale_pos_weight = 127.4` for `future_dpd30`, LightGBM learns that
positives are rare but important. It shifts predicted probabilities upward
for suspected positives — but because the actual prevalence is 0.78%,
even a well-calibrated model will have most predictions below 0.5.

At threshold 0.5: nearly all scores fall below → zero predicted positives → F1 = 0.

**This is not the model failing. It is the threshold being wrong for this data.**

The correct response is not to retrain but to find the threshold that
maximises the desired metric on a held-out set.

---

## Optimal Threshold Selection

### Method: Precision-Recall Curve

For each model × target, compute `precision_recall_curve(y_true, y_prob)` on
the validation set. This returns precision and recall at every possible threshold.

The optimal threshold is where **F1 = 2 × precision × recall / (precision + recall)**
is maximised. This balances false positives (lower precision) against false negatives
(lower recall) equally.

### Results

| Target | Model | Optimal Threshold |
|---|---|---|
| missed_upcoming_emi | RF | 0.6872 |
| missed_upcoming_emi | GB | 0.7311 |
| future_dpd30 | RF | 0.9180 |
| future_dpd30 | GB | 0.9592 |

**Why are the thresholds so high (0.73–0.96)?**
Because the models correctly assign low probabilities to most rows.
With 0.78% positives, the model should only predict positive when it is highly
confident — requiring a score above 0.92 means "I am very sure this borrower
will default." This is sensible behaviour, not a bug.

### Why Use Validation Set for Threshold?

Selecting the threshold on the test set would overfit it to test data —
the threshold would be too precise for new data. The validation set is held
out from model training, so threshold selection on it is unbiased.

---

## False Positive / False Negative Analysis

At the optimal threshold, Stage 6 characterises the errors:

**False Positives (FP):**
- Creditworthy borrowers predicted to default
- In production: loan denied or restricted → financial exclusion harm
- Predominantly: borrowers with inconsistent payment histories
  (e.g. missed one payment early in the loan, then perfect record)

**False Negatives (FN):**
- Defaulting borrowers predicted to be safe
- In production: loan extended to someone who will not repay
- Predominantly: borrowers with strong historical records but sudden income shock
  (these are inherently hard to predict from historical behaviour alone)

---

## Slice-Based Performance

Performance is computed separately for each demographic subgroup using the
columns available in the test set predictions:

- `CODE_GENDER` (F / M)
- `NAME_INCOME_TYPE` (Working, Pensioner, Commercial associate, State servant, ...)
- `NAME_EDUCATION_TYPE` (Secondary, Higher education, Incomplete higher, ...)
- `NAME_FAMILY_STATUS` (Married, Single, Separated, Widow, Civil marriage)
- `NAME_HOUSING_TYPE` (House/apartment, With parents, Rented apartment, ...)
- `OCCUPATION_TYPE` (30+ categories)

For each group: ROC-AUC, PR-AUC, F1, precision, recall are reported.

### Notable findings

| Group | Observation |
|---|---|
| Office apartment (housing) | ROC-AUC = 0.32 for GB on `future_dpd30` — near-random |
| Secretaries (occupation) | ROC-AUC = 0.60 for GB — well below average |
| Academic degree (education) | Lower positive prediction rate; DI ≈ 0.70 |
| Pensioners (income type) | ROC-AUC = 0.79, higher than average |

The Office apartment and Secretaries findings are likely due to small group sizes
(19,362 and 10,305 rows respectively) combined with very low positive rates in
those groups (< 0.1%), making reliable estimation impossible. These groups are
flagged for monitoring rather than treated as evidence of model failure.

---

## Calibration Analysis

Calibration measures whether predicted probabilities correspond to actual frequencies.
A model that predicts 0.8 for a borrower should be correct 80% of the time.

### Reliability curve
The test set is binned into 10 probability buckets (0–0.1, 0.1–0.2, ...).
Within each bucket, we compare mean predicted probability to actual positive rate.
A perfectly calibrated model follows the diagonal line.

### Brier Score
`Brier = mean((y_prob - y_true)^2)` — lower is better.
Equivalent to mean squared error on probability predictions.

### Findings
- RF is reasonably calibrated in the mid-range (0.2–0.7) but overestimates at extremes
- GB is slightly overconfident (predicts higher probabilities than the actual rate)
  particularly for `future_dpd30`, consistent with high `scale_pos_weight`

Poor calibration means the predicted probability cannot be used directly as a
"confidence score" for the individual borrower. Platt scaling or isotonic regression
can correct calibration but were not applied here — that is a recommended next step.

---

## Outputs

```
artifacts/raw_checks/
    stage6_optimal_thresholds.csv
    stage6_predictions_{target}_{model}.parquet   (scores + demographic cols)
    stage6_fp_fn_analysis.csv
    stage6_slice_performance.csv
    stage6_calibration.csv
    stage6_notes.md
    stage6_manifest.json
```

The prediction parquets are the most important output — they contain
`y_prob`, `y_pred_05`, `y_pred_optimal`, and all demographic columns.
Stage 8 reads these directly for fairness analysis.

---

## Connects to Stage 7

Stage 7 loads `stage6_optimal_thresholds.csv` to apply the same threshold
when evaluating federated models. This ensures a fair comparison: centralized
and federated models are evaluated at the same operating point.
