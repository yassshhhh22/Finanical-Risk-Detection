# Stage 8: Ethical Evaluation

## Purpose

Predictive performance alone is not enough for credit-risk systems.
Stage 8 evaluates whether model behavior is acceptable across three dimensions:

1. Privacy: what information could leak in centralized/federated setups
2. Fairness: whether error patterns differ systematically across groups
3. Explainability: whether model logic is transparent and domain-consistent

This stage transforms Stage 6 prediction artifacts and Stage 7 federation outcomes
into governance-ready evidence.

---

## Inputs and Why They Matter

Primary inputs:

- Stage 6 prediction files (`y_prob`, `y_pred_optimal`, demographics)
- Stage 5 trained models and preprocessors
- Stage 4 column manifest for consistent schema

Why Stage 6 predictions are reused:

- They already contain calibrated probabilities and optimal-threshold labels
- They include demographic slices used for fairness diagnostics
- Reusing them prevents metric drift from recomputing predictions differently

---

## Fairness Evaluation Framework

Sensitive/group attributes evaluated (when available):

- `CODE_GENDER`
- `NAME_INCOME_TYPE`
- `NAME_EDUCATION_TYPE`
- `NAME_FAMILY_STATUS`
- `NAME_HOUSING_TYPE`

Minimum group size filter:

- groups with fewer than 500 rows are skipped to reduce noisy conclusions

### Metrics computed per group

1. Demographic parity:

- positive prediction rate by group
- disparate impact (DI) ratio relative to overall rate
- rule-of-thumb concern zone: DI < 0.8 or DI > 1.25

2. Equalized odds:

- TPR (recall) by group
- FPR by group
- gap vs overall baseline (`tpr_gap_vs_overall`, `fpr_gap_vs_overall`)

3. Predictive parity:

- precision by group

4. Group quality metrics:

- ROC-AUC
- PR-AUC
- F1

5. Calibration fairness:

- Brier score per group
- calibration gap = mean predicted probability - observed prevalence

Why multiple fairness metrics are necessary:

- A model can satisfy one notion of fairness while violating another
- Credit policy decisions involve both selection rates (parity) and error burden
  (equalized odds / harm)

---

## Harm Analysis Framework

The project explicitly maps model errors to social and business harms.

False Positive (FP):

- Borrower is creditworthy but flagged risky
- Harm type: exclusion harm (denial/restriction of credit)

False Negative (FN):

- Borrower defaults but predicted safe
- Harm type: institutional harm (losses, potential over-indebtedness)

Per-group outputs include:

- FP/FN counts
- FP rate among actual negatives
- FN rate among actual positives

Why this framing is important:

- Harm is not symmetric in credit risk
- The same global accuracy can hide unequal burden distribution

---

## Explainability (SHAP)

Method:

- `TreeExplainer` on RF and LightGBM models
- global importance via mean absolute SHAP values
- top-ranked features saved per model and target

Why SHAP here:

- Tree models are used; SHAP is exact/efficient with tree explainers
- It supports both governance reporting and model debugging
- It checks whether learned signal aligns with known risk logic

Observed pattern:

- installment-history features dominate both targets
- examples: `hist_on_time_streak`, `hist_late_count_last_6`,
  `hist_recent_delay`, `hist_delay_max`

Interpretation:

- model focus is behavior-based (payment history) rather than directly
  demographic-based, which is directionally positive for fairness governance

---

## Privacy Assessment

### Centralized training risk

- Raw data pooled in one environment
- broad access footprint
- higher blast radius in case of breach

### Federated simulation implications (from Stage 7)

- raw rows are not exchanged
- shared artifacts are trees or prediction scores
- this reduces direct exposure but does not provide formal privacy guarantees

Residual privacy risks:

- membership inference from overfit model artifacts
- reconstruction risk under repeated query access

Recommended mitigations for production:

- secure aggregation
- differential privacy noise on shared updates
- strict model access controls and logging
- periodic privacy red-team testing

---

## Key Findings (Practical Summary)

- Gender DI ratios are generally within acceptable screening range
  (approximately 0.99 to 1.13 in final summaries).
- Larger fairness variation appears in some education and housing subgroups.
- Small-sample groups exhibit unstable TPR/FPR and are flagged for monitoring,
  not immediate policy automation.
- SHAP confirms dominant reliance on payment behavior features.

Overall ethical readout:

- no immediate severe fairness red flags at aggregate level
- but deployment should include subgroup monitoring, calibration maintenance,
  and explicit adverse-impact governance thresholds

---

## Limitations

- Observational fairness only; no causal fairness test is performed.
- Demographic fields are constrained by dataset definitions.
- Results are sensitive to threshold choices from Stage 6.
- Simulated federation cannot capture full cross-institution domain shift.

---

## Outputs

```
artifacts/metrics/
    stage8_fairness_metrics.csv
    stage8_harm_analysis.csv
    stage8_shap_importance.csv
    stage8_calibration_fairness.csv

artifacts/raw_checks/
    stage8_notes.md
    stage8_manifest.json
```

---

## Connects to Stage 9

Stage 9 consumes these outputs to generate:

- fairness summary tables
- SHAP feature charts
- harm visualizations
- ethics sections in `FINAL_REPORT.md`
