# Stage 7: Simulated Federated Learning

## Purpose

Stage 7 answers a practical deployment question:

Can we preserve most centralized-model performance without sharing raw borrower data?

Instead of training one model on pooled data, we simulate a federated setting where
multiple clients train locally and only model artifacts are aggregated. This stage
provides the quantitative evidence needed for Stage 8 privacy/fairness discussion and
for Stage 9 final reporting.

---

## Federated Setup

### Client construction

- Number of pseudo-clients: `K = 5`
- Partitioning strategy: stratified random partition on the training split
- Each client receives roughly equal class balance (positives/negatives)

Why stratified partitioning?

- The targets are highly imbalanced (especially `future_dpd30`)
- A naive random split could create clients with almost no positives
- Federated comparison would become unstable and unfair if one client has no signal

### Evaluation protocol

Each target/model is evaluated in three modes on the same global test set:

1. `centralized`:
   model from Stage 5 trained on full pooled data
2. `local`:
   best and worst single-client models evaluated globally
3. `federated`:
   aggregated model using only client-level model artifacts

This gives a full picture:

- centralized = upper bound
- local = lower bound (single-site behavior)
- federated = privacy-preserving compromise

---

## Aggregation Design by Model Family

### Random Forest: FedForest

Method:

- Each client trains `n_estimators / K` trees locally
- All local trees are concatenated into one final global forest

Why this works:

- Random Forest predictions are an average over independent trees
- A tree trained on client A and a tree trained on client B are both valid estimators
- Concatenating trees is mathematically consistent with bagged-tree ensembling

Why this is privacy-relevant:

- Raw rows never leave clients
- Only tree structures are shared

Known risk:

- Overfit leaves can leak membership patterns if not controlled
- Mitigation: depth limits and minimum leaf sizes (already enforced in Stage 5)

### Gradient Boosting: FedEnsemble

Method:

- Each client trains a local LightGBM independently
- Final global prediction is the average of local model probabilities

Why this strategy (instead of direct tree merge):

- Boosted trees are sequentially dependent
- You cannot safely concatenate trees from separately trained GBMs like RF trees
- Probability averaging preserves local signal while remaining model-agnostic

Expected tradeoff:

- Slightly weaker than centralized GB because clients do not share gradient history
- Still typically close if clients are sufficiently representative

---

## Thresholding and Fair Comparison

Stage 7 loads `stage6_optimal_thresholds.csv` and evaluates models at the same
operating thresholds used in centralized diagnostics.

Why this matters:

- If centralized and federated models use different thresholds, comparisons are biased
- Holding thresholds fixed isolates the effect of federation itself

---

## Results Summary

### Global test ROC-AUC / F1

#### `missed_upcoming_emi`

- `rf`
  - Centralized: ROC-AUC 0.7801, F1 0.1994
  - Federated: ROC-AUC 0.7776, F1 0.1978
  - Gap: -0.0025 ROC-AUC

- `gb`
  - Centralized: ROC-AUC 0.7789, F1 0.2006
  - Federated: ROC-AUC 0.7808, F1 0.1995
  - Gap: +0.0019 ROC-AUC

#### `future_dpd30`

- `rf`
  - Centralized: ROC-AUC 0.8217, F1 0.3070
  - Federated: ROC-AUC 0.8058, F1 0.3012
  - Gap: -0.0159 ROC-AUC

- `gb`
  - Centralized: ROC-AUC 0.7701, F1 0.2037
  - Federated: ROC-AUC 0.7698, F1 0.1927
  - Gap: -0.0003 ROC-AUC

### Interpretation

- For three out of four target/model combinations, federated performance remains
  within roughly 0.00 to 0.02 ROC-AUC of centralized training.
- Largest loss appears in `future_dpd30 + rf`, which is expected because:
  - prevalence is extremely low
  - positive examples are split across clients
  - each local client sees less minority-class diversity

Overall conclusion:
Federated learning is viable for this dataset with only minor performance loss.

---

## What This Stage Proves

1. Privacy-preserving training can be practical for tabular credit-risk models.
2. FedForest is an effective federated approximation for RF.
3. FedEnsemble can keep GB close to centralized quality for most targets.
4. Performance cost is measurable and acceptable for policy-level deployment discussion.

---

## Limitations

- Clients are simulated from one dataset; real institutions have stronger domain shift.
- No secure aggregation protocol is implemented in this simulation.
- Communication/computation overhead is not benchmarked here.
- No differential privacy noise is injected in this stage.

These limitations are addressed as recommendations in Stage 8.

---

## Outputs

```
artifacts/metrics/
    stage7_client_metrics.csv
    stage7_comparison.csv

artifacts/raw_checks/
    stage7_notes.md
    stage7_manifest.json
```

---

## Connects to Stage 8

Stage 8 uses Stage 7 outcomes to reason about privacy-accuracy tradeoffs and to
document production-safe mitigations (secure aggregation, DP, monitoring).

## Connects to Stage 9

Stage 9 converts `stage7_comparison.csv` into federated comparison tables and
Figure 5 (`fig_05_federated_comparison.png`) in the final report.
