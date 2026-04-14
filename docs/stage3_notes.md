# Stage 3: Feature Engineering

## Purpose

Raw tables contain events (payments, bureau updates, balance entries).
Models need numbers. This stage transforms raw events into **predictive signals**
for each borrower-time snapshot: one row, one vector of numbers, ready for a classifier.

The goal is to distil everything that is known about a borrower *up to but not
including* the snapshot date into a fixed-width feature vector.

---

## Feature Families

Features are grouped into five families, each cached separately so they can be
rebuilt independently if the raw data changes.

### 1. Application Features (48 features)
Source: `cleaned_application_train.parquet`
These are borrower-level static attributes measured at loan origination:

| Feature group | Examples |
|---|---|
| Credit/income ratios | `AMT_CREDIT`, `AMT_INCOME_TOTAL`, `AMT_CREDIT / AMT_INCOME_TOTAL` |
| Demographics | `CODE_GENDER`, `NAME_INCOME_TYPE`, `NAME_EDUCATION_TYPE` |
| Family/housing | `NAME_FAMILY_STATUS`, `NAME_HOUSING_TYPE`, `CNT_FAM_MEMBERS` |
| Employment | `DAYS_EMPLOYED`, `ORGANIZATION_TYPE` |
| Documentation | `FLAG_DOCUMENT_3`, `FLAG_DOCUMENT_6`, ... |

These features are **static** — they don't change across snapshots for the same borrower.

### 2. Previous Application Features (8 features)
Source: `cleaned_previous_application.parquet`

Captures the borrower's history with Home Credit before the current loan:

| Feature | Meaning |
|---|---|
| `prev_application_count` | How many times the borrower has applied |
| `prev_approved_count` | How many past applications were approved |
| `prev_refused_count` | How many were refused |
| `prev_credit_max` | Largest credit ever granted |
| `prev_days_decision_min` | How long ago the most recent decision was |

**Why this matters:** A borrower who has been refused multiple times but keeps
applying is a very different risk profile from a first-time applicant.

### 3. Bureau Features (10 features)
Source: `cleaned_bureau.parquet`, `cleaned_bureau_balance.parquet`

External credit bureau records (from other lenders):

| Feature | Meaning |
|---|---|
| `bureau_active_count` | Number of currently open credits at other institutions |
| `bureau_overdue_count` | Credits currently overdue externally |
| `bureau_total_debt` | Total outstanding debt across all external credits |
| `bureau_avg_days_credit` | Average age of external credit relationships |
| `bureau_max_dpd` | Worst ever days past due recorded at bureau |

**Why this matters:** A borrower who looks fine internally but has five overdue
loans externally is a major risk. Without bureau data, the model would miss this.

### 4. Installment History Features (14 features)
Source: `cleaned_installment_events_sorted.parquet`

This is the highest-signal family. It tracks actual payment behaviour over time.
Computed as rolling statistics up to the snapshot date:

| Feature | Meaning |
|---|---|
| `hist_total_installments` | Total number of payments ever made |
| `hist_late_count` | Total payments ever made late |
| `hist_late_count_last_3` | Late payments in last 3 months |
| `hist_late_count_last_6` | Late payments in last 6 months |
| `hist_on_time_streak` | Consecutive on-time payments (most recent) |
| `hist_recent_delay` | Days overdue on the most recent payment |
| `hist_delay_mean_last_6` | Mean delay (days) over last 6 months |
| `hist_delay_max` | Maximum delay ever recorded |
| `hist_underpayment_rate` | Fraction of payments where `AMT_PAYMENT < AMT_INSTALMENT` |

**Why these are the most predictive features:**
Recent payment behaviour is the strongest predictor of near-future behaviour.
A borrower who paid on time for 24 months and then missed two is very different
from one who has always been 5 days late. The rolling windows capture this trajectory.

**Why rolling windows (3-month, 6-month)?**
Older behaviour decays in relevance. A missed payment 2 years ago should carry
less weight than one last month. By computing separate windows, the model can
learn this decay implicitly from data without us hard-coding a weighting scheme.

### 5. POS/Cash Balance Features (6 features)
Source: `cleaned_pos_cash_balance.parquet`

Monthly balance snapshots for revolving products:

| Feature | Meaning |
|---|---|
| `pos_months_balance_min` | Earliest month in the POS history |
| `pos_cnt_instalment_avg` | Average remaining instalments |
| `pos_sk_dpd_max` | Maximum DPD ever in POS history |

---

## Total Feature Count

- **48 numeric features** + **6 categorical features** = 54 features per snapshot
- The 6 categorical: `CODE_GENDER`, `NAME_INCOME_TYPE`, `NAME_EDUCATION_TYPE`,
  `NAME_FAMILY_STATUS`, `NAME_HOUSING_TYPE`, `OCCUPATION_TYPE`

---

## Why Cache Feature Families?

Each feature family joins millions of rows. Bureau features alone take 5–10 minutes
to compute. If we rebuild all features every time we change one thing, iteration
becomes impossibly slow.

By caching each family as a Parquet file, Stage 3 only recomputes what has changed.
The `--rebuild-cache` flag forces a full rebuild when needed.

---

## Why Not Use the Full Application Feature Set?

The raw application table has 120+ columns. Many are:
- Highly correlated (e.g. 20 `FLAG_DOCUMENT_*` columns)
- Mostly missing (> 70% null)
- Leaky (derived from events that happen after the snapshot date)

Manual selection to 48 features removes noise and prevents the model from
finding spurious patterns in near-empty columns.

---

## Feature Quality Report

Stage 3 outputs `stage3_feature_quality_report.csv` with:
- Missing rate per feature
- Mean, std, min, max
- Correlation with label (Pearson, for reference only)

Features with > 80% missing are flagged but not dropped — they may still carry
signal in the non-missing rows.

---

## Outputs

```
artifacts/processed/
    features_application.parquet           (cached)
    features_previous_application.parquet  (cached)
    features_bureau.parquet                (cached)
    features_installment_history.parquet   (cached)
    snapshot_feature_base.parquet          (all families joined)
    model_features_missed_upcoming_emi.parquet
    model_features_future_dpd30.parquet

artifacts/raw_checks/
    stage3_feature_dictionary.csv
    stage3_feature_quality_report.csv
    stage3_manifest.json
```

---

## Connects to Stage 4

Stage 4 reads `model_features_{target}.parquet` and applies chronological
splitting. The `snapshot_day` column (the date of each snapshot) is used to
ensure no future information leaks into the training set.
