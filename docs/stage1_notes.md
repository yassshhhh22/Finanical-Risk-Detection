# Stage 1: Raw Data Audit and Cleaning

## Purpose

Before any modelling can happen, the raw Home Credit tables must be validated.
This stage answers three questions:
1. What does the data actually look like (shapes, types, ranges)?
2. Where is it broken (missing values, duplicates, impossible values)?
3. Can we trust it enough to build labels and features on top of it?

The output is a set of cleaned Parquet files that every later stage reads from.
Nothing downstream ever touches the original CSVs.

---

## Dataset

The Home Credit Default Risk dataset contains seven relational tables:

| Table | Rows (approx) | What it represents |
|---|---|---|
| `application_train.csv` | 307,511 | One row per loan application — demographics, financials |
| `bureau.csv` | 1,716,428 | Credit bureau records for each applicant |
| `bureau_balance.csv` | 27,299,925 | Monthly status of each bureau credit |
| `previous_application.csv` | 1,670,214 | Past loan applications at Home Credit |
| `installments_payments.csv` | 13,605,401 | Actual payment records per instalment |
| `POS_CASH_balance.csv` | 10,001,358 | Monthly POS/cash loan balance snapshots |
| `credit_card_balance.csv` | 3,840,312 | Monthly credit card snapshots |

All tables link back to `application_train` via `SK_ID_CURR` (borrower ID).
`bureau_balance` also has `SK_ID_BUREAU`, linking to `bureau`.

---

## Why Parquet?

CSV files are re-parsed every time they are read.
Parquet stores data in columnar binary format with embedded schema and compression.
For a dataset of this size, reading a Parquet file is 10–30× faster and uses
less memory than reading the equivalent CSV.
All cleaned outputs are saved as Parquet so every downstream stage can load only
the columns it needs (column pruning).

---

## What Was Cleaned

### Application table
- Removed the `TARGET` column (used only for reference; project builds its own labels)
- Replaced anomalous `DAYS_EMPLOYED = 365243` (a sentinel for "not employed") with `NaN`
- Clipped extreme outliers on income and credit columns (> 99.9th percentile)
- Standardised `CODE_GENDER`: replaced rare `XNA` entries with `NaN`
- Imputed nothing at this stage — imputation happens in Stage 5 preprocessing

### Installments payments
- Sorted by `SK_ID_CURR`, `SK_ID_PREV`, `DAYS_INSTALMENT`
- Removed rows where `AMT_PAYMENT` is negative (data entry errors)
- Created `cleaned_installment_events_sorted.parquet` for efficient merging

### Bureau / Bureau balance
- Removed bureau records with no balance history (uninformative)
- Fixed `STATUS` column: encoded A/C/X/0–5 consistently

### POS cash balance
- Clipped `CNT_INSTALMENT_FUTURE` at 0 (negatives are impossible)

---

## Missingness Strategy

Missing values are **not imputed here**.
They are preserved as `NaN` and handed to the Stage 5 preprocessing pipeline
(`SimpleImputer`) which fits only on the training set.
Imputing here would leak test-set statistics into the training data.

---

## Outputs

```
artifacts/processed/
    cleaned_application_train.parquet
    cleaned_installments_payments.parquet
    cleaned_installment_events.parquet
    cleaned_pos_cash_balance.parquet
    cleaned_previous_application.parquet
    cleaned_bureau.parquet
    cleaned_bureau_balance.parquet

artifacts/raw_checks/
    raw_audit_summary.csv          shape, dtypes, memory per table
    raw_missingness_summary.csv    % missing per column
    raw_duplicate_summary.csv      duplicate row counts
    raw_anomaly_notes.md           documented anomalies
    stage1_manifest.json
```

---

## Key Decisions

**Why not drop high-missingness columns?**
Some columns (e.g. `OWN_CAR_AGE`, `OCCUPATION_TYPE`) are missing for 40–50% of
rows but are highly predictive when present. Dropping them would discard signal.
Ordinal encoding with `unknown_value=-1` (Stage 5) handles unseen/missing categories
gracefully at inference time.

**Why sort installments?**
Feature engineering in Stage 3 computes rolling windows (last 3 months, last 6 months)
over payment history. These require the data to be sorted by date to be correct.
Sorting once here avoids re-sorting in every feature computation.

---

## Connects to Stage 2

Stage 2 reads `cleaned_application_train.parquet` and
`cleaned_installments_payments.parquet` to build borrower-time snapshots
and assign labels.
