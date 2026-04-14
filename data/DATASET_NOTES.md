## Downloaded Datasets

### 0. Official Home Credit Full Competition Data

- Local folder: `data/home_credit_full/`
- Source: `https://www.kaggle.com/competitions/home-credit-default-risk/data`
- Status: downloaded and extracted successfully

Key files now available locally:

- `application_train.csv`
- `application_test.csv`
- `installments_payments.csv`
- `POS_CASH_balance.csv`
- `previous_application.csv`
- `bureau.csv`
- `bureau_balance.csv`
- `HomeCredit_columns_description.csv`

Why this is the best dataset for the project:

- contains borrower basics in `application_train.csv`
- contains loan/application details
- contains installment-level repayment history in `installments_payments.csv`
- contains delinquency-style fields in `POS_CASH_balance.csv` and `bureau.csv`
- supports time-based target construction better than the earlier summary datasets

Project fit:

- `missed upcoming EMI`: supported
- `30+ DPD within future window`: supported

Important note:

- Home Credit uses relative day/month offsets rather than clean calendar due-date fields, so target engineering will be sequence-based from these offsets
- this is still the best exact match we have found for your topic

### 1. Nigerian BNPL

- Local file: `data/nigerian_bnpl_full.parquet`
- Source: `https://huggingface.co/datasets/electricsheepafrica/nigerian-banking-bnpl`
- Rows: `2,000,000`
- Best for:
  - installment-style risk prediction
  - 30-day / 90-day delinquency targets
  - simulated federated learning via `provider`

Key columns:

- `purchase_date`
- `first_payment_due`
- `principal_ngn`
- `tenor_days`
- `num_installments`
- `provider`
- `credit_score`
- `default_30d`
- `default_90d`

Main limitation:

- no full per-installment payment history
- synthetic dataset

### 2. Home Credit Train

- Local file: `data/home_credit/train.csv`
- Source mirror: `https://github.com/sultanbeishenkulov/home-credit-default-risk`
- Best for:
  - borrower basics
  - loan application features
  - strong centralized credit-risk baseline modeling

Examples of useful columns:

- `TARGET`
- `AMT_INCOME_TOTAL`
- `AMT_CREDIT`
- `AMT_ANNUITY`
- `NAME_INCOME_TYPE`
- `OCCUPATION_TYPE`
- `DAYS_AGE`
- `DAYS_EMPLOYMENT`

Main limitation:

- this mirror is a transformed training table, not the full raw Kaggle package
- installment-level payment history is not included here

## Recommendation

Use:

- `data/home_credit_full/` as the main project dataset
- `data/nigerian_bnpl_full.parquet` as an optional secondary comparison dataset for federated simulation experiments

If we later obtain the full `installments_payments.csv` Home Credit table, that will become the best dataset for your original two-target design.
