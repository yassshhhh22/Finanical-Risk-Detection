# Stage 2: Target Construction

## Purpose

A loan application is a single point in time, but credit risk is a temporal problem.
A borrower who pays reliably for 12 months and then misses 3 consecutive payments
is far more dangerous than someone who has never missed a payment.

This stage converts the static application table into **borrower-time snapshots**:
one row per borrower per 30-day window, across their active loan history.
Each snapshot is then labelled with two distinct risk signals.

---

## Why Two Labels?

Different business decisions need different prediction horizons:

| Label | Meaning | Horizon | Use case |
|---|---|---|---|
| `missed_upcoming_emi` | Will the borrower miss the *next* scheduled payment? | 0–30 days | Operational alert: flag loans for outreach before the payment date |
| `future_dpd30` | Will the borrower go 30+ days past due in the next 90 days? | 30–90 days | Provisioning: reserve capital for expected losses |

Having both allows the models to be evaluated on both short-term operational
accuracy and medium-term strategic accuracy — which is important for the ethical
evaluation in Stage 8.

---

## What is a Borrower-Time Snapshot?

For each active loan, we generate one snapshot every 30 days.
A snapshot records the state of the borrower *at that point in time*:
- How many payments have they made so far?
- What is their current overdue balance?
- What is their on-time streak?

**Why 30-day windows?**
EMI schedules are monthly. A 30-day resolution aligns with the natural payment
cycle and avoids sub-monthly noise in the payment records.

---

## Label Construction

### `missed_upcoming_emi`

For each snapshot at day `d`:
1. Find the next scheduled installment after day `d`
2. Look up whether `AMT_PAYMENT` for that installment was recorded within
   the due date plus a 5-day grace period
3. If no payment or underpayment: label = 1

**Positive rate: 4.9%**
This is a moderately imbalanced classification problem.

### `future_dpd30`

For each snapshot at day `d`:
1. Look forward 90 days
2. If any payment during that window has `DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT > 30`,
   meaning it was paid 30+ days late: label = 1

**Positive rate: 0.78%**
This is a severely imbalanced problem — only about 1 in 128 snapshots is positive.
This drives the choice of `scale_pos_weight` and calibration analysis later.

---

## Why Not Use the Original `TARGET` Column?

The original `TARGET` in the application table is a binary flag indicating whether
the borrower *ever* defaulted during the entire loan. It has no temporal resolution.
Our snapshot-level labels allow the model to learn *when* risk materialises,
not just *whether* it materialises. This is essential for real-time credit monitoring.

---

## Scale

| Split | Rows |
|---|---|
| Full train snapshots | ~6,075,794 |
| Validation snapshots | ~2,842,665 |
| Test snapshots | ~3,693,872 (missed_upcoming_emi) / ~2,871,365 (future_dpd30) |

The large row counts come from the many 30-day windows per borrower.
A borrower with a 36-month loan contributes up to 36 snapshots.

---

## Outputs

```
artifacts/processed/
    target_missed_upcoming_emi.parquet
    target_future_dpd30.parquet

artifacts/raw_checks/
    stage2_target_summary.csv
    stage2_target_notes.md
    stage2_manifest.json
```

---

## Key Design Decisions

**Grace period of 5 days for `missed_upcoming_emi`:**
Payment processing systems often delay posting by 1–3 days.
Without a grace period, legitimate on-time payments would be mislabelled as missed.

**90-day forward window for `future_dpd30`:**
Shorter windows (30 days) would be too reactive — a borrower who is 29 days overdue
today would not yet be labelled positive. Longer windows (180 days) would blur the
signal by including events too far in the future to be actionable.

**Why not `future_dpd60` or `future_dpd90`?**
30+ DPD is the standard industry threshold for "Special Mention" classification
under most banking regulations. It also corresponds to the point at which
provisioning requirements typically kick in.

---

## Connects to Stage 3

Stage 3 reads both target files and enriches them with features.
The `snapshot_day` column (the day the snapshot was taken relative to loan start)
becomes the primary join key for feature engineering.
