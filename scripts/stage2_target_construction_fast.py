from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd


FUTURE_WINDOW_DAYS = 90


def configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_inputs(processed_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    logging.info("Loading cleaned Stage 1 artifacts...")
    inst = pd.read_parquet(
        processed_dir / "cleaned_installment_events.parquet",
        columns=[
            "SK_ID_PREV",
            "SK_ID_CURR",
            "NUM_INSTALMENT_VERSION",
            "NUM_INSTALMENT_NUMBER",
            "DAYS_INSTALMENT",
            "first_payment_day",
            "AMT_INSTALMENT",
            "total_amount_paid",
            "missing_payment_flag",
            "underpaid_after_aggregation_flag",
            "aggregated_payment_delay_days",
        ],
    )
    pos = pd.read_parquet(
        processed_dir / "cleaned_pos_cash_balance.parquet",
        columns=[
            "SK_ID_PREV",
            "SK_ID_CURR",
            "MONTHS_BALANCE",
            "SK_DPD",
            "SK_DPD_DEF",
        ],
    )
    logging.info("Installment events shape: %s", inst.shape)
    logging.info("POS cash shape: %s", pos.shape)
    return inst, pos


def build_snapshots(inst: pd.DataFrame) -> pd.DataFrame:
    logging.info("Building borrower-time snapshots...")
    t0 = time.time()
    inst = inst.sort_values(
        ["SK_ID_CURR", "DAYS_INSTALMENT", "NUM_INSTALMENT_NUMBER", "NUM_INSTALMENT_VERSION"]
    ).reset_index(drop=True)
    inst["snapshot_day"] = inst["DAYS_INSTALMENT"]
    inst["snapshot_month_approx"] = np.floor(inst["snapshot_day"] / 30.0).astype("int32")
    inst["borrower_event_order"] = inst.groupby("SK_ID_CURR").cumcount().astype("int32")

    next_cols = [
        "SK_ID_PREV",
        "NUM_INSTALMENT_VERSION",
        "NUM_INSTALMENT_NUMBER",
        "DAYS_INSTALMENT",
        "first_payment_day",
        "AMT_INSTALMENT",
        "total_amount_paid",
        "missing_payment_flag",
        "underpaid_after_aggregation_flag",
        "aggregated_payment_delay_days",
    ]
    next_events = inst.groupby("SK_ID_CURR")[next_cols].shift(-1).add_prefix("next_")
    snapshots = pd.concat([inst, next_events], axis=1)
    snapshots["has_next_installment"] = snapshots["next_DAYS_INSTALMENT"].notna().astype("int8")
    snapshots["next_installment_gap_days"] = snapshots["next_DAYS_INSTALMENT"] - snapshots["snapshot_day"]
    snapshots["snapshot_id"] = (
        snapshots["SK_ID_CURR"].astype("int64").astype(str)
        + "_"
        + snapshots["borrower_event_order"].astype(str)
    )
    logging.info("Built %s snapshots in %.1fs", f"{len(snapshots):,}", time.time() - t0)
    return snapshots


def build_missed_upcoming_emi(snapshots: pd.DataFrame) -> pd.DataFrame:
    logging.info("Constructing missed_upcoming_emi target...")
    t0 = time.time()
    missed = snapshots.loc[snapshots["has_next_installment"] == 1].copy()
    missed["missed_upcoming_emi"] = (
        (missed["next_missing_payment_flag"].fillna(0) > 0)
        | (missed["next_underpaid_after_aggregation_flag"].fillna(0) > 0)
        | (missed["next_aggregated_payment_delay_days"].fillna(-999999) > 0)
    ).astype("int8")
    missed["target_available_flag"] = 1
    missed["target_name"] = "missed_upcoming_emi"

    cols = [
        "snapshot_id",
        "SK_ID_CURR",
        "SK_ID_PREV",
        "borrower_event_order",
        "snapshot_day",
        "snapshot_month_approx",
        "DAYS_INSTALMENT",
        "first_payment_day",
        "aggregated_payment_delay_days",
        "missing_payment_flag",
        "underpaid_after_aggregation_flag",
        "next_SK_ID_PREV",
        "next_NUM_INSTALMENT_NUMBER",
        "next_DAYS_INSTALMENT",
        "next_first_payment_day",
        "next_aggregated_payment_delay_days",
        "next_missing_payment_flag",
        "next_underpaid_after_aggregation_flag",
        "next_installment_gap_days",
        "missed_upcoming_emi",
        "target_available_flag",
        "target_name",
    ]
    out = missed[cols].sort_values(["SK_ID_CURR", "snapshot_day"]).reset_index(drop=True)
    logging.info(
        "missed_upcoming_emi rows=%s positive_rate=%.4f built in %.1fs",
        f"{len(out):,}",
        out["missed_upcoming_emi"].mean(),
        time.time() - t0,
    )
    return out


def build_flagged_day_dict(df: pd.DataFrame, borrower_col: str, day_col: str, flag_col: str) -> dict[int, np.ndarray]:
    flagged = df.loc[df[flag_col] == 1, [borrower_col, day_col]].copy()
    if flagged.empty:
        return {}
    flagged = flagged.sort_values([borrower_col, day_col])
    return {
        int(borrower_id): group[day_col].to_numpy(dtype="float64", copy=True)
        for borrower_id, group in flagged.groupby(borrower_col, sort=False)
    }


def build_future_dpd30(
    snapshots: pd.DataFrame,
    inst: pd.DataFrame,
    pos: pd.DataFrame,
    future_window_days: int,
    limit_borrowers: int | None = None,
) -> pd.DataFrame:
    logging.info("Preparing future_dpd30 inputs...")
    t0 = time.time()

    inst_future = inst[["SK_ID_CURR", "DAYS_INSTALMENT", "aggregated_payment_delay_days"]].copy()
    inst_future["inst_dpd30_flag"] = (inst_future["aggregated_payment_delay_days"].fillna(-999999) >= 30).astype("int8")

    pos_future = pos.copy()
    pos_future["approx_day_from_month"] = pos_future["MONTHS_BALANCE"] * 30
    pos_future["pos_dpd30_flag"] = (
        (pos_future["SK_DPD"].fillna(0) >= 30) | (pos_future["SK_DPD_DEF"].fillna(0) >= 30)
    ).astype("int8")

    borrower_max_future_day = inst.groupby("SK_ID_CURR")["DAYS_INSTALMENT"].max().rename("borrower_max_installment_day")
    snapshots = snapshots.merge(borrower_max_future_day, on="SK_ID_CURR", how="left")
    snapshots["future_window_complete_flag"] = (
        snapshots["borrower_max_installment_day"] >= (snapshots["snapshot_day"] + future_window_days)
    ).astype("int8")

    inst_dpd30_days = build_flagged_day_dict(inst_future, "SK_ID_CURR", "DAYS_INSTALMENT", "inst_dpd30_flag")
    pos_dpd30_days = build_flagged_day_dict(pos_future, "SK_ID_CURR", "approx_day_from_month", "pos_dpd30_flag")
    logging.info("Borrowers with installment 30+ DPD hits: %s", f"{len(inst_dpd30_days):,}")
    logging.info("Borrowers with POS 30+ DPD hits: %s", f"{len(pos_dpd30_days):,}")

    snapshot_groups = snapshots.groupby("SK_ID_CURR", sort=False).indices
    borrower_items = list(snapshot_groups.items())
    if limit_borrowers is not None:
        borrower_items = borrower_items[:limit_borrowers]
        selected_indices = np.concatenate([idx for _, idx in borrower_items])
        snapshots = snapshots.iloc[selected_indices].copy().reset_index(drop=True)
        snapshot_groups = snapshots.groupby("SK_ID_CURR", sort=False).indices
        borrower_items = list(snapshot_groups.items())
        logging.warning("Limiting Stage 2 run to first %s borrowers for testing.", limit_borrowers)

    inst_hits = np.zeros(len(snapshots), dtype=np.int8)
    pos_hits = np.zeros(len(snapshots), dtype=np.int8)

    total_borrowers = len(borrower_items)
    logging.info("Starting future_dpd30 construction for %s borrowers...", f"{total_borrowers:,}")
    loop_start = time.time()

    for idx, (borrower_id, row_idx) in enumerate(borrower_items, start=1):
        borrower_snapshot_days = snapshots.iloc[row_idx]["snapshot_day"].to_numpy(dtype="float64", copy=False)
        window_end_days = borrower_snapshot_days + future_window_days

        inst_days = inst_dpd30_days.get(int(borrower_id))
        if inst_days is not None and len(inst_days) > 0:
            left = np.searchsorted(inst_days, borrower_snapshot_days, side="right")
            right = np.searchsorted(inst_days, window_end_days, side="right")
            inst_hits[row_idx] = (right > left).astype(np.int8)

        pos_days = pos_dpd30_days.get(int(borrower_id))
        if pos_days is not None and len(pos_days) > 0:
            left = np.searchsorted(pos_days, borrower_snapshot_days, side="right")
            right = np.searchsorted(pos_days, window_end_days, side="right")
            pos_hits[row_idx] = (right > left).astype(np.int8)

        if idx == 1 or idx % 5000 == 0 or idx == total_borrowers:
            logging.info(
                "Processed %s/%s borrowers in %.1fs",
                f"{idx:,}",
                f"{total_borrowers:,}",
                time.time() - loop_start,
            )

    snapshots["future_inst_dpd30_hit"] = inst_hits
    snapshots["future_pos_dpd30_hit"] = pos_hits
    snapshots["future_dpd30"] = (
        (snapshots["future_inst_dpd30_hit"] > 0) | (snapshots["future_pos_dpd30_hit"] > 0)
    ).astype("int8")

    out = snapshots.loc[snapshots["future_window_complete_flag"] == 1].copy()
    out["target_available_flag"] = 1
    out["target_name"] = "future_dpd30"
    cols = [
        "snapshot_id",
        "SK_ID_CURR",
        "SK_ID_PREV",
        "borrower_event_order",
        "snapshot_day",
        "snapshot_month_approx",
        "borrower_max_installment_day",
        "future_window_complete_flag",
        "future_inst_dpd30_hit",
        "future_pos_dpd30_hit",
        "future_dpd30",
        "target_available_flag",
        "target_name",
    ]
    out = out[cols].sort_values(["SK_ID_CURR", "snapshot_day"]).reset_index(drop=True)
    logging.info(
        "future_dpd30 rows=%s positive_rate=%.4f built in %.1fs",
        f"{len(out):,}",
        out["future_dpd30"].mean(),
        time.time() - t0,
    )
    return out


def save_outputs(
    processed_dir: Path,
    raw_checks_dir: Path,
    missed_target: pd.DataFrame,
    future_target: pd.DataFrame,
    future_window_days: int,
) -> None:
    logging.info("Saving Stage 2 outputs...")
    missed_target_path = processed_dir / "target_missed_upcoming_emi.parquet"
    future_target_path = processed_dir / "target_future_dpd30.parquet"
    label_summary_path = raw_checks_dir / "stage2_target_summary.csv"
    label_notes_path = raw_checks_dir / "stage2_target_notes.md"

    missed_target.to_parquet(missed_target_path, index=False)
    future_target.to_parquet(future_target_path, index=False)

    summary = pd.DataFrame(
        [
            {
                "target_name": "missed_upcoming_emi",
                "row_count": len(missed_target),
                "positive_count": int(missed_target["missed_upcoming_emi"].sum()),
                "positive_rate": float(missed_target["missed_upcoming_emi"].mean()),
            },
            {
                "target_name": "future_dpd30",
                "row_count": len(future_target),
                "positive_count": int(future_target["future_dpd30"].sum()),
                "positive_rate": float(future_target["future_dpd30"].mean()),
            },
        ]
    )
    summary.to_csv(label_summary_path, index=False)

    notes = "\n".join(
        [
            "# Stage 2 Target Notes",
            "",
            "- Snapshot anchor: observed borrower installment event at `snapshot_day = DAYS_INSTALMENT`.",
            "- `missed_upcoming_emi` uses the next borrower installment event after the snapshot.",
            f"- `future_dpd30` uses a fixed {future_window_days}-day future window.",
            "- `future_dpd30` is triggered by either installment delay >= 30 days or POS DPD >= 30 in the window.",
            "- Incomplete future windows are excluded from `future_dpd30`.",
        ]
    )
    label_notes_path.write_text(notes + "\n", encoding="utf-8")

    manifest = {
        "targets": [missed_target_path.name, future_target_path.name],
        "summary": label_summary_path.name,
        "notes": label_notes_path.name,
        "future_window_days": future_window_days,
    }
    (raw_checks_dir / "stage2_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    logging.info("Saved: %s", missed_target_path)
    logging.info("Saved: %s", future_target_path)
    logging.info("Saved: %s", label_summary_path)
    logging.info("Saved: %s", label_notes_path)
    logging.info("Target summary:\n%s", summary.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast Stage 2 target construction for Home Credit.")
    parser.add_argument("--future-window-days", type=int, default=FUTURE_WINDOW_DAYS)
    parser.add_argument("--limit-borrowers", type=int, default=None, help="Optional test limit for quick validation.")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)

    root = project_root()
    processed_dir = root / "artifacts" / "processed"
    raw_checks_dir = root / "artifacts" / "raw_checks"
    raw_checks_dir.mkdir(parents=True, exist_ok=True)

    total_start = time.time()
    inst, pos = load_inputs(processed_dir)
    snapshots = build_snapshots(inst)
    missed_target = build_missed_upcoming_emi(snapshots)
    future_target = build_future_dpd30(
        snapshots=snapshots,
        inst=inst,
        pos=pos,
        future_window_days=args.future_window_days,
        limit_borrowers=args.limit_borrowers,
    )
    save_outputs(
        processed_dir=processed_dir,
        raw_checks_dir=raw_checks_dir,
        missed_target=missed_target,
        future_target=future_target,
        future_window_days=args.future_window_days,
    )
    logging.info("Stage 2 completed in %.1fs", time.time() - total_start)


if __name__ == "__main__":
    main()
