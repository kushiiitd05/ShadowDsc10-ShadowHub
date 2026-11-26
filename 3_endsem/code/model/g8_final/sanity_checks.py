"""Sanity checks for a final-retrain prediction file (submission layout: Id, OrderVolume).

Usage (real run):
  python sanity_checks.py <out_dir>/submission_final.csv --frame <proj>/data/processed/test_forecast_features.csv.gz \
      --reference <control>/submission_control.csv [--sample-submission <proj>/datset/sample_submission.csv] [--out report.json]
Usage (dry run, file written by final_retrain.py --dry-run-fold N; reference = the fold's actual volumes):
  python sanity_checks.py <out_dir>/dry_run_fold4_predictions.csv --dry-run

Hard checks (exit code 1 when any fails): row count, Id order, finite values, no negatives, closed rows exactly 0.
Distribution checks against the reference compare log1p means, quantiles, correlation and per-horizon means with thresholds.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

EXPECTED_ROWS = 46_830
QUANTILES = (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)
MAX_MEAN_LOG_DIFF = 0.10
MAX_HORIZON_MEAN_LOG_DIFF = 0.15
MIN_CORRELATION = 0.95
# dry run compares with the fold's real volumes (a single day carries real noise), not with a similar model: looser limits
DRY_RUN_LIMITS = {"max_mean": 0.10, "max_horizon": 0.25, "min_corr": 0.90}


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str


def _log(values: np.ndarray | pd.Series) -> np.ndarray:
    return np.log1p(np.clip(np.asarray(values, dtype=float), 0, None))


def check_layout(pred: pd.DataFrame, expected_rows: int, id_order: np.ndarray | None, id_label: str) -> list[Check]:
    checks = [Check("columns", list(pred.columns[:2]) == ["Id", "OrderVolume"], f"columns {list(pred.columns)}"),
              Check("row_count", len(pred) == expected_rows, f"{len(pred)} rows, expected {expected_rows}"),
              Check("id_unique", bool(pred["Id"].is_unique), "Id values are unique" if pred["Id"].is_unique else "duplicate Id values")]
    if id_order is not None:
        same = len(id_order) == len(pred) and bool(np.array_equal(pred["Id"].to_numpy(), id_order))
        checks.append(Check(f"id_order_vs_{id_label}", same, "Id sequence identical" if same else "Id sequence differs"))
    return checks


def check_values(pred: pd.DataFrame, frame: pd.DataFrame) -> list[Check]:
    """frame holds IsOpen aligned row by row with pred (same Id order)."""
    values = pred["OrderVolume"].to_numpy(dtype=float)
    closed = frame["IsOpen"].to_numpy() == 0
    open_zero = int(((~closed) & (values == 0)).sum())
    return [
        Check("no_nan_inf", bool(np.isfinite(values).all()), f"{int((~np.isfinite(values)).sum())} non-finite values"),
        Check("no_negative", bool((values >= 0).all()), f"{int((values < 0).sum())} negative values"),
        Check("closed_rows_zero", bool((values[closed] == 0).all()), f"{int(closed.sum())} closed rows, {int((values[closed] != 0).sum())} non-zero"),
        Check("open_rows_positive", open_zero == 0, f"{open_zero} open rows predicted exactly 0"),
    ]


def distribution_summary(pred_values: np.ndarray, ref_values: np.ndarray, horizon: np.ndarray, is_open: np.ndarray) -> dict:
    """Mean/quantiles of log1p, correlation and per-horizon means for prediction and reference, over open rows only
    (closed rows are 0 in both and would inflate every agreement figure)."""
    keep = np.asarray(is_open) == 1
    pred_values, ref_values, horizon = np.asarray(pred_values)[keep], np.asarray(ref_values)[keep], np.asarray(horizon)[keep]
    lp, lr = _log(pred_values), _log(ref_values)
    summary = {
        "mean_log1p": {"pred": float(lp.mean()), "ref": float(lr.mean())},
        "quantiles_log1p": {str(q): {"pred": float(np.quantile(lp, q)), "ref": float(np.quantile(lr, q))} for q in QUANTILES},
        "correlation_log1p": float(np.corrcoef(lp, lr)[0, 1]),
        "per_horizon_mean_log1p": {},
    }
    for h in sorted(np.unique(horizon)):
        m = horizon == h
        summary["per_horizon_mean_log1p"][int(h)] = {"pred": float(lp[m].mean()), "ref": float(lr[m].mean())}
    return summary


def check_distribution(summary: dict, max_mean: float, max_horizon: float, min_corr: float) -> list[Check]:
    mean_diff = summary["mean_log1p"]["pred"] - summary["mean_log1p"]["ref"]
    horizon_diffs = {h: v["pred"] - v["ref"] for h, v in summary["per_horizon_mean_log1p"].items()}
    worst_h, worst = max(horizon_diffs.items(), key=lambda kv: abs(kv[1]))
    return [
        Check("mean_log1p_vs_reference", abs(mean_diff) <= max_mean, f"mean log1p diff {mean_diff:+.4f} (limit {max_mean})"),
        Check("correlation_vs_reference", summary["correlation_log1p"] >= min_corr, f"corr {summary['correlation_log1p']:.4f} (min {min_corr})"),
        Check("per_horizon_mean_vs_reference", abs(worst) <= max_horizon, f"worst horizon {worst_h}: {worst:+.4f} (limit {max_horizon})"),
    ]


def run_checks(pred: pd.DataFrame, frame: pd.DataFrame, reference: pd.DataFrame | None, sample_ids: np.ndarray | None = None,
               expected_rows: int = EXPECTED_ROWS, max_mean: float = MAX_MEAN_LOG_DIFF, max_horizon: float = MAX_HORIZON_MEAN_LOG_DIFF,
               min_corr: float = MIN_CORRELATION) -> tuple[list[Check], dict | None]:
    """pred: Id, OrderVolume; frame: IsOpen + Horizon, same row order as pred; reference: Id, OrderVolume (any earlier prediction)."""
    checks = check_layout(pred, expected_rows, None if reference is None else reference["Id"].to_numpy(), "reference")
    if sample_ids is not None:
        checks.extend(c for c in check_layout(pred, expected_rows, sample_ids, "sample_submission") if c.name.startswith("id_order"))
    if len(frame) != len(pred):
        return [*checks, Check("frame_alignment", False, f"frame has {len(frame)} rows, predictions {len(pred)}")], None
    checks.extend(check_values(pred, frame))
    if reference is None or len(reference) != len(pred):
        return checks, None
    summary = distribution_summary(pred["OrderVolume"].to_numpy(), reference["OrderVolume"].to_numpy(), frame["Horizon"].to_numpy(), frame["IsOpen"].to_numpy())
    return [*checks, *check_distribution(summary, max_mean, max_horizon, min_corr)], summary


def _print(checks: list[Check], summary: dict | None) -> None:
    for c in checks:
        print(f"[{'PASS' if c.passed else 'FAIL'}] {c.name}: {c.detail}")
    if summary:
        print("log1p quantiles (pred / ref):", {q: (round(v["pred"], 3), round(v["ref"], 3)) for q, v in summary["quantiles_log1p"].items()})
        print("per-horizon mean log1p (pred / ref):", {h: (round(v["pred"], 3), round(v["ref"], 3)) for h, v in summary["per_horizon_mean_log1p"].items()})


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pred_csv", type=Path)
    ap.add_argument("--frame", type=Path, help="test_forecast_features.csv.gz (Id, IsOpen, Horizon in submission order)")
    ap.add_argument("--reference", type=Path, help="earlier prediction file with Id, OrderVolume")
    ap.add_argument("--sample-submission", type=Path)
    ap.add_argument("--dry-run", action="store_true", help="pred_csv carries IsOpen, Horizon, ActualOrderVolume; the actuals are the reference")
    ap.add_argument("--expected-rows", type=int, default=EXPECTED_ROWS)
    ap.add_argument("--max-mean-diff", type=float, default=None, help=f"default {MAX_MEAN_LOG_DIFF}")
    ap.add_argument("--max-horizon-diff", type=float, default=None, help=f"default {MAX_HORIZON_MEAN_LOG_DIFF} (dry run {DRY_RUN_LIMITS['max_horizon']})")
    ap.add_argument("--min-corr", type=float, default=None, help=f"default {MIN_CORRELATION} (dry run {DRY_RUN_LIMITS['min_corr']})")
    ap.add_argument("--out", type=Path, help="write the report as JSON")
    args = ap.parse_args(argv)

    pred = pd.read_csv(args.pred_csv)
    if args.dry_run:
        frame = pred[["IsOpen", "Horizon"]]
        reference = pred[["Id", "ActualOrderVolume"]].rename(columns={"ActualOrderVolume": "OrderVolume"})
    else:
        if args.frame is None or args.reference is None:
            raise SystemExit("--frame and --reference are required unless --dry-run")
        frame = pd.read_csv(args.frame, usecols=["Id", "IsOpen", "Horizon"])
        reference = pd.read_csv(args.reference)
    sample_ids = pd.read_csv(args.sample_submission)["Id"].to_numpy() if args.sample_submission else None
    base = DRY_RUN_LIMITS if args.dry_run else {"max_mean": MAX_MEAN_LOG_DIFF, "max_horizon": MAX_HORIZON_MEAN_LOG_DIFF, "min_corr": MIN_CORRELATION}
    limits = {"max_mean": args.max_mean_diff if args.max_mean_diff is not None else base["max_mean"],
              "max_horizon": args.max_horizon_diff if args.max_horizon_diff is not None else base["max_horizon"],
              "min_corr": args.min_corr if args.min_corr is not None else base["min_corr"]}
    checks, summary = run_checks(pred, frame, reference, sample_ids, expected_rows=args.expected_rows, **limits)
    if not args.dry_run and "Id" in frame and len(frame) == len(pred) and not np.array_equal(frame["Id"].to_numpy(), pred["Id"].to_numpy()):
        checks.append(Check("id_order_vs_test_frame", False, "prediction Id order differs from the test feature frame"))
    elif not args.dry_run:
        checks.append(Check("id_order_vs_test_frame", True, "prediction Id order equals the test feature frame"))
    _print(checks, summary)
    ok = all(c.passed for c in checks)
    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    if args.out:
        args.out.write_text(json.dumps({"passed": ok, "checks": [asdict(c) for c in checks], "distribution": summary}, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
