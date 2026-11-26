"""TabPFN-TS on the REAL test window (same settings as the dev folds).

Forecast origin = last observed day 2015-06-19; forecast days 2015-06-20 .. 2015-07-31 (42 days, 1,115 hubs). The test frame is
the one final_retrain.py predicts (data/processed/test_forecast_features.csv.gz): it is read, never rebuilt, and the output keeps its
row order and Ids. Settings are those of the fold runs: TabPFN v2 backbone, 365-day context, closed days interpolated in the history,
known-future covariates (open / promo / holiday / school + calendar), closed forecast days forced to 0.

Usage (venv_nf, from the improve/g10_neural dir):
  python neural_testwindow.py <proj_root> <out_dir> [--context 365] [--impute 1]
Outputs in <out_dir>: test_pred_tabpfnts.csv (Id, OrderVolume in frame order), test_pred_tabpfnts.npy (same volumes), testwindow_manifest.json
(SHA-256 of both files, settings, origin, summary stats). The prediction file is the "blend file" accepted by final_retrain.py.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from neural_prep import KEYS, build_future_frame, history_window, volume_from_log  # noqa: E402
from tabpfn_members import predict_tabpfn_ts  # noqa: E402  (tabpfn itself is imported lazily inside)

HORIZON = 42
EXPECTED_ROWS = 46_830
EXPECTED_HUBS = 1_115
FRAME_FILE = "data/processed/test_forecast_features.csv.gz"
BASE_FILE = "data/processed/base_train.csv.gz"
BASE_COLUMNS = ["HubID", "Date", "OrderVolume", "IsOpen", "PromoActive", "RegionalHoliday", "SchoolClosureFlag"]
PRED_CSV, PRED_NPY, MANIFEST = "test_pred_tabpfnts.csv", "test_pred_tabpfnts.npy", "testwindow_manifest.json"


# ---------------------------------------------------------------- pure helpers (unit-tested)
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def check_test_frame(frame: pd.DataFrame, horizon: int = HORIZON, n_hubs: int = EXPECTED_HUBS, expected_rows: int = EXPECTED_ROWS) -> pd.Timestamp:
    """The frame must be the complete hubs x horizon grid with unique Ids and one forecast origin; returns that origin."""
    if len(frame) != expected_rows or frame["HubID"].nunique() != n_hubs or frame["Date"].nunique() != horizon:
        raise AssertionError(f"test frame is not a complete {n_hubs} x {horizon} grid ({len(frame)} rows)")
    if not frame["Id"].is_unique:
        raise AssertionError("test frame Ids are not unique")
    origins = pd.to_datetime(frame["ForecastOrigin"]).unique()
    if len(origins) != 1:
        raise AssertionError("test frame has more than one forecast origin")
    origin = pd.Timestamp(origins[0])
    if pd.to_datetime(frame["Date"]).min() != origin + pd.Timedelta(days=1):
        raise AssertionError("first forecast day is not origin + 1")
    return origin


def history_until(base: pd.DataFrame, origin: pd.Timestamp) -> pd.DataFrame:
    """History rows dated on or before the origin (new frame); raises when there are none."""
    return history_window(base, origin)


def assert_history_before_window(hist: pd.DataFrame, origin: pd.Timestamp) -> None:
    last = pd.to_datetime(hist["Date"]).max()
    if not last <= pd.Timestamp(origin):
        raise AssertionError(f"history reaches {last.date()}, after the forecast origin {pd.Timestamp(origin).date()}")


def assert_hubs_have_history(hist: pd.DataFrame, frame: pd.DataFrame) -> None:
    missing = sorted(set(frame["HubID"].astype(int)) - set(hist["HubID"].astype(int)))
    if missing:
        raise AssertionError(f"{len(missing)} test hubs have no history, e.g. {missing[:5]}")


def align_to_frame(forecast: pd.DataFrame, frame: pd.DataFrame, column: str = "log_pred") -> pd.DataFrame:
    """Log-space forecast (unique_id, ds, column) -> Id, OrderVolume in the frame's own row order.

    Joined on (HubID, Date) with a one-to-one check, never by position. Non-finite forecasts are refused (not zeroed); volumes are
    expm1, clipped at 0, and exactly 0 on closed days. Neither input is modified.
    """
    fc = pd.DataFrame({"HubID": forecast["unique_id"].astype(int).to_numpy(), "Date": pd.to_datetime(forecast["ds"]).to_numpy(),
                       column: forecast[column].to_numpy(dtype="float64")})
    if fc.duplicated(KEYS).any():
        raise ValueError("forecast has duplicate (HubID, Date) rows")
    rows = frame[["Id", "HubID", "Date", "IsOpen"]].copy()
    rows["Date"] = pd.to_datetime(rows["Date"])
    merged = rows.merge(fc, on=KEYS, how="left", validate="one_to_one")
    missing = merged[column].isna()
    if missing.any():
        # a NaN either means the key had no forecast row or the model returned NaN: distinguish for a clear message
        keys_present = rows.merge(fc[KEYS], on=KEYS, how="inner").shape[0]
        if keys_present < len(rows):
            raise ValueError(f"{int(missing.sum())} test rows have no forecast")
        raise ValueError(f"{int(missing.sum())} test rows have a non-finite forecast")
    if not np.isfinite(merged[column].to_numpy()).all():
        raise ValueError("forecast contains non-finite values")
    volume = volume_from_log(merged[column].to_numpy(), merged["IsOpen"].to_numpy())
    return pd.DataFrame({"Id": merged["Id"].to_numpy(), "OrderVolume": volume})


def validate_predictions(table: pd.DataFrame, frame: pd.DataFrame) -> None:
    """Hard checks on the output: layout, Id order equal to the frame, finite, non-negative, closed rows exactly 0."""
    if list(table.columns) != ["Id", "OrderVolume"] or len(table) != len(frame):
        raise AssertionError("prediction table layout or length differs from the test frame")
    if not np.array_equal(table["Id"].to_numpy(), frame["Id"].to_numpy()):
        raise AssertionError("prediction Id order differs from the test frame")
    values = table["OrderVolume"].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise AssertionError("predictions contain NaN or inf")
    if (values < 0).any():
        raise AssertionError("predictions contain negative values")
    if (values[frame["IsOpen"].to_numpy() == 0] != 0).any():
        raise AssertionError("predictions are non-zero on closed rows")


def write_outputs(table: pd.DataFrame, frame: pd.DataFrame, out_dir: Path, origin: pd.Timestamp, settings: dict, seconds: float) -> dict:
    """Write CSV + NPY + manifest (refuses to overwrite) and return the manifest dict."""
    validate_predictions(table, frame)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / PRED_CSV, out_dir / PRED_NPY, out_dir / MANIFEST]
    for p in paths:
        if p.exists():
            raise FileExistsError(f"{p.name} already exists in {out_dir}")
    table.to_csv(paths[0], index=False)
    np.save(paths[1], table["OrderVolume"].to_numpy(dtype="float64"))
    is_open = frame["IsOpen"].to_numpy() == 1
    values = table["OrderVolume"].to_numpy(dtype=float)
    manifest = {
        "member": "tabpfnts", "predictions_file": PRED_CSV, "predictions_sha256": sha256_file(paths[0]),
        "npy_file": PRED_NPY, "npy_sha256": sha256_file(paths[1]), "rows": int(len(table)), "hubs": int(frame["HubID"].nunique()),
        "forecast_origin": str(pd.Timestamp(origin).date()), "first_target_date": str(pd.to_datetime(frame["Date"]).min().date()),
        "last_target_date": str(pd.to_datetime(frame["Date"]).max().date()), "history_last_date": str(pd.Timestamp(origin).date()),
        "settings": settings, "closed_rows": int((~is_open).sum()), "mean_log1p_open": float(np.log1p(values[is_open]).mean()),
        "median_log1p_open": float(np.median(np.log1p(values[is_open]))), "predict_seconds": round(float(seconds), 1),
        "test_frame_sha256": settings.get("test_frame_sha256"),
    }
    paths[2].write_text(json.dumps(manifest, indent=2, default=str))
    return manifest


# ---------------------------------------------------------------- I/O + run
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("proj_root", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--context", type=int, default=365, help="context days per hub (same as the dev folds)")
    ap.add_argument("--impute", type=int, default=1, help="1: interpolate closed-day targets in the history (same as the dev folds)")
    return ap.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    root = args.proj_root.resolve()
    t0 = time.perf_counter()
    frame = pd.read_csv(root / FRAME_FILE, parse_dates=["Date", "ForecastOrigin"])
    origin = check_test_frame(frame, HORIZON, EXPECTED_HUBS, EXPECTED_ROWS)
    base = pd.read_csv(root / BASE_FILE, usecols=BASE_COLUMNS, parse_dates=["Date"])
    hist = history_until(base, origin)
    assert_history_before_window(hist, origin)
    assert_hubs_have_history(hist, frame)
    print(f"test window: origin {origin.date()}, {len(frame)} rows, {frame['HubID'].nunique()} hubs, history rows {len(hist)} "
          f"(last day {pd.to_datetime(hist['Date']).max().date()}), loaded in {time.perf_counter() - t0:.0f}s", flush=True)
    valid = frame[["HubID", "Date", "IsOpen", "PromoActive", "RegionalHoliday", "SchoolClosureFlag"]].copy()
    build_future_frame(valid, origin, HORIZON)  # fail fast on a wrong window before the 15-minute GPU run
    t1 = time.perf_counter()
    forecast = predict_tabpfn_ts(hist, valid, origin, args)
    seconds = time.perf_counter() - t1
    table = align_to_frame(forecast, frame)
    settings = {"backbone": "TabPFN v2", "context": args.context, "impute_closed": args.impute, "horizon": HORIZON,
                "covariates": ["IsOpen", "PromoActive", "RegionalHoliday", "SchoolClosureFlag", "calendar"],
                "test_frame_sha256": sha256_file(root / FRAME_FILE)}
    manifest = write_outputs(table, frame, args.out_dir, origin, settings, seconds)
    print(json.dumps({k: manifest[k] for k in ("rows", "predictions_sha256", "mean_log1p_open", "predict_seconds")}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
