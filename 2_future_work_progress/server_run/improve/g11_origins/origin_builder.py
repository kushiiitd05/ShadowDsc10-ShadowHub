"""G11: training-origin construction helpers (pure functions) plus a small CLI that builds extra origin tables.

Pure part (no I/O, no mutation of inputs): origin date grids, block completeness, label-window purge, season windows,
stale-row filtering, origin-set selection, duplicate statistics, log-space blending.

CLI (needs the project's src.eda_pipeline on sys.path, i.e. run with --root <proj_root>):
    python origin_builder.py plan  --root PROJ --data G11_DATA
    python origin_builder.py build --root PROJ --data G11_DATA --kinds summer,new7 [--fold5]
    python origin_builder.py manifest --root PROJ --data G11_DATA

Layout under G11_DATA: origins/ (parquet; base files are symlinks to the project's own files), folds/ (fold 5 csv.gz),
plan.csv, manifest_all.csv (columns origin,target_start,target_end,rows,path,kind).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HORIZON = 42
FULL_BLOCK_ROWS = 46_830
N_HUBS = 1_115
GAP_START, GAP_END = pd.Timestamp("2014-07-01"), pd.Timestamp("2014-12-31")
FOLD_ORIGINS = ("2015-01-02", "2015-02-13", "2015-03-27", "2015-05-08")
FOLD5_ORIGIN = pd.Timestamp("2014-06-19")
KINDS = ("base", "summer", "new7")
SET_KINDS: dict[str, tuple[str, ...]] = {
    "base": ("base",),
    "summer": ("base", "summer"),
    "dense7": ("base", "new7"),
}


# ----------------------------------------------------------------------------- origin dates / blocks
def origin_grid(first: pd.Timestamp, last: pd.Timestamp, spacing_days: int) -> list[pd.Timestamp]:
    """Origins first, first+spacing, ... up to and including `last`."""
    if spacing_days <= 0:
        raise ValueError("spacing_days must be positive")
    first, last = pd.Timestamp(first), pd.Timestamp(last)
    if last < first:
        return []
    return list(pd.date_range(first, last, freq=f"{spacing_days}D"))


def block_dates(origin: pd.Timestamp, horizon: int = HORIZON) -> tuple[pd.Timestamp, pd.Timestamp]:
    origin = pd.Timestamp(origin)
    return origin + pd.Timedelta(days=1), origin + pd.Timedelta(days=horizon)


def block_row_count(observed: pd.DataFrame, origin: pd.Timestamp, horizon: int = HORIZON) -> int:
    """Number of observed (HubID, Date) rows inside the origin's label window."""
    start, end = block_dates(origin, horizon)
    return int(observed["Date"].between(start, end).sum())


def is_complete_block(observed: pd.DataFrame, origin: pd.Timestamp, horizon: int = HORIZON) -> bool:
    """Same test as the project pipeline: 1115 hubs x 42 dates, 46,830 rows."""
    start, end = block_dates(origin, horizon)
    block = observed.loc[observed["Date"].between(start, end)]
    return len(block) == FULL_BLOCK_ROWS and block["HubID"].nunique() == N_HUBS and block["Date"].nunique() == horizon


def plan_origins(observed: pd.DataFrame, existing: set[pd.Timestamp], first: pd.Timestamp, last: pd.Timestamp) -> pd.DataFrame:
    """Candidate origins on the 7-day grid with their block status and the kind they would get.

    kind: base (already built, complete), new7 (complete, off the 14-day grid), summer (incomplete block, on the 14-day grid),
    skip_incomplete_7 (incomplete and only on the 7-day grid: not used), skip_empty (no label rows at all).
    """
    grid14 = set(origin_grid(first, last, 14))
    rows = []
    for origin in origin_grid(first, last, 7):
        n = block_row_count(observed, origin)
        complete = is_complete_block(observed, origin)
        if origin in existing:
            kind = "base"
        elif complete:
            kind = "new7"
        elif n == 0:
            kind = "skip_empty"
        elif origin in grid14:
            kind = "summer"
        else:
            kind = "skip_incomplete_7"
        rows.append({"origin": origin, "rows_observed": n, "complete": complete, "kind": kind})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- eligibility / purge / season
def eligible_origins(origins: pd.Series, validation_origin: pd.Timestamp, horizon: int = HORIZON) -> pd.Series:
    """Harness rule: an origin is usable if its label window ends on or before the validation origin."""
    return (pd.to_datetime(origins) + pd.Timedelta(days=horizon)) <= pd.Timestamp(validation_origin)


def purge_mask(origins: pd.Series, holdout_origin: pd.Timestamp, horizon: int = HORIZON) -> pd.Series:
    """True for origins that may train: label windows disjoint from the holdout window (|o - h| >= horizon).

    For origins older than the holdout this equals the harness rule o + horizon < h + 1.
    """
    gap = (pd.to_datetime(origins) - pd.Timestamp(holdout_origin)).abs()
    return gap >= pd.Timedelta(days=horizon)


def circular_day_distance(a: pd.Series | pd.Timestamp, b: pd.Timestamp, year_days: float = 365.25) -> pd.Series | float:
    """Calendar distance in days between dates, ignoring the year (wraps around the new year)."""
    delta = (pd.to_datetime(a) - pd.Timestamp(b)) / pd.Timedelta(days=1)
    delta = np.mod(np.asarray(delta, dtype=float), year_days)
    out = np.minimum(delta, year_days - delta)
    return pd.Series(out, index=getattr(a, "index", None)) if isinstance(a, pd.Series) else float(out)


def season_mask(origins: pd.Series, center: pd.Timestamp, half_width_days: int = 45) -> pd.Series:
    """True for origins within +/- half_width_days of `center` on the calendar circle."""
    dist = circular_day_distance(pd.to_datetime(origins), center)
    return pd.Series(np.asarray(dist) <= half_width_days, index=origins.index)


def season_window_for_test(test_origin: pd.Timestamp, half_width_days: int = 45) -> tuple[str, str]:
    """Calendar window (month-day .. month-day) of origins used for the real test window."""
    c = pd.Timestamp(test_origin)
    return ((c - pd.Timedelta(days=half_width_days)).strftime("%m-%d"), (c + pd.Timedelta(days=half_width_days)).strftime("%m-%d"))


def reference_holdout(manifest: pd.DataFrame, validation_origin: pd.Timestamp, kinds: tuple[str, ...] = ("base",)) -> pd.Timestamp:
    """Chronologically last eligible origin among `kinds` (the harness's early-stopping holdout for the base set)."""
    pool = manifest[manifest["kind"].isin(kinds) & eligible_origins(manifest["origin"], validation_origin)]
    if pool.empty:
        raise ValueError(f"no eligible origin before {pd.Timestamp(validation_origin).date()}")
    return pd.Timestamp(pool["origin"].max())


def select_origin_set(manifest: pd.DataFrame, name: str, validation_origin: pd.Timestamp) -> pd.DataFrame:
    """Rows of the manifest usable for a fold under set `name` (base | summer | dense7); sorted by origin."""
    if name not in SET_KINDS:
        raise KeyError(f"unknown origin set: {name}")
    keep = manifest["kind"].isin(SET_KINDS[name]) & eligible_origins(manifest["origin"], validation_origin)
    return manifest.loc[keep].sort_values("origin").reset_index(drop=True)


def split_train_hold(frame_origins: pd.Series, holdout_origin: pd.Timestamp, horizon: int = HORIZON) -> tuple[pd.Series, pd.Series]:
    """Boolean masks (train, hold) over rows given each row's origin; train rows are purged on both sides of the holdout."""
    origins = pd.to_datetime(frame_origins)
    hold = origins == pd.Timestamp(holdout_origin)
    return purge_mask(origins, holdout_origin, horizon), hold


def season_pool(manifest: pd.DataFrame, validation_origin: pd.Timestamp, half_width_days: int = 45) -> pd.DataFrame:
    """Base-set eligible origins within the calendar season of the validation origin (member's own pool)."""
    pool = select_origin_set(manifest, "base", validation_origin)
    return pool.loc[season_mask(pool["origin"], validation_origin, half_width_days)].reset_index(drop=True)


# ----------------------------------------------------------------------------- row filters / blending / stats
def drop_stale_rows(frame: pd.DataFrame, max_stale_days: int = 7) -> pd.DataFrame:
    """Drop rows of hubs with no observation in the `max_stale_days` before the origin (gap hubs re-appearing after the closure)."""
    return frame.loc[frame["HistoryDaysSinceLastObservation"].le(max_stale_days)].reset_index(drop=True)


def log_blend(base_pred: np.ndarray, other_pred: np.ndarray, weight: float) -> np.ndarray:
    """expm1((1-w) log1p(base) + w log1p(other)); w is the weight of `other`."""
    if not 0.0 <= weight <= 1.0:
        raise ValueError("weight must be in [0, 1]")
    lb, lo = np.log1p(np.clip(base_pred, 0, None)), np.log1p(np.clip(other_pred, 0, None))
    return np.expm1((1.0 - weight) * lb + weight * lo)


def duplication_stats(origins: list[pd.Timestamp], horizon: int = HORIZON) -> dict[str, float]:
    """How many origins cover the same target day on average, and mean pairwise window overlap of neighbouring origins."""
    ds = sorted(pd.Timestamp(o) for o in origins)
    if not ds:
        return {"origins": 0, "mean_origins_per_target_day": 0.0, "neighbour_overlap_frac": 0.0}
    days = pd.concat([pd.Series(pd.date_range(*block_dates(o, horizon))) for o in ds])
    gaps = np.diff(np.array(ds, dtype="datetime64[D]")).astype(int)
    overlap = np.clip(1 - gaps / horizon, 0, 1) if len(gaps) else np.array([0.0])
    return {"origins": len(ds), "mean_origins_per_target_day": float(days.value_counts().mean()),
            "neighbour_overlap_frac": float(overlap.mean())}


# ----------------------------------------------------------------------------- building (needs the project pipeline)
def build_frame_incomplete(ep, train: pd.DataFrame, future: pd.DataFrame, metadata: pd.DataFrame, origin: pd.Timestamp) -> pd.DataFrame:
    """Mirror of eda_pipeline.build_forecast_frame (training source) that accepts a block with missing hub-days.

    Only the completeness assertion differs: rows absent from the raw panel stay absent (nothing is padded or imputed),
    and schedule-phase features see only the visible days. Verified identical to the original on complete blocks (see tests).
    """
    origin = pd.Timestamp(origin)
    start, end = block_dates(origin)
    block = future.loc[future["Date"].between(start, end)].copy()
    if block.empty:
        raise ValueError(f"empty label block at {origin.date()}")
    actual_columns = [c for c in ("OrderVolume", "AppSessions") if c in block]
    actuals = block[["HubID", "Date", *actual_columns]].copy()
    known = block.drop(columns=actual_columns, errors="ignore")
    state = ep.historical_state(train, origin)
    known = ep._base_preprocess(known, metadata)
    known["ForecastOrigin"] = origin
    known["Horizon"] = (known["Date"] - origin).dt.days.astype("int8")
    known["ForecastWeek"] = ((known["Horizon"] - 1) // 7 + 1).astype("int8")
    known = ep.add_target_aligned_lags(known, train, origin)
    cols = [*ep.SCHEDULE_COLUMNS, "Weekday", "RegionalHoliday"]
    schedule_history = train.loc[train["Date"].le(origin), ["HubID", "Date", *cols]]
    schedule_future = block[["HubID", "Date", *cols]]
    visible = ep._visible_future_schedule_features(schedule_history, schedule_future, origin)
    known = known.merge(visible, on=["HubID", "Date"], how="left", validate="one_to_one")
    known = known.merge(state, on="HubID", how="left", validate="many_to_one")
    dow, promo, dow_promo = ep._conditional_history(train, origin)
    known = known.merge(dow, on=["HubID", "Weekday"], how="left", validate="many_to_one")
    known = known.merge(promo, on=["HubID", "PromoActive"], how="left", validate="many_to_one")
    known = known.merge(dow_promo, on=["HubID", "Weekday", "PromoActive"], how="left", validate="many_to_one")
    renamed = actuals.rename(columns={c: f"Actual{c}" for c in actual_columns})
    known = known.merge(renamed, on=["HubID", "Date"], how="left", validate="one_to_one")
    return known.sort_values(["Date", "HubID"]).reset_index(drop=True)


def _load_pipeline(root: Path):
    sys.path.insert(0, str(root))
    from src import eda_pipeline as ep  # noqa: WPS433
    return ep


def _origin_file(data: Path, origin: pd.Timestamp) -> Path:
    return data / "origins" / f"origin_{pd.Timestamp(origin).date()}.parquet"


def cmd_plan(root: Path, data: Path) -> pd.DataFrame:
    ep = _load_pipeline(root)
    train = pd.read_csv(ep.Paths(root).raw / ep.RAW_FILES["train"], usecols=["HubID", "Date"], parse_dates=["Date"])
    base = pd.read_csv(root / "data/processed/training_origin_manifest.csv", parse_dates=["origin"])
    first = train["Date"].min() + pd.Timedelta(days=180)
    last = train["Date"].max() - pd.Timedelta(days=HORIZON)
    plan = plan_origins(train, set(base["origin"]), first, last)
    data.mkdir(parents=True, exist_ok=True)
    plan.to_csv(data / "plan.csv", index=False)
    print(plan["kind"].value_counts().to_string())
    return plan


def cmd_build(root: Path, data: Path, kinds: list[str], origins: list[str], fold5: bool, max_stale: int) -> None:
    ep = _load_pipeline(root)
    frames = ep.load_raw(ep.Paths(root))
    train, metadata = frames["train"], frames["metadata"]
    (data / "origins").mkdir(parents=True, exist_ok=True)
    (data / "folds").mkdir(parents=True, exist_ok=True)
    plan = pd.read_csv(data / "plan.csv", parse_dates=["origin"])
    todo = list(pd.to_datetime(origins)) if origins else list(plan.loc[plan["kind"].isin(kinds), "origin"])
    for origin in todo:
        out = _origin_file(data, origin)
        if out.exists():
            continue
        frame = build_frame_incomplete(ep, train, train, metadata, origin)
        n0 = len(frame)
        frame = drop_stale_rows(frame, max_stale)
        frame.to_parquet(out, index=False, compression="zstd")
        print(f"built {origin.date()} rows {len(frame)} (dropped stale {n0 - len(frame)})", flush=True)
    if fold5:
        out = data / "folds" / f"fold_5_{FOLD5_ORIGIN.date()}.csv.gz"
        if not out.exists():
            frame = build_frame_incomplete(ep, train, train, metadata, FOLD5_ORIGIN)
            frame.to_csv(out, index=False, compression="gzip")
            print(f"built fold 5 rows {len(frame)} cols {frame.shape[1]}", flush=True)


def cmd_manifest(root: Path, data: Path) -> pd.DataFrame:
    base = pd.read_csv(root / "data/processed/training_origin_manifest.csv", parse_dates=["origin", "target_start", "target_end"])
    (data / "origins").mkdir(parents=True, exist_ok=True)
    rows = []
    for r in base.itertuples():
        link = _origin_file(data, r.origin)
        if not link.exists() and not link.is_symlink():
            os.symlink(root / r.path, link)
        rows.append({"origin": r.origin, "rows": int(r.rows), "kind": "base"})
    plan = pd.read_csv(data / "plan.csv", parse_dates=["origin"])
    for r in plan[plan["kind"].isin(["summer", "new7"])].itertuples():
        f = _origin_file(data, r.origin)
        if f.exists():
            rows.append({"origin": r.origin, "rows": len(pd.read_parquet(f, columns=["HubID"])), "kind": r.kind})
    out = pd.DataFrame(rows).sort_values("origin").reset_index(drop=True)
    out["target_start"] = out["origin"] + pd.Timedelta(days=1)
    out["target_end"] = out["origin"] + pd.Timedelta(days=HORIZON)
    out["path"] = [f"origins/origin_{o.date()}.parquet" for o in out["origin"]]
    out.to_csv(data / "manifest_all.csv", index=False)
    print(out.groupby("kind")["rows"].agg(["count", "sum"]).to_string())
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=("plan", "build", "manifest"))
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--kinds", default="summer,new7")
    ap.add_argument("--origins", default="", help="explicit comma list of origin dates (overrides --kinds)")
    ap.add_argument("--fold5", action="store_true")
    ap.add_argument("--max-stale-days", type=int, default=7)
    a = ap.parse_args()
    root, data = a.root.resolve(), a.data.resolve()
    if a.command == "plan":
        cmd_plan(root, data)
    elif a.command == "build":
        cmd_build(root, data, [k for k in a.kinds.split(",") if k], [o for o in a.origins.split(",") if o], a.fold5, a.max_stale_days)
    else:
        cmd_manifest(root, data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
