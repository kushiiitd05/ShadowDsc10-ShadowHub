"""Origin-anchored extra features for the forecasting frames. Every feature at origin o uses data dated <= o only.

Column groups (all columns start with `g3_`): trend, weekday_promo, streaks, region.
Public API: add_features, add_features_multi, find_duplicate_columns, confirm_duplicates, select_group_columns.
"""
from __future__ import annotations

import warnings
from typing import Iterable

import numpy as np
import pandas as pd

from feature_panel import Panel, build_panel

ALL_GROUPS = ("trend", "weekday_promo", "streaks", "region")
GROUP_PREFIX = {"trend": "g3_trend_", "weekday_promo": "g3_dow_", "streaks": "g3_streak_", "region": "g3_region_"}
SLOPE_DAYS = 28          # slopes are reported as log-change per 28 days
DOW_WINDOW = 56          # last 8 weeks
CLOSURE_WINDOW = 90


# ------------------------------------------------------------------ helpers
def select_group_columns(columns: Iterable[str], groups: Iterable[str]) -> list[str]:
    prefixes = tuple(GROUP_PREFIX[g] for g in groups)
    return [c for c in columns if c.startswith(prefixes)]


def _log_open(panel: Panel) -> np.ndarray:
    """log1p(orders) on open, positive-volume days; NaN elsewhere (closed, open-but-zero, missing rows)."""
    keep = (panel.is_open == 1) & (panel.orders > 0)
    return np.where(keep, np.log1p(np.where(keep, panel.orders, 0.0)), np.nan)


def _nanmedian(values: np.ndarray, axis: int = 1) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(values, axis=axis)


def _trailing_true(mask: np.ndarray) -> np.ndarray:
    """Length of the run of True ending at the last column, per row."""
    rev = mask[:, ::-1]
    return np.where(rev.all(axis=1), rev.shape[1], np.argmin(rev, axis=1)).astype(float)


def _days_since_true(mask: np.ndarray) -> np.ndarray:
    """Days from the last column back to the latest True (0 if the last column); NaN if never True."""
    rev = mask[:, ::-1]
    return np.where(rev.any(axis=1), np.argmax(rev, axis=1), np.nan)


def _run_starts(mask: np.ndarray, window: int) -> np.ndarray:
    """Number of False->True transitions whose True day lies in the last `window` columns."""
    padded = np.concatenate([np.zeros((mask.shape[0], 1), bool), mask], axis=1)
    starts = padded[:, 1:] & ~padded[:, :-1]
    return starts[:, -window:].sum(axis=1).astype(float)


# ------------------------------------------------------------------ (1) trend
def ridge_trend(y: np.ndarray, window: int, ridge_frac: float, min_points: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-row ridge line over the last `window` columns, x = day offset from the last column (<= 0).

    Returns (slope per day, fitted level at the last column). The ridge penalty acts on the slope only and equals
    `ridge_frac` times the x-variance mass of a full window; rows with < min_points finite values give NaN.
    """
    seg = y[:, -window:]
    x = np.arange(seg.shape[1], dtype=float) - (seg.shape[1] - 1)
    m = np.isfinite(seg)
    n = m.sum(axis=1)
    ys = np.where(m, seg, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        xbar, ybar = (m * x).sum(axis=1) / n, ys.sum(axis=1) / n
        sxx = (m * x ** 2).sum(axis=1) - n * xbar ** 2
        sxy = (ys * x).sum(axis=1) - n * xbar * ybar
        slope = sxy / (sxx + ridge_frac * window ** 3 / 12.0)
        level = ybar - slope * xbar
    ok = n >= min_points
    return np.where(ok, slope, np.nan), np.where(ok, level, np.nan)


def trend_features(panel: Panel, ridge_frac: float = 0.02) -> pd.DataFrame:
    y = _log_open(panel)
    out = {"HubID": panel.hubs}
    for window, min_points in ((91, 10), (365, 30)):
        slope, level = ridge_trend(y, window, ridge_frac, min_points)
        out[f"g3_trend_slope{window}"] = slope * SLOPE_DAYS
        out[f"g3_trend_level{window}"] = level
    return pd.DataFrame(out)


# ------------------------------------------------------------------ (2) weekday x promo
def _weekday_axis(panel: Panel) -> np.ndarray:
    return panel.dates.day_of_week.to_numpy() + 1


def weekday_promo_tables(panel: Panel) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Last-8-week medians of log orders: per (hub, weekday) and, when promo exists, per (hub, weekday, promo)."""
    y = _log_open(panel)[:, -DOW_WINDOW:]
    weekday = _weekday_axis(panel)[-DOW_WINDOW:]
    promo = None if panel.promo is None else panel.promo[:, -DOW_WINDOW:]
    hub_med = _nanmedian(y)
    dow_rows, promo_rows = [], []
    for w in range(1, 8):
        cols = np.where(weekday == w)[0]
        sub = y[:, cols]
        med = _nanmedian(sub)
        dow_rows.append(pd.DataFrame({"HubID": panel.hubs, "Weekday": w, "g3_dow_hub_median8w": hub_med,
                                      "g3_dow_median8w": med, "g3_dow_n8w": np.isfinite(sub).sum(axis=1),
                                      "g3_dow_rel8w": med - hub_med}))
        if promo is None:
            continue
        for p in (0, 1):
            sel = np.where(promo[:, cols] == p, sub, np.nan)
            pmed = _nanmedian(sel)
            promo_rows.append(pd.DataFrame({"HubID": panel.hubs, "Weekday": w, "PromoActive": p,
                                            "g3_dow_promo_median8w": pmed, "g3_dow_promo_n8w": np.isfinite(sel).sum(axis=1),
                                            "g3_dow_promo_rel8w": pmed - hub_med}))
    return pd.concat(dow_rows, ignore_index=True), (pd.concat(promo_rows, ignore_index=True) if promo_rows else None)


# ------------------------------------------------------------------ (3)+(4) streaks
def closure_features(panel: Panel) -> dict[str, np.ndarray]:
    closed = panel.is_open == 0
    seg = closed[:, -CLOSURE_WINDOW:]
    return {"g3_streak_close_since_last": _days_since_true(closed), "g3_streak_close_run_len": _trailing_true(closed),
            "g3_streak_close_runs90": _run_starts(closed, CLOSURE_WINDOW), "g3_streak_close_days90": seg.sum(axis=1).astype(float)}


def promo_streak_features(panel: Panel) -> dict[str, np.ndarray]:
    if panel.promo is None:
        return {}
    on = panel.promo == 1
    year = panel.promo[:, -365:]
    seen = np.isfinite(year).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        share = np.where(seen > 0, (year == 1).sum(axis=1) / seen, np.nan)
    return {"g3_streak_promo_run_len": _trailing_true(on), "g3_streak_promo_since_last": _days_since_true(on),
            "g3_streak_promo_days28": on[:, -28:].sum(axis=1).astype(float),
            "g3_streak_promo_days91": on[:, -91:].sum(axis=1).astype(float), "g3_streak_promo_share365": share}


def streak_features(panel: Panel) -> pd.DataFrame:
    return pd.DataFrame({"HubID": panel.hubs, **closure_features(panel), **promo_streak_features(panel)})


# ------------------------------------------------------------------ (5) region / format proxy
def region_features(panel: Panel) -> pd.DataFrame:
    """Hub 91-day median log level minus the median of that level over hubs sharing the grouping value."""
    level = pd.Series(_nanmedian(_log_open(panel)[:, -91:]))
    out = {"HubID": panel.hubs}
    for name, codes in panel.groups.items():
        series = pd.Series(np.where(codes >= 0, level, np.nan))
        group_median = series.groupby(codes).transform("median")
        out[f"g3_region_{name}_rel91"] = (series - group_median).to_numpy()
    return pd.DataFrame(out)


# ------------------------------------------------------------------ joiner
def _lookup(table: pd.DataFrame, keys: list[str], frame: pd.DataFrame) -> pd.DataFrame:
    wanted = pd.MultiIndex.from_frame(frame[keys]) if len(keys) > 1 else pd.Index(frame[keys[0]])
    found = table.set_index(keys).reindex(wanted)
    found.index = frame.index
    return found.astype("float32")


def features_for_origin(panel: Panel, frame: pd.DataFrame, groups: tuple[str, ...] = ALL_GROUPS) -> pd.DataFrame:
    """Extra columns for `frame` rows; `panel` must already end at the origin (see Panel.upto)."""
    parts: list[pd.DataFrame] = []
    hub_tables = []
    if "trend" in groups:
        hub_tables.append(trend_features(panel))
    if "streaks" in groups:
        hub_tables.append(streak_features(panel))
    if "region" in groups and panel.groups:
        hub_tables.append(region_features(panel))
    for table in hub_tables:
        parts.append(_lookup(table, ["HubID"], frame))
    if "weekday_promo" in groups:
        dow, promo = weekday_promo_tables(panel)
        parts.append(_lookup(dow, ["HubID", "Weekday"], frame))
        if promo is not None and "PromoActive" in frame.columns:
            parts.append(_lookup(promo, ["HubID", "Weekday", "PromoActive"], frame))
    return pd.concat(parts, axis=1) if parts else pd.DataFrame(index=frame.index)


def _check(frame: pd.DataFrame, groups: Iterable[str]) -> None:
    unknown = set(groups) - set(ALL_GROUPS)
    if unknown:
        raise ValueError(f"unknown feature groups: {sorted(unknown)}")
    need = {"HubID"} | ({"Weekday"} if "weekday_promo" in groups else set())
    if need - set(frame.columns):
        raise ValueError(f"frame lacks columns: {sorted(need - set(frame.columns))}")


def _attach(frame: pd.DataFrame, extra: pd.DataFrame) -> pd.DataFrame:
    clash = set(extra.columns) & set(frame.columns)
    if clash:
        raise ValueError(f"frame already has feature columns: {sorted(clash)[:5]}")
    kwargs = {} if int(pd.__version__.split(".")[0]) >= 3 else {"copy": False}   # avoid duplicating the big frame
    return pd.concat([frame, extra], axis=1, **kwargs)


def add_features(frame: pd.DataFrame, raw_history: pd.DataFrame | Panel, origin: pd.Timestamp,
                 groups: tuple[str, ...] = ALL_GROUPS) -> pd.DataFrame:
    """Return a new frame with the extra columns for one forecast origin. Rows dated after `origin` are dropped first."""
    _check(frame, groups)
    panel = raw_history if isinstance(raw_history, Panel) else build_panel(raw_history[pd.to_datetime(raw_history["Date"]) <= origin])
    return _attach(frame, features_for_origin(panel.upto(origin), frame, tuple(groups)))


def add_features_multi(frame: pd.DataFrame, raw_history: pd.DataFrame | Panel, origin_col: str = "_origin",
                       groups: tuple[str, ...] = ALL_GROUPS) -> pd.DataFrame:
    """Like add_features for a stacked frame with one origin per row (column `origin_col`); index must be unique."""
    _check(frame, groups)
    if not frame.index.is_unique:
        raise ValueError("frame index must be unique")
    panel = raw_history if isinstance(raw_history, Panel) else build_panel(raw_history)
    pieces = []
    for origin, rows in frame.groupby(origin_col, sort=True):
        pieces.append(features_for_origin(panel.upto(pd.Timestamp(origin)), rows, tuple(groups)))
    return _attach(frame, pd.concat(pieces).reindex(frame.index))


# ------------------------------------------------------------------ (6) exact duplicate columns
def _canonical(series: pd.Series) -> tuple[str, np.ndarray]:
    if pd.api.types.is_datetime64_any_dtype(series):
        return "t", series.astype("datetime64[ns]").astype("int64").to_numpy()
    if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
        arr = series.to_numpy(dtype=float, na_value=np.nan) + 0.0   # +0.0 folds -0.0 into 0.0
        return "n", np.where(np.isnan(arr), np.nan, arr)
    return "s", series.astype("string").to_numpy(dtype=object, na_value="\x00NA")


def _same(a: tuple[str, np.ndarray], b: tuple[str, np.ndarray]) -> bool:
    if a[0] != b[0] or len(a[1]) != len(b[1]):
        return False
    return bool(np.array_equal(a[1], b[1], equal_nan=True)) if a[0] == "n" else bool(np.array_equal(a[1], b[1]))


def _is_constant(canon: tuple[str, np.ndarray]) -> bool:
    kind, arr = canon
    if len(arr) < 2:
        return True
    if kind == "n":
        return bool(((arr == arr[0]) | (np.isnan(arr) & np.isnan(arr[0]))).all())
    return bool((arr == arr[0]).all())


def find_duplicate_columns(frame: pd.DataFrame, columns: list[str] | None = None, ignore_constant: bool = True) -> dict[str, str]:
    """Map each exact duplicate column -> the first column (in order) holding identical values (NaN == NaN).

    Numeric dtypes are compared by value (int 1 == float 1.0); constants are skipped by default because two constant
    columns are only coincidentally equal. Buckets by a value hash, then verifies with a full comparison.
    """
    names = list(frame.columns if columns is None else columns)
    buckets: dict[tuple, list[tuple[str, tuple[str, np.ndarray]]]] = {}
    found: dict[str, str] = {}
    for name in names:
        canon = _canonical(frame[name])
        if ignore_constant and _is_constant(canon):
            continue
        key = (canon[0], len(canon[1]), int(pd.util.hash_array(canon[1]).sum()))
        match = next((kept for kept, kept_canon in buckets.get(key, []) if _same(canon, kept_canon)), None)
        if match is None:
            buckets.setdefault(key, []).append((name, canon))
        else:
            found[name] = match
    return found


def confirm_duplicates(pairs: dict[str, str], *frames: pd.DataFrame) -> dict[str, str]:
    """Keep only the (dup -> kept) pairs that are exactly equal in every given frame."""
    return {d: k for d, k in pairs.items()
            if all(d in f.columns and k in f.columns and _same(_canonical(f[d]), _canonical(f[k])) for f in frames)}
