"""Holiday / calendar / Sunday / promo features for the cs02 recipe (stage g3c). Every value at origin o uses labels dated <= o and
schedule values (IsOpen, PromoActive, RegionalHoliday, SchoolClosureFlag) dated <= o + 42 days only.

Leakage guard: the label side is cut with `Panel.upto(origin)` (as in g3b); the schedule side is read from a window
[origin - 21 days, origin + WINDOW_DAYS] of the full calendar, so anything dated later is never touched. Schedules beyond the window
exist in the validation folds but not for the real test window, so they are treated as unknown in both.

Conventions
  * log orders = log1p(OrderVolume) on observed open days with positive volume (g3b convention); missing rows are never zeros.
  * RegionalHoliday is a code (0 none, 1 ordinary holiday, 2 Easter Friday/Monday, 3 Christmas 25/26 in the real file); any code > 0 is
    a holiday and the code is the holiday TYPE. Missing hub rows take the date's code when every observed hub agrees on it.
  * holiday (g3c_hh_*): only rows whose target date is a holiday for that hub, NaN otherwise. "any" = previous holiday days of any type,
    "type" = previous holidays with the same code as d, both on days the hub was open. mean / median / n of their log orders (n = 0
    when the hub never worked such a holiday; the statistics are then NaN), ratio = mean - the hub's mean log orders on non-holiday days
    with the same weekday as d in the last 365 days, type_last_log / type_last_days = log orders on and age (days before the origin)
    of the most recent same-code holiday the hub was open on.
  * calendar (g3c_cal_*): for events hol (code > 0), sch (SchoolClosureFlag == 1) and promo: number of event days in the Monday-Sunday week of d
    (_week), the 7 days before d (_prev7), the 7 days after d (_next7), days to the nearest event strictly before / after d (_dprev, _dnext,
    capped at 14, 14 when none is known). Days past origin + 42 are unknown and count as no event. closed_run_before / _after =
    consecutive scheduled-closed days (IsOpen == 0) directly before / after d (cap 14); first_after_closed = d is open and the day before
    was closed; last_before_closed = d is open and the day after is closed.
  * sunday (g3c_sun_*), NaN unless d is a Sunday: share of the hub's observed Sundays of the last 365 days it was open, mean log orders on its
    open Sundays, number of open Sundays.
  * promo (g3c_promo_*): ratio{w} = promo-day mean minus non-promo mean of log orders over the last w in (182, 365) days, computed per
    weekday and averaged over weekdays that have both kinds of day, weighted by the number of promo days (same weekday mix); n{w} = open
    promo days in the window.
Public API: ALL_GROUPS, GROUP_PREFIX, select_group_columns, Calendar, build_calendar, add_features, add_features_multi.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
for _path in (_HERE.parent, _HERE.parent / "g3_features", _HERE.parent / "g3b_features"):
    if str(_path) not in sys.path:
        sys.path.append(str(_path))

from feature_panel import Panel, build_panel  # noqa: E402
from g3b_features import _log_open, _nan_mean_count, _nan_median  # noqa: E402  (read-only reuse of the g3b building blocks)

ALL_GROUPS = ("holiday", "calendar", "sunday", "promo")
GROUP_PREFIX = {"holiday": "g3c_hh_", "calendar": "g3c_cal_", "sunday": "g3c_sun_", "promo": "g3c_promo_"}
SCHEDULE_COLUMNS = ("IsOpen", "PromoActive", "RegionalHoliday", "SchoolClosureFlag")
WINDOW_DAYS = 42              # forecast horizon: schedule dated after origin + 42 is unknown
LOOKBACK = 21                 # schedule columns kept before the origin (week start, 7-day and 14-day look-backs)
CAP = 14
BASELINE_DAYS = 365
SUNDAY_DAYS = 365
PROMO_WINDOWS = (182, 365)
EVENTS = {"hol": "holiday", "sch": "school", "promo": "promo"}
OUT_DTYPE = np.float32
_BIG = 10 ** 6


def select_group_columns(columns: Iterable[str], groups: Iterable[str]) -> list[str]:
    prefixes = tuple(GROUP_PREFIX[g] for g in groups)
    return [c for c in columns if c.startswith(prefixes)]


# ------------------------------------------------------------------ calendar object
@dataclass(frozen=True)
class Calendar:
    """Full-axis (hub x day) labels + schedule. Labels are only ever read through `panel.upto(origin)`."""
    panel: Panel                 # labels (orders), IsOpen and PromoActive on the full date axis
    holiday: np.ndarray          # (H, T) RegionalHoliday code, NaN = no row and no unanimous date code
    school: np.ndarray           # (H, T) SchoolClosureFlag, NaN = no row


def _dense_like(panel: Panel, data: pd.DataFrame, column: str) -> np.ndarray:
    out = np.full((len(panel.hubs), len(panel.dates)), np.nan)
    out[np.searchsorted(panel.hubs, data["HubID"].to_numpy()), (data["Date"] - panel.dates[0]).dt.days.to_numpy()] = data[column].to_numpy(float)
    return out


def _fill_unanimous(codes: np.ndarray) -> np.ndarray:
    """Cells without a row take the date's code when every observed hub carries the same one."""
    ok = np.isfinite(codes)
    with np.errstate(invalid="ignore"):
        lo, hi = np.where(ok, codes, np.inf).min(axis=0), np.where(ok, codes, -np.inf).max(axis=0)
    unanimous = ok.any(axis=0) & (lo == hi)
    return np.where(ok, codes, np.where(unanimous, lo, np.nan)[None, :])


def build_calendar(raw: pd.DataFrame) -> Calendar:
    """`raw`: HubID, Date, IsOpen, PromoActive, RegionalHoliday, SchoolClosureFlag (+ OrderVolume where known, + HubFormat/AssortmentTier).
    May contain schedule-only rows (the test window); rows without OrderVolume carry no label."""
    missing = {"HubID", "Date", "IsOpen", "PromoActive", "RegionalHoliday", "SchoolClosureFlag"} - set(raw.columns)
    if missing:
        raise ValueError(f"raw history lacks columns: {sorted(missing)}")
    data = raw.assign(Date=pd.to_datetime(raw["Date"]).dt.normalize()).drop_duplicates(["HubID", "Date"], keep="last")
    if "OrderVolume" not in data.columns:
        data = data.assign(OrderVolume=np.nan)
    panel = build_panel(data)
    return Calendar(panel, _fill_unanimous(_dense_like(panel, data, "RegionalHoliday")), _dense_like(panel, data, "SchoolClosureFlag"))


# ------------------------------------------------------------------ (A) holiday_history
def _hub_stats(y: np.ndarray, mask: np.ndarray, columns: np.ndarray) -> dict[str, np.ndarray]:
    """mean / median / n / last log / last age of y over `columns` where mask; all (H,)."""
    sel = np.where(mask, y[:, columns], np.nan)
    mean, n = _nan_mean_count(sel, 0, sel.shape[1])
    last_pos = np.where(np.isfinite(sel), np.arange(sel.shape[1])[None, :], -1).max(axis=1)
    has = last_pos >= 0
    rows = np.arange(len(y))
    last_log = np.where(has, sel[rows, np.maximum(last_pos, 0)], np.nan)
    last_col = columns[np.maximum(last_pos, 0)]
    return {"mean": mean, "median": _nan_median(sel), "n": n, "last_log": last_log, "last_col": np.where(has, last_col, -1)}


def _weekday_baseline(y: np.ndarray, hol: np.ndarray, weekdays: np.ndarray) -> np.ndarray:
    """(H, 7) mean log orders on non-holiday days of each weekday over the last BASELINE_DAYS columns."""
    lo = max(y.shape[1] - BASELINE_DAYS, 0)
    seg = np.where(hol[:, lo:] == 0, y[:, lo:], np.nan)
    wd = weekdays[lo:y.shape[1]]
    return np.stack([_nan_mean_count(seg[:, wd == w], 0, int((wd == w).sum()))[0] for w in range(7)], axis=1)


def holiday_tables(y: np.ndarray, hol: np.ndarray, weekdays: np.ndarray) -> dict:
    """Hub-level holiday statistics for the history `y`, `hol` (both (H, width), last column = origin)."""
    width = y.shape[1]
    columns = np.flatnonzero((hol > 0).any(axis=0))
    codes = np.unique(hol[:, columns][hol[:, columns] > 0]).astype(int) if len(columns) else np.zeros(0, dtype=int)
    any_stats = _hub_stats(y, hol[:, columns] > 0, columns)
    typed = [_hub_stats(y, hol[:, columns] == c, columns) for c in codes]
    stack = {k: np.stack([t[k] for t in typed], axis=1) if typed else np.zeros((len(y), 0)) for k in ("mean", "median", "n", "last_log", "last_col")}
    return {"codes": codes, "any": any_stats, "type": stack, "base": _weekday_baseline(y, hol, weekdays), "width": width}


def holiday_features(tables: dict, hub_idx: np.ndarray, matched: np.ndarray, code: np.ndarray, weekday: np.ndarray) -> dict[str, np.ndarray]:
    """Row columns for target rows with holiday `code` (NaN/0 = not a holiday) and weekday (0 = Monday)."""
    is_hol = matched & np.isfinite(code) & (code > 0)
    rows = np.where(is_hol, hub_idx, 0)
    base = tables["base"][rows, weekday]
    out: dict[str, np.ndarray] = {}
    for key in ("mean", "median"):
        out[f"g3c_hh_any_{key}"] = tables["any"][key][rows]
    out["g3c_hh_any_n"] = tables["any"]["n"][rows]
    out["g3c_hh_any_ratio"] = tables["any"]["mean"][rows] - base
    codes, wanted = tables["codes"], np.where(is_hol, code, -1)
    pos = np.minimum(np.searchsorted(codes, wanted), max(len(codes) - 1, 0))
    found = is_hol & (codes[pos] == wanted) if len(codes) else np.zeros(len(rows), dtype=bool)
    cell = (rows, np.where(found, pos, 0))

    def typed(name: str) -> np.ndarray:
        return np.where(found, tables["type"][name][cell], np.nan) if len(codes) else np.full(len(rows), np.nan)
    out["g3c_hh_type_mean"], out["g3c_hh_type_median"] = typed("mean"), typed("median")
    out["g3c_hh_type_n"] = np.where(found, typed("n"), 0.0)
    out["g3c_hh_type_ratio"] = typed("mean") - base
    out["g3c_hh_type_last_log"] = typed("last_log")
    last_col = typed("last_col")
    out["g3c_hh_type_last_days"] = np.where(np.isfinite(last_col) & (last_col >= 0), tables["width"] - 1 - last_col, np.nan)
    return {k: np.where(is_hol, v, np.nan) for k, v in out.items()}


# ------------------------------------------------------------------ (B) holiday_calendar
def _prefix(events: np.ndarray) -> np.ndarray:
    return np.concatenate([np.zeros((events.shape[0], 1)), np.cumsum(events, axis=1)], axis=1)


def _window_count(prefix: np.ndarray, rows: np.ndarray, start: np.ndarray, stop: np.ndarray) -> np.ndarray:
    """Number of events in columns [start, stop] inclusive (clipped to the slice)."""
    length = prefix.shape[1] - 1
    a, b = np.clip(start, 0, length), np.clip(stop + 1, 0, length)
    return prefix[rows, b] - prefix[rows, np.minimum(a, b)]


def event_features(events: np.ndarray, rows: np.ndarray, t: np.ndarray, weekday: np.ndarray, tag: str) -> dict[str, np.ndarray]:
    """Counts and distances for a (H, L) boolean event slice at slice columns t (rows are hub indices)."""
    length = events.shape[1]
    idx = np.arange(length)[None, :]
    prefix = _prefix(events)
    last = np.maximum.accumulate(np.where(events, idx, -_BIG), axis=1)
    nxt = np.minimum.accumulate(np.where(events, idx, _BIG)[:, ::-1], axis=1)[:, ::-1]
    before = np.where(t >= 1, t - last[rows, np.clip(t - 1, 0, length - 1)], _BIG)
    after = np.where(t + 1 < length, nxt[rows, np.clip(t + 1, 0, length - 1)] - t, _BIG)
    start = t - weekday
    return {f"g3c_cal_{tag}_week": _window_count(prefix, rows, start, start + 6),
            f"g3c_cal_{tag}_prev7": _window_count(prefix, rows, t - 7, t - 1),
            f"g3c_cal_{tag}_next7": _window_count(prefix, rows, t + 1, t + 7),
            f"g3c_cal_{tag}_dprev": np.minimum(before, CAP).astype(float), f"g3c_cal_{tag}_dnext": np.minimum(after, CAP).astype(float)}


def closed_run_features(closed: np.ndarray, is_open: np.ndarray, rows: np.ndarray, t: np.ndarray) -> dict[str, np.ndarray]:
    """Closed-run lengths directly before / after t from the scheduled IsOpen slice (missing rows break a run)."""
    length = closed.shape[1]
    idx = np.arange(length)[None, :]
    last_open = np.maximum.accumulate(np.where(~closed, idx, -1), axis=1)
    next_open = np.minimum.accumulate(np.where(~closed, idx, length)[:, ::-1], axis=1)[:, ::-1]
    before = np.where(t >= 1, (t - 1) - last_open[rows, np.clip(t - 1, 0, length - 1)], 0)
    after = np.where(t + 1 < length, next_open[rows, np.clip(t + 1, 0, length - 1)] - (t + 1), 0)
    before, after = np.minimum(before, CAP).astype(float), np.minimum(after, CAP).astype(float)
    now = is_open[rows, t]
    opened = now == 1
    return {"g3c_cal_closed_run_before": before, "g3c_cal_closed_run_after": after,
            "g3c_cal_first_after_closed": np.where(np.isfinite(now), (opened & (before >= 1)).astype(float), np.nan),
            "g3c_cal_last_before_closed": np.where(np.isfinite(now), (opened & (after >= 1)).astype(float), np.nan)}


def calendar_features(cal: Calendar, width: int, hub_idx: np.ndarray, matched: np.ndarray, d_abs: np.ndarray, weekday: np.ndarray) -> dict[str, np.ndarray]:
    lo, hi = max(width - 1 - LOOKBACK, 0), min(width - 1 + WINDOW_DAYS, len(cal.panel.dates) - 1)
    t = d_abs - lo
    valid = matched & (t >= 0) & (t <= hi - lo)
    rows, tt = np.where(valid, hub_idx, 0), np.where(valid, t, 0)
    with np.errstate(invalid="ignore"):
        slices = {"hol": cal.holiday[:, lo:hi + 1] > 0, "sch": cal.school[:, lo:hi + 1] == 1,
                  "promo": (cal.panel.promo[:, lo:hi + 1] == 1) if cal.panel.promo is not None else np.zeros_like(cal.holiday[:, lo:hi + 1], dtype=bool)}
    is_open = cal.panel.is_open[:, lo:hi + 1]
    out: dict[str, np.ndarray] = {}
    for tag, events in slices.items():
        out.update(event_features(events, rows, tt, weekday, tag))
    out.update(closed_run_features(is_open == 0, is_open, rows, tt))
    return {k: np.where(valid, v, np.nan) for k, v in out.items()}


# ------------------------------------------------------------------ (C) sunday_special
def sunday_hub_features(panel: Panel, y: np.ndarray, weekdays: np.ndarray) -> dict[str, np.ndarray]:
    width = y.shape[1]
    cols = np.arange(max(width - SUNDAY_DAYS, 0), width)
    cols = cols[weekdays[cols] == 6]
    observed = np.isfinite(panel.is_open[:, cols])
    opened = panel.is_open[:, cols] == 1
    n_obs, n_open = observed.sum(axis=1).astype(float), opened.sum(axis=1).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        share = np.where(n_obs > 0, n_open / np.maximum(n_obs, 1), np.nan)
    return {"g3c_sun_open_share": share, "g3c_sun_log_mean": _nan_mean_count(y[:, cols], 0, len(cols))[0], "g3c_sun_n_open": n_open}


# ------------------------------------------------------------------ (D) promo_hub_ratio
def promo_hub_features(panel: Panel, y: np.ndarray, weekdays: np.ndarray) -> dict[str, np.ndarray]:
    width, out = y.shape[1], {}
    for w in PROMO_WINDOWS:
        lo = max(width - w, 0)
        seg, promo, wd = y[:, lo:], panel.promo[:, lo:], weekdays[lo:width]
        total, weight = np.zeros(len(y)), np.zeros(len(y))
        for day in range(7):
            part = wd == day
            on, n_on = _nan_mean_count(np.where(promo[:, part] == 1, seg[:, part], np.nan), 0, int(part.sum()))
            off, n_off = _nan_mean_count(np.where(promo[:, part] == 0, seg[:, part], np.nan), 0, int(part.sum()))
            both = (n_on > 0) & (n_off > 0)
            total += np.where(both, n_on * (on - off), 0.0)
            weight += np.where(both, n_on, 0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[f"g3c_promo_ratio{w}"] = np.where(weight > 0, total / np.maximum(weight, 1), np.nan)
        out[f"g3c_promo_n{w}"] = np.nansum(np.where(promo == 1, np.isfinite(seg), False), axis=1).astype(float)
    return out


# ------------------------------------------------------------------ assembly
def _gather(table: np.ndarray, hub_idx: np.ndarray, matched: np.ndarray) -> np.ndarray:
    return np.where(matched, table[np.where(matched, hub_idx, 0)], np.nan)


def features_for_origin(cal: Calendar, origin: pd.Timestamp, hubs: np.ndarray, dates: pd.DatetimeIndex, groups: tuple[str, ...]) -> dict[str, np.ndarray]:
    """Feature columns (name -> (n,) float32) for rows (hub id, target date) at one origin."""
    panel = cal.panel.upto(origin)
    width = len(panel.dates)
    pos = np.searchsorted(panel.hubs, hubs)
    hub_idx = np.minimum(pos, len(panel.hubs) - 1)
    matched = panel.hubs[hub_idx] == hubs
    d_abs = ((dates - cal.panel.dates[0]).days).to_numpy().astype(np.int64)
    weekday = dates.day_of_week.to_numpy().astype(np.int64)
    known = (d_abs >= 0) & (d_abs <= min(width - 1 + WINDOW_DAYS, len(cal.panel.dates) - 1))
    cols: dict[str, np.ndarray] = {}
    needs_y = set(groups) & {"holiday", "sunday", "promo"}
    y = _log_open(panel) if needs_y else None
    weekdays = panel.dates.day_of_week.to_numpy()
    if "holiday" in groups:
        code = np.where(matched & known, cal.holiday[hub_idx, np.clip(d_abs, 0, cal.holiday.shape[1] - 1)], np.nan)
        cols.update(holiday_features(holiday_tables(y, cal.holiday[:, :width], weekdays), hub_idx, matched, code, weekday))
    if "calendar" in groups:
        cols.update(calendar_features(cal, width, hub_idx, matched, d_abs, weekday))
    if "sunday" in groups:
        is_sunday = weekday == 6
        cols.update({k: np.where(is_sunday, _gather(v, hub_idx, matched), np.nan) for k, v in sunday_hub_features(panel, y, weekdays).items()})
    if "promo" in groups:
        cols.update({k: _gather(v, hub_idx, matched) for k, v in promo_hub_features(panel, y, weekdays).items()})
    return {k: np.where(known, v, np.nan).astype(OUT_DTYPE) for k, v in sorted(cols.items())}


def _check(frame: pd.DataFrame, groups: Iterable[str]) -> None:
    unknown = set(groups) - set(ALL_GROUPS)
    if unknown:
        raise ValueError(f"unknown feature groups: {sorted(unknown)}")
    missing = {"HubID", "Date"} - set(frame.columns)
    if missing:
        raise ValueError(f"frame lacks columns: {sorted(missing)}")


def _attach(frame: pd.DataFrame, extra: pd.DataFrame) -> pd.DataFrame:
    clash = set(extra.columns) & set(frame.columns)
    if clash:
        raise ValueError(f"frame already has feature columns: {sorted(clash)[:5]}")
    kwargs = {} if int(pd.__version__.split(".")[0]) >= 3 else {"copy": False}
    return pd.concat([frame, extra], axis=1, **kwargs)


def _as_calendar(history: pd.DataFrame | Calendar) -> Calendar:
    return history if isinstance(history, Calendar) else build_calendar(history)


def add_features(frame: pd.DataFrame, history: pd.DataFrame | Calendar, origin: pd.Timestamp, groups: tuple[str, ...] = ALL_GROUPS) -> pd.DataFrame:
    """New frame with the g3c columns for one forecast origin (rows are target days after `origin`).
    `history` holds labels and schedule; labels dated after `origin` and schedule dated after origin + 42 days are ignored."""
    _check(frame, groups)
    origin = pd.Timestamp(origin)
    cols = features_for_origin(_as_calendar(history), origin, frame["HubID"].to_numpy(), pd.DatetimeIndex(pd.to_datetime(frame["Date"])), tuple(groups))
    return _attach(frame, pd.DataFrame(cols, index=frame.index))


def add_features_multi(frame: pd.DataFrame, history: pd.DataFrame | Calendar, origin_col: str = "_origin",
                       groups: tuple[str, ...] = ALL_GROUPS) -> pd.DataFrame:
    """Like add_features for a stacked frame with one origin per row (column `origin_col`); the index must be unique."""
    _check(frame, groups)
    if not frame.index.is_unique:
        raise ValueError("frame index must be unique")
    cal = _as_calendar(history)
    origins = pd.to_datetime(frame[origin_col]).to_numpy()
    hubs, dates = frame["HubID"].to_numpy(), pd.DatetimeIndex(pd.to_datetime(frame["Date"]))
    pieces = []
    for origin in np.unique(origins):
        pos = np.flatnonzero(origins == origin)
        cols = features_for_origin(cal, pd.Timestamp(origin), hubs[pos], dates[pos], tuple(groups))
        pieces.append(pd.DataFrame(cols, index=frame.index[pos]))
    extra = pd.concat(pieces).reindex(frame.index) if pieces else pd.DataFrame(index=frame.index)
    return _attach(frame, extra)
