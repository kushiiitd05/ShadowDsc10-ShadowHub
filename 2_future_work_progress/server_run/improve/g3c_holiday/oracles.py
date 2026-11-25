"""Slow, loop-based reference implementations (pandas only) that the vectorised g3c features are checked against."""
from __future__ import annotations

import numpy as np
import pandas as pd

DAY = pd.Timedelta(days=1)
WINDOW, CAP = 42, 14


def _log(frame: pd.DataFrame) -> pd.DataFrame:
    keep = frame["IsOpen"].eq(1) & frame["OrderVolume"].gt(0)
    return frame.loc[keep].assign(y=np.log1p(frame.loc[keep, "OrderVolume"]))


def _mean_median(values: pd.Series) -> tuple[float, float, int]:
    return (float(values.mean()) if len(values) else np.nan, float(values.median()) if len(values) else np.nan, len(values))


def holiday_history(raw: pd.DataFrame, hub: int, d: pd.Timestamp, origin: pd.Timestamp) -> dict[str, float]:
    """Expected g3c_hh_* values for one row (hub, target date d) at `origin`."""
    cell = raw[(raw["HubID"] == hub) & (raw["Date"] == d)]
    code = int(cell["RegionalHoliday"].iloc[0])
    keys = ["any_mean", "any_median", "any_n", "any_ratio", "type_mean", "type_median", "type_n", "type_ratio", "type_last_log", "type_last_days"]
    if code == 0:
        return {k: np.nan for k in keys}
    hist = _log(raw[(raw["HubID"] == hub) & (raw["Date"] <= origin)])
    holiday = hist[hist["RegionalHoliday"] > 0]
    same = holiday[holiday["RegionalHoliday"] == code]
    window = hist[(hist["Date"] > origin - 365 * DAY) & hist["RegionalHoliday"].eq(0) & (hist["Date"].dt.day_of_week == d.day_of_week)]
    base = window["y"].mean() if len(window) else np.nan
    out: dict[str, float] = {}
    for label, part in (("any", holiday), ("type", same)):
        mean, median, n = _mean_median(part["y"])
        out.update({f"{label}_mean": mean, f"{label}_median": median, f"{label}_n": float(n), f"{label}_ratio": mean - base})
    last = same.sort_values("Date").tail(1)
    out["type_last_log"] = float(last["y"].iloc[0]) if len(last) else np.nan
    out["type_last_days"] = float((origin - last["Date"].iloc[0]).days) if len(last) else np.nan
    return out


def _schedule(raw: pd.DataFrame, column: str, hub: int, fill_unanimous: bool = False) -> dict[pd.Timestamp, float]:
    sched = raw[raw["HubID"] == hub].set_index("Date")[column].to_dict()
    if fill_unanimous:       # missing hub rows take the date's code when every observed hub agrees
        per_date = raw.groupby("Date")[column].agg(["min", "max"])
        for date in per_date.index[per_date["min"] == per_date["max"]]:
            sched.setdefault(date, per_date.loc[date, "min"])
    return sched


def calendar_row(raw: pd.DataFrame, hub: int, d: pd.Timestamp, origin: pd.Timestamp) -> dict[str, float]:
    """Expected g3c_cal_* values (events known up to origin + 42 days)."""
    last_known = origin + WINDOW * DAY
    out: dict[str, float] = {}
    for tag, column, test, fill in (("hol", "RegionalHoliday", lambda v: v > 0, True), ("sch", "SchoolClosureFlag", lambda v: v == 1, False),
                                    ("promo", "PromoActive", lambda v: v == 1, False)):
        sched = _schedule(raw, column, hub, fill)
        events = sorted(t for t, v in sched.items() if t <= last_known and test(v))
        week_start = d - d.day_of_week * DAY
        out[f"{tag}_week"] = float(sum(week_start <= t <= week_start + 6 * DAY for t in events))
        out[f"{tag}_prev7"] = float(sum(d - 7 * DAY <= t < d for t in events))
        out[f"{tag}_next7"] = float(sum(d < t <= d + 7 * DAY for t in events))
        before = [(d - t).days for t in events if t < d]
        after = [(t - d).days for t in events if t > d]
        out[f"{tag}_dprev"] = float(min(min(before, default=CAP), CAP))
        out[f"{tag}_dnext"] = float(min(min(after, default=CAP), CAP))
    is_open = _schedule(raw, "IsOpen", hub)

    def closed(t: pd.Timestamp) -> bool:
        return t <= last_known and is_open.get(t, 1) == 0
    run_before = run_after = 0
    while run_before < CAP and closed(d - (run_before + 1) * DAY):
        run_before += 1
    while run_after < CAP and closed(d + (run_after + 1) * DAY):
        run_after += 1
    opened = is_open.get(d, np.nan) == 1
    out.update(closed_run_before=float(run_before), closed_run_after=float(run_after),
               first_after_closed=float(opened and run_before >= 1), last_before_closed=float(opened and run_after >= 1))
    return out


def sunday_row(raw: pd.DataFrame, hub: int, d: pd.Timestamp, origin: pd.Timestamp) -> dict[str, float]:
    if d.day_of_week != 6:
        return {"open_share": np.nan, "log_mean": np.nan, "n_open": np.nan}
    hist = raw[(raw["HubID"] == hub) & (raw["Date"] <= origin) & (raw["Date"] > origin - 365 * DAY) & (raw["Date"].dt.day_of_week == 6)]
    opened = hist[hist["IsOpen"] == 1]
    logs = _log(hist)["y"]
    return {"open_share": len(opened) / len(hist) if len(hist) else np.nan, "log_mean": float(logs.mean()) if len(logs) else np.nan,
            "n_open": float(len(opened))}


def promo_row(raw: pd.DataFrame, hub: int, origin: pd.Timestamp, window: int) -> dict[str, float]:
    """promo mean minus non-promo mean, averaged over weekdays with both kinds of day, weighted by the number of promo days."""
    hist = _log(raw[(raw["HubID"] == hub) & (raw["Date"] <= origin) & (raw["Date"] > origin - window * DAY)])
    total, weight = 0.0, 0.0
    for wd in range(7):
        part = hist[hist["Date"].dt.day_of_week == wd]
        on, off = part[part["PromoActive"] == 1]["y"], part[part["PromoActive"] == 0]["y"]
        if len(on) and len(off):
            total += len(on) * (on.mean() - off.mean())
            weight += len(on)
    return {"ratio": total / weight if weight else np.nan, "n": float((hist["PromoActive"] == 1).sum())}
