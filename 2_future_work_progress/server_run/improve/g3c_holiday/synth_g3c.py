"""Synthetic raw history with a schedule for the g3c tests.

Codes mimic the real file: 1 = ordinary holiday, 2 = Easter Friday/Monday (movable), 3 = Christmas 25/26. Most hubs are closed on national
holidays and Sundays; hubs 1-2 are open on both, hub 3 on code-1 holidays and on about half of the Sundays. Hubs 1-5 have an extra regional
holiday (code 1) on a few dates, hubs 6-9 another one. Closures, missing rows and 3 years of daily data as in synth_g3b.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CODE1 = ("2013-01-01", "2013-05-01", "2013-10-03", "2014-01-01", "2014-05-01", "2014-10-03", "2015-01-01", "2015-05-01")
CODE2 = ("2013-03-29", "2013-04-01", "2014-04-18", "2014-04-21", "2015-04-03", "2015-04-06")
CODE3 = ("2013-12-25", "2013-12-26", "2014-12-25", "2014-12-26")
REGIONAL_A = ("2013-05-30", "2014-06-19", "2015-06-04")       # hubs 1..5
REGIONAL_B = ("2013-11-01", "2014-11-01")                     # hubs 6..9
GAP_HUBS = (3, 4)
GAP_START, GAP_END = pd.Timestamp("2014-07-01"), pd.Timestamp("2014-12-31")
HOLIDAY_FACTOR = {0: 1.0, 1: 1.5, 2: 1.8, 3: 2.0}


def _holiday_codes(frame: pd.DataFrame) -> np.ndarray:
    code = np.zeros(len(frame), dtype=int)
    for dates, value in ((CODE1, 1), (CODE2, 2), (CODE3, 3)):
        code[frame["Date"].isin(pd.to_datetime(list(dates))).to_numpy()] = value
    regional = (frame["Date"].isin(pd.to_datetime(list(REGIONAL_A))) & frame["HubID"].between(1, 5)) | \
               (frame["Date"].isin(pd.to_datetime(list(REGIONAL_B))) & frame["HubID"].between(6, 9))
    code[regional.to_numpy()] = 1
    return code


def synthetic_raw(n_hubs: int = 14, n_days: int = 1000, seed: int = 0, start: str = "2013-01-01") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=n_days, freq="D")
    frame = pd.MultiIndex.from_product([np.arange(1, n_hubs + 1), dates], names=["HubID", "Date"]).to_frame(index=False)
    frame["Weekday"] = frame["Date"].dt.day_of_week + 1
    frame["RegionalHoliday"] = _holiday_codes(frame)
    day = (frame["Date"] - dates[0]).dt.days.to_numpy()
    frame["SchoolClosureFlag"] = (((day + 15 * (frame["HubID"] % 3).to_numpy()) % 45) < 10).astype(int)
    frame["PromoActive"] = (rng.random(len(frame)) < 0.25).astype(int)
    national = frame["Date"].isin(pd.to_datetime([*CODE1, *CODE2, *CODE3])).to_numpy()
    open_hol = frame["HubID"].isin([1, 2]) | (frame["HubID"].eq(3) & frame["RegionalHoliday"].eq(1))
    sunday = frame["Weekday"].eq(7)
    sunday_open = frame["HubID"].isin([1, 2]) | (frame["HubID"].eq(3) & (rng.random(len(frame)) < 0.5))
    is_open = np.ones(len(frame), dtype=bool)
    is_open &= ~(national & ~open_hol.to_numpy())
    is_open &= ~(sunday.to_numpy() & ~sunday_open.to_numpy())
    is_open &= rng.random(len(frame)) >= 0.02
    frame["IsOpen"] = is_open.astype(int)
    season = 1 + 0.25 * np.sin(2 * np.pi * day / 365.0)
    base = (50 + 10 * (frame["HubID"] % 4)) * season * (1 + 0.0004 * day)
    factor = np.vectorize(HOLIDAY_FACTOR.get)(frame["RegionalHoliday"].to_numpy()) * np.where(sunday, 0.5, 1.0)
    orders = base * (1 + 0.3 * frame["PromoActive"]) * factor * rng.lognormal(0, 0.1, len(frame))
    frame["OrderVolume"] = np.where(frame["IsOpen"].eq(1), orders.round(), 0.0)
    frame["HubFormat"] = 1 + (frame["HubID"] % 2)
    frame["AssortmentTier"] = 1 + (frame["HubID"] % 3)
    drop = rng.random(len(frame)) < 0.01
    gap = frame["HubID"].isin(GAP_HUBS) & frame["Date"].between(GAP_START, GAP_END)
    return frame.loc[~(drop | gap)].reset_index(drop=True)


def synthetic_target_frame(raw: pd.DataFrame, origin: pd.Timestamp, horizon: int = 42) -> pd.DataFrame:
    """Rows of the forecast window that exist in the raw schedule (hub x date), key columns only."""
    window = raw[raw["Date"].between(origin + pd.Timedelta(days=1), origin + pd.Timedelta(days=horizon))]
    frame = window[["HubID", "Date", "Weekday"]].sort_values(["HubID", "Date"]).reset_index(drop=True)
    frame["Horizon"] = (frame["Date"] - origin).dt.days
    return frame
