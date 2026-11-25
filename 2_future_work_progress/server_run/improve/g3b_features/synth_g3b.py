"""Synthetic raw history for the g3b tests: 3 years of daily rows, a 184-day gap for two hubs, missing rows, yearly seasonality."""
from __future__ import annotations

import numpy as np
import pandas as pd

GAP_HUBS = (3, 4)
GAP_START, GAP_END = pd.Timestamp("2014-07-01"), pd.Timestamp("2014-12-31")


def synthetic_raw(n_hubs: int = 14, n_days: int = 1000, seed: int = 0, start: str = "2013-01-01") -> pd.DataFrame:
    """Sunday closures, random closures/promos, ~1% missing rows, shared yearly seasonality, hubs 3 and 4 have a 184-day gap."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=n_days, freq="D")
    hubs = np.arange(1, n_hubs + 1)
    frame = pd.MultiIndex.from_product([hubs, dates], names=["HubID", "Date"]).to_frame(index=False)
    frame["Weekday"] = frame["Date"].dt.day_of_week + 1
    frame["IsOpen"] = (frame["Weekday"] != 7).astype(int)
    frame.loc[rng.random(len(frame)) < 0.02, "IsOpen"] = 0
    frame["PromoActive"] = (rng.random(len(frame)) < 0.25).astype(int)
    day = (frame["Date"] - dates[0]).dt.days.to_numpy()
    season = 1 + 0.25 * np.sin(2 * np.pi * day / 365.0)
    base = (50 + 10 * (frame["HubID"] % 4)) * season * (1 + 0.0004 * day)
    orders = base * (1 + 0.3 * frame["PromoActive"]) * rng.lognormal(0, 0.1, len(frame))
    frame["OrderVolume"] = np.where(frame["IsOpen"].eq(1), orders.round(), 0.0)
    frame["HubFormat"] = 1 + (frame["HubID"] % 2)
    frame["AssortmentTier"] = 1 + (frame["HubID"] % 3)
    drop = rng.random(len(frame)) < 0.01
    gap = frame["HubID"].isin(GAP_HUBS) & frame["Date"].between(GAP_START, GAP_END)
    return frame.loc[~(drop | gap)].reset_index(drop=True)


def synthetic_target_frame(raw: pd.DataFrame, origin: pd.Timestamp, horizon: int = 42) -> pd.DataFrame:
    """Forecast-window rows for every hub that has any history: key columns only."""
    hubs = np.sort(raw["HubID"].unique())
    dates = pd.date_range(origin + pd.Timedelta(days=1), periods=horizon, freq="D")
    frame = pd.MultiIndex.from_product([hubs, dates], names=["HubID", "Date"]).to_frame(index=False)
    frame["Weekday"] = frame["Date"].dt.day_of_week + 1
    frame["PromoActive"] = (frame["Date"].dt.day % 5 == 0).astype(int)
    frame["Horizon"] = (frame["Date"] - origin).dt.days
    return frame
