"""Shared synthetic raw history for the G3 tests (no real data needed)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
for path in (HERE, HERE.parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def synthetic_raw(n_hubs: int = 12, n_days: int = 500, seed: int = 0, start: str = "2013-01-01") -> pd.DataFrame:
    """Daily panel with Sunday closures, random closures/promos, a few missing rows and two hub groups."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=n_days, freq="D")
    hubs = np.arange(1, n_hubs + 1)
    frame = pd.MultiIndex.from_product([hubs, dates], names=["HubID", "Date"]).to_frame(index=False)
    frame["Weekday"] = frame["Date"].dt.day_of_week + 1
    frame["IsOpen"] = (frame["Weekday"] != 7).astype(int)
    frame.loc[rng.random(len(frame)) < 0.02, "IsOpen"] = 0
    frame["PromoActive"] = (rng.random(len(frame)) < 0.25).astype(int)
    base = 50 + 10 * (frame["HubID"] % 4) + 0.02 * (frame["Date"] - dates[0]).dt.days
    orders = base * (1 + 0.3 * frame["PromoActive"]) * rng.lognormal(0, 0.1, len(frame))
    frame["OrderVolume"] = np.where(frame["IsOpen"].eq(1), orders.round(), 0.0)
    frame["AppSessions"] = frame["OrderVolume"] * 3
    frame["HubFormat"] = 1 + (frame["HubID"] % 2)
    frame["AssortmentTier"] = 1 + (frame["HubID"] % 3)
    drop = rng.random(len(frame)) < 0.01
    return frame.loc[~drop].reset_index(drop=True)


def synthetic_target_frame(raw: pd.DataFrame, origin: pd.Timestamp, horizon: int = 42) -> pd.DataFrame:
    """Forecast-window rows for every hub: key columns only (features are added by the code under test)."""
    hubs = np.sort(raw["HubID"].unique())
    dates = pd.date_range(origin + pd.Timedelta(days=1), periods=horizon, freq="D")
    frame = pd.MultiIndex.from_product([hubs, dates], names=["HubID", "Date"]).to_frame(index=False)
    frame["Weekday"] = frame["Date"].dt.day_of_week + 1
    frame["PromoActive"] = (frame["Date"].dt.day % 5 == 0).astype(int)
    frame["Horizon"] = (frame["Date"] - origin).dt.days
    return frame


@pytest.fixture()
def raw() -> pd.DataFrame:
    return synthetic_raw()


@pytest.fixture()
def origin() -> pd.Timestamp:
    return pd.Timestamp("2014-04-20")
