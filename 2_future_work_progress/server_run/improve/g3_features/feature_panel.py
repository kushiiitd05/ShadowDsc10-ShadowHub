"""Dense hub x day panel built from raw daily history; truncation to an origin is the leakage guard."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

GROUP_SOURCES = ("Region", "RegionID", "HubRegion", "HubFormat", "AssortmentTier")


@dataclass(frozen=True)
class Panel:
    hubs: np.ndarray                 # (H,) sorted HubID
    dates: pd.DatetimeIndex          # (D,) contiguous daily axis
    orders: np.ndarray               # (H, D) OrderVolume, NaN where the row is missing
    is_open: np.ndarray              # (H, D) 1/0, NaN where the row is missing
    promo: np.ndarray | None         # (H, D) 1/0/NaN, None when the raw data has no promo column
    groups: dict[str, np.ndarray]    # static hub grouping name -> (H,) int codes, -1 = unknown

    def upto(self, origin: pd.Timestamp) -> "Panel":
        """New panel whose last column is `origin`; nothing dated after it survives."""
        origin = pd.Timestamp(origin)
        if origin < self.dates[0] or origin > self.dates[-1]:
            raise ValueError(f"origin {origin.date()} outside panel axis {self.dates[0].date()}..{self.dates[-1].date()}")
        stop = int(self.dates.get_loc(origin)) + 1
        return Panel(self.hubs, self.dates[:stop], self.orders[:, :stop], self.is_open[:, :stop],
                     None if self.promo is None else self.promo[:, :stop], self.groups)


def _dense(raw: pd.DataFrame, hub_idx: np.ndarray, day_idx: np.ndarray, shape: tuple[int, int], column: str) -> np.ndarray:
    out = np.full(shape, np.nan)
    out[hub_idx, day_idx] = raw[column].to_numpy(float)
    return out


def _group_codes(raw: pd.DataFrame, hubs: np.ndarray, columns: list[str]) -> dict[str, np.ndarray]:
    static = raw.drop_duplicates("HubID", keep="last").set_index("HubID").reindex(hubs)
    groups = {c: pd.factorize(static[c])[0].astype(np.int64) for c in columns}
    if len(columns) >= 2:
        joined = static[columns].astype("string").agg("|".join, axis=1).where(static[columns].notna().all(axis=1))
        groups["_".join(columns)] = pd.factorize(joined)[0].astype(np.int64)
    return groups


def build_panel(raw: pd.DataFrame) -> Panel:
    """Needs HubID, Date, OrderVolume, IsOpen; uses PromoActive and grouping columns when present."""
    missing = {"HubID", "Date", "OrderVolume", "IsOpen"} - set(raw.columns)
    if missing:
        raise ValueError(f"raw history lacks columns: {sorted(missing)}")
    data = raw.assign(Date=pd.to_datetime(raw["Date"]).dt.normalize()).drop_duplicates(["HubID", "Date"], keep="last")
    hubs = np.sort(data["HubID"].unique())
    dates = pd.date_range(data["Date"].min(), data["Date"].max(), freq="D")
    hub_idx = np.searchsorted(hubs, data["HubID"].to_numpy())
    day_idx = (data["Date"] - dates[0]).dt.days.to_numpy()
    shape = (len(hubs), len(dates))
    promo = _dense(data, hub_idx, day_idx, shape, "PromoActive") if "PromoActive" in data.columns else None
    group_cols = [c for c in GROUP_SOURCES if c in data.columns]
    return Panel(hubs, dates, _dense(data, hub_idx, day_idx, shape, "OrderVolume"),
                 _dense(data, hub_idx, day_idx, shape, "IsOpen"), promo, _group_codes(data, hubs, group_cols))
