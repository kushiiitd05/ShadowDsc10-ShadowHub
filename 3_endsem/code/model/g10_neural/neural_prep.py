"""Pure helpers for the neural / foundation-model pilots (TiDE, NHITS, TabPFN-TS).

Everything here is I/O-free and never mutates its inputs. The three guarantees the pilots rely on:

1. history_window / build_panel: the model only sees rows dated on or before the forecast origin
   (the last observed day; the forecast covers origin+1 .. origin+h). Rows after the origin are dropped
   before anything is computed from them.
2. build_future_frame / forecast_to_frame: forecasts are joined back to the fold rows on (HubID, Date)
   with a strict one-to-one check, never by position.
3. volume_from_log: log1p-space prediction -> order volume, clipped at 0 and forced to 0 on closed days.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

OPEN_COL, PROMO_COL = "IsOpen", "PromoActive"
FLAG_COLUMNS = (OPEN_COL, PROMO_COL, "RegionalHoliday", "SchoolClosureFlag")  # known in the future
FLAG_NAMES = {OPEN_COL: "open", PROMO_COL: "promo", "RegionalHoliday": "holiday", "SchoolClosureFlag": "school"}
STATIC_CATEGORICAL = ("HubFormat", "AssortmentTier")
STATIC_NUMERIC = ("CompetitorDistanceLog1p", "LoyaltyProgram")
KEYS = ["HubID", "Date"]


def calendar_features(dates: pd.Series) -> pd.DataFrame:
    """Deterministic calendar covariates from the date alone (known for any future day)."""
    d = pd.DatetimeIndex(pd.to_datetime(dates))
    out = {f"dow{k}": (d.day_of_week == k).astype("float32") for k in range(7)}
    dom = (d.day.to_numpy() - 1) / d.days_in_month.to_numpy()
    out["dom_sin"] = np.sin(2 * np.pi * dom).astype("float32")
    out["dom_cos"] = np.cos(2 * np.pi * dom).astype("float32")
    doy = (d.day_of_year.to_numpy() - 1) / 365.25
    out["doy_sin"] = np.sin(2 * np.pi * doy).astype("float32")
    out["doy_cos"] = np.cos(2 * np.pi * doy).astype("float32")
    out["month_start"] = d.is_month_start.astype("float32")
    out["month_end"] = d.is_month_end.astype("float32")
    return pd.DataFrame(out, index=range(len(d)))


def futr_columns() -> list[str]:
    return [*FLAG_NAMES.values(), *calendar_features(pd.Series(pd.to_datetime(["2015-01-01"]))).columns]


def history_window(base: pd.DataFrame, origin: pd.Timestamp) -> pd.DataFrame:
    """Rows dated on or before the forecast origin (the last observed day). Never returns a later row."""
    origin = pd.Timestamp(origin)
    out = base.loc[pd.to_datetime(base["Date"]) <= origin].copy()
    if out.empty:
        raise ValueError(f"no history on or before {origin.date()}")
    return out


def _covariates(frame: pd.DataFrame) -> pd.DataFrame:
    """unique_id, ds, flag covariates and calendar covariates for rows that carry the flag columns."""
    out = pd.DataFrame({"unique_id": frame["HubID"].astype(int).astype(str).to_numpy(), "ds": pd.to_datetime(frame["Date"]).to_numpy()})
    for col, name in FLAG_NAMES.items():
        out[name] = frame[col].to_numpy(dtype="float32")
    cal = calendar_features(frame["Date"])
    return pd.concat([out, cal], axis=1)


def build_panel(base: pd.DataFrame, origin: pd.Timestamp) -> pd.DataFrame:
    """Complete daily panel (unique_id, ds, y=log1p(orders), covariates) from the start of history to the origin.

    Days a hub has no row for are filled as closed (y=0, no flags); the calendar covariates are always real.
    """
    hist = history_window(base, origin)
    start, end = pd.to_datetime(hist["Date"]).min(), pd.Timestamp(origin)
    days = pd.date_range(start, end, freq="D")
    hubs = np.sort(hist["HubID"].astype(int).unique())
    grid = pd.MultiIndex.from_product([hubs, days], names=["HubID", "Date"]).to_frame(index=False)
    h = hist.assign(Date=pd.to_datetime(hist["Date"]))[[*KEYS, "OrderVolume", *FLAG_COLUMNS]]
    h = h.drop_duplicates(KEYS, keep="last")
    full = grid.merge(h, on=KEYS, how="left")
    full["OrderVolume"] = full["OrderVolume"].fillna(0.0).clip(lower=0.0)
    for col in FLAG_COLUMNS:
        full[col] = full[col].fillna(0.0)
    panel = _covariates(full)
    panel.insert(2, "y", np.log1p(full["OrderVolume"].to_numpy(dtype="float64")).astype("float32"))
    return panel.sort_values(["unique_id", "ds"], kind="stable").reset_index(drop=True)


def impute_closed(panel: pd.DataFrame) -> pd.DataFrame:
    """Replace y on closed days by linear interpolation between the hub's open days (history only; edges carry the nearest open value).

    Closed days are known zeros, and they are forced back to 0 at prediction time, so the network only has to model
    open-day volume. Returns a new frame; hubs with no open day at all stay 0.
    """
    wide = panel.pivot(index="ds", columns="unique_id", values="y").astype("float64")
    closed = panel.pivot(index="ds", columns="unique_id", values="open").to_numpy() == 0
    filled = wide.mask(closed).interpolate(method="linear", limit_direction="both").fillna(0.0)
    long = filled.rename_axis(columns=None).reset_index().melt(id_vars="ds", var_name="unique_id", value_name="y_filled")
    out = panel.merge(long, on=["ds", "unique_id"], how="left", validate="one_to_one")
    out["y"] = out.pop("y_filled").astype("float32")
    return out


def build_future_frame(valid: pd.DataFrame, origin: pd.Timestamp, horizon: int) -> pd.DataFrame:
    """Known-future covariates for the forecast days. Validates that they are exactly origin+1 .. origin+horizon per hub."""
    origin = pd.Timestamp(origin)
    dates = pd.to_datetime(valid["Date"])
    expected = pd.date_range(origin + pd.Timedelta(days=1), periods=horizon, freq="D")
    if sorted(dates.unique()) != list(expected):
        raise ValueError("forecast days are not origin+1 .. origin+horizon")
    if valid.duplicated(KEYS).any() or (valid.groupby("HubID").size() != horizon).any():
        raise ValueError("each hub must have exactly one row per forecast day")
    out = _covariates(valid.assign(Date=dates))
    return out.sort_values(["unique_id", "ds"], kind="stable").reset_index(drop=True)


def build_static(valid: pd.DataFrame) -> pd.DataFrame:
    """One row per hub: one-hot hub format / assortment tier plus numeric hub metadata (time-invariant)."""
    first = valid.sort_values(KEYS).drop_duplicates("HubID", keep="first").reset_index(drop=True)
    out = pd.DataFrame({"unique_id": first["HubID"].astype(int).astype(str)})
    for col in STATIC_CATEGORICAL:
        for level in sorted(first[col].astype(str).unique()):
            out[f"{col}_{level}"] = (first[col].astype(str) == level).astype("float32")
    for col in STATIC_NUMERIC:
        values = first[col].astype("float64")
        if values.isna().any():  # hub metadata can be missing: median fill plus an indicator
            out[f"{col}_missing"] = values.isna().astype("float32")
            values = values.fillna(values.median() if values.notna().any() else 0.0)
        std = values.std(ddof=0)
        out[col] = ((values - values.mean()) / (std if std > 0 else 1.0)).astype("float32")
    return out.sort_values("unique_id", kind="stable").reset_index(drop=True)


def volume_from_log(log_pred: np.ndarray, is_open: np.ndarray) -> np.ndarray:
    """log1p-space prediction -> order volume; clipped at 0, closed rows forced to 0, non-finite -> 0."""
    log_pred = np.asarray(log_pred, dtype="float64")
    vol = np.clip(np.expm1(np.nan_to_num(log_pred, nan=0.0, posinf=0.0, neginf=0.0)), 0.0, None)
    return np.where(np.asarray(is_open) == 1, vol, 0.0)


def forecast_to_frame(forecast: pd.DataFrame, valid: pd.DataFrame, column: str) -> pd.DataFrame:
    """Attach a log-space forecast (unique_id, ds, column) to the fold rows by (HubID, Date); returns oof columns with pred."""
    fc = forecast[["unique_id", "ds", column]].copy()
    fc["HubID"] = fc["unique_id"].astype(int)
    fc["Date"] = pd.to_datetime(fc["ds"])
    if fc.duplicated(KEYS).any():
        raise ValueError("forecast has duplicate (HubID, Date) rows")
    rows = valid[["HubID", "Date", "Horizon", "IsOpen", "ActualOrderVolume"]].copy()
    rows["Date"] = pd.to_datetime(rows["Date"])
    merged = rows.merge(fc[[*KEYS, column]], on=KEYS, how="left", validate="one_to_one")
    if merged[column].isna().any():
        raise ValueError(f"{int(merged[column].isna().sum())} fold rows have no forecast")
    merged["pred"] = volume_from_log(merged[column].to_numpy(), merged["IsOpen"].to_numpy())
    return merged.drop(columns=[column])
