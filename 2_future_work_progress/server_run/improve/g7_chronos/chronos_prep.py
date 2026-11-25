"""Pure helpers for the zero-shot Chronos-2 member (no model, no disk I/O, inputs are never mutated).

Contract
- history for a fold is every raw row with Date <= ForecastOrigin (the origin day is the last observed day, exactly the
  information the fold feature files use); nothing after the origin reaches the model
- target series = log1p(OrderVolume); closed days are real zeros; rows missing from the raw table become NaN targets
- future covariates (IsOpen, PromoActive, RegionalHoliday, SchoolClosureFlag) are known calendar inputs taken from the fold file
- prediction -> expm1 -> clip at 0 -> forced to 0 on future closed days
"""
from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

ALL_COVARIATES = ("IsOpen", "PromoActive", "RegionalHoliday", "SchoolClosureFlag")
OOF_COLUMNS = ["fold", "HubID", "Date", "Horizon", "IsOpen", "ActualOrderVolume", "pred"]
QUANTILE = 0.5


def volume_from_log(log_pred: np.ndarray, is_open: np.ndarray) -> np.ndarray:
    """log1p-space prediction -> order volume; closed rows are forced to 0."""
    return np.where(np.asarray(is_open) == 1, np.clip(np.expm1(np.asarray(log_pred, dtype=float)), 0.0, None), 0.0)


def build_context(raw: pd.DataFrame, origin: pd.Timestamp, covariates: Sequence[str] = (),
                  hubs: Iterable[int] | None = None) -> pd.DataFrame:
    """Chronos long-format context: item_id, timestamp, target [, covariates], daily and gap-free per hub up to `origin`.

    Each hub's series starts at its first observed day; days missing from the raw table get NaN target and 0 covariates.
    """
    origin = pd.Timestamp(origin)
    hist = raw.loc[raw["Date"] <= origin]
    if hubs is not None:
        hist = hist.loc[hist["HubID"].isin(list(hubs))]
    if hist.empty:
        raise ValueError("no history rows on or before the origin")
    first = hist.groupby("HubID")["Date"].min()
    days = pd.date_range(first.min(), origin, freq="D")
    grid = pd.MultiIndex.from_product([sorted(first.index), days], names=["HubID", "Date"]).to_frame(index=False)
    grid = grid.loc[grid["Date"] >= grid["HubID"].map(first)]
    cols = ["HubID", "Date", "OrderVolume", *covariates]
    merged = grid.merge(hist[cols], on=["HubID", "Date"], how="left", validate="one_to_one")
    out = pd.DataFrame({"item_id": merged["HubID"], "timestamp": merged["Date"],
                        "target": np.log1p(merged["OrderVolume"].to_numpy(dtype="float64"))})
    for c in covariates:
        out[c] = merged[c].fillna(0).to_numpy(dtype="float64")
    return out.sort_values(["item_id", "timestamp"]).reset_index(drop=True)


def build_future(fold_frame: pd.DataFrame, covariates: Sequence[str]) -> pd.DataFrame:
    """Known-future covariates for the forecast window: item_id, timestamp, covariates."""
    out = pd.DataFrame({"item_id": fold_frame["HubID"].to_numpy(), "timestamp": fold_frame["Date"].to_numpy()})
    for c in covariates:
        out[c] = fold_frame[c].to_numpy(dtype="float64")
    return out.sort_values(["item_id", "timestamp"]).reset_index(drop=True)


def forecast_to_oof(forecast: pd.DataFrame, fold_frame: pd.DataFrame, actual: pd.DataFrame, fold_number: int,
                    quantile: str = "0.5") -> pd.DataFrame:
    """Join a Chronos forecast (log1p scale) onto the fold rows by (HubID, Date) and return the OOF frame.

    Every fold row must have exactly one forecast row and one actual; closed future days get prediction 0.
    """
    if forecast.duplicated(["item_id", "timestamp"]).any():
        raise ValueError("duplicate forecast rows for the same hub and day")
    fc = forecast[["item_id", "timestamp", quantile]].rename(columns={"item_id": "HubID", "timestamp": "Date", quantile: "_log"})
    base = fold_frame[["HubID", "Date", "Horizon", "IsOpen"]].merge(
        fc, on=["HubID", "Date"], how="left", validate="one_to_one")
    if base["_log"].isna().any():
        raise ValueError(f"{int(base['_log'].isna().sum())} fold rows have no forecast")
    act = actual[["HubID", "Date", "OrderVolume"]].rename(columns={"OrderVolume": "ActualOrderVolume"})
    base = base.merge(act, on=["HubID", "Date"], how="left", validate="one_to_one")
    if base["ActualOrderVolume"].isna().any():
        raise ValueError(f"{int(base['ActualOrderVolume'].isna().sum())} fold rows have no actual")
    base["pred"] = volume_from_log(base["_log"].to_numpy(), base["IsOpen"].to_numpy())
    base["fold"] = fold_number
    return base[OOF_COLUMNS].reset_index(drop=True)


def forecast_fold(pipeline: Any, raw: pd.DataFrame, fold_frame: pd.DataFrame, fold_number: int,
                  covariates: Sequence[str] = (), batch_size: int = 256, hubs: Iterable[int] | None = None) -> pd.DataFrame:
    """Run one fold: build the no-leak context, call pipeline.predict_df, align to the fold rows. `pipeline` is injectable."""
    origin = pd.Timestamp(fold_frame["ForecastOrigin"].iloc[0])
    if hubs is not None:
        fold_frame = fold_frame.loc[fold_frame["HubID"].isin(list(hubs))]
    horizon = int(fold_frame["Horizon"].max())
    ctx = build_context(raw, origin, covariates, hubs=sorted(fold_frame["HubID"].unique()))
    future = build_future(fold_frame, covariates) if covariates else None
    fc = pipeline.predict_df(ctx, future_df=future, id_column="item_id", timestamp_column="timestamp", target="target",
                             prediction_length=horizon, quantile_levels=[QUANTILE], batch_size=batch_size)
    return forecast_to_oof(fc, fold_frame, raw, fold_number, quantile=str(QUANTILE))
