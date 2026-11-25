"""Item B: two-stage month-ahead model.

Stage 1 predicts the weekly (7 forecast days) hub total; stage 2 is the cs02 day-level model that also receives the stage-1 prediction.
  * week of a row = (Horizon - 1) // 7  (6 weeks per origin)
  * stage-1 rows = (hub, origin, week); features = the day-level feature columns aggregated over the week (flags summed, static categoricals
    taken from the first day, everything else averaged); target = log1p(sum of ActualOrderVolume over the week), weeks with a zero total dropped
  * stage-1 values of training rows are CROSS-FITTED over blocks of training origins (origins whose 42-day label window overlaps a held-out
    block are purged); holdout and validation rows get predictions from one stage-1 model fitted on all (purged) training origins
  * stage-2 extra columns: g12b_s1_week (log weekly total), g12b_s1_per_open (minus log of the open days of the week),
    g12b_s1_w1..w6 (the hub-origin's whole six-week stage-1 profile; full variant only)
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

WEEKS = 6
HORIZON = 42
PURGE_DAYS = 42
FLAG_COLUMNS = ("IsOpen", "PromoActive", "RegionalHoliday", "SchoolClosureFlag")
STATIC_CATEGORICAL = ("HubID", "HubFormat", "AssortmentTier", "LoyaltyProgram", "LoyaltyProgramInterval")
DROP_COLUMNS = ("Weekday", "DayOfMonth", "Horizon", "Date")          # meaningless once averaged over a week (Horizon is re-added as the week index)
S1_PREFIX = "g12b_s1_"
KEY = ["HubID", "_origin", "_week"]

FitPredict = Callable[[pd.DataFrame, pd.DataFrame], np.ndarray]


def week_of(horizon: np.ndarray | pd.Series) -> np.ndarray:
    return ((np.asarray(horizon, dtype=np.int64) - 1) // 7).astype(np.int64)


def week_table(frame: pd.DataFrame, columns: list[str], origin_col: str = "_origin") -> pd.DataFrame:
    """One row per (HubID, origin, week): aggregated features, n_open_days, week index, and `_total` (sum of ActualOrderVolume) when present."""
    work = frame.assign(_week=week_of(frame["Horizon"]))
    keys = ["HubID", origin_col, "_week"]
    if origin_col != "_origin":
        work = work.rename(columns={origin_col: "_origin"})
        keys = KEY
    grouped = work.groupby(keys, sort=True, observed=True)
    parts = []
    for c in dict.fromkeys(columns):
        if c in DROP_COLUMNS or c in ("HubID",) or c not in work.columns:
            continue
        if c in FLAG_COLUMNS:
            parts.append((work[c].fillna(0) > 0).astype(float).groupby([work[k] for k in keys]).sum().rename(f"{c}_days"))
        elif c in STATIC_CATEGORICAL or work[c].dtype == object or str(work[c].dtype) in ("category", "string"):
            parts.append(grouped[c].first())
        else:
            parts.append(grouped[c].mean())
    out = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=grouped.size().index)
    out["n_open_days"] = (work["IsOpen"].fillna(0) > 0).astype(float).groupby([work[k] for k in keys]).sum() if "IsOpen" in work else np.nan
    if "ActualOrderVolume" in work.columns:
        out["_total"] = grouped["ActualOrderVolume"].sum()
    out["week_index"] = out.index.get_level_values("_week").astype(float)
    return out.reset_index()


def stage1_feature_names(table: pd.DataFrame) -> list[str]:
    return [c for c in table.columns if c not in ("_origin", "_week", "_total")]


def blocked_splits(origins: np.ndarray, n_blocks: int, purge_days: int = PURGE_DAYS) -> list[tuple[np.ndarray, np.ndarray]]:
    """Cross-fit splits over sorted unique origins: (held-out origins, purged training origins) per contiguous block.

    A training origin is purged when its label window (origin, origin + 42 days] overlaps a held-out one, i.e. |o - o_held| < purge_days.
    """
    origins = np.sort(pd.to_datetime(np.unique(origins)).to_numpy())
    blocks = [b for b in np.array_split(origins, min(n_blocks, len(origins))) if len(b)]
    splits = []
    for held in blocks:
        gap = np.abs((origins[:, None] - held[None, :]) / np.timedelta64(1, "D")).min(axis=1)
        splits.append((held, origins[gap >= purge_days]))
    return splits


def crossfit_predict(table: pd.DataFrame, fit_predict: FitPredict, n_blocks: int = 6) -> pd.Series:
    """Out-of-fold stage-1 prediction for every row of `table` (index preserved): no row is predicted by a model that saw its origin or a neighbour."""
    out = pd.Series(np.nan, index=table.index, dtype=float)
    origins = table["_origin"].to_numpy()
    for held, train_origins in blocked_splits(origins, n_blocks):
        pred_mask = np.isin(origins, held)
        train_mask = np.isin(origins, train_origins)
        if not train_mask.any():
            continue
        out[pred_mask] = fit_predict(table[train_mask], table[pred_mask])
    return out


def profile(table: pd.DataFrame, pred: pd.Series) -> pd.DataFrame:
    """(HubID, _origin) x week wide table of predictions named g12b_s1_w{1..6}."""
    wide = pd.DataFrame({"HubID": table["HubID"].to_numpy(), "_origin": table["_origin"].to_numpy(), "_week": table["_week"].to_numpy(), "v": pred.to_numpy()})
    wide = wide.pivot_table(index=["HubID", "_origin"], columns="_week", values="v", aggfunc="first")
    wide = wide.reindex(columns=range(WEEKS))
    wide.columns = [f"{S1_PREFIX}w{w + 1}" for w in range(WEEKS)]
    return wide.reset_index()


def attach(frame: pd.DataFrame, table: pd.DataFrame, pred: pd.Series, full: bool, origin_col: str = "_origin") -> pd.DataFrame:
    """New day-level frame with the g12b stage-1 columns (NaN where the (hub, origin, week) has no stage-1 prediction)."""
    clash = [c for c in frame.columns if c.startswith(S1_PREFIX)]
    if clash:
        raise ValueError(f"frame already has stage-1 columns: {clash[:3]}")
    week = pd.DataFrame({"HubID": table["HubID"].to_numpy(), "_origin": table["_origin"].to_numpy(), "_week": table["_week"].to_numpy(),
                         f"{S1_PREFIX}week": pred.to_numpy(float), "_open": table["n_open_days"].to_numpy(float)})
    with np.errstate(invalid="ignore", divide="ignore"):
        week[f"{S1_PREFIX}per_open"] = np.where(week["_open"] > 0, week[f"{S1_PREFIX}week"] - np.log(np.maximum(week["_open"], 1)), np.nan)
    week = week.drop(columns="_open")
    keys = pd.DataFrame({"HubID": frame["HubID"].to_numpy(), "_origin": frame[origin_col].to_numpy(), "_week": week_of(frame["Horizon"])}, index=frame.index)
    merged = keys.merge(week, on=KEY, how="left", validate="many_to_one")
    cols = [f"{S1_PREFIX}week", f"{S1_PREFIX}per_open"]
    if full:
        merged = merged.merge(profile(table, pred), on=["HubID", "_origin"], how="left", validate="many_to_one")
        cols += [f"{S1_PREFIX}w{w + 1}" for w in range(WEEKS)]
    extra = merged[cols].astype("float32")
    extra.index = frame.index
    return pd.concat([frame, extra], axis=1)


def build_stage1(train: pd.DataFrame, hold: pd.DataFrame, valid: pd.DataFrame, valid_origin: pd.Timestamp, columns: list[str],
                 fit_predict: FitPredict, n_blocks: int = 6) -> dict[str, tuple[pd.DataFrame, pd.Series]]:
    """Stage-1 week tables and predictions: train cross-fitted, hold and valid from one model on all purged training origins."""
    wt_train = week_table(train, columns)
    wt_hold = week_table(hold, columns)
    wt_valid = week_table(valid.assign(_origin=pd.Timestamp(valid_origin)), columns)
    wt_valid = wt_valid.drop(columns=[c for c in ("_total",) if c in wt_valid.columns])
    fit_rows = wt_train[wt_train["_total"] > 0]
    oof = crossfit_predict(wt_train, fit_predict, n_blocks)
    full = fit_predict(fit_rows, pd.concat([wt_hold, wt_valid], ignore_index=True))
    n_hold = len(wt_hold)
    return {"train": (wt_train, oof), "hold": (wt_hold, pd.Series(full[:n_hold], index=wt_hold.index)),
            "valid": (wt_valid, pd.Series(full[n_hold:], index=wt_valid.index))}
