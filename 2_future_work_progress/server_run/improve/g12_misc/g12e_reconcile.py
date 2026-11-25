"""Item E: hierarchical reconciliation of hub-day predictions (log-space MinT-lite, offline from OOF).

Only the cs02 hub-day predictions exist (there is no independent format-level or total forecast), so reconciliation reduces to shrinkage of each
hub's prediction toward the aggregate prediction of its group on the same day. Two forms (all in log1p space, open rows with pred > 0 only):
  level    z = gm_{g,d} - lp_{h,d}                 shrinks the hub level toward the group mean log prediction (expected to hurt: hubs differ in size)
  pattern  z = gdev_{g,d} - dev_{h,d}              dev_{h,d} = lp_{h,d} - mean_d lp_{h,.} (within the 42-day window); shrinks the hub's day-to-day
                                                   pattern toward the group pattern, keeping the hub level
  pattern2 [z_format, z_total]                     hub -> format pattern and format -> total pattern, one weight each (two-level)
Shift = w * z with w fitted leave-fold-out (least squares through the origin); also reported with w clipped to [0, 1] (a genuine shrinkage weight).
Groups: HubFormat, HubFormat x AssortmentTier. The fit pool is folds 1-4 minus the evaluated fold.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import offline_common as oc

POOL = (1, 2, 3, 4)


def log_pred(frame: pd.DataFrame) -> np.ndarray:
    return np.log1p(np.clip(frame["pred"].to_numpy(float), 0, None))


def group_day_mean(values: np.ndarray, mask: np.ndarray, group: np.ndarray, date: np.ndarray) -> np.ndarray:
    """Mean of `values` over masked rows sharing (group, date), broadcast to every row (NaN where the cell has no masked row)."""
    cell = pd.DataFrame({"g": group, "d": date, "v": np.where(mask, values, np.nan)})
    return cell.groupby(["g", "d"])["v"].transform("mean").to_numpy()


def hub_deviation(frame: pd.DataFrame, lp: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """lp minus the hub's own mean lp over its masked rows (the fold window)."""
    df = pd.DataFrame({"h": frame["HubID"].to_numpy(), "v": np.where(mask, lp, np.nan)})
    return lp - df.groupby("h")["v"].transform("mean").to_numpy()


def designs_for_fold(frame: pd.DataFrame, group: np.ndarray, kind: str) -> np.ndarray:
    """(n,1) or (n,2) design for one fold. `group` = per-row group code (any hashable codes); rows outside the mask get NaN."""
    mask = oc.correctable(frame)
    lp = log_pred(frame)
    date = frame["Date"].to_numpy()
    nan = np.full(len(frame), np.nan)
    if kind == "level":
        z = group_day_mean(lp, mask, group, date) - lp
        return np.where(mask, z, nan).reshape(-1, 1)
    dev = hub_deviation(frame, lp, mask)
    gdev = group_day_mean(dev, mask, group, date)
    if kind == "pattern":
        return np.where(mask, gdev - dev, nan).reshape(-1, 1)
    if kind == "pattern2":
        tdev = group_day_mean(dev, mask, np.zeros(len(frame), dtype=int), date)
        return np.column_stack([np.where(mask, gdev - dev, nan), np.where(mask, tdev - gdev, nan)])
    raise ValueError(f"unknown reconciliation kind: {kind}")


def group_codes(frame: pd.DataFrame, meta: pd.DataFrame, grouping: str) -> np.ndarray:
    """Per-row group code from hub metadata ('format' or 'format_tier'); hubs without metadata share code -1."""
    m = meta.set_index("HubID")
    if grouping == "format":
        key = m["HubFormat"].astype(int)
    elif grouping == "format_tier":
        key = m["HubFormat"].astype(int) * 10 + m["AssortmentTier"].astype(int)
    else:
        raise ValueError(f"unknown grouping: {grouping}")
    return frame["HubID"].map(key).fillna(-1).astype(int).to_numpy()


def clipped_weights(frames: dict[int, pd.DataFrame], designs: dict[int, np.ndarray], eval_folds: tuple[int, ...]) -> pd.DataFrame:
    """Same as evaluate_correction but each weight is clipped to [0, 1] (pure shrinkage)."""
    fit = {f: (designs[f][oc.correctable(frames[f])], oc.residual(frames[f]).to_numpy()[oc.correctable(frames[f])]) for f in POOL if f in designs}
    rows = []
    for f in eval_folds:
        beta = np.clip(oc.leave_fold_out(fit, f), 0.0, 1.0)
        x = np.nan_to_num(designs[f], nan=0.0)
        base, new = oc.scores(frames[f]), oc.scores(frames[f], oc.apply_shift(frames[f], x @ beta))
        rows.append({"fold": f, "base_all": base["all"], "base_excl": base["excl_open0"], "new_all": new["all"], "new_excl": new["excl_open0"],
                     "beta": ";".join(f"{b:.4f}" for b in beta)})
    out = pd.DataFrame(rows)
    out["gain_all"], out["gain_excl"] = out["base_all"] - out["new_all"], out["base_excl"] - out["new_excl"]
    return out


def run(frames: dict[int, pd.DataFrame], meta: pd.DataFrame, folds: tuple[int, ...] = (1, 2, 3, 4, 5)) -> pd.DataFrame:
    parts = []
    for grouping in ("format", "format_tier"):
        for kind in ("level", "pattern", "pattern2"):
            designs = {f: designs_for_fold(frames[f], group_codes(frames[f], meta, grouping), kind) for f in folds if f in frames}
            parts.append(oc.evaluate_correction(frames, designs, POOL, folds).assign(variant=f"{grouping}_{kind}", weights="ols"))
            parts.append(clipped_weights(frames, designs, folds).assign(variant=f"{grouping}_{kind}", weights="clip01"))
    return pd.concat(parts, ignore_index=True)
