"""Item D: shift adaptation. Shift fold-k predictions by the mean residual of the most recent days before the origin.

The residual comes from the cs02 predictions made at the PREVIOUS origin (the previous fold's out-of-fold rows, whose 42-day window ends
exactly at this fold's origin): so only folds 2, 3, 4 can be corrected; fold 1 (no earlier origin) and fold 5 (a different season, no OOF
rows for its previous origin) are skipped. The last days of a window are the model's horizon-30..42 predictions, so the residual is
a noisy proxy of the current level error.
Variants: literal (shift = residual mean, beta fixed at 1), fit (one beta, leave-fold-out), fit_h (a beta per horizon bucket).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import offline_common as oc

WINDOWS = (14, 28, 42)
HORIZON_BUCKETS = ((1, 14), (15, 28), (29, 42))


def own_recent_residual(frames: dict[int, pd.DataFrame], fold: int, window: int) -> pd.Series | None:
    """Per HubID mean residual over the last `window` days of the previous fold's window; None when there is no previous fold."""
    prev = oc.previous_fold(fold)
    if prev is None or prev not in frames:
        return None
    end = frames[prev]["Date"].max()
    return oc.hub_mean_residual(frames[prev], window, end)


def horizon_days(frame: pd.DataFrame, fold: int) -> np.ndarray:
    return (frame["Date"] - pd.Timestamp(oc.ORIGINS[fold])).dt.days.to_numpy()


def design(frames: dict[int, pd.DataFrame], fold: int, window: int, by_horizon: bool) -> np.ndarray | None:
    m = own_recent_residual(frames, fold, window)
    if m is None:
        return None
    col = frames[fold]["HubID"].map(m).to_numpy(float)
    if not by_horizon:
        return col.reshape(-1, 1)
    h = horizon_days(frames[fold], fold)
    return np.column_stack([np.where((h >= lo) & (h <= hi), col, 0.0) for lo, hi in HORIZON_BUCKETS])


def evaluate_literal(frames: dict[int, pd.DataFrame], folds: tuple[int, ...], window: int) -> pd.DataFrame:
    rows = []
    for f in folds:
        d = design(frames, f, window, False)
        base = oc.scores(frames[f])
        if d is None:
            rows.append({"fold": f, "base_all": base["all"], "base_excl": base["excl_open0"], "new_all": np.nan, "new_excl": np.nan, "beta": ""})
            continue
        new = oc.scores(frames[f], oc.apply_shift(frames[f], np.nan_to_num(d[:, 0])))
        rows.append({"fold": f, "base_all": base["all"], "base_excl": base["excl_open0"], "new_all": new["all"], "new_excl": new["excl_open0"], "beta": "1"})
    out = pd.DataFrame(rows)
    out["gain_all"], out["gain_excl"] = out["base_all"] - out["new_all"], out["base_excl"] - out["new_excl"]
    return out


def run(frames: dict[int, pd.DataFrame], folds: tuple[int, ...] = (1, 2, 3, 4, 5)) -> pd.DataFrame:
    """All D variants -> one long table (variant, window, fold, base/new RMSLE, gains, beta)."""
    parts = []
    for window in WINDOWS:
        parts.append(evaluate_literal(frames, folds, window).assign(variant="literal", window=window))
        for name, by_h in (("fit", False), ("fit_h", True)):
            designs = {f: d for f in folds if (d := design(frames, f, window, by_h)) is not None}
            parts.append(oc.evaluate_correction(frames, designs, fit_folds=oc.DEV_FOLDS, eval_folds=folds).assign(variant=name, window=window))
    return pd.concat(parts, ignore_index=True)
