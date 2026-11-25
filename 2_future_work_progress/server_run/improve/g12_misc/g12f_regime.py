"""Item F: regime experts. Separate LightGBMs for special days (Sunday or regional holiday) and ordinary days, blended with cs02 by a calendar gate.

special = (Weekday == 7) or (RegionalHoliday > 0).  Open special days carry only 1-3 % of the squared error of cs02 (dev folds), so even a perfect
special expert could lower RMSLE by at most ~0.0016; the experiment measures how much of that a separate model recovers.
Pure helpers only (flag, regime frames, log-space blend); the model fitting lives in g12f_runner.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SPECIAL_EXPERT_ROUNDS = 200

# pre-declared blend gates: (weight of the expert on special rows, weight of the expert on ordinary rows); the remainder is the global cs02 prediction
BLENDS: dict[str, tuple[float, float]] = {
    "g12f_hard": (1.0, 1.0),
    "g12f_half": (0.5, 0.5),
    "g12f_spec": (1.0, 0.0),
    "g12f_spec_half": (0.5, 0.0),
}


def special_flag(frame: pd.DataFrame) -> np.ndarray:
    """True for Sunday or regional-holiday rows (numeric coercion of both columns; missing -> not special)."""
    weekday = pd.to_numeric(frame["Weekday"], errors="coerce").to_numpy()
    holiday = pd.to_numeric(frame["RegionalHoliday"], errors="coerce").fillna(0).to_numpy()
    return (weekday == 7) | (holiday > 0)


def regime_ctx(ctx: dict, special: bool) -> dict:
    """New ctx whose train/hold frames only hold the rows of one regime (validation keeps every row; the caller reads the regime rows)."""
    out = {**ctx}
    for key in ("train", "hold"):
        frame = ctx[key]
        keep = special_flag(frame) == special
        out[key] = frame.loc[keep].reset_index(drop=True)
    return out


def blend_log(p_global: np.ndarray, p_expert_special: np.ndarray, p_expert_ordinary: np.ndarray, special: np.ndarray, gate: tuple[float, float]) -> np.ndarray:
    """Log-space blend: g * log1p(expert of the row's regime) + (1 - g) * log1p(global), g = gate[0] on special rows, gate[1] on ordinary rows.

    Rows where the chosen expert predicts exactly 0 while the global prediction is positive keep the global value (closed rows stay 0 in both).
    """
    expert = np.where(special, p_expert_special, p_expert_ordinary)
    g = np.where(special, gate[0], gate[1])
    mixed = g * np.log1p(np.clip(expert, 0, None)) + (1 - g) * np.log1p(np.clip(p_global, 0, None))
    out = np.expm1(mixed)
    out = np.where((expert <= 0) & (p_global > 0), p_global, out)
    return np.clip(out, 0.0, None)


def join_global(valid_oof: pd.DataFrame, global_oof: pd.DataFrame) -> np.ndarray:
    """cs02 predictions aligned with `valid_oof` rows by (fold, HubID, Date); raises when a row has no global prediction."""
    key = ["fold", "HubID", "Date"]
    left = valid_oof[key].assign(_row=np.arange(len(valid_oof)))
    merged = left.merge(global_oof[[*key, "pred"]], on=key, how="left", validate="one_to_one")
    if merged["pred"].isna().any():
        raise ValueError(f"{int(merged['pred'].isna().sum())} validation rows without a global cs02 prediction")
    return merged.sort_values("_row")["pred"].to_numpy(float)
