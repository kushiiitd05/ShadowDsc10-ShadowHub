"""Pure helpers for the screening harness (no I/O, no mutation of inputs)."""
from __future__ import annotations

import numpy as np

LEVEL_COLUMNS = ("Orders_90D_MedianOpen", "Orders_180D_MedianOpen", "Orders_365D_MedianOpen", "HistHubDOWOrdersMean")


def level_baseline(columns: dict[str, np.ndarray]) -> np.ndarray:
    """Recent hub level L: first strictly positive of the fallback chain, else 1.0.

    Every input is a per-origin history statistic, so L never sees the forecast window.
    """
    n = len(next(iter(columns.values())))
    level = np.full(n, np.nan)
    for name in LEVEL_COLUMNS:
        values = np.asarray(columns[name], dtype=float)
        usable = np.isnan(level) & np.isfinite(values) & (values > 0)
        level = np.where(usable, values, level)
    return np.where(np.isnan(level), 1.0, level)


def encode_target(actual: np.ndarray, level: np.ndarray, mode: str) -> np.ndarray:
    """Training target for a variant: 'log' = log1p(y); 'level' = log1p(y) - log1p(L); 'level_init' = raw y."""
    if mode == "log":
        return np.log1p(actual)
    if mode == "level":
        return np.log1p(actual) - np.log1p(level)
    if mode == "level_init":
        return np.asarray(actual, dtype=float)
    raise ValueError(f"unknown target mode: {mode}")


def decode_prediction(raw: np.ndarray, level: np.ndarray, mode: str, offset: float = 0.0) -> np.ndarray:
    """Map a model's RAW score back to order volume.

    `raw` must be the raw score. For 'level_init' LightGBM does not add init_score at
    predict time, so log(L) is added here by hand.
    """
    if mode == "log":
        return np.clip(np.expm1(raw + offset), 0.0, None)
    if mode == "level":
        return np.clip(np.expm1(raw + np.log1p(level) + offset), 0.0, None)
    if mode == "level_init":
        return np.clip(np.exp(raw + np.log(level) + offset), 0.0, None)
    raise ValueError(f"unknown target mode: {mode}")


def rmsle(actual: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.log1p(np.clip(prediction, 0, None)) - np.log1p(actual)) ** 2)))


def fold_scores(actual: np.ndarray, prediction: np.ndarray, is_open: np.ndarray) -> dict[str, float]:
    """RMSLE on all rows, RMSLE without open-but-zero rows, and mean log bias on open positive rows."""
    open_zero = (is_open == 1) & (actual == 0)
    positive = (is_open == 1) & (actual > 0)
    bias = float(np.mean(np.log1p(prediction[positive]) - np.log1p(actual[positive])))
    return {
        "rmsle_all": rmsle(actual, prediction),
        "rmsle_excl_open0": rmsle(actual[~open_zero], prediction[~open_zero]),
        "bias_open": bias,
        "n_open_zero": int(open_zero.sum()),
    }


def recency_weights(origin_dates: np.ndarray, validation_origin: np.datetime64, half_life_days: float) -> np.ndarray:
    """0.5 ** (days between a training origin and the validation origin / half life)."""
    # force day units on both sides: mixing ns and D made the difference come out in nanoseconds and every weight 0
    days = (np.datetime64(validation_origin, "D") - origin_dates.astype("datetime64[D]")).astype(float)
    return 0.5 ** (days / half_life_days)
