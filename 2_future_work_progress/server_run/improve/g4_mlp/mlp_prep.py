"""Preprocessing for the entity-embedding MLP: fit statistics on a training frame, transform any frame.

Pure functions; inputs are never mutated. All statistics (medians, means, stds, vocabularies,
missing-indicator set, log-transformed column set) come from the frame given to `fit_preprocessor`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

MISSING_TOKEN = "__MISSING__"
UNKNOWN_INDEX = 0  # index 0 of every embedding table = category never seen in training (zero vector)


@dataclass(frozen=True)
class Preprocessor:
    num_cols: list[str]
    cat_cols: list[str]
    miss_cols: list[str]        # numeric columns that get a missing indicator (>1% missing in the training frame)
    log_cols: list[str]         # non-negative heavy-tailed columns passed through log1p before standardising
    medians: dict[str, float]   # computed on the (transformed) training values
    means: dict[str, float]
    stds: dict[str, float]
    vocab: dict[str, dict[str, int]]
    clip: float = 10.0

    @property
    def n_features(self) -> int:
        return len(self.num_cols) + len(self.miss_cols)

    def cat_cardinalities(self) -> list[int]:
        return [len(self.vocab[c]) + 1 for c in self.cat_cols]  # +1 for the unknown slot at index 0


def _is_categorical(series: pd.Series, name: str, categorical: set[str]) -> bool:
    numeric = pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series)
    return name in categorical or not numeric  # object / string / str / category dtypes are all categorical


def _numeric_values(series: pd.Series) -> np.ndarray:
    """float64 view of a column with +/-inf mapped to NaN (bool -> 0/1)."""
    values = series.to_numpy(dtype="float64", na_value=np.nan)
    return np.where(np.isfinite(values), values, np.nan)


def _category_keys(series: pd.Series) -> pd.Series:
    """String keys that are identical for int/float/string spellings of the same value (e.g. 3, 3.0, '3')."""
    if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
        keys = pd.Series(_numeric_values(series), index=series.index).astype("string")
        keys = keys.str.replace(r"\.0$", "", regex=True)
    else:
        keys = series.astype("string").str.replace(r"\.0$", "", regex=True)
    return keys.fillna(MISSING_TOKEN)


def _signed_log1p(values: np.ndarray) -> np.ndarray:
    return np.log1p(np.clip(values, 0.0, None))


def fit_preprocessor(frame: pd.DataFrame, columns: list[str], categorical: set[str], missing_threshold: float = 0.01,
                     log_threshold: float = 100.0, clip: float = 10.0) -> Preprocessor:
    num_cols: list[str] = []
    cat_cols: list[str] = []
    for c in columns:
        (cat_cols if _is_categorical(frame[c], c, categorical) else num_cols).append(c)
    miss_cols: list[str] = []
    log_cols: list[str] = []
    medians: dict[str, float] = {}
    means: dict[str, float] = {}
    stds: dict[str, float] = {}
    for c in num_cols:
        values = _numeric_values(frame[c])
        observed = values[~np.isnan(values)]
        n_missing = len(values) - len(observed)
        if len(observed) == 0:
            medians[c], means[c], stds[c] = 0.0, 0.0, 1.0
        else:
            if observed.min() >= 0 and observed.max() > log_threshold:
                log_cols.append(c)
                observed = _signed_log1p(observed)
            medians[c] = float(np.median(observed))
            # mean/std of the median-imputed column, computed without materialising it
            mean = (observed.sum() + n_missing * medians[c]) / len(values)
            var = (np.square(observed - mean).sum() + n_missing * (medians[c] - mean) ** 2) / len(values)
            means[c] = float(mean)
            stds[c] = float(np.sqrt(var)) if var > 1e-12 else 1.0
        if n_missing / max(len(values), 1) > missing_threshold:
            miss_cols.append(c)
    vocab = {}
    for c in cat_cols:
        levels = sorted(_category_keys(frame[c]).unique().tolist())
        vocab[c] = {level: i + 1 for i, level in enumerate(levels)}
    return Preprocessor(num_cols, cat_cols, miss_cols, log_cols, medians, means, stds, vocab, clip)


def transform(pre: Preprocessor, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return (float32 [n, n_features], int64 [n, n_cat]); numeric block = standardised columns then missing indicators."""
    n = len(frame)
    x_num = np.empty((n, pre.n_features), dtype=np.float32)
    log_set = set(pre.log_cols)
    for j, c in enumerate(pre.num_cols):
        values = _numeric_values(frame[c])
        nan_mask = np.isnan(values)
        values = np.where(nan_mask, np.nan, _signed_log1p(values) if c in log_set else values)
        values = np.where(nan_mask, pre.medians[c], values)
        x_num[:, j] = np.clip((values - pre.means[c]) / pre.stds[c], -pre.clip, pre.clip)
    for k, c in enumerate(pre.miss_cols):
        x_num[:, len(pre.num_cols) + k] = np.isnan(_numeric_values(frame[c]))
    x_cat = np.empty((n, len(pre.cat_cols)), dtype=np.int64)
    for k, c in enumerate(pre.cat_cols):
        x_cat[:, k] = _category_keys(frame[c]).map(pre.vocab[c]).fillna(UNKNOWN_INDEX).to_numpy(dtype=np.int64)
    return x_num, x_cat
