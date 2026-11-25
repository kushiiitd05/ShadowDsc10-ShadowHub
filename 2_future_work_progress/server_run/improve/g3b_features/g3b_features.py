"""Origin-anchored feature groups for the cs02 recipe: gap_aware, year_ago, peer_corr. Every value at origin o uses data dated <= o only.

Leakage guard: all arrays are cut with `Panel.upto(origin)` before anything is computed, so the last column IS the origin and
every look-back index is relative to it. Year-ago lookups for a target row at offset h = (Date - origin) use column D-1+h-k
(k >= 357), which is always < D-1 for h <= 42.

Conventions
  * "observed" = the raw file has a row for that hub and date; "observed open day" = observed, IsOpen == 1 and OrderVolume > 0.
    All level statistics use log1p(OrderVolume) on observed open days only; missing dates are never treated as zeros.
  * gap_aware (g3b_gap_*): per hub and window w in {28, 91, 182, 365}: mean, median, count of observed open days (n) and coverage
    (observed rows / w). Fallback when coverage < 0.5 or fewer than 3 observed open days (the hub is mostly unobserved):
    mean/median are replaced by the hub's EXPANDING mean/median up to the origin (all its observed open days); if the hub has none,
    by the median of the same window statistic over strong hubs of the same HubFormat x AssortmentTier (global median when the hub has
    no group). n and cov always stay the honest, un-filled values.  Row level: lag-364 value (NaN if closed/unobserved) + availability flag
    (1 = a row exists at d-364, 0 = unobserved).
  * year_ago (g3b_ya_1y_*, g3b_ya_2y_*): row level smoothed value (mean of d-s-7, d-s, d-s+7 on observed open days) and its count;
    hub level (per origin) 42-day year-ago window mean / closed-or-zero days / promo days / coverage, and uplift = window mean minus
    mean of the 42 days just before it.  s = 364 (1y) and 728 (2y).
  * peer_corr (g3b_peer_*): 5 most correlated other hubs (Pearson over the last 365 days up to the origin, pairwise complete,
    >= 120 overlapping observed open days); peers' mean 28/91-day level minus the hub's own, peers' mean year-ago uplift,
    mean correlation and number of peers found.
Public API: ALL_GROUPS, select_group_columns, add_features, add_features_multi, peer_correlation, top_peers.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
for _path in (_HERE, _HERE.parent / "g3_features"):       # feature_panel lives in the (read-only) g3_features folder
    if str(_path) not in sys.path:
        sys.path.append(str(_path))

from feature_panel import Panel, build_panel  # noqa: E402

ALL_GROUPS = ("gap_aware", "year_ago", "peer_corr")
GROUP_PREFIX = {"gap_aware": "g3b_gap_", "year_ago": "g3b_ya_", "peer_corr": "g3b_peer_"}
GAP_WINDOWS = (28, 91, 182, 365)
MIN_COVERAGE = 0.5            # observed rows / window below this -> fallback
MIN_OPEN_DAYS = 3             # fewer observed open days than this -> fallback
LAG = 364
YEAR_SHIFTS = {"1y": 364, "2y": 728}
SMOOTH_DELTAS = (-7, 0, 7)
WINDOW_DAYS = 42              # forecast horizon = year-ago window length
PEER_K = 5
PEER_MIN_OVERLAP = 120
PEER_WINDOW = 365
PEER_LEVEL_WINDOWS = (28, 91)
OUT_DTYPE = np.float32


def select_group_columns(columns: Iterable[str], groups: Iterable[str]) -> list[str]:
    prefixes = tuple(GROUP_PREFIX[g] for g in groups)
    return [c for c in columns if c.startswith(prefixes)]


# ------------------------------------------------------------------ low-level helpers
def _log_open(panel: Panel) -> np.ndarray:
    """log1p(orders) on observed open positive-volume days; NaN elsewhere (closed, zero, missing rows)."""
    keep = (panel.is_open == 1) & (panel.orders > 0)
    return np.where(keep, np.log1p(np.where(keep, panel.orders, 0.0)), np.nan)


def _bounds(lo: int, hi: int, width: int) -> tuple[int, int]:
    lo, hi = max(lo, 0), min(hi, width)
    return (lo, max(lo, hi))


def _nan_mean_count(arr: np.ndarray, lo: int, hi: int) -> tuple[np.ndarray, np.ndarray]:
    """Row mean and count of finite values in columns [lo, hi) (clipped to the array); NaN mean when the count is 0."""
    lo, hi = _bounds(lo, hi, arr.shape[1])
    seg = arr[:, lo:hi]
    ok = np.isfinite(seg)
    count = ok.sum(axis=1).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(count > 0, np.where(ok, seg, 0.0).sum(axis=1) / count, np.nan)
    return mean, count


def _nan_median(arr: np.ndarray, lo: int = 0, hi: int | None = None) -> np.ndarray:
    lo, hi = _bounds(lo, arr.shape[1] if hi is None else hi, arr.shape[1])
    seg = arr[:, lo:hi]
    if seg.shape[1] == 0:
        return np.full(arr.shape[0], np.nan)
    ok = np.isfinite(seg)
    srt = np.sort(np.where(ok, seg, np.inf), axis=1)         # finite values first, +inf padding last
    n = ok.sum(axis=1)
    rows = np.arange(seg.shape[0])
    lower, upper = np.maximum((n - 1) // 2, 0), np.maximum(n // 2, 0)
    med = 0.5 * (srt[rows, lower] + srt[rows, upper])
    return np.where(n > 0, med, np.nan)


def _count_true(mask: np.ndarray, lo: int, hi: int) -> np.ndarray:
    lo, hi = _bounds(lo, hi, mask.shape[1])
    return mask[:, lo:hi].sum(axis=1).astype(float)


def _group_ids(panel: Panel) -> np.ndarray | None:
    """HubFormat x AssortmentTier id per hub (-1 unknown); a single available grouping is used alone; None if no grouping."""
    parts = [panel.groups[k] for k in ("HubFormat", "AssortmentTier") if k in panel.groups]
    if not parts:
        return None
    key = parts[0].astype(np.int64)
    bad = key < 0
    for part in parts[1:]:
        key = key * (int(part.max()) + 2) + part
        bad |= part < 0
    return np.where(bad, -1, key)


def _group_median(values: np.ndarray, strong: np.ndarray, gid: np.ndarray | None) -> np.ndarray:
    """Median of `values` over strong hubs of the same group; global strong median when the group is unknown or empty."""
    good = np.where(strong & np.isfinite(values), values, np.nan)
    overall = np.nanmedian(good) if np.isfinite(good).any() else np.nan
    if gid is None:
        return np.full(len(values), overall)
    per_group = pd.Series(good).groupby(gid).transform("median").to_numpy()
    return np.where((gid >= 0) & np.isfinite(per_group), per_group, overall)


# ------------------------------------------------------------------ (A) gap_aware
def gap_hub_features(panel: Panel, y: np.ndarray) -> dict[str, np.ndarray]:
    """Hub-level (H,) window stats for GAP_WINDOWS with the documented fallback."""
    width = y.shape[1]
    observed = np.isfinite(panel.is_open)
    exp_mean, _ = _nan_mean_count(y, 0, width)
    exp_median = _nan_median(y)
    gid = _group_ids(panel)
    out: dict[str, np.ndarray] = {}
    for w in GAP_WINDOWS:
        mean, n_open = _nan_mean_count(y, width - w, width)
        median = _nan_median(y, width - w, width)
        cov = _count_true(observed, width - w, width) / w
        weak = (cov < MIN_COVERAGE) | (n_open < MIN_OPEN_DAYS)
        for name, value, expanding in (("mean", mean, exp_mean), ("median", median, exp_median)):
            group = _group_median(value, ~weak, gid)
            fill = np.where(np.isfinite(expanding), expanding, group)
            out[f"g3b_gap_{name}{w}"] = np.where(weak, fill, value)
        out[f"g3b_gap_n{w}"] = n_open
        out[f"g3b_gap_cov{w}"] = cov
    return out


def gap_row_features(panel: Panel, y: np.ndarray, hub_idx: np.ndarray, matched: np.ndarray, offsets: np.ndarray) -> dict[str, np.ndarray]:
    width = y.shape[1]
    col = width - 1 + offsets - LAG
    inside = matched & (col >= 0) & (col <= width - 1)
    rows, cols = np.where(inside, hub_idx, 0), np.where(inside, col, 0)
    value = np.where(inside, y[rows, cols], np.nan)
    avail = np.where(inside, np.isfinite(panel.is_open[rows, cols]), False)
    return {"g3b_gap_lag364": value, "g3b_gap_lag364_avail": avail.astype(float)}


# ------------------------------------------------------------------ (B) year_ago
def year_ago_window_stats(panel: Panel, y: np.ndarray, shift: int) -> dict[str, np.ndarray]:
    """Hub-level stats of the year-ago (`shift` days back) copy of the 42 days after the origin, and of the 42 days before that."""
    width = y.shape[1]
    lo, hi = width - shift, width - shift + WINDOW_DAYS
    mean, n_open = _nan_mean_count(y, lo, hi)
    prev_mean, _ = _nan_mean_count(y, lo - WINDOW_DAYS, lo)
    observed = np.isfinite(panel.is_open)
    n_obs = _count_true(observed, lo, hi)
    seen = n_obs > 0
    stats = {"win_mean": mean, "win_closed": np.where(seen, n_obs - n_open, np.nan), "win_cov": n_obs / WINDOW_DAYS,
             "uplift": mean - prev_mean}
    if panel.promo is not None:
        stats["win_promo"] = np.where(seen, _count_true(panel.promo == 1, lo, hi), np.nan)
    return stats


def year_ago_hub_features(panel: Panel, y: np.ndarray) -> dict[str, np.ndarray]:
    return {f"g3b_ya_{label}_{key}": value for label, shift in YEAR_SHIFTS.items()
            for key, value in year_ago_window_stats(panel, y, shift).items()}


def year_ago_row_features(y: np.ndarray, hub_idx: np.ndarray, matched: np.ndarray, offsets: np.ndarray) -> dict[str, np.ndarray]:
    """Mean of the observed open-day logs at d-s-7, d-s, d-s+7 per target row, and how many of the three were usable."""
    width = y.shape[1]
    out: dict[str, np.ndarray] = {}
    for label, shift in YEAR_SHIFTS.items():
        total, count = np.zeros(len(offsets)), np.zeros(len(offsets))
        for delta in SMOOTH_DELTAS:
            col = width - 1 + offsets - (shift + delta)
            inside = matched & (col >= 0) & (col <= width - 1)
            value = np.where(inside, y[np.where(inside, hub_idx, 0), np.where(inside, col, 0)], np.nan)
            ok = np.isfinite(value)
            total += np.where(ok, value, 0.0)
            count += ok
        with np.errstate(invalid="ignore", divide="ignore"):
            out[f"g3b_ya_{label}_smooth"] = np.where(count > 0, total / count, np.nan)
        out[f"g3b_ya_{label}_smooth_n"] = count
    return out


# ------------------------------------------------------------------ (C) peer_corr
def peer_correlation(y: np.ndarray, min_overlap: int = PEER_MIN_OVERLAP) -> np.ndarray:
    """(H,H) pairwise-complete Pearson correlation of the rows of `y` (NaN = unobserved); NaN where overlap < min_overlap, variance 0 or i == j."""
    ok = np.isfinite(y)
    mask = ok.astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        mu = np.where(ok.any(axis=1), np.where(ok, y, 0.0).sum(axis=1) / np.maximum(ok.sum(axis=1), 1), 0.0)
        yc = np.where(ok, y - mu[:, None], 0.0)             # per-hub centring leaves Pearson unchanged and tames cancellation
        n = mask @ mask.T
        sx = yc @ mask.T                                     # sum of x_i over days shared with j
        sxx = (yc * yc) @ mask.T
        sxy = yc @ yc.T
        var_i = n * sxx - sx * sx
        cov = n * sxy - sx * sx.T
        corr = cov / np.sqrt(var_i * var_i.T)
    corr = np.where((n >= min_overlap) & (var_i > 1e-12) & (var_i.T > 1e-12), np.clip(corr, -1.0, 1.0), np.nan)
    np.fill_diagonal(corr, np.nan)
    return corr


def top_peers(corr: np.ndarray, k: int = PEER_K) -> tuple[np.ndarray, np.ndarray]:
    """(H,k) peer indices ordered by decreasing correlation (-1 padding) and their correlations (NaN padding)."""
    score = np.where(np.isfinite(corr), corr, -np.inf)
    k = min(k, corr.shape[1])
    order = np.argsort(-score, axis=1, kind="stable")[:, :k]
    best = np.take_along_axis(score, order, axis=1)
    found = np.isfinite(best)
    return np.where(found, order, -1), np.where(found, best, np.nan)


def _peer_mean(values: np.ndarray, peers: np.ndarray) -> np.ndarray:
    """Mean over each hub's peers of values[peer], ignoring -1 padding and NaN values."""
    gathered = np.where(peers >= 0, values[np.maximum(peers, 0)], np.nan)
    ok = np.isfinite(gathered)
    count = ok.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, np.where(ok, gathered, 0.0).sum(axis=1) / np.maximum(count, 1), np.nan)


def peer_hub_features(panel: Panel, y: np.ndarray) -> dict[str, np.ndarray]:
    width = y.shape[1]
    peers, corr = top_peers(peer_correlation(y[:, max(width - PEER_WINDOW, 0):]), PEER_K)
    out: dict[str, np.ndarray] = {}
    for w in PEER_LEVEL_WINDOWS:
        own, _ = _nan_mean_count(y, width - w, width)
        out[f"g3b_peer_rel{w}"] = _peer_mean(own, peers) - own
    out["g3b_peer_uplift_1y"] = _peer_mean(year_ago_window_stats(panel, y, YEAR_SHIFTS["1y"])["uplift"], peers)
    out["g3b_peer_corr_mean"] = _row_mean(corr)
    out["g3b_peer_n"] = (peers >= 0).sum(axis=1).astype(float)
    return out


def _row_mean(values: np.ndarray) -> np.ndarray:
    ok = np.isfinite(values)
    count = ok.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, np.where(ok, values, 0.0).sum(axis=1) / np.maximum(count, 1), np.nan)


# ------------------------------------------------------------------ assembly
def _gather(table: np.ndarray, hub_idx: np.ndarray, matched: np.ndarray) -> np.ndarray:
    return np.where(matched, table[np.where(matched, hub_idx, 0)], np.nan)


def features_for_origin(panel: Panel, hubs: np.ndarray, offsets: np.ndarray, groups: tuple[str, ...]) -> dict[str, np.ndarray]:
    """Feature columns (name -> (n,) float32) for rows (hub id, target offset in days after the origin); `panel` must end at the origin."""
    pos = np.searchsorted(panel.hubs, hubs)
    hub_idx = np.minimum(pos, len(panel.hubs) - 1)
    matched = panel.hubs[hub_idx] == hubs
    y = _log_open(panel)
    tables: dict[str, np.ndarray] = {}
    rows: dict[str, np.ndarray] = {}
    if "gap_aware" in groups:
        tables.update(gap_hub_features(panel, y))
        rows.update(gap_row_features(panel, y, hub_idx, matched, offsets))
    if "year_ago" in groups:
        tables.update(year_ago_hub_features(panel, y))
        rows.update(year_ago_row_features(y, hub_idx, matched, offsets))
    if "peer_corr" in groups:
        tables.update(peer_hub_features(panel, y))
    cols = {name: _gather(table, hub_idx, matched) for name, table in tables.items()}
    cols.update({name: np.where(matched, value, np.nan) for name, value in rows.items()})
    return {name: value.astype(OUT_DTYPE) for name, value in sorted(cols.items())}


def _check(frame: pd.DataFrame, groups: Iterable[str]) -> None:
    unknown = set(groups) - set(ALL_GROUPS)
    if unknown:
        raise ValueError(f"unknown feature groups: {sorted(unknown)}")
    missing = {"HubID", "Date"} - set(frame.columns)
    if missing:
        raise ValueError(f"frame lacks columns: {sorted(missing)}")


def _attach(frame: pd.DataFrame, extra: pd.DataFrame) -> pd.DataFrame:
    clash = set(extra.columns) & set(frame.columns)
    if clash:
        raise ValueError(f"frame already has feature columns: {sorted(clash)[:5]}")
    kwargs = {} if int(pd.__version__.split(".")[0]) >= 3 else {"copy": False}
    return pd.concat([frame, extra], axis=1, **kwargs)


def _offsets(dates: pd.Series, origin: pd.Timestamp) -> np.ndarray:
    return ((pd.to_datetime(dates) - pd.Timestamp(origin)).dt.days).to_numpy().astype(np.int64)


def _as_panel(raw_history: pd.DataFrame | Panel, origin: pd.Timestamp | None = None) -> Panel:
    if isinstance(raw_history, Panel):
        return raw_history
    data = raw_history if origin is None else raw_history[pd.to_datetime(raw_history["Date"]) <= origin]
    return build_panel(data)


def add_features(frame: pd.DataFrame, raw_history: pd.DataFrame | Panel, origin: pd.Timestamp,
                 groups: tuple[str, ...] = ALL_GROUPS) -> pd.DataFrame:
    """New frame with the g3b columns for one forecast origin (rows are target days after `origin`). Later raw rows are dropped first."""
    _check(frame, groups)
    origin = pd.Timestamp(origin)
    panel = _as_panel(raw_history, origin).upto(origin)
    cols = features_for_origin(panel, frame["HubID"].to_numpy(), _offsets(frame["Date"], origin), tuple(groups))
    return _attach(frame, pd.DataFrame(cols, index=frame.index))


def add_features_multi(frame: pd.DataFrame, raw_history: pd.DataFrame | Panel, origin_col: str = "_origin",
                       groups: tuple[str, ...] = ALL_GROUPS) -> pd.DataFrame:
    """Like add_features for a stacked frame with one origin per row (column `origin_col`); the index must be unique."""
    _check(frame, groups)
    if not frame.index.is_unique:
        raise ValueError("frame index must be unique")
    panel = _as_panel(raw_history)
    origins = pd.to_datetime(frame[origin_col]).to_numpy()
    hubs, dates = frame["HubID"].to_numpy(), pd.to_datetime(frame["Date"])
    pieces = []
    for origin in np.unique(origins):
        pos = np.flatnonzero(origins == origin)
        stamp = pd.Timestamp(origin)
        cols = features_for_origin(panel.upto(stamp), hubs[pos], _offsets(dates.iloc[pos], stamp), tuple(groups))
        pieces.append(pd.DataFrame(cols, index=frame.index[pos]))
    extra = pd.concat(pieces).reindex(frame.index) if pieces else pd.DataFrame(index=frame.index)
    return _attach(frame, extra)
