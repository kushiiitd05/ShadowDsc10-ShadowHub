"""Item A: low-rank (truncated SVD) denoising of the hub x day log1p order matrix, origin-anchored.

For an origin o the matrix is built from the last WINDOW days up to and including o (nothing later is read):
log1p(orders) on observed open positive-volume days, NaN elsewhere. Days on which fewer than MIN_OPEN_FRAC of the observed hubs
were open (Sundays, public holidays) are dropped from the matrix (they would only teach the SVD the closure pattern). Each hub is
centred on its own mean, missing cells are filled by iterative rank-k SVD imputation, and the final rank-k fit gives
  * g12a_load{1..k}      hub loadings (unit-scale), rotated onto the basis of the first training origin (orthogonal Procrustes) so that a
                         column means the same thing at every origin (raw SVD signs / rotations are arbitrary per origin)
  * g12a_comp{1..3}_{7d,28d}   mean of the top-3 aligned time components over the last 7 / 28 kept days (global per origin)
  * g12a_lvl{7,28,91}    denoised level: mean rank-k reconstruction (+ hub mean) over the last 7 / 28 / 91 kept days
  * g12a_noise28         raw 28-day open-day mean minus the denoised 28-day level;  g12a_resid_sd91: sd of raw - reconstruction (91 d)
  * g12a_ya_recon        row level: denoised value at d - 364 (NaN when that day is outside the window or was dropped)
Public API: add_features, add_features_multi, denoise, procrustes_rotation, hub_features, ALL_PREFIX.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
for _path in (_HERE.parent / "g3_features", _HERE.parent / "g3b_features"):
    if str(_path) not in sys.path:
        sys.path.append(str(_path))

from feature_panel import Panel, build_panel  # noqa: E402

ALL_PREFIX = "g12a_"
RANK = 14
WINDOW = 728
MIN_OPEN_FRAC = 0.3
N_IMPUTE = 5
N_COMP = 3
LEVEL_WINDOWS = (7, 28, 91)
YEAR_LAG = 364
OUT_DTYPE = np.float32


def log_open(panel: Panel) -> np.ndarray:
    """log1p(orders) on observed open positive-volume days; NaN elsewhere."""
    keep = (panel.is_open == 1) & (panel.orders > 0)
    return np.where(keep, np.log1p(np.where(keep, panel.orders, 0.0)), np.nan)


def kept_columns(y: np.ndarray, min_open_frac: float = MIN_OPEN_FRAC) -> np.ndarray:
    """Boolean (D,) mask of days on which at least `min_open_frac` of the hubs that have any data in the window were open."""
    ok = np.isfinite(y)
    has_data = ok.any(axis=1)
    denom = max(int(has_data.sum()), 1)
    return ok[has_data].sum(axis=0) / denom >= min_open_frac


def denoise(y: np.ndarray, rank: int = RANK, n_impute: int = N_IMPUTE) -> dict[str, np.ndarray]:
    """Rank-`rank` SVD of the hub-centred matrix `y` (NaN = missing) with iterative imputation.

    Returns mu (H,), U (H,k), s (k,), Vt (k,T) and recon (H,T) = U diag(s) Vt + mu (every cell, also the missing ones).
    """
    ok = np.isfinite(y)
    count = ok.sum(axis=1)
    mu = np.where(count > 0, np.where(ok, y, 0.0).sum(axis=1) / np.maximum(count, 1), 0.0)
    centred = np.where(ok, y - mu[:, None], 0.0)
    rank = int(min(rank, min(y.shape)))
    fill = centred
    for _ in range(max(n_impute, 1)):
        u, s, vt = np.linalg.svd(fill, full_matrices=False)
        low = (u[:, :rank] * s[:rank]) @ vt[:rank]
        fill = np.where(ok, centred, low)
    u, s, vt = np.linalg.svd(fill, full_matrices=False)
    u, s, vt = u[:, :rank], s[:rank], vt[:rank]
    return {"mu": mu, "U": u, "s": s, "Vt": vt, "recon": (u * s) @ vt + mu[:, None]}


def procrustes_rotation(u: np.ndarray, u_ref: np.ndarray) -> np.ndarray:
    """Orthogonal (k,k) R minimising ||u R - u_ref||_F; u and u_ref must have the same number of columns."""
    w, _, zt = np.linalg.svd(u.T @ u_ref)
    return w @ zt


class Reference:
    """Hub-loading basis (H,k) of the first training origin; every later origin is rotated onto it."""

    def __init__(self, u_ref: np.ndarray):
        self.u_ref = u_ref


def _window(y: np.ndarray, width: int, window: int) -> tuple[np.ndarray, int]:
    start = max(width - window, 0)
    return y[:, start:], start


def _recent_mean(values: np.ndarray, day_index: np.ndarray, width: int, w: int) -> np.ndarray:
    """Mean over kept days with panel index in (width - w, width - 1] of values (H,T) or (T,) -> NaN when no such day."""
    sel = day_index >= width - w
    if not sel.any():
        shape = values.shape[:-1] if values.ndim > 1 else ()
        return np.full(shape, np.nan)
    return values[..., sel].mean(axis=-1)


def hub_features(panel: Panel, rank: int = RANK, reference: Reference | None = None, window: int = WINDOW) -> tuple[dict[str, np.ndarray], dict]:
    """Hub-level (H,) feature columns for a panel that ends at the origin, plus a state dict used for the row features."""
    y_all = log_open(panel)
    width = y_all.shape[1]
    y_win, start = _window(y_all, width, window)
    keep = kept_columns(y_win)
    if keep.sum() < max(rank + 2, 10):
        return {}, {"start": start, "width": width, "keep": keep, "recon": None}
    y = y_win[:, keep]
    day_index = np.flatnonzero(keep) + start
    fit = denoise(y, rank)
    u, s, vt, recon, mu = fit["U"], fit["s"], fit["Vt"], fit["recon"], fit["mu"]
    k = u.shape[1]
    if reference is not None and reference.u_ref.shape[1] == k:
        rot = procrustes_rotation(u, reference.u_ref)
    else:
        rot = np.eye(k)
    u_al = u @ rot                                    # (H,k) in reference basis
    comp = rot.T @ (s[:, None] * vt)                  # (k,T): u diag(s) vt = u_al @ comp
    seen = np.isfinite(y).any(axis=1)
    out: dict[str, np.ndarray] = {}
    scale = np.sqrt(u.shape[0])
    for j in range(k):
        out[f"g12a_load{j + 1}"] = np.where(seen, u_al[:, j] * scale, np.nan)
    for j in range(min(N_COMP, k)):
        for w in (7, 28):
            out[f"g12a_comp{j + 1}_{w}d"] = np.full(len(seen), _recent_mean(comp[j], day_index, width, w))
    for w in LEVEL_WINDOWS:
        out[f"g12a_lvl{w}"] = np.where(seen, _recent_mean(recon, day_index, width, w), np.nan)
    recent = day_index >= width - 28
    obs = np.where(recent[None, :], y, np.nan)
    n_obs = np.isfinite(obs).sum(axis=1)
    raw28 = np.where(n_obs > 0, np.where(np.isfinite(obs), obs, 0.0).sum(axis=1) / np.maximum(n_obs, 1), np.nan)
    out["g12a_noise28"] = np.where(seen, raw28 - out["g12a_lvl28"], np.nan)
    wide = day_index >= width - 91
    res = np.where(wide[None, :] & np.isfinite(y), y - recon, np.nan)
    n_res = np.isfinite(res).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out["g12a_resid_sd91"] = np.where(n_res > 2, np.sqrt(np.nansum(res ** 2, axis=1) / np.maximum(n_res, 1)), np.nan)
    return out, {"start": start, "width": width, "keep": keep, "recon": recon, "day_index": day_index, "u_al": u_al}


def row_features(state: dict, hub_idx: np.ndarray, matched: np.ndarray, offsets: np.ndarray) -> dict[str, np.ndarray]:
    """g12a_ya_recon: denoised value at date d - 364 for each row (hub idx, target offset in days after the origin)."""
    n = len(offsets)
    if state.get("recon") is None:
        return {"g12a_ya_recon": np.full(n, np.nan)}
    col = state["width"] - 1 + offsets - YEAR_LAG
    pos = np.full(state["width"], -1, dtype=np.int64)
    pos[state["day_index"]] = np.arange(len(state["day_index"]))
    inside = matched & (col >= state["start"]) & (col <= state["width"] - 1)
    where = np.where(inside, pos[np.clip(col, 0, state["width"] - 1)], -1)
    ok = where >= 0
    value = np.where(ok, state["recon"][np.where(ok, hub_idx, 0), np.where(ok, where, 0)], np.nan)
    return {"g12a_ya_recon": value}


def make_reference(panel: Panel, first_origin: pd.Timestamp, rank: int = RANK) -> Reference:
    """Basis of the earliest origin (causal for every later origin)."""
    y = log_open(panel.upto(first_origin))
    y_win, _ = _window(y, y.shape[1], WINDOW)
    keep = kept_columns(y_win)
    return Reference(denoise(y_win[:, keep], rank)["U"])


def _offsets(dates: pd.Series, origin: pd.Timestamp) -> np.ndarray:
    return ((pd.to_datetime(dates) - pd.Timestamp(origin)).dt.days).to_numpy().astype(np.int64)


def features_for_origin(panel: Panel, hubs: np.ndarray, offsets: np.ndarray, rank: int = RANK, reference: Reference | None = None,
                        cache: dict | None = None) -> dict[str, np.ndarray]:
    """`panel` must end at the origin. The hub-level part is cached per (origin date, rank) when `cache` is given."""
    key = (pd.Timestamp(panel.dates[-1]), rank)
    if cache is not None and key in cache:
        table, state = cache[key]
    else:
        table, state = hub_features(panel, rank, reference)
        if cache is not None:
            cache[key] = (table, state)
    pos = np.searchsorted(panel.hubs, hubs)
    hub_idx = np.minimum(pos, len(panel.hubs) - 1)
    matched = panel.hubs[hub_idx] == hubs
    cols = {name: np.where(matched, v[hub_idx], np.nan) for name, v in table.items()}
    cols.update({name: np.where(matched, v, np.nan) for name, v in row_features(state, hub_idx, matched, offsets).items()})
    return {name: v.astype(OUT_DTYPE) for name, v in sorted(cols.items())}


def _check(frame: pd.DataFrame) -> None:
    missing = {"HubID", "Date"} - set(frame.columns)
    if missing:
        raise ValueError(f"frame lacks columns: {sorted(missing)}")
    clash = [c for c in frame.columns if c.startswith(ALL_PREFIX)]
    if clash:
        raise ValueError(f"frame already has feature columns: {clash[:5]}")


def _attach(frame: pd.DataFrame, extra: pd.DataFrame) -> pd.DataFrame:
    kwargs = {} if int(pd.__version__.split(".")[0]) >= 3 else {"copy": False}
    return pd.concat([frame, extra], axis=1, **kwargs)


def _as_panel(history: pd.DataFrame | Panel) -> Panel:
    return history if isinstance(history, Panel) else build_panel(history)


def add_features(frame: pd.DataFrame, history: pd.DataFrame | Panel, origin: pd.Timestamp, rank: int = RANK,
                 reference: Reference | None = None, cache: dict | None = None) -> pd.DataFrame:
    """New frame with the g12a columns for one origin (rows are target days after it). Later history rows are dropped first."""
    _check(frame)
    origin = pd.Timestamp(origin)
    panel = _as_panel(history).upto(origin)
    cols = features_for_origin(panel, frame["HubID"].to_numpy(), _offsets(frame["Date"], origin), rank, reference, cache)
    return _attach(frame, pd.DataFrame(cols, index=frame.index))


def add_features_multi(frame: pd.DataFrame, history: pd.DataFrame | Panel, origin_col: str = "_origin", rank: int = RANK,
                       reference: Reference | None = None, cache: dict | None = None) -> pd.DataFrame:
    """add_features for a stacked frame with one origin per row (unique index required)."""
    _check(frame)
    if not frame.index.is_unique:
        raise ValueError("frame index must be unique")
    panel = _as_panel(history)
    origins = pd.to_datetime(frame[origin_col]).to_numpy()
    hubs, dates = frame["HubID"].to_numpy(), pd.to_datetime(frame["Date"])
    pieces = []
    for origin in np.unique(origins):
        pos = np.flatnonzero(origins == origin)
        stamp = pd.Timestamp(origin)
        cols = features_for_origin(panel.upto(stamp), hubs[pos], _offsets(dates.iloc[pos], stamp), rank, reference, cache)
        pieces.append(pd.DataFrame(cols, index=frame.index[pos]))
    extra = pd.concat(pieces).reindex(frame.index) if pieces else pd.DataFrame(index=frame.index)
    return _attach(frame, extra)
