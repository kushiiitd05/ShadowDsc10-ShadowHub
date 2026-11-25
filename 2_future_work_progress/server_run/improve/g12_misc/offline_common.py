"""Shared pure helpers for the offline items C, D, E (post-hoc corrections of saved out-of-fold predictions).

Convention: all corrections act in log1p space on rows with IsOpen == 1 and pred > 0; every other row keeps its prediction.
A correction is `lp' = lp + X @ beta`, where X holds origin-anchored features (they use only actuals and earlier-origin predictions
dated <= the fold origin) and beta is the pooled least-squares fit of the log residual on X over the OTHER folds (leave-fold-out).
Inputs are never mutated.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

KEYS = ["fold", "HubID", "Date"]
ORIGINS = {1: "2015-01-02", 2: "2015-02-13", 3: "2015-03-27", 4: "2015-05-08", 5: "2014-06-19"}
DEV_FOLDS = (2, 3, 4)
SEEDS = (20260807, 20260808, 20260809)
GATE_GAIN = 0.0024
NOISE = 0.0012


def rmsle(actual: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.log1p(np.clip(pred, 0, None)) - np.log1p(actual)) ** 2)))


def load_oof(path: Path, folds: tuple[int, ...] | None = None) -> pd.DataFrame:
    """Columns fold, HubID, Date (datetime), IsOpen, y, pred, sorted by fold/Date/HubID; unique keys enforced."""
    frame = pd.read_csv(path, parse_dates=["Date"]).rename(columns={"ActualOrderVolume": "y"})
    frame = frame[["fold", "HubID", "Date", "IsOpen", "y", "pred"]]
    if folds is not None:
        frame = frame[frame["fold"].isin(folds)]
    if frame.duplicated(KEYS).any():
        raise ValueError("duplicate (fold, HubID, Date) keys in OOF file")
    return frame.sort_values(["fold", "Date", "HubID"]).reset_index(drop=True)


def correctable(frame: pd.DataFrame) -> np.ndarray:
    """Rows a correction may touch: open with a positive prediction."""
    return ((frame["IsOpen"] == 1) & (frame["pred"] > 0)).to_numpy()


def residual(frame: pd.DataFrame) -> pd.Series:
    """log1p(y) - log1p(pred) on observed open positive-volume rows, NaN elsewhere (closed days carry no information on the level error)."""
    ok = (frame["IsOpen"] == 1) & (frame["y"] > 0) & (frame["pred"] > 0)
    res = np.log1p(frame["y"].to_numpy(float)) - np.log1p(np.clip(frame["pred"].to_numpy(float), 0, None))
    return pd.Series(np.where(ok, res, np.nan), index=frame.index)


def apply_shift(frame: pd.DataFrame, shift: np.ndarray) -> np.ndarray:
    """New prediction vector: expm1(log1p(pred) + shift) on correctable rows, pred elsewhere."""
    pred = frame["pred"].to_numpy(float)
    mask = correctable(frame)
    out = pred.copy()
    out[mask] = np.expm1(np.log1p(pred[mask]) + np.asarray(shift, float)[mask])
    return np.clip(out, 0.0, None)


def scores(frame: pd.DataFrame, pred: np.ndarray | None = None) -> dict[str, float]:
    p = frame["pred"].to_numpy(float) if pred is None else np.asarray(pred, float)
    y = frame["y"].to_numpy(float)
    open_zero = ((frame["IsOpen"] == 1) & (frame["y"] == 0)).to_numpy()
    return {"all": rmsle(y, p), "excl_open0": rmsle(y[~open_zero], p[~open_zero])}


def hub_mean_residual(frame: pd.DataFrame, last_days: int, end: pd.Timestamp, min_days: int = 3) -> pd.Series:
    """Per HubID mean residual over the last `last_days` days ending at `end` (inclusive); NaN if fewer than `min_days` usable days."""
    start = end - pd.Timedelta(days=last_days - 1)
    sel = frame[(frame["Date"] >= start) & (frame["Date"] <= end)]
    res = residual(sel)
    grouped = res.groupby(sel["HubID"])
    mean, count = grouped.mean(), grouped.count()
    return mean.where(count >= min_days)


def previous_fold(fold: int) -> int | None:
    """Fold whose 42-day window ends at the origin of `fold` (dev folds 1-4 are consecutive; fold 5 is an earlier season)."""
    return fold - 1 if fold in (2, 3, 4) else None


def ols_through_origin(x: np.ndarray, e: np.ndarray, ridge: float = 0.0) -> np.ndarray:
    """beta minimising ||e - x beta||^2 + ridge ||beta||^2; rows with non-finite x or e are skipped; zeros when nothing is usable."""
    x = np.asarray(x, float).reshape(len(e), -1)
    keep = np.isfinite(x).all(axis=1) & np.isfinite(e)
    if not keep.any():
        return np.zeros(x.shape[1])
    xk, ek = x[keep], np.asarray(e, float)[keep]
    gram = xk.T @ xk + ridge * np.eye(x.shape[1])
    return np.linalg.solve(gram + 1e-12 * np.eye(x.shape[1]), xk.T @ ek)


def leave_fold_out(designs: dict[int, tuple[np.ndarray, np.ndarray]], target: int, ridge: float = 0.0) -> np.ndarray:
    """beta fitted on every fold in `designs` except `target` (designs[f] = (X (n,p), residual (n,)))."""
    others = [f for f in designs if f != target]
    if not others:
        raise ValueError(f"no other fold available to fit the correction for fold {target}")
    x = np.vstack([designs[f][0] for f in others])
    e = np.concatenate([designs[f][1] for f in others])
    return ols_through_origin(x, e, ridge)


def evaluate_correction(frames: dict[int, pd.DataFrame], designs: dict[int, np.ndarray], fit_folds: tuple[int, ...] = DEV_FOLDS,
                        eval_folds: tuple[int, ...] | None = None, ridge: float = 0.0) -> pd.DataFrame:
    """Per-fold baseline and corrected RMSLE (all rows / excluding open-zero rows) and beta, with beta fitted leave-fold-out.

    frames[f] is the OOF frame of fold f; designs[f] is its (n,p) design matrix (row-aligned). Folds in `eval_folds` that have no design
    are reported with NaN. The fit pool is `fit_folds` minus the evaluated fold.
    """
    eval_folds = eval_folds if eval_folds is not None else tuple(sorted(designs))
    fit = {f: (designs[f][correctable(frames[f])], residual(frames[f]).to_numpy()[correctable(frames[f])]) for f in fit_folds if f in designs}
    rows = []
    for f in eval_folds:
        base = scores(frames[f])
        if f not in designs:
            rows.append({"fold": f, "base_all": base["all"], "base_excl": base["excl_open0"], "new_all": np.nan, "new_excl": np.nan, "beta": ""})
            continue
        beta = leave_fold_out(fit, f, ridge)
        x = np.nan_to_num(np.asarray(designs[f], float).reshape(len(frames[f]), -1), nan=0.0)
        new = scores(frames[f], apply_shift(frames[f], x @ beta))
        rows.append({"fold": f, "base_all": base["all"], "base_excl": base["excl_open0"], "new_all": new["all"], "new_excl": new["excl_open0"],
                     "beta": ";".join(f"{b:.4f}" for b in beta)})
    out = pd.DataFrame(rows)
    out["gain_all"] = out["base_all"] - out["new_all"]
    out["gain_excl"] = out["base_excl"] - out["new_excl"]
    return out


def gate_summary(per_fold: pd.DataFrame, dev: tuple[int, ...] = DEV_FOLDS) -> dict[str, float | int | bool]:
    """Gate: mean gain >= GATE_GAIN on >= 2 of the dev folds (positive = lower RMSLE than cs02), also excluding open-zero rows."""
    d = per_fold[per_fold["fold"].isin(dev)].dropna(subset=["gain_all"])
    n_pass = int((d["gain_all"] >= GATE_GAIN).sum())
    n_pass_excl = int((d["gain_excl"] >= GATE_GAIN).sum())
    return {"mean_gain": float(d["gain_all"].mean()) if len(d) else float("nan"), "n_pass": n_pass, "n_pass_excl": n_pass_excl, "n_dev": len(d),
            "gate_all": n_pass >= 2 and len(d) >= 3, "gate_excl": n_pass_excl >= 2 and len(d) >= 3}
