"""Item C: peer-hub residuals. Does the mean residual of a hub's peers over the last observed weeks predict the hub's own residual?

The residuals come from the cs02 predictions of the PREVIOUS fold (window ending at this fold's origin), so folds 2, 3, 4 only (see g12d_shift).
Peers: (group) the other hubs of the same AssortmentTier x HubFormat cell, or (corr) the K hubs with the highest Pearson correlation of
log1p(orders) over the 365 days up to the origin (same definition as the g3b peer features, history <= origin only).
Feature P_h = mean over peers of the peer's mean residual in the last W days. Correction = beta * P_h with beta fitted leave-fold-out.
Variants: group / corr5 / corr10 peers, W in {14, 28, 42}; plus own-residual (D) + peer jointly ('own+group', 'own+corr5').
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
for _path in (HERE.parent / "g3b_features", HERE.parent / "g3_features"):
    if str(_path) not in sys.path:
        sys.path.append(str(_path))

import offline_common as oc  # noqa: E402
from g12d_shift import own_recent_residual  # noqa: E402
from g3b_features import peer_correlation, top_peers  # noqa: E402
from feature_panel import Panel  # noqa: E402

WINDOWS = (14, 28, 42)
CORR_WINDOW = 365
CORR_MIN_OVERLAP = 120


def group_peer_mean(values: pd.Series, groups: pd.Series) -> pd.Series:
    """For each hub: mean of `values` over the OTHER hubs of its group (NaN values ignored; NaN if none). Index = HubID."""
    groups = groups.reindex(values.index)
    ok = values.notna() & groups.notna()
    total = values.where(ok).groupby(groups).transform("sum")
    count = ok.astype(float).groupby(groups).transform("sum")
    own = values.where(ok, 0.0)
    n_other = count - ok.astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (total - own) / n_other
    return out.where((n_other > 0) & groups.notna())


def corr_peers(panel: Panel, origin: pd.Timestamp, k: int) -> pd.DataFrame:
    """(H,k) peer HubIDs (NaN padded) by correlation of log1p(open-day orders) over the 365 days up to `origin`. Index = HubID."""
    cut = panel.upto(origin)
    keep = (cut.is_open == 1) & (cut.orders > 0)
    y = np.where(keep, np.log1p(np.where(keep, cut.orders, 0.0)), np.nan)[:, -CORR_WINDOW:]
    idx, _ = top_peers(peer_correlation(y, CORR_MIN_OVERLAP), k)
    ids = np.where(idx >= 0, cut.hubs[np.maximum(idx, 0)].astype(float), np.nan)
    return pd.DataFrame(ids, index=cut.hubs)


def corr_peer_mean(values: pd.Series, peers: pd.DataFrame) -> pd.Series:
    """Mean of `values` (by HubID) over each hub's correlation peers; NaN peers / values ignored."""
    lookup = values.reindex(peers.to_numpy().ravel()).to_numpy().reshape(peers.shape)
    ok = np.isfinite(lookup)
    count = ok.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(count > 0, np.where(ok, lookup, 0.0).sum(axis=1) / np.maximum(count, 1), np.nan)
    return pd.Series(mean, index=peers.index)


def peer_feature(frames: dict[int, pd.DataFrame], fold: int, window: int, kind: str, groups: pd.Series, panel: Panel) -> pd.Series | None:
    """Hub-level peer residual mean (indexed by HubID) for `fold`, or None when the fold has no previous fold."""
    own = own_recent_residual(frames, fold, window)
    if own is None:
        return None
    if kind == "group":
        return group_peer_mean(own, groups)
    if kind.startswith("corr"):
        peers = corr_peers(panel, pd.Timestamp(oc.ORIGINS[fold]), int(kind[4:]))
        return corr_peer_mean(own.reindex(peers.index), peers)
    raise ValueError(f"unknown peer kind: {kind}")


def to_rows(frame: pd.DataFrame, hub_values: pd.Series) -> np.ndarray:
    return frame["HubID"].map(hub_values).to_numpy(float)


def run(frames: dict[int, pd.DataFrame], groups: pd.Series, panel: Panel, folds: tuple[int, ...] = (1, 2, 3, 4, 5)) -> pd.DataFrame:
    parts = []
    for window in WINDOWS:
        for kind in ("group", "corr5", "corr10"):
            peer = {f: v for f in folds if (v := peer_feature(frames, f, window, kind, groups, panel)) is not None}
            designs = {f: to_rows(frames[f], peer[f]).reshape(-1, 1) for f in peer}
            parts.append(oc.evaluate_correction(frames, designs, oc.DEV_FOLDS, folds).assign(variant=kind, window=window))
            if kind in ("group", "corr5"):
                joint = {f: np.column_stack([to_rows(frames[f], own_recent_residual(frames, f, window)), to_rows(frames[f], peer[f])]) for f in peer}
                parts.append(oc.evaluate_correction(frames, joint, oc.DEV_FOLDS, folds).assign(variant=f"own+{kind}", window=window))
    return pd.concat(parts, ignore_index=True)


def diagnostics(frames: dict[int, pd.DataFrame], groups: pd.Series, panel: Panel, window: int = 28) -> pd.DataFrame:
    """Hub-level correlation between each hub feature (own / peer residual mean at the origin) and the hub's mean residual in the fold."""
    rows = []
    for f in oc.DEV_FOLDS:
        target = oc.residual(frames[f]).groupby(frames[f]["HubID"]).mean()
        feats = {"own": own_recent_residual(frames, f, window)}
        for kind in ("group", "corr5", "corr10"):
            feats[kind] = peer_feature(frames, f, window, kind, groups, panel)
        rows.append({"fold": f, **{k: float(pd.concat([v, target], axis=1).dropna().corr().iloc[0, 1]) for k, v in feats.items()}})
    return pd.DataFrame(rows)
