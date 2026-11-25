"""Opt-in checks of the SVD features on the real raw files."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

import g12a_svd as sv
from feature_panel import build_panel

ORIGIN = pd.Timestamp("2015-01-02")


def _target(panel, origin):
    dates = pd.date_range(origin + pd.Timedelta(days=1), periods=42)
    return pd.MultiIndex.from_product([panel.hubs, dates], names=["HubID", "Date"]).to_frame(index=False)


def test_real_scramble_leakage_alignment_and_runtime(real_history):
    panel = build_panel(real_history)
    first = pd.Timestamp("2013-06-30")
    ref = sv.make_reference(panel, first)
    frame = _target(panel, ORIGIN)
    t0 = time.perf_counter()
    base = sv.add_features(frame, panel, ORIGIN, reference=ref)
    assert time.perf_counter() - t0 < 60
    scr = real_history.copy()
    after = scr["Date"] > ORIGIN
    rng = np.random.default_rng(1)
    scr.loc[after, "OrderVolume"] = rng.permutation(scr.loc[after, "OrderVolume"].to_numpy()) * 7 + 1
    scr.loc[after, "IsOpen"] = rng.integers(0, 2, int(after.sum()))
    pd.testing.assert_frame_equal(base, sv.add_features(frame, build_panel(scr), ORIGIN, reference=ref))
    cols = [c for c in base.columns if c.startswith("g12a_")]
    assert len(cols) == 14 + 6 + 3 + 2 + 1
    assert base[cols].notna().mean().min() > 0.5
    # leading loadings stay comparable across origins after alignment
    other = sv.hub_features(panel.upto(pd.Timestamp("2014-10-02")), sv.RANK, ref)[0]
    mine = sv.hub_features(panel.upto(ORIGIN), sv.RANK, ref)[0]
    assert np.corrcoef(other["g12a_load1"], mine["g12a_load1"])[0, 1] > 0.8
    assert np.corrcoef(other["g12a_load2"], mine["g12a_load2"])[0, 1] > 0.5
