"""Opt-in checks on the real raw files (skipped when orders_train.csv is not on this machine)."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import g3b_features as gf
from feature_panel import build_panel

CANDIDATES = [Path("/Users/bournesmartasfuck_kush/Desktop/Dsc/ShadowHub-repo/1_midterm/data/raw")]
RAW_DIR = next((p for p in CANDIDATES if (p / "orders_train.csv").exists()), None)
pytestmark = pytest.mark.skipif(RAW_DIR is None, reason="real raw files not available")

ORIGIN = pd.Timestamp("2014-10-02")


@pytest.fixture(scope="module")
def history() -> pd.DataFrame:
    hist = pd.read_csv(RAW_DIR / "orders_train.csv", parse_dates=["Date"])
    meta = pd.read_csv(RAW_DIR / "hub_metadata.csv")[["HubID", "HubFormat", "AssortmentTier"]]
    return hist.merge(meta, on="HubID", how="left")[["HubID", "Date", "OrderVolume", "IsOpen", "PromoActive", "HubFormat", "AssortmentTier"]]


def _target(panel, origin: pd.Timestamp) -> pd.DataFrame:
    dates = pd.date_range(origin + pd.Timedelta(days=1), periods=42)
    return pd.MultiIndex.from_product([panel.hubs, dates], names=["HubID", "Date"]).to_frame(index=False)


def test_real_data_scramble_leakage_and_runtime(history) -> None:
    panel = build_panel(history)
    frame = _target(panel, ORIGIN)
    t0 = time.perf_counter()
    base = gf.add_features(frame, panel, ORIGIN)
    assert time.perf_counter() - t0 < 60
    rng = np.random.default_rng(1)
    scr = history.copy()
    after = scr["Date"] > ORIGIN
    scr.loc[after, "OrderVolume"] = rng.permutation(scr.loc[after, "OrderVolume"].to_numpy()) * 7 + 1
    scr.loc[after, "IsOpen"] = rng.integers(0, 2, int(after.sum()))
    scr.loc[after, "PromoActive"] = rng.integers(0, 2, int(after.sum()))
    pd.testing.assert_frame_equal(base, gf.add_features(frame, scr, ORIGIN))
    pd.testing.assert_frame_equal(base, gf.add_features(frame, build_panel(scr), ORIGIN))
    assert base.filter(like="g3b_").shape[1] == 37 and base["g3b_peer_n"].eq(gf.PEER_K).all()


def test_real_data_37_origins_run_in_minutes(history) -> None:
    panel = build_panel(history)
    origins = pd.date_range(end=ORIGIN, periods=37, freq="7D")
    stack = pd.concat([_target(panel, o).assign(_origin=o) for o in origins], ignore_index=True)
    t0 = time.perf_counter()
    out = gf.add_features_multi(stack, panel, "_origin")
    assert time.perf_counter() - t0 < 180 and len(out) == len(stack)
