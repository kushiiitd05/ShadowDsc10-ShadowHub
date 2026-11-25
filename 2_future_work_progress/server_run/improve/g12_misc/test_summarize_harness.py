from __future__ import annotations

import numpy as np
import pandas as pd

import summarize_harness as sh


def _oof(fold, noise, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2015-04-01", periods=5)
    f = pd.MultiIndex.from_product([range(1, 21), dates], names=["HubID", "Date"]).to_frame(index=False)
    f["fold"], f["IsOpen"] = fold, 1
    f["y"] = rng.integers(50, 500, len(f)).astype(float)
    f["pred"] = f["y"] * np.exp(rng.normal(0, noise, len(f)))
    return f


def test_compare_gain_sign_and_join():
    champ = {3: _oof(3, 0.2, 1)}
    better = _oof(3, 0.1, 1).rename(columns={"y": "y"})
    better["pred"] = better["y"] * np.exp(np.random.default_rng(5).normal(0, 0.05, len(better)))
    out = sh.compare(better.rename(columns={"y": "y"}), champ)
    assert out.loc[0, "gain_all"] > 0.05 and out.loc[0, "n"] == 100
    worse = champ[3].assign(pred=champ[3]["pred"] * 1.5)
    assert sh.compare(worse, champ).loc[0, "gain_all"] < 0
    assert sh.compare(better.assign(fold=9), champ).empty
