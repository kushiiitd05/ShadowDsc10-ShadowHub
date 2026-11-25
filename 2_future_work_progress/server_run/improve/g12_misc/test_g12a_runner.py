from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import g12a_runner as rn
import g12a_svd as sv
from synth_g3b import synthetic_target_frame


def _ctx(raw):
    o_val = pd.Timestamp("2014-10-02")
    origins = [pd.Timestamp("2014-06-08"), pd.Timestamp("2014-08-17")]

    def stack(os):
        return pd.concat([synthetic_target_frame(raw, o, 3).assign(_origin=o) for o in os], ignore_index=True)

    def finish(f):
        f["f1"] = np.arange(len(f), dtype=float)
        f["IsOpen"], f["ActualOrderVolume"] = 1, 10.0
        return f

    valid = finish(synthetic_target_frame(raw, o_val, 3))
    valid["ForecastOrigin"] = o_val
    return {"train": finish(stack(origins)), "hold": finish(stack(origins[-1:])), "valid": valid, "origin": o_val,
            "columns": ["Horizon", "Weekday", "PromoActive", "f1"], "fold": 3}


def _fake_fit(seen):
    def fit(ctx, cfg, seed, n_jobs, lr, rounds):
        seen[ctx["variant"]] = list(ctx["columns"])
        oof = ctx["valid"][["HubID", "Date", "Horizon"]].assign(IsOpen=1, ActualOrderVolume=10.0, pred=9.0, fold=ctx["fold"])
        return oof, {"rmsle_all": 0.1, "rmsle_excl_open0": 0.1, "bias_open": 0.0, "fold": ctx["fold"], "best_iteration": 5, "fit_seconds": 0.1}
    return fit


def test_augment_adds_columns_without_mutation(raw):
    ctx = _ctx(raw)
    before = list(ctx["train"].columns)
    out = rn.augment_ctx(ctx, raw, 5)
    assert list(ctx["train"].columns) == before
    for key in ("train", "hold", "valid"):
        assert len(out[key]) == len(ctx[key]) and any(c.startswith("g12a_") for c in out[key].columns)
    assert out["columns"] == ctx["columns"]


def test_augment_ignores_future_data(raw):
    ctx = _ctx(raw)
    scrambled = raw.copy()
    scrambled.loc[scrambled["Date"] > ctx["origin"], "OrderVolume"] = 5e4
    a, b = rn.augment_ctx(ctx, raw, 5), rn.augment_ctx(ctx, scrambled, 5)
    for key in ("train", "hold", "valid"):
        cols = [c for c in a[key].columns if c.startswith("g12a_")]
        pd.testing.assert_frame_equal(a[key][cols], b[key][cols])


def test_main_end_to_end(tmp_path: Path, monkeypatch, raw):
    seen: dict = {}
    monkeypatch.setattr(rn.base, "prepare_fold", lambda root, fold, seed: {**_ctx(raw), "fold": fold})
    monkeypatch.setattr(rn.base, "fit_variant", _fake_fit(seen))
    monkeypatch.setattr(rn, "load_history", lambda root, raw_dir: raw)
    out = tmp_path / "o"
    assert rn.main([str(tmp_path), str(out), "--variants", "g12a_svd12,g12a_svd15", "--folds", "3,4", "--seed", "7"]) == 0
    exp = pd.read_csv(out / "experiments.csv")
    assert set(exp["variant"]) == {"g12a_svd12", "g12a_svd15"}
    assert json.loads(exp.loc[0, "config"])["recipe"] == "cs02"
    oof = pd.read_csv(out / "oof_g12a_svd12_s7.csv.gz")
    assert {"fold", "HubID", "Date", "ActualOrderVolume", "pred"} <= set(oof.columns) and set(oof["fold"]) == {3, 4}
    assert sum(c.startswith("g12a_load") for c in seen["g12a_svd12"]) == 12 and sum(c.startswith("g12a_load") for c in seen["g12a_svd15"]) == 15
    assert seen["g12a_svd12"][:4] == ["Horizon", "Weekday", "PromoActive", "f1"]


def test_unknown_variant_rejected(tmp_path):
    with pytest.raises(SystemExit):
        rn.main([str(tmp_path), str(tmp_path / "o"), "--variants", "nope"])
