from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import g12f_regime as reg
import g12f_runner as rn


def _ctx(fold):
    rng = np.random.default_rng(fold)
    dates = pd.date_range("2015-04-01", periods=14)

    def frame(n_origin):
        f = pd.MultiIndex.from_product([range(1, 6), dates], names=["HubID", "Date"]).to_frame(index=False)
        f["Weekday"] = f["Date"].dt.day_of_week + 1
        f["RegionalHoliday"] = (rng.random(len(f)) < 0.05).astype(int)
        f["IsOpen"] = 1
        f["ActualOrderVolume"] = rng.integers(10, 100, len(f)).astype(float)
        f["Horizon"] = np.tile(np.arange(1, 15), 5)
        return f

    valid = frame(1)
    return {"train": frame(3), "hold": frame(1), "valid": valid, "origin": pd.Timestamp("2015-03-31"), "columns": ["Weekday"], "fold": fold}


def test_expert_configs_differ_and_champion_untouched():
    cfgs = rn.expert_configs()
    assert cfgs["ordinary"] == rn.base.VARIANTS["cs02"]
    assert cfgs["special"].fixed_rounds == reg.SPECIAL_EXPERT_ROUNDS and cfgs["special"].num_leaves < cfgs["ordinary"].num_leaves


def test_main_end_to_end_with_stubs(tmp_path: Path, monkeypatch):
    seen = []

    def fake_fit(ctx, cfg, seed, n_jobs, lr, rounds):
        v = ctx["valid"]
        seen.append((len(ctx["train"]), cfg.fixed_rounds))
        base_pred = 20.0 if cfg.fixed_rounds else 50.0
        oof = v[["HubID", "Date", "Horizon", "IsOpen", "ActualOrderVolume"]].assign(pred=base_pred, fold=ctx["fold"])
        return oof, {"rmsle_all": 0.1, "rmsle_excl_open0": 0.1, "bias_open": 0.0, "fold": ctx["fold"], "best_iteration": 3, "fit_seconds": 0.1}

    monkeypatch.setattr(rn.base, "prepare_fold", lambda root, fold, seed: _ctx(fold))
    monkeypatch.setattr(rn.base, "fit_variant", fake_fit)
    glob = pd.concat([_ctx(f)["valid"][["HubID", "Date"]].assign(fold=f, pred=30.0) for f in (3, 4)])
    gpath = tmp_path / "g.csv"
    glob.to_csv(gpath, index=False)
    out = tmp_path / "o"
    assert rn.main([str(tmp_path), str(out), "--global-oof", str(gpath), "--folds", "3,4", "--seed", "5"]) == 0
    exp = pd.read_csv(out / "experiments.csv")
    assert set(exp["variant"]) == set(reg.BLENDS) | {"g12f_expert_special", "g12f_expert_ordinary"}
    hard = pd.read_csv(out / "oof_g12f_hard_s5.csv.gz", parse_dates=["Date"])
    flag = reg.special_flag(_ctx(3)["valid"])
    got = hard[hard["fold"] == 3].sort_values(["HubID", "Date"])["pred"].to_numpy()
    np.testing.assert_allclose(got, np.where(flag, 20.0, 50.0))
    spec = pd.read_csv(out / "oof_g12f_spec_s5.csv.gz")
    np.testing.assert_allclose(spec[spec["fold"] == 3].sort_values(["HubID", "Date"])["pred"].to_numpy(), np.where(flag, 20.0, 30.0))
    assert len(seen) == 4


def test_missing_global_rows_raise(tmp_path, monkeypatch):
    monkeypatch.setattr(rn.base, "prepare_fold", lambda root, fold, seed: _ctx(fold))
    monkeypatch.setattr(rn.base, "fit_variant", lambda ctx, *a, **k: (ctx["valid"][["HubID", "Date", "Horizon", "IsOpen", "ActualOrderVolume"]].assign(pred=1.0, fold=3), {}))
    g = _ctx(3)["valid"][["HubID", "Date"]].assign(fold=3, pred=1.0).iloc[:5]
    g.to_csv(tmp_path / "g.csv", index=False)
    with pytest.raises(ValueError):
        rn.main([str(tmp_path), str(tmp_path / "o"), "--global-oof", str(tmp_path / "g.csv"), "--folds", "3"])
