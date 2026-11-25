import json
import sys
import types
from argparse import Namespace

import numpy as np
import pandas as pd
import pytest

import g2_runner
from blend import blend_table, load
from test_segments import make_ctx


def test_parse_variants_rejects_unknown() -> None:
    assert g2_runner.parse_variants("cs02, split_horizon") == ["cs02", "split_horizon"]
    with pytest.raises(SystemExit):
        g2_runner.parse_variants("split_horizon,nope")


def test_g2_config_distinguishes_variants_and_keeps_model_field() -> None:
    cfgs = {n: g2_runner.g2_config(n) for n in g2_runner.G2_VARIANTS}
    assert all(c.model == "lightgbm" and c.colsample == 0.2 and c.subsample_freq == 1 for c in cfgs.values())
    hashes = {json.dumps(c.__dict__, sort_keys=True) for c in cfgs.values()}
    assert len(hashes) == 3
    assert cfgs["long_window_only"].segments == "1-21|22-42r"


def test_main_writes_experiments_and_oof_in_exp_runner_schema(tmp_path, monkeypatch) -> None:
    pkg = types.ModuleType("src")
    mod = types.ModuleType("src.model_pipeline")
    mod.CATEGORICAL_COLUMNS = {"Weekday"}
    monkeypatch.setitem(sys.modules, "src", pkg)
    monkeypatch.setitem(sys.modules, "src.model_pipeline", mod)
    ctx = make_ctx()
    monkeypatch.setattr(g2_runner, "prepare_fold", lambda root, fold, seed: {**ctx, "fold": fold})
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["g2_runner", str(tmp_path), str(out), "--variants", "cs02,split_horizon",
                                      "--folds", "2,3", "--seed", "5", "--n-jobs", "1", "--lr", "0.1", "--rounds", "30"])
    assert g2_runner.main() == 0
    exp = pd.read_csv(out / "experiments.csv")
    assert list(exp["variant"]) == ["cs02", "split_horizon"]
    for col in ("mean_all", "mean_excl_open0", "mean_bias_open", "fit_seconds", "fold2_all", "fold3_all", "fold2_best_iter", "config_hash"):
        assert col in exp.columns
    a, b = load(out, "cs02", 5), load(out, "split_horizon", 5)
    assert set(a["fold"]) == {2, 3}
    table = blend_table({"cs02": a, "split_horizon": b})
    assert {"cs02", "split_horizon", "blend"} <= set(table.columns)
    raw = pd.read_csv(out / "oof_cs02_s5.csv.gz")
    assert list(raw.columns) == ["HubID", "Date", "Horizon", "IsOpen", "ActualOrderVolume", "pred", "fold"]
