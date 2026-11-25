from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import g3b_features as gf
import g3b_runner as g3b
from extra_features import ALL_GROUPS as G3_GROUPS, select_group_columns as g3_select
from synth_g3b import synthetic_raw, synthetic_target_frame

NEW_VARIANTS = {"g3b_gap", "g3b_yearago", "g3b_peer", "g3b_all", "g3b_all_plus_g3"}


def _ctx(raw: pd.DataFrame) -> dict:
    o_val = pd.Timestamp("2014-10-02")
    origins = [pd.Timestamp("2014-06-08"), pd.Timestamp("2014-08-17")]

    def stack(os):
        return pd.concat([synthetic_target_frame(raw, o, 3).assign(_origin=o) for o in os], ignore_index=True)

    def finish(f):
        f["f1"] = np.arange(len(f), dtype=float)
        f["IsOpen"], f["ActualOrderVolume"] = 1, 10.0
        return f

    train, hold = finish(stack(origins)), finish(stack(origins[-1:]))
    valid = finish(synthetic_target_frame(raw, o_val, 3))
    valid["ForecastOrigin"] = o_val
    return {"train": train, "hold": hold, "valid": valid, "origin": o_val, "columns": ["Horizon", "Weekday", "PromoActive", "f1"], "fold": 2}


def _fake_fit(seen: dict):
    def fit(ctx, cfg, seed, n_jobs, lr, rounds):
        seen[ctx["variant"]] = list(ctx["columns"])
        oof = ctx["valid"][["HubID", "Date", "Horizon"]].assign(IsOpen=1, ActualOrderVolume=10.0, pred=9.0, fold=ctx["fold"])
        return oof, {"rmsle_all": 0.1, "rmsle_excl_open0": 0.1, "bias_open": 0.0, "fold": ctx["fold"], "best_iteration": 5, "fit_seconds": 0.1}
    return fit


def test_variant_specs() -> None:
    assert set(g3b.VARIANT_SPECS) == NEW_VARIANTS
    S = g3b.VARIANT_SPECS
    assert S["g3b_gap"].groups == ("gap_aware",) and S["g3b_yearago"].groups == ("year_ago",) and S["g3b_peer"].groups == ("peer_corr",)
    assert S["g3b_all"].groups == gf.ALL_GROUPS and not S["g3b_all"].with_g3
    assert S["g3b_all_plus_g3"].groups == gf.ALL_GROUPS and S["g3b_all_plus_g3"].with_g3


def test_variant_columns_is_pure_and_ordered() -> None:
    base, extra = ["a", "b"], ["g3_trend_x", "g3b_gap_mean28", "g3b_ya_1y_uplift", "g3_dow_y", "g3b_peer_n"]
    assert g3b.variant_columns(base, extra, g3b.VARIANT_SPECS["g3b_gap"]) == ["a", "b", "g3b_gap_mean28"]
    assert g3b.variant_columns(base, extra, g3b.VARIANT_SPECS["g3b_all"]) == ["a", "b", "g3b_gap_mean28", "g3b_ya_1y_uplift", "g3b_peer_n"]
    assert g3b.variant_columns(base, extra, g3b.VARIANT_SPECS["g3b_all_plus_g3"]) == [
        "a", "b", "g3b_gap_mean28", "g3b_ya_1y_uplift", "g3b_peer_n", "g3_trend_x", "g3_dow_y"]
    assert base == ["a", "b"]


def test_augment_ctx_adds_only_requested_groups_without_touching_input(raw) -> None:
    ctx = _ctx(raw)
    cols_before = list(ctx["train"].columns)
    out = g3b.augment_ctx(ctx, raw, groups=("gap_aware",), with_g3=False)
    assert list(ctx["train"].columns) == cols_before and out["columns"] == ctx["columns"]
    for key in ("train", "hold", "valid"):
        assert len(out[key]) == len(ctx[key]) and (out[key].index == ctx[key].index).all()
        assert gf.select_group_columns(out[key].columns, ("gap_aware",))
        assert not gf.select_group_columns(out[key].columns, ("year_ago", "peer_corr"))
        assert not g3_select(out[key].columns, G3_GROUPS)
    pd.testing.assert_series_equal(out["valid"]["f1"], ctx["valid"]["f1"])


def test_augment_ctx_with_g3_adds_both_families(raw) -> None:
    out = g3b.augment_ctx(_ctx(raw), raw, groups=gf.ALL_GROUPS, with_g3=True)
    for key in ("train", "hold", "valid"):
        assert g3_select(out[key].columns, G3_GROUPS) and gf.select_group_columns(out[key].columns, gf.ALL_GROUPS)


def test_augment_uses_origin_specific_history(raw) -> None:
    out = g3b.augment_ctx(_ctx(raw), raw, groups=("gap_aware",), with_g3=False)
    first, last = out["train"]["_origin"].min(), out["train"]["_origin"].max()
    a = out["train"].loc[out["train"]["_origin"] == first, "g3b_gap_mean28"]
    b = out["train"].loc[out["train"]["_origin"] == last, "g3b_gap_mean28"]
    assert not np.allclose(a.to_numpy(), b.to_numpy(), equal_nan=True)


def test_augment_valid_features_ignore_data_after_origin(raw) -> None:
    ctx = _ctx(raw)
    scrambled = raw.copy()
    after = scrambled["Date"] > ctx["origin"]
    scrambled.loc[after, "OrderVolume"] = 12345.0
    a = g3b.augment_ctx(ctx, raw, groups=gf.ALL_GROUPS, with_g3=False)
    b = g3b.augment_ctx(ctx, scrambled, groups=gf.ALL_GROUPS, with_g3=False)
    for key in ("train", "hold", "valid"):
        cols = gf.select_group_columns(a[key].columns, gf.ALL_GROUPS)
        pd.testing.assert_frame_equal(a[key][cols], b[key][cols])


def test_main_end_to_end_with_stubbed_fit(tmp_path: Path, monkeypatch, raw) -> None:
    seen: dict[str, list[str]] = {}
    monkeypatch.setattr(g3b.base, "prepare_fold", lambda root, fold, seed: {**_ctx(raw), "fold": fold})
    monkeypatch.setattr(g3b.base, "fit_variant", _fake_fit(seen))
    monkeypatch.setattr(g3b, "load_history", lambda root, raw_dir: raw)
    out = tmp_path / "out"
    names = ",".join(sorted(NEW_VARIANTS))
    assert g3b.main([str(tmp_path), str(out), "--variants", names, "--folds", "2,3", "--seed", "7", "--n-jobs", "2"]) == 0
    exp = pd.read_csv(out / "experiments.csv")
    assert set(exp["variant"]) == NEW_VARIANTS and (exp["seed"] == 7).all()
    assert {"mean_all", "fold2_all", "fold3_all", "config_hash", "variant", "date", "lr", "rounds", "n_jobs"} <= set(exp.columns)
    cfg = json.loads(exp.loc[0, "config"])
    assert cfg["colsample"] == 0.2 and cfg["subsample_freq"] == 1 and cfg["recipe"] == "cs02"
    for name in exp["variant"]:
        oof = pd.read_csv(out / f"oof_{name}_s7.csv.gz")
        assert {"fold", "HubID", "Date", "ActualOrderVolume", "pred"} <= set(oof.columns) and set(oof["fold"]) == {2, 3}
    base_cols = ["Horizon", "Weekday", "PromoActive", "f1"]
    assert all(seen[n][:4] == base_cols for n in NEW_VARIANTS)
    assert set(seen["g3b_gap"][4:]) == set(c for c in seen["g3b_all"] if c.startswith("g3b_gap_"))
    assert not any(c.startswith("g3_") for c in seen["g3b_all"])
    assert any(c.startswith("g3_") for c in seen["g3b_all_plus_g3"])
    assert set(seen["g3b_all"][4:]) <= set(seen["g3b_all_plus_g3"])
    assert len(seen["g3b_all"]) > len(seen["g3b_gap"]) > 4 and len(seen["g3b_yearago"]) > 4 and len(seen["g3b_peer"]) > 4
    import blend
    assert not blend.blend_table({n: blend.load(out, n, 7) for n in ("g3b_gap", "g3b_all")}).empty


def test_main_computes_only_needed_feature_families(tmp_path: Path, monkeypatch, raw) -> None:
    calls: list[tuple] = []
    real = g3b.augment_ctx

    def spy(ctx, history, groups, with_g3):
        calls.append((tuple(groups), with_g3))
        return real(ctx, history, groups, with_g3)

    monkeypatch.setattr(g3b, "augment_ctx", spy)
    monkeypatch.setattr(g3b.base, "prepare_fold", lambda root, fold, seed: {**_ctx(raw), "fold": fold})
    monkeypatch.setattr(g3b.base, "fit_variant", _fake_fit({}))
    monkeypatch.setattr(g3b, "load_history", lambda root, raw_dir: raw)
    g3b.main([str(tmp_path), str(tmp_path / "o"), "--variants", "g3b_gap,g3b_peer", "--folds", "2"])
    assert calls == [(("gap_aware", "peer_corr"), False)]


def test_cli_accepts_same_flags_as_g3_runner() -> None:
    args = g3b.parse_args(["root", "out", "--variants", "g3b_gap", "--folds", "2,3,4", "--seed", "1", "--n-jobs", "4", "--lr", "0.03",
                           "--rounds", "10", "--dedupe-scope", "all", "--raw-dir", "/x"])
    assert args.folds == "2,3,4" and args.dedupe_scope == "all" and args.raw_dir == Path("/x")


def test_main_rejects_unknown_variant(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        g3b.main([str(tmp_path), str(tmp_path / "o"), "--variants", "feat_all"])


def test_with_target_date_adds_origin_plus_horizon_without_mutating() -> None:
    import pandas as pd
    from g3b_runner import with_target_date

    frame = pd.DataFrame({"_origin": pd.to_datetime(["2015-01-02", "2015-01-16"]), "Horizon": [1, 42]})
    out = with_target_date(frame)
    assert out["Date"].tolist() == [pd.Timestamp("2015-01-03"), pd.Timestamp("2015-02-27")]
    assert "Date" not in frame.columns
    assert with_target_date(out) is out
