from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import g3b_features as g3b_feat
import g3c_features as gf
import g3c_runner as g3c
from synth_g3c import synthetic_raw, synthetic_target_frame

NEW_VARIANTS = {"g3c_holiday", "g3c_calendar", "g3c_sunday", "g3c_promo", "g3c_all", "g3c_all_plus_g3b"}


def _ctx(raw: pd.DataFrame, o_val: str = "2014-06-20", fold: int = 2) -> dict:
    o_val = pd.Timestamp(o_val)
    origins = [o_val - pd.Timedelta(days=130), o_val - pd.Timedelta(days=60)]

    def stack(os):
        return pd.concat([synthetic_target_frame(raw, o, 3).assign(_origin=o) for o in os], ignore_index=True).drop(columns=["Date", "Weekday"])

    def finish(f):
        f["f1"] = np.arange(len(f), dtype=float)
        f["IsOpen"], f["ActualOrderVolume"] = 1, 10.0
        return f

    train, hold = finish(stack(origins)), finish(stack(origins[-1:]))
    valid = finish(synthetic_target_frame(raw, o_val, 3))
    valid["ForecastOrigin"] = o_val
    return {"train": train, "hold": hold, "valid": valid, "origin": o_val, "columns": ["Horizon", "f1"], "fold": fold}


def _fake_fit(seen: dict):
    def fit(ctx, cfg, seed, n_jobs, lr, rounds):
        seen[ctx["variant"]] = list(ctx["columns"])
        oof = ctx["valid"][["HubID", "Date", "Horizon"]].assign(IsOpen=1, ActualOrderVolume=10.0, pred=9.0, fold=ctx["fold"])
        return oof, {"rmsle_all": 0.1, "rmsle_excl_open0": 0.1, "bias_open": 0.0, "fold": ctx["fold"], "best_iteration": 5, "fit_seconds": 0.1}
    return fit


def test_variant_specs() -> None:
    assert set(g3c.VARIANT_SPECS) == NEW_VARIANTS
    S = g3c.VARIANT_SPECS
    assert [S[n].groups for n in ("g3c_holiday", "g3c_calendar", "g3c_sunday", "g3c_promo")] == [("holiday",), ("calendar",), ("sunday",), ("promo",)]
    assert S["g3c_all"].groups == gf.ALL_GROUPS and not S["g3c_all"].with_g3b
    assert S["g3c_all_plus_g3b"].groups == gf.ALL_GROUPS and S["g3c_all_plus_g3b"].with_g3b


def test_variant_columns_is_pure_and_ordered() -> None:
    base, extra = ["a", "b"], ["g3b_gap_mean28", "g3c_hh_any_n", "g3c_cal_hol_week", "g3b_peer_n", "g3c_sun_n_open"]
    assert g3c.variant_columns(base, extra, g3c.VARIANT_SPECS["g3c_holiday"]) == ["a", "b", "g3c_hh_any_n"]
    assert g3c.variant_columns(base, extra, g3c.VARIANT_SPECS["g3c_all"]) == ["a", "b", "g3c_hh_any_n", "g3c_cal_hol_week", "g3c_sun_n_open"]
    assert g3c.variant_columns(base, extra, g3c.VARIANT_SPECS["g3c_all_plus_g3b"]) == [
        "a", "b", "g3c_hh_any_n", "g3c_cal_hol_week", "g3c_sun_n_open", "g3b_gap_mean28", "g3b_peer_n"]
    assert base == ["a", "b"]


def test_augment_ctx_adds_only_requested_groups_without_touching_input(raw) -> None:
    ctx = _ctx(raw)
    cols_before = list(ctx["train"].columns)
    out = g3c.augment_ctx(ctx, raw, groups=("sunday",), with_g3b=False)
    assert list(ctx["train"].columns) == cols_before and out["columns"] == ctx["columns"]
    for key in ("train", "hold", "valid"):
        assert len(out[key]) == len(ctx[key]) and (out[key].index == ctx[key].index).all()
        assert gf.select_group_columns(out[key].columns, ("sunday",))
        assert not gf.select_group_columns(out[key].columns, ("holiday", "calendar", "promo"))
        assert not g3b_feat.select_group_columns(out[key].columns, g3b_feat.ALL_GROUPS)
    pd.testing.assert_series_equal(out["valid"]["f1"], ctx["valid"]["f1"])


def test_augment_ctx_with_g3b_adds_both_families(raw) -> None:
    out = g3c.augment_ctx(_ctx(raw), raw, groups=gf.ALL_GROUPS, with_g3b=True)
    for key in ("train", "hold", "valid"):
        assert gf.select_group_columns(out[key].columns, gf.ALL_GROUPS) and g3b_feat.select_group_columns(out[key].columns, g3b_feat.ALL_GROUPS)


def test_g3b_columns_from_the_calendar_panel_equal_g3b_on_its_own_panel(raw) -> None:
    ctx = _ctx(raw)
    mine = g3c.augment_ctx(ctx, raw, groups=(), with_g3b=True)
    from g3b_runner import augment_ctx as g3b_augment
    ref = g3b_augment(ctx, raw, g3b_feat.ALL_GROUPS)
    for key in ("train", "hold", "valid"):
        pd.testing.assert_frame_equal(mine[key], ref[key])


def test_augment_uses_origin_specific_history_and_ignores_later_labels(raw) -> None:
    ctx = _ctx(raw)
    out = g3c.augment_ctx(ctx, raw, groups=("promo",))
    first, last = out["train"]["_origin"].min(), out["train"]["_origin"].max()
    a = out["train"].loc[out["train"]["_origin"] == first, "g3c_promo_n365"]
    b = out["train"].loc[out["train"]["_origin"] == last, "g3c_promo_n365"]
    assert a.mean() != b.mean()
    scrambled = raw.assign(OrderVolume=np.where(raw["Date"] > ctx["origin"], 12345.0, raw["OrderVolume"]))
    other = g3c.augment_ctx(ctx, scrambled, groups=gf.ALL_GROUPS)
    full = g3c.augment_ctx(ctx, raw, groups=gf.ALL_GROUPS)
    for key in ("train", "hold", "valid"):
        cols = gf.select_group_columns(full[key].columns, gf.ALL_GROUPS)
        pd.testing.assert_frame_equal(full[key][cols], other[key][cols])


def test_main_end_to_end_with_stubbed_fit_including_fold_5(tmp_path: Path, monkeypatch, raw) -> None:
    seen: dict[str, list[str]] = {}
    seen_folds: list[int] = []

    def prepare(root, fold, seed):
        seen_folds.append(fold)
        if fold == 5:                                                       # summer-2014 diagnostic: origin 2014-06-19, fewer valid rows
            ctx = _ctx(raw, "2014-06-19", 5)
            return {**ctx, "valid": ctx["valid"].iloc[:-7].reset_index(drop=True)}
        return _ctx(raw, "2014-10-02", fold)

    monkeypatch.setattr(g3c.base, "prepare_fold", prepare)
    monkeypatch.setattr(g3c.base, "fit_variant", _fake_fit(seen))
    monkeypatch.setattr(g3c, "load_history", lambda root, raw_dir: raw)
    out = tmp_path / "out"
    names = ",".join(sorted(NEW_VARIANTS))
    assert g3c.main([str(tmp_path), str(out), "--variants", names, "--folds", "2,5", "--seed", "7", "--n-jobs", "2"]) == 0
    assert seen_folds == [2, 5]
    exp = pd.read_csv(out / "experiments.csv")
    assert set(exp["variant"]) == NEW_VARIANTS and (exp["seed"] == 7).all()
    assert {"mean_all", "fold2_all", "fold5_all", "config_hash", "variant", "date", "lr", "rounds", "n_jobs"} <= set(exp.columns)
    cfg = json.loads(exp.loc[0, "config"])
    assert cfg["colsample"] == 0.2 and cfg["subsample_freq"] == 1 and cfg["recipe"] == "cs02"
    for name in exp["variant"]:
        oof = pd.read_csv(out / f"oof_{name}_s7.csv.gz")
        assert {"fold", "HubID", "Date", "ActualOrderVolume", "pred"} <= set(oof.columns) and set(oof["fold"]) == {2, 5}
    assert all(seen[n][:2] == ["Horizon", "f1"] for n in NEW_VARIANTS)
    assert set(seen["g3c_holiday"][2:]) == {c for c in seen["g3c_all"] if c.startswith("g3c_hh_")}
    assert not any(c.startswith("g3b_") for c in seen["g3c_all"])
    assert any(c.startswith("g3b_") for c in seen["g3c_all_plus_g3b"]) and set(seen["g3c_all"][2:]) <= set(seen["g3c_all_plus_g3b"])
    assert len(seen["g3c_all"]) == 2 + 36 and len(seen["g3c_all_plus_g3b"]) > len(seen["g3c_all"])
    import blend
    assert not blend.blend_table({n: blend.load(out, n, 7) for n in ("g3c_holiday", "g3c_all")}).empty


def test_main_computes_only_needed_feature_families(tmp_path: Path, monkeypatch, raw) -> None:
    calls: list[tuple] = []
    real = g3c.augment_ctx

    def spy(ctx, history, groups, with_g3b):
        calls.append((tuple(groups), with_g3b))
        return real(ctx, history, groups, with_g3b)

    monkeypatch.setattr(g3c, "augment_ctx", spy)
    monkeypatch.setattr(g3c.base, "prepare_fold", lambda root, fold, seed: _ctx(raw, fold=fold))
    monkeypatch.setattr(g3c.base, "fit_variant", _fake_fit({}))
    monkeypatch.setattr(g3c, "load_history", lambda root, raw_dir: raw)
    g3c.main([str(tmp_path), str(tmp_path / "o"), "--variants", "g3c_holiday,g3c_promo", "--folds", "2"])
    assert calls == [(("holiday", "promo"), False)]


def test_cli_accepts_same_flags_as_g3b_runner() -> None:
    args = g3c.parse_args(["root", "out", "--variants", "g3c_all", "--folds", "2,3,4,5", "--seed", "1", "--n-jobs", "4", "--lr", "0.03",
                           "--rounds", "10", "--dedupe-scope", "all", "--raw-dir", "/x"])
    assert args.folds == "2,3,4,5" and args.dedupe_scope == "all" and args.raw_dir == Path("/x")


def test_main_rejects_unknown_variant(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        g3c.main([str(tmp_path), str(tmp_path / "o"), "--variants", "g3b_all"])


def test_load_history_appends_test_schedule_and_metadata(tmp_path: Path, raw) -> None:
    folder = tmp_path / "raw"
    folder.mkdir()
    train = raw[raw["Date"] <= "2014-12-31"]
    train.assign(AppSessions=1)[["HubID", "Weekday", "Date", "OrderVolume", "AppSessions", "IsOpen", "PromoActive", "RegionalHoliday",
                                 "SchoolClosureFlag"]].to_csv(folder / "orders_train.csv", index=False)
    future = raw[raw["Date"].between("2015-01-01", "2015-02-11")].drop(columns=["OrderVolume", "HubFormat", "AssortmentTier"])
    future.assign(Id=1)[["Id", "HubID", "Weekday", "Date", "IsOpen", "PromoActive", "RegionalHoliday", "SchoolClosureFlag"]].to_csv(
        folder / "orders_test.csv", index=False)
    raw.drop_duplicates("HubID")[["HubID", "HubFormat", "AssortmentTier"]].to_csv(folder / "hub_metadata.csv", index=False)
    hist = g3c.load_history(tmp_path, folder)
    assert hist["Date"].max() == pd.Timestamp("2015-02-11") and hist["OrderVolume"].isna().sum() == len(future)
    assert {"HubFormat", "AssortmentTier", "RegionalHoliday"} <= set(hist.columns) and "AppSessions" not in hist.columns
    cal = gf.build_calendar(hist)
    assert np.isnan(cal.panel.orders[:, cal.panel.dates > "2014-12-31"]).all()
    with pytest.raises(FileNotFoundError):
        g3c.load_history(tmp_path / "nowhere", None)


def test_with_target_date_is_reused_from_g3b_runner() -> None:
    frame = pd.DataFrame({"_origin": pd.to_datetime(["2015-01-02"]), "Horizon": [42]})
    assert g3c.with_target_date(frame)["Date"].tolist() == [pd.Timestamp("2015-02-13")] and "Date" not in frame.columns
