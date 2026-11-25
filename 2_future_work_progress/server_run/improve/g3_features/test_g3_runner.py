from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import g3_runner as g3
from conftest import synthetic_raw, synthetic_target_frame
from extra_features import ALL_GROUPS, select_group_columns


def _ctx(raw: pd.DataFrame) -> dict:
    o_val = pd.Timestamp("2014-04-20")
    origins = [pd.Timestamp("2013-10-06"), pd.Timestamp("2013-11-17")]
    def stack(os):
        parts = [synthetic_target_frame(raw, o, 3).assign(_origin=o) for o in os]
        return pd.concat(parts, ignore_index=True)
    def finish(f):
        f["f1"] = np.arange(len(f), dtype=float)
        f["f1_dup"] = f["f1"].astype("float32")      # exact duplicate (different dtype)
        f["f2"] = np.random.default_rng(0).random(len(f))
        f["Conversion_28D_Mean"], f["VolumePerSession_28D_Mean"] = f["f2"] * 2, f["f2"] * 2
        f["IsOpen"], f["ActualOrderVolume"] = 1, 10.0
        f["HubFormat"] = 1 + f["HubID"] % 2
        return f
    train, hold = finish(stack(origins)), finish(stack(origins[-1:]))
    valid = finish(synthetic_target_frame(raw, o_val, 3))
    valid["ForecastOrigin"] = o_val
    return {"train": train, "hold": hold, "valid": valid, "origin": o_val, "columns": ["Horizon", "Weekday", "PromoActive", "f1", "f1_dup", "f2", "Conversion_28D_Mean", "VolumePerSession_28D_Mean"], "fold": 2}


def test_variant_specs_cover_requested_names() -> None:
    assert set(g3.VARIANT_SPECS) == {"feat_trend", "feat_weekday_promo", "feat_streaks", "feat_all", "dedupe_only", "dedupe_plus_all"}
    assert g3.VARIANT_SPECS["feat_all"].groups == ALL_GROUPS and not g3.VARIANT_SPECS["feat_all"].dedupe
    assert g3.VARIANT_SPECS["dedupe_only"].groups == () and g3.VARIANT_SPECS["dedupe_only"].dedupe
    assert g3.VARIANT_SPECS["dedupe_plus_all"].groups == ALL_GROUPS and g3.VARIANT_SPECS["dedupe_plus_all"].dedupe


def test_variant_columns_is_pure_and_ordered() -> None:
    base, extra, dups = ["a", "b", "c"], ["g3_trend_x", "g3_dow_y"], {"b": "a"}
    spec = g3.VARIANT_SPECS["dedupe_plus_all"]
    assert g3.variant_columns(base, extra, spec, dups) == ["a", "c", "g3_trend_x", "g3_dow_y"]
    assert g3.variant_columns(base, extra, g3.VARIANT_SPECS["feat_trend"], dups) == ["a", "b", "c", "g3_trend_x"]
    assert base == ["a", "b", "c"]


def test_augment_ctx_adds_columns_without_touching_input(raw) -> None:
    ctx = _ctx(raw)
    cols_before = list(ctx["train"].columns)
    out = g3.augment_ctx(ctx, raw)
    assert list(ctx["train"].columns) == cols_before
    for key in ("train", "hold", "valid"):
        assert len(out[key]) == len(ctx[key])
        assert select_group_columns(out[key].columns, ALL_GROUPS)
    assert out["columns"] == ctx["columns"]
    pd.testing.assert_series_equal(out["valid"]["f1"], ctx["valid"]["f1"])


def test_augment_uses_origin_specific_history(raw) -> None:
    out = g3.augment_ctx(_ctx(raw), raw)
    first, last = out["train"]["_origin"].min(), out["train"]["_origin"].max()
    a = out["train"].loc[out["train"]["_origin"] == first, "g3_trend_level91"]
    b = out["train"].loc[out["train"]["_origin"] == last, "g3_trend_level91"]
    assert not np.allclose(a.to_numpy(), b.to_numpy(), equal_nan=True)


def test_duplicates_for_ctx_found_in_all_frames(raw) -> None:
    dups = g3.duplicates_for_ctx(_ctx(raw))   # Weekday == Horizon is a genuine coincidence of the 3-day synthetic windows
    assert dups["f1_dup"] == "f1" and "f2" not in dups and "f1" not in dups


def test_filter_scope() -> None:
    d = {"VolumePerSession_28D_Mean": "Conversion_28D_Mean", "f1_dup": "f1"}
    assert g3.filter_scope(d, "vps") == {"VolumePerSession_28D_Mean": "Conversion_28D_Mean"}
    assert g3.filter_scope(d, "all") == d and d["f1_dup"] == "f1"
    with pytest.raises(ValueError):
        g3.filter_scope(d, "nope")


def test_load_history_merges_metadata(tmp_path: Path, raw) -> None:
    (tmp_path / "datset_raw").mkdir()
    raw[["HubID", "Weekday", "Date", "OrderVolume", "AppSessions", "IsOpen", "PromoActive"]].to_csv(tmp_path / "datset_raw/orders_train.csv", index=False)
    meta = raw.drop_duplicates("HubID")[["HubID", "HubFormat", "AssortmentTier"]]
    meta.to_csv(tmp_path / "datset_raw/hub_metadata.csv", index=False)
    hist = g3.load_history(tmp_path, None)
    assert {"HubFormat", "AssortmentTier", "PromoActive"} <= set(hist.columns) and len(hist) == len(raw)
    assert pd.api.types.is_datetime64_any_dtype(hist["Date"])


def test_load_history_falls_back_to_processed_base_table(tmp_path: Path, raw) -> None:
    (tmp_path / "data/processed").mkdir(parents=True)
    raw.to_csv(tmp_path / "data/processed/base_train.csv.gz", index=False)
    assert len(g3.load_history(tmp_path, None)) == len(raw)


def test_load_history_errors_when_nothing_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        g3.load_history(tmp_path, None)


def test_main_end_to_end_with_stubbed_fit(tmp_path: Path, monkeypatch, raw) -> None:
    seen: dict[str, list[str]] = {}

    def fake_fit(ctx, cfg, seed, n_jobs, lr, rounds):
        name = ctx["variant"]
        seen[name] = list(ctx["columns"])
        oof = ctx["valid"][["HubID", "Date", "Horizon"]].assign(IsOpen=1, ActualOrderVolume=10.0, pred=9.0, fold=ctx["fold"])
        return oof, {"rmsle_all": 0.1, "rmsle_excl_open0": 0.1, "bias_open": 0.0, "fold": ctx["fold"], "best_iteration": 5, "fit_seconds": 0.1}

    monkeypatch.setattr(g3.base, "prepare_fold", lambda root, fold, seed: {**_ctx(raw), "fold": fold})
    monkeypatch.setattr(g3.base, "fit_variant", fake_fit)
    monkeypatch.setattr(g3, "load_history", lambda root, raw_dir: raw)
    out = tmp_path / "out"
    code = g3.main([str(tmp_path), str(out), "--variants", "feat_trend,dedupe_only,dedupe_plus_all", "--folds", "2,3", "--seed", "7"])
    assert code == 0
    exp = pd.read_csv(out / "experiments.csv")
    assert list(exp["variant"]) == ["feat_trend", "dedupe_only", "dedupe_plus_all"] and (exp["seed"] == 7).all()
    assert {"mean_all", "fold2_all", "fold3_all", "config_hash"} <= set(exp.columns)
    assert json.loads(exp.loc[0, "config"])["colsample"] == 0.2
    for name in exp["variant"]:
        oof = pd.read_csv(out / f"oof_{name}_s7.csv.gz")
        assert {"fold", "HubID", "Date", "ActualOrderVolume", "pred"} <= set(oof.columns) and set(oof["fold"]) == {2, 3}
    assert "f1_dup" in seen["feat_trend"] and "f1_dup" in seen["dedupe_only"]   # default scope vps keeps non-VPS duplicates
    assert "VolumePerSession_28D_Mean" in seen["feat_trend"] and "VolumePerSession_28D_Mean" not in seen["dedupe_only"]
    assert not select_group_columns(seen["dedupe_only"], ALL_GROUPS)
    assert len(select_group_columns(seen["dedupe_plus_all"], ALL_GROUPS)) > len(select_group_columns(seen["feat_trend"], ALL_GROUPS))
    import blend
    assert not blend.blend_table({n: blend.load(out, n, 7) for n in ("feat_trend", "dedupe_only")}).empty


def test_main_dedupe_scope_all_drops_every_duplicate(tmp_path: Path, monkeypatch, raw) -> None:
    seen: list[list[str]] = []

    def fake_fit(ctx, cfg, seed, n_jobs, lr, rounds):
        seen.append(list(ctx["columns"]))
        oof = ctx["valid"][["HubID", "Date", "Horizon"]].assign(IsOpen=1, ActualOrderVolume=10.0, pred=9.0, fold=ctx["fold"])
        return oof, {"rmsle_all": 0.1, "rmsle_excl_open0": 0.1, "bias_open": 0.0, "fold": ctx["fold"], "best_iteration": 5, "fit_seconds": 0.1}

    monkeypatch.setattr(g3.base, "prepare_fold", lambda root, fold, seed: {**_ctx(raw), "fold": fold})
    monkeypatch.setattr(g3.base, "fit_variant", fake_fit)
    monkeypatch.setattr(g3, "load_history", lambda root, raw_dir: raw)
    g3.main([str(tmp_path), str(tmp_path / "o"), "--variants", "dedupe_only", "--folds", "2", "--dedupe-scope", "all"])
    assert "f1_dup" not in seen[0] and "VolumePerSession_28D_Mean" not in seen[0] and "f1" in seen[0]
    assert json.loads(pd.read_csv(tmp_path / "o/experiments.csv").loc[0, "config"])["dedupe_scope"] == "all"
    assert "f1_dup" in json.loads((tmp_path / "o/duplicates_fold2.json").read_text())


def test_main_rejects_unknown_variant(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        g3.main([str(tmp_path), str(tmp_path / "o"), "--variants", "nope"])
