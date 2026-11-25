import sys
import types

import numpy as np
import pandas as pd
import pytest

from segments import (G2_VARIANTS, Segment, SplitSpec, fit_segmented, merge_oof, segment_label, subset_context,
                      validate_segments)

SHORT_COL, LONG_COL = "Orders_7D_MeanOpen", "Orders_90D_MeanOpen"


def make_frame(origin: str, n_hubs: int = 12, seed: int = 0, with_origin_col: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    o = pd.Timestamp(origin)
    hubs, hz = np.meshgrid(np.arange(1, n_hubs + 1), np.arange(1, 43), indexing="ij")
    hubs, hz = hubs.ravel(), hz.ravel()
    level = 50.0 + 10.0 * hubs
    date = o + pd.to_timedelta(hz, unit="D")
    frame = pd.DataFrame({
        "HubID": hubs, "Horizon": hz, "Date": date, "Weekday": date.day_of_week.to_numpy(),
        "IsOpen": 1, SHORT_COL: level * rng.uniform(0.9, 1.1, len(hubs)), LONG_COL: level, "_level": level,
        "ActualOrderVolume": np.round(level * (1 + 0.1 * (date.day_of_week.to_numpy() == 0)) * rng.uniform(0.95, 1.05, len(hubs))),
        "HubID_TE": np.log1p(level).astype("float32"),
    })
    if with_origin_col:
        frame["_origin"] = o
    return frame


def make_ctx() -> dict:
    train = pd.concat([make_frame(o, seed=i) for i, o in enumerate(["2014-01-01", "2014-02-12", "2014-03-26"])], ignore_index=True)
    hold = make_frame("2014-05-07", seed=9)
    valid = make_frame("2014-06-18", seed=10, with_origin_col=False)
    valid.loc[valid.index[:5], "IsOpen"] = 0
    valid.loc[valid.index[:5], "ActualOrderVolume"] = 0.0
    return {"train": train, "hold": hold, "valid": valid, "origin": pd.Timestamp("2014-06-18"), "fold": 2,
            "columns": ["Horizon", "Weekday", SHORT_COL, LONG_COL]}


def test_validate_segments() -> None:
    validate_segments((Segment(1, 21), Segment(22, 42)))
    for bad in [(), (Segment(1, 20), Segment(22, 42)), (Segment(1, 21), Segment(21, 42)), (Segment(2, 42),), (Segment(1, 40),)]:
        with pytest.raises(ValueError):
            validate_segments(bad)


def test_registered_variants() -> None:
    assert set(G2_VARIANTS) == {"cs02", "split_horizon", "long_window_only"}
    assert G2_VARIANTS["long_window_only"].segments == (Segment(1, 21), Segment(22, 42, True))
    assert all(spec.base == "cs02" for spec in G2_VARIANTS.values())
    assert segment_label(G2_VARIANTS["long_window_only"].segments) == "1-21|22-42r"


def test_subset_context_filters_rows_and_columns_without_mutation() -> None:
    ctx = make_ctx()
    before = {k: ctx[k].copy() for k in ("train", "hold", "valid")}
    sub = subset_context(ctx, Segment(22, 42), ["Horizon", LONG_COL])
    for key in ("train", "hold", "valid"):
        pd.testing.assert_frame_equal(ctx[key], before[key])
        assert sub[key]["Horizon"].between(22, 42).all() and len(sub[key]) > 0
    assert sub["columns"] == ["Horizon", LONG_COL]
    assert SHORT_COL not in sub["train"].columns and SHORT_COL not in sub["valid"].columns
    assert {"_origin", "_level", "ActualOrderVolume", "IsOpen", "HubID_TE"} <= set(sub["train"].columns)
    assert "_origin" not in sub["valid"].columns  # valid never carried it
    assert sub["origin"] == ctx["origin"] and sub["fold"] == 2
    assert sub["valid"].index.isin(ctx["valid"].index).all()  # original row labels kept for stitching


def test_subset_context_empty_segment_raises() -> None:
    ctx = make_ctx()
    ctx["hold"] = ctx["hold"][ctx["hold"]["Horizon"] <= 10]
    with pytest.raises(ValueError, match="hold"):
        subset_context(ctx, Segment(22, 42), ctx["columns"])


def fake_fit(calls: list):
    def fit(ctx, cfg, seed, n_jobs, lr, rounds):
        calls.append(ctx)
        valid = ctx["valid"]
        oof = valid[["HubID", "Date", "Horizon", "IsOpen", "ActualOrderVolume"]].copy()
        oof["pred"] = valid["ActualOrderVolume"].to_numpy(float) * 0 + 100.0 + ctx["columns"].count(SHORT_COL)
        oof["fold"] = ctx["fold"]
        return oof, {"fold": ctx["fold"], "best_iteration": 10 * len(calls), "fit_seconds": 1.5, "train_rows": len(ctx["train"]), "offset": 0.0}
    return fit


def test_merge_oof_restores_original_order_and_checks_coverage() -> None:
    ctx = make_ctx()
    valid = ctx["valid"]
    a = valid[valid["Horizon"] > 21].assign(pred=2.0)
    b = valid[valid["Horizon"] <= 21].assign(pred=1.0)
    merged = merge_oof([a, b], valid.index)
    assert merged.index.equals(valid.index) and set(merged["pred"]) == {1.0, 2.0}
    with pytest.raises(ValueError, match="cover"):
        merge_oof([a], valid.index)
    with pytest.raises(ValueError, match="cover"):
        merge_oof([a, b, b], valid.index)


def test_fit_segmented_split_calls_each_segment_once_with_right_columns() -> None:
    ctx = make_ctx()
    calls: list = []
    oof, scores = fit_segmented(ctx, G2_VARIANTS["long_window_only"], G2_VARIANTS_CFG(), 1, 2, 0.05, 10, fit=fake_fit(calls), log=lambda *_: None)
    assert [sorted(c["valid"]["Horizon"].unique())[0] for c in calls] == [1, 22]
    assert SHORT_COL in calls[0]["columns"] and SHORT_COL not in calls[1]["columns"] and LONG_COL in calls[1]["columns"]
    assert len(oof) == len(ctx["valid"]) and oof.index.equals(ctx["valid"].index)
    assert list(oof.columns) == ["HubID", "Date", "Horizon", "IsOpen", "ActualOrderVolume", "pred", "fold"]
    assert scores["fit_seconds"] == 3.0 and scores["train_rows"] == sum(len(c["train"]) for c in calls)
    assert scores["fold"] == 2 and scores["best_iteration"] == 15 and "rmsle_all" in scores
    expected = np.where(ctx["valid"]["Horizon"] <= 21, 101.0, 100.0)
    assert np.allclose(oof["pred"], expected)


def test_fit_segmented_feeds_each_model_only_its_horizon_range_and_chronological_hold() -> None:
    ctx = make_ctx()
    seen: list[tuple[int, int]] = []

    def spy(sub, cfg, seed, n_jobs, lr, rounds):
        lo, hi = int(sub["valid"]["Horizon"].min()), int(sub["valid"]["Horizon"].max())
        for key in ("train", "hold", "valid"):
            assert sub[key]["Horizon"].between(lo, hi).all()
        assert sub["train"]["_origin"].max() < sub["hold"]["_origin"].min()  # holdout is later than training origins
        assert sub["hold"]["_origin"].max() < sub["valid"]["Date"].min()     # and strictly before the validation window
        seen.append((lo, hi))
        oof = sub["valid"][["HubID", "Date", "Horizon", "IsOpen", "ActualOrderVolume"]].copy()
        oof["pred"] = 1.0
        oof["fold"] = sub["fold"]
        return oof, {"fold": 2, "best_iteration": 1, "fit_seconds": 0.0, "train_rows": 1, "offset": 0.0}

    fit_segmented(ctx, G2_VARIANTS["split_horizon"], G2_VARIANTS_CFG(), 1, 2, 0.05, 10, fit=spy, log=lambda *_: None)
    assert seen == [(1, 21), (22, 42)]


def test_fit_segmented_ignores_validation_labels_when_partitioning() -> None:
    ctx = make_ctx()
    calls_a: list = []
    calls_b: list = []
    ctx_b = {**ctx, "valid": ctx["valid"].assign(ActualOrderVolume=ctx["valid"]["ActualOrderVolume"] * 2 + 1)}
    fit_segmented(ctx, G2_VARIANTS["long_window_only"], G2_VARIANTS_CFG(), 1, 2, 0.05, 10, fit=fake_fit(calls_a), log=lambda *_: None)
    fit_segmented(ctx_b, G2_VARIANTS["long_window_only"], G2_VARIANTS_CFG(), 1, 2, 0.05, 10, fit=fake_fit(calls_b), log=lambda *_: None)
    for x, y in zip(calls_a, calls_b):
        pd.testing.assert_frame_equal(x["train"], y["train"])
        pd.testing.assert_frame_equal(x["hold"], y["hold"])
        assert x["columns"] == y["columns"]


def G2_VARIANTS_CFG():
    from exp_runner import VARIANTS
    return VARIANTS["cs02"]


@pytest.fixture
def stub_src(monkeypatch):
    pkg = types.ModuleType("src")
    mod = types.ModuleType("src.model_pipeline")
    mod.CATEGORICAL_COLUMNS = {"Weekday"}
    monkeypatch.setitem(sys.modules, "src", pkg)
    monkeypatch.setitem(sys.modules, "src.model_pipeline", mod)


def test_end_to_end_with_real_lightgbm_on_synthetic_frames(stub_src) -> None:
    from exp_runner import fit_variant
    from helpers import fold_scores

    ctx = make_ctx()
    cfg = G2_VARIANTS_CFG()
    cfg = type(cfg)(**{**cfg.__dict__, "min_child": 20})
    direct, _ = fit_variant(ctx, cfg, 7, 1, 0.1, 40)
    for name in ("cs02", "split_horizon", "long_window_only"):
        oof, scores = fit_segmented(ctx, G2_VARIANTS[name], cfg, 7, 1, 0.1, 40, fit=fit_variant, log=lambda *_: None)
        assert len(oof) == len(ctx["valid"]) and oof.index.equals(ctx["valid"].index)
        assert (oof.loc[ctx["valid"]["IsOpen"] == 0, "pred"] == 0).all()
        assert np.isfinite(oof["pred"]).all()
        assert scores["rmsle_all"] < 0.6, (name, scores)
        if name == "cs02":  # single full-range segment == the harness fit
            assert np.allclose(oof["pred"], direct["pred"])
        check = fold_scores(ctx["valid"]["ActualOrderVolume"].to_numpy(float), oof["pred"].to_numpy(), ctx["valid"]["IsOpen"].to_numpy())
        assert check["rmsle_all"] == pytest.approx(scores["rmsle_all"])
