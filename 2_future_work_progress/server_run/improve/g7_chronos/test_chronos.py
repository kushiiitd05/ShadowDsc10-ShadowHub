"""Tests for the zero-shot Chronos-2 member: history window (no leak), forecast alignment, closed-day rule, mocked model."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chronos_prep import (  # noqa: E402
    build_context, build_future, forecast_fold, forecast_to_oof, volume_from_log,
)

ORIGIN = pd.Timestamp("2015-01-10")
COV = ("IsOpen", "PromoActive")


def make_raw(days: int = 30, hubs=(1, 2), start="2015-01-01", gap_hub: int | None = None) -> pd.DataFrame:
    rows = []
    for h in hubs:
        for i, d in enumerate(pd.date_range(start, periods=days)):
            if gap_hub == h and 3 <= i <= 5:
                continue  # interior gap (hub rows missing)
            is_open = int(d.day_of_week != 6)
            rows.append({"HubID": h, "Date": d, "OrderVolume": (100 * h + i) * is_open, "IsOpen": is_open,
                         "PromoActive": i % 2, "RegionalHoliday": 0, "SchoolClosureFlag": 0})
    return pd.DataFrame(rows)


def make_fold(hubs=(1, 2), horizon: int = 4) -> pd.DataFrame:
    rows = []
    for h in hubs:
        for k in range(1, horizon + 1):
            d = ORIGIN + pd.Timedelta(days=k)
            rows.append({"HubID": h, "Date": d, "Horizon": k, "IsOpen": int(d.day_of_week != 6), "PromoActive": k % 2,
                         "RegionalHoliday": 0, "SchoolClosureFlag": 0, "ForecastOrigin": ORIGIN})
    return pd.DataFrame(rows)


# ---------- history window ----------
def test_context_never_contains_dates_after_origin() -> None:
    ctx = build_context(make_raw(), ORIGIN, COV)
    assert ctx["timestamp"].max() == ORIGIN
    assert (ctx["timestamp"] <= ORIGIN).all()


def test_context_is_invariant_to_post_origin_data() -> None:
    raw = make_raw()
    poisoned = raw.copy()
    poisoned.loc[poisoned["Date"] > ORIGIN, ["OrderVolume", "PromoActive"]] = 9_999_999
    a = build_context(raw, ORIGIN, COV).reset_index(drop=True)
    b = build_context(poisoned, ORIGIN, COV).reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)


def test_context_target_is_log1p_and_regular_daily_with_nan_for_gaps() -> None:
    raw = make_raw(gap_hub=2)
    ctx = build_context(raw, ORIGIN, COV)
    for h, g in ctx.groupby("item_id"):
        assert (g["timestamp"].diff().dropna() == pd.Timedelta(days=1)).all()  # no holes
        assert g["timestamp"].iloc[0] == pd.Timestamp("2015-01-01")
    g1 = ctx[ctx["item_id"] == 1].set_index("timestamp")
    assert g1.loc["2015-01-03", "target"] == pytest.approx(np.log1p(100 * 1 + 2))
    g2 = ctx[ctx["item_id"] == 2].set_index("timestamp")
    assert g2.loc["2015-01-04":"2015-01-06", "target"].isna().all()  # missing rows -> NaN, not 0
    assert g2.loc["2015-01-05", "IsOpen"] == 0 and g2["target"].notna().sum() == 10 - 3


def test_context_starts_at_each_hubs_first_observation() -> None:
    raw = make_raw()
    raw = raw[~((raw["HubID"] == 2) & (raw["Date"] < "2015-01-04"))]
    ctx = build_context(raw, ORIGIN, COV)
    assert ctx[ctx["item_id"] == 2]["timestamp"].min() == pd.Timestamp("2015-01-04")
    assert ctx[ctx["item_id"] == 1]["timestamp"].min() == pd.Timestamp("2015-01-01")


def test_context_covariate_columns_only_when_requested_and_input_not_mutated() -> None:
    raw = make_raw()
    before = raw.copy()
    uni = build_context(raw, ORIGIN, ())
    assert list(uni.columns) == ["item_id", "timestamp", "target"]
    assert list(build_context(raw, ORIGIN, COV).columns) == ["item_id", "timestamp", "target", *COV]
    pd.testing.assert_frame_equal(raw, before)


def test_context_hub_subset() -> None:
    ctx = build_context(make_raw(hubs=(1, 2, 3)), ORIGIN, (), hubs=[3, 1])
    assert sorted(ctx["item_id"].unique()) == [1, 3]


# ---------- future covariates ----------
def test_future_frame_has_covariates_and_sorted_keys() -> None:
    fut = build_future(make_fold(), COV)
    assert list(fut.columns) == ["item_id", "timestamp", *COV]
    assert fut["timestamp"].min() == ORIGIN + pd.Timedelta(days=1)
    assert fut.equals(fut.sort_values(["item_id", "timestamp"]).reset_index(drop=True))


# ---------- volume / closed days ----------
def test_volume_from_log_clips_and_zeroes_closed_rows() -> None:
    out = volume_from_log(np.array([np.log1p(10.0), -3.0, np.log1p(10.0)]), np.array([1, 1, 0]))
    assert out[0] == pytest.approx(10.0) and out[1] == 0.0 and out[2] == 0.0


# ---------- forecast -> frame alignment ----------
def fake_forecast(fold: pd.DataFrame, q: str = "0.5", shuffle: bool = True) -> pd.DataFrame:
    """Chronos-style predict_df output: item_id, timestamp, target_name, predictions, quantile columns; value encodes (hub, day)."""
    out = pd.DataFrame({"item_id": fold["HubID"], "timestamp": fold["Date"], "target_name": "target"})
    out["predictions"] = 0.0
    out[q] = np.log1p(fold["HubID"].to_numpy() * 10.0 + fold["Horizon"].to_numpy())
    return out.sample(frac=1.0, random_state=0).reset_index(drop=True) if shuffle else out


def test_alignment_matches_by_key_not_by_position() -> None:
    fold = make_fold()
    actual = pd.DataFrame({"HubID": fold["HubID"], "Date": fold["Date"], "OrderVolume": 5})
    oof = forecast_to_oof(fake_forecast(fold), fold, actual, fold_number=4)
    assert list(oof.columns) == ["fold", "HubID", "Date", "Horizon", "IsOpen", "ActualOrderVolume", "pred"]
    open_rows = oof[oof["IsOpen"] == 1]
    expect = open_rows["HubID"] * 10.0 + open_rows["Horizon"]
    np.testing.assert_allclose(open_rows["pred"].to_numpy(), expect.to_numpy(), rtol=1e-9)
    assert (oof.loc[oof["IsOpen"] == 0, "pred"] == 0).all()
    assert (oof["fold"] == 4).all() and len(oof) == len(fold)
    assert (oof["ActualOrderVolume"] == 5).all()


def test_alignment_rejects_missing_or_duplicate_forecast_rows() -> None:
    fold = make_fold()
    actual = pd.DataFrame({"HubID": fold["HubID"], "Date": fold["Date"], "OrderVolume": 5})
    fc = fake_forecast(fold)
    with pytest.raises(ValueError):
        forecast_to_oof(fc.iloc[:-1], fold, actual, fold_number=1)
    with pytest.raises(ValueError):
        forecast_to_oof(pd.concat([fc, fc.iloc[:1]]), fold, actual, fold_number=1)


def test_alignment_rejects_missing_actuals() -> None:
    fold = make_fold()
    actual = pd.DataFrame({"HubID": fold["HubID"], "Date": fold["Date"], "OrderVolume": 5}).iloc[:-1]
    with pytest.raises(ValueError):
        forecast_to_oof(fake_forecast(fold), fold, actual, fold_number=1)


# ---------- mocked model end to end ----------
class MockPipeline:
    """Stands in for Chronos2Pipeline: records its inputs and predicts the last context value + 0.1 for every future day."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def predict_df(self, df, future_df=None, id_column="item_id", timestamp_column="timestamp", target="target",
                   prediction_length=None, quantile_levels=(0.5,), batch_size=256, **kw):
        self.calls.append({"df": df, "future_df": future_df, "prediction_length": prediction_length})
        rows = []
        for h, g in df.groupby(id_column):
            last = g.sort_values(timestamp_column)[target].dropna().iloc[-1]
            for k in range(1, prediction_length + 1):
                rows.append({id_column: h, timestamp_column: g[timestamp_column].max() + pd.Timedelta(days=k),
                             "target_name": target, "predictions": last, "0.5": last + 0.1})
        return pd.DataFrame(rows)


def test_forecast_fold_uses_only_pre_origin_history_and_returns_aligned_oof() -> None:
    raw = make_raw(days=30)                      # data runs to 2015-01-30, i.e. well past the origin
    fold = make_fold(horizon=4)
    pipe = MockPipeline()
    oof = forecast_fold(pipe, raw, fold, fold_number=2, covariates=COV, batch_size=8)
    call = pipe.calls[0]
    assert call["df"]["timestamp"].max() == ORIGIN
    assert call["future_df"]["timestamp"].min() == ORIGIN + pd.Timedelta(days=1)
    assert call["prediction_length"] == 4
    assert len(oof) == len(fold) and oof["pred"].notna().all()
    # actuals come from the raw table on the target dates
    merged = oof.merge(raw, on=["HubID", "Date"])
    assert (merged["ActualOrderVolume"] == merged["OrderVolume"]).all()
    closed = oof[oof["IsOpen"] == 0]
    assert (closed["pred"] == 0).all()


def test_forecast_fold_univariate_sends_no_future_covariates() -> None:
    pipe = MockPipeline()
    forecast_fold(pipe, make_raw(), make_fold(), fold_number=1, covariates=(), batch_size=8)
    assert pipe.calls[0]["future_df"] is None
    assert list(pipe.calls[0]["df"].columns) == ["item_id", "timestamp", "target"]
