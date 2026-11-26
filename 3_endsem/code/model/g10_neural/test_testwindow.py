"""Tests for the TabPFN-TS test-window mode: alignment to the test frame, no history after the origin, closed rows 0, no NaN."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import neural_testwindow as tw  # noqa: E402
from neural_prep import build_future_frame  # noqa: E402

ORIGIN = pd.Timestamp("2015-01-10")
H = 5
HUBS = (3, 1, 2)  # deliberately not sorted: the Id order of the frame must be kept


def make_base(n_days: int = 20, hubs=(1, 2, 3)) -> pd.DataFrame:
    rows = [{"HubID": hub, "Date": d, "OrderVolume": float(100 * hub + k), "IsOpen": int(d.day_of_week != 6), "PromoActive": k % 2,
             "RegionalHoliday": 0, "SchoolClosureFlag": int(k % 5 == 0)}
            for hub in hubs for k, d in enumerate(pd.date_range("2015-01-01", periods=n_days))]
    return pd.DataFrame(rows)


def make_frame(hubs=HUBS) -> pd.DataFrame:
    """Test-frame stand-in: Id follows the row order (hub-major here), flags known for the future days."""
    rows = []
    for hub in hubs:
        for k, d in enumerate(pd.date_range(ORIGIN + pd.Timedelta(days=1), periods=H)):
            rows.append({"HubID": hub, "Date": d, "IsOpen": int(d.day_of_week != 6), "PromoActive": 1, "RegionalHoliday": 0,
                         "SchoolClosureFlag": 0, "ForecastOrigin": ORIGIN})
    frame = pd.DataFrame(rows)
    frame.insert(0, "Id", np.arange(1, len(frame) + 1) * 7)  # non-trivial ids
    return frame


def fake_forecast(frame: pd.DataFrame, value: float = 3.0) -> pd.DataFrame:
    fut = build_future_frame(frame, ORIGIN, H)
    fut["log_pred"] = value + fut["unique_id"].astype(int).to_numpy() * 0.1
    return fut[["unique_id", "ds", "log_pred"]].sample(frac=1.0, random_state=0).reset_index(drop=True)  # shuffled


# ---- alignment ------------------------------------------------------------------
def test_align_keeps_the_frame_row_and_id_order_even_for_shuffled_forecasts() -> None:
    frame = make_frame()
    out = tw.align_to_frame(fake_forecast(frame), frame)
    assert list(out.columns) == ["Id", "OrderVolume"] and out["Id"].tolist() == frame["Id"].tolist()
    hub = frame["HubID"].to_numpy()
    expect = np.expm1(3.0 + hub * 0.1)
    is_open = frame["IsOpen"].to_numpy() == 1
    assert out["OrderVolume"].to_numpy()[is_open] == pytest.approx(expect[is_open])


def test_align_zeroes_closed_rows_and_leaves_inputs_untouched() -> None:
    frame, fc = make_frame(), fake_forecast(make_frame())
    keep_f, keep_fc = frame.copy(), fc.copy()
    out = tw.align_to_frame(fc, frame)
    assert (out["OrderVolume"].to_numpy()[frame["IsOpen"].to_numpy() == 0] == 0).all() and (frame["IsOpen"] == 0).any()
    pd.testing.assert_frame_equal(frame, keep_f)
    pd.testing.assert_frame_equal(fc, keep_fc)


def test_align_refuses_missing_duplicate_and_unknown_rows() -> None:
    frame, fc = make_frame(), fake_forecast(make_frame())
    with pytest.raises(ValueError, match="no forecast"):
        tw.align_to_frame(fc.iloc[1:], frame)
    with pytest.raises(ValueError, match="duplicate"):
        tw.align_to_frame(pd.concat([fc, fc.iloc[:1]], ignore_index=True), frame)


def test_align_turns_nan_forecast_into_a_refusal_not_a_silent_zero() -> None:
    frame, fc = make_frame(), fake_forecast(make_frame())
    fc.loc[0, "log_pred"] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        tw.align_to_frame(fc, frame)


# ---- history / frame guards -------------------------------------------------------
def test_history_has_nothing_after_the_origin() -> None:
    hist = tw.history_until(make_base(), ORIGIN)
    assert pd.to_datetime(hist["Date"]).max() == ORIGIN


def test_history_guard_fires_when_rows_after_origin_would_survive() -> None:
    late = make_base().assign(Date=lambda d: pd.to_datetime(d["Date"]) + pd.Timedelta(days=30))
    with pytest.raises(ValueError):
        tw.history_until(late, ORIGIN)  # nothing on or before the origin


def test_assert_history_before_window_rejects_late_rows() -> None:
    with pytest.raises(AssertionError):
        tw.assert_history_before_window(make_base(), ORIGIN)  # base runs to 2015-01-20 > origin
    tw.assert_history_before_window(tw.history_until(make_base(), ORIGIN), ORIGIN)


def test_check_test_frame_accepts_a_complete_grid_and_rejects_others() -> None:
    frame = make_frame()
    assert tw.check_test_frame(frame, horizon=H, n_hubs=3, expected_rows=15) == ORIGIN
    with pytest.raises(AssertionError):
        tw.check_test_frame(frame.iloc[:-1], horizon=H, n_hubs=3, expected_rows=15)
    with pytest.raises(AssertionError):
        tw.check_test_frame(frame.assign(Id=1), horizon=H, n_hubs=3, expected_rows=15)  # Ids not unique
    with pytest.raises(AssertionError):
        tw.check_test_frame(frame.assign(ForecastOrigin=ORIGIN + pd.Timedelta(days=1)), horizon=H, n_hubs=3, expected_rows=15)


def test_hubs_without_history_are_refused() -> None:
    with pytest.raises(AssertionError, match="no history"):
        tw.assert_hubs_have_history(make_base(hubs=(1, 2)), make_frame())


# ---- output validation + manifest -------------------------------------------------
def test_validate_predictions_flags_nan_negative_and_open_closed_mixups() -> None:
    frame = make_frame()
    good = tw.align_to_frame(fake_forecast(frame), frame)
    tw.validate_predictions(good, frame)
    for bad_value in (np.nan, -1.0):
        bad = good.copy()
        bad.loc[bad.index[frame["IsOpen"] == 1][0], "OrderVolume"] = bad_value
        with pytest.raises(AssertionError):
            tw.validate_predictions(bad, frame)
    bad = good.copy()
    bad.loc[bad.index[frame["IsOpen"] == 0][0], "OrderVolume"] = 5.0
    with pytest.raises(AssertionError, match="closed"):
        tw.validate_predictions(bad, frame)
    with pytest.raises(AssertionError, match="Id"):
        tw.validate_predictions(good.assign(Id=good["Id"].iloc[::-1].to_numpy()), frame)


def test_write_outputs_roundtrip_hash_and_manifest(tmp_path: Path) -> None:
    frame = make_frame()
    table = tw.align_to_frame(fake_forecast(frame), frame)
    manifest = tw.write_outputs(table, frame, tmp_path, origin=ORIGIN, settings={"context": 365, "impute": 1}, seconds=1.5)
    csv, npy = tmp_path / tw.PRED_CSV, tmp_path / tw.PRED_NPY
    assert csv.exists() and npy.exists() and (tmp_path / tw.MANIFEST).exists()
    assert manifest["predictions_sha256"] == tw.sha256_file(csv) and manifest["npy_sha256"] == tw.sha256_file(npy)
    assert manifest["rows"] == len(frame) and manifest["forecast_origin"] == "2015-01-10" and manifest["first_target_date"] == "2015-01-11"
    assert manifest["settings"]["context"] == 365
    assert json.loads((tmp_path / tw.MANIFEST).read_text()) == manifest
    back = pd.read_csv(csv)
    assert back["Id"].tolist() == frame["Id"].tolist() and np.load(npy) == pytest.approx(back["OrderVolume"].to_numpy())
    assert np.isfinite(manifest["mean_log1p_open"])


def test_write_outputs_refuses_to_overwrite(tmp_path: Path) -> None:
    frame = make_frame()
    table = tw.align_to_frame(fake_forecast(frame), frame)
    tw.write_outputs(table, frame, tmp_path, origin=ORIGIN, settings={}, seconds=0.0)
    with pytest.raises(FileExistsError):
        tw.write_outputs(table, frame, tmp_path, origin=ORIGIN, settings={}, seconds=0.0)


# ---- end to end with a stub model --------------------------------------------------
def test_run_end_to_end_with_stub_model(tmp_path: Path, monkeypatch) -> None:
    proc = tmp_path / "proj/data/processed"
    proc.mkdir(parents=True)
    make_base().to_csv(proc / "base_train.csv.gz", index=False)
    make_frame().to_csv(proc / "test_forecast_features.csv.gz", index=False)
    seen = {}

    def stub(hist, valid, origin, args):
        seen["hist_max"] = pd.to_datetime(hist["Date"]).max()
        return fake_forecast(valid)

    monkeypatch.setattr(tw, "predict_tabpfn_ts", stub)
    monkeypatch.setattr(tw, "EXPECTED_ROWS", 15)
    monkeypatch.setattr(tw, "EXPECTED_HUBS", 3)
    monkeypatch.setattr(tw, "HORIZON", H)
    args = tw.parse_args([str(tmp_path / "proj"), str(tmp_path / "out")])
    assert tw.run(args) == 0
    assert seen["hist_max"] == ORIGIN  # history was cut at the origin although the base file runs further
    table = pd.read_csv(tmp_path / "out" / tw.PRED_CSV)
    assert table["Id"].tolist() == make_frame()["Id"].tolist() and len(table) == 15
