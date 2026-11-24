import numpy as np
import pytest

from helpers import decode_prediction, encode_target, fold_scores, level_baseline, recency_weights


def test_level_uses_first_positive_in_chain() -> None:
    cols = {
        "Orders_90D_MedianOpen": np.array([0.0, np.nan, 5.0, 0.0]),
        "Orders_180D_MedianOpen": np.array([7.0, 0.0, 6.0, 0.0]),
        "Orders_365D_MedianOpen": np.array([8.0, 9.0, 0.0, 0.0]),
        "HistHubDOWOrdersMean": np.array([1.0, 1.0, 1.0, np.nan]),
    }
    assert level_baseline(cols).tolist() == [7.0, 9.0, 5.0, 1.0]


def test_level_target_round_trip() -> None:
    y = np.array([0.0, 10.0, 6000.0])
    level = np.array([5.0, 12.0, 5500.0])
    raw = encode_target(y, level, "level")
    assert np.allclose(decode_prediction(raw, level, "level"), y)


def test_tweedie_init_score_added_back_by_hand() -> None:
    level = np.array([100.0, 400.0])
    raw_boost = np.array([0.1, -0.2])  # what LightGBM returns without init_score
    out = decode_prediction(raw_boost, level, "level_init")
    assert np.allclose(out, np.exp(raw_boost + np.log(level)))
    assert not np.allclose(out, np.exp(raw_boost))  # forgetting log(L) would be wrong


def test_closed_rows_do_not_use_level_decode_bias() -> None:
    assert decode_prediction(np.array([0.0]), np.array([10.0]), "log")[0] == 0.0


def test_fold_scores_excludes_only_open_zero_rows() -> None:
    actual = np.array([100.0, 0.0, 0.0, 50.0])
    pred = np.array([100.0, 0.0, 5000.0, 50.0])
    is_open = np.array([1, 0, 1, 1])
    s = fold_scores(actual, pred, is_open)
    assert s["n_open_zero"] == 1
    assert s["rmsle_excl_open0"] == pytest.approx(0.0)
    assert s["rmsle_all"] > 4.0


def test_recency_weights_halve_at_half_life() -> None:
    origins = np.array(["2015-01-02", "2014-01-02"], dtype="datetime64[D]")
    w = recency_weights(origins, np.datetime64("2015-01-02"), 365.0)
    assert w[0] == pytest.approx(1.0)
    assert w[1] == pytest.approx(0.5)
