"""Tests for the pure helpers of the G9 diagnostics (run: python -m pytest test_diag.py)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
for p in (HERE, HERE.parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from adv_validation import adversarial_cv, auc_score, day_key, is_origin_identity, rank_drift, standardize_within  # noqa: E402
from perm_prune import (GROUP_ORDER, SUBGROUP_TO_GROUP, assign_group, assign_subgroup, columns_by, gate_verdict,  # noqa: E402
                        is_calendar_like, permute_columns, select_prune_groups)

REAL_COLUMNS = [c for c in (HERE / "feature_columns.txt").read_text().split() if c]


# ---------------------------------------------------------------- group assignment
@pytest.mark.parametrize("col,group,sub", [
    ("Orders_TargetLag7D", "lags", "lags_orders"),
    ("Sessions_TargetLag364D", "lags", "lags_sessions"),
    ("TargetLag28D_Available", "lags", "lags_avail"),
    ("Orders_AtOriginMinus14D", "lags", "lags_orders"),
    ("Orders_28D_MeanOpen", "rolling", "roll_orders_level"),
    ("Orders_365D_OpenTrend", "rolling", "roll_orders_level"),
    ("Orders_90D_OpenCount", "rolling", "roll_orders_count"),
    ("Sessions_7D_PresentCount", "rolling", "roll_sessions_count"),
    ("Conversion_90D_Median", "rolling", "roll_ratio"),
    ("VolumePerSession_28D_Std", "rolling", "roll_ratio"),
    ("Weekday", "calendar", "cal_date"),
    ("Horizon", "calendar", "cal_date"),
    ("Open_StreakDay", "calendar", "cal_open"),
    ("IsOpen", "calendar", "cal_open"),
    ("PromoActive", "promo", "promo"),
    ("Promo_DaysUntilActiveVisible", "promo", "promo"),
    ("RegionalHoliday_Tomorrow", "holiday", "holiday"),
    ("SchoolClosure_StreakDay", "holiday", "school"),
    ("HubFormat", "hub_stats", "hub_static"),
    ("CompetitorDistanceLog1p", "hub_stats", "hub_static"),
    ("HistHubDOWOrdersMean", "hub_stats", "hub_hist"),
    ("HistoryRows", "hub_stats", "hub_hist"),
    ("HubID", "encoding", "encoding"),
    ("HubID_TE", "encoding", "encoding"),
])
def test_known_columns(col: str, group: str, sub: str) -> None:
    assert assign_group(col) == group and assign_subgroup(col) == sub


def test_unknown_column_is_other() -> None:
    assert assign_subgroup("g3c_mystery") == "other" and assign_group("g3c_mystery") == "other"


def test_every_real_feature_column_is_assigned() -> None:
    assert len(REAL_COLUMNS) > 300
    assert [c for c in REAL_COLUMNS if assign_subgroup(c) == "other"] == []
    assert {assign_group(c) for c in REAL_COLUMNS + ["HubID_TE"]} == set(GROUP_ORDER)
    assert set(SUBGROUP_TO_GROUP.values()) == set(GROUP_ORDER)


def test_columns_by_partitions_without_overlap_and_keeps_order() -> None:
    cols = REAL_COLUMNS + ["HubID_TE"]
    for level in ("group", "subgroup"):
        parts = columns_by(cols, level)
        flat = [c for v in parts.values() for c in v]
        assert sorted(flat) == sorted(cols) and len(flat) == len(set(flat))
        for v in parts.values():
            assert v == [c for c in cols if c in v]


def test_calendar_like_flags_time_columns_not_lags() -> None:
    assert all(is_calendar_like(c) for c in ("Year", "DayOfYear", "ForecastWeek", "Open_StreakDay", "HistoryRows", "DaysSinceRenovationReopen"))
    assert not any(is_calendar_like(c) for c in ("Orders_TargetLag7D", "Orders_28D_MeanOpen", "PromoActive", "HubFormat"))


# ---------------------------------------------------------------- permutation
def _frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({"a": rng.normal(size=200), "b": np.arange(200, dtype="float32"),
                         "c": pd.Categorical(rng.choice(["x", "y", "z"], size=200)), "keep": rng.normal(size=200)})


def test_permute_leaves_input_unmutated() -> None:
    f = _frame()
    snapshot = f.copy(deep=True)
    out = permute_columns(f, ["a", "b", "c"], np.random.default_rng(1))
    pd.testing.assert_frame_equal(f, snapshot)
    assert out is not f and not np.shares_memory(out["a"].to_numpy(), f["a"].to_numpy())


def test_permute_shuffles_only_requested_columns_jointly() -> None:
    f = _frame()
    out = permute_columns(f, ["a", "b", "c"], np.random.default_rng(1))
    pd.testing.assert_series_equal(out["keep"], f["keep"])
    assert not np.array_equal(out["a"].to_numpy(), f["a"].to_numpy())
    assert np.allclose(np.sort(out["a"]), np.sort(f["a"]))
    assert str(out["c"].dtype) == "category" and list(out["c"].cat.categories) == list(f["c"].cat.categories)
    # joint shuffle: the (a, b) row pairs are the same set as before
    assert set(zip(out["a"].round(8), out["b"])) == set(zip(f["a"].round(8), f["b"]))
    # one shared permutation: b is the row id, so a must follow it
    lookup = dict(zip(f["b"], f["a"]))
    assert all(lookup[b] == a for a, b in zip(out["a"], out["b"]))


def test_permute_is_deterministic_per_seed() -> None:
    f = _frame()
    a = permute_columns(f, ["a"], np.random.default_rng(5))
    b = permute_columns(f, ["a"], np.random.default_rng(5))
    c = permute_columns(f, ["a"], np.random.default_rng(6))
    pd.testing.assert_frame_equal(a, b)
    assert not a["a"].equals(c["a"])


# ---------------------------------------------------------------- AUC + adversarial CV
def test_auc_score_known_values() -> None:
    assert auc_score(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert auc_score(np.array([0, 0, 1, 1]), np.array([0.9, 0.8, 0.2, 0.1])) == 0.0
    assert auc_score(np.array([0, 1, 0, 1]), np.array([0.5, 0.5, 0.5, 0.5])) == 0.5
    with pytest.raises(ValueError):
        auc_score(np.zeros(4), np.arange(4))


def test_auc_score_matches_sklearn_on_random_data() -> None:
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(3)
    y, s = rng.integers(0, 2, 500), rng.normal(size=500)
    assert auc_score(y, s) == pytest.approx(roc_auc_score(y, s))


def _synthetic(drift: float, n: int = 3000, hubs: int = 30) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    y = rng.integers(0, 2, n)
    x = pd.DataFrame({"drifty": rng.normal(size=n) + drift * y, "noise1": rng.normal(size=n), "noise2": rng.normal(size=n)})
    return x, y, rng.integers(0, hubs, n)


def test_adversarial_cv_detects_drift_and_names_the_feature() -> None:
    x, y, g = _synthetic(drift=3.0)
    before = x.copy(deep=True)
    auc, sd, share = adversarial_cv(x, y, g, rounds=40, n_jobs=2, seed=0)
    pd.testing.assert_frame_equal(x, before)
    assert auc > 0.9 and sd < 0.1
    assert share.idxmax() == "drifty" and share["drifty"] > 0.7 and share.sum() == pytest.approx(1.0)


def test_adversarial_cv_is_near_chance_without_drift() -> None:
    x, y, g = _synthetic(drift=0.0)
    auc, _, _ = adversarial_cv(x, y, g, rounds=40, n_jobs=2, seed=0)
    assert 0.44 < auc < 0.56


def test_rank_drift_orders_by_share_excludes_calendar_and_needs_beating_placebo() -> None:
    cols = ["Year", "Orders_28D_MeanOpen", "PromoActive"]
    s = {"1": pd.Series({"Orders_28D_MeanOpen": 0.5, "PromoActive": 0.01}), "2": pd.Series({"Orders_28D_MeanOpen": 0.3, "PromoActive": 0.01})}
    real, placebo = pd.Series({"1": 0.9, "2": 0.9}), pd.Series({"1": 0.55, "2": 0.55})
    t = rank_drift(s, {"1": s["1"]}, real, placebo, cols)
    assert list(t["feature"])[0] == "Orders_28D_MeanOpen" and t["rank"].tolist() == [1, 2, 3]
    flags = t.set_index("feature")["drift_candidate"]
    assert flags["Orders_28D_MeanOpen"] and not flags["PromoActive"] and not flags["Year"]
    same = rank_drift(s, {}, pd.Series({"1": 0.9}), pd.Series({"1": 0.88}), cols)
    assert not same["drift_candidate"].any()


def test_standardize_within_removes_origin_level_and_leaves_input_alone() -> None:
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"v": np.r_[rng.normal(10, 2, 300), rng.normal(100, 20, 300)], "const": 5.0, "keep": np.arange(600)})
    by = np.r_[np.zeros(300), np.ones(300)]
    snap = frame.copy(deep=True)
    out = standardize_within(frame, by, ["v", "const"])
    pd.testing.assert_frame_equal(frame, snap)
    for g in (0, 1):
        z = out.loc[by == g, "v"]
        assert abs(z.median()) < 1e-6 and 0.5 < (z.quantile(0.75) - z.quantile(0.25)) < 1.5 + 1e-6
    assert (out["const"] == 0).all() and out["keep"].equals(frame["keep"])


def test_origin_identity_columns() -> None:
    assert all(is_origin_identity(c) for c in ("Year", "PromoActive", "Promo_StreakDay", "RegionalHoliday_Tomorrow", "SchoolClosure_StreakDay",
                                               "TargetLag7D_Available", "Orders_TargetLag728D", "Orders_730D_MeanOpen", "Orders_TargetLag364D", "HistoryRows"))
    assert not any(is_origin_identity(c) for c in ("Orders_TargetLag7D", "Orders_28D_MeanOpen", "Orders_AtOrigin", "HubFormat", "HistHubDOWOrdersMean", "Conversion_90D_Mean"))


def test_day_key_is_integer_days() -> None:
    k = day_key(np.array(["2015-01-02T00:00:00", "2015-01-03T12:00:00"], dtype="datetime64[ns]"))
    assert k[1] - k[0] == 1 and k.dtype == np.int64


# ---------------------------------------------------------------- pruning helpers
def _scores(deltas: dict[tuple[int, int], float], base: float = 0.2) -> pd.DataFrame:
    rows = [{"seed": s, "fold": f, "rmsle_all": base + d, "rmsle_excl_open0": base + d} for (s, f), d in deltas.items()]
    return pd.DataFrame(rows)


def test_gate_passes_when_gain_is_large_consistent_and_excl_holds() -> None:
    keys = [(s, f) for s in (1, 2, 3) for f in (2, 3, 4)]
    v = gate_verdict(_scores({k: 0.0 for k in keys}), _scores({k: -0.003 for k in keys}))
    assert v["gate_pass"] and v["all"]["folds_over_threshold"] == 3


def test_gate_fails_below_threshold_or_mixed_sign() -> None:
    keys = [(s, f) for s in (1, 2, 3) for f in (2, 3, 4)]
    assert not gate_verdict(_scores({k: 0.0 for k in keys}), _scores({k: -0.001 for k in keys}))["gate_pass"]
    cand = {k: (-0.004 if k[0] != 3 else 0.004) for k in keys}
    assert not gate_verdict(_scores({k: 0.0 for k in keys}), _scores(cand))["gate_pass"]


def test_gate_fails_when_gain_vanishes_excluding_open_zero() -> None:
    keys = [(s, f) for s in (1, 2) for f in (2, 3, 4)]
    cand = _scores({k: -0.003 for k in keys})
    cand["rmsle_excl_open0"] = 0.2
    v = gate_verdict(_scores({k: 0.0 for k in keys}), cand)
    assert v["all"]["passes"] and not v["excl_open0"]["passes"] and not v["gate_pass"]


def test_select_prune_groups_requires_nonpositive_cost_on_every_fold() -> None:
    r = pd.DataFrame({"name": ["a", "b", "c"], "cost_f3": [-0.001, 0.0, -0.002], "cost_f4": [-0.0005, 0.0004, 0.001]})
    assert select_prune_groups(r, ["cost_f3", "cost_f4"]) == ["a"]
