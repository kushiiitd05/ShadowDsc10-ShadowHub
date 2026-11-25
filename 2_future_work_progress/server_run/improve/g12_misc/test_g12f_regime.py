from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import g12f_regime as r


def _frame():
    return pd.DataFrame({"Weekday": [1, 7, 3, 7, 2], "RegionalHoliday": [0, 0, 1, 2, 0], "x": range(5)})


def test_special_flag_sunday_or_holiday_and_strings():
    f = _frame()
    assert r.special_flag(f).tolist() == [False, True, True, True, False]
    g = f.assign(Weekday=f["Weekday"].astype(str), RegionalHoliday=["0", "0", "a", "0", None])
    assert r.special_flag(g).tolist() == [False, True, False, True, False]


def test_regime_ctx_filters_train_hold_only_and_is_pure():
    f = _frame()
    ctx = {"train": f, "hold": f.copy(), "valid": f.copy(), "origin": 1}
    before = f.copy()
    s, o = r.regime_ctx(ctx, True), r.regime_ctx(ctx, False)
    assert len(s["train"]) == 3 and len(o["train"]) == 2 and len(s["valid"]) == 5 == len(o["valid"])
    assert s["hold"]["x"].tolist() == [1, 2, 3] and o["hold"]["x"].tolist() == [0, 4]
    pd.testing.assert_frame_equal(f, before)
    assert ctx["train"] is f and s["origin"] == 1


def test_blend_gates():
    p_g = np.array([100.0, 100.0, 100.0, 0.0])
    p_s = np.array([200.0, 200.0, 200.0, 5.0])
    p_o = np.array([50.0, 50.0, 50.0, 7.0])
    sp = np.array([True, False, True, False])
    hard = r.blend_log(p_g, p_s, p_o, sp, (1.0, 1.0))
    np.testing.assert_allclose(hard, [200, 50, 200, 7])
    spec = r.blend_log(p_g, p_s, p_o, sp, (1.0, 0.0))
    np.testing.assert_allclose(spec, [200, 100, 200, 0])
    half = r.blend_log(p_g, p_s, p_o, sp, (0.5, 0.5))
    assert half[0] == pytest.approx(np.expm1(0.5 * np.log1p(200) + 0.5 * np.log1p(100)))
    assert half[3] == pytest.approx(np.expm1(0.5 * np.log1p(7)))
    zero_expert = r.blend_log(np.array([100.0]), np.array([0.0]), np.array([0.0]), np.array([True]), (1.0, 1.0))
    assert zero_expert[0] == 100.0                  # an expert that predicts 0 never closes a day cs02 considers open
    np.testing.assert_allclose(r.blend_log(p_g, p_s, p_o, sp, (0.0, 0.0)), p_g)


def test_join_global_alignment_and_missing():
    key = pd.DataFrame({"fold": 3, "HubID": [2, 1, 3], "Date": pd.Timestamp("2015-04-01")})
    g = key.assign(pred=[20.0, 10.0, 30.0]).sample(frac=1, random_state=0).reset_index(drop=True)
    g.loc[g.HubID == 2, "pred"] = 22.0
    np.testing.assert_allclose(r.join_global(key, g), [22.0, 10.0, 30.0])
    with pytest.raises(ValueError):
        r.join_global(key, g[g.HubID != 3])


def test_blends_registry():
    assert set(r.BLENDS) == {"g12f_hard", "g12f_half", "g12f_spec", "g12f_spec_half"}
    assert all(0 <= a <= 1 and 0 <= b <= 1 for a, b in r.BLENDS.values())
