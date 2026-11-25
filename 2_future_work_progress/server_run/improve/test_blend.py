import numpy as np
import pandas as pd

from blend import blend_table


def _frame(name: str, preds: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"fold": 1, "HubID": [1, 2], "Date": ["2015-01-01"] * 2, "ActualOrderVolume": [100.0, 50.0], name: preds})


def test_blend_is_log_space_mean_and_perfect_when_members_bracket_truth() -> None:
    a, b = _frame("a", [100.0, 50.0]), _frame("b", [100.0, 50.0])
    table = blend_table({"a": a, "b": b})
    assert table.loc[1, "blend"] == 0.0
    c = _frame("c", [200.0, 100.0])
    t2 = blend_table({"a": a, "c": c})
    expected = np.expm1((np.log1p(100.0) + np.log1p(200.0)) / 2)
    assert 100.0 < expected < 200.0 and t2.loc[1, "blend"] < t2.loc[1, "c"]
