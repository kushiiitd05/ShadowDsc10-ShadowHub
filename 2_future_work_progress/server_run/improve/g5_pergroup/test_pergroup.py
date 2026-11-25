import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pergroup import fit_by_group, group_values, stitch, subset_by_values


def _ctx() -> dict:
    def frame(n: int, groups: list[str]) -> pd.DataFrame:
        return pd.DataFrame({"HubID": np.arange(n), "HubFormat": np.resize(groups, n), "IsOpen": 1, "ActualOrderVolume": 5.0})
    return {"train": frame(60, ["a", "a", "a", "b", "c"]), "hold": frame(10, ["a", "b", "c", "a", "a"]),
            "valid": frame(10, ["a", "b", "c", "a", "a"]), "fold": 4}


def test_groups_split_by_training_size() -> None:
    big, small = group_values(_ctx(), "HubFormat", min_rows=20)
    assert big == ["a"] and set(small) == {"b", "c"}


def test_subset_does_not_mutate_and_keeps_index() -> None:
    ctx = _ctx()
    sub = subset_by_values(ctx, "HubFormat", ["a"])
    assert len(ctx["valid"]) == 10 and set(sub["valid"]["HubFormat"]) == {"a"}
    assert sub["valid"].index.isin(ctx["valid"].index).all()


def test_stitch_requires_exact_cover() -> None:
    idx = pd.RangeIndex(4)
    parts = [pd.DataFrame({"pred": [1.0, 2.0]}, index=[0, 2]), pd.DataFrame({"pred": [3.0, 4.0]}, index=[1, 3])]
    assert stitch(parts, idx)["pred"].tolist() == [1.0, 3.0, 2.0, 4.0]
    with pytest.raises(ValueError):
        stitch(parts[:1], idx)


def test_fit_by_group_covers_validation_once_with_fake_fit() -> None:
    def fake_fit(sub, cfg, seed, n_jobs, lr, rounds):
        v = sub["valid"]
        oof = v[["HubID", "IsOpen", "ActualOrderVolume"]].copy()
        oof["Date"], oof["Horizon"], oof["pred"] = "d", 1, 5.0
        return oof, {"best_iteration": 10, "train_rows": len(sub["train"]), "fit_seconds": 0.1}
    sys.modules.setdefault("helpers", pytest.importorskip("helpers"))
    merged, scores = fit_by_group(_ctx(), "HubFormat", None, 0, 1, 0.05, 10, fake_fit, log=lambda *_: None)
    assert len(merged) == 10 and scores["rmsle_all"] == pytest.approx(0.0)
