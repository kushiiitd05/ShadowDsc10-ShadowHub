"""predict_frozen must give exactly what final_retrain.py wrote (run with pytest from 3_endsem)."""
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
for p in (HERE, HERE / "g8_final"):
    sys.path.insert(0, str(p))
from predict_frozen import ROOT, blend, frozen_forecast  # noqa: E402


def test_blend_matches_final_retrain_formula() -> None:
    from final_retrain import blend_external, combine_members
    rng = np.random.default_rng(0)
    is_open = (rng.random(500) > 0.15).astype(int)
    seeds = [rng.lognormal(8, 0.5, 500) for _ in range(3)]
    tabpfn = rng.lognormal(8, 0.5, 500)
    expected = blend_external(combine_members(seeds, [], is_open), tabpfn, 0.2, is_open)
    got = blend(seeds, tabpfn, is_open)
    assert np.allclose(got, expected, rtol=0, atol=1e-9)
    assert (got[is_open == 0] == 0).all()


def test_frozen_forecast_reproduces_the_submission() -> None:
    if not (ROOT / "final" / "forecast_matrix_open.parquet").exists():
        pytest.skip("frozen artifacts not present")
    import pandas as pd
    table, _ = frozen_forecast()
    frozen = pd.read_csv(ROOT / "final" / "submission_final.csv")
    assert (table["Id"].to_numpy() == frozen["Id"].to_numpy()).all()
    assert np.max(np.abs(np.log1p(table["OrderVolume"]) - np.log1p(frozen["OrderVolume"]))) < 1e-9
