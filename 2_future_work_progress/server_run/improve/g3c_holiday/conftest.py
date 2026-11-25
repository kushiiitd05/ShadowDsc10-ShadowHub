"""pytest wiring for the g3c tests: import paths + shared fixtures (the helper module is synth_g3c)."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
IMPROVE = HERE.parent
for path in (HERE, IMPROVE, IMPROVE / "g3_features", IMPROVE / "g3b_features"):
    if str(path) not in sys.path:
        sys.path.append(str(path))

from synth_g3c import synthetic_raw  # noqa: E402


@pytest.fixture(scope="session")
def raw() -> pd.DataFrame:
    return synthetic_raw()


@pytest.fixture()
def origin() -> pd.Timestamp:
    return pd.Timestamp("2014-10-02")
