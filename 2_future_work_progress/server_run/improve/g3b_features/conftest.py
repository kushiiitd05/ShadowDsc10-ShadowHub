"""pytest wiring for the g3b tests: import paths + shared fixtures (the helper module is synth_g3b)."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
for path in (HERE, HERE.parent, HERE.parent / "g3_features"):
    if str(path) not in sys.path:
        sys.path.append(str(path))

from synth_g3b import synthetic_raw  # noqa: E402


@pytest.fixture(scope="session")
def raw() -> pd.DataFrame:
    return synthetic_raw()


@pytest.fixture()
def origin() -> pd.Timestamp:
    return pd.Timestamp("2014-10-02")
