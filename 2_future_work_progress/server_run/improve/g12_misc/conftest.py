"""pytest wiring for g12_misc: import paths + shared synthetic raw history (reuses the g3b synthetic generator)."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
IMPROVE = HERE.parent
for path in (IMPROVE / "g3b_features", IMPROVE / "g3_features", IMPROVE, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from synth_g3b import synthetic_raw  # noqa: E402


@pytest.fixture(scope="session")
def raw() -> pd.DataFrame:
    return synthetic_raw(n_hubs=30, n_days=1000)


@pytest.fixture()
def origin() -> pd.Timestamp:
    return pd.Timestamp("2014-10-02")


RAW_CANDIDATES = [Path("/Users/bournesmartasfuck_kush/Desktop/Dsc/ShadowHub-repo/1_midterm/data/raw")]
RAW_DIR = next((p for p in RAW_CANDIDATES if (p / "orders_train.csv").exists()), None)


@pytest.fixture(scope="session")
def real_history() -> pd.DataFrame:
    if RAW_DIR is None:
        pytest.skip("real raw files not available")
    hist = pd.read_csv(RAW_DIR / "orders_train.csv", parse_dates=["Date"])
    meta = pd.read_csv(RAW_DIR / "hub_metadata.csv")[["HubID", "HubFormat", "AssortmentTier"]]
    return hist.merge(meta, on="HubID", how="left")
