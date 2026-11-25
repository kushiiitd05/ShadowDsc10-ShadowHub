from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import synthetic_raw
from feature_panel import build_panel


def test_panel_shapes_and_missing_rows_are_nan(raw: pd.DataFrame) -> None:
    panel = build_panel(raw)
    assert panel.orders.shape == (raw["HubID"].nunique(), len(panel.dates))
    assert panel.promo is not None
    present = raw.groupby("HubID").size().to_numpy()
    assert (np.isfinite(panel.is_open).sum(axis=1) == present).all()


def test_upto_cuts_future_and_keeps_origin_column(raw: pd.DataFrame, origin: pd.Timestamp) -> None:
    cut = build_panel(raw).upto(origin)
    assert cut.dates[-1] == origin
    assert (cut.dates <= origin).all()


def test_upto_does_not_mutate_source(raw: pd.DataFrame, origin: pd.Timestamp) -> None:
    panel = build_panel(raw)
    before = panel.orders.copy()
    panel.upto(origin)
    assert np.array_equal(before, panel.orders, equal_nan=True)


def test_upto_rejects_origin_outside_axis(raw: pd.DataFrame) -> None:
    with pytest.raises(ValueError):
        build_panel(raw).upto(pd.Timestamp("2030-01-01"))


def test_panel_ignores_rows_after_origin_when_built_from_truncated_raw(raw: pd.DataFrame, origin: pd.Timestamp) -> None:
    full = build_panel(raw).upto(origin)
    trunc = build_panel(raw[raw["Date"] <= origin]).upto(origin)
    assert np.array_equal(full.orders[:, -100:], trunc.orders[:, -100:], equal_nan=True)


def test_groups_present_only_when_columns_exist(raw: pd.DataFrame) -> None:
    with_groups = build_panel(raw)
    without = build_panel(raw.drop(columns=["HubFormat", "AssortmentTier"]))
    assert set(with_groups.groups) == {"HubFormat", "AssortmentTier", "HubFormat_AssortmentTier"}
    assert without.groups == {}


def test_promo_none_when_column_absent() -> None:
    assert build_panel(synthetic_raw().drop(columns=["PromoActive"])).promo is None


def test_duplicate_hub_date_rows_keep_last() -> None:
    raw = synthetic_raw(n_hubs=2, n_days=10)
    dup = raw.iloc[[0]].copy()
    dup["OrderVolume"] = 999.0
    panel = build_panel(pd.concat([raw, dup], ignore_index=True))
    assert panel.orders[0, 0] == 999.0
