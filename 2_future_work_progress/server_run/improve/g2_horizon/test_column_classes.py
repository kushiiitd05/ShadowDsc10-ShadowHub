from pathlib import Path

import pytest

from column_classes import classify_column, split_columns, window_days

SNAPSHOT = Path(__file__).with_name("columns_snapshot.txt")


@pytest.mark.parametrize("name,expected", [
    ("Orders_7D_MeanOpen", 7), ("Sessions_730D_StdOpen", 730), ("Orders_28D_OpenTrend", 28),
    ("Conversion_56D_Mean", 56), ("VolumePerSession_365D_Count", 365),
    ("Orders_TargetLag14D", 14), ("TargetLag364D_Available", 364), ("Orders_AtOriginMinus28D", 28),
    ("Orders_AtOrigin", 0), ("Horizon", None), ("Open_StreakDay", None), ("HistHubDOWOrdersMean", None),
])
def test_window_days(name: str, expected: int | None) -> None:
    assert window_days(name) == expected


@pytest.mark.parametrize("name,label", [
    ("Orders_7D_MeanOpen", "short_history"), ("Sessions_7D_PresentCount", "short_history"),
    ("Orders_AtOrigin", "short_history"), ("Sessions_AtOriginMinus7D", "short_history"),
    ("Orders_AtOriginMinus28D", "short_history"),
    ("Orders_TargetLag7D", "short_history"), ("TargetLag14D_Available", "short_history"),
    ("Orders_28D_MeanOpen", "long_history"), ("Sessions_730D_MaxOpen", "long_history"),
    ("Orders_TargetLag28D", "long_history"), ("Orders_TargetLag364D", "long_history"),
    ("Conversion_28D_Median", "long_history"), ("Orders_90D_OpenTrend", "long_history"),
    ("HistHubDOWOrdersMean", "long_history"), ("HistHubDOWPromoCount", "long_history"),
    ("Horizon", "calendar_static"), ("Weekday", "calendar_static"), ("HubID", "calendar_static"),
    ("HubID_TE", "calendar_static"), ("Open_Tomorrow", "calendar_static"), ("Promo_DaysSinceActive", "calendar_static"),
    ("RegionalHoliday_StreakDay", "calendar_static"), ("CompetitorDistanceLog1p", "calendar_static"),
    ("LoyaltyTenureWeeks", "calendar_static"), ("DayBeforeClosed", "calendar_static"), ("ForecastWeek", "calendar_static"),
    ("HistoryRows", "calendar_static"), ("WasClosedForRenovation2014", "calendar_static"),
    ("SomethingNew", "unknown"),
])
def test_classify_column(name: str, label: str) -> None:
    assert classify_column(name) == label


def test_split_keeps_long_and_static_drops_short_and_does_not_mutate() -> None:
    cols = ["Horizon", "Orders_7D_MeanOpen", "Orders_90D_MeanOpen", "Orders_TargetLag7D", "Open_Yesterday"]
    snapshot = list(cols)
    kept, dropped = split_columns(cols)
    assert cols == snapshot
    assert kept == ["Horizon", "Orders_90D_MeanOpen", "Open_Yesterday"]
    assert dropped == ["Orders_7D_MeanOpen", "Orders_TargetLag7D"]


def test_unknown_policy() -> None:
    cols = ["Horizon", "SomethingNew"]
    assert split_columns(cols)[0] == cols  # default keeps unknown
    kept, dropped = split_columns(cols, unknown="drop")
    assert kept == ["Horizon"] and dropped == ["SomethingNew"]
    with pytest.raises(ValueError):
        split_columns(cols, unknown="bogus")


def test_real_header_snapshot_has_no_unknown_columns() -> None:
    names = [line for line in SNAPSHOT.read_text().splitlines() if line]
    assert len(names) > 300
    assert [c for c in names if classify_column(c) == "unknown"] == []
    kept, dropped = split_columns(names)
    assert len(kept) + len(dropped) == len(names)
    assert all(c in dropped for c in ("Orders_7D_MeanOpen", "Sessions_AtOrigin", "Orders_TargetLag14D"))
    assert all(c in kept for c in ("Orders_28D_MeanOpen", "Horizon", "HistHubDOWOrdersMean", "Orders_TargetLag28D"))
