"""Name-based classification of forecast-table feature columns by the history window they encode.

Pure functions, no I/O. Used by the 'long_window_only' variant: for far-horizon rows (22-42 days after
the origin) a feature built from the last 7 days, the origin day itself or a 7/14 day target lag is stale,
so only windows >= 28 days and calendar / hub-static / known-future-schedule columns are kept.
"""
from __future__ import annotations

import re
from typing import Literal

MIN_LONG_WINDOW_DAYS = 28
Label = Literal["short_history", "long_history", "calendar_static", "unknown"]

_WINDOW_PATTERNS = (
    re.compile(r"_(\d+)D(?:_|$)"),     # Orders_28D_MeanOpen, Conversion_90D_Count
    re.compile(r"Lag(\d+)D"),          # Orders_TargetLag14D, TargetLag14D_Available
    re.compile(r"AtOriginMinus(\d+)D"),  # Orders_AtOriginMinus7D
)
_STATIC_EXACT = frozenset({
    "HubID", "HubID_TE", "HubID_TargetEncoded", "Weekday", "IsOpen", "PromoActive", "RegionalHoliday", "SchoolClosureFlag",
    "Year", "Month", "DayOfMonth", "DayOfYear", "ISOWeek", "IsWeekend", "IsMonthStart", "IsMonthEnd",
    "DaysSinceMonthStart", "DaysToMonthEnd", "Horizon", "ForecastWeek", "HubFormat", "AssortmentTier",
    "DayBeforeClosed", "DayAfterClosed", "WasClosedForRenovation2014", "DaysSinceRenovationReopen",
})
_STATIC_PREFIXES = ("Competitor", "Loyalty", "Open_", "Promo_", "SchoolClosure_", "RegionalHoliday_", "History")


def window_days(name: str) -> int | None:
    """History window / lag (days) encoded in a column name; 0 for the origin-day value; None if the name has none."""
    if name.endswith("_AtOrigin"):
        return 0
    for pattern in _WINDOW_PATTERNS:
        match = pattern.search(name)
        if match:
            return int(match.group(1))
    return None


def classify_column(name: str) -> Label:
    """short_history (<28d window, origin-day value, 7/14d lag) | long_history | calendar_static | unknown."""
    if "AtOrigin" in name:  # single-day values at/near the origin, whatever the offset
        return "short_history"
    days = window_days(name)
    if days is not None:
        return "short_history" if days < MIN_LONG_WINDOW_DAYS else "long_history"
    if name.startswith("HistHub"):  # all-history hub x weekday / promo means
        return "long_history"
    if name in _STATIC_EXACT or name.startswith(_STATIC_PREFIXES):
        return "calendar_static"
    return "unknown"


def split_columns(columns: list[str], unknown: str = "keep") -> tuple[list[str], list[str]]:
    """Return (kept, dropped) as new lists, preserving order. Unknown names follow `unknown` ('keep' | 'drop')."""
    if unknown not in ("keep", "drop"):
        raise ValueError(f"unknown must be 'keep' or 'drop', got {unknown!r}")
    drop_labels = {"short_history"} | ({"unknown"} if unknown == "drop" else set())
    kept = [c for c in columns if classify_column(c) not in drop_labels]
    dropped = [c for c in columns if classify_column(c) in drop_labels]
    return kept, dropped


def describe_split(columns: list[str], unknown: str = "keep") -> str:
    """Human-readable kept/dropped listing for the run log."""
    kept, dropped = split_columns(columns, unknown)
    unknowns = [c for c in columns if classify_column(c) == "unknown"]
    lines = [f"long-horizon columns: kept {len(kept)}, dropped {len(dropped)}, unknown {len(unknowns)} (policy: {unknown})",
             "DROPPED: " + ", ".join(dropped), "KEPT: " + ", ".join(kept)]
    if unknowns:
        lines.append("UNKNOWN (unclassified): " + ", ".join(unknowns))
    return "\n".join(lines)
