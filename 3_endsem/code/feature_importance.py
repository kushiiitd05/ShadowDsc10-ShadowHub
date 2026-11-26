"""Which feature groups the frozen LightGBM boosters actually use (gain importance, summed per group, mean of 3 seeds).

Run from the 3_endsem folder (needs lightgbm):
    python code/feature_importance.py      # writes results/feature_group_importance.csv and results/feature_importance_top30.csv

Gain importance is the total loss reduction of the splits on a column. It shows what the trees rely on, not causation, and
correlated columns (for example rolling windows of the same series) share the credit, so read groups rather than single columns.
"""
from __future__ import annotations

import gzip
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (20260807, 20260808, 20260809)
# first matching rule wins; order matters (e.g. promotion run flags before generic gap flags)
GROUPS = (
    ("Hub x weekday x promotion profile", r"^HistHub"),
    ("Hub identity (cross-fitted encoding, id)", r"^HubID"),
    ("Promotion calendar (known ahead)", r"^Promo"),
    ("Open/closed calendar (known ahead)", r"^IsOpen$|^Open_|^DayAfterClosed$|^DayBeforeClosed$"),
    ("Holidays and school closures (known ahead)", r"^RegionalHoliday|^SchoolClosure"),
    ("History coverage and data-gap flags", r"^History|_(MissingCount|PresentCount|OpenCount)$"),
    ("Sessions, conversion, volume per session", r"^Sessions_|^Conversion_|^VolumePerSession_"),
    ("Same-day lags and value at origin", r"^Orders_TargetLag|^Orders_AtOrigin|^TargetLag"),
    ("Order rolling windows (7-730 days)", r"^Orders_\d+D_"),
    ("Calendar and horizon", r"^(Weekday|DayOfMonth|DayOfYear|Month|Year|ISOWeek|IsWeekend|IsMonthEnd|IsMonthStart|"
                             r"DaysSinceMonthStart|DaysToMonthEnd|ForecastWeek|Horizon)$"),
    ("Hub metadata, competition, loyalty", r"^(HubFormat|AssortmentTier|Competitor|Loyalty|WasClosedForRenovation|DaysSinceRenovation)"),
)


def group_of(column: str) -> str:
    for name, pattern in GROUPS:
        if re.search(pattern, column):
            return name
    return "other"


def booster_gain(path: Path) -> pd.Series:
    import lightgbm as lgb
    with gzip.open(path, "rt") as f:
        booster = lgb.Booster(model_str=f.read())
    return pd.Series(booster.feature_importance("gain"), index=booster.feature_name(), dtype=float)


def run(root: Path = ROOT) -> tuple[pd.DataFrame, pd.DataFrame]:
    shares = pd.concat([(g := booster_gain(root / "final" / "models" / f"lightgbm_s{s}.txt.gz")) / g.sum() for s in SEEDS], axis=1)
    shares.columns = [f"s{s}" for s in SEEDS]
    per_col = pd.DataFrame({"column": shares.index, "group": [group_of(c) for c in shares.index],
                            "gain_share": shares.mean(axis=1).to_numpy(), "seed_min": shares.min(axis=1).to_numpy(),
                            "seed_max": shares.max(axis=1).to_numpy()})
    grouped = (per_col.groupby("group").agg(columns=("column", "size"), gain_share=("gain_share", "sum"))
               .sort_values("gain_share", ascending=False).reset_index())
    seed_group = shares.groupby(per_col.set_index("column")["group"]).sum()
    grouped["seed_min"] = grouped["group"].map(seed_group.min(axis=1))
    grouped["seed_max"] = grouped["group"].map(seed_group.max(axis=1))
    return grouped, per_col.sort_values("gain_share", ascending=False).head(30).reset_index(drop=True)


if __name__ == "__main__":
    grouped, top = run()
    if (grouped["group"] == "other").any():
        raise SystemExit(f"unmapped columns: {grouped}")
    grouped.round(4).to_csv(ROOT / "results" / "feature_group_importance.csv", index=False)
    top.round(5).to_csv(ROOT / "results" / "feature_importance_top30.csv", index=False)
    print(grouped.round(4).to_string(index=False))
    print(top.head(12).round(4).to_string(index=False))
