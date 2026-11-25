"""Stage G3c: cs02 recipe + holiday-history / holiday-calendar / Sunday / promo-ratio features (schedule-aware, origin-anchored).

Usage: python g3c_runner.py <proj_root> <out_dir> --variants g3c_holiday,g3c_calendar,g3c_sunday,g3c_promo,g3c_all,g3c_all_plus_g3b --folds 2,3,4 --seed 20260807 --n-jobs 16
Same CLI and output schema as g3b_features/g3b_runner.py (experiments.csv + oof_<variant>_s<seed>.csv.gz). Fold 5 (summer-2014 diagnostic, origin
2014-06-19, fewer than 46,830 validation rows) works like any other fold: `--folds 2,3,4,5`. `--dedupe-scope` is accepted for CLI parity and ignored.
Needs orders_train.csv with RegionalHoliday / SchoolClosureFlag / PromoActive (and orders_test.csv for the schedule of the real test window when present).
Only the feature families needed by the requested variants are computed; the g3b columns are only built for g3c_all_plus_g3b.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

HERE = Path(__file__).resolve().parent
IMPROVE = HERE.parent
for _path in (IMPROVE / "g3b_features", IMPROVE / "g3_features", IMPROVE, HERE):      # HERE ends up first, so g3c_features.py wins
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import exp_runner as base  # noqa: E402  (harness: prepare_fold, fit_variant, write_record, VARIANTS)
import g3_runner  # noqa: E402  (read-only; reused: raw-folder discovery)
import g3b_features as g3b_feat  # noqa: E402
import g3b_runner  # noqa: E402  (read-only; reused: with_target_date, augment_ctx for the g3b columns)
from g3c_features import ALL_GROUPS, Calendar, add_features, add_features_multi, build_calendar, select_group_columns  # noqa: E402

CHAMPION = "cs02"
SCHEDULE_COLUMNS = ["IsOpen", "PromoActive", "RegionalHoliday", "SchoolClosureFlag"]
TRAIN_COLUMNS = ["HubID", "Date", "OrderVolume", *SCHEDULE_COLUMNS]
TEST_COLUMNS = ["HubID", "Date", *SCHEDULE_COLUMNS]
with_target_date = g3b_runner.with_target_date


@dataclass(frozen=True)
class VariantSpec:
    groups: tuple[str, ...]
    with_g3b: bool = False


VARIANT_SPECS: dict[str, VariantSpec] = {
    "g3c_holiday": VariantSpec(("holiday",)),
    "g3c_calendar": VariantSpec(("calendar",)),
    "g3c_sunday": VariantSpec(("sunday",)),
    "g3c_promo": VariantSpec(("promo",)),
    "g3c_all": VariantSpec(ALL_GROUPS),
    "g3c_all_plus_g3b": VariantSpec(ALL_GROUPS, with_g3b=True),
}


# ---------------------------------------------------------------- raw history + schedule
def load_history(root: Path, raw_dir: Path | None) -> pd.DataFrame:
    """orders_train.csv (labels + schedule) plus the schedule rows of orders_test.csv and the g3b hub grouping from hub_metadata.csv."""
    for folder in g3_runner._orders_dirs(root, raw_dir):
        orders = folder / "orders_train.csv"
        if not orders.exists():
            continue
        parts = [pd.read_csv(orders, parse_dates=["Date"], usecols=TRAIN_COLUMNS)]
        if (folder / "orders_test.csv").exists():
            parts.append(pd.read_csv(folder / "orders_test.csv", parse_dates=["Date"], usecols=TEST_COLUMNS))
        history = pd.concat(parts, ignore_index=True)
        if (folder / "hub_metadata.csv").exists():
            static = pd.read_csv(folder / "hub_metadata.csv")[["HubID", "HubFormat", "AssortmentTier"]]
            history = history.merge(static, on="HubID", how="left", validate="many_to_one")
        return history
    raise FileNotFoundError(f"no orders_train.csv under {[str(d) for d in g3_runner._orders_dirs(root, raw_dir)]}")


# ---------------------------------------------------------------- ctx handling (never mutates its input)
def augment_ctx(ctx: dict, history: pd.DataFrame | Calendar, groups: tuple[str, ...] = ALL_GROUPS, with_g3b: bool = False) -> dict:
    """New ctx whose train/hold/valid frames also carry the g3c columns of `groups` (and the g3b_all columns when `with_g3b`)."""
    cal = history if isinstance(history, Calendar) else build_calendar(history)
    start = g3b_runner.augment_ctx(ctx, cal.panel, g3b_feat.ALL_GROUPS) if with_g3b else ctx
    out = {**start}
    if groups:
        out["train"] = add_features_multi(with_target_date(start["train"]), cal, "_origin", groups)
        out["hold"] = add_features_multi(with_target_date(start["hold"]), cal, "_origin", groups)
        out["valid"] = add_features(start["valid"], cal, ctx["origin"], groups)
    return out


def variant_columns(base_columns: list[str], extra_columns: list[str], spec: VariantSpec) -> list[str]:
    cols = [*base_columns, *select_group_columns(extra_columns, spec.groups)]
    return cols + (g3b_feat.select_group_columns(extra_columns, g3b_feat.ALL_GROUPS) if spec.with_g3b else [])


def variant_ctx(aug: dict, name: str) -> dict:
    extra = [c for c in aug["valid"].columns if c.startswith(("g3c_", "g3b_"))]
    return {**aug, "columns": variant_columns(list(aug["columns"]), extra, VARIANT_SPECS[name]), "variant": name}


def needed_groups(names: list[str]) -> tuple[str, ...]:
    wanted = {g for n in names for g in VARIANT_SPECS[n].groups}
    return tuple(g for g in ALL_GROUPS if g in wanted)


# ---------------------------------------------------------------- CLI
def parse_args(argv: list[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("proj_root", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--variants", required=True, help="comma list of: " + ",".join(VARIANT_SPECS))
    ap.add_argument("--folds", default="2,3,4", help="comma list; 5 = summer-2014 diagnostic fold")
    ap.add_argument("--seed", type=int, default=20260807)
    ap.add_argument("--n-jobs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--rounds", type=int, default=1500)
    ap.add_argument("--dedupe-scope", choices=("vps", "all"), default="vps", help="accepted for CLI parity with g3_runner; unused")
    ap.add_argument("--raw-dir", type=Path, default=None, help="folder with orders_train.csv (+orders_test.csv, hub_metadata.csv); auto-detected if omitted")
    return ap.parse_args(argv)


def record_config(name: str) -> SimpleNamespace:
    cfg, spec = base.VARIANTS[CHAMPION], VARIANT_SPECS[name]
    return SimpleNamespace(**{**cfg.__dict__, "recipe": CHAMPION, "groups": list(spec.groups), "with_g3b": spec.with_g3b})


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.proj_root.resolve()
    sys.path.insert(0, str(root))
    names = [v.strip() for v in args.variants.split(",")]
    unknown = [v for v in names if v not in VARIANT_SPECS]
    if unknown:
        raise SystemExit(f"unknown variants: {unknown}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    groups, with_g3b = needed_groups(names), any(VARIANT_SPECS[n].with_g3b for n in names)
    cal = build_calendar(load_history(root, args.raw_dir))
    results: dict[str, list[dict]] = {n: [] for n in names}
    oofs: dict[str, list[pd.DataFrame]] = {n: [] for n in names}
    for fold in [int(x) for x in args.folds.split(",")]:
        t0 = time.perf_counter()
        aug = augment_ctx(base.prepare_fold(root, fold, args.seed), cal, groups, with_g3b)
        print(f"fold {fold}: ready in {time.perf_counter() - t0:.0f}s, train rows {len(aug['train'])}, valid rows {len(aug['valid'])}, "
              f"extra columns {sum(c.startswith(('g3c_', 'g3b_')) for c in aug['valid'].columns)}", flush=True)
        for n in names:
            oof, scores = base.fit_variant(variant_ctx(aug, n), base.VARIANTS[CHAMPION], args.seed, args.n_jobs, args.lr, args.rounds)
            oofs[n].append(oof)
            results[n].append(scores)
            print(n, "fold", fold, json.dumps(scores), flush=True)
        del aug
    for n in names:
        pd.concat(oofs[n], ignore_index=True).to_csv(args.out_dir / f"oof_{n}_s{args.seed}.csv.gz", index=False, compression="gzip")
        base.write_record(args.out_dir, n, record_config(n), args.seed, results[n], args)
        print("DONE", n, round(sum(r["rmsle_all"] for r in results[n]) / len(results[n]), 5), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
