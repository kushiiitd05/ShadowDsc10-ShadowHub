"""Stage G3 'features': cs02 recipe + extra origin-anchored features and/or exact-duplicate column removal.

Usage: python g3_runner.py <proj_root> <out_dir> --variants feat_trend,feat_all --folds 2,3,4 --seed 20260807 --n-jobs 16
Same CLI and output schema as exp_runner.py (experiments.csv + oof_<variant>_s<seed>.csv.gz); extra option --raw-dir.
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
for _path in (HERE, HERE.parent):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import exp_runner as base  # noqa: E402  (harness: prepare_fold, fit_variant, write_record, VARIANTS)
from extra_features import ALL_GROUPS, add_features, add_features_multi, confirm_duplicates, find_duplicate_columns, select_group_columns  # noqa: E402
from feature_panel import Panel, build_panel  # noqa: E402

CHAMPION = "cs02"
HISTORY_COLUMNS = ["HubID", "Date", "OrderVolume", "IsOpen", "PromoActive", "HubFormat", "AssortmentTier"]


@dataclass(frozen=True)
class VariantSpec:
    groups: tuple[str, ...]
    dedupe: bool = False


VARIANT_SPECS: dict[str, VariantSpec] = {
    "feat_trend": VariantSpec(("trend",)),
    "feat_weekday_promo": VariantSpec(("weekday_promo",)),
    "feat_streaks": VariantSpec(("streaks",)),
    "feat_all": VariantSpec(ALL_GROUPS),
    "dedupe_only": VariantSpec((), dedupe=True),
    "dedupe_plus_all": VariantSpec(ALL_GROUPS, dedupe=True),
}


# ---------------------------------------------------------------- raw history
def _orders_dirs(root: Path, raw_dir: Path | None) -> list[Path]:
    dirs = [raw_dir] if raw_dir else []
    dirs += sorted(p for p in root.glob("dat*set*") if p.is_dir()) + [root / "data" / "raw"]   # e.g. <root>/dat*set*/orders_train.csv
    if len(root.parents) > 1:
        dirs.append(root.parents[1] / "1_midterm" / "data" / "raw")
    return dirs


def load_history(root: Path, raw_dir: Path | None) -> pd.DataFrame:
    """Raw daily history (orders_train.csv + hub_metadata.csv), else the processed base_train table."""
    for folder in _orders_dirs(root, raw_dir):
        orders = folder / "orders_train.csv"
        if orders.exists():
            history = pd.read_csv(orders, parse_dates=["Date"])
            meta = folder / "hub_metadata.csv"
            if meta.exists():
                static = pd.read_csv(meta)[["HubID", "HubFormat", "AssortmentTier"]]
                history = history.merge(static, on="HubID", how="left", validate="many_to_one")
            return history[[c for c in HISTORY_COLUMNS if c in history.columns]]
    processed = root / "data" / "processed" / "base_train.csv.gz"
    if processed.exists():
        usecols = lambda c: c in HISTORY_COLUMNS  # noqa: E731
        return pd.read_csv(processed, parse_dates=["Date"], usecols=usecols)
    raise FileNotFoundError(f"no orders_train.csv under {[str(d) for d in _orders_dirs(root, raw_dir)]} and no {processed}")


# ---------------------------------------------------------------- ctx handling (never mutates its input)
def augment_ctx(ctx: dict, history: pd.DataFrame | Panel) -> dict:
    """New ctx whose train/hold/valid frames also carry every extra column (all groups)."""
    panel = history if isinstance(history, Panel) else build_panel(history)
    return {**ctx,
            "train": add_features_multi(ctx["train"], panel, "_origin"),
            "hold": add_features_multi(ctx["hold"], panel, "_origin"),
            "valid": add_features(ctx["valid"], panel, ctx["origin"])}


def duplicates_for_ctx(ctx: dict) -> dict[str, str]:
    """Exact duplicate model columns (dropped -> kept) identical in train, holdout and validation frames."""
    pairs = find_duplicate_columns(ctx["train"], columns=list(ctx["columns"]))
    return confirm_duplicates(pairs, ctx["hold"], ctx["valid"])


def filter_scope(duplicates: dict[str, str], scope: str) -> dict[str, str]:
    """'vps' = the structural twins VolumePerSession_* (== Conversion_*, 24 columns); 'all' = every confirmed duplicate."""
    if scope == "all":
        return dict(duplicates)
    if scope == "vps":
        return {d: k for d, k in duplicates.items() if d.startswith("VolumePerSession_")}
    raise ValueError(f"unknown dedupe scope: {scope}")


def variant_columns(base_columns: list[str], extra_columns: list[str], spec: VariantSpec, duplicates: dict[str, str]) -> list[str]:
    kept = [c for c in base_columns if not (spec.dedupe and c in duplicates)]
    return kept + select_group_columns(extra_columns, spec.groups)


def variant_ctx(aug: dict, name: str, duplicates: dict[str, str]) -> dict:
    extra = [c for c in aug["valid"].columns if c.startswith("g3_")]
    return {**aug, "columns": variant_columns(list(aug["columns"]), extra, VARIANT_SPECS[name], duplicates), "variant": name}


# ---------------------------------------------------------------- CLI
def parse_args(argv: list[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("proj_root", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--variants", required=True, help="comma list of: " + ",".join(VARIANT_SPECS))
    ap.add_argument("--folds", default="2,3,4")
    ap.add_argument("--seed", type=int, default=20260807)
    ap.add_argument("--n-jobs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--rounds", type=int, default=1500)
    ap.add_argument("--dedupe-scope", choices=("vps", "all"), default="vps",
                    help="vps: drop the 24 VolumePerSession_* twins (default); all: drop every exact duplicate found in train+holdout+valid")
    ap.add_argument("--raw-dir", type=Path, default=None, help="folder with orders_train.csv (+hub_metadata.csv); auto-detected if omitted")
    return ap.parse_args(argv)


def record_config(name: str, scope: str) -> SimpleNamespace:
    cfg, spec = base.VARIANTS[CHAMPION], VARIANT_SPECS[name]
    return SimpleNamespace(**{**cfg.__dict__, "recipe": CHAMPION, "groups": list(spec.groups), "dedupe": spec.dedupe,
                              "dedupe_scope": scope if spec.dedupe else None})


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.proj_root.resolve()
    sys.path.insert(0, str(root))
    names = [v.strip() for v in args.variants.split(",")]
    unknown = [v for v in names if v not in VARIANT_SPECS]
    if unknown:
        raise SystemExit(f"unknown variants: {unknown}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    panel = build_panel(load_history(root, args.raw_dir))
    results: dict[str, list[dict]] = {n: [] for n in names}
    oofs: dict[str, list[pd.DataFrame]] = {n: [] for n in names}
    for fold in [int(x) for x in args.folds.split(",")]:
        t0 = time.perf_counter()
        aug = augment_ctx(base.prepare_fold(root, fold, args.seed), panel)
        dups = filter_scope(duplicates_for_ctx(aug), args.dedupe_scope)
        (args.out_dir / f"duplicates_fold{fold}.json").write_text(json.dumps(dups, indent=1, sort_keys=True))
        print(f"fold {fold}: ready in {time.perf_counter() - t0:.0f}s, train rows {len(aug['train'])}, dropped-if-dedupe columns {len(dups)}", flush=True)
        for n in names:
            oof, scores = base.fit_variant(variant_ctx(aug, n, dups), base.VARIANTS[CHAMPION], args.seed, args.n_jobs, args.lr, args.rounds)
            oofs[n].append(oof)
            results[n].append(scores)
            print(n, "fold", fold, json.dumps(scores), flush=True)
        del aug
    for n in names:
        pd.concat(oofs[n], ignore_index=True).to_csv(args.out_dir / f"oof_{n}_s{args.seed}.csv.gz", index=False, compression="gzip")
        base.write_record(args.out_dir, n, record_config(n, args.dedupe_scope), args.seed, results[n], args)
        print("DONE", n, round(sum(r["rmsle_all"] for r in results[n]) / len(results[n]), 5), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
