"""Stage G3b: cs02 recipe + gap-aware / year-ago / peer-correlation origin-anchored features.

Usage: python g3b_runner.py <proj_root> <out_dir> --variants g3b_gap,g3b_yearago,g3b_peer,g3b_all,g3b_all_plus_g3 --folds 2,3,4 --seed 20260807 --n-jobs 16
Same CLI and output schema as g3_features/g3_runner.py (experiments.csv + oof_<variant>_s<seed>.csv.gz). `--dedupe-scope` is accepted for
CLI compatibility and ignored (no dedupe variants here). Only the feature families needed by the requested variants are computed, and
the g3 columns are only built when g3b_all_plus_g3 is requested.
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
for _path in (IMPROVE / "g3_features", IMPROVE, HERE):      # HERE ends up first, so g3b_features.py beats the folder of the same name
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import exp_runner as base  # noqa: E402  (harness: prepare_fold, fit_variant, write_record, VARIANTS)
import g3_runner  # noqa: E402  (read-only; reused: load_history, augment_ctx for the g3 columns)
from extra_features import ALL_GROUPS as G3_GROUPS, select_group_columns as g3_select_columns  # noqa: E402
from feature_panel import Panel, build_panel  # noqa: E402
from g3b_features import ALL_GROUPS, add_features, add_features_multi, select_group_columns  # noqa: E402

CHAMPION = "cs02"
load_history = g3_runner.load_history


@dataclass(frozen=True)
class VariantSpec:
    groups: tuple[str, ...]
    with_g3: bool = False


VARIANT_SPECS: dict[str, VariantSpec] = {
    "g3b_gap": VariantSpec(("gap_aware",)),
    "g3b_yearago": VariantSpec(("year_ago",)),
    "g3b_peer": VariantSpec(("peer_corr",)),
    "g3b_all": VariantSpec(ALL_GROUPS),
    "g3b_all_plus_g3": VariantSpec(ALL_GROUPS, with_g3=True),
}


# ---------------------------------------------------------------- ctx handling (never mutates its input)
def with_target_date(frame: pd.DataFrame) -> pd.DataFrame:
    """Training and holdout frames do not carry `Date`; it is the origin plus Horizon days (verified on the real fold files)."""
    if "Date" in frame.columns:
        return frame
    return frame.assign(Date=frame["_origin"] + pd.to_timedelta(frame["Horizon"], unit="D"))


def augment_ctx(ctx: dict, history: pd.DataFrame | Panel, groups: tuple[str, ...] = ALL_GROUPS, with_g3: bool = False) -> dict:
    """New ctx whose train/hold/valid frames also carry the g3b columns of `groups` (and the g3 columns when `with_g3`)."""
    panel = history if isinstance(history, Panel) else build_panel(history)
    start = g3_runner.augment_ctx(ctx, panel) if with_g3 else ctx
    out = {**start}
    if groups:
        out["train"] = add_features_multi(with_target_date(start["train"]), panel, "_origin", groups)
        out["hold"] = add_features_multi(with_target_date(start["hold"]), panel, "_origin", groups)
        out["valid"] = add_features(start["valid"], panel, ctx["origin"], groups)
    return out


def variant_columns(base_columns: list[str], extra_columns: list[str], spec: VariantSpec) -> list[str]:
    cols = [*base_columns, *select_group_columns(extra_columns, spec.groups)]
    return cols + (g3_select_columns(extra_columns, G3_GROUPS) if spec.with_g3 else [])


def variant_ctx(aug: dict, name: str) -> dict:
    extra = [c for c in aug["valid"].columns if c.startswith(("g3b_", "g3_"))]
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
    ap.add_argument("--folds", default="2,3,4")
    ap.add_argument("--seed", type=int, default=20260807)
    ap.add_argument("--n-jobs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--rounds", type=int, default=1500)
    ap.add_argument("--dedupe-scope", choices=("vps", "all"), default="vps", help="accepted for CLI parity with g3_runner; unused")
    ap.add_argument("--raw-dir", type=Path, default=None, help="folder with orders_train.csv (+hub_metadata.csv); auto-detected if omitted")
    return ap.parse_args(argv)


def record_config(name: str) -> SimpleNamespace:
    cfg, spec = base.VARIANTS[CHAMPION], VARIANT_SPECS[name]
    return SimpleNamespace(**{**cfg.__dict__, "recipe": CHAMPION, "groups": list(spec.groups), "with_g3": spec.with_g3})


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.proj_root.resolve()
    sys.path.insert(0, str(root))
    names = [v.strip() for v in args.variants.split(",")]
    unknown = [v for v in names if v not in VARIANT_SPECS]
    if unknown:
        raise SystemExit(f"unknown variants: {unknown}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    groups, with_g3 = needed_groups(names), any(VARIANT_SPECS[n].with_g3 for n in names)
    panel = build_panel(load_history(root, args.raw_dir))
    results: dict[str, list[dict]] = {n: [] for n in names}
    oofs: dict[str, list[pd.DataFrame]] = {n: [] for n in names}
    for fold in [int(x) for x in args.folds.split(",")]:
        t0 = time.perf_counter()
        aug = augment_ctx(base.prepare_fold(root, fold, args.seed), panel, groups, with_g3)
        print(f"fold {fold}: ready in {time.perf_counter() - t0:.0f}s, train rows {len(aug['train'])}, "
              f"extra columns {sum(c.startswith(('g3b_', 'g3_')) for c in aug['valid'].columns)}", flush=True)
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
