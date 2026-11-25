"""G9b: permutation importance of the cs02 recipe by feature group, then a pruning test.

Usage:
  python perm_prune.py <proj_root> <out_dir> --stage perm  --perm-folds 3,4 --repeats 3 --seed 20260807 --n-jobs 24
  python perm_prune.py <proj_root> <out_dir> --stage prune --folds 1,2,3,4,5 --seeds 20260807,20260808,20260809 --n-jobs 24

perm : fits cs02 through the harness (exp_runner.fit_variant, lr 0.05, early stopping as in the harness), then shuffles each feature
       group's columns jointly on the fold's validation rows and records the rise in log RMSE (rmsle_all) over 3 repeats. Two levels:
       coarse groups (lags, rolling, calendar, promo, holiday, hub_stats, encoding) and finer subgroups.
prune: drops the subgroups whose cost is <= 0 on BOTH perm folds (or an explicit --drop list), refits on the same rows next to cs02, and
       writes per (seed, fold) scores plus the gate verdict.
Fold 5 needs ALLOW_INCOMPLETE_BLOCK=1 in the environment (set by run_g9.sh).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import zlib
from dataclasses import replace
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

IMPROVE = Path(__file__).resolve().parent.parent
if str(IMPROVE) not in sys.path:
    sys.path.insert(0, str(IMPROVE))

CHAMPION = "cs02"
NOISE_FLOOR = 0.0012
GATE_GAIN = 0.0024
DEV_FOLDS = (2, 3, 4)

# ---------------------------------------------------------------- group assignment (pure)
SUBGROUP_TO_GROUP: dict[str, str] = {
    "lags_orders": "lags", "lags_sessions": "lags", "lags_avail": "lags",
    "roll_orders_level": "rolling", "roll_orders_count": "rolling", "roll_sessions_level": "rolling",
    "roll_sessions_count": "rolling", "roll_ratio": "rolling",
    "cal_date": "calendar", "cal_open": "calendar",
    "promo": "promo",
    "holiday": "holiday", "school": "holiday",
    "hub_static": "hub_stats", "hub_hist": "hub_stats",
    "encoding": "encoding",
}
GROUP_ORDER = ("lags", "rolling", "calendar", "promo", "holiday", "hub_stats", "encoding")
_LAG = re.compile(r"^(Orders|Sessions)_(TargetLag\d+D|AtOrigin(Minus\d+D)?)$")
_LAG_AVAIL = re.compile(r"^TargetLag\d+D_Available$")
_ROLL = re.compile(r"^(Orders|Sessions|Conversion|VolumePerSession)_\d+D_")
_ROLL_COUNT = re.compile(r"_(PresentCount|MissingCount|OpenCount)$")
_DATE_PARTS = {"Weekday", "Year", "Month", "DayOfMonth", "DayOfYear", "ISOWeek", "IsWeekend", "IsMonthStart", "IsMonthEnd",
               "DaysSinceMonthStart", "DaysToMonthEnd", "Horizon", "ForecastWeek"}
_OPEN = {"IsOpen", "DayBeforeClosed", "DayAfterClosed"}
_TIME_COUNTERS = {"HistoryRows", "HistoryDaysSinceLastObservation", "HistoryMissingCalendarDays", "DaysSinceRenovationReopen",
                  "CompetitorTenureMonths", "LoyaltyTenureWeeks", "LoyaltyIntervalMonth"}


def assign_subgroup(col: str) -> str:
    """Finer group of one feature column; 'other' when no rule matches (the real run asserts that never happens)."""
    if col in ("HubID", "HubID_TE"):
        return "encoding"
    if _LAG.match(col):
        return "lags_orders" if col.startswith("Orders_") else "lags_sessions"
    if _LAG_AVAIL.match(col):
        return "lags_avail"
    if _ROLL.match(col):
        if col.startswith(("Conversion_", "VolumePerSession_")):
            return "roll_ratio"
        side = "orders" if col.startswith("Orders_") else "sessions"
        return f"roll_{side}_{'count' if _ROLL_COUNT.search(col) else 'level'}"
    if col == "PromoActive" or col.startswith("Promo_"):
        return "promo"
    if col.startswith("RegionalHoliday"):
        return "holiday"
    if col.startswith("SchoolClosure"):
        return "school"
    if col in _DATE_PARTS:
        return "cal_date"
    if col in _OPEN or col.startswith("Open_"):
        return "cal_open"
    if col in ("HubFormat", "AssortmentTier") or col.startswith(("Competitor", "Loyalty")):
        return "hub_static"
    if col.startswith(("HistHub", "History")) or col in ("WasClosedForRenovation2014", "DaysSinceRenovationReopen"):
        return "hub_hist"
    return "other"


def assign_group(col: str) -> str:
    """Coarse group (lags, rolling, calendar, promo, holiday, hub_stats, encoding) of one feature column."""
    return SUBGROUP_TO_GROUP.get(assign_subgroup(col), "other")


def is_calendar_like(col: str) -> bool:
    """Columns that move with the calendar and so separate future from past rows trivially (date parts, open schedule, time counters)."""
    return assign_group(col) == "calendar" or col in _TIME_COUNTERS


def columns_by(cols: list[str], level: str = "group") -> dict[str, list[str]]:
    """{name: columns} for level 'group' or 'subgroup', preserving input order inside each name."""
    fn = assign_group if level == "group" else assign_subgroup
    out: dict[str, list[str]] = {}
    for c in cols:
        out.setdefault(fn(c), []).append(c)
    return out


# ---------------------------------------------------------------- permutation + scoring (pure)
def permute_columns(frame: pd.DataFrame, cols: list[str], rng: np.random.Generator) -> pd.DataFrame:
    """Copy of `frame` whose `cols` are shuffled with ONE shared row permutation (joint shuffle keeps within-group structure)."""
    out = frame.copy()
    idx = rng.permutation(len(frame))
    for c in cols:
        out[c] = frame[c].iloc[idx].array
    return out


def gate_verdict(base: pd.DataFrame, cand: pd.DataFrame, dev_folds: tuple[int, ...] = DEV_FOLDS, threshold: float = GATE_GAIN) -> dict:
    """Gate for a change. Inputs: columns seed, fold, rmsle_all, rmsle_excl_open0 (one row per seed x fold).

    gain = base - cand (positive = candidate better). Passes when the seed-mean gain >= threshold on >= 2 dev folds, the sign of the
    dev-fold mean gain is the same on every seed (and positive), and the same two conditions hold excluding open-zero rows.
    """
    def one(metric: str) -> dict:
        m = base.merge(cand, on=["seed", "fold"], suffixes=("_b", "_c"))
        m["gain"] = m[f"{metric}_b"] - m[f"{metric}_c"]
        dev = m[m["fold"].isin(dev_folds)]
        per_fold = dev.groupby("fold")["gain"].mean()
        per_seed = dev.groupby("seed")["gain"].mean()
        folds_ok = int((per_fold >= threshold).sum())
        same_sign = bool((per_seed > 0).all() or (per_seed < 0).all())
        return {"per_fold_gain": per_fold.round(5).to_dict(), "per_seed_gain": per_seed.round(5).to_dict(), "folds_over_threshold": folds_ok,
                "same_sign_all_seeds": same_sign, "positive": bool((per_seed > 0).all()),
                "passes": bool(folds_ok >= 2 and same_sign and (per_seed > 0).all())}

    full, excl = one("rmsle_all"), one("rmsle_excl_open0")
    return {"all": full, "excl_open0": excl, "gate_pass": bool(full["passes"] and excl["passes"])}


def select_prune_groups(ranking: pd.DataFrame, cost_cols: list[str], limit: float = 0.0) -> list[str]:
    """Names whose permutation cost is <= `limit` on EVERY fold column."""
    keep = ranking[(ranking[cost_cols] <= limit).all(axis=1)]
    return list(keep["name"])


# ---------------------------------------------------------------- fit + permutation importance (needs lightgbm / the server data)
def fit_capture(ctx: dict, cfg, seed: int, n_jobs: int, lr: float, rounds: int):
    """Run the harness fit and also return the fitted LGBMRegressor (captured through a factory patch, so the fit is the harness fit)."""
    import lightgbm as lgb
    import exp_runner as base

    captured: list = []
    original = lgb.LGBMRegressor

    def factory(**kwargs):
        model = original(**kwargs)
        captured.append(model)
        return model

    with mock.patch.object(lgb, "LGBMRegressor", factory):
        oof, scores = base.fit_variant(ctx, cfg, seed, n_jobs, lr, rounds)
    return captured[-1], oof, scores


def score_valid(raw: np.ndarray, valid: pd.DataFrame, open_mask: np.ndarray, target: str = "log") -> dict:
    from helpers import decode_prediction, fold_scores

    pred = np.zeros(len(valid))
    pred[open_mask] = decode_prediction(raw, valid.loc[open_mask, "_level"].to_numpy(), target)
    return fold_scores(valid["ActualOrderVolume"].to_numpy(float), pred, valid["IsOpen"].to_numpy())


def permutation_costs(model, x_valid: pd.DataFrame, valid: pd.DataFrame, open_mask: np.ndarray, groups: dict[str, list[str]],
                      repeats: int, seed: int) -> tuple[float, list[dict]]:
    base_raw = model.predict(x_valid, raw_score=True)
    base = score_valid(base_raw, valid, open_mask)["rmsle_all"]
    rows = []
    for name, cols in groups.items():
        costs = []
        for r in range(repeats):
            rng = np.random.default_rng([seed, r, zlib.crc32(name.encode()) % 100_000])
            raw = model.predict(permute_columns(x_valid, cols, rng), raw_score=True)
            costs.append(score_valid(raw, valid, open_mask)["rmsle_all"] - base)
        rows.append({"name": name, "n_cols": len(cols), "cost_mean": float(np.mean(costs)), "cost_std": float(np.std(costs)),
                     "costs": json.dumps([round(c, 6) for c in costs])})
    return base, rows


def run_perm(root: Path, out_dir: Path, folds: list[int], repeats: int, seed: int, n_jobs: int, lr: float, rounds: int) -> None:
    import exp_runner as base_mod
    from src.model_pipeline import CATEGORICAL_COLUMNS

    cfg = base_mod.VARIANTS[CHAMPION]
    for fold in folds:
        t0 = time.perf_counter()
        ctx = base_mod.prepare_fold(root, fold, seed)
        model, _, scores = fit_capture(ctx, cfg, seed, n_jobs, lr, rounds)
        feats = list(model.feature_name_)
        valid = ctx["valid"]
        open_mask = valid["IsOpen"].eq(1).to_numpy()
        x_valid = base_mod.prep(valid.loc[open_mask], feats, set(CATEGORICAL_COLUMNS))
        other = [c for c in feats if assign_subgroup(c) == "other"]
        if other:
            raise SystemExit(f"unassigned feature columns: {other}")
        print(f"fold {fold}: fitted in {time.perf_counter() - t0:.0f}s, best_iter {scores['best_iteration']}, rmsle_all {scores['rmsle_all']:.5f}", flush=True)
        for level in ("group", "subgroup"):
            base, rows = permutation_costs(model, x_valid, valid, open_mask, columns_by(feats, level), repeats, seed)
            if abs(base - scores["rmsle_all"]) > 1e-6:
                raise SystemExit(f"re-scored baseline {base} differs from harness {scores['rmsle_all']}")
            frame = pd.DataFrame(rows).assign(level=level, fold=fold, seed=seed, base_rmsle_all=base)
            frame.sort_values("cost_mean", ascending=False).to_csv(out_dir / f"perm_{level}_fold{fold}.csv", index=False)
            print(frame.sort_values("cost_mean", ascending=False)[["name", "n_cols", "cost_mean", "cost_std"]].to_string(index=False), flush=True)
        del ctx, model


def build_ranking(out_dir: Path, folds: list[int], level: str) -> pd.DataFrame:
    """Merge perm_<level>_fold*.csv into one table: name, n_cols, cost_f<k>..., cost_min, cost_max, mean_cost."""
    merged = None
    for f in folds:
        part = pd.read_csv(out_dir / f"perm_{level}_fold{f}.csv")[["name", "n_cols", "cost_mean"]].rename(columns={"cost_mean": f"cost_f{f}"})
        merged = part if merged is None else merged.merge(part[["name", f"cost_f{f}"]], on="name")
    cost_cols = [f"cost_f{f}" for f in folds]
    merged["mean_cost"] = merged[cost_cols].mean(axis=1)
    return merged.sort_values("mean_cost", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------- pruning refit
def run_prune(root: Path, out_dir: Path, folds: list[int], seeds: list[int], drop_names: list[str], drop_cols_override: list[str] | None,
              n_jobs: int, lr: float, rounds: int, tag: str) -> None:
    import exp_runner as base_mod

    cfg = base_mod.VARIANTS[CHAMPION]
    records = []
    for seed in seeds:
        for fold in folds:
            ctx = base_mod.prepare_fold(root, fold, seed)
            cols = list(ctx["columns"]) + (["HubID_TE"] if cfg.te else [])
            drop = set(drop_cols_override) if drop_cols_override is not None else {c for c in cols if assign_subgroup(c) in set(drop_names)}
            kept = [c for c in ctx["columns"] if c not in drop]
            cfg_pruned = replace(cfg, te=cfg.te and "HubID_TE" not in drop)
            for variant, c_cfg, c_cols in (("cs02", cfg, list(ctx["columns"])), (f"pruned_{tag}", cfg_pruned, kept)):
                _, scores = base_mod.fit_variant({**ctx, "columns": c_cols}, c_cfg, seed, n_jobs, lr, rounds)
                row = {"variant": variant, "seed": seed, "fold": fold, "n_features": len(c_cols) + int(c_cfg.te), **scores}
                records.append(row)
                print(json.dumps(row), flush=True)
                pd.DataFrame(records).to_csv(out_dir / f"prune_results_{tag}.csv", index=False)
            del ctx
    frame = pd.DataFrame(records)
    if not frame["fold"].isin(DEV_FOLDS).any():  # diagnostic-only run (e.g. fold 5): scores are in the CSV, no gate
        return
    verdict = gate_verdict(frame[frame.variant == "cs02"], frame[frame.variant == f"pruned_{tag}"])
    verdict["dropped_names"] = drop_names
    verdict["dropped_columns"] = sorted(drop)
    (out_dir / f"prune_verdict_{tag}.json").write_text(json.dumps(verdict, indent=2, default=str))
    print(json.dumps(verdict, default=str), flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("proj_root", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--stage", choices=("perm", "prune"), required=True)
    ap.add_argument("--perm-folds", default="3,4")
    ap.add_argument("--folds", default="1,2,3,4,5")
    ap.add_argument("--seed", type=int, default=20260807)
    ap.add_argument("--seeds", default="20260807,20260808,20260809")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--n-jobs", type=int, default=24)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--rounds", type=int, default=1500)
    ap.add_argument("--drop", default=None, help="explicit comma list of subgroup names to drop (default: those with cost <= 0 on all perm folds)")
    ap.add_argument("--tag", default="p1")
    args = ap.parse_args(argv)
    root = args.proj_root.resolve()
    sys.path.insert(0, str(root))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    perm_folds = [int(x) for x in args.perm_folds.split(",")]
    if args.stage == "perm":
        run_perm(root, args.out_dir, perm_folds, args.repeats, args.seed, args.n_jobs, args.lr, args.rounds)
        for level in ("group", "subgroup"):
            build_ranking(args.out_dir, perm_folds, level).to_csv(args.out_dir / f"perm_ranking_{level}.csv", index=False)
        return 0
    ranking = build_ranking(args.out_dir, perm_folds, "subgroup")
    cost_cols = [f"cost_f{f}" for f in perm_folds]
    names = [n.strip() for n in args.drop.split(",")] if args.drop else select_prune_groups(ranking, cost_cols)
    print(f"prune subgroups ({args.tag}): {names}", flush=True)
    if not names:
        print("no subgroup has cost <= 0 on all perm folds; nothing to prune (use --drop for an exploratory set)", flush=True)
        (args.out_dir / f"prune_verdict_{args.tag}.json").write_text(json.dumps({"dropped_names": [], "note": "no subgroup with cost <= 0 on all perm folds"}))
        return 0
    run_prune(root, args.out_dir, [int(x) for x in args.folds.split(",")], [int(x) for x in args.seeds.split(",")], names, None,
              args.n_jobs, args.lr, args.rounds, args.tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
