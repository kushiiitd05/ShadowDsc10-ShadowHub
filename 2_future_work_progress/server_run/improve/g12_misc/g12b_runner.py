"""Item B runner: cs02 day-level model + cross-fitted stage-1 weekly-total predictions (g12b_two_stage.py).

Usage: python g12_misc/g12b_runner.py <proj_root> <out_dir> --variants g12b_ts,g12b_ts_min --folds 3,4 --seed 20260807 --n-jobs 16
Variants: g12b_ts (stage-1 week, per-open-day value and the six-week profile), g12b_ts_min (week and per-open-day value only).
Same output schema as g3b_runner.py (experiments.csv + oof_<variant>_s<seed>.csv.gz); also writes stage1_<fold>_s<seed>.csv (stage-1 RMSLE on weekly totals).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
IMPROVE = HERE.parent
for _path in (IMPROVE, HERE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import exp_runner as base  # noqa: E402
import g12b_two_stage as ts  # noqa: E402

CHAMPION = "cs02"
STAGE1_ROUNDS = 500
STAGE1_LR = 0.05


@dataclass(frozen=True)
class VariantSpec:
    full: bool


VARIANT_SPECS: dict[str, VariantSpec] = {"g12b_ts": VariantSpec(True), "g12b_ts_min": VariantSpec(False)}


def make_fit_predict(cats: set[str], seed: int, n_jobs: int, rounds: int = STAGE1_ROUNDS, lr: float = STAGE1_LR) -> ts.FitPredict:
    """Stage-1 learner: LightGBM on log1p(weekly total), cs02-style (colsample 0.2, row bagging), a fixed number of rounds (no early stopping)."""
    import lightgbm as lgb

    def fit_predict(train_w: pd.DataFrame, pred_w: pd.DataFrame) -> np.ndarray:
        fit = train_w[train_w["_total"] > 0]
        feats = ts.stage1_feature_names(fit)
        model = lgb.LGBMRegressor(objective="regression", n_estimators=rounds, learning_rate=lr, num_leaves=63, min_child_samples=40, subsample=0.85,
                                  subsample_freq=1, colsample_bytree=0.2, reg_lambda=1.0, random_state=seed, n_jobs=n_jobs, verbosity=-1)
        model.fit(base.prep(fit, feats, cats), np.log1p(fit["_total"].to_numpy(float)))
        return model.predict(base.prep(pred_w, feats, cats), raw_score=True)

    return fit_predict


def augment_ctx(ctx: dict, stage1: dict, full: bool) -> dict:
    """New ctx whose train/hold/valid frames carry the stage-1 columns (inputs are never mutated)."""
    out = {**ctx}
    for key in ("train", "hold"):
        table, pred = stage1[key]
        out[key] = ts.attach(ctx[key], table, pred, full)
    table, pred = stage1["valid"]
    out["valid"] = ts.attach(ctx["valid"].assign(_origin=pd.Timestamp(ctx["origin"])), table, pred, full)
    return out


def stage1_quality(ctx: dict, stage1: dict) -> dict:
    """RMSLE of the stage-1 weekly-total log prediction on holdout and validation weeks with a positive total."""
    out = {}
    for key in ("hold", "valid"):
        table, pred = stage1[key]
        frame = ctx[key] if key == "hold" else ctx[key].assign(_origin=pd.Timestamp(ctx["origin"]))
        total = ts.week_table(frame, ["IsOpen"])
        merged = table[["HubID", "_origin", "_week"]].assign(p=pred.to_numpy()).merge(total[["HubID", "_origin", "_week", "_total"]], on=ts.KEY)
        ok = merged["_total"] > 0
        out[f"s1_rmse_{key}"] = float(np.sqrt(np.mean((merged.loc[ok, "p"] - np.log1p(merged.loc[ok, "_total"])) ** 2)))
    return out


def variant_ctx(aug: dict, name: str) -> dict:
    extra = [c for c in aug["valid"].columns if c.startswith(ts.S1_PREFIX)]
    return {**aug, "columns": [*aug["columns"], *extra], "variant": name}


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("proj_root", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--variants", required=True, help="comma list of: " + ",".join(VARIANT_SPECS))
    ap.add_argument("--folds", default="3,4")
    ap.add_argument("--seed", type=int, default=20260807)
    ap.add_argument("--n-jobs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--rounds", type=int, default=1500)
    ap.add_argument("--blocks", type=int, default=6, help="cross-fit blocks over the training origins")
    return ap.parse_args(argv)


def record_config(name: str) -> SimpleNamespace:
    return SimpleNamespace(**{**base.VARIANTS[CHAMPION].__dict__, "recipe": CHAMPION, "stage1_full": VARIANT_SPECS[name].full,
                              "stage1_rounds": STAGE1_ROUNDS})


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.proj_root.resolve()
    sys.path.insert(0, str(root))
    names = [v.strip() for v in args.variants.split(",")]
    unknown = [v for v in names if v not in VARIANT_SPECS]
    if unknown:
        raise SystemExit(f"unknown variants: {unknown}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cats = set(ts.STATIC_CATEGORICAL)      # day-level calendar categoricals become numeric week aggregates, so only the static ones stay categorical
    results: dict[str, list[dict]] = {n: [] for n in names}
    oofs: dict[str, list[pd.DataFrame]] = {n: [] for n in names}
    quality = []
    for fold in [int(x) for x in args.folds.split(",")]:
        t0 = time.perf_counter()
        ctx = base.prepare_fold(root, fold, args.seed)
        print(f"fold {fold}: loaded in {time.perf_counter() - t0:.0f}s, train rows {len(ctx['train'])}", flush=True)
        t1 = time.perf_counter()
        columns = [*ctx["columns"], "HubID_TE"]
        stage1 = ts.build_stage1(ctx["train"], ctx["hold"], ctx["valid"], ctx["origin"], columns,
                                 make_fit_predict(cats, args.seed, args.n_jobs), args.blocks)
        q = {"fold": fold, "stage1_seconds": round(time.perf_counter() - t1, 1), **stage1_quality(ctx, stage1)}
        quality.append(q)
        print("stage1", json.dumps(q), flush=True)
        for n in names:
            aug = augment_ctx(ctx, stage1, VARIANT_SPECS[n].full)
            oof, scores = base.fit_variant(variant_ctx(aug, n), base.VARIANTS[CHAMPION], args.seed, args.n_jobs, args.lr, args.rounds)
            oofs[n].append(oof)
            results[n].append(scores)
            print(n, "fold", fold, json.dumps(scores), flush=True)
            del aug
        del ctx, stage1
    pd.DataFrame(quality).to_csv(args.out_dir / f"stage1_quality_s{args.seed}.csv", index=False)
    for n in names:
        pd.concat(oofs[n], ignore_index=True).to_csv(args.out_dir / f"oof_{n}_s{args.seed}.csv.gz", index=False, compression="gzip")
        base.write_record(args.out_dir, n, record_config(n), args.seed, results[n], args)
        print("DONE", n, round(sum(r["rmsle_all"] for r in results[n]) / len(results[n]), 5), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
