"""G11 runner: cs02 trained on alternative origin sets, one (set, fold, seed) per call.

Usage: python origins_runner.py <proj_root> <data_root> <out_dir> --set base|summer|dense7|season --fold 1..5 --seed 20260807 --n-jobs 6

Sets (all fit exp_runner.VARIANTS["cs02"] with the harness settings: lr 0.05, <=1500 rounds, early stopping on a holdout origin):
  base    the 14-day complete origins the harness loads today (control)
  summer  base + the incomplete summer-2014 / reopening origins (rows of hubs with observed labels only)
  dense7  base + the extra 7-day-grid origins
  season  base origins within +/-45 calendar days of the validation origin, own in-season holdout (member for the log blend)
Validation rows, features, target encoding recipe and early-stopping holdout (except for `season`) are identical across sets;
training origins are purged on both sides of the holdout. Outputs: res_<set>_f<fold>_s<seed>.json and oof_<set>_f<fold>_s<seed>.csv.gz.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
for _p in (HERE.parent, HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import exp_runner as base  # noqa: E402
import origin_builder as ob  # noqa: E402

LEVEL_COLS = ("Orders_90D_MedianOpen", "Orders_180D_MedianOpen", "Orders_365D_MedianOpen", "HistHubDOWOrdersMean")
SEASON_HALF_WIDTH = 45
MIN_SEASON_TRAIN_ORIGINS = 2


def ensure_view(proj: Path, data: Path) -> Path:
    """Shadow project root whose folds dir holds fold 1-4 (links to the project) plus fold 5; the project itself is not touched."""
    folds = data / "view" / "data" / "processed" / "folds"
    folds.mkdir(parents=True, exist_ok=True)
    sources = list((proj / "data/processed/folds").glob("fold_*.csv.gz")) + list((data / "folds").glob("fold_5_*.csv.gz"))
    for src in sources:
        link = folds / src.name
        if not link.exists():
            os.symlink(src, link)
    return data / "view"


def read_manifest(data: Path) -> pd.DataFrame:
    return pd.read_csv(data / "manifest_all.csv", parse_dates=["origin", "target_start", "target_end"])


def pool_for(manifest: pd.DataFrame, set_name: str, validation_origin: pd.Timestamp) -> pd.DataFrame:
    if set_name == "season":
        return ob.season_pool(manifest, validation_origin, SEASON_HALF_WIDTH)
    return ob.select_origin_set(manifest, set_name, validation_origin)


def install_loaders(data: Path, manifest: pd.DataFrame, set_name: str) -> dict:
    """Patch exp_runner.load_origins / split_holdout for the chosen set; returns a dict that is filled with run facts."""
    facts: dict = {}
    original_split = base.split_holdout

    def load(root: Path, validation_origin: pd.Timestamp, columns: list[str]) -> pd.DataFrame:
        pool = pool_for(manifest, set_name, validation_origin)
        need = list(dict.fromkeys([*columns, "ActualOrderVolume", "IsOpen", "HubID", *LEVEL_COLS]))
        parts = []
        extra = pd.DataFrame()
        if set_name == "season":  # the control's holdout origin is loaded too (flagged), in case the in-season holdout is unusable
            ref = ob.reference_holdout(manifest, validation_origin, ("base",))
            extra = manifest[(manifest["origin"] == ref) & ~manifest["origin"].isin(pool["origin"])]
        for row, inpool in [*[(r, True) for r in pool.itertuples()], *[(r, False) for r in extra.itertuples()]]:
            frame = pd.read_parquet(data / row.path, columns=need)
            frame["_origin"] = row.origin
            frame["_target_start"] = row.target_start
            frame["_inpool"] = inpool
            parts.append(frame)
        facts["pool_origins"] = [str(o.date()) for o in pool["origin"]]
        facts["pool_kinds"] = pool["kind"].value_counts().to_dict()
        facts["validation_origin"] = str(pd.Timestamp(validation_origin).date())
        return pd.concat(parts, ignore_index=True)

    def split(train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        val = pd.Timestamp(facts["validation_origin"])
        reference = ob.reference_holdout(manifest, val, ("base",))
        pooled = train["_inpool"]
        holdout = pd.Timestamp(train.loc[pooled, "_origin"].max()) if set_name == "season" else reference
        keep, hold_mask = ob.split_train_hold(train["_origin"], holdout)
        keep, hold_mask = keep & pooled, hold_mask & pooled
        facts["holdout_fallback"] = False
        if set_name == "season" and train.loc[keep, "_origin"].nunique() < MIN_SEASON_TRAIN_ORIGINS:
            # the in-season origins cluster inside 42 days of each other (fold 5: only Jul 2013 exists), so the purge would leave
            # nothing to train on: early-stop on the same out-of-season holdout as the control instead
            holdout = reference
            keep, hold_mask = ob.split_train_hold(train["_origin"], holdout)
            keep = keep & pooled
            facts["holdout_fallback"] = True
        out_train, out_hold = train[keep].drop(columns="_inpool"), train[hold_mask].drop(columns="_inpool")
        if set_name == "base":  # the new split must reproduce the harness exactly for the control set
            ref_train, ref_hold = original_split(train)
            assert len(ref_train) == len(out_train) and len(ref_hold) == len(out_hold), "control split differs from the harness"
        facts["holdout_origin"] = str(holdout.date())
        facts["train_origins"] = sorted({str(o.date()) for o in out_train["_origin"].unique()})
        facts["train_rows_before_positive_filter"] = int(len(out_train))
        return out_train, out_hold

    base.load_origins = load
    base.split_holdout = split
    return facts


def same_as_base(manifest: pd.DataFrame, set_name: str, val: pd.Timestamp) -> bool:
    return set_name != "base" and pool_for(manifest, set_name, val)["origin"].tolist() == pool_for(manifest, "base", val)["origin"].tolist()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("proj_root", type=Path)
    ap.add_argument("data_root", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--set", dest="set_name", required=True, choices=("base", "summer", "dense7", "season"))
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--seed", type=int, default=20260807)
    ap.add_argument("--n-jobs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--rounds", type=int, default=1500)
    a = ap.parse_args()
    proj, data = a.proj_root.resolve(), a.data_root.resolve()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{a.set_name}_f{a.fold}_s{a.seed}"
    if (a.out_dir / f"res_{stem}.json").exists():
        print("exists, skipping", stem)
        return 0
    sys.path.insert(0, str(proj))
    view = ensure_view(proj, data)
    manifest = read_manifest(data)
    fold_file = next((view / "data/processed/folds").glob(f"fold_{a.fold}_*.csv.gz"))
    val_origin = pd.Timestamp(fold_file.name.split("_")[2].split(".")[0])
    if same_as_base(manifest, a.set_name, val_origin):
        (a.out_dir / f"res_{stem}.json").write_text(json.dumps({"set": a.set_name, "fold": a.fold, "seed": a.seed, "identical_to_base": True}))
        print("training origins identical to base for this fold; recorded marker", stem)
        return 0
    facts = install_loaders(data, manifest, a.set_name)
    t0 = time.perf_counter()
    ctx = base.prepare_fold(view, a.fold, a.seed)
    load_s = time.perf_counter() - t0
    print(f"{stem}: loaded in {load_s:.0f}s train rows {len(ctx['train'])} hold rows {len(ctx['hold'])} origins {len(facts['train_origins'])}", flush=True)
    oof, scores = base.fit_variant(ctx, base.VARIANTS["cs02"], a.seed, a.n_jobs, a.lr, a.rounds)
    dup = ob.duplication_stats([pd.Timestamp(o) for o in facts["train_origins"]])
    record = {**scores, **facts, "set": a.set_name, "seed": a.seed, "load_seconds": round(load_s, 1), "n_jobs": a.n_jobs,
              "train_origin_count": len(facts["train_origins"]), "duplication": dup,
              "peak_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 1)}
    oof.to_csv(a.out_dir / f"oof_{stem}.csv.gz", index=False, compression="gzip")
    (a.out_dir / f"res_{stem}.json").write_text(json.dumps(record, default=str))
    print("DONE", stem, round(scores["rmsle_all"], 5), "excl0", round(scores["rmsle_excl_open0"], 5), "iters", scores["best_iteration"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
