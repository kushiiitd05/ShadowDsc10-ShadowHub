"""G9a: adversarial validation of the cs02 feature set.

Usage: python adv_validation.py <proj_root> <out_dir> --targets 4,3,2,1,5,test --n-jobs 12

For every target (a dev fold, or the real test window) a LightGBM classifier separates
  label 0: open rows of the training origins available at that origin (modes: 'all' = a random sample of all of them,
           'recent' = every row of the latest --recent origins)
  label 1: open rows of the fold's validation block (or of the real test-window feature frame).
CV is grouped by HubID (HubID itself is not a feature), so the classifier cannot memorise a hub's frozen history values.

Why several variants (first run: raw AUC was 1.0 even without calendar columns): every forecast origin is identifiable from
cross-hub shared values (the orders of the origin day, open-day counts of the last 28 days, ...), so a raw classifier separates ANY
origin from the others. Variants:
  raw   : features as the model sees them (feature sets 'all' and 'nocal' = without calendar-like columns)
  std   : every numeric feature is standardised within its origin (median / IQR over hubs), which removes origin-level identity and
          keeps only the cross-hub shape; with the 'core' feature set (also without schedule / history-length columns) it is the variant the drift table is built from
  placebo (raw and std): the last training origin as the "validation" side against the earlier ones. Real drift = AUC above placebo.
Outputs: adv_auc.csv, adv_top15.csv, drift_table.csv. Fold 5 needs ALLOW_INCOMPLETE_BLOCK=1 in the environment.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
for _p in (HERE, HERE.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from perm_prune import assign_group, assign_subgroup, is_calendar_like  # noqa: E402

DATE_COLUMNS = ["Date", "ForecastOrigin", "HistoryStartDate", "HistoryLastDate"]
DRIFT_SHARE = 0.02      # a non-calendar feature is a drift candidate above this mean gain share ...
DRIFT_MARGIN = 0.05     # ... but only when the std classifier beats the placebo AUC by this margin


# ---------------------------------------------------------------- pure helpers
def auc_score(y_true: np.ndarray, score: np.ndarray) -> float:
    """Rank-based ROC AUC (ties get average rank); 0.5 = indistinguishable."""
    from scipy.stats import rankdata

    y = np.asarray(y_true).astype(bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("AUC needs both classes")
    ranks = rankdata(np.asarray(score, dtype=float))
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def standardize_within(frame: pd.DataFrame, by: np.ndarray, cols: list[str]) -> pd.DataFrame:
    """Copy of `frame` whose numeric `cols` are (x - median) / scale per group in `by`; scale = IQR, else std, else 1. Never mutates input."""
    out = frame.copy()
    keys = pd.Series(np.asarray(by), index=frame.index)
    for c in cols:
        x = frame[c].astype("float64")
        g = x.groupby(keys)
        med, q25, q75, sd = (keys.map(g.median()), keys.map(g.quantile(0.25)), keys.map(g.quantile(0.75)), keys.map(g.std()))
        iqr = q75 - q25
        scale = iqr.where(iqr > 0, sd).where(lambda s: s > 0, 1.0).fillna(1.0)
        out[c] = ((x - med) / scale).astype("float32")
    return out


def is_origin_identity(col: str) -> bool:
    """Columns that identify a forecast origin for every hub alike: calendar-like, promo / holiday schedules, history-length dependent
    columns (availability flags and the 364/728/730-day lags and windows, which are NaN or short for early origins)."""
    return (is_calendar_like(col) or assign_group(col) in ("promo", "holiday") or assign_subgroup(col) == "lags_avail"
            or any(tag in col for tag in ("364D", "728D", "730D")))


def day_key(dates: np.ndarray) -> np.ndarray:
    """Integer day number of datetime64 values (stable group key for the within-origin standardisation)."""
    return np.asarray(dates).astype("datetime64[D]").astype("int64")


def adversarial_cv(x: pd.DataFrame, y: np.ndarray, groups: np.ndarray, *, n_splits: int = 3, rounds: int = 150, lr: float = 0.1,
                   n_jobs: int = 4, seed: int = 0) -> tuple[float, float, pd.Series]:
    """Group-CV LightGBM classifier. Returns (mean fold AUC, std of fold AUC, mean normalised gain share per feature). Never mutates x."""
    import lightgbm as lgb
    from sklearn.model_selection import GroupKFold

    shares, aucs = [], []
    for tr, te in GroupKFold(n_splits=n_splits).split(x, y, groups):
        clf = lgb.LGBMClassifier(n_estimators=rounds, learning_rate=lr, num_leaves=31, min_child_samples=100, colsample_bytree=0.5,
                                 subsample=0.8, subsample_freq=1, max_bin=63, importance_type="gain", n_jobs=n_jobs, verbosity=-1,
                                 random_state=seed)
        clf.fit(x.iloc[tr], y[tr])
        aucs.append(auc_score(y[te], clf.predict_proba(x.iloc[te])[:, 1]))
        gain = pd.Series(clf.feature_importances_, index=x.columns, dtype=float)
        shares.append(gain / max(gain.sum(), 1e-12))
    return float(np.mean(aucs)), float(np.std(aucs)), pd.concat(shares, axis=1).mean(axis=1)


def rank_drift(shares: dict[str, pd.Series], recent: dict[str, pd.Series], auc_real: pd.Series, auc_placebo: pd.Series,
               columns: list[str]) -> pd.DataFrame:
    """Ranked drift table. `shares[target]` = std-variant gain share (mode all); `recent[target]` = same for mode recent.

    A feature is a drift candidate when it is not calendar-like, its mean share >= DRIFT_SHARE and the std classifier beats the placebo by DRIFT_MARGIN.
    """
    table = pd.DataFrame({"feature": columns})
    for t, s in shares.items():
        table[f"share_{t}"] = table["feature"].map(s).fillna(0.0)
    share_cols = [c for c in table.columns if c.startswith("share_")]
    table["mean_share_all"] = table[share_cols].mean(axis=1)
    table["mean_share_recent"] = table["feature"].map(pd.DataFrame(recent).mean(axis=1)).fillna(0.0) if recent else np.nan
    table["group"] = table["feature"].map(assign_group)
    table["calendar_like"] = table["feature"].map(is_calendar_like)
    table["mean_auc_std"] = float(auc_real.mean())
    table["mean_auc_placebo_std"] = float(auc_placebo.mean())
    beats = float(auc_real.mean()) >= float(auc_placebo.mean()) + DRIFT_MARGIN
    table["drift_candidate"] = (~table["calendar_like"]) & (table["mean_share_all"] >= DRIFT_SHARE) & beats
    table = table.sort_values("mean_share_all", ascending=False).reset_index(drop=True)
    table.insert(0, "rank", np.arange(1, len(table) + 1))
    return table


# ---------------------------------------------------------------- data (server)
def read_validation_side(root: Path, target: str) -> tuple[pd.DataFrame, pd.Timestamp]:
    """(validation-side frame, forecast origin) for a dev fold number or 'test'."""
    if target == "test":
        valid = pd.read_csv(root / "data/processed/test_forecast_features.csv.gz", parse_dates=DATE_COLUMNS)
    else:
        valid = pd.read_csv(next((root / "data/processed/folds").glob(f"fold_{int(target)}_*.csv.gz")), parse_dates=DATE_COLUMNS)
    return valid, pd.Timestamp(valid["ForecastOrigin"].iloc[0])


def load_open_origins(root: Path, columns: list[str]) -> pd.DataFrame:
    """Open rows of every training origin (float32), with _origin / _target_end columns; loaded once for all targets."""
    manifest = pd.read_csv(root / "data/processed/training_origin_manifest.csv", parse_dates=["origin", "target_start", "target_end"]).sort_values("origin")
    parts = []
    for row in manifest.itertuples():
        frame = pd.read_parquet(root / row.path, columns=list(dict.fromkeys([*columns, "IsOpen", "HubID"])))
        frame = frame[frame["IsOpen"].eq(1)].copy()
        for c in frame.columns:
            if frame[c].dtype == np.float64:
                frame[c] = frame[c].astype("float32")
        frame["_origin"], frame["_target_end"] = row.origin, row.target_end
        parts.append(frame)
    return pd.concat(parts, ignore_index=True)


def build_xy(neg: pd.DataFrame, pos: pd.DataFrame, feats: list[str], neg_by: np.ndarray, pos_by: np.ndarray, standardize: bool):
    import exp_runner as base
    from src.model_pipeline import CATEGORICAL_COLUMNS

    both = pd.concat([neg[feats], pos[feats]], ignore_index=True)
    if standardize:
        numeric = [c for c in feats if pd.api.types.is_numeric_dtype(both[c]) and c not in CATEGORICAL_COLUMNS]
        both = standardize_within(both, np.r_[neg_by, pos_by], numeric)
    x = base.prep(both, feats, set(CATEGORICAL_COLUMNS))
    y = np.r_[np.zeros(len(neg)), np.ones(len(pos))].astype(int)
    return x, y, np.r_[neg["HubID"].to_numpy(), pos["HubID"].to_numpy()]


def run_target(target: str, pool: pd.DataFrame, columns: list[str], root: Path, args: argparse.Namespace, rng: np.random.Generator):
    valid, origin = read_validation_side(root, target)
    valid_open = valid[valid["IsOpen"].eq(1)].copy()
    train = pool[pool["_target_end"].le(origin)]
    origins = sorted(train["_origin"].unique())
    all_neg = train.iloc[np.sort(rng.choice(len(train), size=min(args.neg_sample, len(train)), replace=False))]
    recent_neg = train[train["_origin"].isin(origins[-args.recent:])]
    placebo_pos = train[train["_origin"].eq(origins[-1])]
    placebo_neg = train[train["_origin"].lt(origins[-1])]
    placebo_neg = placebo_neg.iloc[np.sort(rng.choice(len(placebo_neg), size=min(args.neg_sample, len(placebo_neg)), replace=False))]
    vo = pd.Timestamp(origin)
    missing = [c for c in columns if c not in valid_open.columns]
    if missing:
        print(f"{target}: columns missing from the validation-side frame, skipped: {missing}", flush=True)
    cols = [c for c in columns if c in valid_open.columns and c != "HubID"]
    nocal = [c for c in cols if not is_calendar_like(c)]
    core = [c for c in cols if not is_origin_identity(c)]
    print(f"{target}: origin {vo.date()}, train open rows {len(train)} ({len(origins)} origins), valid open rows {len(valid_open)}, "
          f"features all/nocal/core {len(cols)}/{len(nocal)}/{len(core)}", flush=True)
    # (side, mode, variant, feature set, negatives, positives, feature list, standardise)
    plan = [
        ("real", "all", "raw", "all", all_neg, valid_open, cols, False),
        ("placebo", "all", "raw", "nocal", placebo_neg, placebo_pos, nocal, False),
        ("real", "all", "std", "nocal", all_neg, valid_open, nocal, True),
        ("placebo", "all", "std", "nocal", placebo_neg, placebo_pos, nocal, True),
        ("real", "all", "std", "core", all_neg, valid_open, core, True),
        ("real", "recent", "std", "core", recent_neg, valid_open, core, True),
        ("placebo", "all", "std", "core", placebo_neg, placebo_pos, core, True),
    ]
    auc_rows, top_frames, shares = [], [], {}
    for side, mode, variant, fs, neg, pos, feats, std in plan:
        pos_origin = vo if side == "real" else origins[-1]
        x, y, groups = build_xy(neg, pos, feats, day_key(neg["_origin"].to_numpy()), np.full(len(pos), day_key(np.array([pos_origin.to_datetime64()]))[0]), std)
        auc, sd, share = adversarial_cv(x, y, groups, rounds=args.rounds, n_jobs=args.n_jobs, seed=args.seed)
        row = {"target": target, "side": side, "mode": mode, "variant": variant, "feature_set": fs, "n_neg": int((y == 0).sum()),
               "n_pos": int((y == 1).sum()), "n_features": len(feats), "auc": round(auc, 5), "auc_sd": round(sd, 5)}
        auc_rows.append(row)
        print(json.dumps(row), flush=True)
        top = share.sort_values(ascending=False).head(15)
        top_frames.append(pd.DataFrame({"target": target, "side": side, "mode": mode, "variant": variant, "feature_set": fs, "rank": np.arange(1, len(top) + 1),
                                        "feature": top.index, "group": [assign_group(f) for f in top.index], "gain_share": top.to_numpy().round(5)}))
        shares[(side, mode, variant, fs)] = share
        del x
    return auc_rows, top_frames, shares


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("proj_root", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--targets", default="4,3,2,1,5,test")
    ap.add_argument("--n-jobs", type=int, default=12)
    ap.add_argument("--neg-sample", type=int, default=120_000)
    ap.add_argument("--recent", type=int, default=6, help="number of latest training origins in mode 'recent'")
    ap.add_argument("--rounds", type=int, default=80)
    ap.add_argument("--seed", type=int, default=20260807)
    args = ap.parse_args(argv)
    root = args.proj_root.resolve()
    sys.path.insert(0, str(root))
    from src.model_pipeline import feature_columns

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    ref = pd.read_csv(next((root / "data/processed/folds").glob("fold_4_*.csv.gz")), nrows=5, parse_dates=DATE_COLUMNS)
    columns = [c for c in feature_columns(ref, "all") if c != "HubID_TE"]
    t0 = time.perf_counter()
    pool = load_open_origins(root, columns)
    print(f"loaded {len(pool)} open training rows over {pool['_origin'].nunique()} origins in {time.perf_counter() - t0:.0f}s", flush=True)
    auc_rows, tops, share_all, share_recent = [], [], {}, {}
    for target in [t.strip() for t in args.targets.split(",")]:
        a, t, shares = run_target(target, pool, columns, root, args, rng)
        auc_rows += a
        tops += t
        share_all[target], share_recent[target] = shares[("real", "all", "std", "core")], shares[("real", "recent", "std", "core")]
        pd.DataFrame(auc_rows).to_csv(args.out_dir / "adv_auc.csv", index=False)
        pd.concat(tops, ignore_index=True).to_csv(args.out_dir / "adv_top15.csv", index=False)
    aucs = pd.DataFrame(auc_rows)
    sel = aucs[(aucs["mode"] == "all") & (aucs["variant"] == "std") & (aucs["feature_set"] == "core")]
    real = sel[sel["side"] == "real"].set_index("target")["auc"]
    placebo = sel[sel["side"] == "placebo"].set_index("target")["auc"]
    core = [c for c in columns if c != "HubID" and not is_origin_identity(c)]
    rank_drift(share_all, share_recent, real, placebo, core).to_csv(args.out_dir / "drift_table.csv", index=False)
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
