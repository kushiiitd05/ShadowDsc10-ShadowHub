"""LightGBM screening harness for the improvement plan (Phase 0 baseline + Phase 1 variants).

Usage: python exp_runner.py <proj_root> <out_dir> --variants base,no_te --folds 2,3,4 --seed 20260807 --n-jobs 16

Fixed screening settings (same for every variant): lr 0.05, up to 1500 rounds, early stopping on a
chronological holdout (the last eligible training origin), training origins purged so no label window
overlaps the holdout window, HubID encoding cross-fitted. One row per run goes to <out_dir>/experiments.csv.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from helpers import decode_prediction, encode_target, fold_scores, level_baseline, recency_weights  # noqa: E402


@dataclass(frozen=True)
class Config:
    te: bool = True
    target: str = "log"            # log | level | level_init
    offset: bool = False           # global log offset fitted on the chronological holdout
    recency: float | None = None   # half-life in days
    colsample: float = 0.8
    subsample_freq: int = 0
    objective: str = "regression"
    linear_tree: bool = False
    num_leaves: int = 63
    min_child: int = 80
    reg_lambda: float = 1.0
    extra_trees: bool = False
    fixed_rounds: int | None = None  # no early stopping: fit exactly this many rounds
    model: str = "lightgbm"        # lightgbm | catboost (production settings: depth 8, lr 0.04, l2 5)


VARIANTS: dict[str, Config] = {
    "base": Config(),
    "no_te": Config(te=False),
    "level": Config(target="level"),
    "level_off": Config(target="level", offset=True),
    "offset": Config(offset=True),
    "recency365": Config(recency=365.0),
    "recency180": Config(recency=180.0),
    "colsample03": Config(colsample=0.3, subsample_freq=1),
    "colsample03_nobag": Config(colsample=0.3, subsample_freq=0),
    "colsample05": Config(colsample=0.5, subsample_freq=1),
    "colsample02": Config(colsample=0.2, subsample_freq=1),
    "bag_only": Config(subsample_freq=1),
    "tweedie_init": Config(target="level_init", objective="tweedie"),
    "linear": Config(target="level", linear_tree=True),
    # round 2: champion = colsample02 (feature fraction 0.2 with row bagging); everything below is a one-change-at-a-time probe
    "cs02": Config(colsample=0.2, subsample_freq=1),
    "cs015": Config(colsample=0.15, subsample_freq=1),
    "cs01": Config(colsample=0.1, subsample_freq=1),
    "cs02_nobag": Config(colsample=0.2, subsample_freq=0),
    "cs02_no_te": Config(colsample=0.2, subsample_freq=1, te=False),
    "cs02_recency365": Config(colsample=0.2, subsample_freq=1, recency=365.0),
    "cs02_leaves127": Config(colsample=0.2, subsample_freq=1, num_leaves=127),
    "cs02_leaves31": Config(colsample=0.2, subsample_freq=1, num_leaves=31),
    "cs02_mcs200": Config(colsample=0.2, subsample_freq=1, min_child=200),
    "cs02_lambda10": Config(colsample=0.2, subsample_freq=1, reg_lambda=10.0),
    "cs02_extra": Config(colsample=0.2, subsample_freq=1, extra_trees=True),
    # round 3: stack the single-change gains of round 2 (no_te -0.0015, lambda10 -0.0011, leaves127 -0.0010 vs cs02)
    "cs02_no_te_l10": Config(colsample=0.2, subsample_freq=1, te=False, reg_lambda=10.0),
    "cs02_no_te_l10_127": Config(colsample=0.2, subsample_freq=1, te=False, reg_lambda=10.0, num_leaves=127),
    "cs02_l10_127": Config(colsample=0.2, subsample_freq=1, reg_lambda=10.0, num_leaves=127),
    # baseline ensemble members with the production settings, on the corrected folds (run with --lr 0.025 --rounds 3000 for LightGBM)
    # fixed round counts (no early stopping): best iteration moves between ~100 and ~2500 across folds/seeds, so the stopping point is itself noise
    "cs02_fix300": Config(colsample=0.2, subsample_freq=1, fixed_rounds=300),
    "cs02_fix600": Config(colsample=0.2, subsample_freq=1, fixed_rounds=600),
    "cs02_fix1000": Config(colsample=0.2, subsample_freq=1, fixed_rounds=1000),
    "prod_tweedie": Config(target="raw", objective="tweedie"),
    "prod_cat": Config(model="catboost"),
    "cs02_tweedie": Config(target="raw", objective="tweedie", colsample=0.2, subsample_freq=1),
    "cs02_cat": Config(model="catboost", colsample=0.2),
    # round 4: neighbours of the hyper-parameter search winner hp_017 (colsample 0.15, 255 leaves, min_child 160, lambda 24.1)
    "hp17": Config(colsample=0.15, subsample_freq=1, num_leaves=255, min_child=160, reg_lambda=24.1),
    "hp17_l10": Config(colsample=0.15, subsample_freq=1, num_leaves=255, min_child=160, reg_lambda=10.0),
    "hp17_l50": Config(colsample=0.15, subsample_freq=1, num_leaves=255, min_child=160, reg_lambda=50.0),
    "hp17_511": Config(colsample=0.15, subsample_freq=1, num_leaves=511, min_child=160, reg_lambda=24.1),
    "hp17_mcs320": Config(colsample=0.15, subsample_freq=1, num_leaves=255, min_child=320, reg_lambda=24.1),
    "hp17_cs01": Config(colsample=0.10, subsample_freq=1, num_leaves=255, min_child=160, reg_lambda=24.1),
    "hp17_cs02": Config(colsample=0.20, subsample_freq=1, num_leaves=255, min_child=160, reg_lambda=24.1),
    "hp17_127": Config(colsample=0.15, subsample_freq=1, num_leaves=127, min_child=160, reg_lambda=24.1),
    "hp17_fix1000": Config(colsample=0.15, subsample_freq=1, num_leaves=255, min_child=160, reg_lambda=24.1, fixed_rounds=1000),
}


def load_origins(root: Path, validation_origin: pd.Timestamp, columns: list[str]) -> pd.DataFrame:
    manifest = pd.read_csv(root / "data/processed/training_origin_manifest.csv", parse_dates=["origin", "target_start", "target_end"])
    eligible = manifest[manifest["target_end"].le(validation_origin)].sort_values("origin")
    need = list(dict.fromkeys([*columns, "ActualOrderVolume", "IsOpen", "HubID", *[c for c in ("Orders_90D_MedianOpen", "Orders_180D_MedianOpen", "Orders_365D_MedianOpen", "HistHubDOWOrdersMean")]]))
    parts = []
    for row in eligible.itertuples():
        frame = pd.read_parquet(root / row.path, columns=need)
        frame["_origin"] = row.origin
        frame["_target_start"] = row.target_start
        parts.append(frame)
    return pd.concat(parts, ignore_index=True)


def split_holdout(train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological holdout = last origin; purge training origins whose label window reaches it."""
    last = train["_origin"].max()
    hold = train[train["_origin"] == last]
    hold_start = hold["_target_start"].iloc[0]
    # the training label window of origin o ends at o + 42 days; keep only those strictly before the holdout labels
    keep = train["_origin"] + pd.Timedelta(days=42) < hold_start
    return train[keep], hold


def prep(frame: pd.DataFrame, columns: list[str], categorical: set[str]) -> pd.DataFrame:
    out = frame[columns].copy()
    for c in out.columns:
        if c in categorical or out[c].dtype == object:
            out[c] = out[c].astype("string").fillna("__MISSING__").astype("category")
        elif pd.api.types.is_bool_dtype(out[c]):
            out[c] = out[c].astype("int8")
        elif out[c].dtype == np.float64:
            out[c] = out[c].astype("float32")
    return out


def prepare_fold(root: Path, fold_number: int, seed: int) -> dict:
    """Load a fold once: chronological holdout, purged training origins, level baseline, cross-fitted HubID encoding."""
    from sklearn.preprocessing import TargetEncoder
    from src.model_pipeline import feature_columns

    valid = pd.read_csv(next((root / "data/processed/folds").glob(f"fold_{fold_number}_*.csv.gz")),
                        parse_dates=["Date", "ForecastOrigin", "HistoryStartDate", "HistoryLastDate"])
    origin = pd.Timestamp(valid["ForecastOrigin"].iloc[0])
    columns = [c for c in feature_columns(valid, "all") if c != "HubID_TargetEncoded"]
    train_all = load_origins(root, origin, columns)
    train, hold = split_holdout(train_all)
    del train_all
    train, hold = train.reset_index(drop=True), hold.reset_index(drop=True)
    level_cols = ("Orders_90D_MedianOpen", "Orders_180D_MedianOpen", "Orders_365D_MedianOpen", "HistHubDOWOrdersMean")
    for frame in (train, hold, valid):
        frame["_level"] = level_baseline({c: frame[c].to_numpy() for c in level_cols})
    mask = train["IsOpen"].eq(1) & train["ActualOrderVolume"].gt(0)
    enc = TargetEncoder(smooth="auto", random_state=seed)
    cross = enc.fit_transform(train.loc[mask, ["HubID"]], np.log1p(train.loc[mask, "ActualOrderVolume"]))
    train["HubID_TE"] = enc.transform(train[["HubID"]]).ravel()  # rows outside the fit mask: plain transform
    train.loc[mask, "HubID_TE"] = cross.ravel()                  # cross-fitted values for fitted rows
    hold["HubID_TE"] = enc.transform(hold[["HubID"]]).ravel()
    valid["HubID_TE"] = enc.transform(valid[["HubID"]]).ravel()
    for frame in (train, hold, valid):
        frame["HubID_TE"] = frame["HubID_TE"].astype("float32")
    return {"train": train, "hold": hold, "valid": valid, "origin": origin, "columns": columns, "fold": fold_number}


def fit_catboost_model(x_train: pd.DataFrame, y_train: np.ndarray, x_hold: pd.DataFrame, y_hold: np.ndarray,
                       weight: np.ndarray | None, cfg: Config, seed: int, n_jobs: int, rounds: int):
    """Production CatBoost settings; categorical columns passed as strings. Returns (model, raw-score predictor, best iteration)."""
    from catboost import CatBoostRegressor, Pool

    def as_text(frame: pd.DataFrame) -> pd.DataFrame:
        return frame.astype({c: str for c in frame.columns if str(frame[c].dtype) == "category"})

    xt, xh = as_text(x_train), as_text(x_hold)
    cat_idx = [i for i, c in enumerate(xt.columns) if xt[c].dtype == object]
    model = CatBoostRegressor(loss_function="RMSE", eval_metric="RMSE", iterations=rounds, learning_rate=0.04, depth=8, l2_leaf_reg=5.0,
                              random_strength=0.5, rsm=cfg.colsample if cfg.colsample < 0.8 else None, random_seed=seed,
                              thread_count=n_jobs, verbose=False, allow_writing_files=False)
    model.fit(Pool(xt, y_train, cat_features=cat_idx, weight=weight), eval_set=Pool(xh, y_hold, cat_features=cat_idx), early_stopping_rounds=100, verbose=False)
    return model, (lambda frame: model.predict(as_text(frame))), int(model.get_best_iteration())


def fit_variant(ctx: dict, cfg: Config, seed: int, n_jobs: int, lr: float, rounds: int) -> tuple[pd.DataFrame, dict]:
    import lightgbm as lgb
    from src.model_pipeline import CATEGORICAL_COLUMNS

    train, hold, valid, origin = ctx["train"], ctx["hold"], ctx["valid"], ctx["origin"]
    feats = list(ctx["columns"])
    if cfg.linear_tree:  # categorical columns are not used inside leaf models; keep numeric trend features only
        feats = [c for c in feats if c not in CATEGORICAL_COLUMNS and valid[c].dtype != object]
    if cfg.te:
        feats.append("HubID_TE")
    cats = set(CATEGORICAL_COLUMNS)

    def positive(frame: pd.DataFrame) -> pd.Series:
        return frame["IsOpen"].eq(1) & frame["ActualOrderVolume"].gt(0)

    mt, mh = positive(train), positive(hold)
    x_train, x_hold = prep(train.loc[mt], feats, cats), prep(hold.loc[mh], feats, cats)
    y_train = encode_target(train.loc[mt, "ActualOrderVolume"].to_numpy(), train.loc[mt, "_level"].to_numpy(), cfg.target)
    y_hold = encode_target(hold.loc[mh, "ActualOrderVolume"].to_numpy(), hold.loc[mh, "_level"].to_numpy(), cfg.target)
    weight = None
    if cfg.recency:
        weight = recency_weights(train.loc[mt, "_origin"].to_numpy().astype("datetime64[D]"), origin.to_datetime64(), cfg.recency)
    kwargs = {}
    if cfg.target == "level_init":
        kwargs = {"init_score": np.log(train.loc[mt, "_level"].to_numpy()), "eval_init_score": [np.log(hold.loc[mh, "_level"].to_numpy())]}
    params = dict(objective=cfg.objective, n_estimators=rounds, learning_rate=lr, num_leaves=cfg.num_leaves, min_child_samples=cfg.min_child,
                  subsample=0.85, subsample_freq=cfg.subsample_freq, colsample_bytree=cfg.colsample, reg_lambda=cfg.reg_lambda, extra_trees=cfg.extra_trees,
                  random_state=seed, n_jobs=n_jobs, verbosity=-1)
    if cfg.objective == "tweedie":
        params["tweedie_variance_power"] = 1.1
    if cfg.linear_tree:
        params.update(linear_tree=True, linear_lambda=1.0)
    start = time.perf_counter()
    if cfg.model == "catboost":
        model, predict_raw, best_iteration = fit_catboost_model(x_train, y_train, x_hold, y_hold, weight, cfg, seed, n_jobs, rounds)
    else:
        if cfg.fixed_rounds:
            params["n_estimators"] = cfg.fixed_rounds
        model = lgb.LGBMRegressor(**params)
        stop = [] if cfg.fixed_rounds else [lgb.early_stopping(100, verbose=False)]
        model.fit(x_train, y_train, sample_weight=weight, eval_set=[(x_hold, y_hold)], callbacks=stop, **kwargs)
        predict_raw = lambda frame: model.predict(frame, raw_score=True)  # noqa: E731
        best_iteration = int(cfg.fixed_rounds or model.best_iteration_ or rounds)
    fit_seconds = time.perf_counter() - start

    offset = 0.0
    if cfg.offset:  # global additive log offset from the chronological holdout only
        base = decode_prediction(predict_raw(x_hold), hold.loc[mh, "_level"].to_numpy(), cfg.target)
        offset = float(np.mean(np.log1p(hold.loc[mh, "ActualOrderVolume"].to_numpy()) - np.log1p(base)))
    open_mask = valid["IsOpen"].eq(1).to_numpy()
    pred = np.zeros(len(valid))
    raw = predict_raw(prep(valid.loc[open_mask], feats, cats))
    pred[open_mask] = decode_prediction(raw, valid.loc[open_mask, "_level"].to_numpy(), cfg.target, offset)
    scores = fold_scores(valid["ActualOrderVolume"].to_numpy(float), pred, valid["IsOpen"].to_numpy())
    scores.update(fold=ctx["fold"], best_iteration=best_iteration, fit_seconds=round(fit_seconds, 1),
                  train_rows=int(mt.sum()), offset=offset)
    oof = valid[["HubID", "Date", "Horizon", "IsOpen", "ActualOrderVolume"]].copy()
    oof["pred"] = pred
    oof["fold"] = ctx["fold"]
    return oof, scores


def write_record(out_dir: Path, name: str, cfg: Config, seed: int, rows: list[dict], args: argparse.Namespace) -> None:
    record = {
        "date": date.today().isoformat(), "variant": name, "seed": seed, "model": cfg.model,
        "config": json.dumps(cfg.__dict__, sort_keys=True), "config_hash": hashlib.sha256(json.dumps(cfg.__dict__, sort_keys=True).encode()).hexdigest()[:10],
        "lr": args.lr, "rounds": args.rounds, "n_jobs": args.n_jobs,
        "mean_all": float(np.mean([r["rmsle_all"] for r in rows])),
        "mean_excl_open0": float(np.mean([r["rmsle_excl_open0"] for r in rows])),
        "mean_bias_open": float(np.mean([r["bias_open"] for r in rows])),
        "fit_seconds": round(sum(r["fit_seconds"] for r in rows), 1),
    }
    for r in rows:
        record[f"fold{r['fold']}_all"] = r["rmsle_all"]
        record[f"fold{r['fold']}_excl_open0"] = r["rmsle_excl_open0"]
        record[f"fold{r['fold']}_best_iter"] = r["best_iteration"]
    path = out_dir / "experiments.csv"
    frame = pd.DataFrame([record])
    (pd.concat([pd.read_csv(path), frame], ignore_index=True) if path.exists() else frame).to_csv(path, index=False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("proj_root", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--variants", required=True, help="comma list of: " + ",".join(sorted(VARIANTS)))
    ap.add_argument("--folds", default="2,3,4")
    ap.add_argument("--seed", type=int, default=20260807)
    ap.add_argument("--n-jobs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--rounds", type=int, default=1500)
    args = ap.parse_args()
    root = args.proj_root.resolve()
    sys.path.insert(0, str(root))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    names = [v.strip() for v in args.variants.split(",")]
    unknown = [v for v in names if v not in VARIANTS]
    if unknown:
        raise SystemExit(f"unknown variants: {unknown}")
    results: dict[str, list[dict]] = {n: [] for n in names}
    oofs: dict[str, list[pd.DataFrame]] = {n: [] for n in names}
    for f in [int(x) for x in args.folds.split(",")]:
        t0 = time.perf_counter()
        ctx = prepare_fold(root, f, args.seed)
        print(f"fold {f}: loaded in {time.perf_counter() - t0:.0f}s, train rows {len(ctx['train'])}", flush=True)
        for n in names:
            oof, s = fit_variant(ctx, VARIANTS[n], args.seed, args.n_jobs, args.lr, args.rounds)
            oofs[n].append(oof)
            results[n].append(s)
            print(n, "fold", f, json.dumps(s), flush=True)
        del ctx
    for n in names:
        pd.concat(oofs[n], ignore_index=True).to_csv(args.out_dir / f"oof_{n}_s{args.seed}.csv.gz", index=False, compression="gzip")
        write_record(args.out_dir, n, VARIANTS[n], args.seed, results[n], args)
        print("DONE", n, round(float(np.mean([r['rmsle_all'] for r in results[n]])), 5), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
