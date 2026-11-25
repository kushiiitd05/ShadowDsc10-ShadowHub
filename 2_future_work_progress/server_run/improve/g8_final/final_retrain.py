"""Final retrain (plan stage G5): champion recipe cs02, lr 0.025, 1000 fixed rounds, seed-averaged in log space,
optional per-AssortmentTier model set as an extra blend member. All eligible training origins, no holdout, no early stopping.

Usage from the improve dir:
  python g8_final/final_retrain.py <proj_root> <out_dir> --n-jobs 40 [--seeds 20260807,20260808,20260809] [--with-pertier]
  python g8_final/final_retrain.py <proj_root> <out_dir> --n-jobs 16 --dry-run-fold 4     # rehearsal on a validation fold

Real run: trains on every training origin whose label window ends on or before the test forecast origin (2015-06-19), so no label
window reaches the test window that starts 2015-06-20. Dry run: same code, but the "test frame" is fold N and only origins whose
label windows end on or before that fold's origin are used; the fold's actual volumes give an RMSLE.

Outputs (names follow the control run's layout):
  lightgbm_s<seed>.txt, pertier_<tier>_s<seed>.txt   saved LightGBM boosters
  test_pred_<member>.npy                             per-member predictions in order-volume units (closed rows 0)
  submission_final.csv                               Id,OrderVolume in the test frame order (== sample submission order)
  final_manifest.json                                SHA-256 of every model and of the prediction file, config, timings
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
for _p in (HERE.parent, HERE.parent / "g5_pergroup"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
from exp_runner import VARIANTS, Config, load_origins, prep  # noqa: E402
from helpers import decode_prediction, fold_scores  # noqa: E402

SEEDS = (20260807, 20260808, 20260809)
LEARNING_RATE = 0.025
ROUNDS = 1000
RECIPE = "cs02"
GROUP_COLUMN = "AssortmentTier"
BLEND_MODES = ("member", "half")


# ---------------------------------------------------------------- pure helpers (unit-tested)
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def final_config() -> Config:
    """cs02 with a fixed round count; must equal the registered cs02_fix1000 variant."""
    cfg = replace(VARIANTS[RECIPE], fixed_rounds=ROUNDS)
    if cfg != VARIANTS["cs02_fix1000"]:
        raise AssertionError("final config drifted from the cs02_fix1000 variant in exp_runner.py")
    return cfg


def lgb_params(cfg: Config, lr: float, seed: int, n_jobs: int) -> dict:
    """LightGBM arguments exactly as exp_runner.fit_variant builds them for a log-target, fixed-round variant."""
    return dict(objective=cfg.objective, n_estimators=int(cfg.fixed_rounds), learning_rate=lr, num_leaves=cfg.num_leaves,
                min_child_samples=cfg.min_child, subsample=0.85, subsample_freq=cfg.subsample_freq, colsample_bytree=cfg.colsample,
                reg_lambda=cfg.reg_lambda, extra_trees=cfg.extra_trees, random_state=seed, n_jobs=n_jobs, verbosity=-1)


def eligible_origins(manifest: pd.DataFrame, forecast_origin: pd.Timestamp) -> pd.DataFrame:
    """Training origins whose 42-day label window ends on or before the forecast origin (same rule as exp_runner.load_origins)."""
    return manifest[manifest["target_end"].le(forecast_origin)].sort_values("origin").reset_index(drop=True)


def assert_labels_before(eligible: pd.DataFrame, first_target_date: pd.Timestamp) -> None:
    if eligible.empty:
        raise AssertionError("no eligible training origins")
    last_label = eligible["target_end"].max()
    if not last_label < first_target_date:
        raise AssertionError(f"training labels reach {last_label.date()}, not before the first target date {first_target_date.date()}")


def zero_closed(pred: np.ndarray, is_open: np.ndarray) -> np.ndarray:
    """New array: negatives clipped to 0 and every closed row exactly 0."""
    return np.where(np.asarray(is_open) == 1, np.clip(np.asarray(pred, dtype=float), 0, None), 0.0)


def log_mean_blend(preds: list[np.ndarray], is_open: np.ndarray) -> np.ndarray:
    """Equal-weight mean of log1p predictions, back to volume; closed rows 0."""
    if not preds:
        raise ValueError("need at least one member")
    mean_log = np.mean([np.log1p(np.clip(np.asarray(p, dtype=float), 0, None)) for p in preds], axis=0)
    return zero_closed(np.expm1(mean_log), is_open)


def combine_members(global_preds: list[np.ndarray], pertier_preds: list[np.ndarray], is_open: np.ndarray, mode: str = "member") -> np.ndarray:
    """member: every model (seeds + per-tier sets) is one equal member of a single log mean.
    half: log mean of the global seeds and log mean of the per-tier sets, then 50/50."""
    if mode not in BLEND_MODES:
        raise ValueError(f"blend mode must be one of {BLEND_MODES}")
    if not pertier_preds:
        return log_mean_blend(global_preds, is_open)
    if mode == "member":
        return log_mean_blend([*global_preds, *pertier_preds], is_open)
    return log_mean_blend([log_mean_blend(global_preds, is_open), log_mean_blend(pertier_preds, is_open)], is_open)


def member_entry(path: Path, best_iteration: int, **extra) -> dict:
    """Manifest block per saved model, same keys as the control manifest (file, sha256, best_iteration, log_offset_c)."""
    return {"file": Path(path).name, "sha256": sha256_file(path), "best_iteration": int(best_iteration), "log_offset_c": 0.0, **extra}


def safe_name(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_") or "x"


def submission_frame(ids: pd.Series, volumes: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({"Id": ids.to_numpy(), "OrderVolume": volumes})


# ---------------------------------------------------------------- data
def load_target_frame(root: Path, dry_run_fold: int | None) -> pd.DataFrame:
    dates = ["Date", "ForecastOrigin", "HistoryStartDate", "HistoryLastDate"]
    if dry_run_fold is None:
        return pd.read_csv(root / "data/processed/test_forecast_features.csv.gz", parse_dates=dates)
    path = next((root / "data/processed/folds").glob(f"fold_{dry_run_fold}_*.csv.gz"))
    return pd.read_csv(path, parse_dates=dates)


def add_hub_encoding(train: pd.DataFrame, frame: pd.DataFrame, seed: int) -> None:
    """HubID_TE as in exp_runner.prepare_fold: cross-fitted on fitted rows, plain transform elsewhere (adds a float32 column in place)."""
    from sklearn.preprocessing import TargetEncoder
    mask = train["IsOpen"].eq(1) & train["ActualOrderVolume"].gt(0)
    enc = TargetEncoder(smooth="auto", random_state=seed)
    cross = enc.fit_transform(train.loc[mask, ["HubID"]], np.log1p(train.loc[mask, "ActualOrderVolume"]))
    train["HubID_TE"] = enc.transform(train[["HubID"]]).ravel()
    train.loc[mask, "HubID_TE"] = cross.ravel()
    frame["HubID_TE"] = enc.transform(frame[["HubID"]]).ravel()
    for f in (train, frame):
        f["HubID_TE"] = f["HubID_TE"].astype("float32")


# ---------------------------------------------------------------- fitting
def fit_and_save(x_train: pd.DataFrame, y_train: np.ndarray, params: dict, path: Path):
    import lightgbm as lgb
    model = lgb.LGBMRegressor(**params)
    model.fit(x_train, y_train)
    model.booster_.save_model(str(path))
    return model


def verify_saved(model, path: Path, x_probe: pd.DataFrame) -> None:
    """The saved file must reproduce the in-memory model's raw scores on a probe slice."""
    import lightgbm as lgb
    probe = x_probe.head(2000)
    if probe.empty:
        return
    live = model.predict(probe, raw_score=True)
    saved = lgb.Booster(model_file=str(path)).predict(probe, raw_score=True)
    if not np.allclose(live, saved, rtol=0, atol=1e-9):
        raise AssertionError(f"saved model {path.name} does not reproduce the in-memory predictions")


def predict_volume(model, x_open: pd.DataFrame, n_rows: int, open_idx: np.ndarray) -> np.ndarray:
    """cs02 uses the log1p target: volume = expm1(raw score); closed rows stay 0."""
    out = np.zeros(n_rows)
    out[open_idx] = decode_prediction(model.predict(x_open, raw_score=True), np.ones(len(x_open)), "log")
    return out


def fit_pertier(train: pd.DataFrame, positive: pd.Series, frame: pd.DataFrame, open_idx: np.ndarray, feats: list[str], cats: set[str],
                params: dict, seed: int, out_dir: Path, fallback: np.ndarray, log=print) -> tuple[np.ndarray, dict]:
    """One cs02 model per large AssortmentTier (>= 20,000 training rows), all small tiers pooled in one model.
    Open rows not covered by any group (unseen/missing tier, empty training set) take the global model's prediction `fallback`."""
    from pergroup import group_values
    big, small = group_values({"train": train, "valid": frame}, GROUP_COLUMN)
    jobs = [([g], safe_name(g)) for g in big] + ([(small, "small")] if small else [])
    pred = np.zeros(len(frame))
    covered = np.zeros(len(frame), dtype=bool)
    entries: dict[str, dict] = {}
    is_open = np.zeros(len(frame), dtype=bool)
    is_open[open_idx] = True
    for values, label in jobs:
        rows = is_open & frame[GROUP_COLUMN].isin(values).to_numpy()
        tr = positive & train[GROUP_COLUMN].isin(values)
        if not rows.any() or not tr.any():
            log(f"  tier {values}: no {'prediction rows' if not rows.any() else 'training rows'}, skipped")
            continue
        x_tr = prep(train.loc[tr], feats, cats)
        path = out_dir / f"pertier_{label}_s{seed}.txt"
        t0 = time.perf_counter()
        model = fit_and_save(x_tr, np.log1p(train.loc[tr, "ActualOrderVolume"].to_numpy()), params, path)
        x_pred = prep(frame.loc[rows], feats, cats)
        verify_saved(model, path, x_pred)
        pred[rows] = decode_prediction(model.predict(x_pred, raw_score=True), np.ones(int(rows.sum())), "log")
        covered |= rows
        entries[f"pertier_{label}_s{seed}"] = member_entry(path, params["n_estimators"], seed=seed, group=[str(v) for v in values], train_rows=int(tr.sum()))
        log(f"  tier {values}: train_rows {int(tr.sum())}, fit {time.perf_counter() - t0:.0f}s")
        del x_tr
    missing = is_open & ~covered
    if missing.any():
        pred[missing] = fallback[missing]
        log(f"  {int(missing.sum())} open rows had no tier model; global prediction used")
    return zero_closed(pred, frame["IsOpen"].to_numpy()), {"entries": entries, "fallback_rows": int(missing.sum())}


def run(args: argparse.Namespace) -> int:
    root = args.proj_root.resolve()
    sys.path.insert(0, str(root))
    from src.eda_pipeline import RAW_FILES, Paths
    from src.model_pipeline import CATEGORICAL_COLUMNS, feature_columns

    dry = args.dry_run_fold is not None
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    stem = f"dry_run_fold{args.dry_run_fold}" if dry else "final"
    manifest_path = out / (f"{stem}_manifest.json" if dry else "final_manifest.json")
    if manifest_path.exists() and not args.overwrite:
        raise SystemExit(f"{manifest_path.name} already exists in {out}; use a new out_dir or --overwrite")
    seeds = [int(s) for s in args.seeds.split(",")]
    pertier_seeds = [int(s) for s in args.pertier_seeds.split(",")] if args.pertier_seeds else seeds[:1]
    if args.with_pertier and not set(pertier_seeds) <= set(seeds):
        raise SystemExit("--pertier-seeds must be a subset of --seeds")
    cfg = final_config()
    timings: dict[str, float] = {}
    clock = time.perf_counter

    frame = load_target_frame(root, args.dry_run_fold)
    origin = pd.Timestamp(frame["ForecastOrigin"].iloc[0])
    first_target = pd.Timestamp(frame["Date"].min())
    columns = [c for c in feature_columns(frame, "all") if c != "HubID_TargetEncoded"]
    manifest_csv = root / "data/processed/training_origin_manifest.csv"
    origins = pd.read_csv(manifest_csv, parse_dates=["origin", "target_start", "target_end"])
    eligible = eligible_origins(origins, origin)
    assert_labels_before(eligible, first_target)
    print(f"{'DRY RUN fold ' + str(args.dry_run_fold) if dry else 'FINAL'}: origin {origin.date()}, targets from {first_target.date()}, "
          f"{len(eligible)} training origins (labels end {eligible['target_end'].max().date()})", flush=True)

    if args.verify_origins:
        t0 = clock()
        bad = [r.path for r in eligible.itertuples() if sha256_file(root / r.path) != r.sha256]
        if bad:
            raise SystemExit(f"origin files differ from the manifest SHA-256: {bad[:3]}")
        timings["verify_origins"] = clock() - t0
        print(f"origin files verified against the manifest in {timings['verify_origins']:.0f}s", flush=True)

    if args.preflight:
        preflight(root, frame, columns, eligible, dry, Paths(root), RAW_FILES)
        return 0
    t0 = clock()
    train = load_origins(root, origin, columns).reset_index(drop=True).copy()  # copy consolidates the many column blocks
    timings["load_training"] = clock() - t0
    print(f"origins loaded in {timings['load_training']:.0f}s", flush=True)
    t0 = clock()
    add_hub_encoding(train, frame, seeds[0])
    train = train.copy()
    timings["hub_encoding"] = clock() - t0
    print(f"hub encoding in {timings['hub_encoding']:.0f}s", flush=True)
    t0 = clock()
    feats = [*columns, "HubID_TE"]
    cats = set(CATEGORICAL_COLUMNS)
    positive = train["IsOpen"].eq(1) & train["ActualOrderVolume"].gt(0)
    x_train = prep(train.loc[positive], feats, cats)
    y_train = np.log1p(train.loc[positive, "ActualOrderVolume"].to_numpy())
    open_idx = np.flatnonzero(frame["IsOpen"].eq(1).to_numpy())
    x_open = prep(frame.iloc[open_idx], feats, cats)
    is_open = frame["IsOpen"].to_numpy()
    timings["prepare_matrices"] = clock() - t0
    print(f"train rows {int(positive.sum())} (of {len(train)} loaded), {len(feats)} columns, {len(open_idx)} open prediction rows", flush=True)

    manifest: dict[str, dict] = {}
    global_preds: dict[int, np.ndarray] = {}
    for seed in seeds:
        params = lgb_params(cfg, args.lr if args.lr else LEARNING_RATE, seed, args.n_jobs)
        path = out / f"lightgbm_s{seed}.txt"
        t0 = clock()
        model = fit_and_save(x_train, y_train, params, path)
        timings[f"fit_lightgbm_s{seed}"] = clock() - t0
        verify_saved(model, path, x_open)
        pred = zero_closed(predict_volume(model, x_open, len(frame), open_idx), is_open)
        global_preds[seed] = pred
        np.save(out / f"test_pred_lightgbm_s{seed}.npy", pred)
        manifest[f"lightgbm_s{seed}"] = member_entry(path, params["n_estimators"], seed=seed, train_rows=int(positive.sum()))
        print(f"lightgbm seed {seed}: fit {timings[f'fit_lightgbm_s{seed}']:.0f}s", flush=True)
        del model

    pertier_preds: dict[int, np.ndarray] = {}
    fallback_rows = 0
    if args.with_pertier:
        for seed in pertier_seeds:
            params = lgb_params(cfg, args.lr if args.lr else LEARNING_RATE, seed, args.n_jobs)
            t0 = clock()
            print(f"per-{GROUP_COLUMN} models, seed {seed}", flush=True)
            pred, info = fit_pertier(train, positive, frame, open_idx, feats, cats, params, seed, out, global_preds[seed])
            timings[f"fit_pertier_s{seed}"] = clock() - t0
            pertier_preds[seed] = pred
            fallback_rows += info["fallback_rows"]
            manifest.update(info["entries"])
            np.save(out / f"test_pred_pertier_s{seed}.npy", pred)
            print(f"per-tier seed {seed}: {timings[f'fit_pertier_s{seed}']:.0f}s", flush=True)

    final = combine_members(list(global_preds.values()), list(pertier_preds.values()), is_open, args.blend_mode)
    if dry:
        pred_path = out / f"{stem}_predictions.csv"
        table = submission_frame(pd.Series(np.arange(1, len(frame) + 1)), final)
        table = pd.concat([table, frame[["HubID", "Date", "Horizon", "IsOpen", "ActualOrderVolume"]].reset_index(drop=True)], axis=1)
        scores = fold_scores(frame["ActualOrderVolume"].to_numpy(float), final, is_open)
        print(f"DRY RUN fold {args.dry_run_fold} RMSLE_all {scores['rmsle_all']:.5f}  excl_open0 {scores['rmsle_excl_open0']:.5f}  bias_open {scores['bias_open']:+.5f}", flush=True)
        for seed, p in global_preds.items():
            s = fold_scores(frame["ActualOrderVolume"].to_numpy(float), p, is_open)
            print(f"  global seed {seed}: RMSLE_all {s['rmsle_all']:.5f}", flush=True)
        if pertier_preds:
            g = combine_members(list(global_preds.values()), [], is_open)
            print(f"  global-only blend RMSLE_all {fold_scores(frame['ActualOrderVolume'].to_numpy(float), g, is_open)['rmsle_all']:.5f}", flush=True)
    else:
        sample = pd.read_csv(Paths(root).raw / RAW_FILES["submission"])
        table = submission_frame(frame["Id"], final)
        if len(table) != len(sample) or list(table["Id"]) != list(sample["Id"]):
            raise AssertionError("submission rows or Id order differ from the sample submission")
        pred_path = out / "submission_final.csv"
        scores = None
    if not np.isfinite(final).all() or (final < 0).any() or (final[is_open == 0] != 0).any():
        raise AssertionError("final predictions contain NaN/inf, negatives or non-zero closed rows")
    table.to_csv(pred_path, index=False)

    manifest["_run"] = {
        "mode": f"dry_run_fold_{args.dry_run_fold}" if dry else "final",
        "seed": seeds[0], "seeds": seeds, "train_rows": int(positive.sum()), "n_columns": len(feats),
        "training_origins": len(eligible), "last_label_date": str(eligible["target_end"].max().date()),
        "forecast_origin": str(origin.date()), "first_target_date": str(first_target.date()),
        "timings_s": timings, "submission_sha256": sha256_file(pred_path), "predictions_file": pred_path.name,
        "config": {"recipe": RECIPE, "variant": "cs02_fix1000", **cfg.__dict__, "learning_rate": args.lr or LEARNING_RATE,
                   "lgb_params": {k: v for k, v in lgb_params(cfg, args.lr or LEARNING_RATE, seeds[0], args.n_jobs).items() if k not in ("random_state", "n_jobs")}},
        "blend": {"mode": args.blend_mode, "with_pertier": bool(args.with_pertier), "pertier_seeds": pertier_seeds if args.with_pertier else [],
                  "pertier_group_column": GROUP_COLUMN, "pertier_fallback_rows": fallback_rows},
        "inputs": {"target_frame": "fold" if dry else "test_forecast_features", "training_origin_manifest_sha256": sha256_file(manifest_csv)},
        "versions": _versions(),
    }
    if scores:
        manifest["_run"]["dry_run_scores"] = scores
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
    print(json.dumps({k: manifest["_run"][k] for k in ("mode", "train_rows", "submission_sha256")}, indent=2), flush=True)
    return 0


def preflight(root: Path, frame: pd.DataFrame, columns: list[str], eligible: pd.DataFrame, dry: bool, paths, raw_files: dict) -> None:
    """Cheap go/no-go checks, no training: schema of the origin tables, grid size, Id order against the sample submission."""
    import pyarrow.parquet as pq
    if len(frame) != 46_830 or frame["HubID"].nunique() != 1_115 or frame["Date"].nunique() != 42:
        raise AssertionError("target frame is not a complete 1115 x 42 grid")
    for row in (eligible.iloc[0], eligible.iloc[-1]):
        names = set(pq.read_schema(root / row["path"]).names)
        missing = [c for c in [*columns, "ActualOrderVolume", "IsOpen", "HubID"] if c not in names]
        if missing:
            raise AssertionError(f"{row['path']} lacks feature columns {missing[:5]}")
    if not dry:
        sample = pd.read_csv(paths.raw / raw_files["submission"])
        if list(frame["Id"]) != list(sample["Id"]):
            raise AssertionError("test frame Id order differs from the sample submission")
    print(f"PREFLIGHT OK: {len(columns)} feature columns present in the origin tables, {len(eligible)} origins, "
          f"{'fold' if dry else 'test'} frame {len(frame)} rows" + ("" if dry else ", Id order equals the sample submission"), flush=True)


def _versions() -> dict:
    import lightgbm
    import sklearn
    return {"python": sys.version.split()[0], "lightgbm": lightgbm.__version__, "numpy": np.__version__, "pandas": pd.__version__, "scikit-learn": sklearn.__version__}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("proj_root", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    ap.add_argument("--n-jobs", type=int, default=16)
    ap.add_argument("--with-pertier", action="store_true", help="add the per-AssortmentTier model set as an extra blend member")
    ap.add_argument("--pertier-seeds", default="", help="comma list, subset of --seeds (default: the first seed only)")
    ap.add_argument("--blend-mode", choices=BLEND_MODES, default="member", help="member: seeds and per-tier sets in one equal log mean; half: 50/50 of the two averages")
    ap.add_argument("--dry-run-fold", type=int, default=None, help="rehearse on validation fold N instead of the real test frame")
    ap.add_argument("--lr", type=float, default=LEARNING_RATE)
    ap.add_argument("--no-verify-origins", dest="verify_origins", action="store_false", help="skip SHA-256 check of origin files against the manifest")
    ap.add_argument("--preflight", action="store_true", help="check inputs and exit without training")
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
