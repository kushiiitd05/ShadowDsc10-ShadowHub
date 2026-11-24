"""Control rerun of the frozen 0.06976 pipeline, with model saving and timing.

Same logic as 2_future_work_progress/kaggle_modelling/scripts/kaggle_end_to_end_source.py
(including its known flaws F2/F5/F6, kept on purpose so this is a faithful control).
Differences: paths come from argv, processed data may be reused instead of rebuilt,
every fitted model is saved with a SHA-256, and wall-clock timings are recorded.

Usage: python control_rerun.py <proj_root> <out_dir> [--rebuild]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("proj_root", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--rebuild", action="store_true", help="rebuild processed data from raw (about 50 min)")
    args = parser.parse_args()

    root = args.proj_root.resolve()
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(root))

    import src.eda_pipeline as edap
    from sklearn.preprocessing import TargetEncoder
    from src.eda_pipeline import Paths
    from src.model_pipeline import (
        feature_columns,
        fit_catboost,
        fit_lightgbm,
        fit_lightgbm_tweedie,
        load_origin_training,
        predict_model,
    )

    timings: dict[str, float] = {}
    clock = time.perf_counter

    if args.rebuild:
        for step in ("preprocess", "build-training-origins", "build-folds"):
            start = clock()
            edap.main([step, "--root", str(root)])
            timings[f"build_{step}"] = clock() - start

    paths = Paths(root)
    test = pd.read_csv(
        paths.processed / "test_forecast_features.csv.gz",
        parse_dates=["Date", "ForecastOrigin", "HistoryStartDate", "HistoryLastDate"],
    )
    test_origin = pd.Timestamp(test["ForecastOrigin"].iloc[0])
    columns = feature_columns(test, "all")

    start = clock()
    train = load_origin_training(paths, test_origin, columns)
    timings["load_training"] = clock() - start
    print("train rows:", len(train), "n feature cols:", len(columns), flush=True)

    encoder = TargetEncoder(smooth="auto")
    mask = train["IsOpen"].eq(1)
    encoder.fit(train.loc[mask, ["HubID"]], np.log1p(train.loc[mask, "ActualOrderVolume"]))
    train["HubID_TargetEncoded"] = encoder.transform(train[["HubID"]])
    test["HubID_TargetEncoded"] = encoder.transform(test[["HubID"]])
    columns = [*columns, "HubID_TargetEncoded"]

    valid = train.sample(frac=0.05, random_state=edap.SEED)
    fitters = {"catboost": fit_catboost, "lightgbm": fit_lightgbm, "lightgbm_tweedie": fit_lightgbm_tweedie}

    ensemble_log = np.zeros(len(test))
    manifest: dict[str, dict] = {}
    for name, fitter in fitters.items():
        print("fitting", name, flush=True)
        start = clock()
        model = fitter(train, valid, columns)
        timings[f"fit_{name}"] = clock() - start

        model_path = out / (f"{name}.cbm" if name == "catboost" else f"{name}.txt")
        if name == "catboost":
            model.save_model(str(model_path))
            best_iteration = int(model.get_best_iteration())
        else:
            model.booster_.save_model(str(model_path))
            best_iteration = int(model.best_iteration_)
        manifest[name] = {"file": model_path.name, "sha256": sha256(model_path), "best_iteration": best_iteration}

        train_pred = predict_model(model, train, columns, name)
        offset = float(np.mean(np.log1p(train.loc[mask, "ActualOrderVolume"]) - np.log1p(train_pred[mask.to_numpy()])))
        test_pred = predict_model(model, test, columns, name)
        corrected = np.where(test["IsOpen"].eq(1), np.expm1(np.log1p(test_pred) + offset), 0.0)
        ensemble_log += np.log1p(corrected)
        manifest[name]["log_offset_c"] = offset
        np.save(out / f"test_pred_{name}.npy", corrected)
        print(name, "done in", round(timings[f"fit_{name}"], 1), "s; c =", offset, flush=True)

    ensemble_log /= len(fitters)
    final = np.clip(np.where(test["IsOpen"].eq(1), np.expm1(ensemble_log), 0.0), 0, None)
    submission = pd.DataFrame({"Id": test["Id"], "OrderVolume": final})
    sample = pd.read_csv(paths.raw / "sample_submission.csv")
    assert len(submission) == len(sample), "row count mismatch vs sample_submission"
    assert list(submission["Id"]) == list(sample["Id"]), "Id order mismatch vs sample_submission"
    submission.to_csv(out / "submission_control.csv", index=False)

    manifest["_run"] = {
        "seed": edap.SEED,
        "train_rows": int(len(train)),
        "n_columns": len(columns),
        "timings_s": timings,
        "submission_sha256": sha256(out / "submission_control.csv"),
    }
    (out / "control_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest["_run"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
