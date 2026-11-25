"""Stage G7 runner: zero-shot Chronos-2 member for the blend (no training, history strictly up to each fold's origin).

Usage: python chronos_runner.py <proj_root> <out_dir> --folds 4,3,2,1 --seed 20260807 [--covariates] [--limit-hubs 100]

Writes <out_dir>/oof_<name>_s<seed>.csv.gz (fold,HubID,Date,Horizon,IsOpen,ActualOrderVolume,pred; readable by blend.py),
name = chronos (univariate) or chronos_cov (IsOpen/PromoActive/RegionalHoliday/SchoolClosureFlag as known covariates),
and appends one row to <out_dir>/experiments.csv (same schema as the other runners).
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from chronos_prep import ALL_COVARIATES, OOF_COLUMNS, forecast_fold  # noqa: E402
from helpers import fold_scores  # noqa: E402

MODEL_ID = "amazon/chronos-2"
RECORD_COLUMNS = ["date", "variant", "seed", "model", "config", "config_hash", "lr", "rounds", "n_jobs",
                  "mean_all", "mean_excl_open0", "mean_bias_open", "fit_seconds"]
FOLD_COLUMNS = ["HubID", "Date", "Horizon", "IsOpen", "ForecastOrigin", *ALL_COVARIATES]


def build_record(variant: str, rows: list[dict], seed: int, cfg_json: str) -> dict:
    record = {
        "date": date.today().isoformat(), "variant": variant, "seed": seed, "model": "chronos-2-zeroshot",
        "config": cfg_json, "config_hash": hashlib.sha256(cfg_json.encode()).hexdigest()[:10],
        "lr": 0.0, "rounds": 0, "n_jobs": 1,
        "mean_all": float(np.mean([r["rmsle_all"] for r in rows])),
        "mean_excl_open0": float(np.mean([r["rmsle_excl_open0"] for r in rows])),
        "mean_bias_open": float(np.mean([r["bias_open"] for r in rows])),
        "fit_seconds": round(sum(r["fit_seconds"] for r in rows), 1),
    }
    for r in rows:
        record[f"fold{r['fold']}_all"] = r["rmsle_all"]
        record[f"fold{r['fold']}_excl_open0"] = r["rmsle_excl_open0"]
        record[f"fold{r['fold']}_best_iter"] = 0
    return record


def append_record(out_dir: Path, record: dict) -> None:
    path = out_dir / "experiments.csv"
    frame = pd.DataFrame([record])
    (pd.concat([pd.read_csv(path), frame], ignore_index=True) if path.exists() else frame).to_csv(path, index=False)


def load_fold(root: Path, fold_number: int) -> pd.DataFrame:
    path = next((root / "data/processed/folds").glob(f"fold_{fold_number}_*.csv.gz"))
    return pd.read_csv(path, usecols=FOLD_COLUMNS, parse_dates=["Date", "ForecastOrigin"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("proj_root", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--folds", default="4,3,2,1")
    ap.add_argument("--seed", type=int, default=20260807)
    ap.add_argument("--covariates", action="store_true", help="pass the known future calendar flags as covariates")
    ap.add_argument("--limit-hubs", type=int, default=0, help="smoke test: only the first N hubs")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--model-id", default=MODEL_ID)
    args = ap.parse_args()
    import torch
    from chronos import Chronos2Pipeline

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but CUDA is not available")
    torch.manual_seed(args.seed)
    root = args.proj_root.resolve()
    variant = "chronos_cov" if args.covariates else "chronos"
    covariates = ALL_COVARIATES if args.covariates else ()
    raw = pd.read_csv(next(root.glob("*/orders_train.csv")), parse_dates=["Date"])
    pipeline = Chronos2Pipeline.from_pretrained(args.model_id, device_map=args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    oof_path = args.out_dir / f"oof_{variant}_s{args.seed}.csv.gz"
    hubs = sorted(raw["HubID"].unique())[: args.limit_hubs] if args.limit_hubs else None
    oofs: list[pd.DataFrame] = []
    rows: list[dict] = []
    for f in [int(x) for x in args.folds.split(",")]:
        t0 = time.perf_counter()
        fold = load_fold(root, f)
        oof = forecast_fold(pipeline, raw, fold, f, covariates=covariates, batch_size=args.batch_size, hubs=hubs)
        seconds = round(time.perf_counter() - t0, 1)
        scores = fold_scores(oof["ActualOrderVolume"].to_numpy(float), oof["pred"].to_numpy(float), oof["IsOpen"].to_numpy())
        scores.update(fold=f, fit_seconds=seconds, rows=len(oof))
        oofs.append(oof[OOF_COLUMNS])
        rows.append(scores)
        print(variant, "fold", f, json.dumps(scores), flush=True)
        pd.concat(oofs, ignore_index=True).to_csv(oof_path, index=False, compression="gzip")  # keep partial results if a later fold dies
        gc.collect()
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
            print(f"peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB", flush=True)
    cfg = {"model_id": args.model_id, "covariates": list(covariates), "quantile": 0.5, "limit_hubs": args.limit_hubs,
           "batch_size": args.batch_size, "context": "history <= ForecastOrigin"}
    append_record(args.out_dir, build_record(variant, rows, args.seed, json.dumps(cfg, sort_keys=True)))
    print("DONE", variant, round(float(np.mean([r["rmsle_all"] for r in rows])), 5), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
