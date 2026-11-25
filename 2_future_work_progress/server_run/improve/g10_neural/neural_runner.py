"""Stage G10 runner: neural / foundation-model pilot members (TiDE, NHITS via neuralforecast; TabPFN-TS).

Usage: python neural_runner.py <proj_root> <out_dir> --model tide --folds 4,3,2,1 --seed 20260807 [--n-hubs 100]

Each fold: history = rows dated <= the fold's forecast origin only (the last observed day), target log1p(orders),
known-future covariates = open / promo / holiday / school flags + calendar for the 42 forecast days, closed forecast days forced to 0.
Writes <out_dir>/oof_<model>_s<seed>.csv.gz (fold,HubID,Date,Horizon,IsOpen,ActualOrderVolume,pred; readable by blend.py)
and appends one row to <out_dir>/experiments.csv (same schema as g4_runner.py).
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
from helpers import fold_scores  # noqa: E402
from neural_prep import build_future_frame, build_panel, build_static, forecast_to_frame, futr_columns, impute_closed  # noqa: E402

HORIZON = 42
MODELS = ("tide", "nhits", "tabpfnts", "tabpfn")
RECORD_COLUMNS = ["date", "variant", "seed", "model", "config", "config_hash", "lr", "rounds", "n_jobs",
                  "mean_all", "mean_excl_open0", "mean_bias_open", "fit_seconds"]
OOF_COLUMNS = ["fold", "HubID", "Date", "Horizon", "IsOpen", "ActualOrderVolume", "pred"]
BASE_COLUMNS = ["HubID", "Date", "OrderVolume", "IsOpen", "PromoActive", "RegionalHoliday", "SchoolClosureFlag"]


def build_record(variant: str, rows: list[dict], seed: int, lr: float, steps: int, cfg: dict) -> dict:
    cfg_json = json.dumps(cfg, sort_keys=True, default=str)
    record = {"date": date.today().isoformat(), "variant": variant, "seed": seed, "model": variant, "config": cfg_json,
              "config_hash": hashlib.sha256(cfg_json.encode()).hexdigest()[:10], "lr": lr, "rounds": steps, "n_jobs": 1,
              "mean_all": float(np.mean([r["rmsle_all"] for r in rows])),
              "mean_excl_open0": float(np.mean([r["rmsle_excl_open0"] for r in rows])),
              "mean_bias_open": float(np.mean([r["bias_open"] for r in rows])),
              "fit_seconds": round(sum(r["fit_seconds"] for r in rows), 1)}
    for r in rows:
        record[f"fold{r['fold']}_all"] = r["rmsle_all"]
        record[f"fold{r['fold']}_excl_open0"] = r["rmsle_excl_open0"]
        record[f"fold{r['fold']}_best_iter"] = steps
    return record


def append_record(out_dir: Path, record: dict) -> None:
    path = out_dir / "experiments.csv"
    frame = pd.DataFrame([record])
    (pd.concat([pd.read_csv(path), frame], ignore_index=True) if path.exists() else frame).to_csv(path, index=False)


def load_fold(root: Path, fold: int, n_hubs: int | None) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """(history rows up to the origin, fold rows to forecast, origin). n_hubs limits to the first n hubs (smoke runs)."""
    proc = root / "data/processed"
    valid = pd.read_csv(next((proc / "folds").glob(f"fold_{fold}_*.csv.gz")), parse_dates=["Date", "ForecastOrigin"])
    origin = pd.Timestamp(valid["ForecastOrigin"].iloc[0])
    base = pd.read_csv(proc / "base_train.csv.gz", usecols=BASE_COLUMNS, parse_dates=["Date"])
    if n_hubs:
        keep = np.sort(valid["HubID"].unique())[:n_hubs]
        valid, base = valid[valid["HubID"].isin(keep)], base[base["HubID"].isin(keep)]
    return base[base["Date"] <= origin].copy(), valid.reset_index(drop=True), origin


def nf_models(name: str, seed: int, steps: int, lr: float, input_size: int, static_cols: list[str], scaler: str):
    from neuralforecast.losses.pytorch import MSE
    from neuralforecast.models import NHITS, TiDE

    common = dict(h=HORIZON, input_size=input_size, loss=MSE(), max_steps=steps, learning_rate=lr, scaler_type=scaler,
                  futr_exog_list=futr_columns(), stat_exog_list=static_cols, random_seed=seed, batch_size=32,
                  windows_batch_size=1024, val_check_steps=10 ** 9,
                  enable_progress_bar=False, logger=False, enable_checkpointing=False, accelerator="gpu", devices=1)
    if name == "tide":
        return TiDE(hidden_size=256, decoder_output_dim=32, temporal_decoder_dim=128, dropout=0.1, num_encoder_layers=2,
                    num_decoder_layers=2, temporal_width=4, **common)
    return NHITS(stack_types=["identity"] * 3, n_blocks=[1, 1, 1], mlp_units=[[512, 512]] * 3, n_pool_kernel_size=[8, 4, 1],
                 n_freq_downsample=[14, 7, 1], dropout_prob_theta=0.1, **common)


def predict_nf(name: str, hist: pd.DataFrame, valid: pd.DataFrame, origin: pd.Timestamp, args) -> pd.DataFrame:
    from neuralforecast import NeuralForecast

    panel, static, future = build_panel(hist, origin), build_static(valid), build_future_frame(valid, origin, HORIZON)
    if args.impute:
        panel = impute_closed(panel)
    model = nf_models(name, args.seed, args.steps, args.lr, args.input_size, [c for c in static.columns if c != "unique_id"], args.scaler)
    nf = NeuralForecast(models=[model], freq="D")
    nf.fit(df=panel, static_df=static, val_size=0)
    fc = nf.predict(futr_df=future)
    if "unique_id" not in fc.columns:
        fc = fc.reset_index()
    col = model.__class__.__name__  # the forecast column is named after the model; never guess (an index column once slipped in)
    assert col in fc.columns, f"forecast column {col} missing from {list(fc.columns)}"
    return fc[["unique_id", "ds", col]].rename(columns={col: "log_pred"})


def predict_member(name: str, hist: pd.DataFrame, valid: pd.DataFrame, origin: pd.Timestamp, args) -> pd.DataFrame:
    if name in ("tide", "nhits"):
        return predict_nf(name, hist, valid, origin, args)
    from tabpfn_members import predict_tabpfn, predict_tabpfn_ts  # noqa: E402  (separate venv, imported lazily)

    return (predict_tabpfn_ts if name == "tabpfnts" else predict_tabpfn)(hist, valid, origin, args)


def run_fold(root: Path, fold: int, args) -> tuple[pd.DataFrame, dict]:
    t0 = time.perf_counter()
    hist, valid, origin = load_fold(root, fold, args.n_hubs)
    assert hist["Date"].max() <= origin < valid["Date"].min()  # leak guard: nothing at or after the first forecast day
    print(f"fold {fold}: origin {origin.date()} history rows {len(hist)} hubs {valid['HubID'].nunique()} loaded in {time.perf_counter() - t0:.0f}s", flush=True)
    t1 = time.perf_counter()
    fc = predict_member(args.model, hist, valid, origin, args)
    fit_seconds = time.perf_counter() - t1
    oof = forecast_to_frame(fc, valid, "log_pred")
    scores = fold_scores(oof["ActualOrderVolume"].to_numpy(float), oof["pred"].to_numpy(float), oof["IsOpen"].to_numpy())
    scores.update(fold=fold, fit_seconds=round(fit_seconds, 1))
    oof["fold"] = fold
    return oof[OOF_COLUMNS], scores


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("proj_root", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--model", required=True, choices=MODELS)
    ap.add_argument("--folds", default="4,3,2,1")
    ap.add_argument("--seed", type=int, default=20260807)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--input-size", type=int, default=180)
    ap.add_argument("--scaler", default="standard")
    ap.add_argument("--impute", type=int, default=1, help="1: interpolate closed-day targets in the history (closed days are re-zeroed at prediction)")
    ap.add_argument("--n-hubs", type=int, default=None, help="smoke: only the first n hubs")
    ap.add_argument("--context", type=int, default=365, help="tabpfn-ts: context days per hub")
    ap.add_argument("--no-record", action="store_true", help="do not append to experiments.csv")
    args = ap.parse_args()
    root = args.proj_root.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    oof_path = args.out_dir / f"oof_{args.model}_s{args.seed}.csv.gz"
    oofs: list[pd.DataFrame] = []
    rows: list[dict] = []
    for f in [int(x) for x in args.folds.split(",")]:
        oof, scores = run_fold(root, f, args)
        oofs.append(oof)
        rows.append(scores)
        print(args.model, "fold", f, json.dumps(scores), flush=True)
        pd.concat(oofs, ignore_index=True).to_csv(oof_path, index=False, compression="gzip")  # keep partial results if a later fold dies
        gc.collect()
    if not args.no_record:
        cfg = {k: v for k, v in vars(args).items() if k not in ("proj_root", "out_dir")}
        append_record(args.out_dir, build_record(args.model, rows, args.seed, args.lr, args.steps, cfg))
    print("DONE", args.model, round(float(np.mean([r["rmsle_all"] for r in rows])), 5), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
