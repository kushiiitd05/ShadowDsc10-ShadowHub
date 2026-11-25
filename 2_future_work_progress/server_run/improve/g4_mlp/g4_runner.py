"""Stage G4b runner: entity-embedding MLP member for the blend.

Usage: python g4_runner.py <proj_root> <out_dir> --folds 1,2,3,4 --seed 20260807 --epochs 40 --device cuda

Writes <out_dir>/oof_mlp_s<seed>.csv.gz (fold,HubID,Date,Horizon,IsOpen,ActualOrderVolume,pred; readable by blend.py)
and appends one row to <out_dir>/experiments.csv (same schema as exp_runner.py).
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from dataclasses import replace
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from helpers import fold_scores  # noqa: E402
from mlp_prep import fit_preprocessor, transform  # noqa: E402

VARIANT = "mlp"
RECORD_COLUMNS = ["date", "variant", "seed", "model", "config", "config_hash", "lr", "rounds", "n_jobs",
                  "mean_all", "mean_excl_open0", "mean_bias_open", "fit_seconds"]
OOF_COLUMNS = ["fold", "HubID", "Date", "Horizon", "IsOpen", "ActualOrderVolume", "pred"]


def volume_from_log(log_pred: np.ndarray, is_open: np.ndarray) -> np.ndarray:
    """log1p-space prediction -> order volume; closed rows are forced to 0."""
    return np.where(np.asarray(is_open) == 1, np.clip(np.expm1(log_pred), 0.0, None), 0.0)


def build_record(rows: list[dict], seed: int, lr: float, epochs: int, cfg_json: str) -> dict:
    record = {
        "date": date.today().isoformat(), "variant": VARIANT, "seed": seed, "model": "mlp",
        "config": cfg_json, "config_hash": hashlib.sha256(cfg_json.encode()).hexdigest()[:10],
        "lr": lr, "rounds": epochs, "n_jobs": 1,
        "mean_all": float(np.mean([r["rmsle_all"] for r in rows])),
        "mean_excl_open0": float(np.mean([r["rmsle_excl_open0"] for r in rows])),
        "mean_bias_open": float(np.mean([r["bias_open"] for r in rows])),
        "fit_seconds": round(sum(r["fit_seconds"] for r in rows), 1),
    }
    for r in rows:
        record[f"fold{r['fold']}_all"] = r["rmsle_all"]
        record[f"fold{r['fold']}_excl_open0"] = r["rmsle_excl_open0"]
        record[f"fold{r['fold']}_best_iter"] = r["best_epoch"]
    return record


def append_record(out_dir: Path, record: dict) -> None:
    path = out_dir / "experiments.csv"
    frame = pd.DataFrame([record])
    (pd.concat([pd.read_csv(path), frame], ignore_index=True) if path.exists() else frame).to_csv(path, index=False)


def run_fold(ctx: dict, cfg, seed: int, device: str, categorical: set[str], log: bool = True) -> tuple[pd.DataFrame, dict]:
    """Fit preprocessing + MLP on one prepared fold (harness masks: open and positive volume) and score the valid rows."""
    from mlp_model import predict_log, train_mlp

    train, hold, valid, feats = ctx["train"], ctx["hold"], ctx["valid"], list(ctx["columns"])

    def positive(frame: pd.DataFrame) -> np.ndarray:
        return (frame["IsOpen"].eq(1) & frame["ActualOrderVolume"].gt(0)).to_numpy()

    mt, mh = positive(train), positive(hold)
    t_frame, h_frame = train.loc[mt, feats], hold.loc[mh, feats]
    y_train = np.log1p(train.loc[mt, "ActualOrderVolume"].to_numpy(dtype="float64"))
    y_hold = np.log1p(hold.loc[mh, "ActualOrderVolume"].to_numpy(dtype="float64"))
    pre = fit_preprocessor(t_frame, feats, categorical)  # statistics from the training rows only
    xn, xc = transform(pre, t_frame)
    hn, hc = transform(pre, h_frame)
    del t_frame, h_frame
    gc.collect()
    result = train_mlp(pre, xn, xc, y_train, hn, hc, y_hold, cfg, seed, device, log=log)
    del xn, xc
    open_mask = valid["IsOpen"].eq(1).to_numpy()
    vn, vc = transform(pre, valid.loc[open_mask, feats])
    pred = np.zeros(len(valid))
    pred[open_mask] = volume_from_log(predict_log(result.model, vn, vc, device), np.ones(int(open_mask.sum())))
    scores = fold_scores(valid["ActualOrderVolume"].to_numpy(float), pred, valid["IsOpen"].to_numpy())
    scores.update(fold=ctx["fold"], best_epoch=result.best_epoch, fit_seconds=round(result.fit_seconds, 1),
                  train_rows=int(mt.sum()), hold_rmse_log=result.best_rmse)
    oof = valid[["HubID", "Date", "Horizon", "IsOpen", "ActualOrderVolume"]].copy()
    oof["pred"] = pred
    oof["fold"] = ctx["fold"]
    return oof[OOF_COLUMNS], scores


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("proj_root", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--folds", default="1,2,3,4")
    ap.add_argument("--seed", type=int, default=20260807)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--batch-size", type=int, default=4096)
    args = ap.parse_args()
    import torch
    from mlp_model import MlpConfig

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but CUDA is not available")
    root = args.proj_root.resolve()
    sys.path.insert(0, str(root))
    from exp_runner import prepare_fold
    from src.model_pipeline import CATEGORICAL_COLUMNS

    cfg = replace(MlpConfig(), epochs=args.epochs, lr=args.lr, batch_size=args.batch_size)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    oof_path = args.out_dir / f"oof_{VARIANT}_s{args.seed}.csv.gz"
    oofs: list[pd.DataFrame] = []
    rows: list[dict] = []
    for f in [int(x) for x in args.folds.split(",")]:
        t0 = time.perf_counter()
        ctx = prepare_fold(root, f, args.seed)
        print(f"fold {f}: loaded in {time.perf_counter() - t0:.0f}s, train rows {len(ctx['train'])}", flush=True)
        oof, scores = run_fold(ctx, cfg, args.seed, device, set(CATEGORICAL_COLUMNS))
        oofs.append(oof)
        rows.append(scores)
        print(VARIANT, "fold", f, json.dumps(scores), flush=True)
        pd.concat(oofs, ignore_index=True).to_csv(oof_path, index=False, compression="gzip")  # keep partial results if a later fold dies
        del ctx
        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
            print(f"peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB", flush=True)
    append_record(args.out_dir, build_record(rows, args.seed, args.lr, args.epochs, json.dumps(cfg.to_dict(), sort_keys=True)))
    print("DONE", VARIANT, round(float(np.mean([r["rmsle_all"] for r in rows])), 5), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
