"""Export the exact LightGBM input matrix of the forecast window, so the frozen boosters can be re-run anywhere.

It rebuilds the matrix the way final_retrain.run() does (same columns, same cross-fitted hub encoding with the first
seed, same prep), saves the open rows with their Id to a parquet file, and then checks that every frozen booster in
<freeze_dir> reproduces its saved forecast (test_pred_lightgbm_s<seed>.npy) from that file alone.

usage (from code/model, needs the processed origin tables):
    python g8_final/export_forecast_matrix.py <proj_root> <freeze_dir> <out.parquet> --seed 20260807
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
for _p in (_HERE, _HERE.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
from final_retrain import add_hub_encoding, load_target_frame  # noqa: E402
from exp_runner import load_origins, prep  # noqa: E402
from helpers import decode_prediction  # noqa: E402


def build_open_matrix(root: Path, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (x_open with an Id column first, the full forecast frame) exactly as the final retrain saw them."""
    sys.path.insert(0, str(root))
    from src.model_pipeline import CATEGORICAL_COLUMNS, feature_columns

    frame = load_target_frame(root, None)
    origin = pd.Timestamp(frame["ForecastOrigin"].iloc[0])
    columns = [c for c in feature_columns(frame, "all") if c != "HubID_TargetEncoded"]
    train = load_origins(root, origin, columns).reset_index(drop=True).copy()
    add_hub_encoding(train, frame, seed)
    feats = [*columns, "HubID_TE"]
    open_idx = np.flatnonzero(frame["IsOpen"].eq(1).to_numpy())
    x_open = prep(frame.iloc[open_idx], feats, set(CATEGORICAL_COLUMNS)).reset_index(drop=True)
    x_open.insert(0, "Id", frame["Id"].iloc[open_idx].to_numpy())
    return x_open, frame


def check_boosters(x_open: pd.DataFrame, n_rows: int, open_idx: np.ndarray, freeze_dir: Path) -> dict[str, float]:
    """Max |log1p difference| between each booster's prediction from the matrix and its saved forecast."""
    import lightgbm as lgb
    out: dict[str, float] = {}
    for model_file in sorted(freeze_dir.glob("lightgbm_s*.txt")):
        seed = model_file.stem.split("_s")[-1]
        booster = lgb.Booster(model_file=str(model_file))
        pred = np.zeros(n_rows)
        raw = booster.predict(x_open.drop(columns="Id"), raw_score=True)
        pred[open_idx] = decode_prediction(raw, np.ones(len(raw)), "log")
        saved = np.load(freeze_dir / f"test_pred_lightgbm_s{seed}.npy")
        out[seed] = float(np.max(np.abs(np.log1p(pred) - np.log1p(saved))))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("proj_root", type=Path)
    ap.add_argument("freeze_dir", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--seed", type=int, default=20260807, help="seed of the hub encoding = first seed of the final retrain")
    args = ap.parse_args(argv)
    x_open, frame = build_open_matrix(args.proj_root.resolve(), args.seed)
    x_open.to_parquet(args.out, compression="zstd", index=False)
    back = pd.read_parquet(args.out)
    open_idx = np.flatnonzero(frame["IsOpen"].eq(1).to_numpy())
    diffs = check_boosters(back, len(frame), open_idx, args.freeze_dir.resolve())
    print(f"wrote {args.out} rows {len(back)} columns {back.shape[1] - 1}; max |dlog1p| per booster {diffs}", flush=True)
    if not diffs or max(diffs.values()) > 1e-9:
        raise SystemExit("the exported matrix does not reproduce the frozen forecasts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
