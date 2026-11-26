"""Re-run the frozen forecaster anywhere (laptop or Colab): no training, no server, no raw-data pipeline.

Inputs (all in the repository):
    final/models/lightgbm_s2026080{7,8,9}.txt.gz   the three frozen LightGBM boosters (hp17 recipe, 1000 trees each)
    final/forecast_matrix_open.parquet              their exact input matrix: the 40,282 open rows of the forecast window
                                                    (Id + 342 feature columns, written by g8_final/export_forecast_matrix.py)
    artifacts/test_window/tabpfn_mix_test.csv       TabPFN-TS forecast, equal log mix of the 365- and 730-day contexts

Formula (log1p space, closed rows exactly 0):
    forecast = expm1( 0.8 * mean_seed log1p(LightGBM_seed) + 0.2 * log1p(TabPFN-TS mix) )

usage (from the 3_endsem folder):  python code/model/predict_frozen.py [out.csv]
Needs numpy, pandas, pyarrow and lightgbm (any 4.x reads the boosters; 4.7.0 was used to train them).
"""
from __future__ import annotations

import gzip
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SEEDS = (20260807, 20260808, 20260809)
BLEND_WEIGHT = 0.2


def load_booster(path: Path):
    import lightgbm as lgb
    with gzip.open(path, "rt") as f:
        return lgb.Booster(model_str=f.read())


def seed_forecast(booster, x_open: pd.DataFrame, open_pos: np.ndarray, n_rows: int) -> np.ndarray:
    """Volume forecast of one booster in frame order: expm1(raw score) on open rows, 0 on closed rows."""
    out = np.zeros(n_rows)
    out[open_pos] = np.clip(np.expm1(booster.predict(x_open, raw_score=True)), 0.0, None)
    return out


def blend(seed_preds: list[np.ndarray], tabpfn: np.ndarray, is_open: np.ndarray, weight: float = BLEND_WEIGHT) -> np.ndarray:
    """Seed log-mean, then (1-w)/w log-space blend with TabPFN-TS; closed rows exactly 0. Same as final_retrain.py."""
    lgbm = np.expm1(np.mean([np.log1p(np.clip(p, 0, None)) for p in seed_preds], axis=0))
    lgbm = np.where(is_open == 1, np.clip(lgbm, 0, None), 0.0)
    mixed = (1.0 - weight) * np.log1p(lgbm) + weight * np.log1p(np.asarray(tabpfn, dtype=float))
    return np.where(is_open == 1, np.clip(np.expm1(mixed), 0, None), 0.0)


def frozen_forecast(root: Path = ROOT) -> tuple[pd.DataFrame, dict[int, np.ndarray]]:
    """Return (Id, OrderVolume) in the submission's Id order, and each seed's forecast."""
    x_open = pd.read_parquet(root / "final" / "forecast_matrix_open.parquet")
    tabpfn = pd.read_csv(root / "artifacts" / "test_window" / "tabpfn_mix_test.csv")
    ids = tabpfn["Id"].to_numpy()
    position = pd.Series(np.arange(len(ids)), index=ids)
    open_pos = position.loc[x_open["Id"].to_numpy()].to_numpy()
    is_open = np.zeros(len(ids), dtype=int)
    is_open[open_pos] = 1
    features = x_open.drop(columns="Id")
    seeds = {s: seed_forecast(load_booster(root / "final" / "models" / f"lightgbm_s{s}.txt.gz"), features, open_pos, len(ids)) for s in SEEDS}
    volume = blend(list(seeds.values()), tabpfn["OrderVolume"].to_numpy(float), is_open)
    return pd.DataFrame({"Id": ids, "OrderVolume": volume}), seeds


def main(argv: list[str]) -> int:
    table, seeds = frozen_forecast()
    frozen = pd.read_csv(ROOT / "final" / "submission_final.csv")
    for s, p in seeds.items():
        saved = np.load(ROOT / "final" / f"test_pred_lightgbm_s{s}.npy")
        print(f"seed {s}: max |dlog1p| vs saved forecast {np.max(np.abs(np.log1p(p) - np.log1p(saved))):.2e}")
    diff = float(np.max(np.abs(np.log1p(table["OrderVolume"]) - np.log1p(frozen["OrderVolume"]))))
    same_ids = bool((table["Id"].to_numpy() == frozen["Id"].to_numpy()).all())
    print(f"rows {len(table)}, same Id order as final/submission_final.csv: {same_ids}, max |dlog1p| {diff:.2e}")
    if argv:
        table.to_csv(argv[0], index=False)
        print("wrote", argv[0])
    return 0 if same_ids and diff < 1e-9 else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
