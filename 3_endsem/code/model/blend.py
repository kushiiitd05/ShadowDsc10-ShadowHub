"""Offline blend check: python blend.py <dir_with_oof_files> <seed> <name1,name2,...> -> per-fold RMSLE of each member and of the log-space mean blend."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

KEYS = ["fold", "HubID", "Date"]


def rmsle(actual: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.log1p(np.clip(pred, 0, None)) - np.log1p(actual)) ** 2)))


def load(folder: Path, name: str, seed: int) -> pd.DataFrame:
    frame = pd.read_csv(folder / f"oof_{name}_s{seed}.csv.gz")
    return frame[[*KEYS, "ActualOrderVolume", "pred"]].rename(columns={"pred": name})


def blend_table(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    names = list(frames)
    merged = frames[names[0]]
    for n in names[1:]:
        merged = merged.merge(frames[n][[*KEYS, n]], on=KEYS, how="inner", validate="one_to_one")
    merged["blend"] = np.expm1(np.mean([np.log1p(merged[n].clip(lower=0)) for n in names], axis=0))
    rows = []
    for fold, g in merged.groupby("fold"):
        rows.append({"fold": fold, **{n: rmsle(g["ActualOrderVolume"].to_numpy(), g[n].to_numpy()) for n in [*names, "blend"]}})
    out = pd.DataFrame(rows).set_index("fold")
    out.loc["mean"] = out.mean()
    return out


if __name__ == "__main__":
    folder, seed, names = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3].split(",")
    print(blend_table({n: load(folder, n, seed) for n in names}).round(4).to_string())
