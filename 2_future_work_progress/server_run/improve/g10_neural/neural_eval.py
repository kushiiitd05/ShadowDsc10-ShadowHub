"""Stand-alone and log-space blend evaluation of a pilot member against the champion, on identical rows.

Usage: python neural_eval.py <champion_oof.csv.gz> <member_oof.csv.gz> [--name tide] [--gate-folds 2,3,4]
Prints per-fold RMSLE (all rows / excluding open-but-zero rows) alone and blended at 0.1/0.2/0.3, plus the gate verdict.
Gate: mean gain over the gate folds >= 0.0024 AND gain > 0 on at least 2 of the gate folds (blend at the best weight).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

KEYS = ["fold", "HubID", "Date"]
WEIGHTS = (0.1, 0.2, 0.3)
GATE_MEAN_GAIN = 0.0024
GATE_MIN_FOLDS = 2


def rmsle(actual: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.log1p(np.clip(pred, 0, None)) - np.log1p(actual)) ** 2)))


def blend_log(champion: np.ndarray, member: np.ndarray, weight: float) -> np.ndarray:
    """Log-space blend: (1-w) * log1p(champion) + w * log1p(member), back to volume. Zero stays zero."""
    z = (1 - weight) * np.log1p(np.clip(champion, 0, None)) + weight * np.log1p(np.clip(member, 0, None))
    return np.expm1(z)


def align(champion: pd.DataFrame, member: pd.DataFrame) -> pd.DataFrame:
    """Inner join on (fold, HubID, Date); the member's open flag and actuals must agree with the champion's."""
    m = champion[[*KEYS, "ActualOrderVolume", "IsOpen", "pred"]].rename(columns={"pred": "champ"}).merge(
        member[[*KEYS, "ActualOrderVolume", "pred"]].rename(columns={"ActualOrderVolume": "act_m", "pred": "member"}),
        on=KEYS, how="inner", validate="one_to_one")
    if not np.allclose(m["ActualOrderVolume"], m["act_m"]):
        raise ValueError("champion and member disagree on ActualOrderVolume for the same rows")
    return m.drop(columns="act_m")


def evaluate(champion: pd.DataFrame, member: pd.DataFrame) -> pd.DataFrame:
    """One row per fold and row set ('all' / 'excl_open0'): champion, member, blend@w RMSLE and gain@w = champion - blend."""
    m = align(champion, member)
    rows = []
    for fold, g in m.groupby("fold"):
        open_zero = ((g["IsOpen"] == 1) & (g["ActualOrderVolume"] == 0)).to_numpy()
        for rows_name, mask in (("all", np.ones(len(g), bool)), ("excl_open0", ~open_zero)):
            a, c, p = (g[k].to_numpy()[mask] for k in ("ActualOrderVolume", "champ", "member"))
            rec = {"fold": int(fold), "rows": rows_name, "n": int(mask.sum()), "champion": rmsle(a, c), "member": rmsle(a, p)}
            for w in WEIGHTS:
                rec[f"blend{w}"] = rmsle(a, blend_log(c, p, w))
                rec[f"gain{w}"] = rec["champion"] - rec[f"blend{w}"]
            rows.append(rec)
    return pd.DataFrame(rows)


def gate_verdict(table: pd.DataFrame, gate_folds: list[int], rows: str = "all") -> dict:
    """Pass if, at some blend weight, mean gain over gate folds >= 0.0024 and the gain is positive on >= 2 gate folds."""
    sub = table[(table["rows"] == rows) & table["fold"].isin(gate_folds)]
    out = {"rows": rows, "folds": gate_folds, "passed": False, "best_weight": None, "mean_gain": float("-inf"), "folds_improved": 0}
    for w in WEIGHTS:
        gains = sub[f"gain{w}"].to_numpy()
        mean_gain, improved = float(gains.mean()), int((gains > 0).sum())
        ok = mean_gain >= GATE_MEAN_GAIN and improved >= GATE_MIN_FOLDS
        better = (ok, mean_gain) > (out["passed"], out["mean_gain"])
        if better:
            out.update(passed=ok, best_weight=w, mean_gain=mean_gain, folds_improved=improved)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("champion", type=Path)
    ap.add_argument("member", type=Path)
    ap.add_argument("--name", default="member")
    ap.add_argument("--gate-folds", default="2,3,4")
    args = ap.parse_args()
    champion, member = pd.read_csv(args.champion), pd.read_csv(args.member)
    table = evaluate(champion, member)
    print(f"== {args.name} vs champion on identical rows ==")
    print(table.round(4).to_string(index=False))
    for rows in ("all", "excl_open0"):
        print(args.name, gate_verdict(table, [int(x) for x in args.gate_folds.split(",")], rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
