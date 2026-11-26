"""Would any model decision change if development fold 4 (= the end-term test window) had never been looked at?

Fold 4 (9 May - 19 Jun 2015) is both a development fold and the end-term test window. No fold-4 label enters an end-term
fit, a sampling probability or a column choice, but fold-4 scores were part of earlier recipe decisions. This script
re-takes every such decision with fold 4 removed, from saved files only, and reports whether the outcome changes.

Run from the 3_endsem folder:
    python code/leakage_check.py        # writes results/leakage_check_without_fold4.csv

Inputs: artifacts/oof/base_lr05_f1-4_s20260807.csv.gz, artifacts/oof/cs02_lr05_f1-4_s20260807.csv.gz (screening runs,
lr 0.05, seed 07), results/audit_gate_gains.csv (freeze audit), results/tabpfn_context_weight_grid.csv (offline grid).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GATE_MIN_GAIN = 0.0024  # 2 x measured seed noise
NOISE = 0.0012
SEEDS = ("seed07", "seed08", "seed09")


def fold_rmsle(oof: pd.DataFrame) -> dict[int, float]:
    out = {}
    for fold, g in oof.groupby("fold"):
        err = np.log1p(g["pred"].clip(lower=0).to_numpy(float)) - np.log1p(g["ActualOrderVolume"].to_numpy(float))
        out[int(fold)] = float(np.sqrt(np.mean(err ** 2)))
    return out


def recipe_decision(base: dict[int, float], new: dict[int, float], folds: tuple[int, ...]) -> dict:
    """Adopt the new recipe if its mean gain over `folds` clears the gate and most of those folds improve."""
    gains = [base[f] - new[f] for f in folds]
    adopt = float(np.mean(gains)) >= GATE_MIN_GAIN and sum(g > 0 for g in gains) * 2 > len(gains)
    return {"mean_gain": float(np.mean(gains)), "adopt": bool(adopt)}


def gate_with_fold4(gains: pd.DataFrame, candidate: str) -> dict:
    """The recorded gate: 3-seed mean over folds 2-4 >= 0.0024 (also without open-zero rows), >= 2 of 3 folds improve,
    every seed's mean gain positive."""
    rows = gains[gains["candidate"] == candidate].set_index("arm")
    avg = rows.loc["3seed"]
    adopt = (avg["mean_gain_f2_4"] >= GATE_MIN_GAIN and avg["mean_gain_f2_4_excl0"] >= GATE_MIN_GAIN
             and avg["folds_improved_2_4"] >= 2 and all(rows.loc[s, "mean_gain_f2_4"] > 0 for s in SEEDS))
    return {"mean_gain": float(avg["mean_gain_f2_4"]), "adopt": bool(adopt)}


def gate_without_fold4(gains: pd.DataFrame, candidate: str) -> dict:
    """Gate on folds 2-3 only: 3-seed mean >= 0.0024, both folds improve on every seed, fold 5 positive on every seed."""
    rows = gains[gains["candidate"] == candidate].set_index("arm")
    mean23 = float((rows.loc["3seed", "gain_f2"] + rows.loc["3seed", "gain_f3"]) / 2)
    every_seed = bool(all(rows.loc[s, "gain_f2"] > 0 and rows.loc[s, "gain_f3"] > 0 for s in SEEDS))
    fold5 = bool(all(rows.loc[s, "gain_f5"] > 0 for s in SEEDS))
    return {"mean_gain": mean23, "adopt": bool(mean23 >= GATE_MIN_GAIN and every_seed and fold5),
            "every_seed_positive": every_seed, "fold5_positive": fold5}


def weight_decision(grid: pd.DataFrame, folds: tuple[int, ...], keep: float = 0.2) -> dict:
    """Keep w=`keep` unless another weight beats it by more than the noise on `folds` without losing on folds 1 or 5."""
    mix = grid[(grid.s365 == 0.5) & (grid.s540 == 0) & (grid.s730 == 0.5)].copy()
    mix["sel"] = mix[[f"g{f}" for f in folds]].mean(axis=1)
    base = float(mix.loc[mix.w == keep, "sel"].iloc[0])
    rivals = mix[(mix.w != keep) & (mix.g1 > 0) & (mix.g5 > 0)]
    best = rivals.loc[rivals["sel"].idxmax()]
    switch = float(best["sel"]) - base > NOISE
    return {"mean_gain": base, "best_rival_w": float(best["w"]), "rival_margin": float(best["sel"]) - base,
            "chosen_w": float(best["w"]) if switch else keep}


def context_decision(grid: pd.DataFrame, folds: tuple[int, ...], w: float = 0.2) -> dict:
    """At weight w: margin of the best single context over the 365/730 mix on `folds` (mix kept unless margin > noise)."""
    at_w = grid[grid.w == w]
    def sel(q):
        return float(at_w[q][[f"g{f}" for f in folds]].mean(axis=1).iloc[0])
    mix = sel((at_w.s365 == 0.5) & (at_w.s540 == 0) & (at_w.s730 == 0.5))
    single = {"365": sel(at_w.s365 == 1), "730": sel(at_w.s730 == 1)}
    name = max(single, key=single.get)
    return {"mean_gain": mix, "best_single": name, "margin": single[name] - mix,
            "chosen": name if single[name] - mix > NOISE else "mix"}


def run(root: Path = ROOT) -> pd.DataFrame:
    oof = root / "artifacts" / "oof"
    base = fold_rmsle(pd.read_csv(oof / "base_lr05_f1-4_s20260807.csv.gz"))
    cs02 = fold_rmsle(pd.read_csv(oof / "cs02_lr05_f1-4_s20260807.csv.gz"))
    gains = pd.read_csv(root / "results" / "audit_gate_gains.csv")
    grid = pd.read_csv(root / "results" / "tabpfn_context_weight_grid.csv")
    rows = []
    with4, without4 = recipe_decision(base, cs02, (2, 3, 4)), recipe_decision(base, cs02, (2, 3))
    rows.append(("x* recipe: cs02 (feature fraction 0.2 + bagging) over base", "folds 2-4", f"adopt={with4['adopt']} gain {with4['mean_gain']:.4f}",
                 f"adopt={without4['adopt']} gain {without4['mean_gain']:.4f} (fold 1 {base[1] - cs02[1]:+.4f})", with4["adopt"] == without4["adopt"]))
    for cand, label in (("hp17", "forecaster: hp17 alone"), ("candidate", "forecaster: hp17 + TabPFN-TS mix, w 0.2")):
        g4, g = gate_with_fold4(gains, cand), gate_without_fold4(gains, cand)
        rows.append((label, "gate (folds 2-4 / folds 2-3)", f"pass={g4['adopt']} gain {g4['mean_gain']:.4f}",
                     f"pass={g['adopt']} gain {g['mean_gain']:.4f} (every seed positive: {g['every_seed_positive']})", g4["adopt"] == g["adopt"]))
    w4, w23 = weight_decision(grid, (2, 3, 4)), weight_decision(grid, (2, 3))
    rows.append(("forecaster: TabPFN-TS weight", "keep 0.2 unless a rival wins by > noise", f"w={w4['chosen_w']} (best rival {w4['best_rival_w']} +{w4['rival_margin']:.4f})",
                 f"w={w23['chosen_w']} (best rival {w23['best_rival_w']} +{w23['rival_margin']:.4f})", w4["chosen_w"] == w23["chosen_w"]))
    c4, c23 = context_decision(grid, (2, 3, 4)), context_decision(grid, (2, 3))
    rows.append(("forecaster: 365/730 context mix", "mix unless a single context wins by > noise", f"{c4['chosen']} (best single {c4['best_single']} {c4['margin']:+.4f})",
                 f"{c23['chosen']} (best single {c23['best_single']} {c23['margin']:+.4f})", c4["chosen"] == c23["chosen"]))
    for what in ("x* rounds (1184) and ridge lambda: training-only holdout", "x* learning rate 0.05: screening default, not tuned",
                 "coreset: rough models, sensitivities, column importance: training rows only"):
        rows.append((what, "fold 4 not used", "-", "-", True))
    return pd.DataFrame(rows, columns=["decision", "rule", "with_fold4", "without_fold4", "same_decision"])


if __name__ == "__main__":
    table = run()
    out = ROOT / "results" / "leakage_check_without_fold4.csv"
    table.to_csv(out, index=False)
    print(table.to_string(index=False))
    print(out)
