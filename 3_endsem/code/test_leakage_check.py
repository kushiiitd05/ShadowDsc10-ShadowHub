"""Unit tests for leakage_check (toy inputs; run with `python -m pytest code/test_leakage_check.py` from 3_endsem)."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from leakage_check import context_decision, gate_with_fold4, gate_without_fold4, recipe_decision, run, weight_decision  # noqa: E402


def _gains(f2: float, f3: float, f4: float, seed_f2: float = 0.004) -> pd.DataFrame:
    rows = []
    for arm in ("seed07", "seed08", "seed09", "3seed"):
        a2 = seed_f2 if arm == "seed08" else f2
        rows.append({"arm": arm, "candidate": "c", "gain_f1": 0.001, "gain_f2": a2, "gain_f3": f3, "gain_f4": f4, "gain_f5": 0.002,
                     "mean_gain_f2_4": (a2 + f3 + f4) / 3, "folds_improved_2_4": sum(x > 0 for x in (a2, f3, f4)),
                     "mean_gain_f2_4_excl0": (a2 + f3 + f4) / 3})
    return pd.DataFrame(rows)


def test_recipe_decision_needs_gate_and_majority() -> None:
    base, new = {2: 0.12, 3: 0.12, 4: 0.12}, {2: 0.11, 3: 0.125, 4: 0.115}
    assert recipe_decision(base, new, (2, 3, 4))["adopt"]          # mean +0.0033, 2 of 3 improve
    assert not recipe_decision(base, new, (3,))["adopt"]            # only the losing fold


def test_gate_without_fold4_rejects_a_negative_seed() -> None:
    assert gate_without_fold4(_gains(0.005, 0.005, 0.0), "c")["adopt"]
    assert not gate_without_fold4(_gains(0.005, 0.005, 0.0, seed_f2=-0.0003), "c")["adopt"]


def test_gate_with_fold4_uses_the_three_fold_mean() -> None:
    assert gate_with_fold4(_gains(0.003, 0.003, 0.003), "c")["adopt"]
    assert not gate_with_fold4(_gains(0.001, 0.004, 0.0005), "c")["adopt"]   # mean 0.0018 < 0.0024


def _grid() -> pd.DataFrame:
    rows = []
    for s365, s730, base in ((0.5, 0.5, 0.004), (1.0, 0.0, 0.002), (0.0, 1.0, 0.0045)):
        for w, bump, g1 in ((0.2, 0.0, 0.003), (0.3, 0.0005, 0.001), (0.5, 0.002, -0.004)):
            g = base + bump
            rows.append({"s365": s365, "s540": 0.0, "s730": s730, "w": w, "g1": g1, "g2": g, "g3": g, "g4": g, "g5": 0.005})
    return pd.DataFrame(rows)


def test_weight_kept_when_rival_inside_noise_or_losing_fold1() -> None:
    d = weight_decision(_grid(), (2, 3))
    assert d["chosen_w"] == 0.2 and d["best_rival_w"] == 0.3           # 0.5 loses fold 1, 0.3 wins by 0.0005 < noise


def test_context_mix_kept_inside_noise() -> None:
    d = context_decision(_grid(), (2, 3))
    assert d["chosen"] == "mix" and d["best_single"] == "730" and d["margin"] == pytest.approx(0.0005)


def test_real_files_no_decision_changes() -> None:
    root = Path(__file__).resolve().parents[1]
    if not (root / "artifacts" / "oof" / "base_lr05_f1-4_s20260807.csv.gz").exists():
        pytest.skip("artifacts not present")
    assert run(root)["same_decision"].all()
