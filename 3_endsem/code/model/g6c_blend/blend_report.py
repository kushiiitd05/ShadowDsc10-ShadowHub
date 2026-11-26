"""Tables for the G6c study: per-fold blended RMSLE of finalists and controls (pure helpers + a small CLI).

Fold 5 is scored on the hubs present in the TabPFN-TS file only (935 hubs); every model is scored on the
same rows. Folds 2 and 3 come from the study's trials.csv, folds 1, 4, 5 from the exp_runner OOF files.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import optuna_study as g6c  # noqa: E402
from helpers import fold_scores  # noqa: E402


def evaluate_fold(lgbm_oof: pd.DataFrame, tab: pd.DataFrame, fold: int, w: float) -> dict[str, Any]:
    """RMSLE (all rows, excl open-zero rows) of the log-space blend on the rows the TabPFN file covers."""
    sub_tab = tab[tab["fold"] == fold]
    if sub_tab.empty:
        raise ValueError(f"fold {fold}: no tabpfn rows")
    hubs = set(sub_tab["HubID"])
    valid = lgbm_oof[(lgbm_oof["fold"] == fold) & lgbm_oof["HubID"].isin(hubs)].reset_index(drop=True)
    if len(valid) != len(sub_tab):
        raise ValueError(f"fold {fold}: {len(valid)} lgbm rows vs {len(sub_tab)} tabpfn rows")
    tpred = g6c.align_tabpfn(valid, tab, fold, expected_rows=len(valid))
    s = g6c.blend_scores(valid["ActualOrderVolume"].to_numpy(float), valid["IsOpen"].to_numpy(),
                         valid["pred"].to_numpy(float), tpred, w)
    return {"rmsle_all": s["blend_rmsle_all"], "rmsle_excl_open0": s["blend_excl_open0"],
            "lgbm_rmsle_all": s["lgbm_rmsle_all"], "n_rows": len(valid), "n_hubs": len(hubs)}


def markdown_table(header: Sequence[str], rows: Sequence[Sequence[Any]], digits: int = 4) -> str:
    def cell(v: Any) -> str:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "-"
        return f"{v:.{digits}f}" if isinstance(v, (float, np.floating)) else str(v)
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(cell(v) for v in r) + " |" for r in rows]
    return "\n".join(lines)


def study_tag(study_name: str) -> str:
    """Short tag for variant names: 'g6c_blend' -> 'a' (study A), 'g6c_blend_b' -> 'b', anything else -> its last token."""
    return "a" if study_name == "g6c_blend" else study_name.rsplit("_", 1)[-1]


def load_trials(paths: Sequence[Path]) -> list[dict[str, Any]]:
    """Pool the trials.csv files of several studies (same schema) into one list of rows."""
    frames = [pd.read_csv(p) for p in paths]
    return pd.concat(frames, ignore_index=True).to_dict("records")


def score_folds(lgbm_oof: pd.DataFrame, tab: pd.DataFrame, folds: Sequence[int], w: float) -> dict[int, dict[str, Any]]:
    return {f: evaluate_fold(lgbm_oof, tab, f, w) for f in folds}


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="pooled finalists of the g6c studies")
    ap.add_argument("trials", nargs="+", type=Path, help="trials.csv files of all studies")
    ap.add_argument("-k", type=int, default=5)
    args = ap.parse_args(argv)
    top = g6c.select_finalists(load_trials(args.trials), args.k)
    for r in top:
        tag = study_tag(str(r["study"]))
        print(f"# {r['study']} trial {int(r['trial'])} objective {float(r['objective']):.5f}")
        print(g6c.variant_source_line(r, tag))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
