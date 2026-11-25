"""Gate table for the offline items: per variant and seed, gain per dev fold (+ folds 1/5 when computable), gate verdict, seed sign agreement.

Usage: python g12_misc/summarize_offline.py <MISC dir>  -> prints the table and writes offline_summary.csv
Verdict labels: pass (gate met on every seed), challenger (gain > 0 on every dev fold and seed and the seed-mean gain >= noise 0.0012,
but gate not met), not adopted (otherwise).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import offline_common as oc

PRIMARY = {  # pre-declared variants used for the verdict (everything else is a sensitivity row)
    "C_peer_resid": [("group", 28, None), ("corr5", 28, None), ("own+corr5", 28, None)],
    "D_shift": [("fit", 14, None), ("literal", 14, None), ("fit_h", 14, None)],
    "E_reconcile": [("format_pattern", None, "clip01"), ("format_pattern2", None, "clip01"), ("format_pattern2", None, "ols")],
}


def verdict(per_seed: list[dict]) -> str:
    if all(s["gate_all"] and s["gate_excl"] for s in per_seed):
        return "pass"
    means = [s["mean_gain"] for s in per_seed]
    if sum(means) / len(means) >= oc.NOISE and all(s["min_fold_gain"] > 0 for s in per_seed):
        return "challenger"
    return "not adopted"


def summarize(folder: Path) -> pd.DataFrame:
    rows = []
    for item, variants in PRIMARY.items():
        table = pd.read_csv(folder / f"{item}.csv")
        for variant, window, weights in variants:
            sel = table[table["variant"] == variant]
            if window is not None:
                sel = sel[sel["window"] == window]
            if weights is not None:
                sel = sel[sel["weights"] == weights]
            per_seed = []
            for seed, g in sel.groupby("seed"):
                gate = oc.gate_summary(g)
                dev = g[g["fold"].isin(oc.DEV_FOLDS)]
                per_seed.append({**gate, "seed": seed, "min_fold_gain": float(dev["gain_all"].min()),
                                 **{f"gain_f{int(r.fold)}": r.gain_all for r in g.itertuples()},
                                 **{f"excl_f{int(r.fold)}": r.gain_excl for r in g.itertuples()}})
            avg = pd.DataFrame(per_seed).mean(numeric_only=True)
            rows.append({"item": item, "variant": variant, "window": window, "weights": weights, "verdict": verdict(per_seed),
                         **{k: float(avg[k]) for k in avg.index if k.startswith(("gain_f", "excl_f", "mean_gain"))},
                         "seed_mean_gains": ";".join(f"{s['mean_gain']:.5f}" for s in per_seed),
                         "gate_all_by_seed": ";".join(str(bool(s["gate_all"])) for s in per_seed)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    out = summarize(Path(sys.argv[1]))
    out.to_csv(Path(sys.argv[1]) / "offline_summary.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(out.round(5).to_string())
