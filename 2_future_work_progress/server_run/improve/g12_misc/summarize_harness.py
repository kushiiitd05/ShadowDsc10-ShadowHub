"""Compare harness variant OOF files with the cs02 champion OOF per fold: gain (cs02 minus variant), excl-open-zero gain, gate flags.

Usage: python g12_misc/summarize_harness.py <results dir with variant OOFs> <variant,...> [--seeds 20260807,...] [--cs02-dir <dir>]
cs02 OOFs default to results/G5/oof_cs02_s<seed>.csv.gz (folds 1-4) and results/exp_out_F5_<seed>/oof_cs02_s<seed>.csv.gz (fold 5).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import offline_common as oc


def champion_frames(results: Path, seed: int) -> dict[int, pd.DataFrame]:
    dev = oc.load_oof(results / "G5" / f"oof_cs02_s{seed}.csv.gz")
    frames = {int(f): g for f, g in dev.groupby("fold")}
    f5 = results / f"exp_out_F5_{seed}" / f"oof_cs02_s{seed}.csv.gz"
    if f5.exists():
        frames[5] = oc.load_oof(f5)
    return frames


def compare(variant_oof: pd.DataFrame, champion: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """Per-fold base/new RMSLE on the rows both files hold (inner join on HubID, Date)."""
    rows = []
    for fold, g in variant_oof.groupby("fold"):
        if fold not in champion:
            continue
        merged = champion[fold].merge(g[["HubID", "Date", "pred"]], on=["HubID", "Date"], suffixes=("", "_new"), validate="one_to_one")
        base, new = oc.scores(merged), oc.scores(merged, merged["pred_new"].to_numpy())
        rows.append({"fold": int(fold), "n": len(merged), "base_all": base["all"], "base_excl": base["excl_open0"], "new_all": new["all"], "new_excl": new["excl_open0"]})
    out = pd.DataFrame(rows)
    if len(out):
        out["gain_all"], out["gain_excl"] = out["base_all"] - out["new_all"], out["base_excl"] - out["new_excl"]
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", type=Path)
    ap.add_argument("variants")
    ap.add_argument("--seeds", default=str(oc.SEEDS[0]))
    ap.add_argument("--results", type=Path, default=Path.home() / "Desktop/Dsc/endterm/results")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    tables = []
    for seed in [int(s) for s in args.seeds.split(",")]:
        champion = champion_frames(args.results, seed)
        for name in args.variants.split(","):
            path = args.folder / f"oof_{name}_s{seed}.csv.gz"
            if not path.exists():
                continue
            tables.append(compare(oc.load_oof(path), champion).assign(variant=name, seed=seed))
    out = pd.concat(tables, ignore_index=True)
    if args.out:
        out.to_csv(args.out, index=False)
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print(out.round(5).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
