"""Run the offline items C, D, E on the saved cs02 OOF files for the three seeds and write long result tables.

Usage: python g12_misc/run_offline.py --results ~/Desktop/Dsc/endterm/results --raw-dir <folder with orders_train.csv, hub_metadata.csv> --out <dir>
Writes <out>/C_peer_resid.csv, D_shift.csv, E_reconcile.csv (columns seed, variant, window/weights, fold, base_all, new_all, gain_all, ...).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
for _path in (HERE.parent / "g3b_features", HERE.parent / "g3_features", HERE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import g12c_peer_resid as c  # noqa: E402
import g12d_shift as d  # noqa: E402
import g12e_reconcile as e  # noqa: E402
import offline_common as oc  # noqa: E402
from feature_panel import build_panel  # noqa: E402


def load_seed_frames(results: Path, seed: int) -> dict[int, pd.DataFrame]:
    dev = oc.load_oof(results / "G5" / f"oof_cs02_s{seed}.csv.gz")
    frames = {int(f): g.reset_index(drop=True) for f, g in dev.groupby("fold")}
    f5 = results / f"exp_out_F5_{seed}" / f"oof_cs02_s{seed}.csv.gz"
    if f5.exists():
        frames[5] = oc.load_oof(f5).reset_index(drop=True)
    return frames


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--raw-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seeds", default=",".join(str(s) for s in oc.SEEDS))
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(args.raw_dir / "hub_metadata.csv")[["HubID", "HubFormat", "AssortmentTier"]]
    groups = (meta["HubFormat"].astype(str) + "_" + meta["AssortmentTier"].astype(str)).set_axis(meta["HubID"])
    history = pd.read_csv(args.raw_dir / "orders_train.csv", parse_dates=["Date"], usecols=["HubID", "Date", "OrderVolume", "IsOpen"])
    panel = build_panel(history)
    tables: dict[str, list[pd.DataFrame]] = {"C_peer_resid": [], "D_shift": [], "E_reconcile": []}
    diag = []
    for seed in [int(s) for s in args.seeds.split(",")]:
        frames = load_seed_frames(args.results, seed)
        tables["C_peer_resid"].append(c.run(frames, groups, panel).assign(seed=seed))
        tables["D_shift"].append(d.run(frames).assign(seed=seed))
        tables["E_reconcile"].append(e.run(frames, meta).assign(seed=seed))
        diag.append(c.diagnostics(frames, groups, panel).assign(seed=seed))
        print("seed", seed, "done", flush=True)
    for name, parts in tables.items():
        pd.concat(parts, ignore_index=True).to_csv(args.out / f"{name}.csv", index=False)
    pd.concat(diag, ignore_index=True).to_csv(args.out / "C_diagnostics.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
