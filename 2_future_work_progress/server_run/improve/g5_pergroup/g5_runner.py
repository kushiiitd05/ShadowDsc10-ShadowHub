"""Per-group experiment runner (1b.10). Usage from the improve dir:
  python g5_pergroup/g5_runner.py <proj_root> <out_dir> --variants perfmt,pertier --folds 1,2,3,4 --seed 20260807 --n-jobs 16
Variants: perfmt (one cs02 model per HubFormat), pertier (per AssortmentTier). Same output schema as exp_runner.py.
Blend with the global model offline: python blend.py <dir> <seed> cs02,perfmt (copy oof_cs02 into the same dir).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
for _p in (HERE, HERE.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
from exp_runner import VARIANTS, Config, fit_variant, prepare_fold, write_record  # noqa: E402
from pergroup import fit_by_group  # noqa: E402

GROUPED = {"perfmt": "HubFormat", "pertier": "AssortmentTier"}


@dataclass(frozen=True)
class G5Config(Config):
    group: str = ""


def config_for(name: str) -> G5Config:
    return G5Config(**{**VARIANTS["cs02"].__dict__, "group": GROUPED[name]})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("proj_root", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--variants", default="perfmt,pertier")
    ap.add_argument("--folds", default="1,2,3,4")
    ap.add_argument("--seed", type=int, default=20260807)
    ap.add_argument("--n-jobs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--rounds", type=int, default=1500)
    args = ap.parse_args()
    names = [v.strip() for v in args.variants.split(",")]
    if [v for v in names if v not in GROUPED]:
        raise SystemExit(f"choose from {sorted(GROUPED)}")
    root = args.proj_root.resolve()
    sys.path.insert(0, str(root))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    results = {n: [] for n in names}
    oofs = {n: [] for n in names}
    for f in [int(x) for x in args.folds.split(",")]:
        t0 = time.perf_counter()
        ctx = prepare_fold(root, f, args.seed)
        print(f"fold {f}: loaded in {time.perf_counter() - t0:.0f}s, train rows {len(ctx['train'])}", flush=True)
        for n in names:
            oof, s = fit_by_group(ctx, GROUPED[n], config_for(n), args.seed, args.n_jobs, args.lr, args.rounds, fit=fit_variant)
            oofs[n].append(oof)
            results[n].append(s)
            print(n, "fold", f, json.dumps(s), flush=True)
        del ctx
    for n in names:
        pd.concat(oofs[n], ignore_index=True).to_csv(args.out_dir / f"oof_{n}_s{args.seed}.csv.gz", index=False, compression="gzip")
        write_record(args.out_dir, n, config_for(n), args.seed, results[n], args)
        print("DONE", n, round(float(np.mean([r["rmsle_all"] for r in results[n]])), 5), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
