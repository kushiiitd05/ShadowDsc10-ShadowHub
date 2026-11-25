"""Stage G2 'structure': horizon-specialised LightGBM variants, same CLI and output schema as exp_runner.py.

Usage (from the improve dir on the server):
  python g2_horizon/g2_runner.py <proj_root> <out_dir> --variants split_horizon,long_window_only --folds 2,3,4 --seed 20260807 --n-jobs 16

Variants: cs02 (control, single full-range model through this code path), split_horizon (separate cs02 models for
Horizon 1-21 and 22-42), long_window_only (as split_horizon, but the 22-42 model drops short-window columns).
Writes <out_dir>/experiments.csv rows and oof_<variant>_s<seed>.csv.gz exactly like exp_runner.py.
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
from segments import G2_VARIANTS, fit_segmented, segment_label  # noqa: E402


@dataclass(frozen=True)
class G2Config(Config):
    segments: str = ""  # e.g. '1-21|22-42r'; makes the config hash differ per variant


def g2_config(name: str) -> G2Config:
    spec = G2_VARIANTS[name]
    return G2Config(**{**VARIANTS[spec.base].__dict__, "segments": segment_label(spec.segments)})


def parse_variants(text: str) -> list[str]:
    names = [v.strip() for v in text.split(",") if v.strip()]
    unknown = [v for v in names if v not in G2_VARIANTS]
    if unknown:
        raise SystemExit(f"unknown variants: {unknown}; choose from {sorted(G2_VARIANTS)}")
    return names


def once_logger():
    """print() that suppresses repeated column listings (identical for every fold)."""
    seen: set[str] = set()

    def log(message: str) -> None:
        if "restricted columns" in message:
            if message in seen:
                return
            seen.add(message)
        print(message, flush=True)

    return log


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("proj_root", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--variants", required=True, help="comma list of: " + ",".join(sorted(G2_VARIANTS)))
    ap.add_argument("--folds", default="2,3,4")
    ap.add_argument("--seed", type=int, default=20260807)
    ap.add_argument("--n-jobs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--rounds", type=int, default=1500)
    args = ap.parse_args()
    names = parse_variants(args.variants)
    root = args.proj_root.resolve()
    sys.path.insert(0, str(root))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    log = once_logger()
    results: dict[str, list[dict]] = {n: [] for n in names}
    oofs: dict[str, list[pd.DataFrame]] = {n: [] for n in names}
    for f in [int(x) for x in args.folds.split(",")]:
        t0 = time.perf_counter()
        ctx = prepare_fold(root, f, args.seed)
        print(f"fold {f}: loaded in {time.perf_counter() - t0:.0f}s, train rows {len(ctx['train'])}", flush=True)
        for n in names:
            oof, s = fit_segmented(ctx, G2_VARIANTS[n], g2_config(n), args.seed, args.n_jobs, args.lr, args.rounds, fit=fit_variant, log=log)
            oofs[n].append(oof)
            results[n].append(s)
            print(n, "fold", f, json.dumps(s), flush=True)
        del ctx
    for n in names:
        pd.concat(oofs[n], ignore_index=True).to_csv(args.out_dir / f"oof_{n}_s{args.seed}.csv.gz", index=False, compression="gzip")
        write_record(args.out_dir, n, g2_config(n), args.seed, results[n], args)
        print("DONE", n, round(float(np.mean([r["rmsle_all"] for r in results[n]])), 5), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
