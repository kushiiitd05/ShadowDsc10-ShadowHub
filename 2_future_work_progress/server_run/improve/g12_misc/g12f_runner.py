"""Item F runner: special-day and ordinary-day LightGBM experts blended with cs02 by a calendar gate (g12f_regime.py).

Usage: python g12_misc/g12f_runner.py <proj_root> <out_dir> --global-oof <cs02 oof file(s), comma separated> --folds 3,4 --seed 20260807 --n-jobs 16
Fits two experts per fold with the harness (special: Sunday / regional holiday rows only, a small fixed-round model; ordinary: the cs02 recipe on
the remaining rows), then writes one OOF file per pre-declared blend (g12f_hard, g12f_half, g12f_spec, g12f_spec_half) plus the raw experts.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
IMPROVE = HERE.parent
for _path in (IMPROVE, HERE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import exp_runner as base  # noqa: E402
import g12f_regime as reg  # noqa: E402
from helpers import fold_scores  # noqa: E402

CHAMPION = "cs02"


def expert_configs() -> dict[str, base.Config]:
    champ = base.VARIANTS[CHAMPION]
    return {"special": replace(champ, colsample=0.5, num_leaves=15, min_child=20, fixed_rounds=reg.SPECIAL_EXPERT_ROUNDS),
            "ordinary": champ}


def load_global(paths: str) -> pd.DataFrame:
    frames = [pd.read_csv(p, parse_dates=["Date"]) for p in paths.split(",")]
    out = pd.concat(frames, ignore_index=True)
    return out[["fold", "HubID", "Date", "pred"]]


def blended_oof(valid_oof: pd.DataFrame, special_rows: np.ndarray, p_global: np.ndarray, p_special: np.ndarray, p_ordinary: np.ndarray,
                gate: tuple[float, float]) -> pd.DataFrame:
    out = valid_oof.copy()
    out["pred"] = reg.blend_log(p_global, p_special, p_ordinary, special_rows, gate)
    return out


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("proj_root", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--global-oof", required=True, help="cs02 OOF csv(.gz) path(s) for the same seed, comma separated (columns fold, HubID, Date, pred)")
    ap.add_argument("--folds", default="3,4")
    ap.add_argument("--seed", type=int, default=20260807)
    ap.add_argument("--n-jobs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--rounds", type=int, default=1500)
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.proj_root.resolve()
    sys.path.insert(0, str(root))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    global_oof = load_global(args.global_oof)
    cfgs = expert_configs()
    names = [*reg.BLENDS, "g12f_expert_special", "g12f_expert_ordinary"]
    results: dict[str, list[dict]] = {n: [] for n in names}
    oofs: dict[str, list[pd.DataFrame]] = {n: [] for n in names}
    for fold in [int(x) for x in args.folds.split(",")]:
        t0 = time.perf_counter()
        ctx = base.prepare_fold(root, fold, args.seed)
        print(f"fold {fold}: loaded in {time.perf_counter() - t0:.0f}s, special train rows {int(reg.special_flag(ctx['train']).sum())}", flush=True)
        expert_oof, expert_scores = {}, {}
        for regime, flag in (("special", True), ("ordinary", False)):
            oof, scores = base.fit_variant(reg.regime_ctx(ctx, flag), cfgs[regime], args.seed, args.n_jobs, args.lr, args.rounds)
            expert_oof[regime], expert_scores[regime] = oof, scores
            print(regime, "expert fold", fold, json.dumps(scores), flush=True)
        valid = ctx["valid"]
        special_rows = reg.special_flag(valid)
        p_global = reg.join_global(expert_oof["ordinary"], global_oof)
        p_s, p_o = expert_oof["special"]["pred"].to_numpy(), expert_oof["ordinary"]["pred"].to_numpy()
        for blend, gate in reg.BLENDS.items():
            oof = blended_oof(expert_oof["ordinary"], special_rows, p_global, p_s, p_o, gate)
            s = {**fold_scores(oof["ActualOrderVolume"].to_numpy(float), oof["pred"].to_numpy(), oof["IsOpen"].to_numpy()),
                 "fold": fold, "best_iteration": 0, "fit_seconds": 0.0}
            oofs[blend].append(oof)
            results[blend].append(s)
            print(blend, "fold", fold, json.dumps(s), flush=True)
        for regime in ("special", "ordinary"):
            oofs[f"g12f_expert_{regime}"].append(expert_oof[regime])
            results[f"g12f_expert_{regime}"].append(expert_scores[regime])
        del ctx
    for n in names:
        pd.concat(oofs[n], ignore_index=True).to_csv(args.out_dir / f"oof_{n}_s{args.seed}.csv.gz", index=False, compression="gzip")
        cfg = SimpleNamespace(**{**base.VARIANTS[CHAMPION].__dict__, "recipe": CHAMPION, "regime": n})
        base.write_record(args.out_dir, n, cfg, args.seed, results[n], args)
        print("DONE", n, round(sum(r["rmsle_all"] for r in results[n]) / len(results[n]), 5), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
