"""Item A runner: cs02 recipe + truncated-SVD denoising features (g12a_svd.py).

Usage: python g12_misc/g12a_runner.py <proj_root> <out_dir> --variants g12a_svd --folds 3,4 --seed 20260807 --n-jobs 16
Variants: g12a_svd (rank 14), g12a_svd12, g12a_svd15. Same output schema as g3b_runner.py (experiments.csv + oof_<variant>_s<seed>.csv.gz).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

HERE = Path(__file__).resolve().parent
IMPROVE = HERE.parent
for _path in (IMPROVE / "g3b_features", IMPROVE / "g3_features", IMPROVE, HERE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import exp_runner as base  # noqa: E402
import g3_runner  # noqa: E402  (read-only; load_history)
from feature_panel import Panel, build_panel  # noqa: E402
from g3b_runner import with_target_date  # noqa: E402
import g12a_svd as svd  # noqa: E402

CHAMPION = "cs02"
load_history = g3_runner.load_history


@dataclass(frozen=True)
class VariantSpec:
    rank: int


VARIANT_SPECS: dict[str, VariantSpec] = {"g12a_svd": VariantSpec(14), "g12a_svd12": VariantSpec(12), "g12a_svd15": VariantSpec(15)}


def augment_ctx(ctx: dict, history: pd.DataFrame | Panel, rank: int, cache: dict | None = None, reference: svd.Reference | None = None) -> dict:
    """New ctx whose train/hold/valid frames also carry the g12a columns (inputs are never mutated)."""
    panel = history if isinstance(history, Panel) else build_panel(history)
    if reference is None:
        reference = svd.make_reference(panel, pd.Timestamp(ctx["train"]["_origin"].min()), rank)
    return {**ctx,
            "train": svd.add_features_multi(with_target_date(ctx["train"]), panel, "_origin", rank, reference, cache),
            "hold": svd.add_features_multi(with_target_date(ctx["hold"]), panel, "_origin", rank, reference, cache),
            "valid": svd.add_features(ctx["valid"], panel, ctx["origin"], rank, reference, cache)}


def variant_ctx(aug: dict, name: str) -> dict:
    extra = [c for c in aug["valid"].columns if c.startswith(svd.ALL_PREFIX)]
    return {**aug, "columns": [*aug["columns"], *extra], "variant": name}


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("proj_root", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--variants", required=True, help="comma list of: " + ",".join(VARIANT_SPECS))
    ap.add_argument("--folds", default="3,4")
    ap.add_argument("--seed", type=int, default=20260807)
    ap.add_argument("--n-jobs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--rounds", type=int, default=1500)
    ap.add_argument("--raw-dir", type=Path, default=None)
    return ap.parse_args(argv)


def record_config(name: str) -> SimpleNamespace:
    return SimpleNamespace(**{**base.VARIANTS[CHAMPION].__dict__, "recipe": CHAMPION, "svd_rank": VARIANT_SPECS[name].rank})


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.proj_root.resolve()
    sys.path.insert(0, str(root))
    names = [v.strip() for v in args.variants.split(",")]
    unknown = [v for v in names if v not in VARIANT_SPECS]
    if unknown:
        raise SystemExit(f"unknown variants: {unknown}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    panel = build_panel(load_history(root, args.raw_dir))
    caches: dict[int, dict] = {VARIANT_SPECS[n].rank: {} for n in names}
    results: dict[str, list[dict]] = {n: [] for n in names}
    oofs: dict[str, list[pd.DataFrame]] = {n: [] for n in names}
    for fold in [int(x) for x in args.folds.split(",")]:
        t0 = time.perf_counter()
        ctx = base.prepare_fold(root, fold, args.seed)
        print(f"fold {fold}: loaded in {time.perf_counter() - t0:.0f}s, train rows {len(ctx['train'])}", flush=True)
        for n in names:
            rank = VARIANT_SPECS[n].rank
            t1 = time.perf_counter()
            aug = augment_ctx(ctx, panel, rank, caches[rank])
            print(f"{n}: features in {time.perf_counter() - t1:.0f}s, extra columns {sum(c.startswith(svd.ALL_PREFIX) for c in aug['valid'].columns)}", flush=True)
            oof, scores = base.fit_variant(variant_ctx(aug, n), base.VARIANTS[CHAMPION], args.seed, args.n_jobs, args.lr, args.rounds)
            oofs[n].append(oof)
            results[n].append(scores)
            print(n, "fold", fold, json.dumps(scores), flush=True)
            del aug
        del ctx
    for n in names:
        pd.concat(oofs[n], ignore_index=True).to_csv(args.out_dir / f"oof_{n}_s{args.seed}.csv.gz", index=False, compression="gzip")
        base.write_record(args.out_dir, n, record_config(n), args.seed, results[n], args)
        print("DONE", n, round(sum(r["rmsle_all"] for r in results[n]) / len(results[n]), 5), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
