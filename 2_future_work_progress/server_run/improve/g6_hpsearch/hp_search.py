"""Random hyper-parameter search around the cs02 recipe (no extra dependency).

Usage: python g6_hpsearch/hp_search.py <proj_root> <out_dir> --n 24 --folds 1,2,3,4 --seed 20260807 --n-jobs 24 [--sample-seed 1]

Draws N configs (num_leaves, min_child, reg_lambda, colsample, extra_trees), registers them as variants
`hp_000..` in exp_runner.VARIANTS, then runs exp_runner's own loop so scoring, folds and logging are
identical to every other experiment. Screening only: the winner must pass the gate (3 seeds) before adoption.
"""
from __future__ import annotations

import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import exp_runner  # noqa: E402
from exp_runner import Config  # noqa: E402

LEAVES = (31, 63, 127, 255)
MIN_CHILD = (40, 80, 160, 320)


def sample_configs(n: int, sample_seed: int) -> dict[str, Config]:
    """Deterministic draw of n distinct configs; the first one is the cs02 control."""
    rng = random.Random(sample_seed)
    out: dict[str, Config] = {"hp_000": Config(colsample=0.2, subsample_freq=1)}
    seen = {out["hp_000"]}
    while len(out) < n:
        cfg = Config(
            colsample=round(rng.choice([0.1, 0.15, 0.2, 0.3, 0.4]), 2),
            subsample_freq=1,
            num_leaves=rng.choice(LEAVES),
            min_child=rng.choice(MIN_CHILD),
            reg_lambda=round(math.exp(rng.uniform(math.log(0.3), math.log(30.0))), 2),
            extra_trees=rng.random() < 0.25,
        )
        if cfg not in seen:
            seen.add(cfg)
            out[f"hp_{len(out):03d}"] = cfg
    return out


def main() -> int:
    argv = sys.argv[1:]
    n = int(argv[argv.index("--n") + 1]); del argv[argv.index("--n"): argv.index("--n") + 2]
    sample_seed = 1
    if "--sample-seed" in argv:
        i = argv.index("--sample-seed"); sample_seed = int(argv[i + 1]); del argv[i: i + 2]
    configs = sample_configs(n, sample_seed)
    exp_runner.VARIANTS.update(configs)
    sys.argv = [sys.argv[0], *argv, "--variants", ",".join(configs)]
    return exp_runner.main()


if __name__ == "__main__":
    raise SystemExit(main())
