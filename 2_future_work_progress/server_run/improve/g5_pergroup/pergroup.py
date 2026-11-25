"""Per-group models (stage 1b.10): one cs02 LightGBM per HubFormat (or AssortmentTier), stitched into one prediction frame.

Each group is fitted by the unchanged harness `fit_variant` on a sub-context holding only that group's hubs (training origins,
chronological holdout and validation rows alike). Groups with too few training rows fall back to the global model's context
(the whole frame) so every validation row is predicted exactly once.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MIN_TRAIN_ROWS = 20_000
KEEP_ALWAYS = ("HubID", "Date", "Horizon", "IsOpen", "ActualOrderVolume", "_level", "_origin", "HubID_TE")
OOF_COLUMNS = ["HubID", "Date", "Horizon", "IsOpen", "ActualOrderVolume"]


def group_values(ctx: dict, column: str, min_rows: int = MIN_TRAIN_ROWS) -> tuple[list, list]:
    """(groups large enough for their own model, groups too small); size = training rows of that group."""
    counts = ctx["train"][column].value_counts()
    big = [g for g, n in counts.items() if n >= min_rows]
    small = [g for g in ctx["valid"][column].dropna().unique() if g not in big]
    return big, small


def subset_by_values(ctx: dict, column: str, values: list) -> dict:
    """New context with rows whose `column` is in `values` (frames keep their original index for stitching)."""
    out = {k: v for k, v in ctx.items() if k not in ("train", "hold", "valid")}
    for key in ("train", "hold", "valid"):
        frame = ctx[key]
        out[key] = frame.loc[frame[column].isin(values)]
        if key != "valid" and out[key].empty:
            raise ValueError(f"no {key} rows for {column} in {values}")
    return out


def stitch(parts: list[pd.DataFrame], index: pd.Index) -> pd.DataFrame:
    merged = pd.concat(parts)
    if len(merged) != len(index) or merged.index.has_duplicates or not merged.index.isin(index).all():
        raise ValueError("groups must cover every validation row exactly once")
    return merged.loc[index]


def fit_by_group(ctx: dict, column: str, cfg, seed: int, n_jobs: int, lr: float, rounds: int, fit, log=print) -> tuple[pd.DataFrame, dict]:
    """One model per large group; all small groups together share one model trained on their pooled rows."""
    from helpers import fold_scores
    big, small = group_values(ctx, column)
    jobs = [[g] for g in big] + ([small] if small else [])
    parts, part_scores = [], []
    for values in jobs:
        sub = subset_by_values(ctx, column, values)
        if sub["valid"].empty:
            continue
        oof, scores = fit(sub, cfg, seed, n_jobs, lr, rounds)
        parts.append(oof[[*OOF_COLUMNS, "pred"]])
        part_scores.append(scores)
        log(f"  {column}={values}: train_rows {scores['train_rows']}, best_iter {scores['best_iteration']}, fit {scores['fit_seconds']}s")
    merged = stitch(parts, ctx["valid"].index)
    merged["fold"] = ctx["fold"]
    valid = ctx["valid"]
    scores = fold_scores(valid["ActualOrderVolume"].to_numpy(float), merged["pred"].to_numpy(), valid["IsOpen"].to_numpy())
    scores.update(fold=ctx["fold"], best_iteration=int(round(float(np.mean([s["best_iteration"] for s in part_scores])))),
                  fit_seconds=round(sum(s["fit_seconds"] for s in part_scores), 1),
                  train_rows=int(sum(s["train_rows"] for s in part_scores)), offset=0.0)
    return merged, scores
