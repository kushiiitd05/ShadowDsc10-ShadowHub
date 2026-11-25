"""Horizon-segmented fitting on top of exp_runner.fit_variant (stage G2).

A SplitSpec describes a base config (cs02) plus a partition of Horizon 1..42 into segments. Each segment gets its
own LightGBM fit through the unchanged harness `fit_variant`, on a sub-context holding only that segment's rows
(training origins, chronological holdout and validation rows alike), optionally with the restricted long-window
column set. Predictions are stitched back into the original validation row order. Pure helpers never mutate inputs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from column_classes import describe_split, split_columns
from helpers import fold_scores

MAX_HORIZON = 42
# columns fit_variant needs besides the feature list (labels / bookkeeping of the already-split train, hold, valid frames)
KEEP_ALWAYS = ("HubID", "Date", "Horizon", "IsOpen", "ActualOrderVolume", "_level", "_origin", "HubID_TE")
OOF_COLUMNS = ["HubID", "Date", "Horizon", "IsOpen", "ActualOrderVolume"]


@dataclass(frozen=True)
class Segment:
    lo: int
    hi: int
    restrict: bool = False  # True: drop short-window columns for this segment (long_window_only)


@dataclass(frozen=True)
class SplitSpec:
    base: str                       # name of an exp_runner.VARIANTS config
    segments: tuple[Segment, ...]


G2_VARIANTS: dict[str, SplitSpec] = {
    "cs02": SplitSpec("cs02", (Segment(1, MAX_HORIZON),)),                                   # control through the same code path
    "split_horizon": SplitSpec("cs02", (Segment(1, 21), Segment(22, MAX_HORIZON))),
    "long_window_only": SplitSpec("cs02", (Segment(1, 21), Segment(22, MAX_HORIZON, True))),  # short model keeps every column
}


def validate_segments(segments: tuple[Segment, ...]) -> None:
    """Segments must tile Horizon 1..42 contiguously with no overlap."""
    if not segments or segments[0].lo != 1 or segments[-1].hi != MAX_HORIZON:
        raise ValueError(f"segments must span 1..{MAX_HORIZON}: {segments}")
    for seg in segments:
        if seg.lo > seg.hi:
            raise ValueError(f"empty segment {seg}")
    for left, right in zip(segments, segments[1:]):
        if right.lo != left.hi + 1:
            raise ValueError(f"segments must be contiguous and disjoint: {left} -> {right}")


def segment_label(segments: tuple[Segment, ...]) -> str:
    """Compact text for the config record, e.g. '1-21|22-42r' (r = restricted columns)."""
    return "|".join(f"{s.lo}-{s.hi}{'r' if s.restrict else ''}" for s in segments)


def subset_context(ctx: dict, segment: Segment, columns: list[str]) -> dict:
    """New context with rows of Horizon in [lo, hi] and only the needed columns. Row labels are kept for stitching."""
    keep = list(dict.fromkeys([*columns, *KEEP_ALWAYS]))
    out = {k: v for k, v in ctx.items() if k not in ("train", "hold", "valid", "columns")}
    for key in ("train", "hold", "valid"):
        frame = ctx[key]
        mask = frame["Horizon"].between(segment.lo, segment.hi)
        if not mask.any():
            raise ValueError(f"no {key} rows with Horizon in {segment.lo}-{segment.hi}")
        out[key] = frame.loc[mask, [c for c in keep if c in frame.columns]]
    out["columns"] = list(columns)
    return out


def merge_oof(parts: list[pd.DataFrame], index: pd.Index) -> pd.DataFrame:
    """Concatenate per-segment prediction frames and restore the original validation row order."""
    merged = pd.concat(parts)
    if len(merged) != len(index) or merged.index.has_duplicates or not merged.index.isin(index).all():
        raise ValueError("segments must exactly cover the validation rows once")
    return merged.loc[index]


def _segment_columns(ctx: dict, segment: Segment, log: Callable[..., None]) -> list[str]:
    columns = list(ctx["columns"])
    if not segment.restrict:
        return columns
    kept, _ = split_columns(columns)
    log(f"segment {segment.lo}-{segment.hi}: restricted columns\n{describe_split(columns)}")
    return kept


def fit_segmented(ctx: dict, spec: SplitSpec, cfg, seed: int, n_jobs: int, lr: float, rounds: int,
                  fit: Callable, log: Callable[..., None] = print) -> tuple[pd.DataFrame, dict]:
    """Fit one model per segment with `fit` (= exp_runner.fit_variant) and return the stitched (oof, scores)."""
    validate_segments(spec.segments)
    parts: list[pd.DataFrame] = []
    part_scores: list[dict] = []
    for segment in spec.segments:  # one sub-context alive at a time to bound memory
        sub = subset_context(ctx, segment, _segment_columns(ctx, segment, log))
        oof, scores = fit(sub, cfg, seed, n_jobs, lr, rounds)
        parts.append(oof[[*OOF_COLUMNS, "pred"]])
        part_scores.append(scores)
        log(f"  segment {segment.lo}-{segment.hi}{' (restricted)' if segment.restrict else ''}: "
            f"best_iter {scores['best_iteration']}, train_rows {scores['train_rows']}, fit {scores['fit_seconds']}s")
        del sub
    merged = merge_oof(parts, ctx["valid"].index)
    merged["fold"] = ctx["fold"]
    valid = ctx["valid"]
    scores = fold_scores(valid["ActualOrderVolume"].to_numpy(float), merged["pred"].to_numpy(), valid["IsOpen"].to_numpy())
    scores.update(fold=ctx["fold"], best_iteration=int(round(float(np.mean([s["best_iteration"] for s in part_scores])))),
                  fit_seconds=round(sum(s["fit_seconds"] for s in part_scores), 1),
                  train_rows=int(sum(s["train_rows"] for s in part_scores)), offset=0.0)
    return merged, scores
