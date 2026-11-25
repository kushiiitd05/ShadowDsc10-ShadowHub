"""TabPFN-TS member (venv_nf only): per-hub zero-shot forecasting with a local TabPFN v2 backbone.

The v3.5 / v2.5 checkpoints that tabpfn-time-series defaults to require a one-time interactive licence acceptance
(browser login) that is impossible on a headless server, so the pilot pins the TabPFN v2 checkpoint, which
needs no acceptance. Same inputs as the neural members: history up to the origin, known-future flags + calendar.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from neural_prep import build_future_frame, build_panel, impute_closed

COVARIATES = ["open", "promo", "holiday", "school"]


def to_tabpfn_frames(panel: pd.DataFrame, future: pd.DataFrame, context: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(context_df, future_df) in the tabpfn-time-series layout: item_id, timestamp, target + known covariates.

    The context is the last `context` days of each hub's panel (already limited to the origin); the future frame carries
    no target. Pure; inputs are not mutated.
    """
    ctx = panel.sort_values(["unique_id", "ds"]).groupby("unique_id", sort=False).tail(context)
    context_df = ctx.rename(columns={"unique_id": "item_id", "ds": "timestamp", "y": "target"})[["item_id", "timestamp", "target", *COVARIATES]]
    future_df = future.rename(columns={"unique_id": "item_id", "ds": "timestamp"})[["item_id", "timestamp", *COVARIATES]]
    return context_df.reset_index(drop=True), future_df.reset_index(drop=True)


def parse_tabpfn_output(pred: pd.DataFrame) -> pd.DataFrame:
    """tabpfn-time-series output (indexed by item_id, timestamp; 'target' = point forecast) -> unique_id, ds, log_pred."""
    frame = pred.reset_index() if "item_id" not in pred.columns else pred.copy()
    return pd.DataFrame({"unique_id": frame["item_id"].astype(str).to_numpy(), "ds": pd.to_datetime(frame["timestamp"]).to_numpy(),
                         "log_pred": frame["target"].to_numpy(dtype="float64")})


def predict_tabpfn_ts(hist: pd.DataFrame, valid: pd.DataFrame, origin: pd.Timestamp, args) -> pd.DataFrame:
    import torch
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion
    from tabpfn_time_series import TabPFNMode, TabPFNTSPipeline

    panel = build_panel(hist, origin)
    if args.impute:
        panel = impute_closed(panel)
    future = build_future_frame(valid, origin, 42)
    context_df, future_df = to_tabpfn_frames(panel, future, args.context)
    model_path = TabPFNRegressor.create_default_for_version(ModelVersion.V2).model_path
    pipe = TabPFNTSPipeline(max_context_length=args.context, tabpfn_mode=TabPFNMode.LOCAL, tabpfn_output_selection="mean",
                            tabpfn_model_config={"model_path": model_path}, num_workers_per_gpu=1, compile=False)
    out = pipe.predict_df(context_df, future_df=future_df, quantiles=[0.5])
    del pipe
    torch.cuda.empty_cache()
    return parse_tabpfn_output(out)


def predict_tabpfn(*_args, **_kwargs):
    raise NotImplementedError("tabular TabPFN v2 fallback is only built if the TabPFN-TS path fails")
