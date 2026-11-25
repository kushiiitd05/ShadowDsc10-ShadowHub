"""G6c: bounded Optuna study for the LightGBM recipe scored on its blend with TabPFN-TS (weight fixed at 0.2).

Usage (from improve/ on a server; every option has a default, see parse_args):
    python g6c_blend/optuna_study.py --study-name g6c_blend --storage ../optuna_g6c.db --sampler-seed 20260807 \
        --model-seed 20260807 --tabpfn-oof ../tabpfn_oof_mix.csv.gz --tabpfn-oof-alt ../tabpfn_oof_730.csv.gz --n-jobs 24 \
        --n-trials 40 --stop-at 2026-11-26T10:00 --out-dir ../exp_out_G6c

Two different seeds: --sampler-seed drives only Optuna's TPE proposals (which parameter sets are tried); --model-seed
(always 20260807 in both studies) drives LightGBM bagging/feature sampling and the HubID target encoder, so a given
parameter set gives the same fit in either study. Two studies with different sampler seeds can be pooled; their
trials.csv files have the same schema (columns `study`, `sampler_seed`, `model_seed` tell them apart).

Design: TPESampler(multivariate, 10 start-up trials), no pruning, 4 searched LightGBM parameters (feature_fraction
0.10-0.35, num_leaves {63,127,255,383,511}, min_data_in_leaf 60-400, lambda_l2 1-100 log), lr 0.025, 1000 fixed rounds,
bagging 0.85/freq 1, target-encoded HubID, log target. Objective (minimise) = mean over folds 2 and 3 only of the
RMSLE (all rows) of expm1(0.8*log1p(lgbm) + 0.2*log1p(tabpfn)). Folds 1, 4 and 5 are never used here.
"""
from __future__ import annotations

import argparse
import csv
import getpass
import math
import resource
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from helpers import fold_scores  # noqa: E402

SEED = 20260807
STUDY_NAME = "g6c_blend"
FOLDS = (2, 3)
ALLOWED_OBJECTIVE_FOLDS = (2, 3)  # folds 1, 4 and 5 are reserved for confirmation
N_TRIALS = 40
LR = 0.025
ROUNDS = 1000
BLEND_W = 0.2
LEAVES = (63, 127, 255, 383, 511)
DEADLINE = datetime(2026, 11, 26, 10, 0)
EXPECTED_ROWS = 46_830
MAX_CONSECUTIVE_FAILURES = 3
PARAM_NAMES = ("feature_fraction", "num_leaves", "min_data_in_leaf", "lambda_l2")
CONTROLS = (
    {"name": "cs02", "params": {"feature_fraction": 0.2, "num_leaves": 63, "min_data_in_leaf": 80, "lambda_l2": 1.0}},
    {"name": "hp17", "params": {"feature_fraction": 0.15, "num_leaves": 255, "min_data_in_leaf": 160, "lambda_l2": 24.1}},
)
# a run of ours (not another user's) that is using cores: other screening runs and the retrain rehearsal
OUR_RUN_MARKERS = ("exp_runner.py", "g5_runner.py", "final_retrain.py")
_FOLD_FIELDS = ("lgbm_rmsle_all", "lgbm_excl_open0", "blend_rmsle_all", "blend_excl_open0",
                "alt_blend_rmsle_all", "alt_blend_excl_open0", "fit_seconds")


def csv_columns(folds: Sequence[int] = FOLDS) -> tuple[str, ...]:
    return ("study", "sampler_seed", "model_seed", "trial", "state", "tag", *PARAM_NAMES, "blend_w",
            *(f"fold{f}_{k}" for f in folds for k in _FOLD_FIELDS),
            "objective", "lgbm_mean", "mean_blend_excl_open0", "mean_lgbm_excl_open0", "total_fit_seconds",
            "tabpfn_used", "n_jobs", "peak_rss_gb", "finished_at")


# ---------------------------------------------------------------- pure helpers

def suggest_params(trial: Any) -> dict[str, Any]:
    return {
        "feature_fraction": trial.suggest_float("feature_fraction", 0.10, 0.35),
        "num_leaves": trial.suggest_categorical("num_leaves", list(LEAVES)),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 60, 400),
        "lambda_l2": trial.suggest_float("lambda_l2", 1.0, 100.0, log=True),
    }


def config_kwargs(params: Mapping[str, Any]) -> dict[str, Any]:
    """Keyword arguments for exp_runner.Config: searched values plus the fixed recipe."""
    return {
        "colsample": params["feature_fraction"], "num_leaves": params["num_leaves"],
        "min_child": params["min_data_in_leaf"], "reg_lambda": params["lambda_l2"],
        "subsample_freq": 1, "extra_trees": False, "te": True, "target": "log", "fixed_rounds": ROUNDS,
    }


def blend_log(lgbm: np.ndarray, tabpfn: np.ndarray, w: float) -> np.ndarray:
    """Log-space blend; zero stays zero (closed days)."""
    z = (1.0 - w) * np.log1p(np.clip(lgbm, 0, None)) + w * np.log1p(np.clip(tabpfn, 0, None))
    return np.clip(np.expm1(z), 0.0, None)


def align_tabpfn(valid: pd.DataFrame, tab: pd.DataFrame, fold: int, expected_rows: int = EXPECTED_ROWS) -> np.ndarray:
    """TabPFN-TS predictions in the row order of `valid`; every row must match and actuals must agree."""
    sub = tab[tab["fold"] == fold]
    if len(sub) != expected_rows or len(valid) != expected_rows:
        raise ValueError(f"fold {fold}: expected {expected_rows} rows, tabpfn {len(sub)}, valid {len(valid)}")
    keys = pd.DataFrame({"HubID": valid["HubID"].to_numpy(), "Date": pd.to_datetime(valid["Date"]).to_numpy()})
    right = pd.DataFrame({"HubID": sub["HubID"].to_numpy(), "Date": pd.to_datetime(sub["Date"]).to_numpy(),
                          "t_pred": sub["pred"].to_numpy(float), "t_actual": sub["ActualOrderVolume"].to_numpy(float)})
    if right.duplicated(["HubID", "Date"]).any():
        raise ValueError(f"fold {fold}: duplicate (HubID, Date) in tabpfn file")
    merged = keys.merge(right, on=["HubID", "Date"], how="left")
    if merged["t_pred"].isna().any() or len(merged) != len(valid):
        raise ValueError(f"fold {fold}: rows do not match on (HubID, Date)")
    if not np.allclose(merged["t_actual"].to_numpy(), valid["ActualOrderVolume"].to_numpy(float)):
        raise ValueError(f"fold {fold}: actuals disagree between tabpfn file and fold")
    return merged["t_pred"].to_numpy()


def blend_scores(actual: np.ndarray, is_open: np.ndarray, lgbm: np.ndarray, tabpfn: np.ndarray, w: float) -> dict[str, float]:
    base = fold_scores(actual, lgbm, is_open)
    mixed = fold_scores(actual, blend_log(lgbm, tabpfn, w), is_open)
    return {"lgbm_rmsle_all": base["rmsle_all"], "lgbm_excl_open0": base["rmsle_excl_open0"],
            "blend_rmsle_all": mixed["rmsle_all"], "blend_excl_open0": mixed["rmsle_excl_open0"]}


def require_complete(tab: pd.DataFrame, folds: Sequence[int], expected_rows: int = EXPECTED_ROWS, name: str = "tabpfn") -> None:
    """The objective's TabPFN file must hold every row of every objective fold."""
    counts = tab.groupby("fold").size()
    short = {f: int(counts.get(f, 0)) for f in folds if int(counts.get(f, 0)) != expected_rows}
    if short:
        raise ValueError(f"{name} file incomplete, rows per fold (expected {expected_rows}): {short}")


def objective_value(fold_results: Mapping[int, Mapping[str, float]], folds: Sequence[int] = FOLDS) -> float:
    if set(fold_results) != set(folds):
        raise ValueError(f"objective needs exactly folds {tuple(folds)}, got {sorted(fold_results)}")
    vals = [float(fold_results[f]["blend_rmsle_all"]) for f in folds]
    if not all(math.isfinite(v) for v in vals):
        raise ValueError(f"non-finite blended score: {vals}")
    return float(np.mean(vals))


def _mean(fr: Mapping[int, Mapping[str, float]], key: str, folds: Sequence[int]) -> float:
    return float(np.mean([float(fr[f][key]) for f in folds]))


def trial_row(number: int, params: Mapping[str, Any], fold_results: Mapping[int, Mapping[str, Any]], *, study: str,
              sampler_seed: int, model_seed: int, n_jobs: int, peak_rss_gb: float, state: str, finished_at: str, tag: str,
              tabpfn_used: str = "", folds: Sequence[int] = FOLDS) -> dict[str, Any]:
    row: dict[str, Any] = {"study": study, "sampler_seed": sampler_seed, "model_seed": model_seed, "trial": number,
                           "state": state, "tag": tag, **{k: params[k] for k in PARAM_NAMES}, "blend_w": BLEND_W}
    for f in folds:
        for k in _FOLD_FIELDS:
            value = fold_results[f].get(k)
            row[f"fold{f}_{k}"] = "" if value is None else value
    row.update(objective=objective_value(fold_results, folds), lgbm_mean=_mean(fold_results, "lgbm_rmsle_all", folds),
               mean_blend_excl_open0=_mean(fold_results, "blend_excl_open0", folds),
               mean_lgbm_excl_open0=_mean(fold_results, "lgbm_excl_open0", folds),
               total_fit_seconds=round(sum(float(fold_results[f]["fit_seconds"]) for f in folds), 1),
               tabpfn_used=tabpfn_used, n_jobs=n_jobs, peak_rss_gb=peak_rss_gb, finished_at=finished_at)
    return {k: row[k] for k in csv_columns(folds)}


def append_csv_row(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if new:
            writer.writeheader()
        writer.writerow(dict(row))


def _key(params: Mapping[str, Any]) -> tuple:
    return (round(float(params["feature_fraction"]), 9), int(params["num_leaves"]),
            int(params["min_data_in_leaf"]), round(float(params["lambda_l2"]), 9))


def select_finalists(rows: Sequence[Mapping[str, Any]], k: int = 5) -> list[dict[str, Any]]:
    """Top k completed trials by objective over any number of pooled studies; controls and duplicates are dropped."""
    skip = {_key(c["params"]) for c in CONTROLS}
    seen: set[tuple] = set()
    kept = []
    for r in sorted((r for r in rows if r.get("state") == "COMPLETE"),
                    key=lambda r: (float(r["objective"]), str(r.get("study", "")), int(r["trial"]))):
        key = _key(r)
        if key in skip or key in seen:
            continue
        seen.add(key)
        kept.append(dict(r))
    return kept[:k]


def variant_name(trial_number: int, tag: str = "") -> str:
    return f"g6c_{tag + '_' if tag else ''}t{int(trial_number)}"


def variant_source_line(row: Mapping[str, Any], tag: str = "") -> str:
    """One exp_runner VARIANTS entry (full float precision) for a finalist's LightGBM recipe."""
    kw = config_kwargs({k: row[k] for k in PARAM_NAMES})
    return (f'    "{variant_name(row["trial"], tag)}": Config(colsample={float(kw["colsample"])!r}, subsample_freq=1, '
            f'num_leaves={int(kw["num_leaves"])}, min_child={int(kw["min_child"])}, reg_lambda={float(kw["reg_lambda"])!r}, '
            f'fixed_rounds={ROUNDS}),')


def choose_n_jobs(ps_lines: Sequence[str]) -> int:
    """24 threads while one of our other runs (screening, per-tier, retrain rehearsal) is alive, else 32."""
    return 24 if any(m in line for line in ps_lines for m in OUR_RUN_MARKERS) else 32


def stop_reason(now: datetime, n_complete: int, deadline: datetime = DEADLINE, budget: int = N_TRIALS) -> str | None:
    if now >= deadline:
        return "deadline reached"
    if n_complete >= budget:
        return "trial budget reached"
    return None


# ---------------------------------------------------------------- optuna glue

def build_study(storage_url: str, study_name: str = STUDY_NAME, sampler_seed: int = SEED):
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=sampler_seed, multivariate=True, n_startup_trials=10)
    study = optuna.create_study(study_name=study_name, storage=storage_url, sampler=sampler, direction="minimize",
                                pruner=optuna.pruners.NopPruner(), load_if_exists=True)
    if not study.get_trials(deepcopy=False):
        for control in CONTROLS:
            study.enqueue_trial(dict(control["params"]), user_attrs={"control": control["name"]}, skip_if_exists=False)
    return study


def count_complete(study) -> int:
    import optuna

    return len(study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,)))


def fail_stale_running(study) -> int:
    """After a crash a trial can stay RUNNING for ever; mark it FAIL so the study can go on."""
    import optuna

    stale = study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.RUNNING,))
    for t in stale:
        study._storage.set_trial_state_values(t._trial_id, optuna.trial.TrialState.FAIL)
    return len(stale)


def _our_ps_lines() -> list[str]:
    out = subprocess.run(["ps", "-u", getpass.getuser(), "-o", "pid=,args="], capture_output=True, text=True, check=False)
    return out.stdout.splitlines()


def _peak_rss_gb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(peak / (1024 ** 3 if sys.platform == "darwin" else 1024 ** 2), 2)


def _tag(params: Mapping[str, Any]) -> str:
    return next((c["name"] for c in CONTROLS if _key(c["params"]) == _key(params)), "")


def make_objective(contexts: Mapping[int, dict], tab_pred: Mapping[int, np.ndarray], alt_pred: Mapping[int, np.ndarray] | None,
                   csv_path: Path, args: argparse.Namespace, tabpfn_used: str):
    import exp_runner

    folds = args.folds_objective

    def objective(trial) -> float:
        params = suggest_params(trial)
        cfg = exp_runner.Config(**config_kwargs(params))
        n_jobs = args.n_jobs or choose_n_jobs(_our_ps_lines())
        results = {}
        for f in folds:
            valid = contexts[f]["valid"]
            oof, fit = exp_runner.fit_variant(contexts[f], cfg, args.model_seed, n_jobs, LR, ROUNDS)
            pred = oof["pred"].to_numpy(float)
            actual, is_open = valid["ActualOrderVolume"].to_numpy(float), valid["IsOpen"].to_numpy()
            res = blend_scores(actual, is_open, pred, tab_pred[f], BLEND_W)
            res["fit_seconds"] = float(fit["fit_seconds"])
            if alt_pred is not None:
                alt = blend_scores(actual, is_open, pred, alt_pred[f], BLEND_W)
                res.update(alt_blend_rmsle_all=alt["blend_rmsle_all"], alt_blend_excl_open0=alt["blend_excl_open0"])
            results[f] = res
            for k, v in res.items():
                trial.set_user_attr(f"fold{f}_{k}", v)
            print(f"trial {trial.number} fold {f} n_jobs {n_jobs} lgbm {res['lgbm_rmsle_all']:.5f} blend {res['blend_rmsle_all']:.5f} "
                  f"(excl-open0 {res['lgbm_excl_open0']:.5f} / {res['blend_excl_open0']:.5f}, fit {res['fit_seconds']}s) {params}",
                  flush=True)
        row = trial_row(trial.number, params, results, study=args.study_name, sampler_seed=args.sampler_seed,
                        model_seed=args.model_seed, n_jobs=n_jobs, peak_rss_gb=_peak_rss_gb(), state="COMPLETE",
                        finished_at=datetime.now().isoformat(timespec="seconds"), tag=_tag(params), tabpfn_used=tabpfn_used,
                        folds=folds)
        trial.set_user_attr("n_jobs", n_jobs)
        trial.set_user_attr("peak_rss_gb", row["peak_rss_gb"])
        append_csv_row(csv_path, row)
        return row["objective"]

    return objective


def run(args: argparse.Namespace) -> int:
    root = args.proj_root.resolve()
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import exp_runner

    folds = args.folds_objective
    args.out_dir.mkdir(parents=True, exist_ok=True)
    objective_tab = pd.read_csv(args.tabpfn_oof)
    require_complete(objective_tab, folds, name="objective")
    alt_tab = pd.read_csv(args.tabpfn_oof_alt) if args.tabpfn_oof_alt else None
    if alt_tab is not None:
        require_complete(alt_tab, folds, name="alt")
    use = args.tabpfn_oof.name
    print(f"TABPFN objective file: {args.tabpfn_oof} ; alt (logged only): {args.tabpfn_oof_alt}", flush=True)
    (args.out_dir / "tabpfn_files.txt").write_text(f"objective={args.tabpfn_oof}\nalt={args.tabpfn_oof_alt}\n")
    study = build_study(f"sqlite:///{args.storage.resolve()}", args.study_name, args.sampler_seed)
    print(f"study {args.study_name} (sampler seed {args.sampler_seed}, model seed {args.model_seed}): "
          f"{len(study.trials)} trials on disk, stale fixed: {fail_stale_running(study)}", flush=True)
    contexts = {}
    for f in folds:
        t0 = time.perf_counter()
        contexts[f] = exp_runner.prepare_fold(root, f, args.model_seed)
        print(f"fold {f} loaded in {time.perf_counter() - t0:.0f}s, train rows {len(contexts[f]['train'])}, "
              f"peak RSS {_peak_rss_gb()} GB", flush=True)
    tab_pred = {f: align_tabpfn(contexts[f]["valid"], objective_tab, f) for f in folds}
    alt_pred = {f: align_tabpfn(contexts[f]["valid"], alt_tab, f) for f in folds} if alt_tab is not None else None
    print("tabpfn join checks passed for folds", folds, flush=True)
    objective = make_objective(contexts, tab_pred, alt_pred, args.out_dir / "trials.csv", args, use)
    failures, reason = 0, None
    while reason is None:
        reason = stop_reason(datetime.now(), count_complete(study), args.stop_at, args.n_trials)
        if reason:
            break
        try:
            study.optimize(objective, n_trials=1)
            failures = 0
        except Exception:  # noqa: BLE001 - keep the study alive, stop after repeated failures
            traceback.print_exc()
            failures += 1
            if failures >= MAX_CONSECUTIVE_FAILURES:
                reason = f"{failures} consecutive failures"
    done = count_complete(study)
    best = study.best_trial if done else None
    print(f"STOP: {reason}; completed {done} trials; best {best.number if best else None} "
          f"{best.value if best else None}", flush=True)
    return 0 if reason in ("trial budget reached", "deadline reached") else 1


def _folds(text: str) -> tuple[int, ...]:
    try:
        folds = tuple(sorted({int(x) for x in text.split(",")}))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"not a fold list: {text!r}") from exc
    if not folds or not set(folds) <= set(ALLOWED_OBJECTIVE_FOLDS):
        raise argparse.ArgumentTypeError(f"objective folds must be a non-empty subset of {ALLOWED_OBJECTIVE_FOLDS}, got {text!r}")
    return folds


def _positive(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return value


def _stop_at(text: str) -> datetime:
    try:
        return datetime.fromisoformat(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"not an ISO date-time: {text!r}") from exc


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--proj-root", type=Path, default=Path("../proj"))
    ap.add_argument("--study-name", default=STUDY_NAME)
    ap.add_argument("--storage", type=Path, default=Path("../optuna_g6c.db"), help="SQLite file (resumable)")
    ap.add_argument("--sampler-seed", type=int, default=SEED, help="Optuna TPE proposal seed (differs between studies)")
    ap.add_argument("--model-seed", type=int, default=SEED, help="LightGBM / target-encoder seed (same in every study)")
    ap.add_argument("--tabpfn-oof", type=Path, default=Path("../tabpfn_oof_mix.csv.gz"), help="TabPFN-TS OOF file used in the objective")
    ap.add_argument("--tabpfn-oof-alt", type=Path, default=None, help="second OOF file; its blend is logged (alt_*), never optimised")
    ap.add_argument("--n-trials", type=_positive, default=N_TRIALS)
    ap.add_argument("--stop-at", type=_stop_at, default=DEADLINE, help="ISO local date-time; no new trial after it")
    ap.add_argument("--n-jobs", type=_positive, default=None, help="LightGBM threads; default 24 while our other runs are alive, else 32")
    ap.add_argument("--out-dir", type=Path, default=Path("../exp_out_G6c"))
    ap.add_argument("--folds-objective", type=_folds, default=FOLDS)
    return ap.parse_args(argv)


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
