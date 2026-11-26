"""End-term experiments: rows (uniform / hub-stratified / sensitivity coreset) and columns (importance / uniform), LightGBM and ridge.

Usage: python endterm_exp.py <proj_root> <out_dir> [--stage rows,cols] [--reps 5] [--n-jobs 16] [--sizes 2000,...]
Fixed split: P = training origins whose label windows end before fold-4's origin (minus the tuning holdout), test = fold 4.
Loss f_i = squared error on log1p(volume) over open rows with positive volume. Every fit uses the same settings and a fixed
number of rounds chosen once on the training-only holdout. Weights are normalised to mean 1. Results are appended to results.csv
(resume-safe: finished keys are skipped).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "improve"))
from coreset import (column_importance_subset, hub_stratified_indices, relative_gap, relative_l2, sample_coreset,  # noqa: E402
                     sensitivities, uniform_indices, weighted_ridge)

SEED0 = 20261125
SIZES = [2_000, 5_000, 10_000, 25_000, 50_000, 100_000, 200_000]
COL_SIZES = [10, 20, 40, 80, 160]
COL_ROWS = 200_000
LGB = dict(learning_rate=0.05, num_leaves=63, min_child_samples=80, subsample=0.85, subsample_freq=1, colsample_bytree=0.2, reg_lambda=1.0)
ROUGH = dict(learning_rate=0.1, num_leaves=15, min_child_samples=20, colsample_bytree=0.5, n_estimators=60)


class Data:
    """Training matrix P (open, positive-volume rows), tuning holdout and the fixed test set."""

    def __init__(self, root: Path, n_jobs: int) -> None:
        from exp_runner import prep, prepare_fold
        from src.model_pipeline import CATEGORICAL_COLUMNS

        ctx = prepare_fold(root, 4, SEED0)
        self.n_jobs = n_jobs
        feats = [*ctx["columns"], "HubID_TE"]
        cats = set(CATEGORICAL_COLUMNS)
        pos = lambda f: f["IsOpen"].eq(1) & f["ActualOrderVolume"].gt(0)  # noqa: E731
        mt, mh = pos(ctx["train"]), pos(ctx["hold"])
        self.x, self.xh = prep(ctx["train"].loc[mt], feats, cats), prep(ctx["hold"].loc[mh], feats, cats)
        self.y = np.log1p(ctx["train"].loc[mt, "ActualOrderVolume"].to_numpy(float))
        self.yh = np.log1p(ctx["hold"].loc[mh, "ActualOrderVolume"].to_numpy(float))
        self.hubs = ctx["train"].loc[mt, "HubID"].to_numpy()
        valid = ctx["valid"]
        self.open = valid["IsOpen"].eq(1).to_numpy()
        self.actual = valid["ActualOrderVolume"].to_numpy(float)
        self.xt = prep(valid.loc[self.open], feats, cats)
        self.yt = np.log1p(self.actual[self.open])
        self.feats = feats
        numeric = [c for c in feats if str(self.x[c].dtype) not in ("category", "object", "string")]
        self.numeric = numeric
        med = self.x[numeric].median()
        scale = self.x[numeric].fillna(med).std().replace(0, 1.0)
        self.num = lambda f: ((f[numeric].fillna(med) - med) / scale).to_numpy(np.float32)  # noqa: E731
        self.n = len(self.y)


def test_metrics(d: Data, pred_open: np.ndarray) -> dict:
    pred = np.zeros(len(d.actual))
    pred[d.open] = np.clip(np.expm1(pred_open), 0, None)
    err = np.log1p(pred) - np.log1p(d.actual)
    zero = d.open & (d.actual == 0)
    var = np.var(np.log1p(d.actual))
    return {"test_rmsle": float(np.sqrt(np.mean(err ** 2))), "test_r2": float(1 - np.mean(err ** 2) / var),
            "test_rmsle_excl_open0": float(np.sqrt(np.mean(err[~zero] ** 2)))}


class LgbModel:
    name = "lgbm"

    def __init__(self, d: Data, rounds: int, cols: list[str] | None = None) -> None:
        self.d, self.rounds, self.cols = d, rounds, cols or d.feats

    def fit(self, idx: np.ndarray, w: np.ndarray | None, seed: int, params: dict | None = None, rounds: int | None = None):
        import lightgbm as lgb
        p = {**LGB, **(params or {})}
        p.setdefault("n_estimators", rounds or self.rounds)
        m = lgb.LGBMRegressor(random_state=seed, n_jobs=self.d.n_jobs, verbosity=-1, **p)
        m.fit(self.d.x.iloc[idx][self.cols], self.d.y[idx], sample_weight=None if w is None else w / w.mean())
        return m

    def predict(self, m, frame: pd.DataFrame) -> np.ndarray:
        return m.predict(frame[self.cols])


class RidgeModel:
    name = "ridge"

    def __init__(self, d: Data, lam: float) -> None:
        self.d, self.lam = d, lam
        self.xn = d.num(d.x)

    def fit(self, idx: np.ndarray, w: np.ndarray | None, seed: int, params: dict | None = None, rounds: int | None = None):
        w = np.ones(len(idx)) if w is None else w / w.mean()
        return weighted_ridge(self.xn[idx], self.d.y[idx], w, self.lam)

    def predict(self, theta: np.ndarray, frame: pd.DataFrame) -> np.ndarray:
        return self.d.num(frame) @ theta[:-1] + theta[-1]


def tune(d: Data, n_jobs: int) -> tuple[int, float]:
    """Rounds for LightGBM (early stopping on the training-only holdout) and lambda for ridge (grid on the same holdout)."""
    import lightgbm as lgb
    m = lgb.LGBMRegressor(n_estimators=1500, random_state=SEED0, n_jobs=n_jobs, verbosity=-1, **LGB)
    m.fit(d.x, d.y, eval_set=[(d.xh, d.yh)], callbacks=[lgb.early_stopping(100, verbose=False)])
    rounds = int(m.best_iteration_ or 1500)
    rr, best = RidgeModel(d, 0.0), (np.inf, 1.0)
    allidx = np.arange(d.n)
    for lam in (1e-4, 1e-3, 1e-2, 1e-1, 1.0):
        rr.lam = lam
        theta = rr.fit(allidx, None, 0)
        score = float(np.mean((rr.predict(theta, d.xh) - d.yh) ** 2))
        best = min(best, (score, lam))
    return rounds, best[1]


def mean_loss(model, fitted, frame: pd.DataFrame, y: np.ndarray) -> float:
    return float(np.mean((model.predict(fitted, frame) - y) ** 2))


def run_rows(d: Data, model, out: "Writer", reps: int, sizes: list[int], hist_dir: Path) -> None:
    allidx = np.arange(d.n)
    if not out.done(model.name, "full", d.n, 0):
        t0 = time.perf_counter()
        xstar = model.fit(allidx, None, SEED0)
        t_full = time.perf_counter() - t0
        pt = model.predict(xstar, d.xt)
        out.add(model=model.name, method="full", m=d.n, rep=0, t_train=t_full, t_build=0.0, opt=mean_loss(model, xstar, d.x, d.y),
                test_loss_mean=float(np.mean((pt - d.yt) ** 2)), **test_metrics(d, pt))
        np.save(hist_dir / f"xstar_testpred_{model.name}.npy", pt)
        if model.name == "ridge":
            np.save(hist_dir / "xstar_theta_ridge.npy", xstar)
    row = out.get(model.name, "full", d.n, 0)
    opt, ref_test, pt_star = row["opt"], row["test_loss_mean"], np.load(hist_dir / f"xstar_testpred_{model.name}.npy")
    theta_star = np.load(hist_dir / "xstar_theta_ridge.npy") if model.name == "ridge" else None
    for rep in range(reps):
        seed = SEED0 + 1000 * (rep + 1)
        rng = np.random.default_rng(seed)
        s = None
        for method in ("uniform", "hub_stratified", "sensitivity"):
            for m in sizes:
                if out.done(model.name, method, m, rep):
                    continue
                t0 = time.perf_counter()
                extra = {}
                if method == "sensitivity":
                    if s is None:  # five rough models from small uniform samples -> sensitivities (built once per repetition)
                        tb = time.perf_counter()
                        losses, alphas = [], []
                        for t in range(5):
                            r = np.random.default_rng(seed + t)
                            sub = uniform_indices(d.n, max(10_000, d.n // 50), r)
                            if model.name == "lgbm":
                                rough = model.fit(sub, None, seed + t, params=ROUGH, rounds=ROUGH["n_estimators"])
                            else:
                                rough = model.fit(sub, None, seed + t)
                            f = (model.predict(rough, d.x) - d.y) ** 2
                            losses.append(f)
                            alphas.append(float(f.mean() / opt))
                        s = sensitivities(np.vstack(losses))
                        t_sens = time.perf_counter() - tb
                        np.save(hist_dir / f"sens_{model.name}_rep{rep}.npy", (s / s.sum()).astype(np.float32))
                        s_meta = {"S": float(s.sum()), "alpha_max": float(max(alphas))}
                    ts = time.perf_counter()
                    idx, w = sample_coreset(s, m, rng)
                    extra = {**s_meta, "t_build_extra": t_sens, "t_sample": time.perf_counter() - ts}
                elif method == "uniform":
                    idx, w = uniform_indices(d.n, m, rng), None
                else:
                    idx, w = hub_stratified_indices(d.hubs, m, rng), None
                t_build = extra["t_sample"] + extra["t_build_extra"] if method == "sensitivity" else time.perf_counter() - t0
                t1 = time.perf_counter()
                fitted = model.fit(idx, w, seed)
                t_train = time.perf_counter() - t1
                pt = model.predict(fitted, d.xt)
                train_loss = mean_loss(model, fitted, d.x, d.y)
                rec = dict(model=model.name, method=method, m=m, rep=rep, t_build=t_build, t_train=t_train, opt=opt,
                           train_gap=relative_gap(train_loss, opt), test_gap=abs(float(np.mean((pt - d.yt) ** 2)) / opt - 1.0),
                           test_gap_vs_test_star=abs(float(np.mean((pt - d.yt) ** 2)) / ref_test - 1.0),
                           test_pred_l2=relative_l2(pt, pt_star), **test_metrics(d, pt), **{k: v for k, v in extra.items() if k not in ("t_build_extra", "t_sample")})
                if theta_star is not None:
                    rec["param_l2"] = relative_l2(fitted, theta_star)
                out.add(**rec)


def run_cols(d: Data, rounds: int, out: "Writer", reps: int) -> None:
    """Rows held fixed at COL_ROWS uniform draws per repetition; vary the number of kept columns."""
    for rep in range(reps):
        seed = SEED0 + 5000 * (rep + 1)
        rng = np.random.default_rng(seed)
        rows = uniform_indices(d.n, COL_ROWS, rng)
        base = LgbModel(d, rounds)
        t0 = time.perf_counter()
        rough = base.fit(uniform_indices(d.n, 20_000, rng), None, seed, params=ROUGH, rounds=ROUGH["n_estimators"])
        scores = rough.booster_.feature_importance("gain")
        t_scores = time.perf_counter() - t0
        for k in COL_SIZES:
            for method in ("importance", "uniform"):
                if out.done("lgbm_cols", method, k, rep):
                    continue
                t1 = time.perf_counter()
                cols = column_importance_subset(scores, k) if method == "importance" else np.sort(rng.choice(len(d.feats), size=k, replace=False))
                names = [d.feats[i] for i in cols]
                t_build = time.perf_counter() - t1 + (t_scores if method == "importance" else 0.0)
                model = LgbModel(d, rounds, names)
                t2 = time.perf_counter()
                fitted = model.fit(rows, None, seed)
                t_train = time.perf_counter() - t2
                pt = model.predict(fitted, d.xt)
                out.add(model="lgbm_cols", method=method, m=k, rep=rep, t_build=t_build, t_train=t_train, rows_fixed=COL_ROWS, **test_metrics(d, pt))


class Writer:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.rows = pd.read_csv(path).to_dict("records") if path.exists() else []

    def done(self, model: str, method: str, m: int, rep: int) -> bool:
        return self.get(model, method, m, rep) is not None

    def get(self, model: str, method: str, m: int, rep: int) -> dict | None:
        return next((r for r in self.rows if r["model"] == model and r["method"] == method and int(r["m"]) == int(m) and int(r["rep"]) == int(rep)), None)

    def add(self, **row) -> None:
        self.rows.append(row)
        pd.DataFrame(self.rows).to_csv(self.path, index=False)
        print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("proj_root", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--stage", default="rows,cols")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--n-jobs", type=int, default=16)
    ap.add_argument("--sizes", default=",".join(map(str, SIZES)))
    args = ap.parse_args()
    root = args.proj_root.resolve()
    sys.path.insert(0, str(root))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    d = Data(root, args.n_jobs)
    print(f"data ready: P={d.n} rows, {len(d.feats)} features, test={len(d.actual)} rows, load {time.perf_counter() - t0:.0f}s", flush=True)
    tune_path = args.out_dir / "tuned.json"
    if tune_path.exists():
        rounds, lam = json.loads(tune_path.read_text())["rounds"], json.loads(tune_path.read_text())["lam"]
    else:
        rounds, lam = tune(d, args.n_jobs)
        tune_path.write_text(json.dumps({"rounds": rounds, "lam": lam}))
    print(f"tuned on training holdout: rounds={rounds}, ridge lambda={lam}", flush=True)
    out = Writer(args.out_dir / "results.csv")
    sizes = [int(x) for x in args.sizes.split(",")]
    stages = args.stage.split(",")
    if "rows" in stages:
        run_rows(d, LgbModel(d, rounds), out, args.reps, sizes, args.out_dir)
        run_rows(d, RidgeModel(d, lam), out, args.reps, sizes, args.out_dir)
    if "cols" in stages:
        run_cols(d, rounds, out, args.reps)
    print("ALL DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
