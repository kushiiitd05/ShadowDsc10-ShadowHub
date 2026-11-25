"""Aggregate G11 runs: python analyze_orig.py <out_dir> [--seeds 20260807,20260808,20260809] [--folds 1,2,3,4,5]

Reads oof_<set>_f<fold>_s<seed>.csv.gz (+ res_*.json markers) and writes scores_long.csv, gate_table.csv, gate_verdict.csv into <out_dir>.
Variants vs the control (set base = cs02): summer, dense7, season blends at 0.3 / 0.5 (log space, weight on the season member).
Gain = control RMSLE - variant RMSLE (positive = better). Gate: seed-mean gain >= 0.0024 on >= 2 folds, gain > 0 on every seed of
those folds, and the same on the metric without open-but-zero rows.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from origin_builder import log_blend  # noqa: E402

KEYS = ["HubID", "Date"]
THRESH = 0.0024
NOISE = 0.0012


def rmsle(actual: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.log1p(np.clip(pred, 0, None)) - np.log1p(actual)) ** 2)))


def scores(frame: pd.DataFrame, col: str) -> tuple[float, float]:
    actual, pred = frame["ActualOrderVolume"].to_numpy(float), frame[col].to_numpy(float)
    keep = ~((frame["IsOpen"].to_numpy() == 1) & (actual == 0))
    return rmsle(actual, pred), rmsle(actual[keep], pred[keep])


def load_oof(out: Path, set_name: str, fold: int, seed: int) -> pd.DataFrame | None:
    path = out / f"oof_{set_name}_f{fold}_s{seed}.csv.gz"
    if path.exists():
        return pd.read_csv(path)
    marker = out / f"res_{set_name}_f{fold}_s{seed}.json"
    if marker.exists() and json.loads(marker.read_text()).get("identical_to_base"):
        return load_oof(out, "base", fold, seed)
    return None


def long_scores(out: Path, folds: list[int], seeds: list[int]) -> pd.DataFrame:
    rows = []
    for fold in folds:
        for seed in seeds:
            ctrl = load_oof(out, "base", fold, seed)
            if ctrl is None:
                continue
            frame = ctrl.rename(columns={"pred": "cs02"})
            for name in ("summer", "dense7", "season"):
                other = load_oof(out, name, fold, seed)
                if other is not None:
                    frame = frame.merge(other[[*KEYS, "pred"]].rename(columns={"pred": name}), on=KEYS, how="left", validate="one_to_one")
            for w in (0.3, 0.5):
                if "season" in frame:
                    frame[f"blend{int(w * 10):02d}"] = log_blend(frame["cs02"].to_numpy(), frame["season"].to_numpy(), w)
            for name in [c for c in ("cs02", "summer", "dense7", "season", "blend03", "blend05") if c in frame]:
                if frame[name].isna().any():
                    continue
                a, e = scores(frame, name)
                rows.append({"variant": name, "fold": fold, "seed": seed, "all": a, "excl0": e, "rows": len(frame)})
    return pd.DataFrame(rows)


def gate_table(long: pd.DataFrame) -> pd.DataFrame:
    ctrl = long[long["variant"] == "cs02"].set_index(["fold", "seed"])
    out = []
    for name, grp in long[long["variant"] != "cs02"].groupby("variant"):
        g = grp.set_index(["fold", "seed"])
        common = g.index.intersection(ctrl.index)
        for fold in sorted({f for f, _ in common}):
            seeds = [s for f, s in common if f == fold]
            ga = np.array([ctrl.loc[(fold, s), "all"] - g.loc[(fold, s), "all"] for s in seeds])
            ge = np.array([ctrl.loc[(fold, s), "excl0"] - g.loc[(fold, s), "excl0"] for s in seeds])
            out.append({"variant": name, "fold": fold, "n_seeds": len(seeds),
                        "cs02_all": float(np.mean([ctrl.loc[(fold, s), "all"] for s in seeds])),
                        "var_all": float(np.mean([g.loc[(fold, s), "all"] for s in seeds])),
                        "gain_all": float(ga.mean()), "gain_excl0": float(ge.mean()),
                        "gain_min_seed": float(ga.min()), "gain_max_seed": float(ga.max()),
                        "gain_by_seed": ";".join(f"{x:+.4f}" for x in ga),
                        "pass_fold": bool(ga.mean() >= THRESH and (ga > 0).all() and ge.mean() >= THRESH and (ge > 0).all())})
    return pd.DataFrame(out)


def verdicts(gt: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, grp in gt.groupby("variant"):
        for label, folds in (("folds1-4", [1, 2, 3, 4]), ("folds1-5", [1, 2, 3, 4, 5]), ("folds2-4", [2, 3, 4])):
            sub = grp[grp["fold"].isin(folds)]
            rows.append({"variant": name, "scope": label, "folds_scored": len(sub), "folds_passing": int(sub["pass_fold"].sum()),
                         "mean_gain_all": float(sub["gain_all"].mean()) if len(sub) else np.nan,
                         "gate_pass": bool(sub["pass_fold"].sum() >= 2)})
    return pd.DataFrame(rows)


def seedpair_control(out: Path, folds: list[int], seeds: list[int]) -> pd.DataFrame:
    """Gain of log-averaging cs02 (seed s) with cs02 of the next seed vs cs02 (seed s) alone: the variance-reduction part of any 2-model blend."""
    rows = []
    for fold in folds:
        for i, seed in enumerate(seeds):
            a, b = load_oof(out, "base", fold, seed), load_oof(out, "base", fold, seeds[(i + 1) % len(seeds)])
            if a is None or b is None:
                continue
            m = a.merge(b[[*KEYS, "pred"]], on=KEYS, suffixes=("", "_o"), validate="one_to_one")
            single = scores(m.rename(columns={"pred": "x"}), "x")
            for w in (0.3, 0.5):
                m["bl"] = log_blend(m["pred"].to_numpy(), m["pred_o"].to_numpy(), w)
                bl = scores(m, "bl")
                rows.append({"fold": fold, "seed": seed, "weight": w, "gain_all": single[0] - bl[0], "gain_excl0": single[1] - bl[1]})
    return pd.DataFrame(rows)


def resources(out: Path) -> pd.DataFrame:
    rows = []
    for f in sorted(out.glob("res_*_f*_s*.json")):
        d = json.loads(f.read_text())
        if d.get("identical_to_base"):
            continue
        rows.append({"set": d["set"], "fold": d["fold"], "seed": d["seed"], "train_rows": d["train_rows"], "origins": d["train_origin_count"],
                     "hold_origin": d["holdout_origin"], "best_iter": d["best_iteration"], "fit_s": d["fit_seconds"], "load_s": d["load_seconds"],
                     "peak_rss_gb": d["peak_rss_gb"], "origins_per_target_day": d["duplication"]["mean_origins_per_target_day"],
                     "holdout_fallback": d.get("holdout_fallback", False)})
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--seeds", default="20260807,20260808,20260809")
    ap.add_argument("--folds", default="1,2,3,4,5")
    a = ap.parse_args()
    long = long_scores(a.out_dir, [int(x) for x in a.folds.split(",")], [int(x) for x in a.seeds.split(",")])
    if long.empty:
        print("no results yet")
        return 0
    long.to_csv(a.out_dir / "scores_long.csv", index=False)
    gt = gate_table(long)
    gt.to_csv(a.out_dir / "gate_table.csv", index=False)
    vd = verdicts(gt)
    vd.to_csv(a.out_dir / "gate_verdict.csv", index=False)
    pd.set_option("display.width", 200)
    print(long.pivot_table(index=["variant", "fold"], columns="seed", values="all").round(4).to_string())
    print(gt.drop(columns=["gain_by_seed"]).round(4).to_string(index=False))
    print(gt[["variant", "fold", "gain_by_seed"]].to_string(index=False))
    print(vd.round(4).to_string(index=False))
    folds, seeds = [int(x) for x in a.folds.split(",")], [int(x) for x in a.seeds.split(",")]
    sp = seedpair_control(a.out_dir, folds, seeds)
    sp.to_csv(a.out_dir / "seedpair_control.csv", index=False)
    print("seed-pair control (cs02 averaged with another cs02 seed): mean gain over seeds")
    print(sp.groupby(["weight", "fold"])[["gain_all", "gain_excl0"]].mean().round(4).to_string())
    res = resources(a.out_dir)
    res.to_csv(a.out_dir / "resources.csv", index=False)
    print(res.groupby("set")[["train_rows", "origins", "best_iter", "fit_s", "load_s", "peak_rss_gb", "origins_per_target_day"]].mean().round(1).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
