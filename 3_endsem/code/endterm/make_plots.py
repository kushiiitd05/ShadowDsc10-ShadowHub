"""Plots 1-5, the sensitivity histogram and the speed-up table from endterm_exp.py results.

Usage: python make_plots.py <results_dir> <out_dir>   (results_dir holds results.csv and sens_*.npy)
Medians over repetitions with min-max bars; uniform sampling is drawn in every plot.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

COLORS = {"uniform": "#7f7f7f", "hub_stratified": "#1f77b4", "sensitivity": "#d62728", "importance": "#d62728"}
LABELS = {"uniform": "uniform", "hub_stratified": "hub-stratified", "sensitivity": "sensitivity coreset", "importance": "importance columns"}


def band(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    return frame.groupby("m")[column].agg(["median", "min", "max"]).reset_index()


def line(ax, frame: pd.DataFrame, method: str, column: str) -> None:
    b = band(frame[frame.method == method], column)
    ax.errorbar(b["m"], b["median"], yerr=[b["median"] - b["min"], b["max"] - b["median"]], marker="o", ms=4, capsize=2,
                color=COLORS[method], label=LABELS[method])


def full_value(data: pd.DataFrame, model: str, column: str) -> float:
    return float(data[(data.model == model) & (data.method == "full")][column].iloc[0])


def plot1(data: pd.DataFrame, model: str, out: Path) -> None:
    rows = data[(data.model == model) & (data.method != "full")]
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for method in ("uniform", "hub_stratified", "sensitivity"):
        line(ax, rows, method, "test_rmsle")
    ax.axhline(full_value(data, model, "test_rmsle"), color="k", ls="--", label="full model")
    ax.set_xscale("log"); ax.set_xlabel("rows kept"); ax.set_ylabel("test RMSLE"); ax.set_title(f"Plot 1, rows ({model})"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out / f"plot1_rows_{model}.png", dpi=200); plt.close(fig)


def plot1_columns(data: pd.DataFrame, out: Path) -> None:
    rows = data[data.model == "lgbm_cols"]
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for method in ("uniform", "importance"):
        line(ax, rows, method, "test_rmsle")
    ax.axhline(full_value(data, "lgbm", "test_rmsle"), color="k", ls="--", label="full model (all rows, all columns)")
    ax.set_xscale("log"); ax.set_xlabel(f"columns kept (rows fixed at {int(rows['rows_fixed'].iloc[0]):,})"); ax.set_ylabel("test RMSLE")
    ax.set_title("Plot 1, columns (LightGBM)"); ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(out / "plot1_columns_lgbm.png", dpi=200); plt.close(fig)


def plot2(data: pd.DataFrame, model: str, out: Path) -> None:
    rows = data[(data.model == model) & (data.method.isin(["uniform", "sensitivity"]))].copy()
    rows["total"] = rows["t_build"] + rows["t_train"]
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    t_full = full_value(data, model, "t_train")
    ax.axhline(t_full, color="k", ls="--", label="T_full")
    for method in ("uniform", "sensitivity"):
        sub = rows[rows.method == method]
        b = band(sub, "total"); ax.plot(b["m"], b["median"], marker="o", ms=4, color=COLORS[method], label=f"{LABELS[method]}: T_build + T_train")
    b = band(rows[rows.method == "sensitivity"], "t_train"); ax.plot(b["m"], b["median"], marker="s", ms=3, ls=":", color=COLORS["sensitivity"], label="coreset: T_train only")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("rows kept"); ax.set_ylabel("seconds"); ax.set_title(f"Plot 2, time ({model})"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out / f"plot2_time_{model}.png", dpi=200); plt.close(fig)


def gap_plot(data: pd.DataFrame, model: str, column: str, title: str, name: str, out: Path) -> None:
    rows = data[(data.model == model) & (data.method.isin(["uniform", "sensitivity"]))]
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for method in ("uniform", "sensitivity"):
        line(ax, rows, method, column)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("sample size m"); ax.set_title(f"{title} ({model})"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out / f"{name}_{model}.png", dpi=200); plt.close(fig)


def slope(data: pd.DataFrame, model: str, method: str, column: str) -> float:
    b = band(data[(data.model == model) & (data.method == method)], column)
    return float(np.polyfit(np.log10(b["m"]), np.log10(b["median"]), 1)[0])


def histogram(results: Path, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2))
    for ax, model in zip(axes, ("lgbm", "ridge")):
        s = np.load(results / f"sens_{model}_rep0.npy").astype(float)
        ax.hist(np.log10(s), bins=60, color="#d62728"); ax.set_xlabel("log10(s_i / S)"); ax.set_title(f"sensitivity scores ({model})"); ax.set_yscale("log")
    fig.tight_layout(); fig.savefig(out / "hist_sensitivity.png", dpi=200); plt.close(fig)


def speedup_table(data: pd.DataFrame) -> pd.DataFrame:
    out = []
    for model in ("lgbm", "ridge"):
        t_full = full_value(data, model, "t_train"); r_full = full_value(data, model, "test_rmsle")
        for method in ("uniform", "hub_stratified", "sensitivity"):
            for m, sub in data[(data.model == model) & (data.method == method)].groupby("m"):
                total = float((sub["t_build"] + sub["t_train"]).median())
                out.append({"model": model, "method": method, "m": m, "test_rmsle": float(sub["test_rmsle"].median()), "delta_vs_full": float(sub["test_rmsle"].median()) - r_full,
                            "t_build": float(sub["t_build"].median()), "t_train": float(sub["t_train"].median()), "speedup": t_full / total})
    return pd.DataFrame(out)


def main() -> None:
    results, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(results / "results.csv")
    for model in ("lgbm", "ridge"):
        plot1(data, model, out); plot2(data, model, out)
        gap_plot(data, model, "train_gap", "Plot 3: gap on training set", "plot3_train_gap", out)
        gap_plot(data, model, "test_gap_vs_test_star" if "test_gap_vs_test_star" in data and model == "lgbm" else "test_gap", "Plot 4: gap on test set", "plot4_test_gap", out)
        gap_plot(data, model, "param_l2" if model == "ridge" else "test_pred_l2", "Plot 5: model difference", "plot5_model_diff", out)
    plot1_columns(data, out); histogram(results, out)
    table = speedup_table(data); table.to_csv(out / "speedup_table.csv", index=False)
    lines = []
    for model, col5 in (("lgbm", "test_pred_l2"), ("ridge", "param_l2")):
        s3 = slope(data, model, "sensitivity", "train_gap"); s5 = slope(data, model, "sensitivity", col5)
        u3 = slope(data, model, "uniform", "train_gap"); u5 = slope(data, model, "uniform", col5)
        lines.append(f"{model}: log-log slope of Plot 3 coreset {s3:.2f}, uniform {u3:.2f}; Plot 5 coreset {s5:.2f}, uniform {u5:.2f}; ratio Plot5/Plot3 (coreset) {s5 / s3:.2f}")
    sens = data[data.method == "sensitivity"].groupby("model")[["S", "alpha_max"]].median()
    lines.append(sens.to_string())
    (out / "slopes_and_S.txt").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
