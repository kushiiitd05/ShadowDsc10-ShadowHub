"""Report figures, drawn only from the CSVs in results/ (no model is fitted here).

    python code/make_figures.py      # writes figures/*.pdf and figures/*.png

Style: ultramarine = our method / verified claim, amber = limitation or a loss, grey = baseline (uniform, cs02).
Every series also has its own marker and a direct label, so identity never rests on colour alone.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results", ROOT / "figures"
INK, GREY, SOFT, RULE = "#0E1116", "#4A515E", "#878E9A", "#D8DCE2"
ULTRA, AMBER = "#1B36C4", "#C77800"
METHOD_STYLE = {
    "uniform": dict(color=GREY, marker="o", ls="-", label="uniform"),
    "hub_stratified": dict(color=SOFT, marker="s", ls="--", label="hub-stratified"),
    "sensitivity": dict(color=ULTRA, marker="D", ls="-", label="sensitivity coreset"),
    "importance": dict(color=ULTRA, marker="D", ls="-", label="importance columns"),
}

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Charter", "Palatino", "Georgia", "DejaVu Serif"], "font.size": 8,
    "axes.edgecolor": GREY, "axes.labelcolor": INK, "axes.linewidth": 0.6, "axes.grid": True, "grid.color": RULE,
    "grid.linewidth": 0.5, "xtick.color": GREY, "ytick.color": GREY, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "legend.fontsize": 7,
    "lines.linewidth": 1.6, "lines.markersize": 4, "savefig.dpi": 220, "figure.dpi": 110,
})


def save(fig, name: str) -> None:
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


def med_band(df: pd.DataFrame, col: str) -> pd.DataFrame:
    return df.groupby("m")[col].agg(["median", "min", "max"]).reset_index()


def curves(ax, data: pd.DataFrame, model: str, col: str, methods: tuple[str, ...], logy: bool = False) -> None:
    for meth in methods:
        d = data[(data["model"] == model) & (data["method"] == meth)]
        if d.empty:
            continue
        b = med_band(d, col)
        st = METHOD_STYLE[meth]
        ax.plot(b["m"], b["median"], color=st["color"], marker=st["marker"], ls=st["ls"], label=st["label"])
        ax.fill_between(b["m"], b["min"], b["max"], color=st["color"], alpha=0.12, lw=0)
    ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("summary size m (rows, log scale)")


def fig_rows(et: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5))
    for ax, model, title in zip(axes, ("lgbm", "ridge"), ("LightGBM", "Ridge")):
        curves(ax, et, model, "test_rmsle", ("uniform", "hub_stratified", "sensitivity"))
        full = et[(et["model"] == model) & (et["method"] == "full")]["test_rmsle"].iloc[0]
        ax.axhline(full, color=INK, lw=0.8, ls=":")
        ax.text(ax.get_xlim()[0] * 1.1 if ax.get_xlim()[0] > 0 else 2000, full, f" full data {full:.4f}", va="bottom", fontsize=6.5, color=INK)
        ax.set_title(f"Plot 1 - {title}: test RMSLE vs summary size", fontsize=8, loc="left")
        ax.set_ylabel("test RMSLE (median, min-max band)")
    axes[0].legend(loc="upper right")
    save(fig, "fig_plot1_rows")


def fig_columns(et: pd.DataFrame) -> None:
    d = et[et["model"] == "lgbm_cols"].copy()
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    for meth in ("uniform", "importance"):
        s = d[d["method"] == meth].groupby("m")["test_rmsle"].agg(["median", "min", "max"]).reset_index()
        st = METHOD_STYLE[meth]
        lab = "random columns" if meth == "uniform" else st["label"]
        ax.plot(s["m"], s["median"], color=st["color"], marker=st["marker"], ls=st["ls"], label=lab)
        ax.fill_between(s["m"], s["min"], s["max"], color=st["color"], alpha=0.12, lw=0)
    ax.set_xlabel("columns kept k (rows fixed at 200,000)")
    ax.set_ylabel("test RMSLE")
    ax.set_title("Columns: importance-chosen vs random", fontsize=8, loc="left")
    ax.legend()
    save(fig, "fig_plot1_columns")


def fig_time(et: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4))
    for ax, model, title in zip(axes, ("lgbm", "ridge"), ("LightGBM", "Ridge")):
        d = et[et["model"] == model].assign(total=lambda x: x["t_build"].fillna(0) + x["t_train"])
        curves(ax, d, model, "total", ("uniform", "sensitivity"), logy=True)
        tfull = et[(et["model"] == model) & (et["method"] == "full")]["t_train"].iloc[0]
        ax.axhline(tfull, color=INK, lw=0.8, ls=":")
        ax.text(2000, tfull, f" T_full {tfull:.1f} s", va="bottom", fontsize=6.5)
        ax.set_ylabel("T_build + T_train (s, log)")
        ax.set_title(f"Plot 2 - {title}: time (shared server, development)", fontsize=8, loc="left")
    axes[0].legend(loc="lower right")
    save(fig, "fig_plot2_time")


def fig_coreset(et: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.3))
    specs = (("train_gap", "Plot 3: F(P,y*)/opt - 1 (train)"), ("test_gap", "Plot 4: |F(P_test,y*)/F(P,x*) - 1|"),
             ("test_pred_l2", "Plot 5: ||y* - x*|| / ||x*|| (test preds)"))
    for ax, (col, title) in zip(axes, specs):
        curves(ax, et, "lgbm", col, ("uniform", "sensitivity"), logy=True)
        ax.set_title(title, fontsize=7.2, loc="left")
    axes[0].legend(loc="upper right")
    save(fig, "fig_plot345_lgbm")


def fig_coreset_ridge(et: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.3))
    specs = (("train_gap", "Plot 3 ridge: train gap"), ("test_gap", "Plot 4 ridge: test gap"),
             ("param_l2", "Plot 5 ridge: ||y* - x*|| / ||x*|| (coefficients)"))
    for ax, (col, title) in zip(axes, specs):
        curves(ax, et, "ridge", col, ("uniform", "sensitivity"), logy=True)
        ax.set_title(title, fontsize=7.2, loc="left")
    axes[0].legend(loc="upper right")
    save(fig, "fig_plot345_ridge")


def fig_tabpfn_grid() -> None:
    g = pd.read_csv(RES / "tabpfn_context_weight_grid.csv")
    inputs = (((1, 0, 0), "365 days", GREY, "o", "-"), ((0, 0, 1), "730 days", SOFT, "s", "--"),
              ((0, 1, 0), "540 days", SOFT, "^", ":"), ((0.5, 0, 0.5), "mix 365/730 (frozen)", ULTRA, "D", "-"))
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.2), sharex=True)
    for (s365, s540, s730), lab, col, mk, ls in inputs:
        d = g[np.isclose(g["s365"], s365) & np.isclose(g["s540"], s540) & np.isclose(g["s730"], s730)].sort_values("w")
        for ax, c in zip(axes, ("g234", "g1", "g5")):
            ax.plot(d["w"], d[c] * 1000, color=col, marker=mk, ls=ls, label=lab)
    for ax, t in zip(axes, ("folds 2-4 mean gain", "fold 1 (winter) gain", "fold 5 (summer 2014) gain")):
        ax.set_title(t, fontsize=7.2, loc="left")
        ax.axvline(0.2, color=INK, lw=0.7, ls=":")
        ax.axhline(0, color=GREY, lw=0.6)
        ax.set_xlabel("TabPFN-TS weight w")
    axes[0].axhline(2.4, color=INK, lw=0.7, ls="--")
    axes[0].set_ylabel("gain over cs02 (x 1e-3)")
    axes[1].legend(loc="lower left", fontsize=6.2)
    save(fig, "fig_tabpfn_grid")


def fig_gate() -> None:
    g = pd.read_csv(RES / "audit_gate_gains.csv")
    fig, ax = plt.subplots(figsize=(7.0, 2.3))
    folds = [1, 2, 3, 4, 5]
    cands = (("hp17", "hp17 alone", SOFT, "s"), ("fallback_hp17_t365", "hp17 + TabPFN-TS 365", GREY, "o"),
             ("candidate", "hp17 + TabPFN-TS mix (frozen)", ULTRA, "D"))
    width = 0.26
    for i, (c, lab, col, _) in enumerate(cands):
        r = g[(g["arm"] == "3seed") & (g["candidate"] == c)].iloc[0]
        vals = [r[f"gain_f{f}"] * 1000 for f in folds]
        x = np.arange(len(folds)) + (i - 1) * width
        bars = ax.bar(x, vals, width * 0.92, color=col, label=lab, edgecolor="white", linewidth=0.6)
        for b, v in zip(bars, vals):
            if v < 0:
                b.set_edgecolor(AMBER)
                b.set_hatch("////")
                b.set_linewidth(1.0)
    ax.axhline(2.4, color=INK, lw=0.8, ls="--")
    ax.text(-0.45, 2.55, "gate 0.0024", fontsize=6.5, ha="left")
    ax.axhline(0, color=GREY, lw=0.6)
    ax.set_xticks(range(len(folds)), ["fold 1\nwinter", "fold 2", "fold 3", "fold 4", "fold 5\n935 hubs"])
    ax.set_ylabel("gain over cs02 (x 1e-3 RMSLE)")
    ax.set_title("Gain over the like-for-like baseline, 3-seed average (hatched amber edge = worse than baseline)", fontsize=8, loc="left")
    ax.set_ylim(top=8)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=c, label=l) for _, l, c, _ in cands], loc="upper left", ncol=3)
    save(fig, "fig_gate_gains")


def fig_optuna() -> None:
    p = ROOT / "results" / "g6c_pooled_80_trials.csv"
    if not p.exists():
        return
    d = pd.read_csv(p)
    obj = next(c for c in d.columns if c.lower() in ("value", "objective", "blend_rmsle", "rmsle_blend"))
    leaves = next((c for c in d.columns if "leaves" in c.lower()), None)
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    d = d.sort_values(obj).reset_index(drop=True)
    ax.scatter(range(len(d)), d[obj], s=10, color=GREY, label="trial")
    best = d.iloc[0]
    ax.scatter([0], [best[obj]], s=26, color=ULTRA, marker="D", label=f"best {best[obj]:.4f}", zorder=3)
    ax.axhspan(best[obj], best[obj] + 0.0012, color=ULTRA, alpha=0.08, lw=0)
    ax.text(len(d) * 0.35, best[obj] + 0.0002, "noise band 0.0012", fontsize=6.5, color=ULTRA)
    ax.set_xlabel("trials, sorted by objective")
    ax.set_ylabel("folds 2-3 blend RMSLE")
    ax.set_title("Pooled Optuna, 80 trials: a plateau", fontsize=8, loc="left")
    ax.legend(loc="lower right")
    save(fig, "fig_optuna")


SHORT_GROUP = {
    "Hub x weekday x promotion profile": "hub x weekday x promo profile", "Order rolling windows (7-730 days)": "order rolling windows",
    "Promotion calendar (known ahead)": "promotion calendar", "Hub identity (cross-fitted encoding, id)": "hub identity (encoding)",
    "Calendar and horizon": "calendar, horizon", "Same-day lags and value at origin": "same-day lags",
    "Sessions, conversion, volume per session": "sessions, conversion", "Open/closed calendar (known ahead)": "open/closed calendar",
    "Holidays and school closures (known ahead)": "holidays, school closures", "History coverage and data-gap flags": "data-gap flags",
    "Hub metadata, competition, loyalty": "hub metadata",
}


def fig_feature_groups() -> None:
    p = RES / "feature_group_importance.csv"
    if not p.exists():
        return
    d = pd.read_csv(p).sort_values("gain_share")
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    y = np.arange(len(d))
    colors = [ULTRA if v >= 0.2 else GREY for v in d["gain_share"]]
    ax.barh(y, d["gain_share"] * 100, color=colors, height=0.62)
    ax.errorbar(d["gain_share"] * 100, y, xerr=[(d["gain_share"] - d["seed_min"]) * 100, (d["seed_max"] - d["gain_share"]) * 100],
                fmt="none", ecolor=INK, elinewidth=0.6, capsize=1.5)
    for yi, (share, n) in enumerate(zip(d["gain_share"], d["columns"])):
        ax.text(d["seed_max"].iloc[yi] * 100 + 1.0, yi, f"{share * 100:.1f}% ({n})", va="center", fontsize=6.2, color=INK)
    ax.set_yticks(y, [SHORT_GROUP.get(g, g) for g in d["group"]], fontsize=6.6)
    ax.set_xlim(0, 66)
    ax.set_xlabel("% of split gain, 3 frozen boosters (min-max); (n) = columns")
    ax.set_title("Which of the 342 columns the trees use", fontsize=8, loc="left")
    ax.grid(axis="y", visible=False)
    save(fig, "fig_feature_groups")


def fig_model_ladder() -> None:
    f = pd.read_csv(RES / "audit_fold_scores.csv")
    f = f[f["arm"] == "3seed"].sort_values("fold")
    models = (("cs02_all", "cs02 baseline (x* recipe family)", GREY, "o", "-"), ("hp17_all", "hp17 alone (tuned)", SOFT, "s", "--"),
              ("candidate_all", "hp17 + TabPFN-TS, w 0.2 (submitted)", ULTRA, "D", "-"))
    fig, axes = plt.subplots(1, 2, figsize=(5.2, 2.3), gridspec_kw={"width_ratios": [4, 1.3]})
    for col, lab, c, mk, ls in models:
        a = f[f["fold"] <= 4]
        axes[0].plot(a["fold"], a[col], color=c, marker=mk, ls=ls, label=lab)
        b = f[f["fold"] == 5]
        axes[1].plot([5], b[col], color=c, marker=mk, ls="none", markersize=5)
    axes[0].set_xticks([1, 2, 3, 4], ["fold 1\n2 Jan", "fold 2\n13 Feb", "fold 3\n27 Mar", "fold 4\n8 May\n= end-term test"])
    axes[1].set_xticks([5], ["fold 5\n19 Jun 2014\n(935 hubs)"])
    axes[0].set_ylabel("development-fold RMSLE (3-seed avg)")
    axes[0].set_title("Development folds, 42-day forecasts from each origin (no real test labels exist)", fontsize=7.2, loc="left")
    axes[0].legend(loc="upper right", fontsize=6.3)
    save(fig, "fig_model_ladder")


def fig_forecast_examples() -> None:
    raw = ROOT.parent / "1_midterm" / "data" / "raw"
    sub = ROOT / "final" / "submission_final.csv"
    if not (raw / "orders_train.csv").exists() or not sub.exists():
        return
    hist = pd.read_csv(raw / "orders_train.csv", usecols=["HubID", "Date", "OrderVolume"], parse_dates=["Date"])
    frame = pd.read_csv(raw / "orders_test.csv", usecols=["Id", "HubID", "Date"], parse_dates=["Date"]).merge(pd.read_csv(sub), on="Id")
    fig, axes = plt.subplots(3, 1, figsize=(7.0, 3.6), sharex=True)
    start = hist["Date"].max() - pd.Timedelta(days=90)
    for ax, hub in zip(axes, (1, 262, 817)):
        h = hist[(hist["HubID"] == hub) & (hist["Date"] > start)]
        f = frame[frame["HubID"] == hub]
        ax.plot(h["Date"], h["OrderVolume"] / 1000, color=GREY, lw=0.9, label="history (actual)")
        ax.plot(f["Date"], f["OrderVolume"] / 1000, color=ULTRA, lw=1.3, label="submitted forecast (no labels)")
        ax.set_ylabel(f"hub {hub}\n(k orders)", fontsize=6.5)
    axes[0].legend(loc="upper left", ncol=2, fontsize=6.3)
    axes[0].set_title("Last 90 days of history and the 42-day forecast (closed days are exactly 0)", fontsize=7.5, loc="left")
    save(fig, "fig_forecast_examples")


def _box(ax, x: float, y: float, w: float, h: float, text: str, edge: str, face: str = "white", size: float = 6.2) -> None:
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.012", linewidth=0.9, edgecolor=edge, facecolor=face))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, color=INK, linespacing=1.25)


def _arrow(ax, x0: float, y0: float, x1: float, y1: float) -> None:
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color=GREY, lw=0.8, shrinkA=0, shrinkB=0))


def fig_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    size = 5.4
    _box(ax, 0.006, 0.40, 0.114, 0.22, "Raw data\n1,115 hubs, daily\n1 Jan 2013 -\n19 Jun 2015", GREY, size=size)
    _box(ax, 0.145, 0.36, 0.14, 0.30, "37 forecast origins,\n14 days apart\n342 features from\nhistory up to origin\ntarget log1p(orders)\non days o+1 ... o+42", GREY, size=size)
    _arrow(ax, 0.12, 0.51, 0.145, 0.51)
    ax.text(0.33, 0.955, "TRACK A  -  end-term rubric (coreset study)", fontsize=6.8, color=ULTRA, weight="bold")
    ax.text(0.33, 0.425, "TRACK B  -  the forecaster we submit", fontsize=6.8, color=INK, weight="bold")
    top = (("Fixed split\ntrain: labels to 8 May 2015\ntest: 9 May-19 Jun 2015\n46,830 rows", GREY),
           ("x* = LightGBM cs02\n1.13M rows x 342 cols\nsecond model: ridge\nsame settings on all", ULTRA),
           ("Coreset\n5 rough models ->\nsensitivities s_i ->\nm rows, w = 1/(m p_i)", ULTRA),
           ("Plots 1-5, time\nm = 50k: 0.1070\n(uniform 0.1102)\n2.75x faster", ULTRA))
    bottom = (("Dev folds 1-5\ngate fixed in advance\n>= 0.0024, all seeds", GREY),
              ("cs02 -> hp17 (tuned)\n+ TabPFN-TS 365/730\nweight 0.2 (log space)", INK),
              ("Final retrain\nall 37 origins, 3 seeds\nfrozen: SHA-256,\n13 checks", INK),
              ("Forecast CSV\n20 Jun - 31 Jul 2015\n46,830 rows\nno labels, no score", INK))
    xs, w, gap = (0.33, 0.50, 0.67, 0.84), 0.145, 0.025
    for lane, y in ((top, 0.57), (bottom, 0.05)):
        for i, ((text, edge), x) in enumerate(zip(lane, xs)):
            _box(ax, x, y, w, 0.33, text, edge, size=size)
            if i:
                _arrow(ax, x - gap, y + 0.165, x, y + 0.165)
    _arrow(ax, 0.285, 0.60, 0.33, 0.72)
    _arrow(ax, 0.285, 0.42, 0.33, 0.22)
    ax.text(0.0, 0.25, "Leakage guards:\n- features stop at the origin\n- training-origin purge\n- cross-fitted hub encoding\n"
            "- fixed rounds, no early\n  stopping on scored folds\n- test rows: scoring only", fontsize=5.5, color=GREY, va="top", linespacing=1.3)
    save(fig, "fig_pipeline")


def main() -> None:
    et = pd.read_csv(RES / "endterm_run1_results.csv")
    fig_feature_groups()
    fig_model_ladder()
    fig_pipeline()
    fig_forecast_examples()
    fig_rows(et)
    fig_columns(et)
    fig_time(et)
    fig_coreset(et)
    fig_coreset_ridge(et)
    fig_tabpfn_grid()
    fig_gate()
    fig_optuna()
    print(sorted(p.name for p in FIG.glob("*.pdf")))


if __name__ == "__main__":
    main()
