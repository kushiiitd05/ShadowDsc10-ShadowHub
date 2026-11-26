"""Independent freeze audit: recompute the adoption gate and the final blend from saved artifacts only (no refitting).

Run from the 3_endsem folder:
    python code/freeze_audit.py            # writes results/audit_*.csv and results/audit_summary.json

Inputs (all under artifacts/, pulled from the run folders without modification):
    oof/cs02_fix1000_f1-4_s<seed>.csv.gz, oof/cs02_fix1000_f5_s<seed>.csv.gz   baseline (lr 0.025, 1000 fixed rounds)
    oof/hp17_fix1000_f1-4_s<seed>.csv.gz, oof/hp17_fix1000_f5_s<seed>.csv.gz   candidate LightGBM member
    oof/tabpfn365_*.csv.gz, oof/tabpfn730_*.csv.gz                             TabPFN-TS members (deterministic, one realisation)
    test_window/*.csv                                                          TabPFN-TS predictions for the real test window
    rehearsal_mix/*                                                            rehearsal-2 retrain outputs (real path)
    ../1_midterm/data/raw/orders_test.csv, sample_submission.csv                 test frame (no labels) for Id order and IsOpen

Gate (fixed in plan v4 section 2 and FINAL_PLAN section G, not changed here):
    mean gain on folds 2-4 >= 0.0024 (2 x measured noise 0.0012), >= 2 of 3 folds improve, same sign on seeds 07/08/09,
    still true without open-but-zero rows, fold 5 checked (935 complete-block hubs).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
RES = ROOT / "results"
RAW = ROOT.parent / "1_midterm" / "data" / "raw"
SEEDS = (20260807, 20260808, 20260809)
W_TABPFN = 0.2
GATE_MIN_GAIN = 0.0024
NOISE = 0.0012
KEYS = ["fold", "HubID", "Date"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def rmsle(actual: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.log1p(pred) - np.log1p(actual)) ** 2)))


def scores(df: pd.DataFrame, col: str) -> tuple[float, float]:
    """(RMSLE all rows, RMSLE excluding open-but-zero rows)."""
    keep = ~((df["IsOpen"] == 1) & (df["ActualOrderVolume"] == 0))
    a, p = df["ActualOrderVolume"].to_numpy(float), df[col].to_numpy(float)
    return rmsle(a, p), rmsle(a[keep.to_numpy()], p[keep.to_numpy()])


def read_oof(name: str) -> pd.DataFrame:
    df = pd.read_csv(ART / "oof" / name, parse_dates=["Date"])
    if df.duplicated(KEYS).any():
        raise AssertionError(f"{name}: duplicate keys")
    return df


def lgb_member(recipe: str, seed: int) -> pd.DataFrame:
    a = read_oof(f"{recipe}_fix1000_f1-4_s{seed}.csv.gz")
    b = read_oof(f"{recipe}_fix1000_f5_s{seed}.csv.gz")
    return pd.concat([a, b], ignore_index=True)


def tabpfn(context: int) -> pd.DataFrame:
    if context == 365:
        parts = [read_oof("tabpfn365_f1-4.csv.gz"), read_oof("tabpfn365_f5.csv.gz")]
    else:
        parts = [read_oof("tabpfn730_f1.csv.gz"), read_oof("tabpfn730_f2-4.csv.gz"), read_oof("tabpfn730_f5.csv.gz")]
    df = pd.concat(parts, ignore_index=True)
    if df.duplicated(KEYS).any():
        raise AssertionError(f"TabPFN {context}: duplicate keys across files")
    return df[KEYS + ["pred"]].rename(columns={"pred": f"t{context}"})


def build_panel() -> pd.DataFrame:
    """One row per (fold, hub, date) present in every member, with all member predictions as columns."""
    base = lgb_member("cs02", SEEDS[0])[KEYS + ["Horizon", "IsOpen", "ActualOrderVolume"]]
    panel = base.copy()
    for recipe in ("cs02", "hp17"):
        for s in SEEDS:
            m = lgb_member(recipe, s)
            chk = panel[KEYS + ["ActualOrderVolume"]].merge(m[KEYS + ["ActualOrderVolume"]], on=KEYS, validate="one_to_one")
            if not np.array_equal(chk["ActualOrderVolume_x"].to_numpy(), chk["ActualOrderVolume_y"].to_numpy()):
                raise AssertionError(f"{recipe} {s}: actual volumes differ from the baseline file")
            panel = panel.merge(m[KEYS + ["pred"]].rename(columns={"pred": f"{recipe}_{s}"}), on=KEYS, how="inner", validate="one_to_one")
    full_rows = panel.groupby("fold").size().to_dict()
    for c in (365, 730):
        panel = panel.merge(tabpfn(c), on=KEYS, how="inner", validate="one_to_one")
    return panel, full_rows


def blend(panel: pd.DataFrame, lgb_cols: list[str], w: float, tab_cols: list[str]) -> np.ndarray:
    """expm1((1-w) * mean_s log1p(lgb_s) + w * mean_c log1p(tabpfn_c)); closed rows 0; clip at 0."""
    lg = np.mean([np.log1p(np.clip(panel[c].to_numpy(float), 0, None)) for c in lgb_cols], axis=0)
    if w == 0:
        mixed = lg
    else:
        tb = np.mean([np.log1p(np.clip(panel[c].to_numpy(float), 0, None)) for c in tab_cols], axis=0)
        mixed = (1 - w) * lg + w * tb
    out = np.clip(np.expm1(mixed), 0, None)
    return np.where(panel["IsOpen"].to_numpy() == 1, out, 0.0)


def gate_tables(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, fold_rows = [], []
    arms = {
        "seed07": [20260807], "seed08": [20260808], "seed09": [20260809], "3seed": list(SEEDS),
    }
    for arm, seeds in arms.items():
        cols = {
            "cs02": blend(panel, [f"cs02_{s}" for s in seeds], 0, []),
            "hp17": blend(panel, [f"hp17_{s}" for s in seeds], 0, []),
            "candidate": blend(panel, [f"hp17_{s}" for s in seeds], W_TABPFN, ["t365", "t730"]),
            "fallback_hp17_t365": blend(panel, [f"hp17_{s}" for s in seeds], W_TABPFN, ["t365"]),
        }
        for fold, idx in panel.groupby("fold").groups.items():
            sub = panel.loc[idx, ["IsOpen", "ActualOrderVolume"]].copy()
            for k, v in cols.items():
                sub[k] = v[panel.index.get_indexer(idx)]
            rec = {"arm": arm, "fold": int(fold), "rows": len(sub)}
            for k in cols:
                rec[f"{k}_all"], rec[f"{k}_excl0"] = scores(sub, k)
            fold_rows.append(rec)
    folds = pd.DataFrame(fold_rows)
    for arm, g in folds.groupby("arm", sort=False):
        g = g.set_index("fold")
        for cand in ("hp17", "candidate", "fallback_hp17_t365"):
            gain = g["cs02_all"] - g[f"{cand}_all"]
            gain0 = g["cs02_excl0"] - g[f"{cand}_excl0"]
            rows.append({
                "arm": arm, "candidate": cand,
                **{f"gain_f{f}": float(gain[f]) for f in g.index},
                "mean_gain_f2_4": float(gain.loc[[2, 3, 4]].mean()),
                "folds_improved_2_4": int((gain.loc[[2, 3, 4]] > 0).sum()),
                "mean_gain_f2_4_excl0": float(gain0.loc[[2, 3, 4]].mean()),
                "gain_f5_excl0": float(gain0[5]),
            })
    return folds, pd.DataFrame(rows)


def gate_verdict(gains: pd.DataFrame, cand: str) -> dict:
    g = gains[gains["candidate"] == cand].set_index("arm")
    per_seed = g.loc[["seed07", "seed08", "seed09"]]
    checks = {
        "mean_gain_f2_4_3seed_ge_0.0024": bool(g.loc["3seed", "mean_gain_f2_4"] >= GATE_MIN_GAIN),
        "every_seed_mean_gain_f2_4_ge_0.0024": bool((per_seed["mean_gain_f2_4"] >= GATE_MIN_GAIN).all()),
        "at_least_2_of_3_folds_improve_every_seed": bool((per_seed["folds_improved_2_4"] >= 2).all()),
        "same_sign_all_seeds": bool((per_seed["mean_gain_f2_4"] > 0).all()),
        "holds_excl_open_zero": bool(g.loc["3seed", "mean_gain_f2_4_excl0"] >= GATE_MIN_GAIN),
        "fold5_positive_every_seed": bool((per_seed["gain_f5"] > 0).all()),
    }
    return {"candidate": cand, "checks": checks, "passes": all(checks.values())}


def context_mix_check(panel: pd.DataFrame) -> dict:
    """Rebuild the 365/730 log-space mix and compare with the server's pre-averaged OOF file (folds 1-4)."""
    out = {}
    srv = read_oof("tabpfn_mix_f1-4_server.csv.gz")[KEYS + ["pred"]]
    t365 = tabpfn(365)
    t730 = tabpfn(730)
    m = t365.merge(t730, on=KEYS, validate="one_to_one")
    m["mix"] = np.expm1(0.5 * (np.log1p(m["t365"]) + np.log1p(m["t730"])))
    j = m.merge(srv, on=KEYS, how="inner", validate="one_to_one")
    out["oof_rows_compared"] = len(j)
    out["oof_max_abs_log_diff"] = float(np.max(np.abs(np.log1p(j["mix"]) - np.log1p(j["pred"]))))
    out["oof_folds"] = sorted(int(f) for f in j["fold"].unique())
    srv730 = read_oof("tabpfn730_f1-4_server.csv.gz")[KEYS + ["pred"]].merge(t730, on=KEYS, validate="one_to_one")
    out["oof730_rebuilt_from_split_files_max_abs_diff"] = float(np.max(np.abs(srv730["pred"] - srv730["t730"])))

    tw = ART / "test_window"
    a, b, mix = (pd.read_csv(tw / f) for f in ("tabpfn365_test.csv", "tabpfn730_test.csv", "tabpfn_mix_test.csv"))
    ids_equal = bool(np.array_equal(a["Id"], b["Id"]) and np.array_equal(a["Id"], mix["Id"]))
    rebuilt = np.expm1(0.5 * (np.log1p(a["OrderVolume"].to_numpy()) + np.log1p(b["OrderVolume"].to_numpy())))
    out["test_ids_equal_across_files"] = ids_equal
    out["test_mix_max_abs_log_diff"] = float(np.max(np.abs(np.log1p(rebuilt) - np.log1p(mix["OrderVolume"].to_numpy()))))
    out["test_mix_sha256"] = sha256(tw / "tabpfn_mix_test.csv")
    out["test365_sha256"] = sha256(tw / "tabpfn365_test.csv")
    out["test730_sha256"] = sha256(tw / "tabpfn730_test.csv")
    out["test365_vs_730_corr_log"] = float(np.corrcoef(np.log1p(a["OrderVolume"]), np.log1p(b["OrderVolume"]))[0, 1])
    m365 = json.loads((tw / "tabpfn365_manifest.json").read_text())
    m730 = json.loads((tw / "tabpfn730_manifest.json").read_text())
    out["test365_sha_matches_its_manifest"] = out["test365_sha256"] == m365["predictions_sha256"]
    out["test730_sha_matches_its_manifest"] = out["test730_sha256"] == m730["predictions_sha256"]
    out["history_last_date"] = [m365["history_last_date"], m730["history_last_date"]]
    out["forecast_origin"] = [m365["forecast_origin"], m730["forecast_origin"]]
    out["contexts"] = [m365["settings"]["context"], m730["settings"]["context"]]
    return out


def rehearsal_check(d: Path = ART / "rehearsal_mix") -> dict:
    """Rebuild submission_final.csv of a real-path run from its saved member predictions and the mix file; hard checks."""
    man = json.loads((d / "final_manifest.json").read_text())
    sub = pd.read_csv(d / "submission_final.csv")
    test = pd.read_csv(RAW / "orders_test.csv")
    sample = pd.read_csv(RAW / "sample_submission.csv")
    mix = pd.read_csv(ART / "test_window" / "tabpfn_mix_test.csv")
    seeds = [np.load(d / f"test_pred_lightgbm_s{s}.npy") for s in SEEDS]
    pre = np.load(d / "test_pred_members_pre_blend.npy")
    is_open = test["IsOpen"].fillna(1).to_numpy()
    lg = np.mean([np.log1p(np.clip(p, 0, None)) for p in seeds], axis=0)
    pre_rebuilt = np.where(is_open == 1, np.expm1(lg), 0.0)
    final_rebuilt = np.where(is_open == 1, np.clip(np.expm1(0.8 * lg + 0.2 * np.log1p(mix["OrderVolume"].to_numpy())), 0, None), 0.0)
    v = sub["OrderVolume"].to_numpy(float)
    closed = is_open == 0
    return {
        "submission_sha256": sha256(d / "submission_final.csv"),
        "submission_sha_matches_manifest": sha256(d / "submission_final.csv") == man["_run"]["submission_sha256"],
        "blend_file_sha_matches_manifest": sha256(ART / "test_window" / "tabpfn_mix_test.csv") == man["_run"]["blend_file_sha256"],
        "rows": len(sub),
        "rows_ok": len(sub) == 46_830,
        "id_order_equals_sample_submission": bool(np.array_equal(sub["Id"], sample["Id"])),
        "id_order_equals_test_frame": bool(np.array_equal(sub["Id"], test["Id"])),
        "any_nan_or_inf": bool(~np.isfinite(v).all()),
        "any_negative": bool((v < 0).any()),
        "closed_rows": int(closed.sum()),
        "closed_rows_exact_zero": bool((v[closed] == 0).all()),
        "open_rows_predicted_zero": int((v[~closed] == 0).sum()),
        "pre_blend_max_abs_log_diff": float(np.max(np.abs(np.log1p(pre_rebuilt) - np.log1p(pre)))),
        "final_formula_max_abs_log_diff": float(np.max(np.abs(np.log1p(final_rebuilt) - np.log1p(v)))),
        "manifest_recipe": man["_run"]["recipe"], "manifest_variant": man["_run"]["config"]["variant"],
        "manifest_blend_weight": man["_run"]["blend_weight"], "manifest_space": man["_run"]["blend"]["external"]["space"],
        "manifest_training_origins": man["_run"]["training_origins"], "manifest_last_label_date": man["_run"]["last_label_date"],
        "manifest_has_exact_command": "command" in man["_run"],
        "manifest_versions": man["_run"]["versions"],
        "model_sha256": {k: v["sha256"] for k, v in man.items() if k.startswith("lightgbm_")},
    }


def run_to_run(a: Path, b: Path) -> dict:
    """How far two real-path runs of the same recipe are apart (log1p scale, open rows)."""
    out = {}
    for s in SEEDS:
        x, y = np.load(a / f"test_pred_lightgbm_s{s}.npy"), np.load(b / f"test_pred_lightgbm_s{s}.npy")
        d = np.log1p(x) - np.log1p(y)
        out[f"seed_{s}_max_abs_log_diff"] = float(np.abs(d).max())
        out[f"seed_{s}_rms_log_diff_open"] = float(np.sqrt(np.mean(d[x > 0] ** 2)))
    x = pd.read_csv(a / "submission_final.csv")["OrderVolume"].to_numpy(float)
    y = pd.read_csv(b / "submission_final.csv")["OrderVolume"].to_numpy(float)
    d, o = np.log1p(x) - np.log1p(y), x > 0
    out.update({"forecast_max_abs_log_diff": float(np.abs(d).max()), "forecast_rms_log_diff_open": float(np.sqrt(np.mean(d[o] ** 2))),
                "forecast_mean_log_diff_open": float(d[o].mean()),
                "forecast_corr_log_open": float(np.corrcoef(np.log1p(x[o]), np.log1p(y[o]))[0, 1])})
    return out


def main() -> None:
    RES.mkdir(exist_ok=True)
    panel, full_rows = build_panel()
    folds, gains = gate_tables(panel)
    folds.to_csv(RES / "audit_fold_scores.csv", index=False, float_format="%.5f")
    gains.to_csv(RES / "audit_gate_gains.csv", index=False, float_format="%.5f")
    summary = {
        "rows_per_fold_all_members": panel.groupby("fold").size().to_dict(),
        "rows_per_fold_lightgbm_only": full_rows,
        "fold5_hubs_in_comparison": int(panel.loc[panel["fold"] == 5, "HubID"].nunique()),
        "open_zero_rows_per_fold": panel.assign(z=(panel["IsOpen"] == 1) & (panel["ActualOrderVolume"] == 0)).groupby("fold")["z"].sum().astype(int).to_dict(),
        "gate": {c: gate_verdict(gains, c) for c in ("hp17", "candidate", "fallback_hp17_t365")},
        "context_mix": context_mix_check(panel),
        "rehearsal_2": rehearsal_check(),
    }
    final_dir = ROOT / "final"
    if (final_dir / "final_manifest.json").exists():
        summary["final_retrain"] = rehearsal_check(final_dir)
        summary["final_vs_rehearsal_2"] = run_to_run(final_dir, ART / "rehearsal_mix")
        (final_dir / "freeze_checks.json").write_text(json.dumps(summary["final_retrain"], indent=2, default=str))
    (RES / "audit_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(gains.round(4).to_string(index=False))
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
