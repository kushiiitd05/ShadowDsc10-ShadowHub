from __future__ import annotations

import pandas as pd

import summarize_offline as s


def _seed(gains, excl=None):
    return {"mean_gain": sum(gains) / 3, "min_fold_gain": min(gains), "gate_all": sum(g >= 0.0024 for g in gains) >= 2,
            "gate_excl": sum(g >= 0.0024 for g in (excl or gains)) >= 2}


def test_verdict_labels():
    assert s.verdict([_seed([0.004, 0.003, 0.0]), _seed([0.005, 0.003, 0.001])]) == "pass"
    assert s.verdict([_seed([0.002, 0.0015, 0.001]), _seed([0.0018, 0.002, 0.001])]) == "challenger"
    assert s.verdict([_seed([0.002, 0.002, -0.001]), _seed([0.002, 0.002, 0.001])]) == "not adopted"
    assert s.verdict([_seed([0.0016, 0.0011, 0.0013]), _seed([0.0012, 0.0015, 0.0013])]) == "challenger"
    assert s.verdict([_seed([0.0003, 0.0002, 0.0001])]) == "not adopted"


def test_summarize_reads_tables(tmp_path):
    rows = []
    for seed in (1, 2):
        for fold in (2, 3, 4):
            rows.append({"seed": seed, "variant": "fit", "window": 14, "weights": None, "fold": fold, "gain_all": 0.003, "gain_excl": 0.003})
    base = pd.DataFrame(rows)
    for name in ("C_peer_resid", "D_shift", "E_reconcile"):
        base.assign(variant=lambda d: d["variant"]).to_csv(tmp_path / f"{name}.csv", index=False)
    pd.concat([base.assign(variant=v, window=28) for v in ("group", "corr5", "own+corr5")]).to_csv(tmp_path / "C_peer_resid.csv", index=False)
    pd.concat([base.assign(variant=v, weights="clip01") for v in ("format_pattern", "format_pattern2")] +
              [base.assign(variant="format_pattern2", weights="ols")]).to_csv(tmp_path / "E_reconcile.csv", index=False)
    pd.concat([base.assign(variant=v, window=14) for v in ("fit", "literal", "fit_h")]).to_csv(tmp_path / "D_shift.csv", index=False)
    out = s.summarize(tmp_path)
    assert (out["verdict"] == "pass").all() and len(out) == 9
