import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import final_retrain as fr  # noqa: E402
import sanity_checks as sc  # noqa: E402


# ---------------------------------------------------------------- manifest hashing
def test_sha256_matches_hashlib_and_changes_with_content(tmp_path: Path) -> None:
    p = tmp_path / "m.txt"
    p.write_bytes(b"tree=1\n" * 200_000)  # > one 1 MiB read block
    assert fr.sha256_file(p) == hashlib.sha256(p.read_bytes()).hexdigest()
    before = fr.sha256_file(p)
    p.write_bytes(p.read_bytes() + b"x")
    assert fr.sha256_file(p) != before


def test_member_entry_has_control_manifest_keys(tmp_path: Path) -> None:
    p = tmp_path / "lightgbm_s1.txt"
    p.write_text("model")
    entry = fr.member_entry(p, 1000, seed=1)
    assert {"file", "sha256", "best_iteration", "log_offset_c"} <= set(entry)
    assert entry["file"] == "lightgbm_s1.txt" and entry["best_iteration"] == 1000 and entry["log_offset_c"] == 0.0
    json.dumps(entry)  # serialisable


def test_final_config_is_cs02_fixed_1000() -> None:
    cfg = fr.final_config()
    assert (cfg.colsample, cfg.subsample_freq, cfg.num_leaves, cfg.min_child, cfg.target, cfg.te, cfg.fixed_rounds) == (0.2, 1, 63, 80, "log", True, 1000)
    params = fr.lgb_params(cfg, 0.025, 20260807, 16)
    assert params["n_estimators"] == 1000 and params["learning_rate"] == 0.025 and params["subsample"] == 0.85
    assert params["colsample_bytree"] == 0.2 and params["subsample_freq"] == 1 and params["random_state"] == 20260807


def test_saved_model_reproduces_predictions(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    x = pd.DataFrame({"a": rng.normal(size=500), "b": rng.normal(size=500), "c": pd.Categorical(rng.choice(["u", "v", "w"], 500))})
    y = x["a"].to_numpy() * 2 + rng.normal(size=500) * 0.1
    params = dict(n_estimators=20, learning_rate=0.1, num_leaves=7, min_child_samples=5, random_state=1, n_jobs=1, verbosity=-1)
    path = tmp_path / "m.txt"
    model = fr.fit_and_save(x, y, params, path)
    fr.verify_saved(model, path, x)  # raises when the file differs from the live model


# ---------------------------------------------------------------- averaging and zeroing
def test_log_mean_blend_is_geometric_in_log1p_space() -> None:
    a, b = np.array([0.0, 99.0, 9.0]), np.array([0.0, 399.0, 9.0])
    out = fr.log_mean_blend([a, b], np.array([1, 1, 1]))
    assert out == pytest.approx(np.expm1((np.log1p(a) + np.log1p(b)) / 2))
    assert out[1] == pytest.approx(np.expm1((np.log(100) + np.log(400)) / 2))
    assert out[1] < (99 + 399) / 2  # not the arithmetic mean


def test_log_mean_single_member_is_identity_and_inputs_unchanged() -> None:
    p = np.array([5.0, 120.0])
    keep = p.copy()
    assert fr.log_mean_blend([p], np.array([1, 1])) == pytest.approx(p)
    assert np.array_equal(p, keep)
    with pytest.raises(ValueError):
        fr.log_mean_blend([], np.array([1]))


def test_closed_rows_zeroed_and_negatives_clipped() -> None:
    pred = np.array([10.0, 20.0, -3.0, 5.0])
    is_open = np.array([1, 0, 1, 0])
    out = fr.zero_closed(pred, is_open)
    assert out.tolist() == [10.0, 0.0, 0.0, 0.0]
    assert pred[1] == 20.0  # input untouched
    blended = fr.log_mean_blend([np.array([10.0, 20.0]), np.array([30.0, 40.0])], np.array([1, 0]))
    assert blended[1] == 0.0 and blended[0] > 0


def test_combine_members_modes() -> None:
    g = [np.array([100.0]), np.array([200.0]), np.array([300.0])]
    t = [np.array([1000.0])]
    o = np.array([1])
    member = fr.combine_members(g, t, o, "member")
    expected_member = np.expm1(np.mean([np.log1p(x[0]) for x in [*g, *t]]))
    assert member[0] == pytest.approx(expected_member)
    half = fr.combine_members(g, t, o, "half")
    g_avg = np.mean([np.log1p(x[0]) for x in g])
    assert half[0] == pytest.approx(np.expm1((g_avg + np.log1p(1000.0)) / 2))
    assert fr.combine_members(g, [], o)[0] == pytest.approx(np.expm1(g_avg))
    with pytest.raises(ValueError):
        fr.combine_members(g, t, o, "bogus")


# ---------------------------------------------------------------- origin eligibility (no label leakage)
def _manifest() -> pd.DataFrame:
    origins = pd.to_datetime(["2015-03-27", "2015-05-03", "2015-05-08"])
    return pd.DataFrame({"origin": origins, "target_start": origins + pd.Timedelta(days=1), "target_end": origins + pd.Timedelta(days=42)})


def test_eligible_origins_use_label_end_not_origin() -> None:
    m = _manifest()
    real = fr.eligible_origins(m, pd.Timestamp("2015-06-19"))
    assert len(real) == 3 and real["target_end"].max() == pd.Timestamp("2015-06-19")
    dry = fr.eligible_origins(m, pd.Timestamp("2015-05-08"))
    assert list(dry["origin"].dt.strftime("%Y-%m-%d")) == ["2015-03-27"]


def test_label_leakage_assertion() -> None:
    m = _manifest()
    fr.assert_labels_before(fr.eligible_origins(m, pd.Timestamp("2015-06-19")), pd.Timestamp("2015-06-20"))
    with pytest.raises(AssertionError):
        fr.assert_labels_before(m, pd.Timestamp("2015-06-19"))  # origin 2015-05-08 labels end on 06-19
    with pytest.raises(AssertionError):
        fr.assert_labels_before(m.iloc[0:0], pd.Timestamp("2015-06-20"))


def test_safe_name() -> None:
    assert fr.safe_name("a/b c") == "a_b_c" and fr.safe_name("") == "x"


# ---------------------------------------------------------------- sanity checks on synthetic frames
def _synthetic(n: int = 200, seed: int = 0):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({"Id": np.arange(1, n + 1), "IsOpen": (rng.random(n) > 0.15).astype(int), "Horizon": np.tile(np.arange(1, 5), n // 4)})
    volume = np.where(frame["IsOpen"] == 1, rng.lognormal(8, 0.5, n), 0.0)
    ref = pd.DataFrame({"Id": frame["Id"], "OrderVolume": volume})
    pred = pd.DataFrame({"Id": frame["Id"], "OrderVolume": np.where(frame["IsOpen"] == 1, volume * rng.normal(1.0, 0.02, n), 0.0)})
    return pred, frame, ref


def _names(checks, ok: bool) -> set[str]:
    return {c.name for c in checks if c.passed == ok}


def test_clean_predictions_pass_all_checks() -> None:
    pred, frame, ref = _synthetic()
    checks, summary = sc.run_checks(pred, frame, ref, sample_ids=ref["Id"].to_numpy(), expected_rows=200)
    assert all(c.passed for c in checks), [c for c in checks if not c.passed]
    assert summary["correlation_log1p"] > 0.99 and set(summary["per_horizon_mean_log1p"]) == {1, 2, 3, 4}


def test_closed_row_nonzero_nan_and_negative_are_caught() -> None:
    pred, frame, ref = _synthetic()
    closed = int(np.flatnonzero(frame["IsOpen"] == 0)[0])
    bad = pred.copy()
    bad.loc[closed, "OrderVolume"] = 5.0
    assert "closed_rows_zero" in _names(sc.run_checks(bad, frame, ref, expected_rows=200)[0], False)
    bad = pred.copy()
    bad.loc[3, "OrderVolume"] = np.nan
    assert "no_nan_inf" in _names(sc.run_checks(bad, frame, ref, expected_rows=200)[0], False)
    bad = pred.copy()
    opened = int(np.flatnonzero(frame["IsOpen"] == 1)[0])
    bad.loc[opened, "OrderVolume"] = -1.0
    assert "no_negative" in _names(sc.run_checks(bad, frame, ref, expected_rows=200)[0], False)


def test_id_order_and_row_count_are_caught() -> None:
    pred, frame, ref = _synthetic()
    swapped = pred.copy()
    swapped.loc[[0, 1], "Id"] = [2, 1]
    assert "id_order_vs_reference" in _names(sc.run_checks(swapped, frame, ref, expected_rows=200)[0], False)
    assert "id_order_vs_sample_submission" in _names(sc.run_checks(swapped, frame, None, sample_ids=ref["Id"].to_numpy(), expected_rows=200)[0], False)
    short = pred.iloc[:-1]
    assert "row_count" in _names(sc.run_checks(short, frame, ref, expected_rows=200)[0], False)
    assert "frame_alignment" in _names(sc.run_checks(short, frame, ref, expected_rows=200)[0], False)


def test_distribution_shift_is_caught() -> None:
    pred, frame, ref = _synthetic()
    shifted = pred.copy()
    shifted["OrderVolume"] = shifted["OrderVolume"] * 3.0
    failed = _names(sc.run_checks(shifted, frame, ref, expected_rows=200)[0], False)
    assert {"mean_log1p_vs_reference", "per_horizon_mean_vs_reference"} <= failed
    noisy = pred.copy()
    noisy["OrderVolume"] = np.where(frame["IsOpen"] == 1, np.random.default_rng(5).lognormal(8, 0.5, len(pred)), 0.0)
    assert "correlation_vs_reference" in _names(sc.run_checks(noisy, frame, ref, expected_rows=200)[0], False)


def test_cli_dry_run_mode_and_exit_code(tmp_path: Path) -> None:
    pred, frame, ref = _synthetic()
    table = pred.assign(IsOpen=frame["IsOpen"], Horizon=frame["Horizon"], ActualOrderVolume=ref["OrderVolume"])
    path = tmp_path / "dry.csv"
    table.to_csv(path, index=False)
    assert sc.main([str(path), "--dry-run", "--expected-rows", "200"]) == 0
    assert sc.main([str(path), "--dry-run"]) == 1  # default expects the real 46,830 rows
