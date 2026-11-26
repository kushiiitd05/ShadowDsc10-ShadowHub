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


# ---------------------------------------------------------------- recipe selection (cs02 default, hp17 on request)
def test_recipe_default_is_cs02_and_hp17_matches_the_registered_variant() -> None:
    assert fr.final_config() == fr.final_config("cs02")
    cfg = fr.final_config("hp17")
    assert (cfg.colsample, cfg.subsample_freq, cfg.num_leaves, cfg.min_child, cfg.reg_lambda, cfg.fixed_rounds) == (0.15, 1, 255, 160, 24.1, 1000)
    params = fr.lgb_params(cfg, 0.025, 7, 16)
    assert params["learning_rate"] == 0.025 and params["n_estimators"] == 1000 and params["num_leaves"] == 255
    assert params["min_child_samples"] == 160 and params["reg_lambda"] == 24.1 and params["colsample_bytree"] == 0.15
    with pytest.raises(ValueError):
        fr.final_config("hp18")


def test_recipe_drift_from_the_registered_variant_is_refused(monkeypatch) -> None:
    from dataclasses import replace
    drifted = replace(fr.VARIANTS["hp17_fix1000"], num_leaves=511)
    monkeypatch.setitem(fr.VARIANTS, "hp17_fix1000", drifted)
    with pytest.raises(AssertionError, match="drifted"):
        fr.final_config("hp17")
    fr.final_config("cs02")  # the other recipe is unaffected


def test_cli_defaults_keep_the_cs02_behaviour() -> None:
    args = fr.parse_args(["proj", "out"])
    assert args.recipe == "cs02" and args.pertier_recipe == "cs02" and args.blend_file is None and args.blend_weight is None
    hp = fr.parse_args(["proj", "out", "--recipe", "hp17", "--pertier-recipe", "hp17", "--blend-file", "x.csv"])
    assert hp.recipe == "hp17" and hp.pertier_recipe == "hp17"
    with pytest.raises(SystemExit):
        fr.parse_args(["proj", "out", "--recipe", "other"])


def test_blend_arguments_must_come_as_a_pair() -> None:
    assert fr.resolve_blend(fr.parse_args(["p", "o"])) == (None, 0.0)
    ok = fr.resolve_blend(fr.parse_args(["p", "o", "--blend-file", "f.csv"]))
    assert ok == (Path("f.csv"), fr.BLEND_WEIGHT_DEFAULT) and fr.BLEND_WEIGHT_DEFAULT == 0.2
    assert fr.resolve_blend(fr.parse_args(["p", "o", "--blend-file", "f.csv", "--blend-weight", "0.3"]))[1] == 0.3
    with pytest.raises(SystemExit):
        fr.resolve_blend(fr.parse_args(["p", "o", "--blend-weight", "0.2"]))  # weight without a file
    for bad in ("0", "1.5", "-0.1"):
        with pytest.raises(SystemExit):
            fr.resolve_blend(fr.parse_args(["p", "o", "--blend-file", "f.csv", "--blend-weight", bad]))


# ---------------------------------------------------------------- external blend in log1p space
def test_external_blend_is_log1p_space_arithmetic() -> None:
    base, ext = np.array([100.0, 0.0, 50.0, 7.0]), np.array([400.0, 10.0, 0.0, 7.0])
    out = fr.blend_external(base, ext, 0.2, np.array([1, 1, 1, 1]))
    assert out == pytest.approx(np.expm1(0.8 * np.log1p(base) + 0.2 * np.log1p(ext)))
    assert out[0] == pytest.approx(np.expm1(0.8 * np.log(101) + 0.2 * np.log(401)))
    assert out[0] < 0.8 * 100 + 0.2 * 400  # not the arithmetic blend
    assert out[3] == pytest.approx(7.0)  # equal inputs are a fixed point
    assert fr.blend_external(base, ext, 0.0, np.ones(4)) == pytest.approx(base)
    assert fr.blend_external(base, ext, 1.0, np.ones(4)) == pytest.approx(ext)


def test_external_blend_on_top_of_the_seed_mean_matches_the_formula() -> None:
    seeds = [np.array([100.0, 30.0]), np.array([200.0, 40.0]), np.array([300.0, 50.0])]
    pertier = [np.array([150.0, 35.0])]
    ext = np.array([500.0, 20.0])
    is_open = np.array([1, 1])
    base = fr.combine_members(seeds, [], is_open, "member")
    mean_log = np.mean([np.log1p(m) for m in seeds], axis=0)
    assert fr.blend_external(base, ext, 0.2, is_open) == pytest.approx(np.expm1(0.8 * mean_log + 0.2 * np.log1p(ext)))
    base = fr.combine_members(seeds, pertier, is_open, "member")
    mean_log = np.mean([np.log1p(m) for m in [*seeds, *pertier]], axis=0)
    assert fr.blend_external(base, ext, 0.2, is_open) == pytest.approx(np.expm1(0.8 * mean_log + 0.2 * np.log1p(ext)))


def test_external_blend_zeroes_closed_rows_and_never_mutates_inputs() -> None:
    base, ext = np.array([100.0, 100.0, 5.0]), np.array([300.0, 300.0, 5.0])
    is_open = np.array([1, 0, 1])
    keep = base.copy(), ext.copy(), is_open.copy()
    out = fr.blend_external(base, ext, 0.2, is_open)
    assert out[1] == 0.0 and out[0] > 0 and (out >= 0).all()
    assert all(np.array_equal(a, b) for a, b in zip(keep, (base, ext, is_open)))


def test_external_blend_refuses_bad_weights_and_non_finite_or_negative_external() -> None:
    base, o = np.array([1.0, 2.0]), np.array([1, 1])
    for w in (-0.1, 1.1, float("nan")):
        with pytest.raises(ValueError):
            fr.blend_external(base, np.array([1.0, 2.0]), w, o)
    for bad in (np.nan, np.inf, -3.0):
        with pytest.raises(ValueError):
            fr.blend_external(base, np.array([1.0, bad]), 0.2, o)
    with pytest.raises(ValueError):
        fr.blend_external(base, np.array([1.0]), 0.2, o)  # length mismatch


# ---------------------------------------------------------------- the external file must match the test frame
def _test_frame() -> pd.DataFrame:
    return pd.DataFrame({"Id": [11, 12, 13, 14], "HubID": [1, 1, 2, 2], "IsOpen": [1, 0, 1, 1],
                         "Date": pd.to_datetime(["2015-06-20", "2015-06-21"] * 2)})


def test_load_external_accepts_exact_ids_and_returns_frame_order(tmp_path: Path) -> None:
    path = tmp_path / "ext.csv"
    pd.DataFrame({"Id": [11, 12, 13, 14], "OrderVolume": [5.0, 0.0, 7.0, 9.0]}).to_csv(path, index=False)
    out = fr.load_external_test(path, _test_frame()["Id"])
    assert out.tolist() == [5.0, 0.0, 7.0, 9.0]


@pytest.mark.parametrize("ids", [[11, 12, 13, 99], [11, 12, 14, 13], [11, 12, 13], [11, 12, 13, 14, 15], [11, 11, 13, 14]])
def test_load_external_refuses_any_id_mismatch(tmp_path: Path, ids) -> None:
    path = tmp_path / "ext.csv"
    pd.DataFrame({"Id": ids, "OrderVolume": np.arange(1.0, len(ids) + 1)}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Id"):
        fr.load_external_test(path, _test_frame()["Id"])


def test_load_external_refuses_missing_columns_and_bad_values(tmp_path: Path) -> None:
    path = tmp_path / "ext.csv"
    pd.DataFrame({"Id": [11, 12, 13, 14], "pred": [1.0, 2.0, 3.0, 4.0]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="columns"):
        fr.load_external_test(path, _test_frame()["Id"])
    pd.DataFrame({"Id": [11, 12, 13, 14], "OrderVolume": [1.0, np.nan, 3.0, 4.0]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="non-finite"):
        fr.load_external_test(path, _test_frame()["Id"])


def test_load_external_oof_aligns_by_key_for_the_dry_run(tmp_path: Path) -> None:
    frame = _test_frame()
    oof = pd.DataFrame({"fold": 4, "HubID": [2, 1, 2, 1], "Date": ["2015-06-21", "2015-06-20", "2015-06-20", "2015-06-21"],
                        "pred": [90.0, 50.0, 70.0, 0.0]})
    other_fold = oof.assign(fold=3, pred=1e6)
    path = tmp_path / "oof.csv.gz"
    pd.concat([oof, other_fold]).to_csv(path, index=False)
    out = fr.load_external_oof(path, frame, 4)
    assert out.tolist() == [50.0, 0.0, 70.0, 90.0]  # frame order (hub, date), not file order, other fold ignored
    with pytest.raises(ValueError, match="no prediction"):
        fr.load_external_oof(path, frame, 2)
    path2 = tmp_path / "oof2.csv.gz"
    oof.iloc[:3].to_csv(path2, index=False)
    with pytest.raises(ValueError, match="no prediction"):
        fr.load_external_oof(path2, frame, 4)
    path3 = tmp_path / "oof3.csv.gz"
    pd.concat([oof, oof.iloc[:1]]).to_csv(path3, index=False)
    with pytest.raises(ValueError, match="duplicate"):
        fr.load_external_oof(path3, frame, 4)


# ---------------------------------------------------------------- manifest fields
def test_run_manifest_records_recipe_blend_weight_and_hashes(tmp_path: Path) -> None:
    blend = tmp_path / "ext.csv"
    blend.write_text("Id,OrderVolume\n1,2\n")
    cfg = fr.final_config("hp17")
    info = fr.recipe_block("hp17", "cs02", cfg, 0.025)
    assert info["recipe"] == "hp17" and info["variant"] == "hp17_fix1000" and info["pertier_recipe"] == "cs02"
    assert info["learning_rate"] == 0.025 and info["fixed_rounds"] == 1000 and info["num_leaves"] == 255
    block = fr.blend_block(mode="member", with_pertier=True, pertier_seeds=[1], fallback_rows=0, blend_file=blend, weight=0.2)
    assert block["external"] == {"file": "ext.csv", "sha256": hashlib.sha256(blend.read_bytes()).hexdigest(), "weight": 0.2, "space": "log1p"}
    none = fr.blend_block(mode="member", with_pertier=False, pertier_seeds=[], fallback_rows=0, blend_file=None, weight=0.0)
    assert none["external"] is None
    json.dumps([info, block, none])
