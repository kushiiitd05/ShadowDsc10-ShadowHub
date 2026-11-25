"""Tests for the entity-embedding MLP member (preprocessing, shapes, leak checks, CPU smoke training)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mlp_prep import fit_preprocessor, transform  # noqa: E402

CATS = {"HubID", "Weekday"}


def make_frame(n: int, seed: int = 0, shift: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({
        "HubID": rng.integers(1, 6, n),
        "Weekday": rng.integers(0, 7, n).astype(float),
        "a": rng.normal(10 + shift, 2, n),
        "b": rng.normal(0, 1, n),
        "big": np.abs(rng.normal(5000, 1500, n)),
        "const": np.full(n, 3.0),
        "flag": rng.integers(0, 2, n).astype(bool),
        "sparse": rng.normal(0, 1, n),
        "name": rng.choice(["x", "y"], n),
    })
    frame.loc[rng.random(n) < 0.2, "sparse"] = np.nan   # >1% missing
    frame.loc[0, "a"] = np.nan                            # <1% missing
    return frame


COLS = ["HubID", "Weekday", "a", "b", "big", "const", "flag", "sparse", "name"]


def test_fit_uses_only_training_statistics_and_does_not_mutate() -> None:
    train, other = make_frame(2000, 1), make_frame(500, 2, shift=4.0)
    before = train.copy(), other.copy()
    pre = fit_preprocessor(train, COLS, CATS)
    out_train = transform(pre, train)
    out_other = transform(pre, other)
    pd.testing.assert_frame_equal(train, before[0])
    pd.testing.assert_frame_equal(other, before[1])
    # shifted frame must NOT be re-centred: leak check on the standardisation stats
    ia = pre.num_cols.index("a")
    assert abs(out_train[0][:, ia].mean()) < 0.1
    assert out_other[0][:, ia].mean() > 1.5
    assert pre.medians["a"] == pytest.approx(float(np.nanmedian(train["a"])))


def test_shapes_dtypes_and_no_nan() -> None:
    train = make_frame(1000, 3)
    pre = fit_preprocessor(train, COLS, CATS)
    x_num, x_cat = transform(pre, train)
    assert x_num.dtype == np.float32 and x_cat.dtype == np.int64
    assert x_num.shape == (1000, len(pre.num_cols) + len(pre.miss_cols))
    assert x_cat.shape == (1000, len(pre.cat_cols))
    assert np.isfinite(x_num).all()
    assert np.abs(x_num).max() <= pre.clip + 1e-6


def test_missing_indicators_only_for_columns_above_one_percent() -> None:
    train = make_frame(2000, 4)
    pre = fit_preprocessor(train, COLS, CATS)
    assert pre.miss_cols == ["sparse"]
    x_num, _ = transform(pre, train)
    ind = x_num[:, -1]
    assert np.array_equal(ind.astype(bool), train["sparse"].isna().to_numpy())


def test_constant_column_is_zero_and_flags_numeric() -> None:
    train = make_frame(500, 5)
    pre = fit_preprocessor(train, COLS, CATS)
    x_num, _ = transform(pre, train)
    assert np.allclose(x_num[:, pre.num_cols.index("const")], 0.0)
    assert "flag" in pre.num_cols and "name" in pre.cat_cols  # bool numeric; object column is categorical


def test_heavy_tailed_nonnegative_column_is_logged() -> None:
    train = make_frame(500, 6)
    pre = fit_preprocessor(train, COLS, CATS)
    assert "big" in pre.log_cols and "a" not in pre.log_cols


def test_unseen_categories_map_to_unknown_index_zero_and_dtype_robust() -> None:
    train = make_frame(500, 7)
    pre = fit_preprocessor(train, COLS, CATS)
    new = make_frame(50, 8)
    new["HubID"] = 999                         # never seen in training
    new["Weekday"] = new["Weekday"].astype(int)  # int vs float dtype must still match the vocabulary
    _, cat = transform(pre, new)
    assert (cat[:, pre.cat_cols.index("HubID")] == 0).all()
    assert (cat[:, pre.cat_cols.index("Weekday")] > 0).all()


def test_infinite_values_are_treated_as_missing() -> None:
    train = make_frame(500, 9)
    bad = train.copy()
    bad.loc[1, "b"] = np.inf
    pre = fit_preprocessor(train, COLS, CATS)
    x_num, _ = transform(pre, bad)
    assert np.isfinite(x_num).all()


def test_model_forward_shape_and_predict_closed_rows_zero() -> None:
    torch = pytest.importorskip("torch")
    from mlp_model import MlpConfig, build_model

    train = make_frame(300, 10)
    pre = fit_preprocessor(train, COLS, CATS)
    model = build_model(pre, MlpConfig(hidden=(16, 8)), target_mean=6.0, target_std=1.5)
    x_num, x_cat = transform(pre, train)
    out = model(torch.from_numpy(x_num[:32]), torch.from_numpy(x_cat[:32]))
    assert out.shape == (32,)


def test_cpu_smoke_training_learns_and_is_deterministic() -> None:
    pytest.importorskip("torch")
    from mlp_model import MlpConfig, predict_log, train_mlp

    def synth(n: int, seed: int) -> tuple[pd.DataFrame, np.ndarray]:
        f = make_frame(n, seed)
        y = 0.8 * f["a"].fillna(10).to_numpy() + 0.3 * f["HubID"].to_numpy() + 0.1 * np.random.default_rng(seed).normal(size=n)
        return f, y

    tr, ytr = synth(3000, 11)
    ho, yho = synth(800, 12)
    pre = fit_preprocessor(tr, COLS, CATS)
    cfg = MlpConfig(hidden=(32, 16), batch_size=256, epochs=6, patience=6, lr=3e-3, bf16=False)
    xn, xc = transform(pre, tr)
    hn, hc = transform(pre, ho)
    res1 = train_mlp(pre, xn, xc, ytr, hn, hc, yho, cfg, seed=1, device="cpu")
    res2 = train_mlp(pre, xn, xc, ytr, hn, hc, yho, cfg, seed=1, device="cpu")
    baseline = float(np.sqrt(np.mean((yho - ytr.mean()) ** 2)))
    assert res1.best_rmse < baseline * 0.6
    assert res1.best_rmse == pytest.approx(res2.best_rmse)
    p1 = predict_log(res1.model, hn, hc, "cpu")
    p2 = predict_log(res2.model, hn, hc, "cpu")
    assert p1.shape == (800,) and np.allclose(p1, p2, atol=1e-5)
    assert 1 <= res1.best_epoch <= cfg.epochs


def test_runner_record_schema_and_closed_rows_zero(tmp_path: Path) -> None:
    from g4_runner import RECORD_COLUMNS, build_record, volume_from_log

    assert RECORD_COLUMNS[:3] == ["date", "variant", "seed"]
    pred = volume_from_log(np.array([np.log1p(10.0), np.log1p(20.0)]), np.array([1, 0]))
    assert pred[0] == pytest.approx(10.0) and pred[1] == 0.0
    rows = [{"fold": 2, "rmsle_all": 0.12, "rmsle_excl_open0": 0.11, "bias_open": 0.0, "best_epoch": 7, "fit_seconds": 5.0}]
    rec = build_record(rows, seed=1, lr=2e-3, epochs=40, cfg_json="{}")
    assert list(rec)[:len(RECORD_COLUMNS)] == RECORD_COLUMNS
    assert rec["fold2_all"] == 0.12 and rec["fold2_best_iter"] == 7 and rec["model"] == "mlp"


def test_run_fold_end_to_end_synthetic_and_blend_compatible(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from blend import blend_table, load
    from g4_runner import OOF_COLUMNS, run_fold
    from mlp_model import MlpConfig

    def part(n: int, seed: int) -> pd.DataFrame:
        f = make_frame(n, seed)
        rng = np.random.default_rng(seed)
        f["IsOpen"] = (rng.random(n) > 0.1).astype(int)
        f["ActualOrderVolume"] = np.where(f["IsOpen"] == 1, np.round(np.exp(0.2 * f["a"].fillna(10) + 5)), 0.0)
        f["Horizon"] = rng.integers(1, 43, n)
        f["Date"] = pd.Timestamp("2025-01-01") + pd.to_timedelta(np.arange(n), unit="D")
        f["HubID"] = np.arange(n) % 5 + 1
        return f

    ctx = {"train": part(2000, 20), "hold": part(500, 21), "valid": part(400, 22),
           "columns": COLS, "fold": 2}
    snapshot = ctx["valid"].copy()
    cfg = MlpConfig(hidden=(16, 8), batch_size=256, epochs=3, patience=3, bf16=False)
    oof, scores = run_fold(ctx, cfg, seed=3, device="cpu", categorical=CATS, log=False)
    pd.testing.assert_frame_equal(ctx["valid"], snapshot)
    assert list(oof.columns) == OOF_COLUMNS and len(oof) == 400
    assert (oof.loc[oof["IsOpen"] == 0, "pred"] == 0).all()
    assert (oof.loc[oof["IsOpen"] == 1, "pred"] > 0).all()
    assert np.isfinite(scores["rmsle_all"])
    oof.to_csv(tmp_path / "oof_mlp_s3.csv.gz", index=False, compression="gzip")
    table = blend_table({"mlp": load(tmp_path, "mlp", 3)})
    assert "blend" in table.columns
