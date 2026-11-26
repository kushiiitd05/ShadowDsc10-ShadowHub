import numpy as np
import pytest

from coreset import column_importance_subset, hub_stratified_indices, relative_gap, relative_l2, sample_coreset, sensitivities, uniform_indices


def test_sensitivity_total_is_between_two_and_t_plus_one() -> None:
    rng = np.random.default_rng(0)
    losses = rng.random((5, 1000)) ** 3  # skewed losses of 5 rough models
    s = sensitivities(losses)
    assert s.shape == (1000,)
    assert 2.0 <= s.sum() <= 6.0 + 1e-9  # S in [2, T+1]
    assert (s >= 1.0 / 1000).all()


def test_sensitivity_uses_sum_total_not_mean() -> None:
    losses = np.array([[1.0, 1.0, 1.0, 1.0]])
    s = sensitivities(losses)
    assert np.allclose(s, 1.0 / 4 + 1.0 / 4)  # f_i / F_sum + 1/n


def test_coreset_weights_are_unbiased_for_total_loss() -> None:
    rng = np.random.default_rng(1)
    f = rng.random(2000) ** 2
    s = sensitivities(f[None, :])
    totals = []
    for seed in range(300):
        idx, w = sample_coreset(s, 200, np.random.default_rng(seed))
        totals.append(float(np.sum(w * f[idx])))
    assert np.mean(totals) == pytest.approx(f.sum(), rel=0.03)


def test_samplers_are_deterministic_and_sized() -> None:
    a = uniform_indices(1000, 50, np.random.default_rng(3))
    b = uniform_indices(1000, 50, np.random.default_rng(3))
    assert np.array_equal(a, b) and len(a) == 50 and len(np.unique(a)) == 50


def test_hub_stratified_covers_every_hub_when_large_enough() -> None:
    hubs = np.repeat(np.arange(20), [100, 5, 5] + [10] * 17)
    idx = hub_stratified_indices(hubs, 100, np.random.default_rng(0))
    assert len(idx) == 100 and len(np.unique(idx)) == 100
    assert set(np.unique(hubs[idx])) == set(range(20))


def test_relative_gap_and_l2() -> None:
    assert relative_gap(1.1, 1.0) == pytest.approx(0.1)
    assert relative_l2(np.array([3.0, 4.0]), np.array([3.0, 4.0])) == 0.0
    assert relative_l2(np.array([0.0, 0.0]), np.array([3.0, 4.0])) == pytest.approx(1.0)


def test_column_subset_picks_top_scores_and_is_fixed_by_scores_only() -> None:
    scores = np.array([0.1, 5.0, 0.2, 4.0, 0.0])
    assert column_importance_subset(scores, 2).tolist() == [1, 3]


def test_weighted_ridge_recovers_linear_signal_and_respects_weights() -> None:
    from coreset import weighted_ridge

    rng = np.random.default_rng(5)
    x = rng.normal(size=(5000, 4))
    y = x @ np.array([1.0, -2.0, 0.5, 0.0]) + 3.0
    theta = weighted_ridge(x, y, np.ones(5000), lam=1e-6)
    assert np.allclose(theta[:4], [1.0, -2.0, 0.5, 0.0], atol=1e-3) and theta[4] == pytest.approx(3.0, abs=1e-3)
    # rows with weight 0 must not influence the fit
    y_bad = y.copy()
    y_bad[:1000] = 100.0
    w = np.ones(5000)
    w[:1000] = 0.0
    assert np.allclose(weighted_ridge(x, y_bad, w, lam=1e-6), theta, atol=1e-3)
