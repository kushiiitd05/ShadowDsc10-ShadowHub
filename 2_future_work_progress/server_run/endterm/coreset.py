"""Pure sampling / coreset helpers for the end-term experiments (no I/O, inputs never mutated)."""
from __future__ import annotations

import numpy as np


def sensitivities(losses: np.ndarray) -> np.ndarray:
    """s_i = max_t [ f_ti / F_t + 1/n ] with F_t = sum_i f_ti (sum, not mean). `losses` has shape (T, n)."""
    losses = np.asarray(losses, dtype=float)
    n = losses.shape[1]
    totals = losses.sum(axis=1, keepdims=True)
    return (losses / totals + 1.0 / n).max(axis=0)


def sample_coreset(s: np.ndarray, m: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Draw m indices with replacement, p_i = s_i / sum s; weight 1 / (m p_i) so that sum(w f) is unbiased for sum f."""
    p = s / s.sum()
    idx = rng.choice(len(s), size=m, replace=True, p=p)
    return idx, 1.0 / (m * p[idx])


def uniform_indices(n: int, m: int, rng: np.random.Generator) -> np.ndarray:
    return np.sort(rng.choice(n, size=m, replace=False))


def hub_stratified_indices(hubs: np.ndarray, m: int, rng: np.random.Generator) -> np.ndarray:
    """Proportional allocation by hub with at least one row per hub when m allows; leftover rows go to random hubs."""
    labels, inverse, counts = np.unique(hubs, return_inverse=True, return_counts=True)
    quota = np.minimum(counts, np.maximum(1, np.floor(m * counts / counts.sum()).astype(int)))
    if quota.sum() > m:  # more hubs than rows: keep a random subset of hubs
        keep = rng.choice(len(labels), size=m, replace=False)
        quota = np.zeros_like(quota)
        quota[keep] = 1
    spare = counts - quota
    while quota.sum() < m and spare.sum() > 0:
        pick = rng.choice(len(labels), p=spare / spare.sum())
        quota[pick] += 1
        spare[pick] -= 1
    while quota.sum() > m:  # floors of 1 can overshoot
        pick = rng.choice(np.flatnonzero(quota > 1))
        quota[pick] -= 1
    chosen = [rng.choice(np.flatnonzero(inverse == g), size=int(q), replace=False) for g, q in enumerate(quota) if q > 0]
    return np.sort(np.concatenate(chosen))


def relative_gap(value: float, optimum: float) -> float:
    return float(value / optimum - 1.0)


def relative_l2(candidate: np.ndarray, reference: np.ndarray) -> float:
    return float(np.linalg.norm(candidate - reference) / np.linalg.norm(reference))


def column_importance_subset(scores: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k highest-scoring columns (scores come from the training set only), sorted."""
    return np.sort(np.argsort(-np.asarray(scores), kind="stable")[:k])


def weighted_ridge(x: np.ndarray, y: np.ndarray, w: np.ndarray, lam: float, chunk: int = 200_000) -> np.ndarray:
    """Minimise sum_i w_i (y_i - x_i.b - c)^2 / sum_i w_i + lam * |b|^2 (intercept c unpenalised).

    Returns theta = [b..., c]. Accumulates the normal equations in float64 over row chunks.
    """
    d = x.shape[1]
    gram = np.zeros((d + 1, d + 1))
    rhs = np.zeros(d + 1)
    for start in range(0, len(y), chunk):
        block = np.hstack([x[start:start + chunk].astype(np.float64), np.ones((min(chunk, len(y) - start), 1))])
        wb = w[start:start + chunk]
        gram += block.T @ (block * wb[:, None])
        rhs += block.T @ (wb * y[start:start + chunk])
    penalty = np.eye(d + 1) * (lam * w.sum())
    penalty[d, d] = 0.0
    return np.linalg.solve(gram + penalty + 1e-9 * np.eye(d + 1), rhs)
