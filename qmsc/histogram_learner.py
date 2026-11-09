"""
The smoothed histogram learner of the paper (Sec. V) on a finite chain, with exact population errors.

The learner never sees the target distribution p, only N i.i.d. samples x^(1..N) ~ p.  On a chain small
enough for exact diagonalization, p is used afterwards to evaluate the population error without sampling
noise:

    E(R, N) = D_KL(p || q_hat_{R,N}),      B(R) = D_KL(p || q*_R),
    L(R, N) = E(R, N) - B(R)  >=  0        (q*_R is the best-in-class radius-R distribution).

Model class: the radius-R local autoregressive class, one Bernoulli parameter per site i and context
c in {0,1}^{min(R, i-1)}.

Estimator: the add-alpha (Krichevsky-Trofimov for alpha = 1/2) conditional

    q_hat(x_i = 1 | c) = (n_{i,c,1} + alpha) / (n_{i,c} + 2 alpha),

optionally clipped to [eps, 1-eps].  The paper's estimation cost is derived for the unclipped estimator.
For alpha = 1/2 the clip can bind only in a cell visited more than 1/(2 eps) - 1 times with every outcome
equal, and ``fit_tables`` records how often that happens (``clip_cells``), so the clipped and unclipped
predictors can be compared directly.

Sampling design: nested N (one draw of size N_max per seed, prefixes for the smaller N) and the same
samples reused for every R.

Bit conventions match :mod:`qmsc.tfim_chain`: flat index b has x_i = (b >> (n-i)) & 1 (site 1 = MSB).
"""
from __future__ import annotations

import numpy as np

__all__ = ["sample_flat", "cell_counts", "fit_tables", "log_model_grid", "population_kl", "learn_and_evaluate"]


# ----------------------------------------------------------------------------------------------
# Sampling
# ----------------------------------------------------------------------------------------------
def sample_flat(p: np.ndarray, N: int, rng: np.random.Generator) -> np.ndarray:
    """N i.i.d. flat basis indices ~ p (array of shape (2,)*n), by inverse CDF (exact)."""
    cum = np.cumsum(p.ravel())
    cum[-1] = 1.0
    return np.searchsorted(cum, rng.random(N), side="right").astype(np.int64)


# ----------------------------------------------------------------------------------------------
# Context counting (vectorized bit operations)
# ----------------------------------------------------------------------------------------------
def cell_counts(samples: np.ndarray, n: int, R: int) -> list[np.ndarray]:
    """Per-site cell counts.  counts[i-1] has shape (2^w, 2), w = min(R, i-1):
    counts[i-1][c, x] = #{k : context(x^(k), i, R) = c and x^(k)_i = x}.

    Context value c packs sites lo..i-1 (lo = max(1, i-R)) with site lo as the most significant bit,
    matching the axis order of the (2,)*n arrays: c = (b >> (n-i+1)) & (2^w - 1).
    """
    out = []
    for i in range(1, n + 1):
        w = min(R, i - 1)
        x_i = (samples >> (n - i)) & 1
        c = (samples >> (n - i + 1)) & ((1 << w) - 1)
        key = (c << 1) | x_i
        out.append(np.bincount(key, minlength=1 << (w + 1)).reshape(-1, 2).astype(np.int64))
    return out


# ----------------------------------------------------------------------------------------------
# Estimator
# ----------------------------------------------------------------------------------------------
def fit_tables(counts: list[np.ndarray], alpha: float = 0.5, eps: float = 1e-4):
    """Add-alpha conditionals, clipped to [eps, 1-eps].

    Returns (tables, stats): tables[i-1][c] = q_hat(x_i = 1 | c); stats records the number of nominal
    and visited cells and how often the clip binds.  alpha = 0 is the maximum-likelihood estimator
    (0/0 -> 1/2 for unvisited cells), then clipped.
    """
    tables, n_cells = [], 0
    visited = clipped_cells = 0
    clipped_weight = 0.0
    N = int(counts[0].sum())
    for cnt in counts:
        n_c = cnt.sum(axis=1)
        if alpha > 0:
            q1 = (cnt[:, 1] + alpha) / (n_c + 2 * alpha)
        else:
            with np.errstate(invalid="ignore"):
                q1 = np.where(n_c > 0, cnt[:, 1] / np.maximum(n_c, 1), 0.5)
        binds = (q1 < eps) | (q1 > 1 - eps)
        clipped_cells += int(binds.sum())
        clipped_weight += float(n_c[binds].sum()) / max(N, 1)
        tables.append(np.clip(q1, eps, 1 - eps))
        n_cells += len(q1)
        visited += int((n_c > 0).sum())
    stats = {"n_cells_nominal": n_cells, "n_cells_visited": visited,
             "clip_cells": clipped_cells, "clip_weight": clipped_weight}
    return tables, stats


# ----------------------------------------------------------------------------------------------
# Exact population error (the exact target distribution enters only here)
# ----------------------------------------------------------------------------------------------
def log_model_grid(tables: list[np.ndarray], n: int, R: int) -> np.ndarray:
    """log q_hat(x) = sum_i log q_hat(x_i | c_{i,R}) on the full (2,)*n grid (broadcast per site)."""
    log_q = np.zeros((2,) * n)
    for i in range(1, n + 1):
        q1 = tables[i - 1]                                   # shape (2^w,), w = min(R, i-1)
        tab = np.stack([np.log(1.0 - q1), np.log(q1)], axis=-1)   # (2^w, 2)
        shape = [1] * n
        lo = max(1, i - R)
        for axis in range(lo - 1, i):
            shape[axis] = 2
        log_q = log_q + tab.reshape(shape)
    return log_q


def population_kl(p: np.ndarray, log_q: np.ndarray) -> float:
    """D_KL(p || q_hat) in nats, exact sum over the whole configuration space (0 log 0 = 0)."""
    mask = p > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(mask, p * (np.log(np.where(mask, p, 1.0)) - log_q), 0.0)
    return float(terms.sum())


# ----------------------------------------------------------------------------------------------
# One full evaluation
# ----------------------------------------------------------------------------------------------
def learn_and_evaluate(p: np.ndarray, samples: np.ndarray, R: int, B_ref: float,
                       alpha: float = 0.5, eps: float = 1e-4) -> dict:
    """Fit the learner on ``samples`` at radius R and evaluate it exactly against p.

    Returns E = D_KL(p || q_hat), L = E - B_ref, the occupancy and clip-binding statistics and the mean
    empirical log loss.  ``B_ref`` is the representation bias B(R) of this (n, g, R), from
    :func:`qmsc.tfim_chain.bias_cmi`.
    """
    n = p.ndim
    N = len(samples)
    counts = cell_counts(samples, n, R)
    tables, stats = fit_tables(counts, alpha=alpha, eps=eps)
    log_q = log_model_grid(tables, n, R)
    E = population_kl(p, log_q)
    emp = float(-log_q.ravel()[samples].mean())
    out = {
        "N": N, "R": R, "alpha": alpha, "eps": eps,
        "E_kl": E, "L": E - B_ref, "B_ref": B_ref,
        "empirical_log_loss": emp,
        **stats,
        "frac_cells_visited": stats["n_cells_visited"] / stats["n_cells_nominal"],
    }
    return out
