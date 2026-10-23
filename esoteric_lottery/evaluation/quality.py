"""Descriptive checks of the draw history itself (are the draws consistent with a
uniform, independent k-of-N lottery?). They are reported, not used for any decision.

* frequency uniformity: chi-square-type statistic sum((O - E)^2 / E) over the numbers;
  because each draw holds exactly k numbers the textbook chi-square df do not apply, so
  the p-value comes from simulating uniform draws.
* consecutive-draw overlap: under independence the overlap of two draws is again
  hypergeometric, and successive overlaps are independent, so the exact test applies.
* longest run: the longest streak of consecutive draws containing a number, maximised
  over all numbers, against simulation.
"""
from __future__ import annotations

import numpy as np

from methods.common import Pool

from .stats import exact_test, hit_pmf


def _longest_runs(matrix: np.ndarray) -> np.ndarray:
    cur = np.zeros(matrix.shape[1], dtype=int)
    best = np.zeros(matrix.shape[1], dtype=int)
    for row in matrix:
        cur = (cur + 1) * row
        best = np.maximum(best, cur)
    return best


def data_quality(matrix: np.ndarray, pool: Pool, reps: int, seed: int) -> dict:
    n = len(matrix)
    counts = matrix.sum(axis=0)
    expected = n * pool.pick / pool.size
    stat = float(((counts - expected) ** 2 / expected).sum())
    runs = _longest_runs(matrix)

    rng = np.random.Generator(np.random.PCG64(seed))
    cur =np.zeros((reps, pool.size), dtype=int)
    best = np.zeros((reps, pool.size), dtype=int)
    sim_counts = np.zeros((reps, pool.size))
    rows = np.arange(reps)[:, None]
    for _ in range(n):
        picks = np.argpartition(rng.random((reps, pool.size)), pool.pick - 1, axis=1)[:, : pool.pick]
        x = np.zeros((reps, pool.size), dtype=int)
        x[rows, picks] = 1
        sim_counts += x
        cur = (cur + 1) * x
        best = np.maximum(best, cur)
    sim_stat = ((sim_counts - expected) ** 2 / expected).sum(axis=1)
    sim_maxrun = best.max(axis=1)

    overlaps = (matrix[1:] & matrix[:-1]).sum(axis=1)
    pmf = hit_pmf(pool)
    return {
        "n_draws": n,
        "frequency": {"counts": {str(i + 1): int(c) for i, c in enumerate(counts)}, "expected_each": expected,
                      "statistic": stat, "p_value_simulated": float((1 + np.sum(sim_stat >= stat)) / (reps + 1)),
                      "simulations": reps},
        "consecutive_overlap": {"mean": float(overlaps.mean()), "expected": float(np.dot(np.arange(len(pmf)), pmf)),
                                "exact_test": exact_test(overlaps, pmf)},
        "longest_run": {"per_number": {str(i + 1): int(r) for i, r in enumerate(runs)},
                        "max_observed": int(runs.max()),
                        "number_with_max": [int(i + 1) for i in np.flatnonzero(runs == runs.max())],
                        "p_value_simulated_max_over_numbers": float((1 + np.sum(sim_maxrun >= runs.max())) / (reps + 1)),
                        "simulated_max_median": float(np.median(sim_maxrun))},
    }
