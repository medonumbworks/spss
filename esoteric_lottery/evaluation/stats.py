"""Statistics for the backtest: exact null distributions, confidence intervals,
goodness of fit, the uniform random baseline, AUC and multiple-testing corrections.

Null hypothesis for every strategy: the ticket is fixed before the draw and the draw is
a uniform random k-subset independent of it. Then the hits of each draw are i.i.d.
hypergeometric (k-of-N pools) or binomial (Joker positions) *whatever* the ticket, so
the exact distribution of the total over n draws is the n-fold convolution.
"""
from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

from methods.common import Pool


def hit_pmf(pool: Pool, drawn: int | None = None) -> np.ndarray:
    if pool.kind == "digits":
        return stats.binom(pool.pick, 1.0 / pool.size).pmf(np.arange(pool.pick + 1))
    drawn = drawn or pool.pick
    return stats.hypergeom(pool.size, drawn, pool.pick).pmf(np.arange(pool.pick + 1))


def sum_pmf(pmf: np.ndarray, n: int) -> np.ndarray:
    """PMF of the sum of n i.i.d. draws from ``pmf`` (exponentiation by squaring)."""
    result, base = np.array([1.0]), np.asarray(pmf, dtype=float)
    while n:
        if n & 1:
            result = np.convolve(result, base)
        base = np.convolve(base, base)
        n >>= 1
    return result / result.sum()


def exact_test(hits: Sequence[int], pmf: np.ndarray) -> dict:
    hits = np.asarray(hits)
    total = int(hits.sum())
    dist = sum_pmf(pmf, len(hits))
    lower, upper = float(dist[: total + 1].sum()), float(dist[total:].sum())
    return {"total_hits": total, "p_lower": min(1.0, lower), "p_upper": min(1.0, upper),
            "p_two_sided": min(1.0, 2.0 * min(lower, upper))}


def mean_ci_t(x: Sequence[float], conf: float = 0.95) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    se = x.std(ddof=1) / math.sqrt(len(x))
    h = stats.t.ppf(0.5 + conf / 2, len(x) - 1) * se
    return float(x.mean() - h), float(x.mean() + h)


def bootstrap_mean_ci(x: Sequence[float], reps: int, seed: int, conf: float = 0.95) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    rng = np.random.Generator(np.random.PCG64(seed))
    means = x[rng.integers(0, len(x), size=(reps, len(x)))].mean(axis=1)
    a = (1 - conf) / 2
    return float(np.quantile(means, a)), float(np.quantile(means, 1 - a))


def wilson(k: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    z = stats.norm.ppf(0.5 + conf / 2)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return float(centre - half), float(centre + half)


def pooled_bins(expected: np.ndarray, min_expected: float = 5.0) -> list[list[int]]:
    bins, cur = [], []
    for k in range(len(expected)):
        cur.append(k)
        if expected[cur].sum() >= min_expected:
            bins.append(cur)
            cur = []
    if cur:
        if bins:
            bins[-1] += cur
        else:
            bins.append(cur)
    return bins


def chi2_gof(hits: Sequence[int], pmf: np.ndarray, min_expected: float = 5.0) -> dict:
    hits = np.asarray(hits)
    n = len(hits)
    observed = np.bincount(hits, minlength=len(pmf))[: len(pmf)].astype(float)
    expected = n * pmf
    bins = pooled_bins(expected, min_expected)
    o = np.array([observed[b].sum() for b in bins])
    e = np.array([expected[b].sum() for b in bins])
    if len(bins) < 2:
        return {"bins": bins, "statistic": None, "df": 0, "p_value": None}
    res = stats.chisquare(o, e * o.sum() / e.sum())
    return {"bins": [f"{b[0]}-{b[-1]}" if len(b) > 1 else str(b[0]) for b in bins], "observed": o.tolist(),
            "expected": e.tolist(), "statistic": float(res.statistic), "df": len(bins) - 1,
            "p_value": float(res.pvalue)}


def holm(pvals: Mapping[str, float]) -> dict[str, float]:
    keys = sorted(pvals, key=lambda k: pvals[k])
    m, running, out = len(keys), 0.0, {}
    for i, k in enumerate(keys):
        running = max(running, min(1.0, (m - i) * pvals[k]))
        out[k] = running
    return out


def benjamini_hochberg(pvals: Mapping[str, float]) -> dict[str, float]:
    keys = list(pvals)
    adj = stats.false_discovery_control(np.array([pvals[k] for k in keys]), method="bh")
    return {k: float(a) for k, a in zip(keys, adj)}


def minimum_detectable_effect(n: int, sd: float, alpha: float, m_tests: int, power: float = 0.8) -> float:
    """Smallest shift of the mean hits a two-sided Bonferroni-level z-test detects with ``power``."""
    z_a = stats.norm.ppf(1 - alpha / (2 * m_tests))
    z_b = stats.norm.ppf(power)
    return float((z_a + z_b) * sd / math.sqrt(n))


def random_ticket_null(draw_matrix: np.ndarray, pool: Pool, reps: int, seed: int) -> np.ndarray:
    """Mean hits of ``reps`` uniform random strategies (a fresh random ticket per draw)
    against the actual draws in ``draw_matrix`` (draws x N, 0/1)."""
    rng = np.random.Generator(np.random.PCG64(seed))
    totals = np.zeros(reps)
    for row in draw_matrix:
        keys = rng.random((reps, pool.size))
        tickets = np.argpartition(keys, pool.pick - 1, axis=1)[:, : pool.pick]
        totals += row[tickets].sum(axis=1)
    return totals / len(draw_matrix)


def mc_p_value(observed_mean: float, null_means: np.ndarray, expected: float) -> float:
    dev = abs(observed_mean - expected)
    return float((1 + np.sum(np.abs(null_means - expected) >= dev - 1e-12)) / (len(null_means) + 1))


def auc_with_ci(scores: np.ndarray, labels: np.ndarray, reps: int, seed: int) -> dict:
    """Pooled ROC AUC of per-number scores (draws x N) against appearance, with a
    bootstrap over draws."""
    point = float(roc_auc_score(labels.ravel(), scores.ravel()))
    rng = np.random.Generator(np.random.PCG64(seed))
    boots = []
    n = len(scores)
    for _ in range(reps):
        idx = rng.integers(0, n, size=n)
        boots.append(roc_auc_score(labels[idx].ravel(), scores[idx].ravel()))
    return {"auc": point, "ci95": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))]}


def summarize_hits(hits: Sequence[int], pmf: np.ndarray, reps: int, seed: int, tail: tuple[int, int] | None) -> dict:
    hits = np.asarray(hits, dtype=int)
    n = len(hits)
    k_max = len(pmf) - 1
    mean0 = float(np.dot(np.arange(len(pmf)), pmf))
    counts = np.bincount(hits, minlength=len(pmf))[: len(pmf)]
    out = {
        "n_draws": n, "mean_hits": float(hits.mean()), "expected_mean_hits": mean0,
        "difference": float(hits.mean() - mean0), "sd_hits": float(hits.std(ddof=1)),
        "ci95_t": mean_ci_t(hits), "ci95_bootstrap": bootstrap_mean_ci(hits, reps, seed),
        "hit_distribution": {str(k): int(counts[k]) for k in range(len(pmf))},
        "expected_distribution": {str(k): float(n * pmf[k]) for k in range(len(pmf))},
        "exact_test": exact_test(hits, pmf), "chi2_gof": chi2_gof(hits, pmf),
        "max_hits": int(hits.max()), "min_hits": int(hits.min()),
        "top_tier_events": int(np.sum(hits == k_max)),
        "top_tier_expected": float(n * pmf[k_max]),
    }
    if tail is not None:
        lo, hi = tail
        k = int(np.sum((hits <= lo) | (hits >= hi)))
        p0 = float(pmf[: lo + 1].sum() + pmf[hi:].sum())
        out["extreme_rate"] = {"definition": f"hits <= {lo} or hits >= {hi}", "count": k, "rate": k / n,
                               "expected_rate": p0, "ci95_wilson": wilson(k, n),
                               "binomial_p_two_sided": float(stats.binomtest(k, n, p0).pvalue)}
        out["zero_hit_events"] = int(np.sum(hits == 0))
        out["zero_hit_expected"] = float(n * pmf[0])
    return out
