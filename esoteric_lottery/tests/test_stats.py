import math

import numpy as np
import pytest
from scipy import stats as sps

from evaluation import stats
from methods.common import Pool

SIN = Pool("main", 22, 11)
EJ = Pool("main", 50, 5)
JOKER = Pool("digits", 10, 6, "digits")


def test_hit_pmf_sve_ili_nista():
    pmf = stats.hit_pmf(SIN)
    assert pmf.sum() == pytest.approx(1)
    assert np.dot(np.arange(12), pmf) == pytest.approx(5.5)
    assert pmf[11] == pytest.approx(1 / math.comb(22, 11)) and pmf[0] == pytest.approx(pmf[11])
    assert pmf[5] == pytest.approx(math.comb(11, 5) ** 2 / math.comb(22, 11))


def test_hit_pmf_other_games():
    assert stats.hit_pmf(EJ)[5] == pytest.approx(1 / math.comb(50, 5))
    assert stats.hit_pmf(JOKER)[6] == pytest.approx(1e-6)


def test_sum_pmf_matches_moments():
    pmf = stats.hit_pmf(SIN)
    d = stats.sum_pmf(pmf, 37)
    k = np.arange(len(d))
    var1 = np.dot((np.arange(12) - 5.5) ** 2, pmf)
    assert np.dot(k, d) == pytest.approx(37 * 5.5)
    assert np.dot((k - 37 * 5.5) ** 2, d) == pytest.approx(37 * var1)


def test_exact_test_symmetry_and_extremes():
    pmf = stats.hit_pmf(SIN)
    centre = stats.exact_test([5, 6] * 50, pmf)
    assert centre["p_two_sided"] == pytest.approx(1.0)
    high = stats.exact_test([7] * 100, pmf)
    assert high["p_two_sided"] < 1e-10 and high["p_upper"] < 1e-10


def test_exact_test_is_calibrated_under_null():
    pmf = stats.hit_pmf(SIN)
    rng = np.random.Generator(np.random.PCG64(1))
    ps = [stats.exact_test(rng.choice(12, size=60, p=pmf), pmf)["p_two_sided"] for _ in range(400)]
    assert 0.02 < np.mean(np.array(ps) < 0.05) < 0.09


def test_ci_and_wilson():
    lo, hi = stats.mean_ci_t([5, 6, 5, 6])
    assert lo < 5.5 < hi
    blo, bhi = stats.bootstrap_mean_ci([5, 6, 5, 6] * 10, 2000, 3)
    assert blo < 5.5 < bhi
    wl, wh = stats.wilson(10, 100)
    assert wl == pytest.approx(0.0552, abs=1e-3) and wh == pytest.approx(0.1744, abs=1e-3)


def test_chi2_pooling():
    pmf = stats.hit_pmf(SIN)
    exp = 270 * pmf
    bins = stats.pooled_bins(exp)
    assert bins[0] == [0, 1, 2, 3] and bins[-1] == [8, 9, 10, 11]
    assert all(exp[b].sum() >= 5 for b in bins)


def test_holm_and_bh():
    p = {"a": 0.01, "b": 0.04, "c": 0.03}
    h = stats.holm(p)
    assert h == pytest.approx({"a": 0.03, "c": 0.06, "b": 0.06})
    bh = stats.benjamini_hochberg(p)
    assert bh == pytest.approx({"a": 0.03, "b": 0.04, "c": 0.04})


def test_random_ticket_null_centred():
    rng = np.random.Generator(np.random.PCG64(5))
    draws = np.zeros((200, 22), dtype=int)
    for row in draws:
        row[rng.choice(22, 11, replace=False)] = 1
    null = stats.random_ticket_null(draws, SIN, 3000, 9)
    assert null.mean() == pytest.approx(5.5, abs=0.01)
    assert null.std() == pytest.approx(math.sqrt(np.dot((np.arange(12) - 5.5) ** 2, stats.hit_pmf(SIN)) / 200), rel=0.1)
    assert stats.mc_p_value(5.5, null, 5.5) == pytest.approx(1.0)


def test_mde():
    sd = math.sqrt(11 * 0.5 * 0.5 * 11 / 21)
    assert stats.minimum_detectable_effect(270, sd, 0.05, 1) == pytest.approx(
        (sps.norm.ppf(0.975) + sps.norm.ppf(0.8)) * sd / math.sqrt(270))


def test_auc_random_scores_near_half():
    rng = np.random.Generator(np.random.PCG64(2))
    labels = (rng.random((300, 22)) < 0.5).astype(int)
    res = stats.auc_with_ci(rng.random((300, 22)), labels, 200, 1)
    assert res["ci95"][0] < 0.5 < res["ci95"][1]
