"""The tier-reach arithmetic, checked against brute force wherever brute force is possible."""
import itertools
from math import comb

import pytest

from evaluation.tiers import (combinations_for_probability, combinations_to_guarantee,
                              draws_reaching, min_shared_for_collision, packing_bound,
                              schonheim, tier_probability, tier_reach)
from methods.common import Pool

LOTO = Pool("main", 45, 6)


def test_draws_reaching_matches_brute_force_on_a_small_pool():
    small = Pool("main", 9, 4)
    fixed = (1, 2, 3, 4)
    for h in range(5):
        brute = sum(1 for d in itertools.combinations(range(1, 10), 4)
                    if len(set(d) & set(fixed)) >= h)
        assert draws_reaching(small, h) == brute


def test_loto_five_hit_tier():
    assert draws_reaching(LOTO, 5) == comb(6, 5) * 39 + 1 == 235
    assert draws_reaching(LOTO, 6) == 1
    assert draws_reaching(LOTO, 0) == comb(45, 6)
    assert tier_probability(LOTO, 5) == pytest.approx(235 / comb(45, 6))
    with pytest.raises(ValueError):
        draws_reaching(LOTO, 7)


def test_two_combinations_sharing_too_little_cannot_both_win():
    """2*min_hits - pick is exactly the threshold, and it is tight."""
    assert min_shared_for_collision(LOTO, 5) == 4
    assert min_shared_for_collision(LOTO, 3) == 0        # low tiers collide freely
    small = Pool("main", 12, 4)
    need = min_shared_for_collision(small, 3)            # 2*3 - 4 = 2
    pool = list(range(1, 13))
    for a, b in itertools.combinations(itertools.combinations(pool, 4), 2):
        if len(set(a) & set(b)) >= need:
            continue
        assert not any(len(set(d) & set(a)) >= 3 and len(set(d) & set(b)) >= 3
                       for d in itertools.combinations(pool, 4))


def test_union_bound_is_exact_for_a_spread_set_and_an_upper_bound_otherwise():
    small = Pool("main", 12, 4)
    p = tier_probability(small, 3)
    pool = list(range(1, 13))

    def exact(tickets):
        return sum(1 for d in itertools.combinations(pool, 4)
                   if any(len(set(d) & set(t)) >= 3 for t in tickets)) / comb(12, 4)

    spread = [(1, 2, 3, 4), (1, 5, 6, 7), (2, 5, 8, 9)]   # pairwise overlap <= 1 < 2
    assert exact(spread) == pytest.approx(len(spread) * p)
    close = [(1, 2, 3, 4), (1, 2, 3, 5)]                  # share 3 >= 2, so they collide
    assert exact(close) < len(close) * p


def test_tier_reach_reports_the_three_figures():
    got = tier_reach(LOTO, 5, 323)
    assert got["expected_winners"] == pytest.approx(323 * 235 / comb(45, 6))
    assert got["p_upper"] == pytest.approx(got["expected_winners"])
    assert got["p_if_independent"] < got["p_upper"]       # independence understates it here
    assert got["disjoint_when_shared_below"] == 4
    assert tier_reach(LOTO, 5, 10**6)["p_upper"] == 1.0


def test_combinations_for_probability():
    n = combinations_for_probability(LOTO, 5, 0.5)
    p = tier_probability(LOTO, 5)
    assert (n - 1) * p < 0.5 <= n * p
    with pytest.raises(ValueError):
        combinations_for_probability(LOTO, 5, 0.0)


def test_covering_and_packing_bounds():
    # Steiner systems hit both bounds exactly: S(5,6,12) has 132 blocks, S(5,8,24) has 759.
    assert schonheim(12, 6, 5) == packing_bound(12, 6, 5) == 132
    assert schonheim(24, 8, 5) == packing_bound(24, 8, 5) == 759
    assert schonheim(45, 6, 5) == 208628
    assert combinations_to_guarantee(LOTO, 5) == 208628
    # 323 combinations on 20 numbers cannot avoid a pair sharing 4
    assert packing_bound(20, 6, 4) == 276 < 323
