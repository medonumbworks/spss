"""How much has to be played to reach a given prize tier — and why the usual
"1 - (1-p)^M" answer is the wrong one.

A played combination of ``pick`` numbers reaches the tier "at least ``h`` of the drawn
numbers" exactly when it holds h or more of them. For a FIXED combination T the number of
draws that do this is a sum of hypergeometric counts and does not depend on which numbers
T holds, so

    p(h) = draws_reaching(pool, h) / C(size, pick)

is the same for every combination. The interesting part is what happens when M
combinations are played together.

Two combinations T1, T2 can only BOTH reach h hits on the same draw if they overlap a lot:

    |T1 n D| + |T2 n D| = |D n (T1 u T2)| + |D n T1 n T2| <= pick + |T1 n T2|,

so both reaching h forces |T1 n T2| >= 2h - pick. For Loto 6/45 and the 5-hit tier that is
|T1 n T2| >= 4: any two played combinations sharing 3 or fewer numbers can never win that
tier on the same draw. Their events are *disjoint*, not independent, and then

    P(at least one reaches h) = M * p(h)   exactly.

That is why the independence model understates the answer: it assumes the events can
coincide, when for a well-spread set they almost never can. The exact probability is
M * p(h) minus the overlap of the few pairs that do share 2h - pick numbers or more, so
M * p(h) is an upper bound that is tight for a spread set (``tests/test_tiers.py`` checks
both halves by enumerating the winning draws).

Guaranteeing a tier is a covering problem: the played combinations must contain every
h-subset of the pool, and the Schonheim bound says how many that takes at the very least.
"""
from __future__ import annotations

from math import ceil, comb

from methods.common import Pool


def draws_reaching(pool: Pool, min_hits: int) -> int:
    """How many draws make a fixed combination hold at least ``min_hits`` drawn numbers."""
    if not 0 <= min_hits <= pool.pick:
        raise ValueError(f"min_hits must lie in 0..{pool.pick}")
    return sum(comb(pool.pick, h) * comb(pool.size - pool.pick, pool.pick - h)
               for h in range(min_hits, pool.pick + 1))


def tier_probability(pool: Pool, min_hits: int) -> float:
    """Probability that one played combination reaches the tier. Same for every combination."""
    return draws_reaching(pool, min_hits) / comb(pool.size, pool.pick)


def min_shared_for_collision(pool: Pool, min_hits: int) -> int:
    """Two combinations can only both reach the tier if they share this many numbers."""
    return max(0, 2 * min_hits - pool.pick)


def tier_reach(pool: Pool, min_hits: int, combinations: int) -> dict:
    """Playing ``combinations`` distinct combinations: expectation (exact) and P (bounded).

    ``expected_winners`` is exact and depends on nothing but the count. ``p_upper`` is the
    union bound, which is an equality when no two of the combinations share
    ``min_shared_for_collision`` numbers or more.
    """
    p = tier_probability(pool, min_hits)
    return {"p_per_combination": p,
            "combinations": combinations,
            "expected_winners": combinations * p,
            "p_upper": min(1.0, combinations * p),
            "p_if_independent": 1.0 - (1.0 - p) ** combinations,
            "disjoint_when_shared_below": min_shared_for_collision(pool, min_hits)}


def combinations_for_probability(pool: Pool, min_hits: int, target: float) -> int:
    """Fewest combinations whose union bound reaches ``target``. A lower bound on reality."""
    if not 0 < target <= 1:
        raise ValueError("target must lie in (0, 1]")
    return ceil(target / tier_probability(pool, min_hits))


def schonheim(n: int, k: int, t: int) -> int:
    """Schonheim lower bound on the covering number C(n, k, t): the fewest k-subsets that
    contain every t-subset. Playing fewer than this cannot guarantee the tier."""
    if t == 0:
        return 1
    return -(-n * schonheim(n - 1, k - 1, t - 1) // k)


def packing_bound(n: int, k: int, t: int) -> int:
    """Johnson upper bound on the packing number D(n, k, t): the most k-subsets in which no
    t-subset repeats. Above it, some pair of combinations must share t numbers."""
    if t == 0:
        return 1
    return n * packing_bound(n - 1, k - 1, t - 1) // k


def combinations_to_guarantee(pool: Pool, min_hits: int) -> int:
    """Fewest combinations that could possibly guarantee the tier on every draw."""
    return schonheim(pool.size, pool.pick, min_hits)
