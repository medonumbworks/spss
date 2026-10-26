"""Building a large block of single combinations under one global constraint.

The goal is to land in a top tier — 6, 5+1 or 5 hits. Two played combinations can both
reach 5 hits on the same draw only if they share at least 2*5 - 6 = 4 numbers
(``evaluation/tiers.py`` proves it), so a set in which no two combinations share more
than 3 numbers has pairwise *disjoint* top-tier events and

    P(at least one top-tier win) = M * 235 / C(45, 6)   exactly,

which is the largest value any M combinations can have. Dropping the bound to 2 shared
numbers costs nothing and buys the lowest tier too: a combination wins something when it
holds 3 of the drawn numbers, so what matters there is how many *distinct* triples the
set covers, and with pairwise overlap at most 2 every combination contributes all
C(6,3) = 20 of its own. Hence one ledger of used triples governs the whole set: a
combination is admissible exactly when none of its 20 triples has been used, which is
the same thing as sharing at most 2 numbers with everything already chosen.

Inside that constraint the choice of numbers is free — every combination has the same
1 : 8.145.060 — so each block picks its numbers by its own criterion, and the criteria
only decide *which* of the equally likely combinations get played. The one criterion
that changes money rather than probability is unpopularity: Loto is pari-mutuel, and
Cook & Clotfelter (1993), Farrell et al. (2000) and Baker & McHale (2009) all find that
numbers above 31 are systematically underplayed, so a win on them is shared with fewer
people.
"""
from __future__ import annotations

from itertools import combinations
from typing import Callable, Iterable, Sequence

import numpy as np

Ticket = tuple[int, ...]
BANDS = ((1, 9), (10, 18), (19, 27), (28, 36), (37, 45))


def band_of(n: int) -> int:
    for i, (lo, hi) in enumerate(BANDS):
        if lo <= n <= hi:
            return i
    raise ValueError(n)


class TripleLedger:
    """Every triple may be used once, which is exactly "share at most 2 numbers"."""

    def __init__(self, existing: Iterable[Ticket] = ()):
        self.used: set[Ticket] = set()
        for t in existing:
            self.add(t)

    def fits(self, ticket: Sequence[int]) -> bool:
        return not any(c in self.used for c in combinations(sorted(ticket), 3))

    def add(self, ticket: Sequence[int]) -> None:
        self.used.update(combinations(sorted(ticket), 3))


# ------------------------------------------------------------------ shape predicates

def empirical_ok(t: Sequence[int]) -> bool:
    """Inside the envelope the 882 drawn results of 2018-2026 actually occupy:
    sum in the 10-90 percentile, 2-4 odd numbers, 3-5 numbers at or below 31."""
    s = sorted(t)
    return (98 <= sum(s) <= 176
            and 2 <= sum(x % 2 for x in s) <= 4
            and 3 <= sum(x <= 31 for x in s) <= 5)


def unpopular_ok(t: Sequence[int]) -> bool:
    """The conscious-selection filter: push the combination where the crowd is not."""
    s = sorted(t)
    if sum(x > 31 for x in s) < 4:
        return False
    if sum(x <= 12 for x in s) > 1:          # day-of-month and month numbers
        return False
    if 7 in s:                                # the single most played number
        return False
    if sum(s) < 165:                          # crowd averages well below 138
        return False
    if max(sum(1 for x in s if (x - 1) // 10 == d) for d in range(5)) > 3:
        return False
    d = [s[i + 1] - s[i] for i in range(5)]
    return not any(d[i] == d[i + 1] for i in range(4))   # no arithmetic run of three


def not_a_pattern(t: Sequence[int]) -> bool:
    """Shapes players draw on the slip, which therefore get shared: three in a row,
    one-sided slips, repeated last digits, constant gaps, all-odd or all-even."""
    s = sorted(t)
    if any(s[i] + 1 == s[i + 1] == s[i + 2] - 1 for i in range(4)):
        return False
    if max(sum(1 for x in s if x % 10 == r) for r in range(10)) > 2:
        return False
    if max(sum(1 for x in s if (x - 1) // 10 == d) for d in range(5)) > 3:
        return False
    if sum(x % 2 for x in s) in (0, 6):
        return False
    d = [s[i + 1] - s[i] for i in range(5)]
    return not any(d[i] == d[i + 1] == d[i + 2] for i in range(3))


def spread_ok(t: Sequence[int], min_bands: int = 4, max_per_band: int = 2) -> bool:
    bands = [band_of(x) for x in t]
    return (len(set(bands)) >= min_bands
            and max(bands.count(b) for b in set(bands)) <= max_per_band)


# ----------------------------------------------------------------------- the builders
#
# Rejection sampling stalls once the ledger is full: most random 6-subsets reuse a
# triple, and the last few per cent of a block are never found. So a combination is
# *searched* instead — numbers are added one at a time and a branch is abandoned the
# moment the number just added closes a triple that is already spent. The order in which
# numbers are offered is a weighted shuffle, which is where each block's own criterion
# enters: the search returns the first admissible combination it meets, so numbers the
# block favours are tried first.

def weighted_order(rng: np.random.Generator, weights: Sequence[float]) -> list[int]:
    """A shuffle of 1..45 in which heavy numbers come early (Gumbel top-k trick)."""
    w = np.asarray(weights, dtype=float).copy()
    w[w <= 0] = 1e-12
    keys = np.log(w) - np.log(-np.log(rng.random(w.size)))
    return [int(i) + 1 for i in np.argsort(-keys)]


def search_ticket(ledger: TripleLedger, order: Sequence[int], accept: Callable[[Ticket], bool],
                  banned: set[Ticket], max_nodes: int = 400_000) -> Ticket | None:
    """First combination, in the given order of numbers, whose 20 triples are all free."""
    used = ledger.used
    nodes = 0

    def rec(start: int, pick: list[int]) -> Ticket | None:
        nonlocal nodes
        if len(pick) == 6:
            t = tuple(sorted(pick))
            return t if t not in banned and accept(t) else None
        for i in range(start, len(order)):
            n = order[i]
            nodes += 1
            if nodes > max_nodes:
                return None
            if any(tuple(sorted((a, b, n))) in used for a, b in combinations(pick, 2)):
                continue
            hit = rec(i + 1, pick + [n])
            if hit is not None:
                return hit
        return None

    return rec(0, [])


def sample_block(count: int, order_fn: Callable[[], Sequence[int]],
                 accept: Callable[[Ticket], bool], ledger: TripleLedger, banned: set[Ticket],
                 attempts_per_ticket: int = 200,
                 on_accept: Callable[[Ticket], None] | None = None) -> list[Ticket]:
    """Search ``count`` combinations, re-rolling the order of numbers on each attempt."""
    out: list[Ticket] = []
    while len(out) < count:
        for _ in range(attempts_per_ticket):
            t = search_ticket(ledger, order_fn(), accept, banned)
            if t is None:
                continue
            ledger.add(t)
            banned.add(t)
            out.append(t)
            if on_accept:
                on_accept(t)
            break
        else:
            raise RuntimeError(f"stuck after {len(out)}/{count}: no combination fits")
    return out


def weighted_block(count: int, rng: np.random.Generator, weights: Sequence[float],
                   accept: Callable[[Ticket], bool], ledger: TripleLedger,
                   banned: set[Ticket]) -> list[Ticket]:
    return sample_block(count, lambda: weighted_order(rng, weights), accept, ledger, banned)


def coverage_block(count: int, rng: np.random.Generator, ledger: TripleLedger,
                   banned: set[Ticket], sharpness: float = 3.0) -> list[Ticket]:
    """Spread over 1..45 as evenly as the arithmetic allows: with 6*count slots over 45
    numbers every number wants about count*6/45 appearances, so a number is offered
    earlier the less it has been used. No combination may lean on one band of nine."""
    used = np.zeros(46, dtype=float)

    def order() -> list[int]:
        return weighted_order(rng, (1.0 + used[1:].max() - used[1:]) ** sharpness)

    return sample_block(count, order, lambda t: spread_ok(t) and not_a_pattern(t),
                        ledger, banned,
                        on_accept=lambda t: used.__setitem__(list(t), used[list(t)] + 1))


def frequency_table(tickets: Sequence[Ticket]) -> dict[int, int]:
    out = {n: 0 for n in range(1, 46)}
    for t in tickets:
        for n in t:
            out[n] += 1
    return out
