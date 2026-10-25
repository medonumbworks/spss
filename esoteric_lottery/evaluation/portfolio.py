"""Choosing a *set* of tickets: the one thing about a lottery entry that is genuinely
optimizable.

The per-ticket distribution of hits cannot be improved. Under the null (the draw is a
uniform random k-subset, independent of the ticket) every ticket of 11 numbers has the
same hypergeometric hit distribution, so every ticket has the same 8.61 % chance of
landing in a paying tier of Sve ili Ništa and the same 1-in-705 432 chance of the
jackpot. The backtest in ``results/`` finds nothing that beats that, and nothing here
claims to.

What a *set* of m tickets can change is the joint distribution — how the m hit counts
move together:

* ``E[sum of hits]`` is invariant: m x 5.5, whatever the tickets. Nothing buys more
  expected hits.
* ``P(at least one ticket wins something)`` is NOT invariant. It ranges from 8.6 %
  (m identical tickets) to about 30.8 % (m tickets whose pairwise overlaps sit near the
  middle), because only then do their outcomes decorrelate.
* Two *complementary* tickets (overlap 0) are the trap: hits(A) + hits(Ā) = 11, so
  A lands in a paying tier exactly when Ā does. Their outcomes are perfectly correlated
  for the purpose of winning anything, which is why a "maximally spread" set built from
  complementary pairs scores *worse* (16.7 %) than a merely well-spread one.

So the optimum avoids both extremes: no pair nearly identical, no pair nearly
complementary. Crucially this depends only on how the tickets overlap *each other*, never
on which numbers are drawn: the objective is invariant under relabelling the 22 numbers
(``tests/test_portfolio.py`` asserts this). It therefore carries no prediction and cannot
leak anything about the draw — it is combinatorics, not forecasting.

The tickets themselves still come from the esoteric methods; this module only chooses
which m of the candidate tickets to play together.

Prize structure used: the paying hit counts of Sve ili Ništa are symmetric — 11 or 0
(jackpot), 10 or 1, 9 or 2, 8 or 3; 4 to 7 pays nothing. The *tiers* are consistent
across public sources, the *amounts* are not, so only tier membership is used and no
money figure is computed anywhere.
"""
from __future__ import annotations

import itertools
from fractions import Fraction
from math import comb
from typing import Iterable, Sequence


from methods.common import Pool, tiebreak_values

Ticket = tuple[int, ...]


# --------------------------------------------------------------- exact joint analysis

def _atoms(tickets: Sequence[Ticket], size: int) -> list[tuple[tuple[int, ...], int]]:
    """Partition 1..size by which tickets contain each number: (membership pattern, count)."""
    counts: dict[tuple[int, ...], int] = {}
    for n in range(1, size + 1):
        pattern = tuple(1 if n in set(t) else 0 for t in tickets)
        counts[pattern] = counts.get(pattern, 0) + 1
    return sorted(counts.items())


def joint_hit_distribution(tickets: Sequence[Ticket], pool: Pool) -> dict[tuple[int, ...], Fraction]:
    """Exact joint distribution of the hit counts of every ticket, as exact fractions.

    The numbers are grouped into atoms by which tickets contain them; the draw assigns a
    count to each atom (multivariate hypergeometric), and every number of an atom counts
    for exactly the tickets in its pattern. Dynamic programming over the atoms.
    """
    atoms = _atoms(tickets, pool.size)
    # state: (numbers drawn so far, hits per ticket) -> number of ways
    states: dict[tuple[int, tuple[int, ...]], int] = {(0, tuple([0] * len(tickets))): 1}
    for pattern, atom_size in atoms:
        nxt: dict[tuple[int, tuple[int, ...]], int] = {}
        for (drawn, hits), ways in states.items():
            for take in range(0, min(atom_size, pool.pick - drawn) + 1):
                new_hits = tuple(h + take * p for h, p in zip(hits, pattern))
                key = (drawn + take, new_hits)
                nxt[key] = nxt.get(key, 0) + ways * comb(atom_size, take)
        states = nxt
    total = comb(pool.size, pool.pick)
    out: dict[tuple[int, ...], Fraction] = {}
    for (drawn, hits), ways in states.items():
        if drawn == pool.pick:
            out[hits] = out.get(hits, Fraction(0)) + Fraction(ways, total)
    assert sum(out.values()) == 1, "joint distribution must sum to 1"
    return out


def evaluate_set(tickets: Sequence[Ticket], pool: Pool, paying: Iterable[int],
                 top_hits: Iterable[int] | None = None) -> dict:
    """Exact figures for a set of tickets. No simulation, no approximation.

    ``tickets`` need not hold ``pool.pick`` numbers: a full system of s numbers is a
    ticket of size s, and its hit count is how many of the drawn numbers it contains.
    ``top_hits`` names the hit counts that are the top prize tier (both ends for a game
    like Sve ili Ništa, only the high end for Loto)."""
    paying = set(paying)
    top = set(top_hits) if top_hits is not None else {0, pool.pick}
    joint = joint_hit_distribution(tickets, pool)
    p_any = Fraction(0)
    p_all_lose = Fraction(0)
    winners_dist: dict[int, Fraction] = {}
    max_dist: dict[int, Fraction] = {}
    e_sum = Fraction(0)
    e_max = Fraction(0)
    p_any_top = Fraction(0)   # at least one ticket with every number right, or none right
    for hits, p in joint.items():
        n_win = sum(1 for h in hits if h in paying)
        winners_dist[n_win] = winners_dist.get(n_win, Fraction(0)) + p
        if n_win:
            p_any += p
        else:
            p_all_lose += p
        m = max(hits)
        max_dist[m] = max_dist.get(m, Fraction(0)) + p
        e_sum += p * sum(hits)
        e_max += p * m
        if any(h in top for h in hits):
            p_any_top += p
    overlaps = [len(set(a) & set(b)) for a, b in itertools.combinations(tickets, 2)]
    return {
        "tickets": [list(t) for t in tickets],
        "n_tickets": len(tickets),
        "pairwise_overlaps": overlaps,
        "min_overlap": min(overlaps) if overlaps else None,
        "max_overlap": max(overlaps) if overlaps else None,
        "p_at_least_one_paying": float(p_any),
        "p_nothing_at_all": float(p_all_lose),
        "p_at_least_one_jackpot_tier": float(p_any_top),
        "top_tier_hits": sorted(top),
        "ticket_sizes": sorted({len(t) for t in tickets}),
        "one_in_at_least_one_paying": round(1 / float(p_any)) if p_any else None,
        "expected_total_hits": float(e_sum),
        "expected_best_ticket_hits": float(e_max),
        "distribution_of_winning_tickets": {str(k): float(v) for k, v in sorted(winners_dist.items())},
        "distribution_of_best_ticket_hits": {str(k): float(v) for k, v in sorted(max_dist.items())},
    }


def single_ticket_reference(pool: Pool, paying: Iterable[int], ticket_size: int | None = None,
                            top_hits: Iterable[int] | None = None) -> dict:
    """The same figures for one ticket of ``ticket_size`` numbers (default ``pool.pick``).

    Hits are hypergeometric: of the ``pool.pick`` drawn numbers, how many fall inside the
    ticket's ``ticket_size`` numbers."""
    paying = set(paying)
    size = ticket_size or pool.pick
    top = set(top_hits) if top_hits is not None else {0, pool.pick}
    total = comb(pool.size, pool.pick)
    pmf = {h: Fraction(comb(size, h) * comb(pool.size - size, pool.pick - h), total)
           for h in range(min(size, pool.pick) + 1)}
    assert sum(pmf.values()) == 1
    p = sum(v for h, v in pmf.items() if h in paying)
    return {"ticket_size": size, "p_paying": float(p), "one_in": round(1 / float(p)) if p else None,
            "p_jackpot_tier": float(sum(v for h, v in pmf.items() if h in top)),
            "expected_hits": float(sum(h * v for h, v in pmf.items())),
            "hit_pmf": {str(h): float(v) for h, v in pmf.items()}}


# --------------------------------------------------------------------- the selection

def candidate_tickets(readings: list[dict], game_key: str, pool: Pool, system_size: int | None = None,
                      kinds: tuple[str, ...] = ("method", "ensemble")) -> list[dict]:
    """Every distinct ticket the methods and ensembles produced, across all readings.

    With ``system_size`` the ticket is the method's top ``system_size`` numbers from its
    full ranking of the pool — a full system, which is exactly what the ranking already
    expresses. ``kinds`` restricts the sources (methods only, or ensembles only)."""
    seen: dict[Ticket, dict] = {}

    def take(block: dict, key: str, reading: int) -> None:
        source = block["ranking"] if system_size else block["picks"]
        if game_key not in source:
            return
        numbers = source[game_key][pool.name]
        t: Ticket = tuple(sorted(numbers[:system_size] if system_size else numbers))
        seen.setdefault(t, {"ticket": list(t), "sources": []})["sources"].append(
            {"reading": reading, "source": key})

    for r in readings:
        if "method" in kinds:
            for key, out in r["outputs"].items():
                if out.status == "failed":
                    continue
                take({"picks": out.picks, "ranking": out.ranking}, key, r["reading"])
        if "ensemble" in kinds:
            for key, ens in r.get("ensembles", {}).items():
                if game_key not in ens["picks"]:
                    continue
                take(ens, key, r["reading"])
    return [seen[t] for t in sorted(seen)]


def select_portfolio(candidates: list[dict], m: int, pool: Pool, paying: Iterable[int],
                     overlap_range: tuple[int, int], max_exact: int, key: str,
                     top_hits: Iterable[int] | None = None, max_subsets: int = 1_000_000) -> dict:
    """The m candidate tickets whose joint structure maximises P(at least one wins).

    Two stages, both deterministic: subsets whose pairwise overlaps all lie inside
    ``overlap_range`` are ranked by a cheap proxy (squared deviation from the neutral
    overlap pick^2/size), and the best ``max_exact`` of them are scored exactly. The
    objective is a function of the overlaps alone, so no draw data is involved.
    """
    tickets = [tuple(c["ticket"]) for c in candidates]
    if len(tickets) < m:
        raise ValueError(f"need at least {m} distinct candidate tickets, got {len(tickets)}")
    sizes = {len(t) for t in tickets}
    neutral = (sum(sizes) / len(sizes)) ** 2 / pool.size   # expected overlap of two random tickets
    n_subsets = comb(len(tickets), m)
    if n_subsets > max_subsets:
        raise ValueError(f"{n_subsets} subsets of {len(tickets)} candidates exceeds max_subsets={max_subsets}; "
                         "narrow the candidate pool (e.g. fewer readings) rather than sampling silently")
    ov = {(i, j): len(set(tickets[i]) & set(tickets[j]))
          for i, j in itertools.combinations(range(len(tickets)), 2)}
    lo, hi = overlap_range

    scored = []
    skipped_by_filter = 0
    for subset in itertools.combinations(range(len(tickets)), m):
        pairs = [ov[(i, j)] for i, j in itertools.combinations(subset, 2)]
        if min(pairs) < lo or max(pairs) > hi:
            skipped_by_filter += 1
            continue
        scored.append((sum((p - neutral) ** 2 for p in pairs), subset))
    relaxed = False
    if not scored:   # the filter was too strict for these candidates: rank them all
        relaxed = True
        for subset in itertools.combinations(range(len(tickets)), m):
            pairs = [ov[(i, j)] for i, j in itertools.combinations(subset, 2)]
            scored.append((sum((p - neutral) ** 2 for p in pairs), subset))
    # Deterministic order: proxy, then a hash of the subset (never the numbers' size).
    tb = tiebreak_values(key, len(scored))
    scored = [s + (float(tb[i]),) for i, s in enumerate(sorted(scored))]
    shortlist = sorted(scored, key=lambda s: (s[0], s[2]))[:max_exact]

    best = None
    for proxy, subset, tiebreak in shortlist:
        ev = evaluate_set([tickets[i] for i in subset], pool, paying, top_hits)
        rank = (-ev["p_at_least_one_paying"], -ev["expected_best_ticket_hits"], tiebreak)
        if best is None or rank < best[0]:
            best = (rank, subset, ev)
    _, subset, ev = best
    ev["selection"] = {
        "candidates": len(tickets), "subsets_considered": len(scored),
        "subsets_rejected_by_overlap_filter": skipped_by_filter,
        "overlap_filter": [lo, hi], "overlap_filter_relaxed": relaxed,
        "exactly_evaluated": len(shortlist), "neutral_overlap": neutral, "subsets_total": n_subsets,
        "objective": "P(at least one ticket in a paying tier), exact; "
                     "tie-break: expected hits of the best ticket, then SHA-256",
        "sources": [candidates[i]["sources"] for i in subset],
    }
    return ev


def compare_structures(chosen: Sequence[Ticket], alternatives: dict[str, Sequence[Ticket]],
                       pool: Pool, paying: Iterable[int], top_hits: Iterable[int] | None = None) -> dict:
    """The chosen set against named alternatives, exactly."""
    out = {"chosen": evaluate_set(chosen, pool, paying, top_hits)}
    for name, tickets in alternatives.items():
        if tickets and len(tickets) == len(chosen):
            out[name] = evaluate_set(tickets, pool, paying, top_hits)
    return out


def hits_against(tickets: Sequence[Ticket], drawn: Sequence[int]) -> list[int]:
    d = set(drawn)
    return [len(set(t) & d) for t in tickets]
