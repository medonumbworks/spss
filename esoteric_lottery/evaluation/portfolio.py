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
                      kinds: tuple[str, ...] = ("method", "ensemble"), slices: int = 1) -> list[dict]:
    """Every distinct ticket the methods and ensembles produced, across all readings.

    With ``system_size`` the ticket is the method's top ``system_size`` numbers from its
    full ranking of the pool — a full system, which is exactly what the ranking already
    expresses, and ``slices`` takes that many consecutive windows of it (the top s, the
    next s, ...), which a ranking of the whole pool supports naturally and which a
    many-ticket packing needs. ``kinds`` restricts the sources (methods only, or
    ensembles only)."""
    seen: dict[Ticket, dict] = {}

    def take(block: dict, key: str, reading: int) -> None:
        source = block["ranking"] if system_size else block["picks"]
        if game_key not in source:
            return
        numbers = source[game_key][pool.name]
        if not system_size:
            windows = [(0, tuple(sorted(numbers)))]
        else:
            # Consecutive windows of the method's own ranking: its top s, its next s, and
            # so on. Each is that method's ordering, and windows of one method are
            # disjoint by construction.
            windows = [(w, tuple(sorted(numbers[w * system_size:(w + 1) * system_size])))
                       for w in range(slices)
                       if len(numbers[w * system_size:(w + 1) * system_size]) == system_size]
        for w, t in windows:
            entry = seen.setdefault(t, {"ticket": list(t), "sources": []})
            entry["sources"].append({"reading": reading, "source": key,
                                     **({"rank_window": w + 1} if system_size else {})})

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


# ------------------------------------------------- many tickets: covering and counting

def evaluate_set_enumerated(tickets: Sequence[Ticket], pool: Pool, paying: Iterable[int],
                            top_hits: Iterable[int] | None = None, chunk: int = 400_000) -> dict:
    """Exact figures by enumerating every possible draw.

    ``joint_hit_distribution`` tracks a hit count per ticket, so its state space grows
    like (pick+1)^tickets and it stops being usable past a handful of tickets. Counting
    straight over all C(size, pick) equally likely draws does not: for Loto that is
    8 145 060 draws, which numpy handles in one pass. The result is exact — a count, not
    a sample — and ``tests`` check it against the DP where both are feasible.
    """
    import numpy as np

    paying = set(paying)
    top = set(top_hits) if top_hits is not None else {0, pool.pick}
    masks = np.zeros((len(tickets), pool.size + 1), dtype=np.int8)
    for i, t in enumerate(tickets):
        masks[i, list(t)] = 1
    total = comb(pool.size, pool.pick)
    winners_hist = np.zeros(len(tickets) + 1, dtype=np.int64)
    max_hist = np.zeros(pool.pick + 1, dtype=np.int64)
    per_ticket_hits = np.zeros((len(tickets), pool.pick + 1), dtype=np.int64)
    top_any = 0
    seen = 0
    it = itertools.combinations(range(1, pool.size + 1), pool.pick)
    while True:
        block = list(itertools.islice(it, chunk))
        if not block:
            break
        draws = np.array(block, dtype=np.int16)
        seen += len(draws)
        hits = np.stack([masks[i][draws].sum(axis=1) for i in range(len(tickets))], axis=1)
        for i in range(len(tickets)):
            per_ticket_hits[i] += np.bincount(hits[:, i], minlength=pool.pick + 1)
        is_win = np.isin(hits, list(paying))
        winners_hist += np.bincount(is_win.sum(axis=1), minlength=len(tickets) + 1)
        max_hist += np.bincount(hits.max(axis=1), minlength=pool.pick + 1)
        top_any += int(np.isin(hits, list(top)).any(axis=1).sum())
    assert seen == total, f"enumerated {seen} of {total} draws"
    overlaps = [len(set(a) & set(b)) for a, b in itertools.combinations(tickets, 2)]
    p_any = Fraction(int(total - winners_hist[0]), total)
    e_sum = sum(Fraction(int(c) * h, total) for i in range(len(tickets))
                for h, c in enumerate(per_ticket_hits[i]))
    e_max = sum(Fraction(int(c) * h, total) for h, c in enumerate(max_hist))
    return {
        "tickets": [list(t) for t in tickets],
        "n_tickets": len(tickets),
        "method": "exact enumeration of all draws",
        "draws_enumerated": total,
        "pairwise_overlaps": overlaps,
        "min_overlap": min(overlaps) if overlaps else None,
        "max_overlap": max(overlaps) if overlaps else None,
        "p_at_least_one_paying": float(p_any),
        "p_nothing_at_all": float(Fraction(int(winners_hist[0]), total)),
        "p_at_least_one_jackpot_tier": float(Fraction(top_any, total)),
        "one_in_at_least_one_paying": round(1 / float(p_any)) if p_any else None,
        "expected_total_hits": float(e_sum),
        "expected_best_ticket_hits": float(e_max),
        "top_tier_hits": sorted(top),
        "ticket_sizes": sorted({len(t) for t in tickets}),
        "distribution_of_winning_tickets": {str(k): float(Fraction(int(c), total))
                                            for k, c in enumerate(winners_hist) if c},
        "distribution_of_best_ticket_hits": {str(k): float(Fraction(int(c), total))
                                             for k, c in enumerate(max_hist) if c},
    }


def evaluate_auto(tickets: Sequence[Ticket], pool: Pool, paying: Iterable[int],
                  top_hits: Iterable[int] | None = None, dp_limit: int = 6) -> dict:
    """Whichever exact method is feasible: the DP for few tickets, enumeration for many."""
    if len(tickets) <= dp_limit:
        return evaluate_set(tickets, pool, paying, top_hits)
    return evaluate_set_enumerated(tickets, pool, paying, top_hits)


def covered_subsets(tickets: Sequence[Ticket], k: int) -> set[tuple[int, ...]]:
    """Every k-subset of numbers contained in at least one ticket."""
    out: set[tuple[int, ...]] = set()
    for t in tickets:
        out.update(itertools.combinations(sorted(t), k))
    return out


def select_portfolio_greedy(candidates: list[dict], m: int, pool: Pool, key: str,
                            max_overlap: int, fixed: Sequence[Ticket] = (),
                            min_hits_to_win: int = 3) -> dict:
    """Pick m tickets when exhaustive search is hopeless: a packing, built greedily.

    A system of s numbers wins when at least ``min_hits_to_win`` of the drawn numbers lie
    inside it, i.e. when one of the k-subsets of the drawn numbers is contained in it. So
    the set wins more often the more DISTINCT k-subsets it covers, and a pair of tickets
    sharing k or more numbers wastes coverage. Greedy: repeatedly add the candidate that
    covers the most new k-subsets, never exceeding ``max_overlap`` with anything already
    chosen (including ``fixed``, the tickets already committed elsewhere).

    Like every other objective here this counts subsets of the tickets themselves and
    never looks at a draw, so it stays invariant under relabelling the numbers.
    """
    chosen: list[Ticket] = []
    covered = covered_subsets(fixed, min_hits_to_win)
    pool_of = {tuple(c["ticket"]): c for c in candidates}
    remaining = [tuple(c["ticket"]) for c in candidates if tuple(c["ticket"]) not in set(fixed)]
    tb = tiebreak_values(key, len(remaining))
    order = {t: float(tb[i]) for i, t in enumerate(remaining)}
    blocked = 0
    while len(chosen) < m and remaining:
        best, best_rank = None, None
        for t in remaining:
            if any(len(set(t) & set(o)) > max_overlap for o in list(fixed) + chosen):
                continue
            new = len(covered_subsets([t], min_hits_to_win) - covered)
            rank = (-new, order[t])
            if best_rank is None or rank < best_rank:
                best, best_rank = t, rank
        if best is None:          # nothing left within the overlap limit
            blocked = len(remaining)
            break
        chosen.append(best)
        covered |= covered_subsets([best], min_hits_to_win)
        remaining.remove(best)
    return {
        "tickets": [list(t) for t in chosen],
        "sources": [pool_of[t]["sources"] for t in chosen],
        "selection": {
            "strategy": "greedy packing", "candidates": len(candidates), "requested": m,
            "selected": len(chosen), "max_overlap": max_overlap,
            "fixed_tickets": [list(t) for t in fixed],
            "subset_size": min_hits_to_win,
            "distinct_subsets_covered": len(covered),
            "subsets_in_pool": comb(pool.size, min_hits_to_win),
            "candidates_blocked_by_overlap": blocked,
            "objective": f"maximise distinct {min_hits_to_win}-subsets covered, "
                         f"pairwise overlap <= {max_overlap}",
        },
    }


def assign_companion_digits(sources: Sequence[Sequence[dict]], digits: dict[tuple[str, int], list[int]],
                            key: str, taken: Iterable[Sequence[int]] = ()) -> list[dict]:
    """Give every selected ticket its own distinct companion number (the Joker).

    Several systems can come from the same method (different windows of one ranking), and
    they would otherwise carry the same Joker. Playing one Joker twice cannot raise the
    chance that some Joker wins — it only doubles a win that already happened — so each
    ticket gets a distinct one: its own method's, then the same method's other readings,
    then any unused one in SHA-256 order.
    """
    available = {k: tuple(v) for k, v in digits.items()}
    used: set[tuple[int, ...]] = {tuple(t) for t in taken}   # already carried by other tickets
    order = sorted(available)
    tb = tiebreak_values(key, len(order))
    fallback = [order[i] for i in sorted(range(len(order)), key=lambda i: float(tb[i]))]
    out = []
    for src in sources:
        first = src[0]
        name, reading = first["source"], first["reading"]
        preferred = [(name, reading)] + [k for k in order if k[0] == name] + fallback
        pick = next((k for k in preferred if k in available and available[k] not in used), None)
        if pick is None:
            raise ValueError("not enough distinct companion numbers for the selected tickets")
        used.add(available[pick])
        out.append({"digits": list(available[pick]), "from": {"source": pick[0], "reading": pick[1]},
                    "own_source": pick == (name, reading)})
    return out


def select_max_weight_system(weights: dict[int, float], size: int, existing: Sequence[Ticket],
                             max_overlap: int, pool: Pool) -> dict:
    """The ``size`` numbers with the highest total weight that share at most
    ``max_overlap`` numbers with every ticket already played.

    When every number of the pool is already in play, a system disjoint in NUMBERS is
    impossible. The useful notion is disjoint in COVERAGE: with an overlap of at most 2,
    no triple — and so no combination — of the new system is one that is already being
    played, which is what makes the new tickets add coverage instead of repeating it.

    ``weights`` carries the esoteric part (the methods' collective Borda preference over
    the pool); the constraint carries the combinatorial part. Branch and bound over the
    numbers in decreasing weight, pruning on the best score the remaining numbers could
    still reach, so the answer is provably optimal rather than greedy.
    """
    order = sorted(range(1, pool.size + 1), key=lambda n: (-weights.get(n, 0.0), n))
    member = [[i for i, t in enumerate(existing) if n in set(t)] for n in order]
    best: dict = {"score": float("-inf"), "numbers": None}
    nodes = 0

    def dfs(i: int, chosen: list[int], score: float, counts: list[int]) -> None:
        nonlocal nodes
        nodes += 1
        if len(chosen) == size:
            if score > best["score"]:
                best.update(score=score, numbers=sorted(chosen))
            return
        if i >= len(order):
            return
        optimistic = sum(weights.get(order[j], 0.0) for j in range(i, min(i + size - len(chosen), len(order))))
        if score + optimistic <= best["score"]:
            return
        if all(counts[j] < max_overlap for j in member[i]):
            for j in member[i]:
                counts[j] += 1
            dfs(i + 1, chosen + [order[i]], score + weights.get(order[i], 0.0), counts)
            for j in member[i]:
                counts[j] -= 1
        dfs(i + 1, chosen, score, counts)

    dfs(0, [], 0.0, [0] * len(existing))
    if best["numbers"] is None:
        raise ValueError(f"no system of {size} shares at most {max_overlap} numbers with all "
                         f"{len(existing)} tickets already played")
    numbers = tuple(best["numbers"])
    overlaps = [len(set(numbers) & set(t)) for t in existing]
    return {
        "ticket": list(numbers), "weight": best["score"], "nodes_explored": nodes,
        "max_overlap_with_played": max(overlaps) if overlaps else 0,
        "overlaps_with_played": overlaps,
        "numbers_unused_by_played": sorted(set(range(1, pool.size + 1)) - {n for t in existing for n in t}),
        "combinations": comb(size, pool.pick),
        "objective": f"maximum total weight over {size} numbers, overlap <= {max_overlap} with every "
                     "ticket already played (branch and bound, provably optimal)",
    }


def borda_weights(readings: list[dict], game_key: str, pool: Pool,
                  source: str = "ensemble_equal") -> dict[int, float]:
    """The methods' collective preference over the whole pool, summed across readings."""
    out = {n: 0.0 for n in range(1, pool.size + 1)}
    for r in readings:
        ens = r.get("ensembles", {}).get(source)
        if not ens or game_key not in ens["points"]:
            continue
        for i, p in enumerate(ens["points"][game_key][pool.name], start=1):
            out[i] += float(p)
    return out
