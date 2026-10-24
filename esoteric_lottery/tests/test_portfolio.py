import itertools
from fractions import Fraction

import numpy as np
import pytest
from scipy.stats import hypergeom

from evaluation.portfolio import (candidate_tickets, evaluate_set, hits_against, joint_hit_distribution,
                                  select_portfolio, single_ticket_reference)
from methods.common import Pool, seeded_rng

POOL = Pool("main", 22, 11)
PAYING = [0, 1, 2, 3, 8, 9, 10, 11]
A = tuple(range(1, 12))
B = tuple(range(12, 23))


def test_single_ticket_matches_hypergeometric():
    ref = single_ticket_reference(POOL, PAYING)
    pmf = hypergeom(22, 11, 11).pmf(np.arange(12))
    assert [ref["hit_pmf"][str(k)] for k in range(12)] == pytest.approx(pmf.tolist())
    assert ref["p_paying"] == pytest.approx(float(sum(pmf[k] for k in PAYING)))
    assert ref["one_in"] == 12


def test_joint_distribution_is_a_distribution():
    joint = joint_hit_distribution([A, B, (1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17)], POOL)
    assert sum(joint.values()) == Fraction(1)
    assert all(isinstance(p, Fraction) and p > 0 for p in joint.values())
    # Complementary tickets: the hit counts always add up to 11.
    for (ha, hb, _), p in joint.items():
        assert ha + hb == 11


def test_marginals_equal_the_single_ticket_distribution():
    """Whatever the set, every ticket on its own is still hypergeometric."""
    tickets = [A, (1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17), (2, 6, 9, 10, 11, 15, 16, 18, 19, 20, 22)]
    joint = joint_hit_distribution(tickets, POOL)
    pmf = hypergeom(22, 11, 11).pmf(np.arange(12))
    for i in range(len(tickets)):
        marginal = {}
        for hits, p in joint.items():
            marginal[hits[i]] = marginal.get(hits[i], Fraction(0)) + p
        assert [float(marginal.get(k, 0)) for k in range(12)] == pytest.approx(pmf.tolist())


def test_expected_total_hits_is_invariant():
    """The one thing no ticket set can change."""
    sets = [[A] * 4,
            [A, B, (1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17), (6, 7, 8, 9, 10, 11, 18, 19, 20, 21, 22)],
            [(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), (1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17),
             (1, 6, 7, 8, 12, 13, 14, 18, 19, 20, 21), (2, 6, 9, 10, 11, 15, 16, 18, 19, 20, 22)]]
    for s in sets:
        assert evaluate_set(s, POOL, PAYING)["expected_total_hits"] == pytest.approx(22.0)


def test_objective_is_invariant_under_relabelling_the_numbers():
    """The decisive honesty check: the objective depends only on how the tickets overlap
    each other, never on which numbers they hold. A set and its relabelled copy score
    identically, so the selection cannot encode any prediction about the draw."""
    tickets = [(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), (1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17),
               (1, 6, 7, 8, 12, 13, 14, 18, 19, 20, 21), (2, 6, 9, 10, 11, 15, 16, 18, 19, 20, 22)]
    base = evaluate_set(tickets, POOL, PAYING)
    rng = seeded_rng("relabel")
    for _ in range(5):
        perm = {i + 1: int(v) + 1 for i, v in enumerate(rng.permutation(22))}
        relabelled = [tuple(sorted(perm[n] for n in t)) for t in tickets]
        other = evaluate_set(relabelled, POOL, PAYING)
        assert other["p_at_least_one_paying"] == base["p_at_least_one_paying"]
        assert other["expected_best_ticket_hits"] == base["expected_best_ticket_hits"]
        assert sorted(other["pairwise_overlaps"]) == sorted(base["pairwise_overlaps"])


def test_identical_tickets_gain_nothing_and_complements_gain_little():
    single = single_ticket_reference(POOL, PAYING)["p_paying"]
    same = evaluate_set([A] * 4, POOL, PAYING)
    assert same["p_at_least_one_paying"] == pytest.approx(single)
    # Complementary pairs win together, so two pairs are only two independent chances.
    comp = evaluate_set([A, B, (1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17),
                         (6, 7, 8, 9, 10, 11, 18, 19, 20, 21, 22)], POOL, PAYING)
    spread = evaluate_set([(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), (1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17),
                           (1, 6, 7, 8, 12, 13, 14, 18, 19, 20, 21),
                           (2, 6, 9, 10, 11, 15, 16, 18, 19, 20, 22)], POOL, PAYING)
    assert same["p_at_least_one_paying"] < comp["p_at_least_one_paying"] < spread["p_at_least_one_paying"]
    assert comp["p_at_least_one_jackpot_tier"] == pytest.approx(2 * single_ticket_reference(POOL, PAYING)["p_jackpot_tier"])
    assert spread["p_at_least_one_jackpot_tier"] == pytest.approx(4 * single_ticket_reference(POOL, PAYING)["p_jackpot_tier"])


def test_exact_matches_monte_carlo():
    tickets = [(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), (1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17),
               (1, 6, 7, 8, 12, 13, 14, 18, 19, 20, 21), (2, 6, 9, 10, 11, 15, 16, 18, 19, 20, 22)]
    exact = evaluate_set(tickets, POOL, PAYING)
    rng = np.random.Generator(np.random.PCG64(7))
    reps, wins = 200_000, 0
    for _ in range(reps):
        drawn = rng.permutation(22)[:11] + 1
        if any(h in set(PAYING) for h in hits_against(tickets, drawn)):
            wins += 1
    se = (exact["p_at_least_one_paying"] * (1 - exact["p_at_least_one_paying"]) / reps) ** 0.5
    assert abs(wins / reps - exact["p_at_least_one_paying"]) < 4 * se


def test_selection_picks_the_best_of_the_candidates():
    rng = seeded_rng("candidates")
    cands = [{"ticket": sorted(int(v) + 1 for v in rng.permutation(22)[:11]), "sources": []} for _ in range(12)]
    chosen = select_portfolio(cands, 4, POOL, PAYING, (3, 8), 400, "k")
    tickets = [tuple(t) for t in chosen["tickets"]]
    assert len(tickets) == 4 and len(set(tickets)) == 4
    best = max(evaluate_set(list(c), POOL, PAYING)["p_at_least_one_paying"]
               for c in itertools.combinations([tuple(c["ticket"]) for c in cands], 4))
    assert chosen["p_at_least_one_paying"] == pytest.approx(best)
    assert chosen["expected_total_hits"] == pytest.approx(22.0)
    # Deterministic.
    assert select_portfolio(cands, 4, POOL, PAYING, (3, 8), 400, "k")["tickets"] == chosen["tickets"]


def test_selection_relaxes_an_impossible_filter():
    cands = [{"ticket": list(A), "sources": []}, {"ticket": list(B), "sources": []},
             {"ticket": sorted(A[:6] + B[:5]), "sources": []}, {"ticket": sorted(A[6:] + B[5:]), "sources": []}]
    chosen = select_portfolio(cands, 4, POOL, PAYING, (5, 6), 50, "k")
    assert chosen["selection"]["overlap_filter_relaxed"] is True
    assert len(chosen["tickets"]) == 4


def test_candidate_tickets_dedupes_across_readings(cfg, subject, event, games):
    from methods import build_methods
    from methods.ensemble import aggregate, equal_weights
    readings = []
    for k in (1, 2):
        outs = {m.key: m.run(subject, event, games.values()) for m in build_methods(cfg, seed=cfg["seed"] + k - 1)}
        ens = {"ensemble_equal": aggregate(outs.values(), equal_weights(outs.values()), games.values(), f"e{k}")}
        readings.append({"reading": k, "outputs": outs, "ensembles": ens})
    cands = candidate_tickets(readings, "sve_ili_nista", POOL)
    assert len(cands) == len({tuple(c["ticket"]) for c in cands})
    assert all(len(c["ticket"]) == 11 for c in cands)
    assert any(len(c["sources"]) > 1 for c in cands)   # the seed-independent methods repeat
