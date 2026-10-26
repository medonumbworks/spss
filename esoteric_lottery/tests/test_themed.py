"""The themed blocks, and the structural promises made about the delivered set."""
import csv
import itertools
import pathlib
from math import comb

import numpy as np
import pytest

from evaluation.themed import (TripleLedger, band_of, coverage_block, empirical_ok,
                               frequency_table, not_a_pattern, search_ticket, spread_ok,
                               unpopular_ok, weighted_block, weighted_order)
from methods.common import seeded_rng

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_ledger_is_the_same_thing_as_overlap_at_most_two():
    """Every triple used once <=> no two combinations share three numbers."""
    rng = seeded_rng("ledger")
    led = TripleLedger()
    chosen = []
    for _ in range(40):
        t = tuple(sorted(int(x) for x in rng.choice(np.arange(1, 46), 6, replace=False)))
        if led.fits(t):
            led.add(t)
            chosen.append(t)
    assert len(chosen) > 5
    assert max(len(set(a) & set(b)) for a, b in itertools.combinations(chosen, 2)) <= 2
    # and the ledger refuses anything that would share three
    a = chosen[0]
    clash = tuple(sorted(a[:3] + (46 - a[0], 46 - a[1], 46 - a[2])[:3]))
    if len(set(clash)) == 6:
        assert not led.fits(clash)


def test_bands_cover_the_pool_once():
    assert sorted(band_of(n) for n in range(1, 46)) == sorted(list(range(5)) * 9)


def test_shape_predicates():
    assert unpopular_ok((19, 27, 32, 33, 38, 41))
    assert not unpopular_ok((1, 2, 3, 4, 5, 6))          # nothing above 31
    assert not unpopular_ok((7, 33, 35, 38, 41, 44))     # holds the number 7
    assert not unpopular_ok((2, 5, 33, 36, 40, 44))      # two numbers at or below 12
    assert not unpopular_ok((13, 20, 27, 34, 41, 45))    # 13,20,27,34,41 is an arithmetic run
    assert empirical_ok((3, 15, 22, 29, 37, 42))
    assert not empirical_ok((40, 41, 42, 43, 44, 45))    # sum far outside the envelope
    assert not not_a_pattern((1, 2, 3, 10, 20, 30))      # three in a row
    assert not not_a_pattern((1, 3, 5, 7, 9, 11))        # all odd
    assert not not_a_pattern((2, 12, 22, 32, 42, 5))     # five numbers ending in 2
    assert spread_ok((3, 12, 20, 29, 38, 44))
    assert not spread_ok((1, 2, 3, 4, 10, 11))           # leans on one band


def test_weighted_order_favours_heavy_numbers():
    rng = seeded_rng("order")
    w = [1.0] * 44 + [500.0]                              # 45 is very heavy
    firsts = [weighted_order(rng, w)[0] for _ in range(60)]
    assert firsts.count(45) > 30
    assert all(sorted(weighted_order(rng, w)) == list(range(1, 46)) for _ in range(5))


def test_search_ticket_respects_the_ledger_and_gives_up_cleanly():
    led = TripleLedger()
    order = list(range(1, 46))
    first = search_ticket(led, order, lambda t: True, set())
    assert first == (1, 2, 3, 4, 5, 6)
    led.add(first)
    again = search_ticket(led, order, lambda t: True, set())
    assert again is not None and len(set(again) & set(first)) <= 2
    tiny = TripleLedger()
    assert search_ticket(tiny, [1, 2, 3], lambda t: True, set()) is None   # too few numbers


def test_coverage_block_is_even_and_non_overlapping():
    led = TripleLedger()
    block = coverage_block(60, seeded_rng("cov"), led, set(), sharpness=6.0)
    assert len(block) == len(set(block)) == 60
    assert max(len(set(a) & set(b)) for a, b in itertools.combinations(block, 2)) <= 2
    freq = frequency_table(block)
    assert set(freq) == set(range(1, 46))
    target = 60 * 6 / 45
    assert max(freq.values()) - min(freq.values()) <= 4
    assert abs(sum(freq.values()) / 45 - target) < 1e-9
    assert all(spread_ok(t) and not_a_pattern(t) for t in block)


def test_weighted_block_honours_its_filter():
    led = TripleLedger()
    w = [1.0 + 2.0 * (n > 31) for n in range(1, 46)]
    block = weighted_block(12, seeded_rng("wb"), w,
                           lambda t: unpopular_ok(t) and not_a_pattern(t), led, set())
    assert len(block) == 12
    assert all(unpopular_ok(t) and not_a_pattern(t) for t in block)
    assert max(len(set(a) & set(b)) for a, b in itertools.combinations(block, 2)) <= 2


def test_delivered_231_for_2026_10_29():
    path = ROOT / "results/draw_2026-10-29/loto_231_kombinacija.csv"
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    tickets = [tuple(int(r[f"b{i}"]) for i in range(1, 7)) for r in rows]
    assert len(tickets) == 231 == len(set(tickets))
    assert all(len(set(t)) == 6 and all(1 <= n <= 45 for n in t) for t in tickets)
    assert all(list(t) == sorted(t) for t in tickets)
    # the promise the probabilities rest on
    assert max(len(set(a) & set(b)) for a, b in itertools.combinations(tickets, 2)) == 2
    for k, per in ((3, 20), (4, 15), (5, 6)):
        covered = {c for t in tickets for c in itertools.combinations(t, k)}
        assert len(covered) == 231 * per, k          # nothing counted twice
    blocks = {}
    for r, t in zip(rows, tickets):
        blocks.setdefault(r["blok"], []).append(t)
    assert {k: len(v) for k, v in blocks.items()} == {
        "pokrivenost": 100, "statistika": 50, "ezoterija": 50, "prosudba": 31}
    assert all(unpopular_ok(t) and not_a_pattern(t) for t in blocks["prosudba"])
    assert all(empirical_ok(t) and not_a_pattern(t) for t in blocks["statistika"])
    assert all(spread_ok(t) and not_a_pattern(t) for t in blocks["pokrivenost"])
    freq = frequency_table(blocks["pokrivenost"])
    assert min(freq.values()) >= 12 and max(freq.values()) <= 15
    assert set(frequency_table(tickets)) == set(range(1, 46))
    assert min(frequency_table(tickets).values()) > 0      # every number is in play
    # and the headline figure
    assert 231 * 235 / comb(45, 6) == pytest.approx(0.0066648, abs=1e-6)
