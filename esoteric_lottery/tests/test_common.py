from datetime import timedelta

import numpy as np
import pytest

from methods.common import (Pool, digit_root, digits_from_sequence, local_datetime, normalize_latin, rank_pool,
                            reduce_to_range, seeded_rng, sequence_scores, tiebreak_values)

SET22 = Pool("main", 22, 11)
DIG = Pool("digits", 10, 6, "digits")


def test_reduce_to_range():
    assert [reduce_to_range(v, SET22) for v in (1, 22, 23, 0, 44, -1)] == [1, 22, 1, 22, 22, 21]
    assert [reduce_to_range(v, DIG) for v in (0, 9, 10, 123)] == [0, 9, 0, 3]


def test_sequence_scores_harmonic_and_cumulative():
    s = sequence_scores([3, 25, 4], SET22)  # 25 -> 3
    assert s[2] == pytest.approx(1 + 1 / 2)
    assert s[3] == pytest.approx(1 / 3)
    assert s.sum() == pytest.approx(1 + 1 / 2 + 1 / 3)


def test_rank_pool_orders_by_score_then_hash():
    scores = np.zeros(22)
    scores[[4, 9]] = [2.0, 1.0]
    r = rank_pool(scores, SET22, "k")
    assert r[:2] == [5, 10]
    assert sorted(r) == list(range(1, 23))
    assert r == rank_pool(scores, SET22, "k")
    assert r[2:] != rank_pool(scores, SET22, "other")[2:]


def test_tiebreak_is_reproducible_and_in_unit_interval():
    a = tiebreak_values("x", 50)
    assert np.array_equal(a, tiebreak_values("x", 50))
    assert ((a >= 0) & (a < 1)).all()


def test_digits_fill_when_short():
    d, derived = digits_from_sequence([12, 7], DIG, "k")
    assert d[:2] == [2, 7] and len(d) == 6 and derived == 2
    assert all(0 <= x <= 9 for x in d)


def test_digit_root_masters():
    assert digit_root(1991) == 2
    assert digit_root(29, masters=(11, 22, 33)) == 11
    assert digit_root(29) == 2
    assert digit_root(0) == 0


def test_normalize_latin_croatian():
    assert normalize_latin("Četvrtak, Đakovo, Šibenik-Žminj") == "CETVRTAKDAKOVOSIBENIKZMINJ"


def test_seeded_rng_reproducible():
    assert seeded_rng("a", 1).integers(0, 1 << 30) == seeded_rng("a", 1).integers(0, 1 << 30)
    assert seeded_rng("a", 1).integers(0, 1 << 30) != seeded_rng("a", 2).integers(0, 1 << 30)


def test_timezones_use_historical_dst():
    birth = local_datetime("1991-05-31", "18:20", "Europe/Zagreb")
    draw = local_datetime("2026-10-23", "19:00", "Europe/Zagreb")
    winter = local_datetime("2026-12-01", "19:00", "Europe/Zagreb")
    assert birth.utcoffset() == timedelta(hours=2)
    assert draw.utcoffset() == timedelta(hours=2)
    assert winter.utcoffset() == timedelta(hours=1)


def test_pool_validation():
    with pytest.raises(ValueError):
        Pool("x", 5, 6)
    with pytest.raises(ValueError):
        Pool("x", 5, 2, "bag")
