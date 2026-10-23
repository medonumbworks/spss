from fractions import Fraction

import numpy as np
import pytest

from methods.common import seeded_rng
from methods.iching import (KING_WEN, TRIGRAMS, IChingMethod, cast_hexagram, change, hexagram_from_trigrams,
                            king_wen_number, mutual, yarrow_line)
from methods.meihua import MeihuaMethod, cast_from_numbers, five_phase_relation
from tests.helpers import assert_reproducible, assert_valid_output


def test_king_wen_is_a_bijection():
    assert sorted(KING_WEN.values()) == list(range(1, 65))


@pytest.mark.parametrize("upper,lower,number", [
    ("Qian", "Qian", 1), ("Kun", "Kun", 2), ("Kan", "Zhen", 3), ("Gen", "Kan", 4), ("Kan", "Qian", 5),
    ("Kun", "Qian", 11), ("Qian", "Kun", 12), ("Dui", "Li", 49), ("Dui", "Gen", 31), ("Qian", "Xun", 44),
    ("Kan", "Li", 63), ("Li", "Kan", 64),
])
def test_known_hexagrams(upper, lower, number):
    assert king_wen_number(hexagram_from_trigrams(upper, lower)) == number


def test_earlier_heaven_numbers():
    assert {k: v["number"] for k, v in TRIGRAMS.items()} == {
        "Qian": 1, "Dui": 2, "Li": 3, "Zhen": 4, "Xun": 5, "Kan": 6, "Gen": 7, "Kun": 8}


def test_mutual_and_change():
    ge = hexagram_from_trigrams("Dui", "Li")
    assert king_wen_number(mutual(ge)) == 44
    assert king_wen_number(change(ge, [1])) == 31


def test_meihua_classic_plum_blossom_example():
    """《梅花易數》 觀梅占: 辰年(5) 十二月 十七日 申時(9) -> 澤火革, 初爻動 -> 澤山咸, 互 乾/巽."""
    c = cast_from_numbers(5, 12, 17, 9)
    assert c["sum_ymd"] == 34 and c["sum_ymdh"] == 43
    assert (c["upper_number"], c["lower_number"], c["moving_line"]) == (2, 3, 1)
    assert c["original"]["king_wen"] == 49
    assert c["changed"]["king_wen"] == 31
    assert {c["mutual"]["upper"], c["mutual"]["lower"]} == {"Qian", "Xun"}
    assert c["use_trigram"] == "Li" and c["body_trigram"] == "Dui"


def test_meihua_remainder_zero_maps_to_eight_and_six():
    c = cast_from_numbers(7, 9, 14, 10)  # sums 30 and 40
    assert (c["upper_number"], c["lower_number"], c["moving_line"]) == (6, 8, 4)
    assert c["original"]["king_wen"] == 8 and c["changed"]["king_wen"] == 45 and c["mutual"]["king_wen"] == 23


def test_five_phase_relation():
    assert five_phase_relation("metal", "fire") == "use controls body (用克體)"
    assert five_phase_relation("fire", "wood") == "use generates body (用生體)"
    assert five_phase_relation("earth", "earth") == "same element (比和)"


def test_meihua_target_draw(subject, event, games):
    out = MeihuaMethod({"hour_basis": "true_solar"}).run(subject, event, games.values())
    assert_valid_output(out, games)
    draw = out.intermediate["draw_cast"]
    assert draw["calendar"]["lunar_date"]["month"] == 9 and draw["calendar"]["lunar_date"]["day"] == 14
    assert draw["calendar"]["hour_branch"] == "酉"
    civil = MeihuaMethod({"hour_basis": "civil"}).run(subject, event, games.values())
    assert civil.intermediate["draw_cast"]["calendar"]["hour_branch"] == "戌"
    assert_reproducible(MeihuaMethod, {}, subject, event, games)


def test_yarrow_line_values_and_removals():
    rng = seeded_rng("t")
    for _ in range(200):
        v, steps = yarrow_line(rng)
        assert v in (6, 7, 8, 9)
        assert steps[0]["removed"] in (5, 9)
        assert all(s["removed"] in (4, 8) for s in steps[1:])


def exact_yarrow_distribution():
    """Exact line probabilities of the simulated procedure (uniform splits)."""
    def rem(n):
        return 4 if n % 4 == 0 else n % 4
    dist = {49: Fraction(1)}
    for _ in range(3):
        nxt = {}
        for b, p in dist.items():
            lefts = range(1, b - 1)
            for left in lefts:
                nb = b - (1 + rem(left) + rem(b - left - 1))
                nxt[nb] = nxt.get(nb, 0) + p / len(lefts)
        dist = nxt
    return {b // 4: float(p) for b, p in dist.items()}


def test_yarrow_probabilities_close_to_tradition():
    exact = exact_yarrow_distribution()
    for v, p in {6: 1 / 16, 7: 5 / 16, 8: 7 / 16, 9: 3 / 16}.items():
        assert exact[v] == pytest.approx(p, abs=0.03)
    rng = seeded_rng("probabilities")
    vals = np.array([yarrow_line(rng)[0] for _ in range(40000)])
    for v in (6, 7, 8, 9):
        assert float(np.mean(vals == v)) == pytest.approx(exact[v], abs=0.012)


def test_cast_relating_hexagram():
    c = cast_hexagram(seeded_rng("x"))
    if c["moving_lines"]:
        assert c["relating"]["king_wen"] != c["primary"]["king_wen"]
    else:
        assert c["relating"] is None


def test_iching_method(subject, event, games):
    out = IChingMethod({"casts": 12}).run(subject, event, games.values())
    assert_valid_output(out, games)
    assert all(d in (6, 7, 8, 9) for d in out.picks["joker"]["digits"])
    assert_reproducible(IChingMethod, {"casts": 12}, subject, event, games)
