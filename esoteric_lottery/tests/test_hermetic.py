from datetime import date

import pytest

from methods.common import Event, Place, local_datetime
from methods.gematria import hebrew_value
from methods.hermetic import (CHALDEAN, DAY_RULER, KAMEAS, MAGIC_CONSTANT, SEPHIRA_OF, HermeticMethod,
                              cell_of, lines_through, planetary_hour, read_kamea)
from tests.helpers import assert_reproducible, assert_valid_output


@pytest.mark.parametrize("planet", list(KAMEAS))
def test_every_kamea_is_a_genuine_magic_square(planet):
    """Agrippa II.22, transcribed: this both states the property and checks the copy."""
    k = KAMEAS[planet]
    n, c = len(k), MAGIC_CONSTANT[planet]
    assert sorted(v for row in k for v in row) == list(range(1, n * n + 1))
    assert all(sum(row) == c for row in k)
    assert all(sum(k[i][j] for i in range(n)) == c for j in range(n))
    assert sum(k[i][i] for i in range(n)) == c
    assert sum(k[i][n - 1 - i] for i in range(n)) == c
    assert c == n * (n * n + 1) // 2


def test_kamea_orders_match_the_planets():
    assert [len(KAMEAS[p]) for p in CHALDEAN] == [3, 4, 5, 6, 7, 8, 9]
    assert MAGIC_CONSTANT["Sol"] == 111 and MAGIC_CONSTANT["Luna"] == 369


def test_day_rulers_are_the_classical_ones():
    assert DAY_RULER[7] == "Sol" and DAY_RULER[1] == "Luna" and DAY_RULER[6] == "Saturn"
    assert set(DAY_RULER.values()) == set(CHALDEAN)
    # The classical weekday order follows from stepping the Chaldean order by 24 hours.
    for wd in range(1, 8):
        nxt = wd % 7 + 1
        i = CHALDEAN.index(DAY_RULER[wd])
        assert CHALDEAN[(i + 24) % 7] == DAY_RULER[nxt]


def test_lines_through_a_cell():
    k = KAMEAS["Saturn"]
    assert cell_of(k, 5) == (1, 1)
    lines = lines_through(k, (1, 1))
    assert lines["row"] == [3, 5, 7] and lines["column"] == [9, 5, 1]
    assert lines["diagonal"] == [4, 5, 6] and lines["antidiagonal"] == [2, 5, 8]
    assert all(sum(v) == 15 for v in lines.values())
    assert "diagonal" not in lines_through(k, (0, 1))


def test_read_kamea_wraps_the_seed_into_the_square():
    r = read_kamea("Saturn", 5)
    assert r["seed_value"] == 5 and r["magic_constant"] == 15
    assert read_kamea("Saturn", 14)["seed_value"] == 5     # 14 -> (14-1) mod 9 + 1
    assert read_kamea("Sol", 6)["order"] == 6


def test_sephirot_planetary_attributions():
    assert SEPHIRA_OF == {"Saturn": 3, "Jupiter": 4, "Mars": 5, "Sol": 6, "Venus": 7,
                          "Mercury": 8, "Luna": 9}


def test_divine_name_gematria_matches_the_gematria_module():
    assert hebrew_value("אהיה") == 21
    assert hebrew_value("יה") == 15
    assert hebrew_value("אל") == 31
    assert hebrew_value("אלהים") == 86
    assert hebrew_value("יהוה") == 26
    assert hebrew_value("אדני") == 65


def test_planetary_hour_on_a_known_sunrise(event):
    h = planetary_hour(event)
    assert h["hour_ruler"] in CHALDEAN and 1 <= h["hour_index"] <= 24
    assert h["day_ruler"] == DAY_RULER[date.fromisoformat(h["day_of_sunrise"]).isoweekday()]
    assert 30 < h["hour_length_minutes"] < 90
    # The first unequal hour after sunrise always belongs to the day's ruler.
    # Sunrise in Zagreb on 2026-10-25 is 05:26 UT = 06:26 local (CET), hours of 52 min.
    zagreb = Place("Zagreb", 45.8150, 15.9819, "Europe/Zagreb")
    first = planetary_hour(Event(local_datetime("2026-10-25", "07:00", "Europe/Zagreb"), zagreb))
    assert first["hour_index"] == 1 and first["hour_ruler"] == first["day_ruler"]
    assert first["day_ruler"] == "Sol"          # 2026-10-25 is a Sunday
    assert planetary_hour(Event(local_datetime("2026-10-25", "07:30", "Europe/Zagreb"),
                                zagreb))["hour_index"] == 2
    # Unequal hours: in late October the night hours are the longer ones.
    night = planetary_hour(Event(local_datetime("2026-10-25", "19:00", "Europe/Zagreb"), zagreb))
    assert night["part"] == "night" and night["hour_length_minutes"] > first["hour_length_minutes"]


def test_hermetic_method(subject, event, games):
    out = HermeticMethod({}).run(subject, event, games.values())
    assert_valid_output(out, games)
    i = out.intermediate
    assert i["draw_kamea"]["planet"] == i["draw_planetary_hour"]["hour_ruler"]
    assert i["draw_sephira"]["number"] == SEPHIRA_OF[i["draw_kamea"]["planet"]]
    assert i["paths_of_wisdom"] == 32
    assert_reproducible(HermeticMethod, {}, subject, event, games)
