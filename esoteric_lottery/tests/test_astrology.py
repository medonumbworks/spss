import numpy as np
import pytest

from methods.common import Pool
from methods.vedic_astrology import (DASHA_YEARS, VedicAstrologyMethod, karana, nakshatra, tithi, tithi_name,
                                     vimshottari, yoga)
from methods.western_astrology import WesternAstrologyMethod, arc_scores, find_aspects, separation
from tests.helpers import assert_reproducible, assert_valid_output


def test_separation():
    assert separation(10, 350) == pytest.approx(20)
    assert separation(0, 180) == pytest.approx(180)


def test_find_aspects_picks_closest_within_orb():
    t = {"Sun": {"lon": 100.0}}
    n = {"Moon": {"lon": 221.0}, "Mars": {"lon": 300.0}}
    orbs = {"conjunction": 8, "sextile": 4, "square": 6, "trine": 6, "opposition": 8}
    asp = find_aspects(t, n, orbs)
    assert len(asp) == 1 and asp[0]["aspect"] == "trine" and asp[0]["natal"] == "Moon"
    assert asp[0]["strength"] == pytest.approx(1 - 1 / 6)


def test_arc_scores_peak_at_point():
    pool = Pool("euro", 12, 2)
    s = arc_scores([(45.0, 1.0)], pool)  # 15° Taurus -> sign 2
    assert int(np.argmax(s)) == 1


def test_western_method(subject, event, games):
    out = WesternAstrologyMethod({}).run(subject, event, games.values())
    assert_valid_output(out, games)
    natal = out.intermediate["natal_chart"]["points"]
    assert natal["Sun"]["sign"] == "Gemini" and natal["ASC"]["sign"] == "Scorpio"
    assert out.derived["sve_ili_nista"]["main"] == 11
    assert_reproducible(WesternAstrologyMethod, {}, subject, event, games)


def test_panchanga_formulas():
    assert tithi(0.0, 11.9) == 1 and tithi(0.0, 12.0) == 2 and tithi(10.0, 9.0) == 30
    assert tithi_name(15) == "Shukla Purnima" and tithi_name(30) == "Krishna Amavasya"
    assert nakshatra(0.0)[:2] == (1, 1) and nakshatra(359.99)[0] == 27
    assert nakshatra(13 + 1 / 3 + 3.4)[1] == 2
    assert yoga(0.0, 0.0) == 1 and yoga(200.0, 159.9) == 27
    assert karana(0.0, 3.0) == (1, "Kimstughna")
    assert karana(0.0, 6.0) == (2, "Bava")
    assert karana(0.0, 6 * 57 + 1) == (58, "Shakuni")
    assert karana(0.0, 359.0) == (60, "Naga")


def test_vimshottari_structure():
    assert sum(DASHA_YEARS.values()) == 120
    d0 = vimshottari(0.0, 2451545.0, 2451545.0)  # Ashwini start -> Ketu
    assert d0["periods"][0]["lord"] == "Ketu" and d0["periods"][1]["lord"] == "Ketu"
    # 7 Ketu years + 1 day after birth at the start of Ashwini -> Venus mahadasha, Venus antardasha.
    d1 = vimshottari(0.0, 2451545.0, 2451545.0 + 7 * 365.25 + 1)
    assert d1["periods"][0]["lord"] == "Venus" and d1["periods"][1]["lord"] == "Venus"


def test_vedic_method(subject, event, games):
    out = VedicAstrologyMethod({}).run(subject, event, games.values())
    assert_valid_output(out, games)
    p = out.intermediate["panchanga_draw"]
    assert 1 <= p["tithi"] <= 30 and 1 <= p["nakshatra"] <= 27 and p["vara_name"].startswith("Shukravara")
    assert out.intermediate["natal_moon"]["nakshatra_name"] == "Purva Ashadha"
    assert out.intermediate["vimshottari_at_draw"]["birth_nakshatra_lord"] == "Venus"
    assert_reproducible(VedicAstrologyMethod, {}, subject, event, games)
