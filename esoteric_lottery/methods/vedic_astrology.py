"""Vedic astrology (Jyotish): sidereal positions, Panchanga and Vimshottari dasha.

Traditional quantities (sidereal zodiac, Lahiri/Chitrapaksha ayanamsa, whole-sign
houses, mean lunar node for Rahu, Ketu = Rahu + 180°):

* Panchanga at the draw moment: tithi = floor((Moon - Sun) / 12°) + 1 (1..30);
  nakshatra = floor(Moon / 13°20') + 1 (1..27) and its pada (1..4);
  yoga = floor((Sun + Moon) / 13°20') + 1 (1..27);
  karana = half-tithi 1..60 (Kimstughna, 7 movable karanas x 8, Shakuni, Chatushpada,
  Naga); vara = weekday counted from sunrise (Sunday = 1 ... Saturday = 7).
* Lagna (whole-sign ascendant) for the draw and for the birth.
* Vimshottari dasha from the natal Moon (Brihat Parashara Hora Shastra ch. 46):
  lords Ketu 7, Venus 20, Sun 6, Moon 10, Mars 7, Rahu 18, Jupiter 16, Saturn 19,
  Mercury 17 years (120 in all), starting from the lord of the Moon's nakshatra with the
  unelapsed fraction; antar- and pratyantar-dashas are proportional. A year is taken as
  365.25 days (conventions differ: 365.2425, 360).
* Planet numbers of Indian numerology: Sun 1, Moon 2, Jupiter 3, Rahu 4, Mercury 5,
  Venus 6, Ketu 7, Saturn 8, Mars 9.

Experimental mapping: the sequence [tithi, nakshatra, yoga, karana, vara, pada, draw
lagna, natal Moon nakshatra, natal lagna, maha/antar/pratyantar dasha lord numbers,
the sign of each graha, the nakshatra of each graha] reduced into the lottery range.
"""
from __future__ import annotations

from datetime import timedelta

import swisseph as swe

from . import ephemeris as eph
from .common import Component, Event, Method, MethodNumbers, Subject

NAKSHATRA_SPAN = 360.0 / 27.0
NAKSHATRAS = ["Ashwini", "Bharani", "Krittika", "Rohini", "Mrigashira", "Ardra", "Punarvasu", "Pushya",
              "Ashlesha", "Magha", "Purva Phalguni", "Uttara Phalguni", "Hasta", "Chitra", "Swati", "Vishakha",
              "Anuradha", "Jyeshtha", "Mula", "Purva Ashadha", "Uttara Ashadha", "Shravana", "Dhanishta",
              "Shatabhisha", "Purva Bhadrapada", "Uttara Bhadrapada", "Revati"]
YOGAS = ["Vishkambha", "Priti", "Ayushman", "Saubhagya", "Shobhana", "Atiganda", "Sukarma", "Dhriti", "Shula",
         "Ganda", "Vriddhi", "Dhruva", "Vyaghata", "Harshana", "Vajra", "Siddhi", "Vyatipata", "Variyana",
         "Parigha", "Shiva", "Siddha", "Sadhya", "Shubha", "Shukla", "Brahma", "Indra", "Vaidhriti"]
TITHI_NAMES = ["Pratipada", "Dwitiya", "Tritiya", "Chaturthi", "Panchami", "Shashthi", "Saptami", "Ashtami",
               "Navami", "Dashami", "Ekadashi", "Dwadashi", "Trayodashi", "Chaturdashi"]
MOVABLE_KARANAS = ["Bava", "Balava", "Kaulava", "Taitila", "Garaja", "Vanija", "Vishti"]
RASHIS = ["Mesha", "Vrishabha", "Mithuna", "Karka", "Simha", "Kanya", "Tula", "Vrishchika", "Dhanu", "Makara",
          "Kumbha", "Meena"]
VARAS = ["Ravivara (Sun)", "Somavara (Moon)", "Mangalavara (Mars)", "Budhavara (Mercury)",
         "Guruvara (Jupiter)", "Shukravara (Venus)", "Shanivara (Saturn)"]
DASHA_LORDS = ["Ketu", "Venus", "Sun", "Moon", "Mars", "Rahu", "Jupiter", "Saturn", "Mercury"]
DASHA_YEARS = {"Ketu": 7, "Venus": 20, "Sun": 6, "Moon": 10, "Mars": 7, "Rahu": 18, "Jupiter": 16,
               "Saturn": 19, "Mercury": 17}
PLANET_NUMBER = {"Sun": 1, "Moon": 2, "Jupiter": 3, "Rahu": 4, "Mercury": 5, "Venus": 6, "Ketu": 7,
                 "Saturn": 8, "Mars": 9}
GRAHAS = {"Sun": swe.SUN, "Moon": swe.MOON, "Mars": swe.MARS, "Mercury": swe.MERCURY, "Jupiter": swe.JUPITER,
          "Venus": swe.VENUS, "Saturn": swe.SATURN}
YEAR_DAYS = 365.25


def nakshatra(lon: float) -> tuple[int, int, float]:
    """(nakshatra 1..27, pada 1..4, fraction of the nakshatra elapsed)."""
    idx = int(lon // NAKSHATRA_SPAN) % 27
    frac = (lon % NAKSHATRA_SPAN) / NAKSHATRA_SPAN
    return idx + 1, int(frac * 4) + 1, frac


def tithi(sun: float, moon: float) -> int:
    return int(((moon - sun) % 360.0) // 12.0) + 1


def tithi_name(t: int) -> str:
    paksha = "Shukla" if t <= 15 else "Krishna"
    n = (t - 1) % 15
    name = TITHI_NAMES[n] if n < 14 else ("Purnima" if t == 15 else "Amavasya")
    return f"{paksha} {name}"


def yoga(sun: float, moon: float) -> int:
    return int(((sun + moon) % 360.0) // NAKSHATRA_SPAN) + 1


def karana(sun: float, moon: float) -> tuple[int, str]:
    k = int(((moon - sun) % 360.0) // 6.0)  # 0..59
    if k == 0:
        name = "Kimstughna"
    elif k <= 56:
        name = MOVABLE_KARANAS[(k - 1) % 7]
    else:
        name = ["Shakuni", "Chatushpada", "Naga"][k - 57]
    return k + 1, name


def grahas(jd: float) -> dict[str, float]:
    pos = {name: eph.longitude(jd, body, sidereal=True)[0] for name, body in GRAHAS.items()}
    pos["Rahu"] = eph.mean_node(jd, sidereal=True)
    pos["Ketu"] = (pos["Rahu"] + 180.0) % 360.0
    return pos


def vimshottari(moon_lon: float, birth_jd: float, at_jd: float) -> dict:
    """Maha-, antar- and pratyantar-dasha running at ``at_jd``."""
    nak, _, frac = nakshatra(moon_lon)
    first = (nak - 1) % 9
    lords = [DASHA_LORDS[(first + i) % 9] for i in range(9)]
    # The first mahadasha (virtually) began before birth by the elapsed fraction.
    start = birth_jd - frac * DASHA_YEARS[lords[0]] * YEAR_DAYS
    elapsed_years = (at_jd - start) / YEAR_DAYS
    levels, seq_start, span = [], 0.0, 120.0
    lord_list = lords
    for level in ("maha", "antar", "pratyantar"):
        t = 0.0
        for i in range(9):
            lord = lord_list[i]
            length = span * DASHA_YEARS[lord] / 120.0
            if elapsed_years < t + length or i == 8:
                levels.append({"level": level, "lord": lord, "number": PLANET_NUMBER[lord],
                               "start_years_from_cycle": t + seq_start, "length_years": length})
                elapsed_years -= t
                seq_start += t
                span = length
                k = DASHA_LORDS.index(lord)
                lord_list = [DASHA_LORDS[(k + j) % 9] for j in range(9)]
                break
            t += length
    return {"birth_nakshatra_lord": lords[0], "balance_at_birth_years": (1 - frac) * DASHA_YEARS[lords[0]],
            "periods": levels}


def vara_number(event: Event) -> tuple[int, float]:
    """Vedic weekday (Sunday = 1): the civil weekday, or the previous one before sunrise."""
    jd = eph.julian_day_ut(event.local)
    rise = eph.sunrise_before(jd, event.place.lat, event.place.lon)
    rise_local = eph.jd_to_utc(rise).astimezone(event.local.tzinfo)
    day = event.local.date() if rise_local.date() == event.local.date() else event.local.date() - timedelta(days=1)
    return (day.isoweekday() % 7) + 1, rise


class VedicAstrologyMethod(Method):
    key = "vedic_astrology"
    label = "Vedska astrologija (Jyotish, Panchanga, Vimshottari)"
    kind = "traditional quantities + experimental mapping"
    sources = (
        "Brihat Parashara Hora Shastra (tr. R. Santhanam, 1984), ch. 46 – Vimshottari dasha",
        "Surya Siddhanta / standard Panchanga definitions of tithi, nakshatra, yoga, karana, vara",
        "Swiss Ephemeris SIDM_LAHIRI (Lahiri/Chitrapaksha ayanamsa, Indian Calendar Reform Committee 1955)",
        "Planet numbers of Indian numerology (Sun 1 … Mars 9), e.g. Cheiro, Book of Numbers, 1926",
    )

    def numbers(self, subject: Subject, event: Event) -> MethodNumbers:
        jd_birth = eph.julian_day_ut(subject.birth_local)
        jd_draw = eph.julian_day_ut(event.local)
        natal, now = grahas(jd_birth), grahas(jd_draw)
        lagna_draw = eph.houses(jd_draw, event.place.lat, event.place.lon, "W", sidereal=True)["asc"]
        lagna_birth = eph.houses(jd_birth, subject.place.lat, subject.place.lon, "W", sidereal=True)["asc"]

        t = tithi(now["Sun"], now["Moon"])
        nak, pada, _ = nakshatra(now["Moon"])
        y = yoga(now["Sun"], now["Moon"])
        k_idx, k_name = karana(now["Sun"], now["Moon"])
        v, sunrise_jd = vara_number(event)
        natal_nak, natal_pada, _ = nakshatra(natal["Moon"])
        dasha = vimshottari(natal["Moon"], jd_birth, jd_draw)
        lagna_sign_draw = int(lagna_draw // 30) + 1
        lagna_sign_birth = int(lagna_birth // 30) + 1

        order = ["Sun", "Moon", "Mars", "Mercury", "Jupiter", "Venus", "Saturn", "Rahu", "Ketu"]
        signs_now = [int(now[g] // 30) + 1 for g in order]
        naks_now = [nakshatra(now[g])[0] for g in order]
        sequence = [t, nak, y, k_idx, v, pada, lagna_sign_draw, natal_nak, lagna_sign_birth] + \
                   [p["number"] for p in dasha["periods"]] + signs_now + naks_now

        def table(pos: dict[str, float]) -> dict:
            return {g: {"sidereal_lon": pos[g], "rashi": RASHIS[int(pos[g] // 30)],
                        "nakshatra": NAKSHATRAS[nakshatra(pos[g])[0] - 1], "pada": nakshatra(pos[g])[1]}
                    for g in order}

        intermediate = {
            "ayanamsa_lahiri": {"birth": eph.ayanamsa_lahiri(jd_birth), "draw": eph.ayanamsa_lahiri(jd_draw)},
            "panchanga_draw": {
                "tithi": t, "tithi_name": tithi_name(t), "nakshatra": nak, "nakshatra_name": NAKSHATRAS[nak - 1],
                "pada": pada, "yoga": y, "yoga_name": YOGAS[y - 1], "karana_index": k_idx, "karana_name": k_name,
                "vara": v, "vara_name": VARAS[v - 1], "sunrise_utc": eph.jd_to_utc(sunrise_jd).isoformat(),
            },
            "lagna": {"draw": lagna_draw, "draw_rashi": RASHIS[lagna_sign_draw - 1],
                      "birth": lagna_birth, "birth_rashi": RASHIS[lagna_sign_birth - 1]},
            "natal_grahas": table(natal), "draw_grahas": table(now),
            "natal_moon": {"nakshatra": natal_nak, "nakshatra_name": NAKSHATRAS[natal_nak - 1], "pada": natal_pada},
            "vimshottari_at_draw": dasha,
        }
        return MethodNumbers(
            sequence=sequence,
            digit_sequence=[t, nak, y, k_idx, v, pada],
            intermediate=intermediate,
            components=[Component("sidereal_positions_lahiri", "ran"), Component("panchanga", "ran"),
                        Component("whole_sign_lagna", "ran"), Component("vimshottari_dasha", "ran",
                                                                         "maha/antar/pratyantar")],
        )
