"""Hermetic and Kabbalistic numbers: planetary magic squares, sephirot, divine names.

Traditional material, all of it attested:

* **Planetary kameas** (magic squares) exactly as tabulated by H. C. Agrippa,
  *De occulta philosophia libri tres* (1533), II.22: Saturn 3x3 (1-9, constant 15),
  Jupiter 4x4 (1-16, 34), Mars 5x5 (1-25, 65), Sol 6x6 (1-36, 111), Venus 7x7 (1-49,
  175), Mercury 8x8 (1-64, 260), Luna 9x9 (1-81, 369). Every row, column and both
  diagonals of each square sum to its constant; ``tests/test_hermetic.py`` checks all
  seven, which is also a transcription check.
* **Planetary ruler of the moment**: the day's ruler is the classical one (Sunday Sol …
  Saturday Saturn), and the *planetary hours* are the unequal hours — daylight split
  into twelve from sunrise to sunset, night into twelve from sunset to sunrise — with
  the first hour of the day ruled by the day's ruler and the rest following the
  Chaldean order Saturn, Jupiter, Mars, Sol, Venus, Mercury, Luna (Ptolemy,
  *Tetrabiblos*; Agrippa II.34).
* **Sephirot**: the ten of the Tree of Life, numbered 1 Kether … 10 Malkuth, with the
  usual planetary attributions (3 Binah Saturn, 4 Chesed Jupiter, 5 Geburah Mars,
  6 Tiphareth Sol, 7 Netzach Venus, 8 Hod Mercury, 9 Yesod Luna). The thirty-two paths
  of wisdom are the ten sephirot plus the twenty-two Hebrew letters (*Sefer Yetzirah*
  1:1).
* **Divine names** in their standard spelling, valued with the same mispar hechrachi
  used by the gematria module, so the two agree by construction: אהיה 21, יה 15,
  אל 31, אלהים 86, יהוה 26, שדי 314, אדני 65.

Experimental: which numbers of the chosen kamea are read (the row, the column and the
diagonals through the cell holding the sephira's number), and the mapping into the
lottery range. No tradition assigns lottery numbers.
"""
from __future__ import annotations

import swisseph as swe

from . import ephemeris as eph
from .common import Component, Event, Method, MethodNumbers, Subject
from .gematria import hebrew_value

# Agrippa, De occulta philosophia II.22.
KAMEAS: dict[str, list[list[int]]] = {
    "Saturn": [[4, 9, 2], [3, 5, 7], [8, 1, 6]],
    "Jupiter": [[4, 14, 15, 1], [9, 7, 6, 12], [5, 11, 10, 8], [16, 2, 3, 13]],
    "Mars": [[11, 24, 7, 20, 3], [4, 12, 25, 8, 16], [17, 5, 13, 21, 9],
             [10, 18, 1, 14, 22], [23, 6, 19, 2, 15]],
    "Sol": [[6, 32, 3, 34, 35, 1], [7, 11, 27, 28, 8, 30], [19, 14, 16, 15, 23, 24],
            [18, 20, 22, 21, 17, 13], [25, 29, 10, 9, 26, 12], [36, 5, 33, 4, 2, 31]],
    "Venus": [[22, 47, 16, 41, 10, 35, 4], [5, 23, 48, 17, 42, 11, 29],
              [30, 6, 24, 49, 18, 36, 12], [13, 31, 7, 25, 43, 19, 37],
              [38, 14, 32, 1, 26, 44, 20], [21, 39, 8, 33, 2, 27, 45],
              [46, 15, 40, 9, 34, 3, 28]],
    "Mercury": [[8, 58, 59, 5, 4, 62, 63, 1], [49, 15, 14, 52, 53, 11, 10, 56],
                [41, 23, 22, 44, 45, 19, 18, 48], [32, 34, 35, 29, 28, 38, 39, 25],
                [40, 26, 27, 37, 36, 30, 31, 33], [17, 47, 46, 20, 21, 43, 42, 24],
                [9, 55, 54, 12, 13, 51, 50, 16], [64, 2, 3, 61, 60, 6, 7, 57]],
    "Luna": [[37, 78, 29, 70, 21, 62, 13, 54, 5], [6, 38, 79, 30, 71, 22, 63, 14, 46],
             [47, 7, 39, 80, 31, 72, 23, 55, 15], [16, 48, 8, 40, 81, 32, 64, 24, 56],
             [57, 17, 49, 9, 41, 73, 33, 65, 25], [26, 58, 18, 50, 1, 42, 74, 34, 66],
             [67, 27, 59, 10, 51, 2, 43, 75, 35], [36, 68, 19, 60, 11, 52, 3, 44, 76],
             [77, 28, 69, 20, 61, 12, 53, 4, 45]],
}
MAGIC_CONSTANT = {"Saturn": 15, "Jupiter": 34, "Mars": 65, "Sol": 111, "Venus": 175,
                  "Mercury": 260, "Luna": 369}
# Descending apparent speed: the order the planetary hours run in.
CHALDEAN = ["Saturn", "Jupiter", "Mars", "Sol", "Venus", "Mercury", "Luna"]
# isoweekday 1 = Monday.
DAY_RULER = {1: "Luna", 2: "Mars", 3: "Mercury", 4: "Jupiter", 5: "Venus", 6: "Saturn", 7: "Sol"}
SEPHIROT = [(1, "Kether", None), (2, "Chokmah", None), (3, "Binah", "Saturn"), (4, "Chesed", "Jupiter"),
            (5, "Geburah", "Mars"), (6, "Tiphareth", "Sol"), (7, "Netzach", "Venus"),
            (8, "Hod", "Mercury"), (9, "Yesod", "Luna"), (10, "Malkuth", None)]
SEPHIRA_OF = {planet: n for n, _, planet in SEPHIROT if planet}
SEPHIRA_NAME = {n: name for n, name, _ in SEPHIROT}
DIVINE_NAMES = {"Kether": "אהיה", "Chokmah": "יה", "Chesed": "אל", "Binah": "אלהים",
                "Tiphareth": "יהוה", "Yesod": "שדי", "Malkuth": "אדני"}
PATHS_OF_WISDOM = 32   # Sefer Yetzirah 1:1: ten sephirot and twenty-two letters


def planetary_hour(event: Event) -> dict:
    """Which unequal hour of which planet the moment falls in."""
    jd = eph.julian_day_ut(event.local)
    lat, lon = event.place.lat, event.place.lon
    rise = eph.sunrise_before(jd, lat, lon)
    set_res, set_t = swe.rise_trans(rise, swe.SUN, swe.CALC_SET, (lon, lat, 0.0), 1013.25, 10.0,
                                    swe.FLG_MOSEPH)
    next_res, next_rise = swe.rise_trans(rise + 0.5, swe.SUN, swe.CALC_RISE, (lon, lat, 0.0), 1013.25,
                                         10.0, swe.FLG_MOSEPH)
    if set_res != 0 or next_res != 0:
        raise RuntimeError("no sunset or next sunrise found")
    sunset, next_sunrise = set_t[0], next_rise[0]
    if jd < sunset:                                   # daytime
        length, index, part = (sunset - rise) / 12.0, int((jd - rise) / ((sunset - rise) / 12.0)), "day"
    else:                                             # night
        length = (next_sunrise - sunset) / 12.0
        index, part = 12 + int((jd - sunset) / length), "night"
    index = min(index, 23)
    # The civil day whose ruler starts the sequence begins at that sunrise.
    day = eph.jd_to_utc(rise).astimezone(event.local.tzinfo).date()
    day_ruler = DAY_RULER[day.isoweekday()]
    start = CHALDEAN.index(day_ruler)
    ruler = CHALDEAN[(start + index) % 7]
    return {"day_of_sunrise": day.isoformat(), "day_ruler": day_ruler, "hour_index": index + 1,
            "part": part, "hour_ruler": ruler, "hour_length_minutes": length * 24 * 60,
            "sunrise_utc": eph.jd_to_utc(rise).isoformat(),
            "sunset_utc": eph.jd_to_utc(sunset).isoformat()}


def cell_of(kamea: list[list[int]], value: int) -> tuple[int, int] | None:
    for r, row in enumerate(kamea):
        for c, v in enumerate(row):
            if v == value:
                return r, c
    return None


def lines_through(kamea: list[list[int]], cell: tuple[int, int]) -> dict:
    """The row, the column and whichever diagonals pass through a cell."""
    r, c = cell
    n = len(kamea)
    out = {"row": list(kamea[r]), "column": [kamea[i][c] for i in range(n)]}
    if r == c:
        out["diagonal"] = [kamea[i][i] for i in range(n)]
    if r + c == n - 1:
        out["antidiagonal"] = [kamea[i][n - 1 - i] for i in range(n)]
    return out


def read_kamea(planet: str, seed_value: int) -> dict:
    """The lines of a planet's square through the cell holding ``seed_value``."""
    kamea = KAMEAS[planet]
    n = len(kamea)
    value = (seed_value - 1) % (n * n) + 1
    cell = cell_of(kamea, value)
    lines = lines_through(kamea, cell)
    return {"planet": planet, "order": n, "magic_constant": MAGIC_CONSTANT[planet],
            "seed_value": value, "cell": {"row": cell[0] + 1, "column": cell[1] + 1},
            "lines": lines, "numbers": [v for line in lines.values() for v in line]}


class HermeticMethod(Method):
    key = "hermetic"
    label = "Hermetika i kabala (planetarni kamee, sefirot, božanska imena)"
    kind = "traditional correspondences + experimental mapping"
    sources = (
        "H. C. Agrippa, De occulta philosophia libri tres (1533), II.22 – the seven planetary kameas; "
        "II.34 – the planetary hours",
        "Ptolemy, Tetrabiblos – the Chaldean order of the planets and the rulers of the days",
        "Sefer Yetzirah 1:1 – the thirty-two paths of wisdom (ten sephirot and twenty-two letters)",
        "Standard sephirotic planetary attributions (Kircher tree, as used by the Golden Dawn)",
        "Divine names valued with mispar hechrachi, the same cipher as methods/gematria.py",
    )

    def numbers(self, subject: Subject, event: Event) -> MethodNumbers:
        components = []
        hour = planetary_hour(event)
        components.append(Component("planetary_hour", "ran",
                                    f"{hour['hour_ruler']} hour {hour['hour_index']} ({hour['part']})"))
        # The birth moment's ruler, for the querent's own planet.
        birth_event = Event(subject.birth_local, subject.place)
        birth_hour = planetary_hour(birth_event)
        components.append(Component("natal_planetary_hour", "ran", birth_hour["hour_ruler"]))

        draw_planet, birth_planet = hour["hour_ruler"], birth_hour["hour_ruler"]
        draw_sephira = SEPHIRA_OF.get(draw_planet)
        birth_sephira = SEPHIRA_OF.get(birth_planet)
        draw_kamea = read_kamea(draw_planet, draw_sephira or 1)
        birth_kamea = read_kamea(birth_planet, birth_sephira or 1)
        components.append(Component("planetary_kamea", "ran",
                                    f"{draw_planet} {draw_kamea['order']}x{draw_kamea['order']}, "
                                    f"constant {draw_kamea['magic_constant']}"))

        names = {sephira: {"name": text, "gematria": hebrew_value(text)}
                 for sephira, text in DIVINE_NAMES.items()}
        components.append(Component("divine_name_gematria", "ran", f"{len(names)} names"))

        sequence = (draw_kamea["numbers"] + [draw_kamea["magic_constant"], draw_sephira or 1]
                    + birth_kamea["numbers"] + [birth_kamea["magic_constant"], birth_sephira or 1]
                    + [PATHS_OF_WISDOM, hour["hour_index"]]
                    + [v["gematria"] for v in names.values()])
        return MethodNumbers(
            sequence=sequence,
            digit_sequence=[draw_kamea["magic_constant"], draw_sephira or 1, hour["hour_index"],
                            birth_kamea["magic_constant"], birth_sephira or 1, PATHS_OF_WISDOM],
            intermediate={
                "draw_planetary_hour": hour, "birth_planetary_hour": birth_hour,
                "draw_sephira": {"number": draw_sephira, "name": SEPHIRA_NAME.get(draw_sephira)},
                "birth_sephira": {"number": birth_sephira, "name": SEPHIRA_NAME.get(birth_sephira)},
                "draw_kamea": draw_kamea, "birth_kamea": birth_kamea,
                "divine_names": names, "paths_of_wisdom": PATHS_OF_WISDOM,
            },
            components=components,
        )


