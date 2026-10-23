"""Western (tropical) astrology: natal chart, transit chart and transit-to-natal aspects.

Astronomy (standard): tropical geocentric longitudes of the Sun, Moon and planets
Mercury-Pluto from the Swiss Ephemeris; Placidus houses, ascendant and midheaven for the
birth place (Zadar) and the draw place (Zagreb).

Aspects (Ptolemaic set, Ptolemy *Tetrabiblos* I.13): conjunction 0°, sextile 60°,
square 90°, trine 120°, opposition 180°. Orbs are fixed a priori in ``config.yaml``
(conjunction/opposition 8°, trine/square 6°, sextile 4°) and never tuned on draw data.

Experimental mapping ("zodiacal density"): the zodiac is cut into N equal arcs, one per
lottery number (22 arcs of 16.36° for Sve ili Ništa, 50 of 7.2° for Eurojackpot, 12
arcs = the 12 signs for the Euro numbers). Every natal and transit point carries weight
1 plus the strength (1 - deviation/orb) of each transit-natal aspect it takes part in,
and spreads that weight with a von Mises kernel (scipy.stats.vonmises, width = half an
arc). A number's score is the kernel density at the centre of its arc. Joker digits are
the last digit of the integer longitude of the transiting Sun, Moon, Mercury, Venus,
Mars and Jupiter.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.stats import vonmises

from . import ephemeris as eph
from .common import Component, Event, Method, MethodNumbers, Pool, Subject

ASPECTS = {"conjunction": 0.0, "sextile": 60.0, "square": 90.0, "trine": 120.0, "opposition": 180.0}
DEFAULT_ORBS = {"conjunction": 8.0, "sextile": 4.0, "square": 6.0, "trine": 6.0, "opposition": 8.0}


@lru_cache(maxsize=4096)
def chart(jd: float, lat: float, lon: float, hsys: str = "P") -> dict:
    h = eph.houses(jd, lat, lon, hsys)
    points = {}
    for name, body in eph.PLANETS.items():
        lam, speed = eph.longitude(jd, body)
        points[name] = {"lon": lam, "speed": speed, "retrograde": speed < 0}
    points["ASC"] = {"lon": h["asc"], "speed": None, "retrograde": False}
    points["MC"] = {"lon": h["mc"], "speed": None, "retrograde": False}
    for p in points.values():
        num, sign, deg = eph.sign_of(p["lon"])
        p.update({"sign_number": num, "sign": sign, "degree_in_sign": deg, "house": eph.house_of(p["lon"], h["cusps"])})
    return {"points": points, "cusps": h["cusps"], "house_system": hsys}


def separation(a: float, b: float) -> float:
    d = abs(a - b) % 360.0
    return 360.0 - d if d > 180.0 else d


def find_aspects(transit: dict, natal: dict, orbs: dict) -> list[dict]:
    out = []
    for tn, tp in transit.items():
        for nn, np_ in natal.items():
            sep = separation(tp["lon"], np_["lon"])
            best = None
            for asp, angle in ASPECTS.items():
                dev = abs(sep - angle)
                if dev <= orbs[asp] and (best is None or dev < best["deviation"]):
                    best = {"transit": tn, "natal": nn, "aspect": asp, "deviation": dev,
                            "strength": 1.0 - dev / orbs[asp]}
            if best:
                out.append(best)
    out.sort(key=lambda a: (a["deviation"], a["transit"], a["natal"]))
    return out


def arc_scores(points: list[tuple[float, float]], pool: Pool) -> np.ndarray:
    """Von Mises kernel density of weighted longitudes at the centre of each of N arcs."""
    n = pool.size
    centres = np.deg2rad((np.arange(n) + 0.5) * 360.0 / n)
    sigma = 0.5 * 2 * np.pi / n
    kappa = 1.0 / sigma**2
    scores = np.zeros(n)
    for lam, w in points:
        scores += w * vonmises.pdf(centres, kappa, loc=np.deg2rad(lam))
    return scores


class WesternAstrologyMethod(Method):
    key = "western_astrology"
    label = "Zapadna (tropska) astrologija"
    kind = "standard astronomy + experimental mapping"
    sources = (
        "Swiss Ephemeris 2.10 (Astrodienst), Moshier ephemeris, Placidus houses",
        "Claudius Ptolemy, Tetrabiblos I.13 – the five aspects",
        "Orbs and the zodiacal-density mapping are experimental, fixed a priori in config.yaml",
    )

    def numbers(self, subject: Subject, event: Event) -> MethodNumbers:
        hsys = self.params.get("house_system", "P")
        orbs = {**DEFAULT_ORBS, **self.params.get("orbs", {})}
        jd_birth = eph.julian_day_ut(subject.birth_local)
        jd_draw = eph.julian_day_ut(event.local)
        natal = chart(jd_birth, subject.place.lat, subject.place.lon, hsys)
        transit = chart(jd_draw, event.place.lat, event.place.lon, hsys)
        aspects = find_aspects(transit["points"], natal["points"], orbs)

        weights = {("natal", k): 1.0 for k in natal["points"]}
        weights.update({("transit", k): 1.0 for k in transit["points"]})
        for a in aspects:
            weights[("transit", a["transit"])] += a["strength"]
            weights[("natal", a["natal"])] += a["strength"]
        weighted = [(natal["points"][k]["lon"] if side == "natal" else transit["points"][k]["lon"], w)
                    for (side, k), w in sorted(weights.items())]

        digit_bodies = ["Sun", "Moon", "Mercury", "Venus", "Mars", "Jupiter"]
        digit_seq = [int(transit["points"][b]["lon"]) for b in digit_bodies]
        sequence = [int(p["lon"]) for p in transit["points"].values()] + [int(p["lon"]) for p in natal["points"].values()]
        return MethodNumbers(
            sequence=sequence,
            digit_sequence=digit_seq,
            intermediate={"natal_chart": natal, "transit_chart": transit, "aspects_transit_to_natal": aspects,
                          "point_weights": {f"{s}:{k}": w for (s, k), w in sorted(weights.items())},
                          "jd_ut": {"birth": jd_birth, "draw": jd_draw}},
            components=[Component("natal_chart", "ran", f"Placidus ({hsys}), Moshier ephemeris"),
                        Component("transit_chart", "ran"),
                        Component("transit_to_natal_aspects", "ran", f"{len(aspects)} aspects within orb")],
            pool_scores=lambda pool: arc_scores(weighted, pool),
        )
