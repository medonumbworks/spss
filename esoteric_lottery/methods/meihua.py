"""Meihua Yishu (梅花易數, "Plum Blossom numerology"), time-based casting.

Traditional algorithm (年月日時起卦, attributed to Shao Yong 邵雍, 1011-1077; text in
《梅花易數》 vol. 1, worked example 觀梅占 "watching the plum blossoms"):

* year number  = earthly branch of the lunar year (子 = 1 ... 亥 = 12)
* month, day   = lunar month and lunar day (a leap month keeps its number)
* hour number  = double-hour branch (子 = 1 ... 亥 = 12)
* upper trigram = (year + month + day) mod 8,           0 -> 8
* lower trigram = (year + month + day + hour) mod 8,    0 -> 8
* moving line   = (year + month + day + hour) mod 6,    0 -> 6
* trigram numbers follow the Earlier Heaven order 乾1 兌2 離3 震4 巽5 坎6 艮7 坤8.

From the original hexagram come the mutual (互) and changed (變) hexagrams, and the
body/use (體/用) trigrams: the trigram holding the moving line is "use".

The classic example (dragon year 辰=5, 12th month, 17th day, 申 hour = 9) gives 澤火革
(49) changing to 澤山咸 (31) with mutual trigrams 乾/巽; it is a unit test.

Choices (config ``meihua``): the hour is taken from local *true solar* time (sundial
time, the only clock the tradition knew) and the lunar date from the Chinese calendar
day of that local date; ``hour_basis: civil`` uses the clock instead.

Experimental part: casting both for the draw moment (Zagreb) and the birth moment
(Zadar) and mapping [King Wen numbers of original, changed and mutual hexagrams,
upper, lower, moving line, the two sums] for the draw, then the same for the birth,
into the lottery range.
"""
from __future__ import annotations

from datetime import datetime

from . import ephemeris as eph
from .chinese_calendar import EARTHLY_BRANCHES, hour_branch, lunar_date
from .common import Component, Event, Method, MethodNumbers, Subject
from .iching import TRIGRAMS, TRIGRAM_BY_NUMBER, change, describe, hexagram_from_trigrams, mutual

_GENERATES = {"wood": "fire", "fire": "earth", "earth": "metal", "metal": "water", "water": "wood"}
_CONTROLS = {"wood": "earth", "earth": "water", "water": "fire", "fire": "metal", "metal": "wood"}


def cast_from_numbers(year: int, month: int, day: int, hour: int) -> dict:
    s1 = year + month + day
    s2 = s1 + hour
    upper_n = s1 % 8 or 8
    lower_n = s2 % 8 or 8
    moving = s2 % 6 or 6
    upper, lower = TRIGRAM_BY_NUMBER[upper_n], TRIGRAM_BY_NUMBER[lower_n]
    original = hexagram_from_trigrams(upper, lower)
    changed = change(original, [moving])
    use, body = (lower, upper) if moving <= 3 else (upper, lower)
    return {
        "inputs": {"year": year, "month": month, "day": day, "hour": hour},
        "sum_ymd": s1, "sum_ymdh": s2,
        "upper_number": upper_n, "lower_number": lower_n, "moving_line": moving,
        "original": describe(original), "mutual": describe(mutual(original)), "changed": describe(changed),
        "body_trigram": body, "use_trigram": use,
        "body_use_relation": five_phase_relation(TRIGRAMS[body]["element"], TRIGRAMS[use]["element"]),
    }


def five_phase_relation(body: str, use: str) -> str:
    if body == use:
        return "same element (比和)"
    if _GENERATES[use] == body:
        return "use generates body (用生體)"
    if _GENERATES[body] == use:
        return "body generates use (體生用)"
    if _CONTROLS[body] == use:
        return "body controls use (體克用)"
    return "use controls body (用克體)"


def local_reference_time(moment: datetime, lon: float, basis: str) -> datetime:
    if basis == "civil":
        return moment.replace(tzinfo=None)
    if basis == "true_solar":
        return eph.true_solar_time(moment, lon)
    raise ValueError(f"unknown hour_basis {basis!r}")


def cast_for_moment(moment: datetime, lon: float, basis: str) -> dict:
    ref = local_reference_time(moment, lon, basis)
    ld = lunar_date(ref.date())
    hb = hour_branch(ref.hour + ref.minute / 60)
    cast = cast_from_numbers(ld.year_branch, ld.month, ld.day, hb)
    cast["calendar"] = {"reference_time": ref.isoformat(timespec="minutes"), "hour_basis": basis,
                        "lunar_date": ld.to_json(), "hour_branch_number": hb,
                        "hour_branch": EARTHLY_BRANCHES[hb - 1]}
    return cast


def _cast_sequence(cast: dict) -> list[int]:
    return [cast["original"]["king_wen"], cast["changed"]["king_wen"], cast["mutual"]["king_wen"],
            cast["upper_number"], cast["lower_number"], cast["moving_line"], cast["sum_ymd"], cast["sum_ymdh"]]


class MeihuaMethod(Method):
    key = "meihua"
    label = "Meihua Yishu (梅花易數, vremensko bacanje)"
    kind = "traditional casting + experimental mapping"
    sources = (
        "《梅花易數》 attributed to Shao Yong 邵雍 (Song dynasty), vol. 1: 年月日時起卦, example 觀梅占",
        "GB/T 33661-2017 Calculation and promulgation of the Chinese calendar (lunar date rules)",
        "Swiss Ephemeris (Moshier) for new moons, solar terms and the equation of time",
    )

    def numbers(self, subject: Subject, event: Event) -> MethodNumbers:
        basis = self.params.get("hour_basis", "true_solar")
        draw = cast_for_moment(event.local, event.place.lon, basis)
        birth = cast_for_moment(subject.birth_local, subject.place.lon, basis)
        return MethodNumbers(
            sequence=_cast_sequence(draw) + _cast_sequence(birth),
            digit_sequence=[draw["upper_number"], draw["lower_number"], draw["moving_line"],
                            draw["original"]["king_wen"], draw["mutual"]["king_wen"], draw["changed"]["king_wen"]],
            intermediate={"draw_cast": draw, "birth_cast": birth},
            components=[Component("chinese_lunar_calendar", "ran", "computed from Swiss Ephemeris new moons"),
                        Component("draw_time_cast", "ran"), Component("birth_time_cast", "ran")],
        )
