"""Chaldean numerology after Cheiro.

Traditional (Cheiro = W. J. Warner, *Cheiro's Book of Numbers*, 1926):

* letters 1-8 only (9 is held sacred and never assigned):
  1 A I J Q Y, 2 B K R, 3 C G L S, 4 D M T, 5 E H N X, 6 U V W, 7 O Z, 8 F P
* compound numbers (10-52) are kept alongside the single digit they reduce to;
  no master numbers
* birth number = day of birth; destiny ("fadic") number = all digits of the birth date
* planets: 1 Sun, 2 Moon, 3 Jupiter, 4 Uranus, 5 Mercury, 6 Venus, 7 Neptune,
  8 Saturn, 9 Mars; a weekday takes the number of its classical ruler
  (Sun 1, Mon 2, Tue 9, Wed 5, Thu 3, Fri 6, Sat 8)
* name number: only if ``subject.name`` is set, otherwise skipped

Experimental: applying the letter table to the place names, and mapping the ordered
sequence [draw day compound/reduced, draw date compound/reduced, draw weekday planet,
birth day compound/reduced, destiny compound/reduced, birth weekday planet, name, places]
into the lottery range.
"""
from __future__ import annotations

from datetime import date

from .common import Component, Event, Method, MethodNumbers, Subject, digit_root, normalize_latin

CHALDEAN = {}
for value, letters in {1: "AIJQY", 2: "BKR", 3: "CGLS", 4: "DMT", 5: "EHNX", 6: "UVW", 7: "OZ", 8: "FP"}.items():
    for c in letters:
        CHALDEAN[c] = value

WEEKDAY_PLANET = {7: ("Sun", 1), 1: ("Moon", 2), 2: ("Mars", 9), 3: ("Mercury", 5), 4: ("Jupiter", 3),
                  5: ("Venus", 6), 6: ("Saturn", 8)}  # key: isoweekday


def chaldean_value(text: str) -> int:
    return sum(CHALDEAN[c] for c in normalize_latin(text))


def full_date_compound(d: date) -> int:
    return sum(int(c) for c in d.strftime("%d%m%Y"))


def date_block(d: date) -> dict:
    compound = full_date_compound(d)
    planet, pnum = WEEKDAY_PLANET[d.isoweekday()]
    return {"day_compound": d.day, "day_reduced": digit_root(d.day), "date_compound": compound,
            "date_reduced": digit_root(compound), "weekday_planet": planet, "weekday_planet_number": pnum}


class ChaldeanMethod(Method):
    key = "chaldean"
    label = "Kaldejska numerologija (Cheiro)"
    kind = "traditional (Cheiro) + experimental mapping"
    sources = (
        "Cheiro (W. J. Warner), Cheiro's Book of Numbers, London 1926 – letter table 1-8, compound numbers, planets",
        "Place-name values and the lottery mapping are experimental",
    )

    def numbers(self, subject: Subject, event: Event) -> MethodNumbers:
        draw, birth = date_block(event.local.date()), date_block(subject.birth_local.date())
        sequence = [draw["day_compound"], draw["day_reduced"], draw["date_compound"], draw["date_reduced"],
                    draw["weekday_planet_number"], birth["day_compound"], birth["day_reduced"],
                    birth["date_compound"], birth["date_reduced"], birth["weekday_planet_number"]]
        components = [Component("draw_date_numbers", "ran"), Component("birth_date_numbers", "ran")]
        intermediate = {"draw": draw, "birth": birth}
        if subject.name:
            v = chaldean_value(subject.name)
            intermediate["name"] = {"compound": v, "reduced": digit_root(v)}
            sequence += [v, digit_root(v)]
            components.append(Component("name_number", "ran"))
        else:
            components.append(Component("name_number", "skipped", "no name in config.yaml (subject.name is null)"))
        places = {}
        for label, text in (("birth_place", subject.place.name), ("draw_place", event.place.name)):
            v = chaldean_value(text)
            places[label] = {"text": text, "compound": v, "reduced": digit_root(v)}
            sequence += [v, digit_root(v)]
        intermediate["places"] = places
        components.append(Component("place_name_numbers", "ran", "experimental"))
        return MethodNumbers(
            sequence=sequence,
            digit_sequence=[birth["day_reduced"], birth["date_reduced"], draw["day_reduced"], draw["date_reduced"],
                            draw["weekday_planet_number"], birth["day_compound"]],
            intermediate=intermediate,
            components=components,
        )
