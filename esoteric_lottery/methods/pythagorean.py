"""Pythagorean (modern Western) numerology.

Traditional within modern numerology (the attribution to Pythagoras is legendary; the
system was shaped by L. Dow Balliett, c. 1908, and codified by J. Jordan, *Numerology:
The Romance in Your Name*, 1965, and M. O. Goodwin, *Numerology: The Complete Guide*,
1981):

* letters: 1 AJS, 2 BKT, 3 CLU, 4 DMV, 5 ENW, 6 FOX, 7 GPY, 8 HQZ, 9 IR
* reduction: repeated digit sum to 1..9, keeping the master numbers 11, 22, 33 where
  the tradition keeps them (life path, pinnacles, name numbers)
* life path = reduce(reduce(month) + reduce(day) + reduce(year))  (Goodwin's method)
* birthday = day (compound and reduced); attitude = reduce(month + day)
* pinnacles P1 = month+day, P2 = day+year, P3 = P1+P2, P4 = month+year;
  challenges C1 = |month-day|, C2 = |day-year|, C3 = |C1-C2|, C4 = |month-year|
* personal year = reduce(month + day + draw year), personal month = reduce(PY + draw
  month), personal day = reduce(PM + draw day); universal day = reduce(all digits of
  the draw date). Personal cycles are reduced to 1..9.
* name numbers (expression, soul urge, personality) need the querent's name: they are
  skipped unless ``subject.name`` is set in config.yaml.

Experimental: the numerology of the birth- and draw-place names, and the mapping of
the ordered sequence into the lottery range.
"""
from __future__ import annotations

from datetime import date

from .common import Component, Event, Method, MethodNumbers, Subject, digit_root, normalize_latin

MASTERS = (11, 22, 33)
LETTER_VALUES = {c: (i % 9) + 1 for i, c in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ")}
VOWELS = set("AEIOU")


def name_value(text: str, letters: set[str] | None = None) -> int:
    s = normalize_latin(text)
    return sum(LETTER_VALUES[c] for c in s if letters is None or (c in letters))


def date_numbers(birth: date) -> dict:
    m, d, y = digit_root(birth.month), digit_root(birth.day), digit_root(birth.year)
    m_m, d_m, y_m = (digit_root(x, MASTERS) for x in (birth.month, birth.day, birth.year))
    p1 = digit_root(m + d, MASTERS)
    p2 = digit_root(d + y, MASTERS)
    p3 = digit_root(digit_root(p1) + digit_root(p2), MASTERS)
    p4 = digit_root(m + y, MASTERS)
    c1, c2, c4 = abs(m - d), abs(d - y), abs(m - y)
    c3 = abs(c1 - c2)
    return {
        "life_path": digit_root(m_m + d_m + y_m, MASTERS),
        "life_path_components": {"month": m_m, "day": d_m, "year": y_m},
        "birthday_compound": birth.day, "birthday": digit_root(birth.day, MASTERS),
        "attitude": digit_root(birth.month + birth.day),
        "pinnacles": [p1, p2, p3, p4], "challenges": [c1, c2, c3, c4],
    }


def cycle_numbers(birth: date, on: date) -> dict:
    py = digit_root(digit_root(birth.month) + digit_root(birth.day) + digit_root(on.year))
    pm = digit_root(py + digit_root(on.month))
    pd = digit_root(pm + digit_root(on.day))
    ud = digit_root(sum(int(c) for c in on.strftime("%Y%m%d")))
    return {"personal_year": py, "personal_month": pm, "personal_day": pd, "universal_day": ud}


def name_numbers(name: str) -> dict:
    return {"expression": digit_root(name_value(name), MASTERS),
            "soul_urge": digit_root(name_value(name, VOWELS), MASTERS),
            "personality": digit_root(name_value(name, set(LETTER_VALUES) - VOWELS), MASTERS),
            "expression_compound": name_value(name)}


class PythagoreanMethod(Method):
    key = "pythagorean"
    label = "Pitagorejska numerologija"
    kind = "traditional (modern numerology) + experimental mapping"
    sources = (
        "J. Jordan, Numerology: The Romance in Your Name, 1965 – letter table, core numbers",
        "M. O. Goodwin, Numerology: The Complete Guide, 1981 – life path, pinnacles, challenges, personal cycles",
        "Place-name numbers and the lottery mapping are experimental",
    )

    def numbers(self, subject: Subject, event: Event) -> MethodNumbers:
        b, on = subject.birth_local.date(), event.local.date()
        dn, cn = date_numbers(b), cycle_numbers(b, on)
        places = {}
        for label, text in (("birth_place", subject.place.name), ("draw_place", event.place.name)):
            v = name_value(text)
            places[label] = {"text": text, "compound": v, "reduced": digit_root(v, MASTERS)}
        components = [Component("date_numbers", "ran", "life path, birthday, attitude, pinnacles, challenges"),
                      Component("personal_cycles", "ran", "personal year/month/day, universal day")]
        sequence = [cn["personal_day"], cn["universal_day"], dn["life_path"], cn["personal_month"],
                    cn["personal_year"], dn["birthday_compound"], dn["birthday"], dn["attitude"],
                    *dn["pinnacles"], *dn["challenges"]]
        intermediate = {"date_numbers": dn, "cycles_at_draw": cn, "place_numbers": places}
        if subject.name:
            nn = name_numbers(subject.name)
            intermediate["name_numbers"] = nn
            sequence += [nn["expression"], nn["soul_urge"], nn["personality"], nn["expression_compound"]]
            components.append(Component("name_numbers", "ran"))
        else:
            components.append(Component("name_numbers", "skipped", "no name in config.yaml (subject.name is null)"))
        for p in places.values():
            sequence += [p["compound"], p["reduced"]]
        components.append(Component("place_name_numbers", "ran", "experimental"))
        return MethodNumbers(
            sequence=sequence,
            digit_sequence=[dn["life_path"], cn["personal_year"], cn["personal_month"], cn["personal_day"],
                            cn["universal_day"], dn["birthday"]],
            intermediate=intermediate,
            components=components,
        )
