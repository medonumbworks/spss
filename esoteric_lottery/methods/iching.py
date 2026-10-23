"""I Ching hexagram generation by a simulated yarrow-stalk cast.

Traditional algorithm (Xici zhuan / "Great Commentary" I.9, 大衍之數五十，其用四十有九;
procedure as described in R. Wilhelm, *I Ging*, 1924, English ed. 1950, "On Consulting
the Oracle"):

* 50 stalks, one set aside, 49 used. Each line takes three "changes": split the bundle
  into two heaps, take one stalk from the right heap, count both heaps off by fours
  (a remainder of 0 counts as 4) and remove the stalks taken. After three changes the
  remaining bundle divided by 4 is 6, 7, 8 or 9.
* 6 = old yin (changing), 7 = young yang, 8 = young yin, 9 = old yang (changing).
  With equally likely remainders this gives P(6, 7, 8, 9) = (1, 5, 7, 3)/16; the
  uniform splits simulated here give exactly (0.052, 0.289, 0.448, 0.211) (tested).
* Lines are built bottom to top; changing lines flip to give the relating hexagram.

Experimental part: the "random" heap splits come from a numpy PCG64 generator seeded
by SHA-256(method, birth moment, draw moment, global seed), so the cast is
reproducible. Each split is uniform over the allowed sizes. Mapping: the King Wen
numbers of the primary and relating hexagrams of ``casts`` successive casts, reduced
into the lottery range; the six line values (6-9) of the first cast are the Joker digits.

King Wen sequence: Wilhelm/Baynes, *The I Ching or Book of Changes* (Princeton, 1950),
key for identifying the hexagrams.
"""
from __future__ import annotations

from .common import Component, Event, Method, MethodNumbers, Subject, seeded_rng

# Trigrams with their Earlier Heaven (先天) numbers, lines bottom -> top (1 = yang).
TRIGRAMS = {
    "Qian": {"number": 1, "char": "乾", "lines": (1, 1, 1), "element": "metal", "image": "heaven"},
    "Dui": {"number": 2, "char": "兌", "lines": (1, 1, 0), "element": "metal", "image": "lake"},
    "Li": {"number": 3, "char": "離", "lines": (1, 0, 1), "element": "fire", "image": "fire"},
    "Zhen": {"number": 4, "char": "震", "lines": (1, 0, 0), "element": "wood", "image": "thunder"},
    "Xun": {"number": 5, "char": "巽", "lines": (0, 1, 1), "element": "wood", "image": "wind"},
    "Kan": {"number": 6, "char": "坎", "lines": (0, 1, 0), "element": "water", "image": "water"},
    "Gen": {"number": 7, "char": "艮", "lines": (0, 0, 1), "element": "earth", "image": "mountain"},
    "Kun": {"number": 8, "char": "坤", "lines": (0, 0, 0), "element": "earth", "image": "earth"},
}
TRIGRAM_BY_NUMBER = {v["number"]: k for k, v in TRIGRAMS.items()}
TRIGRAM_BY_LINES = {v["lines"]: k for k, v in TRIGRAMS.items()}

# King Wen number by (upper, lower) trigram.
_KW_ORDER = ["Qian", "Zhen", "Kan", "Gen", "Kun", "Xun", "Li", "Dui"]
_KW_TABLE = [  # rows: upper trigram, columns: lower trigram, both in _KW_ORDER
    [1, 25, 6, 33, 12, 44, 13, 10],
    [34, 51, 40, 62, 16, 32, 55, 54],
    [5, 3, 29, 39, 8, 48, 63, 60],
    [26, 27, 4, 52, 23, 18, 22, 41],
    [11, 24, 7, 15, 2, 46, 36, 19],
    [9, 42, 59, 53, 20, 57, 37, 61],
    [14, 21, 64, 56, 35, 50, 30, 38],
    [43, 17, 47, 31, 45, 28, 49, 58],
]
KING_WEN = {(u, l): _KW_TABLE[i][j] for i, u in enumerate(_KW_ORDER) for j, l in enumerate(_KW_ORDER)}

YARROW_LINE = {6: "old yin (changing)", 7: "young yang", 8: "young yin", 9: "old yang (changing)"}


def hexagram_from_trigrams(upper: str, lower: str) -> tuple[int, ...]:
    return TRIGRAMS[lower]["lines"] + TRIGRAMS[upper]["lines"]


def trigrams_of(lines: tuple[int, ...]) -> tuple[str, str]:
    """(upper, lower) trigram names of a hexagram given bottom -> top."""
    return TRIGRAM_BY_LINES[tuple(lines[3:6])], TRIGRAM_BY_LINES[tuple(lines[0:3])]


def king_wen_number(lines: tuple[int, ...]) -> int:
    upper, lower = trigrams_of(lines)
    return KING_WEN[(upper, lower)]


def mutual(lines: tuple[int, ...]) -> tuple[int, ...]:
    """Nuclear / mutual hexagram (互卦): lines 2-4 below, 3-5 above."""
    return tuple(lines[1:4]) + tuple(lines[2:5])


def change(lines: tuple[int, ...], positions: list[int]) -> tuple[int, ...]:
    """Flip the given 1-based line positions."""
    out = list(lines)
    for p in positions:
        out[p - 1] = 1 - out[p - 1]
    return tuple(out)


def describe(lines: tuple[int, ...]) -> dict:
    upper, lower = trigrams_of(lines)
    return {"king_wen": king_wen_number(lines), "upper": upper, "lower": lower,
            "upper_char": TRIGRAMS[upper]["char"], "lower_char": TRIGRAMS[lower]["char"],
            "lines_bottom_to_top": list(lines)}


def _remainder(n: int) -> int:
    r = n % 4
    return 4 if r == 0 else r


def yarrow_line(rng) -> tuple[int, list[dict]]:
    """One line by three changes of the yarrow stalks."""
    bundle, steps = 49, []
    for _ in range(3):
        left = int(rng.integers(1, bundle - 1))  # 1 .. bundle-2, so the right heap keeps >= 2
        right = bundle - left - 1                  # one stalk from the right heap between the fingers
        removed = 1 + _remainder(left) + _remainder(right)
        steps.append({"bundle": bundle, "left": left, "right": right + 1, "removed": removed})
        bundle -= removed
    return bundle // 4, steps


def cast_hexagram(rng) -> dict:
    values, details = [], []
    for _ in range(6):
        v, steps = yarrow_line(rng)
        values.append(v)
        details.append(steps)
    primary = tuple(1 if v in (7, 9) else 0 for v in values)
    moving = [i + 1 for i, v in enumerate(values) if v in (6, 9)]
    relating = change(primary, moving)
    return {"line_values": values, "moving_lines": moving, "primary": describe(primary),
            "relating": describe(relating) if moving else None, "yarrow_steps": details}


class IChingMethod(Method):
    key = "iching"
    label = "I Ching (simulirano bacanje stabljika stolisnika)"
    kind = "traditional procedure, pseudo-random simulation (experimental)"
    sources = (
        "Xici zhuan (Great Commentary) I.9 – yarrow-stalk procedure (大衍之數)",
        "R. Wilhelm / C. F. Baynes, The I Ching or Book of Changes, Princeton 1950 – consulting the oracle, King Wen key",
        "numpy PCG64 generator seeded with SHA-256 (reproducible stand-in for manual stalk division)",
    )

    def numbers(self, subject: Subject, event: Event) -> MethodNumbers:
        rng = seeded_rng(self.key, subject.birth_local.isoformat(), event.local.isoformat(), self.seed)
        casts = [cast_hexagram(rng) for _ in range(int(self.params.get("casts", 12)))]
        sequence = []
        for c in casts:
            sequence.append(c["primary"]["king_wen"])
            if c["relating"]:
                sequence.append(c["relating"]["king_wen"])
        compact = [{k: v for k, v in c.items() if k != "yarrow_steps"} for c in casts]
        intermediate = {"casts": compact, "first_cast_yarrow_steps": casts[0]["yarrow_steps"]}
        return MethodNumbers(
            sequence=sequence,
            digit_sequence=casts[0]["line_values"],
            intermediate=intermediate,
            components=[Component("yarrow_stalk_cast", "ran", f"{len(casts)} hexagrams cast"),
                        Component("king_wen_lookup", "ran")],
        )
