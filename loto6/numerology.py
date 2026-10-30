"""Numeroloske tablice: kaldejska, hebrejska gematrija (standardne vrijednosti, bez sofita), pitagorejska."""
from __future__ import annotations
import datetime as dt

CHALDEAN = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 8, "G": 3, "H": 5, "I": 1, "J": 1, "K": 2, "L": 3, "M": 4,
            "N": 5, "O": 7, "P": 8, "Q": 1, "R": 2, "S": 3, "T": 4, "U": 6, "V": 6, "W": 6, "X": 5, "Y": 1, "Z": 7}

PYTHAGOREAN = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "H": 8, "I": 9, "J": 1, "K": 2, "L": 3, "M": 4,
               "N": 5, "O": 6, "P": 7, "Q": 8, "R": 9, "S": 1, "T": 2, "U": 3, "V": 4, "W": 5, "X": 6, "Y": 7, "Z": 8}

HEBREW_VALUES = {"א": 1, "ב": 2, "ג": 3, "ד": 4, "ה": 5, "ו": 6, "ז": 7, "ח": 8, "ט": 9, "י": 10, "כ": 20, "ך": 20,
                 "ל": 30, "מ": 40, "ם": 40, "נ": 50, "ן": 50, "ס": 60, "ע": 70, "פ": 80, "ף": 80, "צ": 90, "ץ": 90,
                 "ק": 100, "ר": 200, "ש": 300, "ת": 400}

GEMATRIA_WORDS = [("מזל", 77), ("ברכה", 227), ("שפע", 450), ("זהב", 14), ("חי", 18), ("הצלחה", 138), ("כסף", 160),
                  ("עושר", 576)]


def _letters(word: str, table: dict[str, int]) -> list[tuple[str, int]]:
    return [(ch, table[ch]) for ch in word.upper() if ch in table]


def chaldean_letters(word: str) -> list[tuple[str, int]]:
    return _letters(word, CHALDEAN)


def chaldean_word(word: str) -> int:
    return sum(v for _, v in chaldean_letters(word))


def pythagorean_letters(word: str) -> list[tuple[str, int]]:
    return _letters(word, PYTHAGOREAN)


def pythagorean_word(word: str) -> int:
    return sum(v for _, v in pythagorean_letters(word))


def hebrew_letters(word: str) -> list[tuple[str, int]]:
    return [(ch, HEBREW_VALUES[ch]) for ch in word if ch in HEBREW_VALUES]


def hebrew_word(word: str) -> int:
    return sum(v for _, v in hebrew_letters(word))


def digit_sum(n: int) -> int:
    return sum(int(c) for c in str(abs(int(n))))


def reduce_le45(n: int) -> tuple[int, int]:
    """Ponavlja zbroj znamenki dok vrijednost nije <= 45; vraca (vrijednost, broj koraka)."""
    steps = 0
    n = int(n)
    while n > 45:
        n = digit_sum(n)
        steps += 1
    return n, steps


def reduce_single(n: int) -> int:
    n = abs(int(n))
    while n > 9:
        n = digit_sum(n)
    return n


def date_digit_sum(d: dt.date) -> int:
    return digit_sum(int(d.strftime("%d%m%Y")))


def selftest() -> None:
    for w, v in GEMATRIA_WORDS:
        got = hebrew_word(w)
        assert got == v, f"gematrija {w}: izracunato {got}, zadano {v}"
    assert pythagorean_word("IVAN") == 19
    assert digit_sum(2026) == 10
    assert date_digit_sum(dt.date(2026, 11, 1)) == 13
    assert reduce_le45(46) == (10, 1)
    assert reduce_le45(2062) == (10, 1)
    assert reduce_le45(29) == (29, 0)
    assert reduce_single(29) == 2
    assert chaldean_word("IVAN") == 1 + 6 + 1 + 5
    for w in ["LOTO", "HRVATSKA LUTRIJA", "ZAGREB", "IVAN", "NEDJELJA", "SRIJEDA"]:
        print(f"  kaldejski {w} = {chaldean_word(w)}  {chaldean_letters(w)}")
    print("numerology selftest OK")


if __name__ == "__main__":
    selftest()
