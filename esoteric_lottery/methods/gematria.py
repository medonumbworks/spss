"""Gematria: letter-number ciphers applied to the texts of the reading.

Traditional ciphers implemented and tested:

* Hebrew *mispar hechrachi* (standard values, final forms = their base letter):
  א1 … ט9, י10 … צ90, ק100 ר200 ש300 ת400. E.g. שלום = 376.
* Greek isopsephy (Milesian numerals incl. ϛ 6, ϙ 90, ϡ 900). E.g. Ἰησοῦς = 888.
* Latin alphanumerics in the Hebrew/Greek decimal pattern as tabulated by H. C. Agrippa,
  *De occulta philosophia* II.20 (1533): A1 … I9, K10 … S90, T100 V200 X300 Y400 Z500
  (J -> I, U -> V and W -> VV, as in the classical 23-letter alphabet).
* English ordinal (A1 … Z26) and reverse ordinal (A26 … Z1), the simple modern ciphers.

Hebrew and Greek values are only used when Hebrew/Greek text is supplied in
config.yaml (``gematria.extra_texts``); otherwise that component is reported as
skipped. Experimental: the choice of texts (Croatian weekday and month names of the
draw and of the birth, the draw and birth places, the querent's name if given) and the
mapping of [ordinal, reverse ordinal, Agrippa] of each text into the lottery range.
"""
from __future__ import annotations

import unicodedata

from .common import (CROATIAN_MONTHS, CROATIAN_WEEKDAYS, Component, Event, Method, MethodNumbers, Subject,
                     normalize_latin)

HEBREW = {"א": 1, "ב": 2, "ג": 3, "ד": 4, "ה": 5, "ו": 6, "ז": 7, "ח": 8, "ט": 9, "י": 10, "כ": 20, "ך": 20,
          "ל": 30, "מ": 40, "ם": 40, "נ": 50, "ן": 50, "ס": 60, "ע": 70, "פ": 80, "ף": 80, "צ": 90, "ץ": 90,
          "ק": 100, "ר": 200, "ש": 300, "ת": 400}
GREEK = {"α": 1, "β": 2, "γ": 3, "δ": 4, "ε": 5, "ϛ": 6, "ϝ": 6, "ζ": 7, "η": 8, "θ": 9, "ι": 10, "κ": 20,
         "λ": 30, "μ": 40, "ν": 50, "ξ": 60, "ο": 70, "π": 80, "ϙ": 90, "ϟ": 90, "ρ": 100, "σ": 200, "ς": 200,
         "τ": 300, "υ": 400, "φ": 500, "χ": 600, "ψ": 700, "ω": 800, "ϡ": 900}
AGRIPPA = dict(zip("ABCDEFGHIKLMNOPQRSTVXYZ", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90,
                                                100, 200, 300, 400, 500]))


def hebrew_value(text: str) -> int:
    return sum(HEBREW.get(c, 0) for c in text)


def greek_value(text: str) -> int:
    base = "".join(c for c in unicodedata.normalize("NFD", text.lower()) if not unicodedata.combining(c))
    return sum(GREEK.get(c, 0) for c in base)


def ordinal(text: str) -> int:
    return sum(ord(c) - 64 for c in normalize_latin(text))


def reverse_ordinal(text: str) -> int:
    return sum(27 - (ord(c) - 64) for c in normalize_latin(text))


def agrippa(text: str) -> int:
    s = normalize_latin(text).replace("J", "I").replace("U", "V").replace("W", "VV")
    return sum(AGRIPPA[c] for c in s)


def is_hebrew(text: str) -> bool:
    return any(c in HEBREW for c in text)


def is_greek(text: str) -> bool:
    return greek_value(text) > 0


class GematriaMethod(Method):
    key = "gematria"
    label = "Gematrija (hebrejska, grčka, latinska/Agrippa, engleska ordinalna)"
    kind = "traditional ciphers + experimental text selection and mapping"
    sources = (
        "Hebrew mispar hechrachi – standard letter values (e.g. G. Scholem, Kabbalah, 1974, 'Gematria')",
        "Greek isopsephy – Milesian numerals",
        "H. C. Agrippa, De occulta philosophia libri tres (1533), II.20 – Latin letter numbers",
        "English ordinal / reverse ordinal – modern simple ciphers",
    )

    def texts(self, subject: Subject, event: Event) -> list[tuple[str, str]]:
        d, b = event.local, subject.birth_local
        out = [("draw_weekday", CROATIAN_WEEKDAYS[d.weekday()]), ("draw_month", CROATIAN_MONTHS[d.month - 1]),
               ("draw_place", event.place.name), ("birth_weekday", CROATIAN_WEEKDAYS[b.weekday()]),
               ("birth_month", CROATIAN_MONTHS[b.month - 1]), ("birth_place", subject.place.name)]
        if subject.name:
            out.append(("name", subject.name))
        return out

    def numbers(self, subject: Subject, event: Event) -> MethodNumbers:
        values, sequence = [], []
        for label, text in self.texts(subject, event):
            v = {"label": label, "text": text, "normalized": normalize_latin(text), "ordinal": ordinal(text),
                 "reverse_ordinal": reverse_ordinal(text), "agrippa": agrippa(text)}
            values.append(v)
            sequence += [v["ordinal"], v["reverse_ordinal"], v["agrippa"]]
        components = [Component("latin_ciphers", "ran", "ordinal, reverse ordinal, Agrippa")]
        if not subject.name:
            components.append(Component("name_cipher", "skipped", "no name in config.yaml (subject.name is null)"))
        extra = [t for t in self.params.get("extra_texts", []) or []]
        heb = [{"text": t, "mispar_hechrachi": hebrew_value(t)} for t in extra if is_hebrew(t)]
        grk = [{"text": t, "isopsephy": greek_value(t)} for t in extra if is_greek(t)]
        for h in heb:
            sequence.append(h["mispar_hechrachi"])
        for g in grk:
            sequence.append(g["isopsephy"])
        components.append(Component("hebrew_mispar_hechrachi", "ran" if heb else "skipped",
                                    "" if heb else "no Hebrew text supplied (gematria.extra_texts)"))
        components.append(Component("greek_isopsephy", "ran" if grk else "skipped",
                                    "" if grk else "no Greek text supplied (gematria.extra_texts)"))
        return MethodNumbers(
            sequence=sequence,
            digit_sequence=sequence[:6],
            intermediate={"texts": values, "hebrew": heb, "greek": grk},
            components=components,
        )
