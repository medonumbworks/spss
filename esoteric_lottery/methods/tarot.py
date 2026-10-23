"""Tarot numerical mappings.

Traditional numbering: the Rider-Waite-Smith deck (A. E. Waite, *The Pictorial Key to
the Tarot*, 1910), with Strength = 8 and Justice = 11; 22 Major Arcana 0-21 and four
suits (Wands, Cups, Swords, Pentacles) of Ace-10, Page, Knight, Queen, King. The Fool is
counted as 22 when it closes the sequence, as in M. K. Greer's birth-card system.

Traditional (modern) birth cards: M. K. Greer, *Tarot for Your Self* (1984) /
*Who Are You in the Tarot?* (2002): add month + day + year of birth; reduce by digit
sum until the total is 22 or less; that card and its further digit-sum reductions are the
birth cards (1991-05-31: 5 + 31 + 1991 = 2027 -> 11 Justice -> 2 High Priestess).
The year card uses the draw year instead of the birth year.

Card index used for the lottery (experimental): Majors 1-21 by number, the Fool 22,
then Wands 23-36, Cups 37-50, Swords 51-64, Pentacles 65-78. For Sve ili Ništa the 22
Majors map one-to-one onto the numbers 1-22.

Experimental: a deterministic "shuffle" (numpy PCG64 seeded with SHA-256 of birth and
draw moment) and a spread of ``spread_size`` cards in the order drawn; reversals are
recorded but do not change numbers. Sequence = birth cards, year cards, spread. Joker
digits = the pip value (Major number, or rank 1-14 for Minors) of the first six spread
cards, mod 10.
"""
from __future__ import annotations

from .common import Component, Event, Method, MethodNumbers, Subject, digit_sum, seeded_rng

MAJORS = ["The Fool", "The Magician", "The High Priestess", "The Empress", "The Emperor", "The Hierophant",
          "The Lovers", "The Chariot", "Strength", "The Hermit", "Wheel of Fortune", "Justice", "The Hanged Man",
          "Death", "Temperance", "The Devil", "The Tower", "The Star", "The Moon", "The Sun", "Judgement",
          "The World"]
SUITS = ["Wands", "Cups", "Swords", "Pentacles"]
RANKS = ["Ace", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten", "Page", "Knight", "Queen",
         "King"]


def build_deck() -> list[dict]:
    deck = []
    for n, name in enumerate(MAJORS):
        deck.append({"index": n if n else 22, "name": name, "arcana": "major", "value": n})
    for s, suit in enumerate(SUITS):
        for r, rank in enumerate(RANKS):
            deck.append({"index": 23 + 14 * s + r, "name": f"{rank} of {suit}", "arcana": "minor", "suit": suit,
                         "value": r + 1})
    return deck


DECK = build_deck()


def card_chain(total: int) -> list[int]:
    """Greer reduction: digit sums until <= 22, then onward down to a single digit."""
    n = total
    while n > 22:
        n = digit_sum(n)
    chain = [n]
    while chain[-1] > 9:
        chain.append(digit_sum(chain[-1]))
    return chain


def major_name(n: int) -> str:
    return MAJORS[0] if n == 22 else MAJORS[n]


class TarotMethod(Method):
    key = "tarot"
    label = "Tarot (Rider-Waite-Smith, Greerine rođendanske karte)"
    kind = "traditional numbering and birth cards + experimental seeded spread"
    sources = (
        "A. E. Waite, The Pictorial Key to the Tarot, 1910 – Rider-Waite-Smith numbering (Strength 8, Justice 11)",
        "M. K. Greer, Tarot for Your Self, 1984; Who Are You in the Tarot?, 2002 – birth and year cards",
        "Seeded shuffle (numpy PCG64 + SHA-256) – experimental stand-in for a physical shuffle",
    )

    def numbers(self, subject: Subject, event: Event) -> MethodNumbers:
        b = subject.birth_local
        birth_total = b.month + b.day + b.year
        year_total = b.month + b.day + event.local.year
        birth_cards, year_cards = card_chain(birth_total), card_chain(year_total)
        rng = seeded_rng(self.key, subject.birth_local.isoformat(), event.local.isoformat(), self.seed)
        order = rng.permutation(len(DECK))
        reversed_flags = rng.random(len(DECK)) < 0.5
        size = int(self.params.get("spread_size", 22))
        spread = [{**DECK[i], "reversed": bool(reversed_flags[j])} for j, i in enumerate(order[:size])]
        sequence = birth_cards + year_cards + [c["index"] for c in spread]
        return MethodNumbers(
            sequence=sequence,
            digit_sequence=[c["value"] for c in spread[:6]],
            intermediate={
                "birth_cards": {"total": birth_total, "chain": birth_cards, "names": [major_name(n) for n in birth_cards]},
                "year_cards": {"year": event.local.year, "total": year_total, "chain": year_cards,
                               "names": [major_name(n) for n in year_cards]},
                "spread": spread,
            },
            components=[Component("greer_birth_cards", "ran"), Component("greer_year_cards", "ran"),
                        Component("seeded_spread", "ran", f"{size} cards from a 78-card deck")],
        )
