"""Lenormand numerical mappings.

Traditional numbering of the 36-card Petit Lenormand, inherited from J. K. Hechtel's
*Das Spiel der Hoffnung* (Nuremberg, c. 1799), later sold under Mlle Lenormand's name
(modern card list e.g. in M. Katz & T. Goodwin, *Learning Lenormand*, 2013):
1 Rider … 36 Cross.

Experimental (no lottery tradition exists): the card whose number equals the day of
birth and the card of the draw day (1-31), then a deterministic Grand Tableau (all 36
cards, numpy PCG64 seeded by SHA-256 of birth and draw moment) read row by row. The
card numbers in that order are reduced into the lottery range; Joker digits are the
first six tableau card numbers mod 10. The querent card (Gentleman 28 / Lady 29) is not
used: the querent's gender is not an input.
"""
from __future__ import annotations

from .common import Component, Event, Method, MethodNumbers, Subject, seeded_rng

CARDS = ["Rider", "Clover", "Ship", "House", "Tree", "Clouds", "Snake", "Coffin", "Bouquet", "Scythe", "Whip",
         "Birds", "Child", "Fox", "Bear", "Stars", "Stork", "Dog", "Tower", "Garden", "Mountain", "Crossroads",
         "Mice", "Heart", "Ring", "Book", "Letter", "Gentleman", "Lady", "Lily", "Sun", "Moon", "Key", "Fish",
         "Anchor", "Cross"]


def card(n: int) -> dict:
    return {"number": n, "name": CARDS[n - 1]}


class LenormandMethod(Method):
    key = "lenormand"
    label = "Lenormand (36 karata, Grand Tableau)"
    kind = "traditional numbering + experimental seeded tableau"
    sources = (
        "J. K. Hechtel, Das Spiel der Hoffnung, c. 1799 – the 36 cards and their numbers",
        "M. Katz & T. Goodwin, Learning Lenormand, 2013 – modern card list and Grand Tableau",
        "Seeded tableau (numpy PCG64 + SHA-256) and day-of-month cards – experimental",
    )

    def numbers(self, subject: Subject, event: Event) -> MethodNumbers:
        rng = seeded_rng(self.key, subject.birth_local.isoformat(), event.local.isoformat(), self.seed)
        tableau = [int(i) + 1 for i in rng.permutation(36)]
        birth_card, draw_card = subject.birth_local.day, event.local.day
        sequence = [birth_card, draw_card] + tableau
        return MethodNumbers(
            sequence=sequence,
            digit_sequence=tableau[:6],
            intermediate={"birth_day_card": card(birth_card), "draw_day_card": card(draw_card),
                          "grand_tableau_rows": [[card(n) for n in tableau[r * 9:(r + 1) * 9]] for r in range(4)]},
            components=[Component("day_of_month_cards", "ran", "experimental"),
                        Component("seeded_grand_tableau", "ran", "4 x 9")],
        )
