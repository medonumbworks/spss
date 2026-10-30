"""Lenormand (36) i Tarot (78): mijesanje seedom, kaskade i tekstualni prikaz."""
from __future__ import annotations
import numpy as np

LENORMAND = ["Jahac", "Djetelina", "Brod", "Kuca", "Drvo", "Oblaci", "Zmija", "Lijes", "Buket", "Kosa", "Bic", "Ptice",
             "Dijete", "Lisica", "Medvjed", "Zvijezde", "Roda", "Pas", "Toranj", "Vrt", "Planina", "Raskrizje", "Misevi",
             "Srce", "Prsten", "Knjiga", "Pismo", "Muskarac", "Zena", "Ljiljani", "Sunce", "Mjesec", "Kljuc", "Ribe",
             "Sidro", "Kriz"]

_MAJOR = ["Mag", "Visoka svecenica", "Carica", "Car", "Hijerofant", "Ljubavnici", "Kocija", "Snaga", "Pustinjak",
          "Kolo srece", "Pravda", "Objeseni", "Smrt", "Umjerenost", "Davao", "Kula", "Zvijezda", "Mjesec", "Sunce", "Sud",
          "Svijet", "Luda"]
_RANKS = ["As", "2", "3", "4", "5", "6", "7", "8", "9", "10", "Paz", "Vitez", "Kraljica", "Kralj"]
TAROT = list(_MAJOR)
for _suit in ("stapova", "pehara", "maceva", "diskova"):
    TAROT += [f"{r} {_suit}" for r in _RANKS]
assert len(TAROT) == 78 and len(LENORMAND) == 36


def lenormand_cascade(rng: np.random.Generator) -> tuple[list[list[int]], list[tuple[str, int]]]:
    perm = [int(x) for x in rng.permutation(36) + 1]
    tableau = [perm[0:9], perm[9:18], perm[18:27], perm[27:36]]
    casc: list[tuple[str, int]] = []
    r1 = tableau[0]
    for a, b in zip(r1[:-1], r1[1:]):
        casc.append((f"r1:{LENORMAND[a-1]}({a})+{LENORMAND[b-1]}({b})", a + b))
    for row in tableau[1:]:
        casc += [(LENORMAND[c - 1], c) for c in row]
    return tableau, casc


def lenormand_text(tableau: list[list[int]]) -> str:
    lines = []
    for i, row in enumerate(tableau, start=1):
        lines.append(f"red {i}: " + "  ".join(f"{c:2d} {LENORMAND[c-1]}" for c in row))
    return "\n".join(lines)


def tarot_cascade(rng: np.random.Generator) -> tuple[list[int], list[tuple[str, int]]]:
    order = [int(x) for x in rng.permutation(78) + 1]
    casc = [(f"{c} {TAROT[c-1]}", c) for c in order if c <= 45]
    return order, casc


def tarot_text(order: list[int]) -> str:
    lines = ["prvih 20 karata mijesanja (* = indeks <= 45, ulazi u kaskadu):"]
    for pos, c in enumerate(order[:20], start=1):
        lines.append(f"  {pos:2d}. {c:2d} {TAROT[c-1]}{' *' if c <= 45 else ''}")
    return "\n".join(lines)


def selftest() -> None:
    assert len(set(LENORMAND)) == 36 and len(set(TAROT)) == 78
    rng = np.random.default_rng(123)
    tab, casc = lenormand_cascade(rng)
    assert sorted(c for r in tab for c in r) == list(range(1, 37))
    assert len(casc) == 35 and all(v == tab[0][i] + tab[0][i + 1] for i, (_, v) in enumerate(casc[:8]))
    tab2, casc2 = lenormand_cascade(np.random.default_rng(123))
    assert tab2 == tab and casc2 == casc
    order, tc = tarot_cascade(np.random.default_rng(7))
    assert sorted(order) == list(range(1, 79)) and len(tc) == 45 and all(v <= 45 for _, v in tc)
    assert len({v for _, v in tc}) == 45
    assert tarot_cascade(np.random.default_rng(7))[0] == order
    print(lenormand_text(tab))
    print(tarot_text(order))
    print("cards selftest OK")


if __name__ == "__main__":
    selftest()
