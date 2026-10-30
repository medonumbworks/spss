"""Agrippine planetarne kameje (De occulta philosophia II.22) s provjerom redaka, stupaca i dijagonala."""
from __future__ import annotations

KAMEA: dict[str, list[list[int]]] = {
    "Saturn": [[4, 9, 2], [3, 5, 7], [8, 1, 6]],
    "Jupiter": [[4, 14, 15, 1], [9, 7, 6, 12], [5, 11, 10, 8], [16, 2, 3, 13]],
    "Mars": [[11, 24, 7, 20, 3], [4, 12, 25, 8, 16], [17, 5, 13, 21, 9], [10, 18, 1, 14, 22], [23, 6, 19, 2, 15]],
    "Sunce": [[6, 32, 3, 34, 35, 1], [7, 11, 27, 28, 8, 30], [19, 14, 16, 15, 23, 24], [18, 20, 22, 21, 17, 13],
              [25, 29, 10, 9, 26, 12], [36, 5, 33, 4, 2, 31]],
    "Venera": [[22, 47, 16, 41, 10, 35, 4], [5, 23, 48, 17, 42, 11, 29], [30, 6, 24, 49, 18, 36, 12],
               [13, 31, 7, 25, 43, 19, 37], [38, 14, 32, 1, 26, 44, 20], [21, 39, 8, 33, 2, 27, 45],
               [46, 15, 40, 9, 34, 3, 28]],
    "Merkur": [[8, 58, 59, 5, 4, 62, 63, 1], [49, 15, 14, 52, 53, 11, 10, 56], [41, 23, 22, 44, 45, 19, 18, 48],
               [32, 34, 35, 29, 28, 38, 39, 25], [40, 26, 27, 37, 36, 30, 31, 33], [17, 47, 46, 20, 21, 43, 42, 24],
               [9, 55, 54, 12, 13, 51, 50, 16], [64, 2, 3, 61, 60, 6, 7, 57]],
    "Mjesec": [[37, 78, 29, 70, 21, 62, 13, 54, 5], [6, 38, 79, 30, 71, 22, 63, 14, 46], [47, 7, 39, 80, 31, 72, 23, 55, 15],
               [16, 48, 8, 40, 81, 32, 64, 24, 56], [57, 17, 49, 9, 41, 73, 33, 65, 25], [26, 58, 18, 50, 1, 42, 74, 34, 66],
               [67, 27, 59, 10, 51, 2, 43, 75, 35], [36, 68, 19, 60, 11, 52, 3, 44, 76], [77, 28, 69, 20, 61, 12, 53, 4, 45]],
}


def magic_constant(n: int) -> int:
    return n * (n * n + 1) // 2


def check(planet: str) -> None:
    sq = KAMEA[planet]
    n = len(sq)
    c = magic_constant(n)
    assert all(len(r) == n for r in sq), f"{planet}: nije kvadrat"
    assert sorted(v for r in sq for v in r) == list(range(1, n * n + 1)), f"{planet}: brojevi nisu 1..{n*n}"
    for i, r in enumerate(sq):
        assert sum(r) == c, f"{planet}: red {i+1} = {sum(r)} != {c}"
    for j in range(n):
        s = sum(sq[i][j] for i in range(n))
        assert s == c, f"{planet}: stupac {j+1} = {s} != {c}"
    d1 = sum(sq[i][i] for i in range(n))
    d2 = sum(sq[i][n - 1 - i] for i in range(n))
    assert d1 == c and d2 == c, f"{planet}: dijagonale {d1}, {d2} != {c}"


def _middle_rows(n: int) -> list[int]:
    return [n // 2] if n % 2 else [n // 2 - 1, n // 2]


def kamea_cascade(planet: str) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    """(kaskada, rezerva). Kaskada: sredisnji red (dva za parni n), konstanta, ukupni zbroj.
    Rezerva: ostali redovi odozgo prema dolje, zatim stupci slijeva."""
    check(planet)
    sq = KAMEA[planet]
    n = len(sq)
    mid = _middle_rows(n)
    casc: list[tuple[str, int]] = []
    for i in mid:
        casc += [(f"red{i+1}[{j+1}]", v) for j, v in enumerate(sq[i])]
    casc.append(("konstanta", magic_constant(n)))
    casc.append(("ukupni zbroj", n * n * (n * n + 1) // 2))
    res: list[tuple[str, int]] = []
    for i, row in enumerate(sq):
        if i in mid:
            continue
        res += [(f"red{i+1}[{j+1}]", v) for j, v in enumerate(row)]
    for j in range(n):
        res += [(f"stupac{j+1}[{i+1}]", sq[i][j]) for i in range(n)]
    return casc, res


def kamea_text(planet: str) -> str:
    sq = KAMEA[planet]
    n = len(sq)
    w = len(str(n * n))
    lines = [f"{planet} {n}x{n}, konstanta {magic_constant(n)}, zbroj {n*n*(n*n+1)//2}"]
    for i, r in enumerate(sq):
        mark = " <- sredisnji" if i in _middle_rows(n) else ""
        lines.append(" ".join(f"{v:{w}d}" for v in r) + mark)
    return "\n".join(lines)


def selftest() -> None:
    for p in KAMEA:
        check(p)
    c, _ = kamea_cascade("Saturn")
    assert [v for _, v in c] == [3, 5, 7, 15, 45], c
    c, _ = kamea_cascade("Sunce")
    assert [v for _, v in c][:12] == [19, 14, 16, 15, 23, 24, 18, 20, 22, 21, 17, 13] and c[-2][1] == 111 and c[-1][1] == 666
    c, r = kamea_cascade("Mjesec")
    assert c[-2][1] == 369 and c[-1][1] == 3321 and len(r) == 8 * 9 + 81
    for p in KAMEA:
        print(kamea_text(p))
    print("kamea selftest OK")


if __name__ == "__main__":
    selftest()
