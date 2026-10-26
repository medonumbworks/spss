"""Zajednički alati za Loto 6/45: podaci, registar odigranih, pakiranje, egzaktno nabrajanje.

Sve funkcije rade s kombinacijama kao sortiranim tuple-ovima od 6 brojeva iz 1..45.
"""
from __future__ import annotations

import csv
import itertools
import os
import re
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
N, K = 45, 6
TOTAL = 8_145_060  # C(45,6)

# 10 fiksnih kombinacija iz PDF-a "Loto 6/45 — 10 fiksnih kombinacija za godinu dana"
PDF_FIKSNE = [
    (19, 27, 32, 33, 38, 41), (16, 23, 34, 39, 42, 45), (14, 29, 35, 36, 40, 44),
    (21, 28, 33, 37, 43, 45), (17, 26, 32, 38, 41, 44), (13, 24, 34, 35, 39, 42),
    (20, 31, 36, 40, 43, 44), (15, 22, 37, 41, 42, 45), (18, 25, 33, 38, 39, 44),
    (11, 30, 32, 36, 40, 45),
]


# ----------------------------------------------------------------- podaci
def load_draws(path: Path | None = None) -> list[dict]:
    """Vraća listu izvlačenja: date, year, draw, nums (sorted tuple), bonus, joker."""
    path = path or DATA / "loto_6_45_2018_2026.csv"
    out = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            nums = tuple(sorted(int(r[f"n{i}"]) for i in range(1, 7)))
            out.append(dict(date=r["date"], year=int(r["year"]), draw=int(r["draw"]),
                            nums=nums, bonus=int(r["bonus"]), joker=r["joker"]))
    return out


def load_played() -> dict[str, list[tuple]]:
    """Sve već odigrane/objavljene kombinacije: registar 16.08.2026 (456), sistemi CSV (131), PDF (10)."""
    groups: dict[str, list[tuple]] = {}
    # registar (markdown)
    text = (DATA / "loto6_odigrane_16082026.md").read_text(encoding="utf-8")
    reg: list[tuple] = []
    for line in text.splitlines():
        m = re.match(r"^\d+\.\s+((?:\d{2}\s·\s)+\d{2})", line.strip())
        if not m:
            continue
        nums = [int(x) for x in m.group(1).split(" · ")]
        if len(nums) == 6:
            reg.append(tuple(sorted(nums)))
        elif len(nums) == 7:
            reg.extend(tuple(c) for c in itertools.combinations(sorted(nums), 6))
        else:
            raise ValueError(line)
    groups["registar_16082026"] = reg
    # sistemi CSV
    sis: list[tuple] = []
    with open(DATA / "loto_6-45_sistemi.csv", newline="") as f:
        for r in csv.DictReader(f):
            sis.append(tuple(sorted(int(r[f"b{i}"]) for i in range(1, 7))))
    groups["sistemi_csv"] = sis
    groups["pdf_10_fiksnih"] = [tuple(sorted(c)) for c in PDF_FIKSNE]
    return groups


def all_played() -> list[tuple]:
    return [c for g in load_played().values() for c in g]


def parse_md_set(path: Path) -> list[tuple]:
    """Izvlači sve `aa · bb · cc · dd · ee · ff` kombinacije iz markdown datoteke, redom."""
    text = Path(path).read_text(encoding="utf-8")
    out = []
    for m in re.finditer(r"`(\d{2}) · (\d{2}) · (\d{2}) · (\d{2}) · (\d{2}) · (\d{2})`", text):
        out.append(tuple(sorted(int(x) for x in m.groups())))
    return out


# ----------------------------------------------------------- pakiranje
def triples(c) -> list[tuple]:
    return list(itertools.combinations(sorted(c), 3))


class Packing:
    """Skup kombinacija u kojem nijedan par ne dijeli više od 2 broja (nijedna trojka dvaput)."""

    def __init__(self, blocks=()):
        self.blocks: list[tuple] = []
        self.covered: set[tuple] = set()
        self.freq = Counter()
        for b in blocks:
            self.add(b)

    def fits(self, c) -> bool:
        c = tuple(sorted(c))
        return len(set(c)) == 6 and all(1 <= x <= N for x in c) and not any(t in self.covered for t in triples(c))

    def add(self, c):
        c = tuple(sorted(c))
        if not self.fits(c):
            raise ValueError(f"ne stane u pakiranje: {c}")
        self.blocks.append(c)
        self.covered.update(triples(c))
        self.freq.update(c)

    def remove(self, c):
        c = tuple(sorted(c))
        self.blocks.remove(c)
        self.covered.difference_update(triples(c))
        self.freq.subtract(c)

    def max_overlap(self) -> int:
        return max((len(set(a) & set(b)) for a, b in itertools.combinations(self.blocks, 2)), default=0)


def max_pairwise_overlap(blocks) -> int:
    sets = [set(b) for b in blocks]
    return max((len(a & b) for a, b in itertools.combinations(sets, 2)), default=0)


def overlap_matrix_counts(blocks) -> Counter:
    sets = [set(b) for b in blocks]
    return Counter(len(a & b) for a, b in itertools.combinations(sets, 2))


# ------------------------------------------------------- egzaktno nabrajanje
_D_CACHE = HERE / "out" / "all_draws_masks.npy"


def mask(c) -> int:
    return sum(1 << (x - 1) for x in c)


def all_draw_masks() -> np.ndarray:
    """Maske svih 8.145.060 mogućih izvlačenja (int64), keširano na disku."""
    if _D_CACHE.exists():
        return np.load(_D_CACHE)
    _D_CACHE.parent.mkdir(exist_ok=True)
    bits = np.array([1 << i for i in range(N)], dtype=np.int64)
    idx = np.fromiter(itertools.chain.from_iterable(itertools.combinations(range(N), K)),
                      dtype=np.int8, count=TOTAL * K).reshape(TOTAL, K)
    D = bits[idx].sum(axis=1)
    np.save(_D_CACHE, D)
    return D


def exact_stats(blocks, D: np.ndarray | None = None) -> dict:
    """Za skup kombinacija izračunaj po svim izvlačenjima: distribuciju najboljeg pogotka i
    očekivani broj kombinacija s točno 3/4/5/6 pogodaka. Egzaktno, bez simulacije."""
    D = all_draw_masks() if D is None else D
    best = np.zeros(len(D), dtype=np.int8)
    cnt = {3: np.zeros(len(D), dtype=np.int16), 4: np.zeros(len(D), dtype=np.int16),
           5: np.zeros(len(D), dtype=np.int16), 6: np.zeros(len(D), dtype=np.int16)}
    for b in blocks:
        h = np.bitwise_count(D & np.int64(mask(b))).astype(np.int8)
        np.maximum(best, h, out=best)
        for k in (3, 4, 5, 6):
            cnt[k] += (h == k)
    dist = Counter(best.tolist())
    n = len(D)
    out = {
        "n_blocks": len(blocks),
        "p_best": {k: dist.get(k, 0) / n for k in range(7)},
        "p_ge5": sum(dist.get(k, 0) for k in (5, 6)) / n,
        "p_ge4": sum(dist.get(k, 0) for k in (4, 5, 6)) / n,
        "p_ge3": sum(dist.get(k, 0) for k in (3, 4, 5, 6)) / n,
        "p_nothing": sum(dist.get(k, 0) for k in (0, 1, 2)) / n,
        "exp_count": {k: float(cnt[k].sum()) / n for k in (3, 4, 5, 6)},
        "max_simultaneous": {k: int(cnt[k].max()) for k in (3, 4, 5, 6)},
    }
    return out


def hits_against_history(blocks, draws) -> Counter:
    """Koliko bi puta skup kombinacija pogodio 3/4/5/6 u stvarnim prošlim izvlačenjima."""
    c = Counter()
    for d in draws:
        s = set(d["nums"])
        for b in blocks:
            h = len(s & set(b))
            if h >= 3:
                c[h] += 1
    return c


# ---------------------------------------------------------------- profil
def profile(c) -> dict:
    c = sorted(c)
    return dict(
        sum=sum(c), odd=sum(x % 2 for x in c), low=sum(x <= 22 for x in c),
        le31=sum(x <= 31 for x in c), gt31=sum(x > 31 for x in c),
        consec=sum(1 for i in range(5) if c[i + 1] - c[i] == 1),
        decades=Counter((x - 1) // 10 for x in c), bands9=Counter((x - 1) // 9 for x in c),
        endings=Counter(x % 10 for x in c), spread=c[-1] - c[0],
    )


def fmt(c) -> str:
    return " · ".join(f"{x:02d}" for x in sorted(c))
