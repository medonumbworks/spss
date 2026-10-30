"""Zajednicka pravila pretvorbe: m45, kaskada -> kombinacija, Joker, rng, pravilo seta."""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from config import read_seed

Cascade = list[tuple[str, int]]   # (oznaka, sirova vrijednost prije m45)


def m45(n: int) -> int:
    return ((int(n) - 1) % 45) + 1


def rng_for(draw_ordinal: int, system_id: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence(read_seed(), spawn_key=(int(draw_ordinal), int(system_id))))


def joker_from(cascade: Cascade, reserve: Cascade | None = None) -> str:
    vals = [v for _, v in cascade] + [v for _, v in (reserve or [])]
    digits = [str(abs(int(v)))[-1] for v in vals[:6]]
    while len(digits) < 6:
        digits.append("0")
    return "".join(digits)


@dataclass
class Combo:
    system_id: int
    system: str
    numbers: list[int]                 # u redoslijedu kaskade (ne sortirano)
    cascade: Cascade                   # glavna kaskada (sirovo)
    reserve: Cascade                   # rezervna kaskada (sirovo)
    joker: str
    used_from: list[str]               # oznake iskoristenih kandidata, istim redom kao numbers
    dopuna: bool = False               # dopunjeno iz rezervne kaskade
    dopuna_krajnja: bool = False       # dopunjeno krajnjom rezervom (najmanji slobodan broj)
    izvod: str = ""                    # kratak tekst izvoda
    zamjene: list[str] = field(default_factory=list)
    _next_idx: int = 0                 # pokazivac na sljedeceg kandidata u (cascade + reserve)

    def candidates(self) -> Cascade:
        return list(self.cascade) + list(self.reserve)

    def sorted_numbers(self) -> list[int]:
        return sorted(self.numbers)

    def next_candidate(self, exclude: set[int]) -> tuple[str, int] | None:
        """Sljedeci kandidat (oznaka, m45 vrijednost) koji nije u exclude; pomice pokazivac."""
        cands = self.candidates()
        while self._next_idx < len(cands):
            lab, raw = cands[self._next_idx]
            self._next_idx += 1
            v = m45(raw)
            if v not in exclude:
                return (f"{lab}={raw}->{v}", v)
        return None


def build_combo(system_id: int, system: str, cascade: Cascade, reserve: Cascade, izvod: str = "") -> Combo:
    numbers: list[int] = []
    used: list[str] = []
    dopuna = False
    dopuna_krajnja = False
    cands = list(cascade) + list(reserve)
    idx = 0
    while len(numbers) < 6 and idx < len(cands):
        lab, raw = cands[idx]
        idx += 1
        v = m45(raw)
        if v in numbers:
            continue
        if idx > len(cascade):
            dopuna = True
        numbers.append(v)
        used.append(f"{lab}={raw}->{v}")
    while len(numbers) < 6:
        dopuna_krajnja = True
        v = min(set(range(1, 46)) - set(numbers))
        numbers.append(v)
        used.append(f"krajnja_rezerva->{v}")
    c = Combo(system_id, system, numbers, list(cascade), list(reserve), joker_from(cascade, reserve), used,
              dopuna, dopuna_krajnja, izvod)
    c._next_idx = idx
    return c


def overlap(a: list[int], b: list[int]) -> int:
    return len(set(a) & set(b))


def apply_set_rule(combos: list[Combo], max_shared: int = 2, forbidden: list[set[int]] | None = None) -> list[Combo]:
    """Nijedan par ne dijeli vise od max_shared brojeva. Kombinacije su poredane po rangu (indeks 0 = najvisi).
    Prekrsaj: u kombinaciji NIZEG ranga zamijeni zadnji (po redoslijedu kaskade) od ZAJEDNICKIH brojeva
    sljedecim kandidatom iz njezine kaskade koji ne stvara novi prekrsaj; sve zamjene se biljeze.
    `forbidden`: skupovi vec odigranih kombinacija (odigrano.csv) koje se ne smiju ponoviti."""
    forbidden = forbidden or []
    for i, c in enumerate(combos):
        guard = 0
        while True:
            guard += 1
            if guard > 500:
                raise RuntimeError(f"pravilo seta: nije konvergiralo za {c.system}")
            viol = None
            for j in range(i):
                h = combos[j]
                if overlap(c.numbers, h.numbers) > max_shared:
                    viol = ("preklapanje", h, set(c.numbers) & set(h.numbers))
                    break
            if viol is None:
                for f in forbidden:
                    if set(c.numbers) == f:
                        viol = ("odigrano", None, set(c.numbers))
                        break
            if viol is None:
                break
            kind, h, shared = viol
            # zadnji zajednicki broj po redoslijedu kaskade
            pos = max(k for k, n in enumerate(c.numbers) if n in shared)
            old = c.numbers[pos]
            # trazi kandidata koji ne stvara prekrsaj s visim rangovima
            replaced = False
            tried = []
            while True:
                nc = c.next_candidate(set(c.numbers))
                if nc is None:
                    break
                lab, v = nc
                trial = c.numbers[:pos] + c.numbers[pos + 1:] + [v]
                ok = all(overlap(trial, combos[j].numbers) <= max_shared for j in range(i)) and \
                    all(set(trial) != f for f in forbidden)
                tried.append(str(v))
                if ok:
                    c.numbers = trial
                    c.used_from = c.used_from[:pos] + c.used_from[pos + 1:] + [lab]
                    who = h.system if h is not None else "odigrano.csv"
                    c.zamjene.append(f"{old}->{v} ({kind} s {who}: {sorted(shared)}; kandidat {lab})")
                    replaced = True
                    break
            if not replaced:
                # krajnja rezerva: najmanji broj koji ne stvara prekrsaj
                for v in range(1, 46):
                    if v in c.numbers:
                        continue
                    trial = c.numbers[:pos] + c.numbers[pos + 1:] + [v]
                    if all(overlap(trial, combos[j].numbers) <= max_shared for j in range(i)) and \
                            all(set(trial) != f for f in forbidden):
                        c.numbers = trial
                        c.used_from = c.used_from[:pos] + c.used_from[pos + 1:] + [f"krajnja_rezerva->{v}"]
                        who = h.system if h is not None else "odigrano.csv"
                        c.zamjene.append(f"{old}->{v} ({kind} s {who}: {sorted(shared)}; KRAJNJA REZERVA, kaskada iscrpljena)")
                        c.dopuna_krajnja = True
                        replaced = True
                        break
            if not replaced:
                raise RuntimeError(f"pravilo seta: nema zamjene za {c.system}")
    return combos


def selftest() -> None:
    assert m45(1) == 1 and m45(45) == 45 and m45(46) == 1 and m45(0) == 45 and m45(-1) == 44 and m45(90) == 45
    casc = [("a", 77), ("b", 227), ("c", 450), ("d", 14), ("e", 18), ("f", 138), ("g", 160), ("h", 576)]
    c = build_combo(6, "Gematrija", casc, [])
    assert c.numbers == [32, 2, 45, 14, 18, 3], c.numbers
    assert c.joker == "770488", c.joker
    assert not c.dopuna
    c2 = build_combo(99, "t", [("x", 1), ("y", 46), ("z", 91)], [("r", 2), ("s", 3)])
    assert c2.numbers == [1, 2, 3, 4, 5, 6] and c2.dopuna and c2.dopuna_krajnja, c2.numbers
    assert c2.joker == "161230", c2.joker
    # pravilo seta
    a = build_combo(1, "A", [("", v) for v in [1, 2, 3, 4, 5, 6]], [])
    b = build_combo(2, "B", [("", v) for v in [1, 2, 3, 9, 10, 11, 12, 13]], [])
    assert b.numbers == [1, 2, 3, 9, 10, 11]
    apply_set_rule([a, b])
    assert b.numbers == [1, 2, 9, 10, 11, 12], b.numbers   # zadnji zajednicki (3) zamijenjen sa 12
    assert len(b.zamjene) == 1
    # odigrano
    d = build_combo(3, "D", [("", v) for v in [20, 21, 22, 23, 24, 25, 26]], [])
    apply_set_rule([d], forbidden=[{20, 21, 22, 23, 24, 25}])
    assert d.numbers == [20, 21, 22, 23, 24, 26], d.numbers
    print("core selftest OK")


if __name__ == "__main__":
    selftest()
