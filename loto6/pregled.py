"""Citljiv pregled svih kombinacija iz loto6_godina.csv i loto6_fiksni_set.csv -> kombinacije_pregled.md"""
from __future__ import annotations
import collections
import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> None:
    rows = list(csv.DictReader(open(HERE / "loto6_godina.csv", encoding="utf-8-sig"), delimiter=";"))
    fixed = list(csv.DictReader(open(HERE / "loto6_fiksni_set.csv", encoding="utf-8-sig"), delimiter=";"))
    seed = rows[0]["seed"]
    by = collections.OrderedDict()
    for r in rows:
        by.setdefault((r["datum"], r["dan"], r["vrijeme"]), []).append(r)
    L = [f"# Loto 6/45 - kombinacije po kolu (seed {seed})", "",
         f"Kola: {len(by)}; kombinacija po kolu: {len(next(iter(by.values())))}; ukupno redaka: {len(rows)}.",
         "Plecko = konvergencija (brojevi koje nominira najvise sustava u tom kolu). Zvjezdica (*) = kombinacija sadrzi zamjenu po pravilu seta ili dopunu; detalji u stupcima izvod/zamjena CSV-a.", ""]
    L += ["## Fiksni set (nepromijenjen cijelo razdoblje)", "", "| sustav | n1 | n2 | n3 | n4 | n5 | n6 | joker |", "|---|---|---|---|---|---|---|---|"]
    for r in fixed:
        L.append(f"| {r['sustav']}{'*' if r['zamjena'] else ''} | " + " | ".join(r[f"n{i}"] for i in range(1, 7)) + f" | {r['joker']} |")
    L.append("")
    for (d, dan, t), rs in by.items():
        L += [f"## {d} {dan} {t}", "", "| sustav | n1 | n2 | n3 | n4 | n5 | n6 | joker |", "|---|---|---|---|---|---|---|---|"]
        for r in rs:
            L.append(f"| {r['sustav']}{'*' if r['zamjena'] else ''} | " + " | ".join(r[f"n{i}"] for i in range(1, 7)) + f" | {r['joker']} |")
        L.append("")
    (HERE / "kombinacije_pregled.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"zapisano kombinacije_pregled.md ({len(L)} redaka)")


if __name__ == "__main__":
    main()
