"""Neovisna provjera skupa od 231 kombinacije (out/set231.json ili markdown s `aa · bb · …` zapisima).

Pokretanje: python3 verify.py [datoteka]   → ispis + out/verify.json
Sve vjerojatnosti su egzaktne (nabrajanje svih 8.145.060 izvlačenja).
"""
from __future__ import annotations

import itertools
import json
import math
import sys
from collections import Counter
from pathlib import Path

import common as C


def load_set(path: Path):
    if path.suffix == ".json":
        d = json.load(open(path, encoding="utf-8"))
        blocks = {int(k): [tuple(c) for c in v] for k, v in d["blokovi"].items()}
        allb = [tuple(c) for c in d["sve"]]
        return blocks, allb
    allb = C.parse_md_set(path)
    return {1: allb[:100], 2: allb[100:150], 3: allb[150:200], 4: allb[200:]}, allb


def main(path: Path):
    blocks, allb = load_set(path)
    rep = {}
    rep["n"] = len(allb)
    rep["distinct"] = len(set(allb))
    rep["valid_format"] = all(len(c) == 6 and len(set(c)) == 6 and all(1 <= x <= 45 for x in c) for c in allb)
    rep["max_overlap"] = C.max_pairwise_overlap(allb)
    rep["overlap_dist"] = {str(k): v for k, v in sorted(C.overlap_matrix_counts(allb).items())}
    played = C.load_played()
    allp = set(C.all_played())
    rep["n_played"] = len(allp)
    rep["equal_to_played"] = sorted(set(allb) & allp)
    draws = C.load_draws()
    hist = {d["nums"] for d in draws}
    rep["equal_to_past_draw"] = sorted(set(allb) & hist)
    rep["max_overlap_with_pdf10"] = max(len(set(a) & set(b)) for a in allb for b in played["pdf_10_fiksnih"])
    rep["overlap_with_pdf10_dist"] = {str(k): v for k, v in sorted(Counter(
        max(len(set(a) & set(b)) for b in played["pdf_10_fiksnih"]) for a in allb).items())}
    fr = Counter(x for c in allb for x in c)
    rep["freq_all"] = {x: fr.get(x, 0) for x in range(1, 46)}
    rep["freq_min_max"] = (min(fr.values()), max(fr.values()))
    # trojke / četvorke / petorke
    tr = Counter(t for c in allb for t in itertools.combinations(c, 3))
    rep["triples_distinct"] = len(tr); rep["triples_max_mult"] = max(tr.values())
    q = Counter(t for c in allb for t in itertools.combinations(c, 4))
    rep["quads_distinct"] = len(q); rep["quads_max_mult"] = max(q.values())
    pr = Counter(p for c in allb for p in itertools.combinations(c, 2))
    rep["pairs_covered_all"] = len(pr); rep["pairs_max_all"] = max(pr.values())
    # blok 1
    b1 = blocks.get(1, [])
    if b1:
        p1 = Counter(p for c in b1 for p in itertools.combinations(c, 2))
        f1 = Counter(x for c in b1 for x in c)
        rep["b1"] = dict(n=len(b1), pairs_covered=len(p1), pairs_max=max(p1.values()),
                         pairs_dist={str(k): v for k, v in sorted(Counter(p1.values()).items())},
                         freq_min_max=(min(f1.values()), max(f1.values())),
                         band_patterns={str(k): v for k, v in Counter(
                             tuple(sorted(Counter((x - 1) // 9 for x in c).values(), reverse=True)) for c in b1).items()},
                         sum_mean=sum(sum(c) for c in b1) / len(b1),
                         odd_mean=sum(sum(x % 2 for x in c) for c in b1) / len(b1),
                         le31_mean=sum(sum(x <= 31 for x in c) for c in b1) / len(b1))
    b4 = blocks.get(4, [])
    if b4:
        rep["b4"] = dict(n=len(b4),
                         rules_ok=all(sum(x > 31 for x in c) >= 4 and sum(x <= 12 for x in c) <= 1 and 7 not in c
                                      and sum(c) >= 165 for c in b4),
                         sum_mean=sum(sum(c) for c in b4) / len(b4),
                         gt31_mean=sum(sum(x > 31 for x in c) for c in b4) / len(b4),
                         consec_share=sum(1 for c in b4 if any(c[i + 1] - c[i] == 1 for i in range(5))) / len(b4))
    # egzaktno
    D = C.all_draw_masks()
    st = C.exact_stats(allb, D)
    rep["exact"] = st
    per = 235 / C.TOTAL
    rep["p_ge5_theoretical_max"] = len(allb) * per
    rep["p_ge5_is_max"] = abs(st["p_ge5"] - len(allb) * per) < 1e-12
    rep["history_hits"] = {str(k): v for k, v in sorted(C.hits_against_history(allb, draws).items())}
    (C.HERE / "out").mkdir(exist_ok=True)
    json.dump(rep, open(C.HERE / "out" / "verify.json", "w"), indent=1, ensure_ascii=False, default=str)

    print(f"kombinacija: {rep['n']}  različitih: {rep['distinct']}  format OK: {rep['valid_format']}")
    print(f"max presjek para: {rep['max_overlap']}  raspodjela: {rep['overlap_dist']}")
    print(f"jednake odigranima ({rep['n_played']}): {rep['equal_to_played']}  jednake prošlim izvlačenjima: {rep['equal_to_past_draw']}")
    print(f"max presjek s 10 fiksnih iz PDF-a: {rep['max_overlap_with_pdf10']}  {rep['overlap_with_pdf10_dist']}")
    print(f"frekvencije min/max: {rep['freq_min_max']}")
    print(f"trojke: {rep['triples_distinct']} (max {rep['triples_max_mult']}×), četvorke {rep['quads_distinct']} (max {rep['quads_max_mult']}×), "
          f"parovi {rep['pairs_covered_all']}/990 (max {rep['pairs_max_all']}×)")
    if b1:
        print("blok 1:", rep["b1"])
    if b4:
        print("blok 4:", rep["b4"])
    print("EGZAKTNO: P(≥5 negdje) = %.6f = 1 u %.1f  (teorijski max za %d: %.6f, dosegnut: %s)" %
          (st["p_ge5"], 1 / st["p_ge5"], len(allb), rep["p_ge5_theoretical_max"], rep["p_ge5_is_max"]))
    print("          P(≥4 negdje) = %.4f  P(≥3 negdje) = %.4f  P(ništa) = %.6f = 1 u %.0f" %
          (st["p_ge4"], st["p_ge3"], st["p_nothing"], 1 / st["p_nothing"]))
    print("          očekivani broj kombinacija s 3/4/5/6:", {k: round(v, 4) for k, v in st["exp_count"].items()})
    print("          najviše istodobno s 3/4/5/6:", st["max_simultaneous"])
    print("povijest 882 kola — pogodaka 3/4/5/6:", rep["history_hits"])


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else C.HERE / "out" / "set231.json")
