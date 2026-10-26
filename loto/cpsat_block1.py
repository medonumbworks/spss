"""Blok 1 kao egzaktni model (CP-SAT): 100 kombinacija koje
  * izbjegavaju sve već zauzete trojke (blokovi 2–4 + 10 fiksnih iz PDF-a) i međusobno ne dijele trojku,
  * pokrivaju svaki od 990 parova barem 1× i najviše 2×,
  * svaki broj 13–14×, ≤3 iz istog pojasa od 9 i ≥4 pojasa, zbroj u 10.–90. percentilu,
  * nisu jednake nijednoj odigranoj/prošloj kombinaciji.

Kandidati: svi 6-podskupovi koji zadovoljavaju lokalna pravila i čije su sve trojke slobodne; CP-SAT bira 100.
"""
from __future__ import annotations

import itertools
import random
import time
from collections import Counter

import numpy as np

import common as C


def enumerate_candidates(used_triples: set, forbidden: set, sum_lo: float, sum_hi: float, rng: random.Random,
                         max_cands: int | None = None) -> list[tuple]:
    """Brzo nabrajanje: numpy maske svih 8.145.060 kombinacija, filter po zauzetim trojkama bitmaskama."""
    D = C.all_draw_masks()
    # 1) zbroj i pojasevi preko bit-popisa: izračunaj kroz indekse (sporo u Pythonu) → radimo vektorski
    # vektorski zbroj: Σ x·bit
    sums = np.zeros(len(D), dtype=np.int32)
    bands = [np.zeros(len(D), dtype=np.int8) for _ in range(5)]
    for x in range(1, 46):
        bit = ((D >> (x - 1)) & 1).astype(np.int8)
        sums += bit.astype(np.int32) * x
        bands[(x - 1) // 9] += bit
    ok = (sums >= sum_lo) & (sums <= sum_hi)
    bmax = np.maximum.reduce(bands)
    nb = sum((b > 0).astype(np.int8) for b in bands)
    ok &= (bmax <= 3) & (nb >= 4)
    # 2) zauzete trojke: za svaku zauzetu trojku isključi sve kombinacije koje je sadrže
    for t in used_triples:
        m = np.int64(C.mask(t))
        ok &= (D & m) != m
    idx = np.nonzero(ok)[0]
    cands = []
    for i in idx:
        m = int(D[i])
        c = tuple(x for x in range(1, 46) if m >> (x - 1) & 1)
        if c not in forbidden:
            cands.append(c)
    if max_cands and len(cands) > max_cands:
        cands = rng.sample(cands, max_cands)
    return cands


def solve(cands: list[tuple], n_blocks=100, time_limit=300.0, workers=4, seed=20261029, log=print,
          pair_cap=2, hint: list[tuple] | None = None):
    from ortools.sat.python import cp_model
    m = cp_model.CpModel()
    x = [m.NewBoolVar(f"b{i}") for i in range(len(cands))]
    by_pair: dict[tuple, list] = {}
    by_triple: dict[tuple, list] = {}
    by_num: dict[int, list] = {k: [] for k in range(1, 46)}
    for i, c in enumerate(cands):
        for p in itertools.combinations(c, 2):
            by_pair.setdefault(p, []).append(x[i])
        for t in itertools.combinations(c, 3):
            by_triple.setdefault(t, []).append(x[i])
        for k in c:
            by_num[k].append(x[i])
    m.Add(sum(x) == n_blocks)
    for p in itertools.combinations(range(1, 46), 2):
        lst = by_pair.get(p, [])
        if not lst:
            log(f"par {p} nema kandidata")
            return None
        m.Add(sum(lst) >= 1)
        m.Add(sum(lst) <= pair_cap)
    for t, lst in by_triple.items():
        if len(lst) > 1:
            m.AddAtMostOne(lst)
    for k in range(1, 46):
        m.Add(sum(by_num[k]) >= 13)
        m.Add(sum(by_num[k]) <= 14)
    if hint:
        hs = set(hint)
        for i, c in enumerate(cands):
            m.AddHint(x[i], 1 if c in hs else 0)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_search_workers = workers
    solver.parameters.random_seed = seed
    solver.parameters.log_search_progress = False
    t0 = time.time()
    st = solver.Solve(m)
    log(f"CP-SAT status {solver.StatusName(st)} za {time.time() - t0:.0f}s, kandidata {len(cands)}")
    if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None
    return [cands[i] for i in range(len(cands)) if solver.Value(x[i])]


if __name__ == "__main__":
    import json, sys
    import generate as G
    d = json.load(open(C.HERE / "out" / "set231.json"))
    others = [tuple(c) for k in ("2", "3", "4") for c in d["blokovi"][k]]
    used = set()
    for c in others + G.PLAYED["pdf_10_fiksnih"]:
        used.update(C.triples(c))
    rng = random.Random(20261029)
    t0 = time.time()
    cands = enumerate_candidates(used, G.FORBIDDEN, G.SUM_LO, G.SUM_HI, rng)
    print("kandidata:", len(cands), "u %.0fs" % (time.time() - t0))
    b1 = [tuple(c) for c in d["blokovi"]["1"]]
    n_sample = int(sys.argv[2]) if len(sys.argv) > 2 else 40000
    cap = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    pool = list(set(rng.sample(cands, min(n_sample, len(cands))) + [c for c in b1 if c in set(cands)]))
    print("bazen:", len(pool), "cap", cap)
    sol = solve(pool, time_limit=float(sys.argv[1]) if len(sys.argv) > 1 else 300, pair_cap=cap, hint=b1)
    if sol:
        pairs = Counter(p for c in sol for p in itertools.combinations(c, 2))
        freq = Counter(x for c in sol for x in c)
        print("rješenje:", len(sol), "parova", len(pairs), "max", max(pairs.values()), "freq", min(freq.values()), max(freq.values()))
        print("max presjek sa svima:", C.max_pairwise_overlap(sol + others))
        json.dump([list(c) for c in sol], open(C.HERE / "out" / "_cpsat_b1.json", "w"))
