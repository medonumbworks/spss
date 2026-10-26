"""LNS-popravak bloka 1 egzaktnim pod-modelom (CP-SAT).

Zadano: trenutni blok 1 (100 kombinacija, možda s nepokrivenim parovima) i zauzete vanjske trojke.
U svakom krugu oslobodi K kombinacija (one oko nepokrivenih parova + nasumične), fiksiraj ostale i
egzaktno izaberi K zamjena iz bazena kandidata tako da: svaki par 1–2×, nijedna trojka dvaput, svaki broj 13–14×.
"""
from __future__ import annotations

import itertools
import random
import time
from collections import Counter

import common as C
from cpsat_block1 import enumerate_candidates


def lns_repair(blocks: list[tuple], all_cands: list[tuple], rounds=12, free_k=20, time_limit=90.0,
               seed=20261029, log=print, workers=4, pair_cap=2):
    from ortools.sat.python import cp_model
    rng = random.Random(seed)
    blocks = list(blocks)
    cand_set = set(all_cands)
    for rnd in range(rounds):
        pairs = Counter(p for c in blocks for p in itertools.combinations(c, 2))
        freq = Counter(x for c in blocks for x in c)
        unc = [p for p in itertools.combinations(range(1, 46), 2) if pairs[p] == 0]
        over = [p for p, v in pairs.items() if v > pair_cap]
        bad_freq = [x for x in range(1, 46) if not 13 <= freq[x] <= 14]
        log(f"LNS krug {rnd}: nepokriveno {len(unc)}, prekobrojno {len(over)}, loša frekvencija {bad_freq}")
        if not unc and not over and not bad_freq:
            return blocks
        hot = set(x for p in unc for x in p) | set(x for p in over for x in p) | set(bad_freq)

        def score(i):
            c = blocks[i]
            return sum(x in hot for x in c) * 3 + sum(1 for p in itertools.combinations(c, 2) if pairs[p] >= 2) * 0.3 \
                + rng.random() * 2
        order = sorted(range(len(blocks)), key=score, reverse=True)
        free_idx = set(order[:free_k])
        fixed = [blocks[i] for i in range(len(blocks)) if i not in free_idx]
        fixed_pairs = Counter(p for c in fixed for p in itertools.combinations(c, 2))
        fixed_tr = set(t for c in fixed for t in itertools.combinations(c, 3))
        fixed_freq = Counter(x for c in fixed for x in c)
        full_pairs = {p for p, v in fixed_pairs.items() if v >= pair_cap}
        full_nums = {x for x in range(1, 46) if fixed_freq[x] >= 14}
        cands = [c for c in all_cands
                 if not any(x in full_nums for x in c)
                 and not any(p in full_pairs for p in itertools.combinations(c, 2))
                 and not any(t in fixed_tr for t in itertools.combinations(c, 3))]
        log(f"  kandidata za pod-model: {len(cands)}")
        m = cp_model.CpModel()
        x = [m.NewBoolVar(f"b{i}") for i in range(len(cands))]
        by_pair: dict[tuple, list] = {}
        by_tr: dict[tuple, list] = {}
        by_num: dict[int, list] = {k: [] for k in range(1, 46)}
        for i, c in enumerate(cands):
            for p in itertools.combinations(c, 2):
                by_pair.setdefault(p, []).append(x[i])
            for t in itertools.combinations(c, 3):
                by_tr.setdefault(t, []).append(x[i])
            for k in c:
                by_num[k].append(x[i])
        m.Add(sum(x) == free_k)
        feasible = True
        for p in itertools.combinations(range(1, 46), 2):
            lst = by_pair.get(p, [])
            need = 1 - fixed_pairs[p]
            room = pair_cap - fixed_pairs[p]
            if need > 0 and not lst:
                feasible = False
                break
            if lst:
                if need > 0:
                    m.Add(sum(lst) >= need)
                m.Add(sum(lst) <= room)
        if not feasible:
            log("  par bez kandidata — mijenjam susjedstvo")
            continue
        for t, lst in by_tr.items():
            if len(lst) > 1:
                m.AddAtMostOne(lst)
        for k in range(1, 46):
            m.Add(sum(by_num[k]) >= 13 - fixed_freq[k])
            m.Add(sum(by_num[k]) <= 14 - fixed_freq[k])
        # savjet: trenutne oslobođene kombinacije
        cur_free = {blocks[i] for i in free_idx}
        for i, c in enumerate(cands):
            if c in cur_free:
                m.AddHint(x[i], 1)
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = time_limit
        solver.parameters.num_search_workers = workers
        solver.parameters.random_seed = seed + rnd
        t0 = time.time()
        st = solver.Solve(m)
        log(f"  CP-SAT {solver.StatusName(st)} za {time.time() - t0:.0f}s")
        if st in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            new = [cands[i] for i in range(len(cands)) if solver.Value(x[i])]
            blocks = fixed + new
            return lns_repair_check(blocks, log) or blocks
        free_k = min(free_k + 5, 45)
    return blocks


def lns_repair_check(blocks, log):
    pairs = Counter(p for c in blocks for p in itertools.combinations(c, 2))
    freq = Counter(x for c in blocks for x in c)
    unc = sum(1 for p in itertools.combinations(range(1, 46), 2) if pairs[p] == 0)
    tr = Counter(t for c in blocks for t in itertools.combinations(c, 3))
    log(f"  rezultat: {len(blocks)} blokova, nepokriveno {unc}, max par {max(pairs.values())}, "
        f"freq {min(freq.values())}–{max(freq.values())}, max trojka {max(tr.values())}")
    return None


if __name__ == "__main__":
    import json, sys
    import generate as G
    d = json.load(open(C.HERE / "out" / "set231.json"))
    others = [tuple(c) for k in ("2", "3", "4") for c in d["blokovi"][k]]
    b1 = [tuple(c) for c in d["blokovi"]["1"]]
    used = set()
    for c in others + G.PLAYED["pdf_10_fiksnih"]:
        used.update(C.triples(c))
    rng = random.Random(20261029)
    t0 = time.time()
    cands = enumerate_candidates(used, G.FORBIDDEN, G.SUM_LO, G.SUM_HI, rng)
    print("kandidata:", len(cands), "u %.0fs" % (time.time() - t0))
    res = lns_repair(b1, cands, time_limit=float(sys.argv[1]) if len(sys.argv) > 1 else 90,
                     free_k=int(sys.argv[2]) if len(sys.argv) > 2 else 20, pair_cap=int(sys.argv[3]) if len(sys.argv) > 3 else 2)
    lns_repair_check(res, print)
    print("max presjek sa svima:", C.max_pairwise_overlap(res + others))
    json.dump([list(c) for c in res], open(C.HERE / "out" / "_lns_b1.json", "w"))
