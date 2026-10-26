"""Generator 231 kombinacije Loto 6/45 za četvrtak 29.10.2026.

Redoslijed gradnje (zbog pakiranja): Blok 3 (ezoterija, najkrući) → Blok 2 (statistika) →
Blok 4 (prosudba) → Blok 1 (pokrivenost, lokalno pretraživanje). Ispis je u redoslijedu 1-2-3-4.

Globalna pravila za svih 231 (+ 10 fiksnih iz PDF-a koje se igraju svako kolo):
  * nijedan par kombinacija ne dijeli više od 2 broja (nijedna trojka se ne ponavlja);
  * nijedna nije jednaka nekoj od 597 već odigranih/objavljenih ni nekom od 882 prošlih izvlačenja.

Pokretanje: python3 generate.py [iteracije_blok1]  → out/set231.json
"""
from __future__ import annotations

import itertools
import json
import math
import os
import random
import sys
from collections import Counter
from pathlib import Path

import common as C

SEED = 20261029
OUT = C.HERE / "out"
EZO_DIR = Path(os.environ.get("EZO_DIR", OUT))
EZO_ORDER = [
    ("kaldejska", "kaldejska numerologija"), ("gematrija", "gematrija"),
    ("hermetika", "hermetika (pentakli / kamee)"), ("iching", "I Ching"), ("lenormand", "Lenormand"),
    ("meihua", "Meihua Yishu"), ("pitagorejska", "pitagorejska numerologija"), ("tarot", "Tarot"),
    ("jyotish", "vedska astrologija (jyotiš)"), ("zapadna", "zapadna astrologija"),
]

draws = C.load_draws()
HIST = {d["nums"] for d in draws}
PLAYED = C.load_played()
FORBIDDEN = HIST | set(C.all_played())
STATS = json.load(open(OUT / "stats.json"))
PCT = STATS["sum"]["pct"]
SUM_LO, SUM_HI = PCT["10"], PCT["90"]


def band9(c):
    return Counter((x - 1) // 9 for x in c)


def is_ap(c, k=4):
    """Sadrži li aritmetički niz od k članova (vizualni uzorak na listiću)."""
    s = sorted(c)
    for sub in itertools.combinations(s, k):
        d = sub[1] - sub[0]
        if d and all(sub[i + 1] - sub[i] == d for i in range(k - 1)):
            return True
    return False


def consec(c):
    s = sorted(c)
    return sum(1 for i in range(5) if s[i + 1] - s[i] == 1)


class Builder:
    def __init__(self):
        self.pack = C.Packing()
        for c in PLAYED["pdf_10_fiksnih"]:      # igraju se svako kolo → njihove trojke su zauzete
            self.pack.covered.update(C.triples(c))
        self.rng = random.Random(SEED)
        self.blocks: dict[int, list[tuple]] = {1: [], 2: [], 3: [], 4: []}
        self.notes: dict[tuple, str] = {}
        self.log: list[str] = []

    # ------------------------------------------------------------ zajedničko
    def ok(self, c) -> bool:
        c = tuple(sorted(c))
        return len(set(c)) == 6 and all(1 <= x <= 45 for x in c) and c not in FORBIDDEN and self.pack.fits(c)

    def add(self, block: int, c, note: str = ""):
        c = tuple(sorted(c))
        self.pack.add(c)
        self.blocks[block].append(c)
        self.notes[c] = note

    def triple_free(self, x, chosen) -> bool:
        cov = self.pack.covered
        return not any(tuple(sorted((a, b, x))) in cov for a, b in itertools.combinations(chosen, 2))

    def construct(self, pools, weight, cap, cap_max, final_ok, tries=400, damp=1.0):
        """Gradi kombinaciju broj po broj: za svaki (bazen, koliko) bira brojeve težinski
        (weight[x] / (1 + damp·cap[x])), samo one čije su trojke s već izabranima slobodne."""
        rng = self.rng
        for _ in range(tries):
            chosen: list[int] = []
            for pool, cnt in pools:
                for _ in range(cnt):
                    opts = [x for x in pool if x not in chosen and cap[x] < cap_max and self.triple_free(x, chosen)]
                    if not opts:
                        break
                    ws = [weight[x] / (1 + damp * cap[x]) for x in opts]
                    chosen.append(rng.choices(opts, ws)[0])
            if len(chosen) != 6:
                continue
            c = tuple(sorted(chosen))
            if final_ok(c) and self.ok(c):
                return c
        return None

    # --------------------------------------------------------------- blok 3
    def build_block3(self):
        """5 kombinacija po metodi, po rangu metode, prva koja stane u pakiranje.
        Jedini dodatni filtar: bez očitih 'popularnih' uzoraka (aritmetički niz od 4, svih 6 ≤ 12, zbroj < 60)."""
        for slug, name in EZO_ORDER:
            path = EZO_DIR / f"ezo_{slug}.json"
            if not path.exists():
                self.log.append(f"NEDOSTAJE {path}")
                continue
            data = json.load(open(path, encoding="utf-8"))
            cands = sorted(data["kandidati"], key=lambda k: k["rang"])
            got, skipped = 0, []
            for k in cands:
                c = tuple(sorted(int(x) for x in k["brojevi"]))
                if is_ap(c, 4) or all(x <= 12 for x in c) or sum(c) < 60:
                    skipped.append(("uzorak", c)); continue
                if self.ok(c):
                    self.add(3, c, f"{name}: {k.get('obrazlozenje', '')}")
                    got += 1
                    if got == 5:
                        break
                else:
                    skipped.append(("pakiranje/odigrano", c))
            self.log.append(f"blok3 {slug}: {got}/5 iz {len(cands)} kandidata (preskočeno {len(skipped)})")
            if got < 5:
                self.log.append(f"  !!! {slug} ima samo {got}")

    # --------------------------------------------------------------- blok 2
    def build_block2(self):
        f_all = {int(k): v for k, v in STATS["freq_all"].items()}
        f_60 = {int(k): v for k, v in STATS["freq_60"].items()}
        gaps = {int(k): v for k, v in STATS["gaps"].items()}
        hot12 = STATS["hot12_60"]; cold12 = STATS["cold12_60"]; late12 = STATS["late12"]
        pareto9 = STATS["pareto9"]
        coldlate = sorted(set(cold12) | set(late12))
        top_pairs = [tuple(p) for p, n in STATS["top_pairs"][:12]]
        ALL = list(range(1, 46))

        def borda(d):
            order = sorted(ALL, key=lambda x: (-d[x], x))
            return {x: 45 - i for i, x in enumerate(order)}

        b_hot, b_all, b_late = borda(f_60), borda(f_all), borda(gaps)
        w = {x: 0.40 * b_hot[x] + 0.35 * b_all[x] + 0.25 * b_late[x] for x in ALL}
        for x in pareto9:
            w[x] *= 1.6
        self.weights2 = w

        def envelope(c):
            p = C.profile(c)
            return SUM_LO <= p["sum"] <= SUM_HI and 2 <= p["odd"] <= 4 and 3 <= p["le31"] <= 5 and not is_ap(c, 4)

        cap = Counter()
        plan = [
            ("Pareto: ≥3 od 9 najčešćih ukupno + jedan od 12 najčešćih parova", "PARETO",
             lambda c: envelope(c) and sum(x in pareto9 for x in c) >= 3),
            ("vruće-hladni mix: 3 vruća + 3 hladna/kasneća", [(hot12, 3), (coldlate, 3)], envelope),
            ("vrući (zadnjih 60 kola): 4 od 12 najčešćih", [(hot12, 4), (ALL, 2)], envelope),
            ("hladni + kašnjenje: 4 iz 12 najrjeđih / 12 najkašnjenijih", [(coldlate, 4), (ALL, 2)], envelope),
            ("Borda težine (0,40 vrući + 0,35 ukupno + 0,25 kašnjenje), modalni profil", [(ALL, 6)],
             lambda c: envelope(c) and 116 <= sum(c) <= 159 and consec(c) <= 1 and len(C.profile(c)["decades"]) >= 3),
        ]
        for name, pools, fin in plan:
            n = 0
            for j in range(10):
                if pools == "PARETO":
                    pair = list(top_pairs[j % len(top_pairs)])
                    pools_j = [(pair, 2), (pareto9, 2), (ALL, 2)]
                else:
                    pools_j = pools
                c = self.construct(pools_j, w, cap, cap_max=10, final_ok=fin,
                                   tries=1500 if pools == "PARETO" else 600, damp=0.6)
                if c is None:
                    break
                self.add(2, c, name); cap.update(c); n += 1
            self.log.append(f"blok2 {name}: {n}/10")

    # --------------------------------------------------------------- blok 4
    def build_block4(self):
        """Svjesni odabir (conscious selection): nepopularna zona radi nedijeljenja dobitka.
        Pravila: ≥4 broja >31, najviše 1 broj ≤12, bez 3 i 7 (najigraniji 'sretni' brojevi), zbroj ≥165,
        ≤3 iz istog desetljeća, bez aritmetičkog niza od 4, u ~70 % uzastopni par; 13 i 32–45 pojačani."""
        high = list(range(32, 46))
        mid = [x for x in range(13, 32)]
        low = [x for x in range(1, 13) if x not in (3, 7)]
        weight = {x: 1.0 for x in range(1, 46)}
        for x in high: weight[x] = 3.0
        weight[13] = 2.5
        for x in low: weight[x] = 0.4
        cap = Counter()

        def ok4(c):
            s = sorted(c)
            return (sum(x > 31 for x in s) >= 4 and sum(x <= 12 for x in s) <= 1 and 7 not in s and 3 not in s
                    and sum(s) >= 165 and max(Counter((x - 1) // 10 for x in s).values()) <= 3
                    and not is_ap(c, 4) and 1 <= sum(x % 2 for x in s) <= 5)

        want_consec = [True] * 22 + [False] * 9
        self.rng.shuffle(want_consec)
        for i in range(31):
            nh = self.rng.choice([4, 4, 4, 4, 5, 5, 6])
            pools = [(high, nh), (mid + low, 6 - nh)]
            c = self.construct(pools, weight, cap, cap_max=11,
                               final_ok=lambda c, i=i: ok4(c) and (consec(c) >= 1) == want_consec[i], tries=800, damp=0.8)
            if c is None:
                c = self.construct([(high, 4), (mid + low, 2)], weight, cap, cap_max=12, final_ok=ok4, tries=800, damp=0.8)
            if c is None:
                self.log.append("blok4: nema kandidata na koraku %d" % i)
                break
            self.add(4, c, "svjesni odabir: ≥4 broja >31, bez 3 i 7, zbroj ≥165" + (", uzastopni par" if consec(c) else ""))
            cap.update(c)
        self.log.append(f"blok4: {len(self.blocks[4])}/31")

    # --------------------------------------------------------------- blok 1
    def build_block1(self, iters=600_000):
        """100 kombinacija: svaki broj 13–14×, ≤2 iz istog pojasa od 9 (po mogućnosti ≥4 pojasa), svih 990 parova
        pokriveno i nijedan par više od 2×, zbroj u 10.–90. percentilu; lokalno pretraživanje uz globalno pakiranje."""
        rng = self.rng
        lo, hi = SUM_LO, SUM_HI
        pack = self.pack

        def band_ok(c):
            b = band9(c)
            return max(b.values()) <= 3 and len(b) >= 4

        def hard_ok(c):
            return band_ok(c) and c not in FORBIDDEN and pack.fits(c)

        blocks: list[tuple] = []
        freq = Counter()

        pairs = Counter()

        def construct_one():
            """Pohlepno: broj s najviše još nepokrivenih parova prema već izabranima, niskom frekvencijom i šumom."""
            for attempt in range(600):
                chosen: list[int] = []
                bands = Counter()
                noise = 3 + 0.2 * attempt
                while len(chosen) < 6:
                    best, best_s = None, -1e9
                    for x in range(1, 46):
                        bx = (x - 1) // 9
                        if x in chosen or bands[bx] >= 3 or not self.triple_free(x, chosen):
                            continue
                        if bx not in bands and len(bands) + (6 - len(chosen)) < 4:
                            pass
                        if bx in bands and len(bands) + (6 - len(chosen)) - 1 < 4:
                            continue  # mora ostati moguće doseći 4 pojasa
                        unc = sum(1 for a in chosen if pairs[tuple(sorted((a, x)))] == 0)
                        over = sum(1 for a in chosen if pairs[tuple(sorted((a, x)))] >= 2)
                        sc = 10 * unc - 4 * over - 1.5 * freq[x] + rng.random() * noise
                        if sc > best_s:
                            best, best_s = x, sc
                    if best is None:
                        break
                    chosen.append(best); bands[(best - 1) // 9] += 1
                if len(chosen) < 6:
                    continue
                c = tuple(sorted(chosen))
                if len(bands) >= 4 and (lo <= sum(c) <= hi or attempt > 300) \
                        and c not in FORBIDDEN and pack.fits(c):
                    return c
            return None

        while len(blocks) < 100:
            c = construct_one()
            if c is None:
                raise RuntimeError("blok1: početna konstrukcija nije uspjela kod %d" % len(blocks))
            pack.add(c); blocks.append(c); freq.update(c)
            pairs.update(itertools.combinations(c, 2))

        W_UNC, W_OVER, W_FREQ, W_SUM, W_ODD = 100, 15, 15, 5, 3

        def soft(c):
            return W_SUM * (0 if lo <= sum(c) <= hi else 1) + W_ODD * (0 if 2 <= sum(x % 2 for x in c) <= 4 else 1)

        def objective():
            unc = sum(1 for p in itertools.combinations(range(1, 46), 2) if pairs[p] == 0)
            over = sum(max(0, v - 2) for v in pairs.values())
            fr = sum(max(0, abs(freq[x] - 13.5) - 0.5) for x in range(1, 46))
            return W_UNC * unc + W_OVER * over + W_FREQ * fr + sum(soft(c) for c in blocks)

        def delta(old, new):
            d = 0.0
            op, np_ = set(itertools.combinations(old, 2)), set(itertools.combinations(new, 2))
            for p in op - np_:
                v = pairs[p]
                if v == 1: d += W_UNC
                if v > 2: d -= W_OVER
            for p in np_ - op:
                v = pairs[p]
                if v == 0: d -= W_UNC
                if v >= 2: d += W_OVER
            for x in set(old) ^ set(new):
                before = freq[x]; after = freq[x] - (x in old) + (x in new)
                d += W_FREQ * (max(0, abs(after - 13.5) - 0.5) - max(0, abs(before - 13.5) - 0.5))
            return d + soft(new) - soft(old)

        def apply(i, new):
            old = blocks[i]
            pack.remove(old); pack.add(new)
            for p in itertools.combinations(old, 2): pairs[p] -= 1
            for p in itertools.combinations(new, 2): pairs[p] += 1
            freq.subtract(old); freq.update(new)
            blocks[i] = new

        def problem_block():
            """Indeks kombinacije koja sadrži prekobrojan par ili prečest broj (ako postoji)."""
            bad = [i for i, c in enumerate(blocks)
                   if any(pairs[p] > 2 for p in itertools.combinations(c, 2)) or any(freq[x] > 14 for x in c)]
            return rng.choice(bad) if bad else rng.randrange(100)

        cur = objective()
        self.log.append(f"blok1 pohlepni start: cilj={cur:.0f} nepokrivenih="
                        f"{sum(1 for p in itertools.combinations(range(1, 46), 2) if pairs[p] == 0)}")

        def uncovered():
            return [p for p in itertools.combinations(range(1, 46), 2) if pairs[p] == 0]

        def build_with(fixed, exclude_idx):
            """Novi blok koji sadrži `fixed`, pohlepno dopunjen (nepokriveni parovi, niska frekvencija, šum),
            uz pakiranje bez bloka exclude_idx."""
            V = blocks[exclude_idx]
            pack.remove(V)
            result = None
            fo = Counter(V)
            for attempt in range(40):
                chosen = list(fixed)
                bands = Counter((x - 1) // 9 for x in chosen)
                if max(bands.values()) > 3 or not all(self.triple_free(x, chosen[:k]) for k, x in enumerate(chosen)):
                    break
                noise = 2 + 0.5 * attempt
                okb = True
                while len(chosen) < 6:
                    best, best_s = None, -1e9
                    for x in range(1, 46):
                        bx = (x - 1) // 9
                        if x in chosen or bands[bx] >= 3 or not self.triple_free(x, chosen):
                            continue
                        if bx in bands and len(bands) + (6 - len(chosen)) - 1 < 4:
                            continue
                        unc = sum(1 for a in chosen if pairs[tuple(sorted((a, x)))] - (a in V and x in V) == 0)
                        over = sum(1 for a in chosen if pairs[tuple(sorted((a, x)))] - (a in V and x in V) >= 2)
                        fx = freq[x] - fo[x]
                        sc = 10 * unc - 4 * over - 3 * max(0, fx - 13) + rng.random() * noise
                        if sc > best_s:
                            best, best_s = x, sc
                    if best is None:
                        okb = False
                        break
                    chosen.append(best); bands[(best - 1) // 9] += 1
                if not okb:
                    continue
                c = tuple(sorted(chosen))
                if len(bands) >= 4 and c not in FORBIDDEN and pack.fits(c) and c not in blocks:
                    result = c
                    break
            pack.add(V)
            return result

        def removal_cost(i):
            c = blocks[i]
            d = 0
            for p in itertools.combinations(c, 2):
                if pairs[p] == 1: d += W_UNC
                elif pairs[p] > 2: d -= W_OVER
            for x in c:
                d += W_FREQ * (max(0, abs(freq[x] - 1 - 13.5) - 0.5) - max(0, abs(freq[x] - 13.5) - 0.5))
            return d

        T = 6.0
        it = 0
        best_cur, best_state, last_improve = cur, (list(blocks), Counter(pairs), Counter(freq)), 0

        def restore_best():
            nonlocal cur
            bl, pr, fq = best_state
            for i, c in enumerate(blocks):
                if c != bl[i]:
                    pack.remove(c)
            for i, c in enumerate(bl):
                if blocks[i] != c:
                    pack.add(c)
            blocks[:] = list(bl); pairs.clear(); pairs.update(pr); freq.clear(); freq.update(fq)
            cur = best_cur

        for it in range(iters):
            if cur == 0:
                break
            if cur < best_cur:
                best_cur, best_state, last_improve = cur, (list(blocks), Counter(pairs), Counter(freq)), it
            if it % 10_000 == 0:
                T = max(2.0, T * 0.9)
            if it - last_improve > 60_000:
                restore_best()
                # udarac: 2 prisilna "ruin & recreate" poteza
                for _ in range(2):
                    unc = uncovered()
                    fixed = list(rng.choice(unc)) if unc else [rng.randrange(1, 46)]
                    i = rng.randrange(100)
                    new = build_with(fixed, i)
                    if new is not None:
                        d = delta(blocks[i], new); apply(i, new); cur += d
                last_improve = it
                T = 6.0
            r = rng.random()
            if r < 0.40:
                # ruin & recreate: nepokriveni par ugradi u novi blok umjesto najjeftinije žrtve
                unc = uncovered()
                if unc:
                    fixed = list(rng.choice(unc))
                else:
                    i0 = problem_block(); fixed = [rng.choice(blocks[i0])]
                costs = sorted(range(100), key=lambda i: removal_cost(i) + rng.random() * 30)
                i = rng.choice(costs[:6])
                new = build_with(fixed, i)
                if new is None:
                    continue
                d = delta(blocks[i], new)
                if d <= 0 or rng.random() < math.exp(-d / T):
                    apply(i, new); cur += d
                continue
            if 0.40 <= r < 0.65:
                # zamjena brojeva između dvije kombinacije (frekvencije ostaju iste)
                i = problem_block() if rng.random() < 0.5 else rng.randrange(100)
                j = rng.randrange(100)
                if i == j:
                    continue
                A, B = blocks[i], blocks[j]
                a, b = rng.choice(A), rng.choice(B)
                if a in B or b in A:
                    continue
                nA = tuple(sorted([t for t in A if t != a] + [b]))
                nB = tuple(sorted([t for t in B if t != b] + [a]))
                pack.remove(A); ok1 = hard_ok(nA); pack.add(A)
                if not ok1:
                    continue
                d1 = delta(A, nA); apply(i, nA)
                pack.remove(B); ok2 = hard_ok(nB); pack.add(B)
                if not ok2:
                    apply(i, A); continue
                d2 = delta(B, nB)
                if d1 + d2 <= 0 or rng.random() < math.exp(-(d1 + d2) / T):
                    apply(j, nB); cur += d1 + d2
                else:
                    apply(i, A)
                continue
            if r < 0.85:
                # ciljani potez: nepokriveni par (x, y) → ubaci y u kombinaciju koja sadrži x
                unc = uncovered()
                if unc:
                    x, y = rng.choice(unc)
                    if rng.random() < 0.5:
                        x, y = y, x
                    withx = [i for i, c in enumerate(blocks) if x in c and y not in c]
                    if not withx:
                        continue
                    cands = []
                    for i in withx:
                        old = blocks[i]
                        pack.remove(old)
                        for z in old:
                            if z == x:
                                continue
                            new = tuple(sorted([t for t in old if t != z] + [y]))
                            if hard_ok(new):
                                cands.append((delta(old, new), i, new))
                        pack.add(old)
                    if not cands:
                        continue
                    cands.sort(key=lambda t: t[0])
                    d, i, new = cands[0] if rng.random() < 0.8 else rng.choice(cands)
                    if d <= 0 or rng.random() < math.exp(-d / T):
                        apply(i, new); cur += d
                    continue
                i = problem_block()
            else:
                i = problem_block() if rng.random() < 0.5 else rng.randrange(100)
            old = blocks[i]
            pos = rng.randrange(6)
            pack.remove(old)
            cands = []
            for x in range(1, 46):
                if x in old:
                    continue
                new = list(old); new[pos] = x; new = tuple(sorted(new))
                if hard_ok(new):
                    cands.append((delta(old, new), new))
            pack.add(old)
            if not cands:
                continue
            cands.sort(key=lambda t: t[0])
            d, new = cands[0] if rng.random() < 0.8 else rng.choice(cands)
            if d <= 0 or rng.random() < math.exp(-d / T):
                apply(i, new); cur += d
        if cur > best_cur:
            restore_best()
        unc = [p for p in itertools.combinations(range(1, 46), 2) if pairs[p] == 0]
        chk = objective()
        self.log.append(f"blok1: cilj={cur:.0f} (provjera {chk:.0f}) nepokrivenih parova={len(unc)} "
                        f"max par={max(pairs.values())} freq min/max={min(freq.values())}/{max(freq.values())} "
                        f"pojasevi≥4: {sum(1 for c in blocks if len(band9(c)) >= 4)}/100 iteracija={it + 1}")
        for c in blocks:
            self.blocks[1].append(c)
            self.notes[c] = "pokrivenost 1–45"
        self.block1_pairs = pairs

    # ------------------------------------------------------------------ out
    def dump(self, path=OUT / "set231.json"):
        allb = self.blocks[1] + self.blocks[2] + self.blocks[3] + self.blocks[4]
        data = {
            "datum": "2026-10-29", "vrijeme": "20:00 CET", "seed": SEED,
            "blokovi": {str(k): [list(c) for c in v] for k, v in self.blocks.items()},
            "napomene": {" ".join(map(str, c)): n for c, n in self.notes.items()},
            "sve": [list(c) for c in allb],
            "log": self.log,
        }
        json.dump(data, open(path, "w"), indent=1, ensure_ascii=False)
        return allb


def main():
    b = Builder()
    b.build_block3()
    b.build_block2()
    b.build_block4()
    b.build_block1(iters=int(sys.argv[1]) if len(sys.argv) > 1 else 600_000)
    allb = b.dump()
    print("\n".join(b.log))
    print("ukupno", len(allb), "različitih", len(set(allb)), "max presjek", C.max_pairwise_overlap(allb))


if __name__ == "__main__":
    main()
