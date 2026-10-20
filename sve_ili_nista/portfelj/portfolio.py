"""Egzaktna optimizacija portfelja listića za Sve ili Ništa (11/22).

Nabraja svih 705 432 izvlačenja. Za zadanih 10 listića traži 3 dodatna koja maksimiziraju
P(barem jedan dobitak) = |unija dobitnih skupova| / 705 432 (primarno), zatim |unija skupova ≥260 €| (sekundarno).
Pretraga: egzaktni pohlepni odabir + koordinatni uspon sa zamjenama, s egzaktnim odbacivanjem kandidata
preko gornje međe (dobitak ≤ 6294 − preklapanje s unijom 10). Više početaka iz najboljih prvih izbora.
"""
import sys, json, time, itertools, math
import numpy as np
import os
W = os.environ.get('SIN_WORK', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = f'{W}/portfolio'
sys.path.insert(0, W); import sin_lib as L
t0 = time.time()
def log(*a): print(f'[{time.time()-t0:7.1f}s]', *a, flush=True)

TOTAL = 705432; PRICE = 1.30
PAY = np.array([80000, 260, 26, 0, 0, 0, 0, 0, 0, 26, 260, 80000], dtype=np.int64)   # isplata po broju pogodaka 0..11
LUT_ALL = (PAY > 0)                      # bilo koji dobitak: 0,1,2,9,10,11
LUT_HI = (PAY >= 260)                    # ≥260 €: 0,1,10,11
LUT_JP = (PAY >= 80000)                  # 0,11
SINGLE_ALL, SINGLE_HI, SINGLE_JP = 6294, 244, 2   # ≥260 € uključuje i glavni dobitak (242 + 2)
popcount = np.bitwise_count

def mask(t): return np.uint32(sum(1 << (n - 1) for n in t))
def unmask(m): return [n for n in range(1, 23) if (int(m) >> (n - 1)) & 1]
def comp_mask(m): return np.uint32((1 << 22) - 1) ^ m

# --- sva izvlačenja kao 22-bitni brojevi
idx = np.fromiter(itertools.chain.from_iterable(itertools.combinations(range(22), 11)), dtype=np.int8, count=TOTAL * 11).reshape(TOTAL, 11)
DRAWS = (np.left_shift(np.uint32(1), idx.astype(np.uint32))).sum(1).astype(np.uint32)
assert len(np.unique(DRAWS)) == TOTAL
log('draws enumerated')
# --- kandidati: svi 11-podskupovi koji sadrže broj 1 (predstavnici parova komplemenata)
CANDS = DRAWS[(DRAWS & np.uint32(1)) == 1]
assert len(CANDS) == 352716

def hits(tm): return popcount(DRAWS & tm)
def prize_sets(tm):
    h = hits(tm); return LUT_ALL[h], LUT_HI[h], LUT_JP[h]

def evaluate(tickets, label=''):
    """Egzaktne metrike portfelja (lista maski)."""
    k = len(tickets)
    H = np.stack([hits(t) for t in tickets])                   # k × TOTAL
    pay = PAY[H]                                                # k × TOTAL
    total = pay.sum(0)
    any_all = LUT_ALL[H].any(0); any_hi = LUT_HI[H].any(0); any_jp = LUT_JP[H].any(0)
    vals, cnts = np.unique(total, return_counts=True)
    ov = np.array([[int(popcount(a & b)) for b in tickets] for a in tickets])
    return dict(label=label, n_tickets=k, tickets=[unmask(t) for t in tickets],
                union_all=int(any_all.sum()), union_hi=int(any_hi.sum()), union_jp=int(any_jp.sum()),
                p_any_prize=any_all.sum() / TOTAL, p_hi_prize=any_hi.sum() / TOTAL, p_jackpot=any_jp.sum() / TOTAL,
                ev_total_eur=float(total.sum() / TOTAL), cost_eur=k * PRICE, ev_net_eur=float(total.sum() / TOTAL - k * PRICE),
                redundancy_all=int(k * SINGLE_ALL - any_all.sum()), redundancy_hi=int(k * SINGLE_HI - any_hi.sum()),
                payout_distribution=[dict(payout_eur=int(v), n_draws=int(c), p=c / TOTAL) for v, c in zip(vals, cnts)],
                pairwise_overlap=ov.tolist(), mean_pairwise_overlap=float(ov[np.triu_indices(k, 1)].mean()),
                p_two_or_more_prizes=float((LUT_ALL[H].sum(0) >= 2).sum() / TOTAL))

def overlap_counts(cands, U, lut, chunk=256):
    """Za svaki kandidat: broj izvlačenja iz U (niz maski) u kojima kandidat dobiva prema lut."""
    out = np.zeros(len(cands), dtype=np.int32)
    if len(U) == 0: return out
    for i in range(0, len(cands), chunk):
        c = cands[i:i + chunk]
        out[i:i + chunk] = lut[popcount(c[:, None] & U[None, :])].sum(1)
    return out

# --- postojećih 10
EX = json.load(open(f'{OUT}/existing10.json'))
T10 = [mask(t) for t in EX['tickets']]
base = evaluate(T10, 'original 10'); log('10 tickets: union_all', base['union_all'], 'union_hi', base['union_hi'], 'union_jp', base['union_jp'])
H10 = np.stack([hits(t) for t in T10])
M10_all = LUT_ALL[H10].any(0); M10_hi = LUT_HI[H10].any(0)
U10 = DRAWS[M10_all]; U10h = DRAWS[M10_hi]
forbidden = set(int(t) for t in T10) | set(int(comp_mask(t)) for t in T10)

# --- f(k): preklapanje dobitnih skupova dvaju listića s k zajedničkih brojeva
A = mask(range(1, 12)); fk = []
for k in range(12):
    B = mask(list(range(1, k + 1)) + list(range(12, 23 - k)))
    assert popcount(A & B) == k
    ha, hb = hits(A), hits(B)
    fk.append(dict(k=k, joint_prize_draws=int((LUT_ALL[ha] & LUT_ALL[hb]).sum()), joint_hi_draws=int((LUT_HI[ha] & LUT_HI[hb]).sum()),
                   union_prize_draws=int((LUT_ALL[ha] | LUT_ALL[hb]).sum())))
log('f(k):', [(d['k'], d['joint_prize_draws']) for d in fk])

# --- egzaktno preklapanje svakog kandidata s unijom 10 (jedini veliki prolaz)
log('scanning', len(CANDS), 'candidates vs', len(U10), 'winning draws of the 10 ...')
A_all = overlap_counts(CANDS, U10, LUT_ALL)
A_hi = overlap_counts(CANDS, U10h, LUT_HI)
log('scan done; min overlap', int(A_all.min()), 'max', int(A_all.max()), 'mean', float(A_all.mean()))
order = np.argsort(A_all, kind='stable')
valid = np.array([int(c) not in forbidden for c in CANDS])

def best_given(extra_masks, exclude):
    """Egzaktno najbolji kandidat uz unija10 ∪ prize(extra). Odbacivanje: dobitak ≤ 6294 − A_all."""
    if extra_masks:
        He = np.stack([hits(t) for t in extra_masks])
        Wm = LUT_ALL[He].any(0) & ~M10_all; Whm = LUT_HI[He].any(0) & ~M10_hi
        Wd = DRAWS[Wm]; Whd = DRAWS[Whm]
    else:
        Wd = DRAWS[:0]; Whd = DRAWS[:0]
    best = (-1, -1); best_i = -1; checked = 0; B = 2048
    for s in range(0, len(order), B):
        sel = order[s:s + B]
        if SINGLE_ALL - A_all[sel[0]] < best[0]: break          # nijedan daljnji kandidat ne može biti bolji (ni izjednačen)
        sel = sel[valid[sel] & ~np.isin(CANDS[sel].astype(np.int64), list(exclude))]
        if len(sel) == 0: continue
        g_all = SINGLE_ALL - A_all[sel] - overlap_counts(CANDS[sel], Wd, LUT_ALL)
        g_hi = SINGLE_HI - A_hi[sel] - overlap_counts(CANDS[sel], Whd, LUT_HI)
        checked += len(sel)
        j = np.lexsort((-g_hi, -g_all))[0]
        cand = (int(g_all[j]), int(g_hi[j]))
        if cand > best: best, best_i = cand, int(sel[j])
    return best_i, best, checked

def local_search(new, exclude0):
    """Koordinatni uspon: svaki novi listić zamijeni egzaktno najboljim uz ostala dva."""
    improved = True; rounds = 0
    while improved and rounds < 10:
        improved = False; rounds += 1
        for i in range(len(new)):
            others = [t for j, t in enumerate(new) if j != i]
            excl = exclude0 | {int(t) for t in others} | {int(comp_mask(t)) for t in others}
            cur_gain = gain_of(new[i], others)
            bi, bg, _ = best_given(others, excl)
            if bg > cur_gain and CANDS[bi] != new[i]:
                new[i] = CANDS[bi]; improved = True
    return new

def gain_of(tm, others):
    He = np.stack([hits(t) for t in others]) if others else None
    Ua = M10_all | (LUT_ALL[He].any(0) if others else False); Uh = M10_hi | (LUT_HI[He].any(0) if others else False)
    pa, ph, _ = prize_sets(tm)
    return (int((pa & ~Ua).sum()), int((ph & ~Uh).sum()))

# --- pohlepno + lokalna pretraga s više početaka
RESTARTS = 6
first_candidates = [int(i) for i in order if valid[i]][:RESTARTS]
# zapravo rangiraj prve izbore egzaktno (uz praznu uniju-dodatak gornja međa je točan dobitak)
results = []
for r, fi in enumerate(first_candidates):
    new = [CANDS[fi]]
    excl = set(forbidden) | {int(new[0]), int(comp_mask(new[0]))}
    for step in range(2):
        bi, bg, chk = best_given(new, excl)
        new.append(CANDS[bi]); excl |= {int(CANDS[bi]), int(comp_mask(CANDS[bi]))}
        log(f'restart {r}: greedy pick {len(new)} gain {bg} checked {chk}')
    new = local_search(new, set(forbidden))
    ev = evaluate(T10 + new, f'restart {r}')
    results.append((ev['union_all'], ev['union_hi'], r, new, ev))
    log(f'restart {r}: final union_all {ev["union_all"]} union_hi {ev["union_hi"]} tickets {[unmask(t) for t in new]}')
results.sort(key=lambda x: (-x[0], -x[1], x[2]))
best_new = results[0][3]; opt = evaluate(T10 + best_new, 'optimized 13')
log('BEST union_all', opt['union_all'], 'union_hi', opt['union_hi'], 'tickets', opt['tickets'][10:])

# --- usporedbe
rng = np.random.default_rng(2026)
def random_triple():
    while True:
        tr = []; ex = set(forbidden)
        while len(tr) < 3:
            m = mask(sorted(rng.choice(np.arange(1, 23), 11, replace=False).tolist()))
            if int(m) in ex: continue
            tr.append(m); ex |= {int(m), int(comp_mask(m))}
        return tr
rand = []
for _ in range(300):
    ev = evaluate(T10 + random_triple()); rand.append((ev['union_all'], ev['union_hi']))
rand = np.array(rand)
random_summary = dict(n=300, union_all_mean=float(rand[:, 0].mean()), union_all_sd=float(rand[:, 0].std()), union_all_min=int(rand[:, 0].min()), union_all_max=int(rand[:, 0].max()),
                      union_hi_mean=float(rand[:, 1].mean()), frac_random_below_optimized=float((rand[:, 0] < opt['union_all']).mean()))
log('random triples: mean union_all', random_summary['union_all_mean'], 'max', random_summary['union_all_max'])

# frekvencijski listići iz 656 kola
draws = L.load_draws(EX['dataset'])
M = np.zeros((len(draws), 22), int)
for i, d in enumerate(draws): M[i, np.array(d['nums']) - 1] = 1
def top11(score): return sorted(int(n + 1) for n in sorted(range(22), key=lambda n: (-score[n], n))[:11])
gaps = L.gaps(draws)
freq_tickets = [top11(M.sum(0)), top11(M[-30:].sum(0)), top11(np.array([gaps[n + 1] for n in range(22)]))]
freq_masks = []
for t in freq_tickets:
    m = mask(t)
    if int(m) in forbidden or int(m) in {int(x) for x in freq_masks} or int(comp_mask(m)) in {int(x) for x in freq_masks}:
        log('frequency ticket duplicates/complements an existing one:', t)
    freq_masks.append(m)
freq_ev = evaluate(T10 + freq_masks, '10 + 3 frequency-based (hot-all, hot-30, overdue)')

# "maksimalna raznolikost": maksimiziraj najmanju simetričnu razliku prema svim listićima (bez točnih komplemenata)
ex_masks = list(T10); div_masks = []
ov_all = np.stack([popcount(CANDS & t) for t in T10]).astype(np.int32)      # 10 × ncand
for _ in range(3):
    if div_masks: ov_all = np.vstack([ov_all, np.stack([popcount(CANDS & t) for t in div_masks[-1:]]).astype(np.int32)])
    dist = np.minimum(ov_all, 11 - ov_all)                   # udaljenost do listića ili njegova komplementa (manja = sličniji nekom od para)
    score = dist.min(0) * 1000 + dist.sum(0)                 # maksimiziraj najmanju "pravu" udaljenost, pa zbroj
    score[~valid] = -1
    for dm in div_masks: score[(CANDS == dm) | (CANDS == comp_mask(dm))] = -1
    bi = int(np.argmax(score)); div_masks.append(CANDS[bi])
div_ev = evaluate(T10 + div_masks, '10 + 3 maximum-diversity (max-min overlap distance)')
# naivna raznolikost: maksimalna Hammingova udaljenost (bez točnih komplemenata) -> skoro-komplementi
naive_masks = []
ov0 = np.stack([popcount(CANDS & t) for t in T10]).astype(np.int32)
for _ in range(3):
    if naive_masks: ov0 = np.vstack([ov0, popcount(CANDS & naive_masks[-1])[None, :].astype(np.int32)])
    score = (11 - ov0).min(0) * 1000 + (11 - ov0).sum(0); score[~valid] = -1
    score[(ov0 == 0).any(0)] = -1
    for dm in naive_masks: score[(CANDS == dm) | (CANDS == comp_mask(dm))] = -1
    bi = int(np.argmax(score)); naive_masks.append(CANDS[bi])
naive_ev = evaluate(T10 + naive_masks, '10 + 3 naive max-Hamming-distance (near-complements)')

out = dict(base=base, optimized=opt, new_tickets=opt['tickets'][10:], restarts=[dict(restart=r, union_all=u, union_hi=h, tickets=[unmask(t) for t in n]) for u, h, r, n, _ in results],
           random=random_summary, frequency=freq_ev, diversity=div_ev, naive_diversity=naive_ev, f_k=fk,
           scan=dict(min_overlap_with_union10=int(A_all.min()), mean_overlap=float(A_all.mean()), n_candidates=int(len(CANDS))),
           single_ticket=dict(prize_draws=SINGLE_ALL, hi_draws=SINGLE_HI, jp_draws=SINGLE_JP, ev_eur=380220 / TOTAL), seconds=time.time() - t0)
json.dump(out, open(f'{OUT}/portfolio_result.json', 'w'), indent=1)
log('saved portfolio_result.json')
for k in ('base', 'optimized', 'frequency', 'diversity', 'naive_diversity'):
    e = out[k]; print(f"{e['label']:55} union_all {e['union_all']:6d} ({e['p_any_prize']:.5%})  union_hi {e['union_hi']:5d} ({e['p_hi_prize']:.5%})  jp {e['union_jp']}  EV {e['ev_total_eur']:.4f} €  cost {e['cost_eur']:.2f}")
print('random mean', random_summary['union_all_mean'], 'max', random_summary['union_all_max'])
print('NEW TICKETS:', out['new_tickets'])
