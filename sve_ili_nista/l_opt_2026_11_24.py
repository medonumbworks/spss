# -*- coding: utf-8 -*-  (pokretati PRIJE dopuna_2026_11_24.py: prepisuje CSV bez stupca redoslijed)
"""§5–§6 za 24.11.2026. (prošireni l_opt.py): L1–L3 (greedy + lokalna zamjena + ponovna pokretanja),
točne vjerojatnosti, isplate iz potvrđenih iznosa (donja granica), redoslijed igranja, backtest.
python3 -I l_opt_2026_11_24.py <sin csv> <kola csv> <rezultati json> <sekunde>"""
import sys, json, time, secrets, itertools
from math import comb, ceil
from fractions import Fraction
import numpy as np, pandas as pd
SIN, KOLA, RJ, BUD = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
RNG = secrets.SystemRandom()
STAKE = Fraction(13, 10)                      # 1,30 € po kombinaciji
PAY = {'11/0': Fraction(80000), '8/3': Fraction(26, 10)}   # potvrđeno; 10/1 i 9/2 nepoznato
s = pd.read_csv(SIN, sep=';')
B = [f'B{i}' for i in range(1, 12)]
fixed = [list(map(int, r)) for r in s[s['#'].str[0].isin(['E', 'S'])][B].values]
meta = s[s['#'].str[0].isin(['E', 'S'])][['#', 'tip', 'sustav']].values.tolist()
prevL = [list(map(int, r)) for r in s[s['#'].str[0] == 'L'][B].values]
TOTAL = comb(22, 11)
ALL = np.arange(1 << 22, dtype=np.uint32); ALL = ALL[np.bitwise_count(ALL) == 11]
assert len(ALL) == TOTAL
WINK = (0, 1, 2, 3, 8, 9, 10, 11)
WL = np.zeros(12, bool); WL[list(WINK)] = True
full = (1 << 22) - 1
def bm(c): return sum(1 << (v - 1) for v in c)
def karr(c): return np.bitwise_count(ALL & np.uint32(bm(c)))
def winarr(c): return WL[karr(c)]
def runmax(c):
    return max(len(list(g)) for _, g in itertools.groupby(enumerate(sorted(c)), lambda t: t[1] - t[0]))
def ok(c, others):
    if len(set(c)) != 11: return False
    if sum(1 for v in c if v % 2 == 0) not in (5, 6) or runmax(c) > 4: return False
    return all(3 <= len(set(c) & set(o)) <= 8 for o in others)
cov_fixed = np.zeros(TOTAL, bool)
for c in fixed: cov_fixed |= winarr(c)
def rc(): return sorted(RNG.sample(range(1, 23), 11))
def total(L):
    cv = cov_fixed.copy()
    for c in L: cv |= winarr(c)
    return int(cv.sum())
def greedy(ncand):
    L = []
    for j in range(3):
        cv = cov_fixed.copy()
        for c in L: cv |= winarr(c)
        best, bg, n = None, -1, 0
        while n < ncand:
            c = rc()
            if not ok(c, fixed + L): continue
            n += 1; g = int((winarr(c) & ~cv).sum())
            if g > bg: best, bg = c, g
        L.append(best)
    return L
def local(L, tlim):
    improved = True
    while improved and time.time() < tlim:
        improved = False
        for j in range(3):
            oth = fixed + [L[i] for i in range(3) if i != j]
            cv = cov_fixed.copy()
            for i in range(3):
                if i != j: cv |= winarr(L[i])
            bc, bv = L[j], int((winarr(L[j]) & ~cv).sum())
            for o in L[j]:
                for n in range(1, 23):
                    if n in L[j]: continue
                    t = sorted([v for v in L[j] if v != o] + [n])
                    if not ok(t, oth): continue
                    v = int((winarr(t) & ~cv).sum())
                    if v > bv: bc, bv = t, v
            if bc != L[j]: L[j] = bc; improved = True
    return L
T = time.time(); TL = T + BUD
log = []; best, bestv = prevL, total(prevL); start_v = bestv
while time.time() < TL:
    L = local(greedy(800), TL); v = total(L); log.append(v)
    if v > bestv: best, bestv = [c[:] for c in L], v
combos = fixed + best
ids = [m[0] for m in meta] + ['L1', 'L2', 'L3']
for i in range(6, 9): assert ok(combos[i], combos[:i] + combos[i + 1:])
ovs = [len(set(a) & set(b)) for a, b in itertools.combinations(combos, 2)]
ms = {bm(c) for c in combos}
chk = dict(duplikati=len(ms) != 9, komplementi=any((full ^ m) in ms for m in ms), ov_min=min(ovs), ov_max=max(ovs))

# ---- točno za cijeli set
K = [karr(c) for c in combos]
W = [WL[k] for k in K]
J = [(k == 0) | (k == 11) for k in K]
C83 = [(k == 3) | (k == 8) for k in K]
UNK = [(k == 1) | (k == 10) | (k == 2) | (k == 9) for k in K]
def union(idx):
    a = np.zeros(TOTAL, bool)
    for i in idx: a |= W[i]
    return a
anyw = int(union(range(9)).sum())
cls = {}
for nm, ks in [('11/0', (11, 0)), ('10/1', (10, 1)), ('9/2', (9, 2)), ('8/3', (8, 3))]:
    a = np.zeros(TOTAL, bool)
    for k in K: a |= np.isin(k, ks)
    cls[nm] = int(a.sum())
def payout_stats(idx):
    nJ = np.zeros(TOTAL, np.int16); n83 = np.zeros(TOTAL, np.int16); unk = np.zeros(TOTAL, bool)
    for i in idx: nJ += J[i]; n83 += C83[i]; unk |= UNK[i]
    stake = STAKE * len(idx)
    need83 = ceil(stake / PAY['8/3'])          # broj 8/3 dobitaka potreban da potvrđena isplata ≥ ulog
    ge = (nJ > 0) | (n83 >= need83)
    exp_pay = (PAY['11/0'] * int(nJ.sum()) + PAY['8/3'] * int(n83.sum())) / TOTAL   # točno
    return dict(stake=float(stake), need83=need83, p_ge=int(ge.sum()), exp_pay=float(exp_pay), exp_pay_frac=str(exp_pay),
                n83_dist={str(x): int((n83 == x).sum()) for x in range(0, 10)}, p_unk_only=int((unk & ~ge).sum()), p_unk=int(unk.sum()))
PS = payout_stats(range(9))
nwin = np.zeros(TOTAL, np.int16)
for w in W: nwin += w
E_win = Fraction(int(nwin.sum()), TOTAL)

# ---- 1000 slučajnih setova (bez komplemenata i duplikata)
def rset():
    while True:
        st = [rc() for _ in range(9)]; m = {bm(c) for c in st}
        if len(m) == 9 and not any((full ^ x) in m for x in m): return st
RS = [rset() for _ in range(1000)]
rsa, rsge = [], []
for st in RS:
    a = np.zeros(TOTAL, bool); n83 = np.zeros(TOTAL, np.int8); jj = np.zeros(TOTAL, bool)
    for c in st:
        k = karr(c); a |= WL[k]; n83 += ((k == 3) | (k == 8)); jj |= (k == 0) | (k == 11)
    rsa.append(int(a.sum())); rsge.append(int((jj | (n83 >= PS['need83'])).sum()))
rsa = np.array(rsa); rsge = np.array(rsge)
def pct(arr, v): return float((arr < v).mean() * 100 + (arr == v).mean() * 50)

# ---- redoslijed igranja (pohlepno po doprinosu P(barem jedan dobitak))
order = []
pair_best = [max(int((W[i] | W[j]).sum()) for j in range(9) if j != i) for i in range(9)]
first = max(range(9), key=lambda i: (int(W[i].sum()), pair_best[i], -i))   # sve imaju istu P → presuda: najbolji par, pa redoslijed u tablici
order.append(first); U = W[first].copy()
while len(order) < 9:
    nxt = max([i for i in range(9) if i not in order], key=lambda i: (int((U | W[i]).sum()), -i))
    order.append(nxt); U |= W[nxt]
seq = []
for k in range(1, 10):
    idx = order[:k]
    pany = int(union(idx).sum())
    bestk = max(itertools.combinations(range(9), k), key=lambda sub: int(union(sub).sum()))
    bv = int(union(bestk).sum())
    ps = payout_stats(idx)
    seq.append(dict(k=k, add=ids[order[k - 1]], p_any=pany, stake=ps['stake'], need83=ps['need83'], p_ge=ps['p_ge'], exp_pay=ps['exp_pay'],
                    best_subset=[ids[i] for i in bestk], best_p=bv, greedy_optimal=pany == bv))

# ---- backtest
k = pd.read_csv(KOLA, sep=';', encoding='utf-8-sig').dropna(subset=B)
DM = np.array([bm(list(map(int, r))) for r in k[B].values], dtype=np.uint32); N = len(DM)
bt = []
for nm, c in zip(ids, combos):
    h = np.bitwise_count(DM & np.uint32(bm(c)))
    dist = {kk: int((h == kk).sum()) for kk in range(12)}
    cl = {n2: int(sum(dist[x] for x in ks)) for n2, ks in [('11/0', (11, 0)), ('10/1', (10, 1)), ('9/2', (9, 2)), ('8/3', (8, 3))]}
    bt.append(dict(id=nm, dist=dist, cls=cl, W=sum(cl.values()), pay=float(PAY['11/0'] * cl['11/0'] + PAY['8/3'] * cl['8/3'])))
setW = sum(b['W'] for b in bt); setPay = sum(b['pay'] for b in bt)
stake_bt = float(STAKE * 9 * N)
rsW, rsPay = [], []
for st in RS:
    w = 0; pay = 0.0
    for c in st:
        h = np.bitwise_count(DM & np.uint32(bm(c)))
        w += int(WL[h].sum()); pay += 80000 * int(((h == 0) | (h == 11)).sum()) + 2.6 * int(((h == 3) | (h == 8)).sum())
    rsW.append(w); rsPay.append(pay)
rsW = np.array(rsW); rsPay = np.array(rsPay)
draws_any = int(sum(1 for d in DM if any(WL[bin(int(d) & bm(c)).count('1')] for c in combos)))
P_WIN = sum(comb(11, x) * comb(11, 11 - x) for x in WINK) / TOTAL
out = dict(restarts=len(log), restart_values=log, start_value=start_v, L=best, any=anyw, fixed_only=int(cov_fixed.sum()), chk=chk, cls=cls,
           P_SiN=cls['11/0'], E_win=float(E_win), E_win_frac=str(E_win), payout=PS,
           rs=dict(mean=float(rsa.mean()), sd=float(rsa.std()), min=int(rsa.min()), max=int(rsa.max()), pct=pct(rsa, anyw)),
           rs_ge=dict(mean=float(rsge.mean()), sd=float(rsge.std()), min=int(rsge.min()), max=int(rsge.max()), pct=pct(rsge, PS['p_ge'])),
           order=[ids[i] for i in order], seq=seq, pair_best={ids[i]: pair_best[i] for i in range(9)},
           bt=bt, setW=setW, expW=9 * N * P_WIN, setPay=setPay, stake_bt=stake_bt,
           rsW=dict(mean=float(rsW.mean()), sd=float(rsW.std()), pct=pct(rsW, setW)),
           rsPay=dict(mean=float(rsPay.mean()), sd=float(rsPay.std()), pct=pct(rsPay, setPay)), draws_any=draws_any, N=N)
R = json.load(open(RJ, encoding='utf-8')); R['L_opt'] = out
json.dump(R, open(RJ, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
with open(SIN, 'w', encoding='utf-8') as fh:
    fh.write('#;tip;sustav;' + ';'.join(B) + '\n')
    for m, c in zip(meta, fixed): fh.write(f"{m[0]};{m[1]};{m[2]};" + ';'.join(map(str, c)) + '\n')
    for j, c in enumerate(best): fh.write(f"L{j+1};logička;maks. pokrivenost (greedy + lokalna zamjena, više pokretanja);" + ';'.join(map(str, c)) + '\n')
print(json.dumps({x: out[x] for x in ['restarts', 'start_value', 'L', 'any', 'fixed_only', 'chk', 'cls', 'E_win_frac', 'payout', 'rs', 'rs_ge', 'order', 'pair_best', 'setW', 'expW', 'setPay', 'stake_bt', 'rsW', 'rsPay', 'draws_any']}, ensure_ascii=False))
for r in seq: print(r)
for b in bt: print(b)
