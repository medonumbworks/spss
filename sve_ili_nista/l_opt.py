# -*- coding: utf-8 -*-
"""§5–§6 ponovno: L1–L3 uz više greedy pokretanja + lokalnu zamjenu (E1–E3, S1–S3 fiksni iz CSV-a).
python3 -I l_opt.py <sin csv> <kola csv> <rezultati.json> <sekunde>"""
import sys, json, time, secrets, itertools
from math import comb
import numpy as np, pandas as pd
SIN, KOLA, RJ, BUD = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
RNG = secrets.SystemRandom()
s = pd.read_csv(SIN, sep=';')
B = [f'B{i}' for i in range(1, 12)]
fixed = [list(map(int, r)) for r in s[s['#'].str[0].isin(['E', 'S'])][B].values]
meta = s[s['#'].str[0].isin(['E', 'S'])][['#', 'tip', 'sustav']].values.tolist()
TOTAL = comb(22, 11)
ALL = np.arange(1 << 22, dtype=np.uint32); ALL = ALL[np.bitwise_count(ALL) == 11]
WINK = (0, 1, 2, 3, 8, 9, 10, 11)
WL = np.zeros(12, bool); WL[list(WINK)] = True
full = (1 << 22) - 1
def bm(c): return sum(1 << (v - 1) for v in c)
def winarr(c): return WL[np.bitwise_count(ALL & np.uint32(bm(c)))]
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
prevL = [list(map(int, r)) for r in s[s['#'].str[0] == 'L'][B].values]
log = []; best, bestv = prevL, total(prevL)
while time.time() < TL:
    L = local(greedy(800), TL); v = total(L); log.append(v)
    if v > bestv: best, bestv = [c[:] for c in L], v
combos = fixed + best
assert all(ok(combos[i], combos[:i] + combos[i + 1:]) or i < 6 for i in range(9))
ovs = [len(set(a) & set(b)) for a, b in itertools.combinations(combos, 2)]
ms = {bm(c) for c in combos}
chk = dict(duplikati=len(ms) != 9, komplementi=any((full ^ m) in ms for m in ms), ov_min=min(ovs), ov_max=max(ovs))
cls = {}
for nm, ks in [('11/0', (11, 0)), ('10/1', (10, 1)), ('9/2', (9, 2)), ('8/3', (8, 3))]:
    a = np.zeros(TOTAL, bool)
    for c in combos: a |= np.isin(np.bitwise_count(ALL & np.uint32(bm(c))), ks)
    cls[nm] = int(a.sum())
anyw = total(best)
# 1000 slučajnih setova (bez komplemenata i duplikata)
def rset():
    while True:
        st = [rc() for _ in range(9)]; m = {bm(c) for c in st}
        if len(m) == 9 and not any((full ^ x) in m for x in m): return st
RS = [rset() for _ in range(1000)]
rsa = []
for st in RS:
    a = np.zeros(TOTAL, bool)
    for c in st: a |= winarr(c)
    rsa.append(int(a.sum()))
rsa = np.array(rsa)
fixed_only = int(cov_fixed.sum())
# backtest
k = pd.read_csv(KOLA, sep=';', encoding='utf-8-sig').dropna(subset=B)
DM = np.array([bm(list(map(int, r))) for r in k[B].values], dtype=np.uint32); N = len(DM)
bt = []
for nm, c in zip([m[0] for m in meta] + ['L1', 'L2', 'L3'], combos):
    h = np.bitwise_count(DM & np.uint32(bm(c)))
    dist = {kk: int((h == kk).sum()) for kk in range(12)}
    cl = {n2: int(sum(dist[x] for x in ks)) for n2, ks in [('11/0', (11, 0)), ('10/1', (10, 1)), ('9/2', (9, 2)), ('8/3', (8, 3))]}
    bt.append(dict(id=nm, dist=dist, cls=cl, W=sum(cl.values())))
setW = sum(b['W'] for b in bt)
rsW = np.array([sum(int(WL[np.bitwise_count(DM & np.uint32(bm(c)))].sum()) for c in st) for st in RS])
draws_any = int(sum(1 for d in DM if any(WL[bin(int(d) & bm(c)).count('1')] for c in combos)))
rsD = np.array([sum(1 for d in DM if any(WL[bin(int(d) & bm(c)).count('1')] for c in st)) for st in RS[:200]])
P_WIN = sum(comb(11, x) * comb(11, 11 - x) for x in WINK) / TOTAL
out = dict(restarts=len(log), restart_values=log, L=best, any=anyw, fixed_only=fixed_only, chk=chk, cls=cls,
           P_SiN=cls['11/0'], E_win=9 * P_WIN, rs=dict(mean=float(rsa.mean()), sd=float(rsa.std()), min=int(rsa.min()), max=int(rsa.max()),
           pct=float((rsa < anyw).mean() * 100 + (rsa == anyw).mean() * 50)),
           bt=bt, setW=setW, expW=9 * N * P_WIN, rsW=dict(mean=float(rsW.mean()), sd=float(rsW.std()), pct=float((rsW < setW).mean() * 100 + (rsW == setW).mean() * 50)),
           draws_any=draws_any, draws_any_rand200=dict(mean=float(rsD.mean()), pct=float((rsD < draws_any).mean() * 100 + (rsD == draws_any).mean() * 50)), N=N)
R = json.load(open(RJ, encoding='utf-8')); R['L_opt'] = out
json.dump(R, open(RJ, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
with open(SIN, 'w', encoding='utf-8') as fh:
    fh.write('#;tip;sustav;' + ';'.join(B) + '\n')
    for m, c in zip(meta, fixed): fh.write(f"{m[0]};{m[1]};{m[2]};" + ';'.join(map(str, c)) + '\n')
    for j, c in enumerate(best): fh.write(f"L{j+1};logička;maks. pokrivenost (greedy + lokalna zamjena, više pokretanja);" + ';'.join(map(str, c)) + '\n')
print(json.dumps({x: out[x] for x in ['restarts', 'restart_values', 'L', 'any', 'fixed_only', 'chk', 'cls', 'rs', 'setW', 'expW', 'rsW', 'draws_any', 'draws_any_rand200']}, ensure_ascii=False))
for b in bt: print(b)
