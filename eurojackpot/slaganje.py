"""§6–§9 – Plečkov filtar, slaganje seta, točne vjerojatnosti, backtest, CSV."""
import json, os, datetime as dt, itertools
from collections import defaultdict
from zoneinfo import ZoneInfo
import numpy as np

OUT = os.path.dirname(os.path.abspath(__file__))
N_KOMB, CIJENA, KOLO = 6, 2.0, 95
EZ = json.load(open(os.path.join(OUT, 'ezoterija_danas.json')))
TEST = json.load(open(os.path.join(OUT, 'test_ezoterije.json')))
R = json.load(open(os.path.join(OUT, 'slucajni_elementi.json')))
ODIGRANO = []  # već odigrano: ništa
tez_map = {k: v['tezina'] for k, v in TEST['sustavi'].items()}
SUST = [k for k in EZ if not k.startswith('_')]
tez = {k: tez_map.get('Kamea' if k.startswith('Kamea') else k, 1) for k in SUST}

# ---------- §6 Plečkov filtar ----------
vg, pg, ve, pe = defaultdict(float), defaultdict(int), defaultdict(float), defaultdict(int)
glasaci_g, glasaci_e = defaultdict(list), defaultdict(list)
for k in SUST:
    for pos, x in enumerate(EZ[k]['glavni'][:5], 1):
        vg[x] += tez[k]; pg[x] += pos; glasaci_g[x].append(k)
    for pos, x in enumerate(EZ[k]['euro'][:2], 1):
        ve[x] += tez[k]; pe[x] += pos; glasaci_e[x].append(k)
rank_g = sorted(range(1, 51), key=lambda x: (-vg[x], pg[x] if vg[x] else 999, x))
rank_e = sorted(range(1, 13), key=lambda x: (-ve[x], pe[x] if ve[x] else 999, x))
signal = [x for x in rank_g if vg[x] >= 2]
sum_g = [x for x in rank_g if vg[x] == 1]
sig_e = [x for x in rank_e if ve[x] >= 2]

# ---------- slaganje ----------
def susjed(x, komb):
    return any(abs(x - y) <= 1 for y in komb)

used = set(x for c in ODIGRANO for x in c[0])
komb, oznake, punila = [], [], []
pool = list(signal)
preostali_sust = list(SUST)
for c in range(N_KOMB):
    # izvor: sustav s najviše neiskorištenih signalnih brojeva među svojih prvih 5
    def score(k):
        t = [x for x in EZ[k]['glavni'][:5] if x in pool]
        return (len(t), sum(vg[x] for x in t), -SUST.index(k))
    src = max(preostali_sust, key=score)
    preostali_sust.remove(src)
    kk, pun = [], []
    for x in EZ[src]['glavni'][:5]:
        if x in pool and not susjed(x, kk) and len(kk) < 5:
            kk.append(x); pool.remove(x)
    for x in list(pool):
        if len(kk) < 5 and not susjed(x, kk):
            kk.append(x); pool.remove(x)
    while len(kk) < 5:  # punilo: najviše glasova (šum prije nule), najmanje korišten, najudaljeniji, manji
        iskor = set(y for z in komb for y in z) | used | set(pool)
        kand = [x for x in range(1, 51) if x not in iskor and x not in kk and not susjed(x, kk)]
        x = min(kand, key=lambda x: (-vg[x], -min([abs(x - y) for y in kk] or [99]), x))
        kk.append(x); pun.append(x)
    komb.append(sorted(kk)); oznake.append(src); punila.append(pun)
assert not pool, pool

# euro: n = 6 -> svih 12 točno jednom, parovi nesusjedni
def moze_upariti(ost):
    if not ost: return True
    a = ost[0]
    return any(abs(a - b) != 1 and moze_upariti([x for x in ost[1:] if x != b]) for b in ost[1:])

eu, ostalo = [], list(range(1, 13))
for c in range(N_KOMB):
    src = oznake[c]
    pref = [x for x in EZ[src]['euro'][:2]]
    if len(pref) == 2 and all(x in ostalo for x in pref) and abs(pref[0] - pref[1]) != 1 and \
            moze_upariti([x for x in ostalo if x not in pref]):
        par = sorted(pref)
    else:
        kand = [p for p in itertools.combinations(sorted(ostalo, key=lambda x: rank_e.index(x)), 2)
                if abs(p[0] - p[1]) != 1 and moze_upariti([x for x in ostalo if x not in p])]
        # prednost: euro broj izvora, zatim zbroj glasova, zatim rang
        par = sorted(max(kand, key=lambda p: (sum(x in pref for x in p), ve[p[0]] + ve[p[1]], -rank_e.index(p[0]) - rank_e.index(p[1]))))
    eu.append(par); ostalo = [x for x in ostalo if x not in par]

SET = [(g, e) for g, e in zip(komb, eu)]
# provjera §3
prov = dict(susjedni_glavni=[c for c, (g, e) in enumerate(SET, 1) if any(b - a == 1 for a, b in zip(g, g[1:]))],
            max_zajednickih=max(len(set(a[0]) & set(b[0])) for a, b in itertools.combinations(SET + ODIGRANO, 2)),
            susjedni_euro=[c for c, (g, e) in enumerate(SET, 1) if abs(e[0] - e[1]) == 1],
            euro_parovi_jedinstveni=len(set(tuple(e) for _, e in SET)) == len(SET),
            euro_pokrivenost=sorted(x for _, e in SET for x in e) == list(range(1, 13)),
            zbroj_glasova=sum(vg[x] for g, _ in SET for x in g),
            maks_moguci_zbroj_30=sum(sorted(vg.values(), reverse=True)[:30]))
print(prov)

# ---------- §7 točne vjerojatnosti ----------
C = np.array(list(itertools.combinations(range(50), 5)), dtype=np.uint64)
DM = np.zeros(len(C), np.uint64)
for j in range(5):
    DM |= np.left_shift(np.uint64(1), C[:, j])
del C
EP = list(itertools.combinations(range(12), 2))
NG, NE = len(DM), len(EP)
assert NG == 2118760 and NE == 66

def maska(g):
    return np.uint64(sum(1 << (x - 1) for x in g))

def bits66(f):
    return sum(1 << j for j, p in enumerate(EP) if f(p))

def vjerojatnosti(S):
    n = len(S)
    m = np.stack([np.bitwise_count(DM & maska(g)) for g, _ in S])  # n x NG
    es = [set(x - 1 for x in e) for _, e in S]
    B1 = [bits66(lambda p, s=s: len(s & set(p)) >= 1) for s in es]
    B2 = [bits66(lambda p, s=s: len(s & set(p)) == 2) for s in es]
    w = (1 << np.arange(n)).astype(np.int64)[:, None]
    A = (m >= 3).any(0)
    S2 = ((m == 2) * w).sum(0); S1 = ((m == 1) * w).sum(0)
    tab = np.zeros(1 << (2 * n), np.int64)
    for s2 in range(1 << n):
        u2 = 0
        for c in range(n):
            if s2 >> c & 1: u2 |= B1[c]
        for s1 in range(1 << n):
            u = u2
            for c in range(n):
                if s1 >> c & 1: u |= B2[c]
            tab[s2 | (s1 << n)] = bin(u).count('1')
    idx = (S2 | (S1 << n))[~A]
    any_cnt = A.sum() * NE + tab[idx].sum()
    # visoki razredi
    S5 = ((m == 5) * w).sum(0); S4 = ((m == 4) * w).sum(0)
    def tabB(B):
        t = np.zeros(1 << n, np.int64)
        for s in range(1 << n):
            u = 0
            for c in range(n):
                if s >> c & 1: u |= B[c]
            t[s] = bin(u).count('1')
        return t
    t2 = tabB(B2)
    any5 = S5 > 0
    p52 = t2[S5].sum()
    top = any5.sum() * NE + t2[S4[~any5]].sum()
    tot = NG * NE
    # očekivani broj dobitnih kombinacija (zbroj pojedinačnih)
    pm = np.bincount(m.ravel(), minlength=6) / NG / n
    return dict(P_5p2=p52 / tot, P_top=top / tot, P_bilo_koji=any_cnt / tot,
                ocek_dobitnih=n * P1, jedan_na_bilo_koji=tot / any_cnt)

from math import comb
pmh = {k: comb(5, k) * comb(45, 5 - k) / comb(50, 5) for k in range(6)}
peh = {k: comb(2, k) * comb(10, 2 - k) / comb(12, 2) for k in range(3)}
RAZ = [(5, 2), (5, 1), (5, 0), (4, 2), (4, 1), (4, 0), (3, 2), (2, 2), (3, 1), (3, 0), (1, 2), (2, 1)]
P1 = sum(pmh[a] * peh[b] for a, b in RAZ)

V_set = vjerojatnosti(SET + ODIGRANO)
print('SET', V_set)

# redoslijed igranja (pohlepno)
red, ost, koraci = [], list(range(N_KOMB)), []
while ost:
    best = max(ost, key=lambda c: (round(vjerojatnosti([SET[i] for i in red + [c]])['P_bilo_koji'], 12), -c))
    red.append(best); ost.remove(best)
    v = vjerojatnosti([SET[i] for i in red])
    koraci.append(dict(k=len(red), komb=best + 1, ulog=len(red) * CIJENA, **v))
    print(koraci[-1])

# lokalna zamjena: provjera (set s disjunktnim glavnima i disjunktnim euro parovima je invarijantan na permutacije brojeva)
zamjena = []
rngz = np.random.default_rng(R['numpy_seed'] + 1)
for _ in range(20):
    c = int(rngz.integers(N_KOMB)); i = int(rngz.integers(5))
    slob = [x for x in range(1, 51) if all(x not in g for g, _ in SET)]
    x = int(rngz.choice(slob))
    g2 = sorted([y for y in SET[c][0] if y != SET[c][0][i]] + [x])
    if any(b - a == 1 for a, b in zip(g2, g2[1:])): continue
    S2 = [s if j != c else (g2, s[1]) for j, s in enumerate(SET)]
    d = vjerojatnosti(S2)['P_bilo_koji'] - V_set['P_bilo_koji']
    zamjena.append((c + 1, SET[c][0][i], x, d))
max_dobit_zamjene = max(z[3] for z in zamjena)
print('max dobit zamjene (pp)', max_dobit_zamjene * 100)

# 1000 slučajnih setova bez pravila
rng = np.random.default_rng(R['numpy_seed'] + 2)
rand_sets = [[(sorted((rng.choice(50, 5, replace=False) + 1).tolist()), sorted((rng.choice(12, 2, replace=False) + 1).tolist()))
              for _ in range(N_KOMB)] for _ in range(1000)]
rp = np.array([vjerojatnosti(s)['P_bilo_koji'] for s in rand_sets])
percentil = float((rp < V_set['P_bilo_koji']).mean() * 100 + (rp == V_set['P_bilo_koji']).mean() * 50)
print('percentil P(bilo koji)', percentil, rp.min(), rp.mean(), rp.max())

# ---------- §8 backtest ----------
hist = [l.strip().split(';') for l in open(os.path.join(OUT, 'povijest_eurojackpot_2022-03-25_2026-11-24.csv')).readlines()[1:]]
hist = [(r[0], set(map(int, r[1:6])), set(map(int, r[6:8]))) for r in hist]
def backtest(S):
    raz = defaultdict(int); kola = 0
    for _, G, E in hist:
        w = False
        for g, e in S:
            k = (len(G & set(g)), len(E & set(e)))
            if k in RAZ: raz[k] += 1; w = True
        kola += w
    return raz, kola
bt_raz, bt_kola = backtest(SET)
bt_rand = np.array([sum(backtest(s)[0].values()) for s in rand_sets])
bt_tot = sum(bt_raz.values())
bt_pct = float((bt_rand < bt_tot).mean() * 100 + (bt_rand == bt_tot).mean() * 50)
print('backtest', dict(bt_raz), bt_tot, bt_kola, 'percentil', bt_pct, 'oček.', len(hist) * N_KOMB * P1)

# ---------- §9 CSV ----------
sada = dt.datetime.now(ZoneInfo('Europe/Zagreb')).strftime('%Y-%m-%d %H:%M')
hdr = 'ID;izvor;datum_i_sat_izrade;kolo;sustav;G1;G2;G3;G4;G5;E1;E2;punila;seed;pogoci_glavni;pogoci_euro;razred;isplata'
rows = []
for c, (g, e) in enumerate(SET):
    rows.append(';'.join([f"EJ-2026-11-27-K{KOLO}-{c + 1:02d}", 'Claude Code', sada, str(KOLO), oznake[c]] + list(map(str, g)) +
                         list(map(str, e)) + [','.join(f"{x}*" for x in punila[c]), str(R['numpy_seed']), '', '', '', '']))
fn = os.path.join(OUT, 'eurojackpot_2026-11-27_kolo95.csv')
open(fn, 'w').write(hdr + '\n' + '\n'.join(rows) + '\n')
ev = os.path.join(OUT, 'evidencija_kombinacija.csv')
if not os.path.exists(ev):
    open(ev, 'w').write(hdr + '\n')
postojeci = open(ev).read()
with open(ev, 'a') as f:
    for r in rows:
        if r.split(';')[0] not in postojeci: f.write(r + '\n')

json.dump(dict(glasovi_g={x: vg[x] for x in rank_g if vg[x]}, glasaci_g={x: glasaci_g[x] for x in rank_g if vg[x]},
               glasovi_e={x: ve[x] for x in rank_e if ve[x]}, glasaci_e={x: glasaci_e[x] for x in rank_e if ve[x]},
               signal=signal, sum=sum_g, signal_e=sig_e, set=SET, oznake=oznake, punila=punila, provjera=prov,
               vjerojatnosti=V_set, redoslijed=koraci, zamjena_max_pp=max_dobit_zamjene * 100,
               slucajni_setovi=dict(min=rp.min(), sred=rp.mean(), max=rp.max(), percentil=percentil),
               backtest=dict(razredi={f"{a}+{b}": v for (a, b), v in sorted(bt_raz.items(), reverse=True)}, ukupno=bt_tot,
                             kola_s_dobitkom=bt_kola, ocekivano=len(hist) * N_KOMB * P1, percentil=bt_pct,
                             slucajni_min_sred_max=[int(bt_rand.min()), float(bt_rand.mean()), int(bt_rand.max())]),
               P1=P1, rang_g=rank_g, rang_e=rank_e),
          open(os.path.join(OUT, 'rezultat.json'), 'w'), ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
print('SET'); [print(c + 1, oznake[c], g, e, punila[c]) for c, (g, e) in enumerate(SET)]
print('signal', [(x, vg[x]) for x in signal]); print('sum', sum_g); print('euro', [(x, ve[x]) for x in rank_e])
