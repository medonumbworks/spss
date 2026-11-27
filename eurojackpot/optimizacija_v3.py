"""v3 – R3/R4 (bez bazena), X3/X4 (bazen), zamjene (protocol_v3.md §4–§5). Upotreba: python optimizacija_v3.py A|B"""
import json, os, sys, time, datetime as dt
from zoneinfo import ZoneInfo
import numpy as np
import v3_jezgra as J

T_START = time.time()
TAB = sys.argv[1] if len(sys.argv) > 1 else 'A'
OUT = J.OUT
P = json.load(open(os.path.join(OUT, 'v3_privremeni.json')))
R = json.load(open(os.path.join(OUT, 'slucajni_elementi.json'))); SEED = R['numpy_seed']
v2 = json.load(open(os.path.join(OUT, 'set_v2_konacni.json')))
vg = {int(k): v for k, v in v2['glasovi_g'].items()}; ve = {int(k): v for k, v in v2['glasovi_e'].items()}
BAZEN, BAZEN_E = P['bazen'], P['bazen_e']
SVI, SVI_E = list(range(1, 51)), list(range(1, 13))
tab = {tuple(map(int, k.split('+'))): v for k, v in P['isplate'].items()}
LUT = J.lut(tab, 'med')
U = J.Uzorak(SEED + 30, 300_000)
if TAB == 'A':
    FIX = []; E0 = [(list(g), list(e)) for g, e in P['A0']]; CAP = float(sys.argv[2]) if len(sys.argv) > 2 else 70
else:
    FIX = [(list(g), list(e)) for g, e in P['uplaceni']]; E0 = [(list(g), list(e)) for g, e in P['B0']]; CAP = float(sys.argv[2]) if len(sys.argv) > 2 else 25
PRAG = 2600
fix_tot = np.zeros(U.n, np.int32)
for g, e in FIX:
    fix_tot += U.isplata(g, e, LUT)

class Stanje:
    def __init__(self, S):
        self.S = [(sorted(g), sorted(e)) for g, e in S]
        self.pay = [U.isplata(g, e, LUT) for g, e in self.S]
        self.tot = fix_tot + sum(self.pay)
    def kljuc(self, tot=None):
        t = self.tot if tot is None else tot
        return int(np.count_nonzero(t >= PRAG)), int(np.count_nonzero(t))
    def probaj(self, c, g, e):
        p = U.isplata(g, e, LUT)
        tot = self.tot - self.pay[c] + p
        return self.kljuc(tot), p, tot
    def primijeni(self, c, g, e, p, tot):
        self.S[c] = (sorted(g), sorted(e)); self.pay[c] = p; self.tot = tot

def dopusteno(S, c, g, e):
    if not J.ok_komb(g, e): return False
    drugi = FIX + [S[d] for d in range(len(S)) if d != c]
    return J.ok_prema(g, e, drugi)

def potezi(S, c, glavni, euro):
    g, e = S[c]
    for i in range(5):
        for v in glavni:
            if v in g: continue
            g2 = sorted(g[:i] + g[i + 1:] + [v])
            yield (i, g[i], v), g2, e
    for i in range(2):
        for v in euro:
            if v in e: continue
            e2 = sorted(e[:i] + e[i + 1:] + [v])
            yield (5 + i, e[i], v), g, e2

def lokalna(S0, glavni, euro, cilj, cap, rng):
    st = Stanje(S0); t0 = time.time()
    kk = (lambda k: k) if cilj == 3 else (lambda k: (k[1], k[0]))
    best = kk(st.kljuc()); popr = True
    while popr and time.time() - t0 < cap:
        popr = False
        for c in rng.permutation(len(st.S)):
            mv = list(potezi(st.S, c, glavni, euro)); rng.shuffle(mv)
            for _, g, e in mv:
                if not dopusteno(st.S, c, g, e): continue
                k, p, tot = st.probaj(c, g, e)
                if kk(k) > best:
                    st.primijeni(c, g, e, p, tot); best = kk(k); popr = True
            if time.time() - t0 > cap: break
    return st

def slucajni_start(glavni, euro, rng):
    S = []
    while len(S) < len(E0):
        g = sorted(rng.choice(glavni, 5, replace=False).tolist()); e = sorted(rng.choice(euro, 2, replace=False).tolist())
        if J.ok_komb(g, e) and J.ok_prema(g, e, FIX + S): S.append((g, e))
    return S

def pretraga(glavni, euro, cilj, seed_off):
    rng = np.random.default_rng(SEED + seed_off)
    starts = [E0] + [slucajni_start(glavni, euro, rng) for _ in range(3)]
    best = None
    for s in starts:
        st = lokalna(s, glavni, euro, cilj, CAP / 4, rng)
        k = st.kljuc(); k = k if cilj == 3 else (k[1], k[0])
        if best is None or k > best[0]: best = (k, [tuple(x) for x in st.S])
    return best[1]

ref = {}
ref['R3'] = pretraga(SVI, SVI_E, 3, 31); ref['R4'] = pretraga(SVI, SVI_E, 4, 32)
ref['X3'] = pretraga(BAZEN, BAZEN_E, 3, 33); ref['X4'] = pretraga(BAZEN, BAZEN_E, 4, 34)
ks = {k: Stanje(v).kljuc() for k, v in ref.items()}
k0 = Stanje(E0).kljuc()
print('uzorak', {k: (v[0] / U.n, v[1] / U.n) for k, v in ks.items()}, 'E0', (k0[0] / U.n, k0[1] / U.n), round(time.time() - T_START))
TOL = int(0.003 * U.n)
cilj3 = max(ks['R3'][0], ks['X3'][0]) - TOL   # R3 = najbolje pronađeno bez ograničenja bazena
cilj4 = max(ks['R4'][1], ks['X4'][1]) - TOL

def glas(i, v):
    return ve.get(v, 0) if i >= 5 else vg.get(v, 0)

def zamjene(S0, cilj, uvjet3, cilj_br):
    st = Stanje(S0); log = []
    for _ in range(40):
        k = st.kljuc()
        if (k[0] if cilj == 3 else k[1]) >= cilj_br: return st, log, True
        poz = [(c, i, (st.S[c][0] + st.S[c][1])[i]) for c in range(len(st.S)) for i in range(7)]
        razine = sorted({glas(i, v) for _, i, v in poz})
        izbor = None
        for L in razine:
            for c, i, v in poz:
                if glas(i, v) > L: continue
                g, e = st.S[c]
                for nv in (BAZEN if i < 5 else BAZEN_E):
                    if nv in (g if i < 5 else e): continue
                    g2 = sorted([x for x in g if x != v] + [nv]) if i < 5 else g
                    e2 = sorted([x for x in e if x != v] + [nv]) if i >= 5 else e
                    if not dopusteno(st.S, c, g2, e2): continue
                    k2, p, tot = st.probaj(c, g2, e2)
                    if cilj == 4 and k2[0] < uvjet3: continue
                    dob = (k2[0], k2[1]) if cilj == 3 else (k2[1], k2[0])
                    if dob <= ((k[0], k[1]) if cilj == 3 else (k[1], k[0])): continue
                    kl = (dob, glas(i, nv), -nv)
                    if izbor is None or kl > izbor[0]: izbor = (kl, c, i, v, nv, g2, e2, p, tot)
            if izbor: break
        if not izbor: return st, log, False
        _, c, i, v, nv, g2, e2, p, tot = izbor
        st.primijeni(c, g2, e2, p, tot)
        log.append(f"K{c + 1 + len(FIX)}: {'glavni' if i < 5 else 'euro'} {v} → {nv} (glas {glas(i, v)} → {glas(i, nv)})")
    k = st.kljuc()
    return st, log, (k[0] if cilj == 3 else k[1]) >= cilj_br

st3, log3, ok3 = zamjene(E0, 3, None, cilj3)
S3 = [tuple(x) for x in st3.S] if ok3 else E0
uvjet3 = cilj3 if ok3 else k0[0]
st4, log4, ok4 = zamjene(S3, 4, uvjet3, cilj4)
SF = [tuple(x) for x in st4.S] if ok4 else S3
print('korak3', ok3, log3); print('korak4', ok4, log4)

# točno
luts = {s: J.lut(tab, s) for s in ('p25', 'med', 'p75')}
toc = {}
for ime, S in [('E0', E0), ('konacni', SF)] + list(ref.items()):
    toc[ime] = J.tocno(FIX + [tuple(x) for x in S], luts); print(ime, toc[ime])
res = dict(tablica=TAB, vrijeme=dt.datetime.now(ZoneInfo('Europe/Zagreb')).isoformat(timespec='seconds'),
           uzorak=dict(n=U.n, seed=SEED + 30, kljucevi={k: v for k, v in ks.items()}, E0=k0, cilj3=cilj3, cilj4=cilj4),
           reference={k: [list(map(list, x)) for x in v] for k, v in ref.items()}, korak3=dict(uspjeh=ok3, zamjene=log3),
           korak4=dict(uspjeh=ok4, zamjene=log4), E0=E0, konacni=[list(map(list, x)) for x in SF], tocno=toc,
           provjera=J.provjera(SF, FIX), trajanje_s=round(time.time() - T_START, 1))
json.dump(res, open(os.path.join(OUT, f'v3_optimizacija_{TAB}.json'), 'w'), ensure_ascii=False, indent=1,
          default=lambda o: o.item() if hasattr(o, 'item') else str(o))
print('trajanje', res['trajanje_s'])
