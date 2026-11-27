"""v3 – isplate po razredima, točne vjerojatnosti (2.118.760 × 66), uzorak, provjere §4, slaganje E0."""
import json, os, re, glob, itertools, datetime as dt
import numpy as np
from vjerojatnosti_v2 import DM, EP, NG, NE, TOT, maska, euro_matrica

OUT = os.path.dirname(os.path.abspath(__file__))
PAYDIR = '/tmp/dl/pay'
RAZ = [(5, 2), (5, 1), (5, 0), (4, 2), (4, 1), (3, 2), (4, 0), (2, 2), (3, 1), (3, 0), (1, 2), (2, 1)]
LAB = {('5', '2'): (5, 2), ('5', '1'): (5, 1), ('5', None): (5, 0), ('4', '2'): (4, 2), ('4', '1'): (4, 1), ('3', '2'): (3, 2),
       ('4', None): (4, 0), ('2', '2'): (2, 2), ('3', '1'): (3, 1), ('3', None): (3, 0), ('1', '2'): (1, 2), ('2', '1'): (2, 1)}

def povijest():
    h = [l.strip().split(';') for l in open(os.path.join(OUT, 'povijest_eurojackpot_2012-03-23_2026-11-24.csv')).readlines()[1:]]
    return [(r[0], set(map(int, r[1:6])), set(map(int, r[6:8])), r[8]) for r in h]

def isplate_kola(d):
    """{razred: (iznos €, broj dobitnika)} s euro-jackpot.net; None ako stranica nije potpuna."""
    try:
        h = open(f'{PAYDIR}/{d}.html', errors='ignore').read()
    except FileNotFoundError:
        return None
    i = h.find('Prize Breakdown'); j = h.find('European Prize Breakdowns', i)
    t = re.sub(r'\s+', ' ', re.sub(r'<[^>]+>', ' | ', h[i:j if j > 0 else i + 8000]))
    out = {}
    for m in re.finditer(r'Match (\d)(?: and (\d) Euro Numbers?)? [\s|]*&euro;([\d,\.]+) [\s|]*([\d,]+)', t):
        out[LAB[(m.group(1), m.group(2))]] = (float(m.group(3).replace(',', '')), int(m.group(4).replace(',', '')))
    return out if len(out) == 12 else None

def tablice_isplata(n_zadnjih=100):
    H = povijest()
    d212 = [d for d, _, _, f in H if f == '2/12']
    zadnji = d212[-n_zadnjih:]
    P = {d: isplate_kola(d) for d in zadnji}
    nedost = [d for d, v in P.items() if v is None]
    tab = {}
    for k in RAZ:
        v = np.array([P[d][k][0] for d in zadnji if P[d] and P[d][k][1] >= 1])
        tab[k] = dict(n=len(v), p25=float(np.percentile(v, 25)), med=float(np.median(v)), p75=float(np.percentile(v, 75)))
    return tab, nedost, (zadnji[0], zadnji[-1])

def lut(tab, kljuc, clip=2600):
    """6x3 tablica isplata u centima (odrezano na clip) – indeks m*3+e."""
    L = np.zeros((6, 3), np.int32)
    for (m, e), v in tab.items():
        L[m, e] = min(int(round(v[kljuc] * 100)), clip)
    return L.ravel()

# ---------------- točno ----------------
def tocno(S, luts, pragovi=(2600,)):
    """S = [(g, e)]; luts = {ime: lut}; vraća P(5+2), P(top), P(bilo koji), P(isplata ≥ prag) po scenariju."""
    n = len(S)
    M = [np.bitwise_count(DM & maska(g)).astype(np.int32) for g, _ in S]
    E = euro_matrica([e for _, e in S])  # 66 x n
    # top razredi i 5+2 (samo izvlačenja s ≥ 4 glavna u nekoj kombinaciji)
    Mm = np.stack(M)
    sel = np.nonzero((Mm >= 4).any(0))[0]; m4 = Mm[:, sel]
    top = ((m4.T[:, None, :] == 5) | ((m4.T[:, None, :] == 4) & (E[None, :, :] == 2))).any(2).sum()
    p52 = ((m4.T[:, None, :] == 5) & (E[None, :, :] == 2)).any(2).sum()
    M3 = [m * 3 for m in M]
    rez = dict(P_5p2=p52 / TOT, P_top=top / TOT)
    for ime, L in luts.items():
        cnt_any = 0; cnt = {p: 0 for p in pragovi}
        for j in range(NE):
            tot = np.zeros(NG, np.int32)
            for c in range(n):
                tot += L[M3[c] + E[j, c]]
            cnt_any += int(np.count_nonzero(tot))
            for p in pragovi:
                cnt[p] += int(np.count_nonzero(tot >= p))
        rez[f'P_bilo_koji'] = cnt_any / TOT
        for p in pragovi:
            rez[f'P_ge_{p // 100}_{ime}'] = cnt[p] / TOT
    return rez

# ---------------- uzorak ----------------
class Uzorak:
    def __init__(self, seed, n=300_000):
        rng = np.random.default_rng(seed)
        self.gi = rng.integers(0, NG, n); self.ei = rng.integers(0, NE, n)
        self.DMs = DM[self.gi]
        em = np.array([(1 << a) | (1 << b) for a, b in EP], dtype=np.uint64)
        self.EMs = em[self.ei]; self.n = n
    def isplata(self, g, e, L):
        m = np.bitwise_count(self.DMs & maska(g)).astype(np.int32)
        x = np.bitwise_count(self.EMs & np.uint64(sum(1 << (k - 1) for k in e))).astype(np.int32)
        return L[m * 3 + x]

# ---------------- §4 ----------------
def ok_komb(g, e):
    g = sorted(g)
    return (len(set(g)) == 5 and all(1 <= x <= 50 for x in g) and all(b - a > 1 for a, b in zip(g, g[1:]))
            and len(set(e)) == 2 and abs(e[0] - e[1]) != 1 and all(1 <= x <= 12 for x in e))

def ok_prema(g, e, drugi):
    return all(len(set(g) & set(h)) <= 2 and sorted(e) != sorted(f) for h, f in drugi)

def provjera(S, uplaceni=()):
    sve = list(uplaceni) + list(S)
    pre = [len(set(a[0]) & set(b[0])) for a, b in itertools.combinations(sve, 2)]
    return dict(susjedni=[i + 1 for i, (g, e) in enumerate(S) if any(b - a == 1 for a, b in zip(sorted(g), sorted(g)[1:]))],
                max_zajednickih=max(pre), parova_s_2=sum(p == 2 for p in pre), parova_s_1=sum(p == 1 for p in pre),
                susjedni_euro=[i + 1 for i, (g, e) in enumerate(S) if abs(e[0] - e[1]) == 1],
                euro_parovi_razliciti=len({tuple(sorted(e)) for _, e in sve}) == len(sve),
                euro_brojevi_u_setu=sorted({x for _, e in sve for x in e}),
                razlicitih_glavnih=len({x for g, _ in sve for x in g}))
