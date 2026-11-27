"""§5 – empirijski test determinističkih sustava (prema protocol.md)."""
import json, os, math, datetime as dt
import numpy as np
from math import comb
from scipy.stats import norm
import sustavi as S

OUT = os.path.dirname(os.path.abspath(__file__))
hist = [l.strip().split(';') for l in open(os.path.join(OUT, 'povijest_eurojackpot_2022-03-25_2026-11-24.csv')).readlines()[1:]]
hist = [(dt.date.fromisoformat(r[0]), list(map(int, r[1:6])), list(map(int, r[6:8]))) for r in hist]
R = json.load(open(os.path.join(OUT, 'slucajni_elementi.json')))

# točne vjerojatnosti razreda po listiću
pm = {m: comb(5, m) * comb(45, 5 - m) / comb(50, 5) for m in range(6)}
pe = {e: comb(2, e) * comb(10, 2 - e) / comb(12, 2) for e in range(3)}
RAZREDI = [(5, 2), (5, 1), (5, 0), (4, 2), (4, 1), (4, 0), (3, 2), (2, 2), (3, 1), (3, 0), (1, 2), (2, 1)]
P_WIN = sum(pm[m] * pe[e] for m, e in RAZREDI)
VM, VE = 5 * 0.1 * 0.9 * 45 / 49, 2 * (2 / 12) * (10 / 12) * (10 / 11)

def hl_kolo(i):
    d = hist[i][0]
    return sum(1 for x, _, _ in hist[:i] if x.year == d.year) + 1

SUST = ['Sinkronicitet', 'Arhetipovi', 'Meihua', 'Jyotiš', 'Zapadna', 'Kamea', 'Kabala', 'Numerologija', 'Planetarni sat']

def listici(i):
    d = hist[i][0]
    prev5 = hist[i - 5:i][::-1]
    out = {}
    raw, er, _ = S.sinkronicitet(d, prev5, hl_kolo(i)); out['Sinkronicitet'] = S.popis(raw, er)
    out['Arhetipovi'] = S.popis(S.ARHETIPOVI)
    out['Meihua'] = S.popis(S.meihua(d)[0])
    out['Jyotiš'] = S.popis(S.jyotish(d)[0])
    out['Zapadna'] = S.popis(S.zapadna(d)[0])
    out['Kamea'] = S.popis(S.kamea(d)[0])
    out['Kabala'] = S.popis(S.kabala(d)[0])
    out['Numerologija'] = S.popis(S.numerologija(d)[0])
    out['Planetarni sat'] = S.popis(S.planetarni_sat(d)[0])
    return {k: S.listic(*v) for k, v in out.items()}

idx = list(range(5, len(hist)))
N = len(idx); ntr = N // 2
gm = {k: [] for k in SUST}; ge = {k: [] for k in SUST}; gw = {k: [] for k in SUST}
for i in idx:
    L = listici(i)
    G, E = set(hist[i][1]), set(hist[i][2])
    for k, (g, e) in L.items():
        m = len(G & set(g)); x = len(E & set(e))
        gm[k].append(m); ge[k].append(x); gw[k].append((m, x) in RAZREDI)

def z_of(T, n):
    return (T - n * 5 / 6) / math.sqrt(n * (VM + VE))

rez = {}
for k in SUST:
    m = np.array(gm[k]); e = np.array(ge[k]); w = np.array(gw[k]); t = m + e
    ztr = z_of(t[:ntr].sum(), ntr); zte = z_of(t[ntr:].sum(), N - ntr)
    ptr = 2 * norm.sf(abs(ztr))
    pte = norm.sf(zte) if ztr >= 0 else norm.cdf(zte)
    tez = 2 if (ptr < 0.05 / len(SUST) and pte < 0.05) else 1
    zall = z_of(t.sum(), N)
    rez[k] = dict(N=N, glavni_po_kolu=round(m.mean(), 4), euro_po_kolu=round(e.mean(), 4), pogodaka_po_kolu=round(t.mean(), 4),
                  z_glavni=round((m.sum() - N * 0.5) / math.sqrt(N * VM), 3), z_euro=round((e.sum() - N / 3) / math.sqrt(N * VE), 3),
                  z_ukupno=round(zall, 3), p_ukupno_bonf=min(1, round(2 * norm.sf(abs(zall)) * len(SUST), 4)),
                  dobitni=int(w.sum()), ocekivano_dobitnih=round(N * P_WIN, 2),
                  z_trening=round(ztr, 3), p_trening=round(ptr, 4), p_trening_bonf=min(1, round(ptr * len(SUST), 4)),
                  z_test=round(zte, 3), p_test_isti_smjer=round(pte, 4), tezina=tez)
    print(k, rez[k])

# kontrola: 1000 fiksnih slučajnih listića
rng = np.random.default_rng(R['numpy_seed'])
K = 1000
tick_g = np.array([np.sort(rng.choice(50, 5, replace=False) + 1) for _ in range(K)])
tick_e = np.array([np.sort(rng.choice(12, 2, replace=False) + 1) for _ in range(K)])
Dg = np.zeros((N, 51), bool); De = np.zeros((N, 13), bool)
for r, i in enumerate(idx):
    Dg[r, hist[i][1]] = True; De[r, hist[i][2]] = True
hits = Dg[:, tick_g].sum(2) + De[:, tick_e].sum(2)  # N x K
ztr_r = np.array([z_of(hits[:ntr, j].sum(), ntr) for j in range(K)])
zte_r = np.array([z_of(hits[ntr:, j].sum(), N - ntr) for j in range(K)])
best = int(np.argmax(ztr_r))
ptr_r = 2 * norm.sf(np.abs(ztr_r))
kontrola = dict(najbolji_listic=[tick_g[best].tolist(), tick_e[best].tolist()], z_trening=round(ztr_r[best], 3),
                p_trening=float(ptr_r[best]), z_test=round(zte_r[best], 3), p_test=round(float(norm.sf(zte_r[best])), 4),
                udio_prelazi_bonf_trening=float((ptr_r < 0.05 / len(SUST)).mean()),
                udio_p05_trening=float((ptr_r < 0.05).mean()),
                korelacija_z_trening_test=round(float(np.corrcoef(ztr_r, zte_r)[0, 1]), 4))
print('KONTROLA', kontrola)
json.dump(dict(N=N, trening=ntr, test=N - ntr, razdoblje=[str(hist[idx[0]][0]), str(hist[idx[-1]][0])],
               trening_do=str(hist[idx[ntr - 1]][0]), test_od=str(hist[idx[ntr]][0]), P_dobitak_po_listicu=P_WIN,
               sustavi=rez, kontrola=kontrola), open(os.path.join(OUT, 'test_ezoterije.json'), 'w'), ensure_ascii=False, indent=1)
