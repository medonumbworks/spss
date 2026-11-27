"""v2 – vremenska stabilnost, glasovi, slaganje 8 kombinacija (protocol_v2.md D–F).
Upotreba: python slaganje_v2.py privremeni|konacni"""
import json, os, sys, time, itertools, datetime as dt
from collections import defaultdict
from zoneinfo import ZoneInfo
import numpy as np
import sustavi as S
import vjerojatnosti_v2 as V

T0 = time.time()
OUT = os.path.dirname(os.path.abspath(__file__))
FAZA = sys.argv[1] if len(sys.argv) > 1 else 'privremeni'
D = dt.date(2026, 11, 27)
N_KOMB, CIJENA, KOLO = 8, 2.0, 95
ODIGRANO = []  # „već odigrano" nije upisano → ništa
SATI = [19.0, 19.25, 19.5, 19.75, 20.0]
R = json.load(open(os.path.join(OUT, 'slucajni_elementi.json')))
SEED = R['numpy_seed']

# ---------- težine ----------
if FAZA == 'privremeni':
    tez_src = json.load(open(os.path.join(OUT, 'test_ezoterije.json')))['sustavi']
else:
    tez_src = json.load(open(os.path.join(OUT, 'test_ezoterije_v2.json')))['sustavi']
def tezina(k):
    return tez_src.get('Kamea' if k.startswith('Kamea') else k, {'tezina': 1})['tezina']

# ---------- popisi sustava u svakom trenutku ----------
hist = [l.strip().split(';') for l in open(os.path.join(OUT, 'povijest_eurojackpot_2022-03-25_2026-11-24.csv')).readlines()[1:]]
hist = [(dt.date.fromisoformat(r[0]), list(map(int, r[1:6])), list(map(int, r[6:8]))) for r in hist]
prev5 = hist[-5:][::-1]
hl = sum(1 for d, _, _ in hist if d.year == 2026) + 1
assert hl == KOLO

def popisi_u(h):
    out = {}
    raw, er, _ = S.sinkronicitet(D, prev5, hl); out['Sinkronicitet'] = S.popis(raw, er)
    out['Arhetipovi'] = S.popis(S.ARHETIPOVI)
    out['Meihua'] = S.popis(S.meihua(D, h)[0])
    out['I Ching'] = S.popis(S.iching_bacanja(R['iching'])[0])
    out['Jyotiš'] = S.popis(S.jyotish(D, h)[0])
    out['Zapadna'] = S.popis(S.zapadna(D, h)[0])
    out['Kamea (Venera)'] = S.popis(S.kamea(D)[0])
    out['Kabala'] = S.popis(S.kabala(D)[0])
    dek, _ = S.tarot_dekan(D, h); out['Tarot'] = S.popis([dek] + R['tarot'])
    L = R['lenormand']; out['Lenormand'] = S.popis(L + [L[i] + L[i + 1] for i in range(8)])
    out['Numerologija'] = S.popis(S.numerologija(D)[0])
    out['Planetarni sat'] = S.popis(S.planetarni_sat(D, h)[0])
    return out

P_T = {h: popisi_u(h) for h in SATI}
SUST = list(P_T[19.0].keys())
sustav = {}
promjene = {}
for k in SUST:
    gl, el = [], []
    for h in SATI:
        for x in P_T[h][k][0]:
            if x not in gl: gl.append(x)
        for x in P_T[h][k][1]:
            if x not in el: el.append(x)
    top_g = [set(P_T[h][k][0][:6]) for h in SATI]; top_e = [set(P_T[h][k][1][:2]) for h in SATI]
    vg = [x for x in gl if any(x in t for t in top_g)]
    ve = [x for x in el if any(x in t for t in top_e)]
    stab_g = {x: 1.0 if all(x in t for t in top_g) else 0.5 for x in vg}
    stab_e = {x: 1.0 if all(x in t for t in top_e) else 0.5 for x in ve}
    sustav[k] = dict(popis_g=gl, popis_e=el, glasa_g=vg, glasa_e=ve, stab_g=stab_g, stab_e=stab_e, tezina=tezina(k))
    pr = {str(h): (P_T[h][k][0][:6], P_T[h][k][1][:2]) for h in SATI}
    if len({json.dumps(v) for v in pr.values()}) > 1:
        promjene[k] = pr

# ---------- glasovi (Plečko) ----------
vg, pg, ve, pe = defaultdict(float), defaultdict(int), defaultdict(float), defaultdict(int)
for k, s in sustav.items():
    for x in s['glasa_g']:
        vg[x] += s['tezina'] * s['stab_g'][x]; pg[x] += s['popis_g'].index(x) + 1
    for x in s['glasa_e']:
        ve[x] += s['tezina'] * s['stab_e'][x]; pe[x] += s['popis_e'].index(x) + 1
rank_g = sorted(range(1, 51), key=lambda x: (-vg[x], pg[x] if vg[x] else 999, x))
rank_e = sorted(range(1, 13), key=lambda x: (-ve[x], pe[x] if ve[x] else 999, x))
signal = [x for x in rank_g if vg[x] >= 2]
sum_g = [x for x in rank_g if 0 < vg[x] < 2]

# ---------- slaganje ----------
def susjed(x, kk):
    return any(abs(x - y) <= 1 for y in kk)

def dopusten(x, kk, ranije):
    return x not in kk and not susjed(x, kk) and all(len((set(kk) | {x}) & set(p)) <= 1 for p in ranije)

def punilo(kk, ranije):
    kand = [x for x in range(1, 51) if dopusten(x, kk, ranije)]
    return min(kand, key=lambda x: (-vg[x], -min([abs(x - y) for y in kk] or [99]), x))

ranije = [g for g, _ in ODIGRANO]
komb, izvori, punila_l = [], [], []
# K1 = Plečko
kk, pun = [], []
for x in signal:
    if len(kk) < 5 and dopusten(x, kk, ranije): kk.append(x)
while len(kk) < 5:
    x = punilo(kk, ranije); kk.append(x); pun.append(x)
komb.append(kk); izvori.append('Plečko'); punila_l.append(pun)
# redoslijed sustava za K2–K8
def kljuc(k):
    u = [x for x in sustav[k]['glasa_g'] if x in signal]
    return (-len(u), -sum(vg[x] for x in u), k.casefold())
red_sust = sorted(SUST, key=kljuc)
for k in red_sust[:N_KOMB - 1]:
    kk, pun = [], []
    pre = ranije + komb
    for x in sustav[k]['popis_g']:
        if len(kk) < 5 and dopusten(x, kk, pre): kk.append(x)
    while len(kk) < 5:
        x = punilo(kk, pre); kk.append(x); pun.append(x)
    komb.append(kk); izvori.append(k); punila_l.append(pun)

# euro
kap = {x: (2 if x in rank_e[:4] else 1) for x in range(1, 13)}
iskor_parovi = set(tuple(sorted(e)) for _, e in ODIGRANO)

def dovrsivo(kap, br, iskor):
    """Može li se od preostalih kapaciteta složiti br parova (različiti, nesusjedni, nekorišteni)?"""
    ost = sorted(x for x, c in kap.items() for _ in range(c))
    if len(ost) != 2 * br: return False
    def rek(ost, iskor):
        if not ost: return True
        a = ost[0]
        for b in sorted(set(ost[1:])):
            if b != a and abs(a - b) != 1 and (a, b) not in iskor:
                o2 = list(ost[1:]); o2.remove(b)
                if rek(o2, iskor | {(a, b)}): return True
        return False
    return rek(ost, iskor)

eu = []
for c in range(N_KOMB):
    if izvori[c] == 'Plečko':
        top2, el = rank_e[:2], rank_e
    else:
        top2, el = sustav[izvori[c]]['glasa_e'], sustav[izvori[c]]['popis_e']
    kand = []
    for a, b in itertools.combinations(range(1, 13), 2):
        if kap[a] > 0 and kap[b] > 0 and abs(a - b) != 1 and (a, b) not in iskor_parovi:
            k2 = dict(kap); k2[a] -= 1; k2[b] -= 1
            if dovrsivo(k2, N_KOMB - c - 1, iskor_parovi | {(a, b)}):
                pos = lambda x: el.index(x) + 1 if x in el else 99
                kand.append(((-(a in top2) - (b in top2), pos(a) + pos(b), -(ve[a] + ve[b]), a, b), (a, b)))
    par = min(kand)[1]
    eu.append(list(par)); kap[par[0]] -= 1; kap[par[1]] -= 1; iskor_parovi.add(par)

SET = [(sorted(g), e) for g, e in zip(komb, eu)]

# ---------- referenca: najbolji raspored za 8 kombinacija ----------
G40 = [list(range(1 + 5 * i, 6 + 5 * i)) for i in range(8)]  # 40 različitih glavnih (raspored glavnih ne utječe)
C40 = V.glavni_cache(G40)
def oblik(unutarnji, ciklus=0):
    v = list(range(1, 13)); deg2 = v[:4]; listovi = v[4:]; parovi = []
    i2 = 0
    if ciklus:
        cyc = deg2[:ciklus]; i2 = ciklus
        parovi += [(cyc[j], cyc[(j + 1) % ciklus]) for j in range(ciklus)]
    li = 0
    for k in unutarnji:
        put = [listovi[li]] + deg2[i2:i2 + k] + [listovi[li + 1]]; li += 2; i2 += k
        parovi += list(zip(put, put[1:]))
    assert len(parovi) == 8
    return [sorted(p) for p in parovi]
OBLICI = {'4,0,0,0': oblik([4, 0, 0, 0]), '3,1,0,0': oblik([3, 1, 0, 0]), '2,2,0,0': oblik([2, 2, 0, 0]),
          '2,1,1,0': oblik([2, 1, 1, 0]), '1,1,1,1': oblik([1, 1, 1, 1]), 'C3+1,0,0,0': oblik([1, 0, 0, 0], 3),
          'C4+0,0,0,0': oblik([0, 0, 0, 0], 4)}
ref_oblici = {k: V.vjerojatnosti(list(zip(G40, e)), C40)['P_bilo_koji'] for k, e in OBLICI.items()}
best_oblik = max(ref_oblici, key=ref_oblici.get)
# neograničena lokalna pretraga
rng = np.random.default_rng(SEED + 20)
def P_eu(Eu):
    return V.vjerojatnosti(list(zip(G40, Eu)), C40)['P_bilo_koji']
best_free, best_free_eu = 0, None
for start in range(20 if FAZA == 'konacni' else 4):
    Eu = [sorted((rng.choice(12, 2, replace=False) + 1).tolist()) for _ in range(8)]
    p = P_eu(Eu); popr = True
    while popr:
        popr = False
        for c in range(8):
            for i in range(2):
                for x in range(1, 13):
                    if x in Eu[c]: continue
                    E2 = [list(e) for e in Eu]; E2[c][i] = x; E2[c] = sorted(E2[c])
                    p2 = P_eu(E2)
                    if p2 > p + 1e-15: Eu, p, popr = E2, p2, True
    if p > best_free: best_free, best_free_eu = p, Eu
P_REF = max(ref_oblici[best_oblik], best_free)

# ---------- provjera cilja 3 i zamjene ----------
def provjera3(S_):
    G = [set(g) for g, _ in S_ + ODIGRANO]
    return dict(susjedni_glavni=[c + 1 for c, (g, e) in enumerate(S_) if any(b - a == 1 for a, b in zip(sorted(g), sorted(g)[1:]))],
                max_zajednickih=max(len(a & b) for a, b in itertools.combinations(G, 2)),
                parova_s_1_zajednickim=sum(len(a & b) == 1 for a, b in itertools.combinations(G, 2)),
                susjedni_euro=[c + 1 for c, (g, e) in enumerate(S_) if abs(e[0] - e[1]) == 1],
                euro_parovi_razliciti=len(set(tuple(e) for _, e in S_ + ODIGRANO)) == len(S_ + ODIGRANO),
                euro_pojavljivanja={x: sum(x in e for _, e in S_) for x in range(1, 13)},
                razlicitih_glavnih=len(set().union(*[set(g) for g, _ in S_])))
Vs = V.vjerojatnosti(SET + ODIGRANO)
P_prije_zamjena = Vs['P_bilo_koji']; SET_prije = [(list(g), list(e)) for g, e in SET]; prov_prije = provjera3(SET)
zamjene = []; zamjenski = [[] for _ in range(N_KOMB)]
while Vs['P_bilo_koji'] < P_REF - 0.003:
    zaj = [(c, x) for c, (g, _) in enumerate(SET) for x in g if any(x in SET[d][0] for d in range(N_KOMB) if d != c)]
    if zaj:
        c, x = min(zaj, key=lambda t: (vg[t[1]], -t[0], -t[1]))
        ost = [y for y in SET[c][0] if y != x]; dr = [SET[d][0] for d in range(N_KOMB) if d != c] + ranije
        u_setu = set(y for g, _ in SET for y in g)
        kand = [y for y in range(1, 51) if y not in u_setu and dopusten(y, ost, dr)]
        y = min(kand, key=lambda y: (-vg[y], -min(abs(y - z) for z in ost), y))
        SET[c] = (sorted(ost + [y]), SET[c][1]); zamjene.append(f"K{c + 1}: glavni {x} → {y}")
        if x in punila_l[c]: punila_l[c].remove(x)
        if x in zamjenski[c]: zamjenski[c].remove(x)
        zamjenski[c].append(y)
    else:
        best = None
        for c, d in itertools.combinations(range(N_KOMB), 2):
            for i, j in itertools.product(range(2), range(2)):
                e1, e2 = list(SET[c][1]), list(SET[d][1]); e1[i], e2[j] = e2[j], e1[i]
                if e1[0] == e1[1] or e2[0] == e2[1] or abs(e1[0] - e1[1]) == 1 or abs(e2[0] - e2[1]) == 1: continue
                S2 = [s for s in SET]; S2[c] = (SET[c][0], sorted(e1)); S2[d] = (SET[d][0], sorted(e2))
                if len(set(tuple(e) for _, e in S2)) < N_KOMB: continue
                p = V.vjerojatnosti(S2)['P_bilo_koji']
                if best is None or p > best[0]: best = (p, S2, f"euro K{c + 1}↔K{d + 1}: {SET[c][1][i]}↔{SET[d][1][j]}")
        SET = best[1]; zamjene.append(best[2])
    Vs = V.vjerojatnosti(SET + ODIGRANO)
Vnovi = V.vjerojatnosti(SET)
prov = provjera3(SET)

# ---------- redoslijed igranja ----------
red, ost, koraci = [], list(range(N_KOMB)), []
while ost:
    best = max(ost, key=lambda c: (round(V.vjerojatnosti([SET[i] for i in red + [c]])['P_bilo_koji'], 13), -c))
    red.append(best); ost.remove(best)
    v = V.vjerojatnosti([SET[i] for i in red])
    koraci.append(dict(k=len(red), komb=best + 1, ulog=len(red) * CIJENA, P_bilo_koji=v['P_bilo_koji'], P_5p2=v['P_5p2'], P_top=v['P_top']))

# ---------- 1000 slučajnih setova ----------
rngs = np.random.default_rng(SEED + 12)
rand_sets = [[(sorted((rngs.choice(50, 5, replace=False) + 1).tolist()), sorted((rngs.choice(12, 2, replace=False) + 1).tolist()))
              for _ in range(N_KOMB)] for _ in range(1000)]
n_rand = 1000 if FAZA == 'konacni' else 200
rp = np.array([V.vjerojatnosti(s)['P_bilo_koji'] for s in rand_sets[:n_rand]])
pct = float((rp < Vs['P_bilo_koji']).mean() * 100 + (rp == Vs['P_bilo_koji']).mean() * 50)

glas_k = [sum(vg[x] for x in g) for g, _ in SET]
rez = dict(faza=FAZA, vrijeme=dt.datetime.now(ZoneInfo('Europe/Zagreb')).isoformat(timespec='seconds'),
           tezine={k: s['tezina'] for k, s in sustav.items()}, sustavi=sustav, promjene_u_prozoru=promjene,
           glasovi_g={x: vg[x] for x in rank_g if vg[x]}, glasovi_e={x: ve[x] for x in rank_e if ve[x]},
           signal=signal, sum=sum_g, rang_e=rank_e, red_sustava=red_sust,
           set=SET, izvori=izvori, punila=punila_l, zamjenski=zamjenski, set_prije_zamjena=SET_prije, P_prije_zamjena=P_prije_zamjena, provjera_prije=prov_prije, glasovi_po_komb=glas_k, zbroj_glasova=sum(glas_k), zamjene=zamjene,
           provjera=prov, vjerojatnosti_set=Vs, vjerojatnosti_nove=Vnovi,
           referenca=dict(oblici=ref_oblici, najbolji_oblik=best_oblik, neograniceno=best_free, neograniceno_euro=best_free_eu,
                          P_ref=P_REF, razlika_pp=(P_REF - Vs['P_bilo_koji']) * 100),
           redoslijed=koraci, slucajni=dict(n=n_rand, min=rp.min(), sred=rp.mean(), max=rp.max(), percentil=pct),
           trajanje_s=round(time.time() - T0, 1))
json.dump(rez, open(os.path.join(OUT, f'set_v2_{FAZA}.json'), 'w'), ensure_ascii=False, indent=1,
          default=lambda o: o.item() if hasattr(o, 'item') else str(o))
print('promjene', json.dumps(promjene, ensure_ascii=False))
print('signal', [(x, vg[x]) for x in signal]); print('šum', [(x, vg[x]) for x in sum_g])
print('euro', [(x, ve[x]) for x in rank_e]); print('red sustava', red_sust)
for c, (g, e) in enumerate(SET):
    print(f"K{c + 1} {izvori[c]:15s} {g} {e} punila={punila_l[c]} zamjenski={zamjenski[c]} glasovi={[vg[x] for x in g]} E={[ve[x] for x in e]}")
print('P prije zamjena', P_prije_zamjena, prov_prije, SET_prije); print('zamjene', zamjene); print('provjera', prov)
print('V set', Vs); print('ref', rez['referenca']); print('slučajni', rez['slucajni'])
for k in koraci: print(k)
print('trajanje', rez['trajanje_s'])
