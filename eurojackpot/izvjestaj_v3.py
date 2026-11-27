"""v3 – konačne tablice, redoslijed igranja, slučajni setovi, backtest (stvarne isplate), očekivani povrat, CSV."""
import json, os, time, datetime as dt
from zoneinfo import ZoneInfo
from math import comb
import numpy as np
import v3_jezgra as J

T_START = time.time()
OUT = J.OUT
P = json.load(open(os.path.join(OUT, 'v3_privremeni.json')))
OA = json.load(open(os.path.join(OUT, 'v3_optimizacija_A.json'))); OB = json.load(open(os.path.join(OUT, 'v3_optimizacija_B.json')))
v2 = json.load(open(os.path.join(OUT, 'set_v2_konacni.json')))
R = json.load(open(os.path.join(OUT, 'slucajni_elementi.json'))); SEED = R['numpy_seed']
SUST = v2['sustavi']; signal = v2['signal']
vg = {int(k): v for k, v in v2['glasovi_g'].items()}; ve = {int(k): v for k, v in v2['glasovi_e'].items()}
tab = {tuple(map(int, k.split('+'))): v for k, v in P['isplate'].items()}
LUT = J.lut(tab, 'med')
UPL = [(list(g), list(e)) for g, e in P['uplaceni']]
A = [(list(g), list(e)) for g, e in OA['konacni']]; B = [(list(g), list(e)) for g, e in OB['konacni']]
IZV_A = [o['sustav'] for o in P['oznake_A']]; IZV_B = P['izvori_B']

def oznake(g, e, src):
    own = signal if src == 'Plečko' else SUST[src]['popis_g']
    own_e = v2['rang_e'] if src == 'Plečko' else SUST[src]['popis_e']
    return [x for x in g if x not in own], [x for x in e if x not in own_e]

# ---------- očekivani povrat ----------
pm = {k: comb(5, k) * comb(45, 5 - k) / comb(50, 5) for k in range(6)}
pe = {k: comb(2, k) * comb(10, 2 - k) / comb(12, 2) for k in range(3)}
E1 = {s: sum(pm[m] * pe[e] * v[s] for (m, e), v in tab.items()) for s in ('p25', 'med', 'p75')}
E1_bez_jp = {s: sum(pm[m] * pe[e] * v[s] for (m, e), v in tab.items() if (m, e) != (5, 2)) for s in ('p25', 'med', 'p75')}

# ---------- redoslijed igranja A (uzorak) + točno ----------
U = J.Uzorak(SEED + 30, 300_000)
payA = [U.isplata(g, e, LUT) for g, e in A]
red, ost, tot = [], list(range(len(A))), np.zeros(U.n, np.int32)
while ost:
    c = max(ost, key=lambda c: (np.count_nonzero(tot + payA[c]), -c))
    red.append(c); ost.remove(c); tot = tot + payA[c]
koraci = []
for k in range(1, len(A) + 1):
    S = [A[i] for i in red[:k]]
    t = J.tocno(S, {'med': LUT}, pragovi=(200 * k,))
    koraci.append(dict(k=k, komb=red[k - 1] + 1, ulog=2 * k, P_bilo_koji=t['P_bilo_koji'], P_isplata_ge_ulog=t[f'P_ge_{2 * k}_med']))
    print(koraci[-1])

# ---------- slučajni setovi (uzorak) ----------
def uz(S):
    t = sum(U.isplata(g, e, LUT) for g, e in S)
    return np.count_nonzero(t) / U.n, np.count_nonzero(t >= 2600) / U.n
rngA = np.random.default_rng(SEED + 40); rngB = np.random.default_rng(SEED + 41)
def rnd(rng, n):
    return [(sorted((rng.choice(50, 5, replace=False) + 1).tolist()), sorted((rng.choice(12, 2, replace=False) + 1).tolist())) for _ in range(n)]
RA = [rnd(rngA, 13) for _ in range(1000)]; RB = [UPL + rnd(rngB, 2) for _ in range(1000)]
ra = np.array([uz(s) for s in RA]); rb = np.array([uz(s) for s in RB])
uA, uB, u8 = uz(A), uz(UPL + B), uz(UPL)
pct = lambda a, x: float((a < x).mean() * 100 + (a == x).mean() * 50)
slucajni = dict(A=dict(P_bilo_koji=[ra[:, 0].min(), ra[:, 0].mean(), ra[:, 0].max(), pct(ra[:, 0], uA[0])],
                       P_ge26=[ra[:, 1].min(), ra[:, 1].mean(), ra[:, 1].max(), pct(ra[:, 1], uA[1])], uzorak_set=uA),
                B=dict(P_bilo_koji=[rb[:, 0].min(), rb[:, 0].mean(), rb[:, 0].max(), pct(rb[:, 0], uB[0])],
                       P_ge26=[rb[:, 1].min(), rb[:, 1].mean(), rb[:, 1].max(), pct(rb[:, 1], uB[1])], uzorak_set=uB), v2_8_uzorak=u8)
print('slučajni', slucajni)

# ---------- backtest (stvarne isplate euro-jackpot.net) ----------
H = [x for x in J.povijest() if x[3] == '2/12']
ISP = {d: J.isplate_kola(d) for d, _, _, _ in H}
def bt(S, ulog):
    raz = {}; ukupno = 0.0; kola_ge = 0; kola_dob = 0
    for d, G, E, _ in H:
        s = 0.0
        for g, e in S:
            k = (len(G & set(g)), len(E & set(e)))
            if k in ISP[d]:
                raz[k] = raz.get(k, 0) + 1; s += ISP[d][k][0]
        ukupno += s; kola_ge += s >= ulog; kola_dob += s > 0
    return dict(razredi={f'{a}+{b}': v for (a, b), v in sorted(raz.items(), reverse=True)}, isplate=round(ukupno, 2),
                ulog=ulog * len(H), povrat_po_euru=round(ukupno / (ulog * len(H)), 4), kola_s_isplatom_ge_ulog=kola_ge, kola_s_dobitkom=kola_dob)
btA = bt(A, 26); btB = bt(UPL + B, 20); bt8 = bt(UPL, 16)
bra = np.array([bt(s, 26)['isplate'] for s in RA]); brb = np.array([bt(s, 20)['isplate'] for s in RB])
btA['percentil_isplata'] = pct(bra, btA['isplate']); btB['percentil_isplata'] = pct(brb, btB['isplate'])
btA['slucajni_isplate'] = [float(bra.min()), float(np.median(bra)), float(bra.max())]; btB['slucajni_isplate'] = [float(brb.min()), float(np.median(brb)), float(brb.max())]
print('bt A', btA); print('bt B', btB)

# ---------- CSV ----------
sada = dt.datetime.now(ZoneInfo('Europe/Zagreb')).strftime('%Y-%m-%d %H:%M')
hdr = 'ID;izvor;datum_i_sat_izrade;kolo;tablica;status;sustav;G1;G2;G3;G4;G5;E1;E2;brojevi_iz_drugog_sustava;stabilnost;glasovi;seed;pogoci_glavni;pogoci_euro;razred;isplata'
def stab(x, src):
    if src == 'Plečko': return '1'
    s = SUST[src]; v = s['stab_g'].get(str(x))
    if v is None:
        pr = v2['promjene_u_prozoru'].get(src); v = 1.0 if pr is None or all(x in p[0] for p in pr.values()) else 0.5
    return str(v).replace('.', ',')
def red_csv(ID, tabl, status, src, g, e):
    tg, te = oznake(g, e, src) if src in SUST or src == 'Plečko' else ([], [])
    st = '/'.join('†' if x in tg else stab(x, src) for x in g) if (src in SUST or src == 'Plečko') else ''
    gl = 'G:' + '/'.join(str(vg.get(x, 0)).replace('.', ',') for x in g) + ' E:' + '/'.join(str(ve.get(x, 0)).replace('.', ',') for x in e)
    return ';'.join([ID, 'Claude Code', sada, '95', tabl, status, src] + list(map(str, g)) + list(map(str, e)) +
                    [','.join([f'{x}†' for x in tg] + [f'E{x}†' for x in te]), st, gl, str(SEED), '', '', '', ''])
v2ids = [l.split(';')[0] for l in open(os.path.join(OUT, 'eurojackpot_2026-11-27_kolo95_v2.csv')).read().splitlines()[1:]]
v2src = [l.split(';')[4] for l in open(os.path.join(OUT, 'eurojackpot_2026-11-27_kolo95_v2.csv')).read().splitlines()[1:]]
rows = [red_csv(f'EJ-2026-11-27-K95-v3-A{c + 1:02d}', 'A', 'NOVO', IZV_A[c], g, e) for c, (g, e) in enumerate(A)]
rowsB_upl = [';'.join([v2ids[c], 'Claude Code', sada, '95', 'B', 'uplaćeno', v2src[c]] + list(map(str, g)) + list(map(str, e)) + ['', '', '', str(SEED), '', '', '', ''])
             for c, (g, e) in enumerate(UPL)]
rowsB_new = [red_csv(f'EJ-2026-11-27-K95-v3-B{9 + c:02d}', 'B', 'NOVO', IZV_B[c], g, e) for c, (g, e) in enumerate(B)]
open(os.path.join(OUT, 'eurojackpot_2026-11-27_kolo95_v3.csv'), 'w').write(hdr + '\n' + '\n'.join(rows + rowsB_upl + rowsB_new) + '\n')
ev = os.path.join(OUT, 'evidencija_kombinacija.csv')
lines = open(ev).read().splitlines()
HDR_EV = 'ID;izvor;datum_i_sat_izrade;kolo;tablica;status;sustav;G1;G2;G3;G4;G5;E1;E2;punila;brojevi_iz_drugog_sustava;stabilnost;glasovi;seed;pogoci_glavni;pogoci_euro;razred;isplata'
if lines[0] != HDR_EV:  # stari format (v2): umetni tablica;status iza kola i brojevi_iz_drugog_sustava iza punila
    nove = [HDR_EV]
    for l in lines[1:]:
        p = l.split(';'); nove.append(';'.join(p[:4] + ['', ''] + p[4:13] + [''] + p[13:]))
    lines = nove
ids = {l.split(';')[0] for l in lines}
for r_ in rows + rowsB_new:  # uplaćeni iz B već postoje pod v2 ID-jevima
    p = r_.split(';')
    if p[0] not in ids: lines.append(';'.join(p[:14] + [''] + p[14:]))
open(ev, 'w').write('\n'.join(lines) + '\n')

res = dict(vrijeme=dt.datetime.now(ZoneInfo('Europe/Zagreb')).isoformat(timespec='seconds'), redoslijed_A=koraci, slucajni=slucajni,
           backtest=dict(A=btA, B=btB, v2_8=bt8), ocekivana_isplata_po_kombinaciji=E1, bez_jackpota=E1_bez_jp,
           oznake_A=[oznake(g, e, s) for (g, e), s in zip(A, IZV_A)], oznake_B=[oznake(g, e, s) for (g, e), s in zip(B, IZV_B)],
           trajanje_s=round(time.time() - T_START, 1))
json.dump(res, open(os.path.join(OUT, 'v3_izvjestaj.json'), 'w'), ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
print('E1', E1, 'bez jp', E1_bez_jp); print('\n'.join(rows + rowsB_upl + rowsB_new)); print('trajanje', res['trajanje_s'])
