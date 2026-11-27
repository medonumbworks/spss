"""v3 – ezoterijski rasporedi E0 za tablice A i B (protocol_v3.md §1–§2) + točne vjerojatnosti (privremene tablice)."""
import json, os, sys, time, itertools, datetime as dt
from zoneinfo import ZoneInfo
import numpy as np
import v3_jezgra as J

T0 = time.time()
OUT = J.OUT
v2 = json.load(open(os.path.join(OUT, 'set_v2_konacni.json')))
SUST = v2['sustavi']
vg = {int(k): v for k, v in v2['glasovi_g'].items()}; ve = {int(k): v for k, v in v2['glasovi_e'].items()}
pos_g = {}
for k, s in SUST.items():
    for x in s['glasa_g']:
        pos_g[x] = pos_g.get(x, 0) + s['popis_g'].index(x) + 1
rank_g = sorted(range(1, 51), key=lambda x: (-vg.get(x, 0), pos_g.get(x, 999) if vg.get(x) else 999, x))
rank_e = v2['rang_e']
signal = v2['signal']
BAZEN = sorted({x for s in SUST.values() for x in s['popis_g']})
BAZEN_E = sorted({x for s in SUST.values() for x in s['popis_e']})
UPLACENI = [(g, e) for g, e in v2['set']]  # v2-01 … v2-08; 3 dodatne nisu poznate

def slozi(izvori, uplaceni):
    S, oznake = [], []
    for src in izvori:
        drugi = list(uplaceni) + S
        own = signal if src == 'Plečko' else SUST[src]['popis_g']
        kand = [(x, False) for x in own] + [(x, True) for x in signal] + [(x, True) for x in rank_g if x in BAZEN]
        kk, tuđi = [], []
        for x, t in kand:
            if len(kk) == 5: break
            if x in kk or x not in BAZEN or any(abs(x - y) <= 1 for y in kk): continue
            if any(len((set(kk) | {x}) & set(h)) > 2 for h, _ in drugi): continue
            kk.append(x)
            if t and x not in own: tuđi.append(x)
        own_e = rank_e if src == 'Plečko' else SUST[src]['popis_e']
        ke = []
        for x in list(own_e) + list(rank_e):
            if x not in ke and x in BAZEN_E: ke.append(x)
        iskor = {tuple(sorted(e)) for _, e in drugi}
        par = next((a, b) for i, a in enumerate(ke) for b in ke[i + 1:] if abs(a - b) != 1 and tuple(sorted((a, b))) not in iskor)
        e_tuđi = [x for x in par if x not in own_e]
        S.append((sorted(kk), sorted(par))); oznake.append(dict(sustav=src, tudji=tuđi, tudji_e=e_tuđi))
    return S, oznake

# A: K1 Plečko + 12 sustava redom v2
IZV_A = ['Plečko'] + v2['red_sustava']
A0, OZN_A = slozi(IZV_A, [])
# B: 2 nova od sustava bez kombinacije u v2
KAND_B = ['Sinkronicitet', 'Lenormand', 'Tarot', 'Kamea (Venera)', 'Numerologija']
def kljuc(k):
    u = [x for x in SUST[k]['glasa_g'] if x in signal]
    return (-len(u), -sum(vg[x] for x in u), k.casefold())
IZV_B = sorted(KAND_B, key=kljuc)[:2]
B0, OZN_B = slozi(IZV_B, UPLACENI)

if __name__ == '__main__':
    tab, ned, raz = J.tablice_isplata()
    luts = {s: J.lut(tab, s) for s in ('p25', 'med', 'p75')}
    res = {}
    for ime, S, upl in [('A', A0, []), ('B', B0, UPLACENI), ('v2_8', UPLACENI, [])]:
        cijeli = list(upl) + list(S)
        res[ime] = dict(tocno=J.tocno(cijeli, luts), provjera=J.provjera(S, upl))
        print(ime, res[ime])
    out = dict(vrijeme=dt.datetime.now(ZoneInfo('Europe/Zagreb')).isoformat(timespec='seconds'), bazen=BAZEN, bazen_e=BAZEN_E,
               rang_g=rank_g, signal=signal, A0=A0, oznake_A=OZN_A, izvori_B=IZV_B, B0=B0, oznake_B=OZN_B,
               uplaceni=UPLACENI, isplate={f'{a}+{b}': v for (a, b), v in tab.items()}, isplate_razdoblje=raz, rezultati=res, trajanje_s=round(time.time() - T0, 1))
    json.dump(out, open(os.path.join(OUT, 'v3_privremeni.json'), 'w'), ensure_ascii=False, indent=1,
              default=lambda o: o.item() if hasattr(o, 'item') else str(o))
    print('bazen', len(BAZEN), BAZEN, 'izvan', sorted(set(range(1, 51)) - set(BAZEN)), 'euro bazen', BAZEN_E)
    for c, ((g, e), o) in enumerate(zip(A0, OZN_A), 1): print('A', c, o['sustav'], g, e, o)
    for c, ((g, e), o) in enumerate(zip(B0, OZN_B), 1): print('B', c, o['sustav'], g, e, o)
    print('trajanje', out['trajanje_s'])
