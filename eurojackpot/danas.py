"""§4 – ezoterija za 27.11.2026. + slučajni elementi (secrets) spremljeni u JSON prije slaganja."""
import json, secrets, os, datetime as dt, math
import ephem, swisseph as swe
import sustavi as S

D = dt.date(2026, 11, 27)
OUT = os.path.dirname(os.path.abspath(__file__))
hist = [l.strip().split(';') for l in open(os.path.join(OUT, 'povijest_eurojackpot_2022-03-25_2026-11-24.csv')).readlines()[1:]]
hist = [(dt.date.fromisoformat(r[0]), list(map(int, r[1:6])), list(map(int, r[6:8]))) for r in hist]
prev5 = hist[-5:][::-1]
hl_kolo = sum(1 for d, _, _ in hist if d.year == 2026) + 1

# --- slučajni elementi: jednom, prije slaganja ---
rpath = os.path.join(OUT, 'slucajni_elementi.json')
if not os.path.exists(rpath):
    R = dict(izvor='Python secrets (os.urandom)', vrijeme=dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds'),
             iching=[[secrets.choice([2, 3]) for _ in range(3)] for _ in range(6)],
             tarot=[], lenormand=[], numpy_seed=secrets.randbits(63))
    pool = list(range(1, 79))
    for _ in range(10):
        R['tarot'].append(pool.pop(secrets.randbelow(len(pool))))
    pool = list(range(1, 37))
    for _ in range(9):
        R['lenormand'].append(pool.pop(secrets.randbelow(len(pool))))
    json.dump(R, open(rpath, 'w'), ensure_ascii=False, indent=1)
R = json.load(open(rpath))

res = {}
raw, rep_e, info = S.sinkronicitet(D, prev5, hl_kolo)
res['Sinkronicitet'] = (raw, rep_e, info)
res['Arhetipovi'] = (S.ARHETIPOVI, [], {})
raw, info = S.meihua(D); res['Meihua'] = (raw, [], info)
raw, info = S.iching_bacanja(R['iching']); res['I Ching'] = (raw, [], info)
raw, info = S.jyotish(D); res['Jyotiš'] = (raw, [], info)
raw, info = S.zapadna(D); res['Zapadna'] = (raw, [], info)
raw, info = S.kamea(D); res['Kamea (Venera)'] = (raw, [], info)
raw, info = S.kabala(D); res['Kabala'] = (raw, [], info)
dek, dek_info = S.tarot_dekan(D)
res['Tarot'] = ([dek] + R['tarot'], [], dict(dekan=dek_info, karta_dekana=dek, karte=R['tarot']))
L = R['lenormand']
res['Lenormand'] = (L + [L[i] + L[i + 1] for i in range(8)], [], dict(karte=L, zbrojevi=[L[i] + L[i + 1] for i in range(8)]))
raw, info = S.numerologija(D); res['Numerologija'] = (raw, [], info)
raw, info = S.planetarni_sat(D); res['Planetarni sat'] = (raw, [], info)

out = {}
for k, (raw, er, info) in res.items():
    g, e = S.popis(raw, er)
    out[k] = dict(sirovo=raw, euro_izricito=er, glavni=g, euro=e, info=info)
    print(f"{k:16s} G10={g[:10]}  E={e[:6]}")
    print('   ', info)

# --- pravilnosti zadnjih 5 kola ---
from collections import Counter
pr = []
for d, g, e in prev5:
    g = sorted(g)
    parovi = [(a, b) for a, b in zip(g, g[1:]) if b - a == 1]
    pr.append(f"{d}: zbroj {sum(g)}, neparnih {sum(x % 2 for x in g)}, susjedni parovi {parovi}, raspon {g[-1]-g[0]}")
ponav = [(prev5[i][0], prev5[i + 1][0], sorted(set(prev5[i][1]) & set(prev5[i + 1][1])), sorted(set(prev5[i][2]) & set(prev5[i + 1][2]))) for i in range(4)]
out['_pravilnosti'] = dict(po_kolu=pr, ponavljanja_uzastopnih=[f"{a}↔{b}: glavni {x}, euro {y}" for a, b, x, y in ponav],
                           hl_kolo=hl_kolo)
print('\n'.join(pr)); print(out['_pravilnosti']['ponavljanja_uzastopnih'], 'HL kolo', hl_kolo)

# --- kontrola drugom bibliotekom (ephem) ---
o = ephem.Observer(); o.lat, o.lon, o.elevation = '60.17', '24.94', 0
o.date = '2026/11/27 19:00:00'
kon = {}
for ime, b, sb in [('Sunce', ephem.Sun(), swe.SUN), ('Mjesec', ephem.Moon(), swe.MOON), ('Venera', ephem.Venus(), swe.VENUS),
                   ('Jupiter', ephem.Jupiter(), swe.JUPITER), ('Merkur', ephem.Mercury(), swe.MERCURY), ('Mars', ephem.Mars(), swe.MARS)]:
    b.compute(o.date, epoch=o.date)  # geocentrično, ekvinocij datuma
    le = math.degrees(ephem.Ecliptic(b, epoch=o.date).lon)
    ls = S.lon_trop(S.jd_ut(D), sb)
    kon[ime] = dict(swe=round(ls, 4), ephem=round(le, 4), razlika_lucne_min=round(((ls - le + 180) % 360 - 180) * 60, 2))
o.date = '2026/11/27 00:00:00'; o.pressure = 0; o.horizon = '-0:34'
s = ephem.Sun()
kon['izlazak_UT_ephem'] = str(o.next_rising(s)); kon['zalazak_UT_ephem'] = str(o.next_setting(s))
o.date = kon['zalazak_UT_ephem']; kon['izlazak_sutra_UT_ephem'] = str(o.next_rising(s))
out['_kontrola_ephem'] = kon
print(json.dumps(kon, ensure_ascii=False, indent=1))
json.dump(out, open(os.path.join(OUT, 'ezoterija_danas.json'), 'w'), ensure_ascii=False, indent=1, default=str)
