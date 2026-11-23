# -*- coding: utf-8 -*-
"""Sve ili ništa (11 od 22) – 9 kombinacija za izvlačenje uto 24.11.2026. (kolo 328, 19:00). Prilagođeno iz sin_2026_11_23.py.
Pokretanje: PYTHONPATH=<pyswisseph build> python3 -I sin_2026_11_23.py <csv> <xlsx> <ephe_dir> <out_dir>
"""
import sys, json, time, secrets, itertools, math, importlib.metadata as md
from datetime import datetime, date
from math import comb
import numpy as np, pandas as pd
from scipy import stats

T0 = time.time()
CSV, XLSX, EPHE, OUT = sys.argv[1:5]
RNG = secrets.SystemRandom()
import os
from datetime import timedelta
CACHE = f'{OUT}/slucajno_2026-11-24.json'   # secrets-izvlačenja se spremaju: ponovno pokretanje ih NE izvlači ponovno
_cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
def cached(key, fn):
    if key not in _cache:
        _cache[key] = fn(); json.dump(_cache, open(CACHE, 'w'))
    return _cache[key]
R = {}  # rezultati za izvještaj

def red(x):
    x = int(x)
    while x > 22: x -= 22
    if x <= 0: x += 22          # 0 -> 22 (pravilo za 0 nije zadano; navedeno u izvještaju)
    return x

def uniq_red(seq):
    out = []
    for v in seq:
        r = red(v)
        if r not in out: out.append(r)
    return out

# ---------------------------------------------------------------- §1 podaci
df = pd.read_csv(CSV, sep=';', encoding='utf-8-sig')
xl = pd.read_excel(XLSX, sheet_name=None)
B = [f'B{i}' for i in range(1, 12)]
chk = {}
chk['csv_redova'] = len(df)
miss = df[df[B].isna().any(axis=1)]
chk['nedostaje'] = miss[['Godina', 'Kolo', 'Datum']].astype(str).values.tolist()
d = df.dropna(subset=B).copy()
d[B] = d[B].astype(int)
d['dt'] = pd.to_datetime(d['Datum'], format='%d.%m.%Y')
nums = d[B].values
chk['svi_1_22'] = bool(((nums >= 1) & (nums <= 22)).all())
chk['11_razlicitih'] = bool(all(len(set(r)) == 11 for r in nums))
chk['uzlazno'] = bool(all(list(r) == sorted(r) for r in nums))
chk['zbroj_ok'] = bool((nums.sum(1) == d['Zbroj'].astype(int).values).all())
chk['parni_ok'] = bool(((nums % 2 == 0).sum(1) == d['Parni'].astype(int).values).all())
chk['dupli_datumi'] = int(df['Datum'].duplicated().sum())
chk['kolo_eq_dan'] = bool((d['dt'].dt.dayofyear == d['Kolo']).all())
xd = xl['Izvlačenja'].dropna(subset=B)
chk['xlsx_csv_isti'] = bool((xd[B].astype(int).values == nums[:len(xd)]).all())  # prvih 635 kola = XLSX
chk['nova_kola_307_327'] = int(((d['Godina'] == 2026) & (d['Kolo'] >= 307)).sum())
chk['praznine'] = [(a.strftime('%d.%m.%Y'), b.strftime('%d.%m.%Y'), int((b - a).days - 1)) for a, b in zip(d['dt'].iloc[:-1], d['dt'].iloc[1:]) if (b - a).days > 1]
chk['prvo'] = d['Datum'].iloc[0]; chk['zadnje'] = d['Datum'].iloc[-1]
chk['kola_s_brojevima'] = len(d)
chk['napomene_xlsx'] = [str(c) for c in xl['Napomene'].columns] + [str(v) for v in xl['Napomene'].iloc[:, 0].tolist()]
R['podaci'] = chk
D = [set(map(int, r)) for r in nums]
N = len(D)
dates = d['Datum'].tolist(); kola = list(zip(d['Godina'], d['Kolo']))

# razredi (pretpostavka, nije potvrđeno): 11/0, 10/1, 9/2, 8/3
WIN_K = {11: '11/0', 0: '11/0', 10: '10/1', 1: '10/1', 9: '9/2', 2: '9/2', 8: '8/3', 3: '8/3'}
TOTAL = comb(22, 11)
PK = {k: comb(11, k) * comb(11, 11 - k) / TOTAL for k in range(12)}
P_WIN = sum(PK[k] for k in WIN_K)
R['P_win_po_kombinaciji'] = P_WIN

# ---------------------------------------------------------------- §3 ezoterija
import swisseph as swe
from lunar_python import Solar
import sxtwl
from zoneinfo import ZoneInfo
swe.set_ephe_path(EPHE)
LAT, LON = 45.815, 15.982
loc = datetime(2026, 11, 24, 19, 0, tzinfo=ZoneInfo('Europe/Zagreb'))
utc = loc.astimezone(ZoneInfo('UTC'))
jd = swe.julday(utc.year, utc.month, utc.day, utc.hour + utc.minute / 60)
E = {'trenutak_lokalno': loc.isoformat(), 'trenutak_UTC': utc.isoformat(), 'JD_UT': jd}
FL = swe.FLG_SWIEPH | swe.FLG_SPEED
def lon_trop(p):
    x, f = swe.calc_ut(jd, p, FL)
    assert f & swe.FLG_SWIEPH, 'nije korištena SWIEPH datoteka'
    return x[0]
SYS = {}      # sustav -> poredan popis 1..22
KEY = {}      # ključni podaci po sustavu

# Sinkronicitet
last5 = list(range(N - 5, N))
cnt = {}
for i in last5:
    for v in D[i]: cnt[v] = cnt.get(v, 0) + 1
lastseen = {v: max(i for i in last5 if v in D[i]) for v in cnt}
rep = sorted([v for v in cnt if cnt[v] >= 4], key=lambda v: (-cnt[v], -lastseen[v], v))
pravilnosti = {
    'kola': [(dates[i], sorted(D[i])) for i in last5],
    'u_svih_5': sorted(v for v in cnt if cnt[v] == 5),
    'u_4': sorted(v for v in cnt if cnt[v] == 4),
    'nijednom_u_5': sorted(set(range(1, 23)) - set(cnt)),
    'ponovljeno_iz_prethodnog': [len(D[i] & D[i - 1]) for i in last5],
    'parni': [sum(1 for v in D[i] if v % 2 == 0) for i in last5],
    'zbroj': [sum(D[i]) for i in last5],
    'najdulji_niz': [max(len(list(g)) for _, g in itertools.groupby(enumerate(sorted(D[i])), lambda t: t[1] - t[0])) for i in last5],
    'broj_1_u_svakom': all(1 in D[i] for i in last5),
    'pojavljivanja': {v: cnt.get(v, 0) for v in range(1, 23)},
    'kola_br': [kola[i][1] for i in last5],
}
SYS['Sinkronicitet'] = uniq_red(rep + [24, 11, 1, 328, 328, 35] + [31, 5, 18, 20] + sorted(D[N - 1]))
KEY['Sinkronicitet'] = f"≥4× u zadnjih 5 (kola {kola[N-5][1]}–{kola[N-1][1]}): {rep}; 24, 11, 2026→1, kolo 328→20, redni dan 328→20, starost 35→13, 31→9, 5, 18, 20, kolo {kola[N-1][1]}"
E['sinkronicitet_pravilnosti'] = pravilnosti

# Arhetipovi
SYS['Arhetipovi'] = uniq_red([4, 12, 1, 16, 7, 10, 28, 22, 40, 36])
KEY['Arhetipovi'] = 'zadani popis'

# I Ching (King Wen tablica: [gornji][donji], red trigrama Qian,Zhen,Kan,Gen,Kun,Xun,Li,Dui)
TRI = {'Qian': (1, 1, 1), 'Dui': (1, 1, 0), 'Li': (1, 0, 1), 'Zhen': (1, 0, 0),
       'Xun': (0, 1, 1), 'Kan': (0, 1, 0), 'Gen': (0, 0, 1), 'Kun': (0, 0, 0)}  # crte odozdo
XT = {'Qian': 1, 'Dui': 2, 'Li': 3, 'Zhen': 4, 'Xun': 5, 'Kan': 6, 'Gen': 7, 'Kun': 8}  # Xiantian broj
XT_INV = {v: k for k, v in XT.items()}
ORD = ['Qian', 'Zhen', 'Kan', 'Gen', 'Kun', 'Xun', 'Li', 'Dui']
KW = [[1, 25, 6, 33, 12, 44, 13, 10], [34, 51, 40, 62, 16, 32, 55, 54], [5, 3, 29, 39, 8, 48, 63, 60],
      [26, 27, 4, 52, 23, 18, 22, 41], [11, 24, 7, 15, 2, 46, 36, 19], [9, 42, 59, 53, 20, 57, 37, 61],
      [14, 21, 64, 56, 35, 50, 30, 38], [43, 17, 47, 31, 45, 28, 49, 58]]
assert sorted(sum(KW, [])) == list(range(1, 65))
LINES2TRI = {v: k for k, v in TRI.items()}
def hexnum(lines):  # lines: 6 crta odozdo (1 yang, 0 yin)
    lo, up = LINES2TRI[tuple(lines[:3])], LINES2TRI[tuple(lines[3:])]
    return KW[ORD.index(up)][ORD.index(lo)], up, lo
throws = cached('iching', lambda: [[RNG.choice([3, 2]) for _ in range(3)] for _ in range(6)])
vals = [sum(t) for t in throws]                      # 6,7,8,9
prim = [1 if v in (7, 9) else 0 for v in vals]
chg_lines = [i + 1 for i, v in enumerate(vals) if v in (6, 9)]
changed = [(1 - l) if (i + 1) in chg_lines else l for i, l in enumerate(prim)]
h1, up1, lo1 = hexnum(prim); h2, _, _ = hexnum(changed)
hn, _, _ = hexnum(prim[1:4] + prim[2:5])
SYS['I Ching'] = uniq_red([h1, h2, hn] + chg_lines + [XT[up1], XT[lo1]])
KEY['I Ching'] = f"bacanja {vals} (odozdo); heksagram {h1} ({up1}/{lo1}) → {h2}; nuklearni {hn}; promjenjive crte {chg_lines or 'nema'}"

# Lunarni datum (lunar_python, provjera sxtwl)
lun = Solar.fromYmdHms(2026, 11, 24, 19, 0, 0).getLunar()
bj = loc.astimezone(ZoneInfo('Asia/Shanghai')); lunBJ = Solar.fromYmdHms(bj.year, bj.month, bj.day, bj.hour, 0, 0).getLunar()
LM, LD = abs(lun.getMonth()), lun.getDay()
sx = sxtwl.fromSolar(2026, 11, 24)
assert (sx.getLunarMonth(), sx.getLunarDay()) == (LM, LD), 'lunar_python i sxtwl se ne slažu'
E['lunarni'] = {'lunar_python': f"{lun.getYearInGanZhi()} god., mj. {lun.getMonth()} ({lun.getMonthInChinese()}), dan {LD} ({lun.getDayInChinese()}), sat {lun.getTimeZhi()}",
                'sxtwl': f"mj. {sx.getLunarMonth()} (prijestupni={sx.isLunarLeap()}), dan {sx.getLunarDay()}", 'slazu_se': True, 'peking': f"{bj:%d.%m.%Y %H:%M} (Peking) = lunarni mj. {lunBJ.getMonth()}, dan {lunBJ.getDay()}, sat {lunBJ.getTimeZhi()}"}
# Meihua
Y, H = 7, ['子', '丑', '寅', '卯', '辰', '巳', '午', '未', '申', '酉', '戌', '亥'].index(lun.getTimeZhi()) + 1
assert H == 11, H
up = (Y + LM + LD) % 8 or 8; lo = (Y + LM + LD + H) % 8 or 8; mv = (Y + LM + LD + H) % 6 or 6
ml = list(TRI[XT_INV[lo]]) + list(TRI[XT_INV[up]])
mh1 = KW[ORD.index(XT_INV[up])][ORD.index(XT_INV[lo])]
ml2 = ml.copy(); ml2[mv - 1] = 1 - ml2[mv - 1]
mh2, _, _ = hexnum(ml2); mhn, _, _ = hexnum(ml[1:4] + ml[2:5])
SYS['Meihua'] = uniq_red([mh1, mh2, mhn, up, lo, mv, Y, LM, LD, H])
KEY['Meihua'] = f"Y=7 (午), M={LM}, D={LD}, H={H} ({lun.getTimeZhi()}); gornji {up} {XT_INV[up]}, donji {lo} {XT_INV[lo]}, crta {mv}; heksagram {mh1} → {mh2}, nuklearni {mhn}"

# Jyotiš (Lahiri)
swe.set_sid_mode(swe.SIDM_LAHIRI)
FS = swe.FLG_SWIEPH | swe.FLG_SIDEREAL
def lon_sid(p):
    x, f = swe.calc_ut(jd, p, FS); assert f & swe.FLG_SWIEPH; return x[0]
moon_s, sun_s, rahu_s = lon_sid(swe.MOON), lon_sid(swe.SUN), lon_sid(swe.MEAN_NODE)
NAK = 360 / 27
nak_m = int(moon_s // NAK) + 1
tithi = int(((moon_s - sun_s) % 360) // 12) + 1
yoga = int(((moon_s + sun_s) % 360) // NAK) + 1
nak_r = int(rahu_s // NAK) + 1
NAKN = ['Ashvini', 'Bharani', 'Krittika', 'Rohini', 'Mrigashira', 'Ardra', 'Punarvasu', 'Pushya', 'Ashlesha', 'Magha', 'P.Phalguni', 'U.Phalguni', 'Hasta', 'Chitra', 'Swati', 'Vishakha', 'Anuradha', 'Jyeshtha', 'Mula', 'P.Ashadha', 'U.Ashadha', 'Shravana', 'Dhanishta', 'Shatabhisha', 'P.Bhadrapada', 'U.Bhadrapada', 'Revati']
SYS['Jyotiš'] = uniq_red([20, nak_m, tithi, 9, yoga, nak_r])
KEY['Jyotiš'] = f"Mjesec {moon_s:.2f}° sid., nakšatra {nak_m} {NAKN[nak_m-1]}; tithi {tithi}; vladar dana Mars=9; yoga {yoga}; srednji Rahu {rahu_s:.2f}° → nakšatra {nak_r} {NAKN[nak_r-1]}; ayanamsa {swe.get_ayanamsa_ut(jd):.4f}°"

# Zapadna astrologija (tropski)
SIGNS = ['Ovan', 'Bik', 'Blizanci', 'Rak', 'Lav', 'Djevica', 'Vaga', 'Škorpion', 'Strijelac', 'Jarac', 'Vodenjak', 'Ribe']
wp = {}
for nm, p in [('Mars', swe.MARS), ('Jupiter', swe.JUPITER), ('Venera', swe.VENUS), ('Merkur', swe.MERCURY), ('Sunce', swe.SUN), ('Mjesec', swe.MOON)]:
    L = lon_trop(p); wp[nm] = (L, SIGNS[int(L // 30)], int(L % 30) + 1)
SYS['Zapadna astrologija'] = uniq_red([wp[k][2] for k in ['Mars', 'Jupiter', 'Venera', 'Merkur', 'Sunce', 'Mjesec']] + [10, 15])
KEY['Zapadna astrologija'] = '; '.join(f"{k} {v[0]:.2f}° ({v[1]} {v[0]%30:.2f}°)→{v[2]}" for k, v in wp.items())

# Kamea Mjeseca – izvor (Agrippa, De occulta philosophia II, 22) nije moguće potvrditi (mreža blokirana)
KM = np.array([[11, 24, 7, 20, 3], [4, 12, 25, 8, 16], [17, 5, 13, 21, 9], [10, 18, 1, 14, 22], [23, 6, 19, 2, 15]])
assert sorted(KM.flatten().tolist()) == list(range(1, 26))
assert all(KM.sum(0) == 65) and all(KM.sum(1) == 65) and int(np.trace(KM)) == 65 and int(np.trace(KM[:, ::-1])) == 65
diag_main = [int(KM[i, i]) for i in range(5)]; diag_anti = [int(KM[i, 4 - i]) for i in range(5)]
SYS['Kamea Marsa'] = uniq_red(diag_main + diag_anti + [5, 65])
KEY['Kamea Marsa'] = f"kvadrat iz upita (Agrippa II, 22); u kodu provjereno: 5 redova, 5 stupaca, 2 dijagonale = 65, brojevi 1–25 jednom; glavna dijagonala {diag_main}, sporedna {diag_anti}; 65→21"

# Kabala
SYS['Kabala'] = uniq_red([95, 5, 67, 77, 51, 26, 18, 10])
KEY['Kabala'] = 'Maadim 95→7, Gevura 5, IVAN 67→1, mazal 77→11, LOTO 51→7 (dupl.), 26→4, 18, 10'

# Tarot: indeks 1–21 velika arkana I–XXI, 22 = Luda (0), 23–36 Štapovi As–Kralj, 37–50 Pehari, 51–64 Mačevi, 65–78 Novčići
SUITS = ['Štapovi', 'Pehari', 'Mačevi', 'Novčići']
def tarot_name(i):
    if i <= 21: return f"Velika arkana {i}"
    if i == 22: return 'Luda (0)'
    s, r = divmod(i - 23, 14); return f"{SUITS[s]} {['As','2','3','4','5','6','7','8','9','10','Paž','Vitez','Kraljica','Kralj'][r]}"
sun_t = lon_trop(swe.SUN); sign = int(sun_t // 30); dec = int((sun_t % 30) // 10)
suit = {0: 0, 1: 3, 2: 2, 3: 1}[sign % 4]   # vatra Štapovi, zemlja Novčići, zrak Mačevi, voda Pehari
rank = [2, 5, 8][sign % 3] + dec            # kardinalni 2–4, fiksni 5–7, promjenjivi 8–10
dec_card = 23 + suit * 14 + (rank - 1)
deck = list(range(1, 79)); drawn = cached('tarot', lambda: RNG.sample(deck, 10))
SYS['Tarot'] = uniq_red([dec_card] + drawn)
KEY['Tarot'] = f"Sunce {SIGNS[sign]} {sun_t%30:.2f}°, {dec+1}. dekan → {tarot_name(dec_card)} ({dec_card}); izvučeno: " + ', '.join(f"{tarot_name(c)}({c})" for c in drawn)

# Numerologija
def nred(n):
    while n > 9 and n not in (11, 22): n = sum(map(int, str(n)))
    return n
uy = nred(sum(map(int, '2026'))); um = nred(uy + nred(11)); udn = nred(um + nred(24))
assert udn == nred(sum(map(int, '24112026')))
py_ = nred(nred(5) + nred(31) + uy); pm_ = nred(py_ + nred(11)); pdn = nred(pm_ + nred(24))
SYS['Numerologija'] = uniq_red([11, 29, 19, 13, 31, pdn, udn])
KEY['Numerologija'] = f"osobna godina {py_}, osobni mjesec {pm_}, osobni dan {pdn} (rođen 31.05.; 24.11.2026.); univerzalni dan {udn}"

# Lenormand
LN = ['Jahač', 'Djetelina', 'Brod', 'Kuća', 'Stablo', 'Oblaci', 'Zmija', 'Lijes', 'Buket', 'Kosa', 'Bič', 'Ptice', 'Dijete', 'Lisica', 'Medvjed', 'Zvijezde', 'Roda', 'Pas', 'Toranj', 'Vrt', 'Planina', 'Raskrižje', 'Miševi', 'Srce', 'Prsten', 'Knjiga', 'Pismo', 'Gospodin', 'Gospođa', 'Ljiljani', 'Sunce', 'Mjesec', 'Ključ', 'Ribe', 'Sidro', 'Križ']
lc = cached('lenormand', lambda: RNG.sample(range(1, 37), 9))
SYS['Lenormand'] = uniq_red(lc + [lc[i] + lc[i + 1] for i in range(8)])
KEY['Lenormand'] = 'karte: ' + ', '.join(f"{c} {LN[c-1]}" for c in lc)

# Planetarni sat
geo = (LON, LAT, 120.0)
def rise_set(jd0, what):
    res = swe.rise_trans(jd0, swe.SUN, what, geo, 0, 0, swe.FLG_SWIEPH)
    return res[1][0]
jd_mid = swe.julday(2026, 11, 23, 23.0)  # 24.11. 00:00 CET
sr = rise_set(jd_mid, swe.CALC_RISE); ss = rise_set(jd_mid, swe.CALC_SET); sr2 = rise_set(ss, swe.CALC_RISE)
def loct(j):
    y, m, dd, h = swe.revjul(j); return (datetime(y, m, dd, tzinfo=ZoneInfo('UTC')) + timedelta(seconds=round(h * 3600))).astimezone(ZoneInfo('Europe/Zagreb'))
CHAL = ['Saturn', 'Jupiter', 'Mars', 'Sunce', 'Venera', 'Merkur', 'Mjesec']
PNUM = {'Sunce': 1, 'Mjesec': 2, 'Jupiter': 3, 'Merkur': 5, 'Venera': 6, 'Saturn': 8, 'Mars': 9}  # u skladu s 'Mars = 9', 'Mjesec = 2'
assert sr < ss < sr2
if jd < ss:
    hlen = (ss - sr) / 12; hidx = int((jd - sr) / hlen) + 1; hstart = sr + (hidx - 1) * hlen
else:
    hlen = (sr2 - ss) / 12; hidx = 12 + int((jd - ss) / hlen) + 1; hstart = ss + (hidx - 13) * hlen
hend = hstart + hlen
ruler = CHAL[(CHAL.index('Mars') + hidx - 1) % 7]
ph = [PNUM[ruler], hidx, hidx - 12 if hidx > 12 else hidx, 9, loct(sr).hour, loct(sr).minute, loct(ss).hour, loct(ss).minute]
kam_note = ''
if ruler == 'Mars': ph += diag_main + diag_anti; kam_note = '; planet sata Mars → dodane dijagonale kamee Marsa'
elif ruler in ('Sunce', 'Jupiter'): kam_note = f'; kamea ({ruler}) NIJE IZRAČUNATA – kvadrat nije zadan ni potvrđen'
SYS['Planetarni sat'] = uniq_red(ph)
E['planetarni_sat'] = dict(od_pocetka_min=(jd - hstart) * 1440, do_kraja_min=(hend - jd) * 1440, start=f"{loct(hstart):%H:%M:%S}", end=f"{loct(hend):%H:%M:%S}")
KEY['Planetarni sat'] = (f"izlazak {loct(sr):%H:%M:%S}, zalazak {loct(ss):%H:%M:%S}, sutrašnji izlazak {loct(sr2):%H:%M:%S} (CET); 19:00 = {hidx}. sat "
    f"({hidx-12 if hidx > 12 else hidx}. {'noćni' if hidx > 12 else 'dnevni'}, {loct(hstart):%H:%M:%S}–{loct(hend):%H:%M:%S}), vladar {ruler} (={PNUM[ruler]}); "
    f"od početka sata {(jd-hstart)*1440:.1f} min, do kraja {(hend-jd)*1440:.1f} min; 1. dnevni sat = Mars" + kam_note)

for k in SYS: assert len(SYS[k]) == len(set(SYS[k])) and all(1 <= v <= 22 for v in SYS[k])

# Plečkov filtar
votes = {v: 0 for v in range(1, 23)}; possum = {v: 0 for v in range(1, 23)}
for k, lst in SYS.items():
    for pos, v in enumerate(lst[:6], 1): votes[v] += 1; possum[v] += pos
PLECKO = sorted(range(1, 23), key=lambda v: (-votes[v], possum[v] if votes[v] else 0, v))
E['plecko'] = [(v, votes[v], possum[v]) for v in PLECKO]

# ---------------------------------------------------------------- §2 pravila
def bm(c): return sum(1 << (v - 1) for v in c)
def runmax(c):
    s = sorted(c); return max(len(list(g)) for _, g in itertools.groupby(enumerate(s), lambda t: t[1] - t[0]))
def hard_ok(c, others):
    if len(set(c)) != 11 or not all(1 <= v <= 22 for v in c): return False
    for o in others:
        ov = len(set(c) & set(o))
        if ov < 3 or ov > 8: return False   # pokriva i komplement (0) i duplikat (11)
    return True
def soft_ok(c):
    ev = sum(1 for v in c if v % 2 == 0); return ev in (5, 6) and runmax(c) <= 4
FIXLOG = {}
def fix(order, source, others, tag=''):
    """order: brojevi redom dodavanja. 1 zamjena (zadnji dodani, pa unatrag), zatim do 2 zamjene sljedećima iz source; tek onda samo tvrda pravila."""
    c = list(order[:11])
    def good(t, soft): return hard_ok(t, others) and (soft_ok(t) or not soft)
    if good(c, True): FIXLOG[tag] = 'bez zamjene'; return c, True
    for pos in range(10, -1, -1):
        for cand in source:
            if cand in c: continue
            t = c.copy(); t[pos] = cand
            if good(t, True): FIXLOG[tag] = f'{c[pos]}→{cand}'; return t, True
    for p1, p2 in itertools.combinations(range(10, -1, -1), 2):
        for a, b in itertools.permutations([x for x in source if x not in c], 2):
            t = c.copy(); t[p1] = a; t[p2] = b
            if good(t, True): FIXLOG[tag] = f'{c[p1]}→{a}, {c[p2]}→{b}'; return t, True
    for pos in range(10, -1, -1):
        for cand in source:
            if cand in c: continue
            t = c.copy(); t[pos] = cand
            if good(t, False): FIXLOG[tag] = f'{c[pos]}→{cand} (meko pravilo nezadovoljeno)'; return t, False
    raise RuntimeError('nema valjane kombinacije')
def interleave(lists, fill):
    out = []
    for i in range(max(map(len, lists))):
        for l in lists:
            if i < len(l) and l[i] not in out and len(out) < 11: out.append(l[i])
    for v in fill:
        if len(out) >= 11: break
        if v not in out: out.append(v)
    return out

FINAL = []   # (oznaka, tip, sustav, kombinacija, soft_ok)
e1, s1ok = fix(PLECKO[:11], PLECKO[11:], [], 'E1')
FINAL.append(('E1', 'ezoterijska', 'Plečkov filtar (svi sustavi)', sorted(e1), s1ok))
e2, s2ok = fix(interleave([SYS['Meihua'], SYS['I Ching'], SYS['Jyotiš']], PLECKO), PLECKO, [f[3] for f in FINAL], 'E2')
FINAL.append(('E2', 'ezoterijska', 'Istok: Meihua + I Ching + Jyotiš', sorted(e2), s2ok))
e3, s3ok = fix(interleave([SYS['Zapadna astrologija'], SYS['Kamea Marsa'], SYS['Kabala'], SYS['Tarot'], SYS['Numerologija']], PLECKO), PLECKO, [f[3] for f in FINAL], 'E3')
FINAL.append(('E3', 'ezoterijska', 'Zapad: zapadna astr. + kamea Marsa + kabala + tarot + numerologija', sorted(e3), s3ok))
R['t_ezo'] = time.time() - T0

# ---------------------------------------------------------------- §4 statistika
def rank_hot(hist):
    f = np.zeros(23, int)
    for s in hist:
        for v in s: f[v] += 1
    return sorted(range(1, 23), key=lambda v: (-f[v], v))
def m_hot_all(h): return rank_hot(h)
def m_hot50(h): return rank_hot(h[-50:])
def m_hot100(h): return rank_hot(h[-100:])
def m_cold(h): return rank_hot(h)[::-1][:11] + rank_hot(h)[::-1][11:]
def m_pareto(h):
    w = h[-max(1, int(round(0.2 * len(h)))):]
    top = rank_hot(w)[:int(round(0.2 * 22))]
    return top + [v for v in rank_hot(h) if v not in top]
def m_overdue(h):
    gap = {}
    for v in range(1, 23):
        g = 0
        for s in reversed(h):
            if v in s: break
            g += 1
        gap[v] = g
    return sorted(range(1, 23), key=lambda v: (-gap[v], v))
def m_last(h):
    last = sorted(h[-1]); rest = sorted(set(range(1, 23)) - h[-1])
    return last + rest
def m_compl(h):
    rest = sorted(set(range(1, 23)) - h[-1]); return rest + sorted(h[-1])
def m_rand(h):
    p = list(range(1, 23)); RNG.shuffle(p); return p
METHODS = {'vrući/hladni (cijela povijest)': m_hot_all, 'vrući (zadnjih 50)': m_hot50, 'vrući (zadnjih 100)': m_hot100,
           'Pareto 20/20': m_pareto, 'prezreli (najdulje neizvučeni)': m_overdue,
           'zadnje kolo / komplement': m_last, 'slučajna (secrets) – kontrola': m_rand}
START = 99  # od 100. kola nadalje
nwf = N - START
var_k = sum(k * k * PK[k] for k in range(12)) - 5.5 ** 2
WF = {}
for name, fn in METHODS.items():
    hits = []
    for t in range(START, N):
        c = set(fn(D[:t])[:11]); hits.append(len(c & D[t]))
    hits = np.array(hits)
    dist = {k: int((hits == k).sum()) for k in range(12)}
    W = int(sum(dist[k] for k in WIN_K))
    cls = {c: int(sum(dist[k] for k in WIN_K if WIN_K[k] == c)) for c in ['11/0', '10/1', '9/2', '8/3']}
    expW = nwf * P_WIN; z = (W - expW) / math.sqrt(nwf * P_WIN * (1 - P_WIN))
    p = 1 - stats.norm.cdf(z); pb = stats.binom.sf(W - 1, nwf, P_WIN)
    zm = (hits.mean() - 5.5) / math.sqrt(var_k / nwf); pm = 2 * (1 - stats.norm.cdf(abs(zm)))
    # χ² GOF raspodjele pogodaka (spojeni repovi 0–3 i 8–11)
    bins = [(0, 1, 2, 3), (4,), (5,), (6,), (7,), (8, 9, 10, 11)]
    O = [sum(dist[k] for k in b) for b in bins]; Ex = [nwf * sum(PK[k] for k in b) for b in bins]
    chi, pchi = stats.chisquare(O, Ex)
    WF[name] = dict(n=nwf, dist=dist, W=W, expW=expW, cls=cls, z=z, p_one=pb, p_bonf=min(1, pb * len(METHODS)),
                    mean=hits.mean(), z_mean=zm, p_mean_bonf=min(1, pm * len(METHODS)), chi=chi, p_chi_bonf=min(1, pchi * len(METHODS)))
R['WF'] = WF
R['expected_dist'] = {k: nwf * PK[k] for k in range(12)}

# testovi slučajnosti
freq = np.array([sum(1 for s in D if v in s) for v in range(1, 23)])
nc = N * 0.25 * 22 / 21
chi_f = float(((freq - N / 2) ** 2).sum() / nc); p_f = float(stats.chi2.sf(chi_f, 21))
chi_naive, p_naive = stats.chisquare(freq)
h1f = np.array([sum(1 for s in D[:N // 2] if v in s) for v in range(1, 23)])
h2f = np.array([sum(1 for s in D[N // 2:] if v in s) for v in range(1, 23)])
rho, prho = stats.spearmanr(h1f, h2f)
ev = np.array([sum(1 for v in s if v % 2 == 0) for s in D])
binsP = [(0, 1, 2, 3), (4,), (5,), (6,), (7,), (8, 9, 10, 11)]
Op = [int(sum((ev == k).sum() for k in b)) for b in binsP]; Ep = [N * sum(PK[k] for k in b) for b in binsP]
chi_p, p_p = stats.chisquare(Op, Ep)
# zbroj: točna raspodjela iz enumeracije
ALL = np.arange(1 << 22, dtype=np.uint32); ALL = ALL[np.bitwise_count(ALL) == 11]
assert len(ALL) == TOTAL
VALS = np.arange(1, 23)
sums_all = np.zeros(len(ALL), np.int32)
for i in range(22): sums_all += ((ALL >> i) & 1).astype(np.int32) * (i + 1)
sm = np.array([sum(s) for s in D])
qs = np.quantile(sums_all, np.linspace(0, 1, 11)[1:-1])
edges = np.concatenate([[-1], qs, [1000]])
Os = np.histogram(sm, edges)[0]; Es = np.histogram(sums_all, edges)[0] / TOTAL * N
chi_s, p_s = stats.chisquare(Os, Es)
z_s = (sm.mean() - 126.5) / math.sqrt(sums_all.var() / N)
R['rand'] = dict(freq=freq.tolist(), chi_f=chi_f, p_f=p_f, chi_naive=float(chi_naive), p_naive=float(p_naive), rho=float(rho), prho=float(prho),
                 par_O=Op, par_E=Ep, chi_p=float(chi_p), p_p=float(p_p), sum_mean=float(sm.mean()), sum_sd=float(sm.std(ddof=1)),
                 sum_theo_sd=float(sums_all.std()), z_s=float(z_s), chi_s=float(chi_s), p_s=float(p_s))

# odabir S1–S3
cand = [m for m in METHODS if 'kontrola' not in m]
ranked = sorted(cand, key=lambda m: (-WF[m]['z'], WF[m]['p_chi_bonf']))
sig = [m for m in cand if WF[m]['p_bonf'] < 0.05]
R['S_signif'] = sig
def typical(c): return 116.5 <= sum(c) <= 136.5 and sum(1 for v in c if v % 2 == 0) in (5, 6)
used = []
for m in ranked:
    if len(used) == 3: break
    rk = METHODS[m](D)
    others = [f[3] for f in FINAL]
    c = list(rk[:11])
    ok = None
    if hard_ok(c, others) and soft_ok(c) and (sig or typical(c)): ok = c
    else:
        for pos in range(10, -1, -1):
            for cnd in rk[11:]:
                if cnd in c: continue
                t = c.copy(); t[pos] = cnd
                if hard_ok(t, others) and soft_ok(t) and (sig or typical(t)): ok = t; break
            if ok: break
        if ok is None:  # dvije zamjene, redom po rangu metode
            for a, b in itertools.combinations(range(11), 2):
                for x, y in itertools.permutations(rk[11:], 2):
                    t = c.copy(); t[10 - a] = x; t[10 - b] = y
                    if len(set(t)) == 11 and hard_ok(t, others) and soft_ok(t) and (sig or typical(t)): ok = t; break
                if ok: break
    if ok is None: continue
    used.append(m)
    FINAL.append((f'S{len(used)}', 'statistička', m + ('' if sig else ' (statistički tipična)'), sorted(ok), True))
R['S_rang'] = ranked
R['t_stat'] = time.time() - T0

# ---------------------------------------------------------------- §5 logičke
WINLUT = np.zeros(12, bool)
for k in WIN_K: WINLUT[k] = True
def winarr(c): return WINLUT[np.bitwise_count(ALL & np.uint32(bm(c)))]
def rand_combo(): return sorted(RNG.sample(range(1, 23), 11))
fixed = [f[3] for f in FINAL]
cov_fixed = np.zeros(TOTAL, bool)
for c in fixed: cov_fixed |= winarr(c)
L = []
for j in range(3):
    others = fixed + L
    cov = cov_fixed.copy()
    for c in L: cov |= winarr(c)
    best, bg = None, -1
    tries = 0
    while tries < 1500:
        c = rand_combo()
        if not (hard_ok(c, others) and soft_ok(c)): continue
        tries += 1
        g = int((winarr(c) & ~cov).sum())
        if g > bg: best, bg = c, g
    L.append(best)
def total_cov(Ls):
    cov = cov_fixed.copy()
    for c in Ls: cov |= winarr(c)
    return int(cov.sum())
cur = total_cov(L); R['L_greedy'] = cur
improved, it, TLIM = True, 0, time.time() + 150
while improved and time.time() < TLIM:
    improved = False; it += 1
    for j in range(3):
        others = fixed + [L[i] for i in range(3) if i != j]
        cov = cov_fixed.copy()
        for i in range(3):
            if i != j: cov |= winarr(L[i])
        base = int(cov.sum()); bestc, bestv = L[j], base + int((winarr(L[j]) & ~cov).sum())
        for out_v in L[j]:
            for in_v in range(1, 23):
                if in_v in L[j]: continue
                t = sorted([v for v in L[j] if v != out_v] + [in_v])
                if not (hard_ok(t, others) and soft_ok(t)): continue
                val = base + int((winarr(t) & ~cov).sum())
                if val > bestv: bestc, bestv = t, val
        if bestc != L[j]: L[j] = bestc; improved = True
cur = total_cov(L); R['L_iter'] = it; R['L_final'] = cur
for j, c in enumerate(L): FINAL.append((f'L{j+1}', 'logička', 'maks. pokrivenost (greedy + lokalna zamjena)', c, True))

combos = [f[3] for f in FINAL]
assert len(combos) == 9 and all(hard_ok(combos[i], combos[:i] + combos[i+1:]) for i in range(9))
ovs = [len(set(a) & set(b)) for a, b in itertools.combinations(combos, 2)]
masks = {bm(c) for c in combos}; full = (1 << 22) - 1
R['set_checks'] = dict(duplikati=len(masks) != 9, komplementi=any((full ^ m) in masks for m in masks), ov_min=min(ovs), ov_max=max(ovs))
covset = np.zeros(TOTAL, bool); jack = np.zeros(TOTAL, bool)
for c in combos:
    k = np.bitwise_count(ALL & np.uint32(bm(c))); covset |= WINLUT[k]; jack |= (k == 11) | (k == 0)
cls_any = {}
for cname, ks in [('11/0', (11, 0)), ('10/1', (10, 1)), ('9/2', (9, 2)), ('8/3', (8, 3))]:
    a = np.zeros(TOTAL, bool)
    for c in combos: a |= np.isin(np.bitwise_count(ALL & np.uint32(bm(c))), ks)
    cls_any[cname] = int(a.sum())
R['set_prob'] = dict(P_SiN=int(jack.sum()), P_any=int(covset.sum()), E_win=9 * P_WIN, cls_any=cls_any, total=TOTAL)

# 1000 slučajnih setova
def rand_set():
    while True:
        s = [rand_combo() for _ in range(9)]; ms = {bm(c) for c in s}
        if len(ms) == 9 and not any((full ^ m) in ms for m in ms): return s
RS = [rand_set() for _ in range(1000)]
rs_any = []
for s in RS:
    a = np.zeros(TOTAL, bool)
    for c in s: a |= winarr(c)
    rs_any.append(int(a.sum()))
rs_any = np.array(rs_any)
R['rand_sets'] = dict(mean=float(rs_any.mean()), sd=float(rs_any.std()), min=int(rs_any.min()), max=int(rs_any.max()),
                      pct=float((rs_any < covset.sum()).mean() * 100 + (rs_any == covset.sum()).mean() * 50))
R['t_L'] = time.time() - T0

# ---------------------------------------------------------------- §6 backtest
DM = np.array([bm(s) for s in D], dtype=np.uint32)
bt = []
for f in FINAL:
    k = np.bitwise_count(DM & np.uint32(bm(f[3])))
    dist = {kk: int((k == kk).sum()) for kk in range(12)}
    cls = {c: int(sum(dist[kk] for kk in WIN_K if WIN_K[kk] == c)) for c in ['11/0', '10/1', '9/2', '8/3']}
    bt.append(dict(id=f[0], dist=dist, cls=cls, W=sum(cls.values())))
setW = sum(b['W'] for b in bt)
rsW = []
for s in RS:
    w = 0
    for c in s: w += int(WINLUT[np.bitwise_count(DM & np.uint32(bm(c)))].sum())
    rsW.append(w)
rsW = np.array(rsW)
R['backtest'] = dict(per=bt, setW=setW, expW=9 * N * P_WIN, rs_mean=float(rsW.mean()), rs_sd=float(rsW.std()),
                     pct=float((rsW < setW).mean() * 100 + (rsW == setW).mean() * 50), draws_any=int(sum(1 for i in range(N) if any(WINLUT[bin(int(DM[i]) & bm(c)).count('1')] for c in combos))))

# ---------------------------------------------------------------- izlaz
y23 = pd.read_csv(f'{OUT}/sin_2026-11-23.csv', sep=';')
assert dates[N - 1] == '23.11.2026'
UH = {'E1': 5, 'E2': 5, 'E3': 6, 'S1': 8, 'S2': 3, 'S3': 6, 'L1': 5, 'L2': 4, 'L3': 4}
R['povratna_327'] = [dict(id=r['#'], hits=len({int(r[b]) for b in B} & D[N - 1]), korisnik=UH[r['#']], razred=WIN_K.get(len({int(r[b]) for b in B} & D[N - 1]), '-')) for _, r in y23.iterrows()]
R['FIXLOG'] = FIXLOG
R['FINAL'] = [dict(id=f[0], tip=f[1], sustav=f[2], brojevi=f[3], parni=sum(1 for v in f[3] if v % 2 == 0), zbroj=sum(f[3]), meko_ok=soft_ok(f[3]), niz=runmax(f[3])) for f in FINAL]
R['SYS'] = SYS; R['KEY'] = KEY; R['E'] = E
R['versions'] = {p: md.version(p) for p in ['numpy', 'pandas', 'scipy', 'lunar_python', 'sxtwl', 'openpyxl']}
R['versions']['pyswisseph'] = swe.version + ' (izgrađen iz pyswisseph-2.10.3.2.tar.gz)'
R['versions']['python'] = sys.version.split()[0]
R['t_total'] = time.time() - T0
with open(f'{OUT}/sin_2026-11-24.csv', 'w', encoding='utf-8') as fh:
    fh.write('#;tip;sustav;' + ';'.join(B) + '\n')
    for f in FINAL: fh.write(f"{f[0]};{f[1]};{f[2]};" + ';'.join(map(str, f[3])) + '\n')
with open(f'{OUT}/rezultati_2026-11-24.json', 'w', encoding='utf-8') as fh:
    json.dump(R, fh, ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
print(json.dumps({k: R[k] for k in ['FINAL', 'set_checks', 'set_prob', 'rand_sets', 't_total']}, ensure_ascii=False, default=str))
