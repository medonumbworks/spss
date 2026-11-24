# -*- coding: utf-8 -*-
"""Eurojackpot (5/50 + 2/12) – 6 kombinacija za uto 24.11.2026. (Helsinki 21:00 EET = 19:00 UTC).
Prilagođeno iz sve_ili_nista/sin_2026_11_24.py i l_opt_2026_11_24.py.
PYTHONPATH=<pyswisseph> python3 ej_2026_11_24.py <ej_povijest.csv> <ephe dir> <HKO T2026e.txt> <izlazni dir> <sekunde za L>"""
import sys, os, json, time, secrets, itertools, math, re, importlib.metadata as md
from math import comb
from fractions import Fraction
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import numpy as np, pandas as pd
from scipy import stats
import swisseph as swe
from lunar_python import Solar
import sxtwl

T0 = time.time()
HIST, EPHE, HKO, OUT, BUD = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], float(sys.argv[5])
RNG = secrets.SystemRandom()
CACHE = f'{OUT}/slucajno_ej_2026-11-24.json'   # secrets-izvlačenja se spremaju i ne izvlače ponovno
_cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
def cached(key, fn):
    if key not in _cache:
        _cache[key] = fn(); json.dump(_cache, open(CACHE, 'w'))
    return _cache[key]
R = {}

def redM(x):
    x = int(x)
    while x > 50: x -= 50
    return x + 50 if x <= 0 else x
def redE(x):
    x = int(x)
    while x > 12: x -= 12
    return x + 12 if x <= 0 else x
def uniq(seq, f):
    out = []
    for v in seq:
        r = f(v)
        if r not in out: out.append(r)
    return out

# ------------------------------------------------------------ §1 podaci
h = pd.read_csv(HIST, sep=';')
MC, EC = ['M1', 'M2', 'M3', 'M4', 'M5'], ['E1', 'E2']
DM = [set(map(int, r)) for r in h[MC].values]; DE = [set(map(int, r)) for r in h[EC].values]
N = len(DM); dates = h['Datum'].tolist()
chk = dict(n=N, prvo=dates[0], zadnje=dates[-1],
           glavni_1_50=bool(all(all(1 <= v <= 50 for v in s) and len(s) == 5 for s in DM)),
           euro_1_12=bool(all(all(1 <= v <= 12 for v in s) and len(s) == 2 for s in DE)),
           dupli_datumi=int(h['Datum'].duplicated().sum()), dupli_izvlacenja=int(len(DM) - len({(tuple(sorted(a)), tuple(sorted(b))) for a, b in zip(DM, DE)})))
USER5 = {'20.11.2026': ([15, 28, 29, 34, 35], [5, 9]), '17.11.2026': ([1, 5, 7, 15, 19], [1, 11]), '13.11.2026': ([5, 12, 27, 35, 44], [4, 12]),
         '10.11.2026': ([7, 15, 17, 40, 44], [5, 11]), '06.11.2026': ([7, 30, 32, 35, 44], [2, 6])}
chk['zadnjih5_podudara'] = all(sorted(DM[dates.index(d)]) == m and sorted(DE[dates.index(d)]) == e for d, (m, e) in USER5.items())
R['podaci'] = chk

# klase dobitaka
CLASSES = [(5, 2), (5, 1), (5, 0), (4, 2), (4, 1), (3, 2), (4, 0), (2, 2), (3, 1), (3, 0), (1, 2), (2, 1)]
CL = {c: f"{c[0]}+{c[1]}" for c in CLASSES}
def is_win(m, e): return (m, e) in CL
PM = {m: Fraction(comb(5, m) * comb(45, 5 - m), comb(50, 5)) for m in range(6)}
PE = {e: Fraction(comb(2, e) * comb(10, 2 - e), comb(12, 2)) for e in range(3)}
P_WIN = sum(PM[m] * PE[e] for m, e in CLASSES)
R['P_win_kombinacija'] = str(P_WIN); R['P_win_float'] = float(P_WIN)

# ------------------------------------------------------------ §3 ezoterija
swe.set_ephe_path(EPHE)
LAT, LON = 60.17, 24.94
loc = datetime(2026, 11, 24, 21, 0, tzinfo=ZoneInfo('Europe/Helsinki'))
utc = loc.astimezone(ZoneInfo('UTC')); assert (utc.hour, utc.minute) == (19, 0)
jd = swe.julday(utc.year, utc.month, utc.day, utc.hour + utc.minute / 60)
E = {'trenutak_lokalno': loc.isoformat(), 'trenutak_zagreb': loc.astimezone(ZoneInfo('Europe/Zagreb')).isoformat(), 'trenutak_UTC': utc.isoformat(), 'JD_UT': jd}
SYS, EUR, KEY = {}, {}, {}
def add(name, raw, key, euro=None):
    SYS[name] = uniq(raw, redM); EUR[name] = euro if euro is not None else uniq(raw, redE); KEY[name] = key

# Sinkronicitet
last5 = list(range(N - 5, N))
cnt = {}
for i in last5:
    for v in DM[i]: cnt[v] = cnt.get(v, 0) + 1
lastseen = {v: max(i for i in last5 if v in DM[i]) for v in cnt}
rep = sorted([v for v in cnt if cnt[v] >= 2], key=lambda v: (-cnt[v], -lastseen[v], v))
ecnt = {}
for i in last5:
    for v in DE[i]: ecnt[v] = ecnt.get(v, 0) + 1
elast = {v: max(i for i in last5 if v in DE[i]) for v in ecnt}
erep = sorted([v for v in ecnt if ecnt[v] >= 2], key=lambda v: (-ecnt[v], -elast[v], v))
doy = loc.timetuple().tm_yday
raw_s = rep + [24, 11, 1, doy, 35, 31, 5, 18, 20] + sorted(DM[N - 1])
def runs(s):
    s = sorted(s); return [(a, b) for a, b in zip(s, s[1:]) if b == a + 1]
E['pravilnosti'] = dict(
    izvlacenja=[(dates[i], sorted(DM[i]), sorted(DE[i])) for i in last5],
    glavni_2plus={v: cnt[v] for v in rep}, euro_2plus={v: ecnt[v] for v in erep},
    neparni=[sum(v % 2 for v in DM[i]) for i in last5], zbroj=[sum(DM[i]) for i in last5],
    susjedni_parovi=[runs(DM[i]) for i in last5], ponovljeno_iz_prethodnog=[len(DM[i] & DM[i - 1]) for i in last5],
    desetice=[sorted({(v - 1) // 10 for v in DM[i]}) for i in last5], nijednom_1_50=len(set(range(1, 51)) - set(cnt)))
add('Sinkronicitet', raw_s, f"glavni ≥2×: {rep}; euro ≥2×: {erep}; redni dan {doy}→{redM(doy)}; starost 35; zadnje izvlačenje {sorted(DM[N-1])}",
    euro=erep + [x for x in uniq(raw_s, redE) if x not in erep])

add('Arhetipovi', [4, 12, 1, 16, 7, 10, 28, 22, 40, 36], 'zadani popis')

# I Ching / Meihua
TRI = {'Qian': (1, 1, 1), 'Dui': (1, 1, 0), 'Li': (1, 0, 1), 'Zhen': (1, 0, 0), 'Xun': (0, 1, 1), 'Kan': (0, 1, 0), 'Gen': (0, 0, 1), 'Kun': (0, 0, 0)}
XT = {'Qian': 1, 'Dui': 2, 'Li': 3, 'Zhen': 4, 'Xun': 5, 'Kan': 6, 'Gen': 7, 'Kun': 8}; XT_INV = {v: k for k, v in XT.items()}
ORD = ['Qian', 'Zhen', 'Kan', 'Gen', 'Kun', 'Xun', 'Li', 'Dui']
KW = [[1, 25, 6, 33, 12, 44, 13, 10], [34, 51, 40, 62, 16, 32, 55, 54], [5, 3, 29, 39, 8, 48, 63, 60], [26, 27, 4, 52, 23, 18, 22, 41],
      [11, 24, 7, 15, 2, 46, 36, 19], [9, 42, 59, 53, 20, 57, 37, 61], [14, 21, 64, 56, 35, 50, 30, 38], [43, 17, 47, 31, 45, 28, 49, 58]]
assert sorted(sum(KW, [])) == list(range(1, 65))
L2T = {v: k for k, v in TRI.items()}
def hexnum(lines):
    lo, up = L2T[tuple(lines[:3])], L2T[tuple(lines[3:])]; return KW[ORD.index(up)][ORD.index(lo)], up, lo
throws = cached('iching', lambda: [[RNG.choice([3, 2]) for _ in range(3)] for _ in range(6)])
vals = [sum(t) for t in throws]
prim = [1 if v in (7, 9) else 0 for v in vals]; chl = [i + 1 for i, v in enumerate(vals) if v in (6, 9)]
chg = [(1 - l) if (i + 1) in chl else l for i, l in enumerate(prim)]
h1, up1, lo1 = hexnum(prim); h2, _, _ = hexnum(chg); hn, _, _ = hexnum(prim[1:4] + prim[2:5])
add('I Ching', [h1, h2, hn] + chl + [XT[up1], XT[lo1]], f"bacanja {vals} (odozdo); heksagram {h1} ({up1}/{lo1}) → {h2}; nuklearni {hn}; promjenjive crte {chl or 'nema'}")

lun = Solar.fromYmdHms(2026, 11, 24, 21, 0, 0).getLunar()
LM, LD = abs(lun.getMonth()), lun.getDay()
sx = sxtwl.fromSolar(2026, 11, 24); assert (sx.getLunarMonth(), sx.getLunarDay()) == (LM, LD)
hk = open(HKO, encoding='utf-8', errors='replace').read().splitlines()
hm_, hd_ = None, None
for line in hk:
    m = re.match(r'(\d{4})/(\d+)/(\d+)\s+(\S+)(?:\s+(\S+))?', line)
    if not m: continue
    if 'Lunar Month' in line: hm_ = int(re.search(r'(\d+)\w\w Lunar Month', line).group(1)); dday = 1
    else: dday = int(m.group(4))
    if (int(m.group(2)), int(m.group(3))) == (11, 24): hd_ = dday; break
assert (hm_, hd_) == (LM, LD), (hm_, hd_, LM, LD)
bj = loc.astimezone(ZoneInfo('Asia/Shanghai')); lunBJ = Solar.fromYmdHms(bj.year, bj.month, bj.day, bj.hour, 0, 0).getLunar()
E['lunarni'] = dict(lunar_python=f"{lun.getYearInGanZhi()}, mj. {LM}, dan {LD}, sat {lun.getTimeZhi()}", sxtwl=f"mj. {sx.getLunarMonth()}, dan {sx.getLunarDay()}",
                    HKO=f"mj. {hm_}, dan {hd_}", peking=f"{bj:%d.%m.%Y %H:%M} = mj. {abs(lunBJ.getMonth())}, dan {lunBJ.getDay()}, sat {lunBJ.getTimeZhi()}")
Y, H = 7, 12
assert lun.getTimeZhi() == '亥'
up = (Y + LM + LD) % 8 or 8; lo = (Y + LM + LD + H) % 8 or 8; mv = (Y + LM + LD + H) % 6 or 6
ml = list(TRI[XT_INV[lo]]) + list(TRI[XT_INV[up]]); mh1 = KW[ORD.index(XT_INV[up])][ORD.index(XT_INV[lo])]
ml2 = ml.copy(); ml2[mv - 1] = 1 - ml2[mv - 1]; mh2, _, _ = hexnum(ml2); mhn, _, _ = hexnum(ml[1:4] + ml[2:5])
add('Meihua', [mh1, mh2, mhn, mv, up, lo], f"Y=7 (午), M={LM}, D={LD}, H=12 (亥); gornji {up} {XT_INV[up]}, donji {lo} {XT_INV[lo]}, crta {mv}; heksagram {mh1} → {mh2}, nuklearni {mhn}")

# Jyotiš
swe.set_sid_mode(swe.SIDM_LAHIRI)
FS = swe.FLG_SWIEPH | swe.FLG_SIDEREAL | swe.FLG_SPEED
def sid(p):
    x, f = swe.calc_ut(jd, p, FS); assert f & swe.FLG_SWIEPH; return x[0], x[3]
(ms, msp), (ss_, ssp), (rs, _) = sid(swe.MOON), sid(swe.SUN), sid(swe.MEAN_NODE)
NAK = 360 / 27
nak_m = int(ms // NAK) + 1; tithi = int(((ms - ss_) % 360) // 12) + 1; yoga = int(((ms + ss_) % 360) // NAK) + 1; nak_r = int(rs // NAK) + 1
NAKN = ['Ashvini', 'Bharani', 'Krittika', 'Rohini', 'Mrigashira', 'Ardra', 'Punarvasu', 'Pushya', 'Ashlesha', 'Magha', 'P.Phalguni', 'U.Phalguni', 'Hasta', 'Chitra', 'Swati', 'Vishakha', 'Anuradha', 'Jyeshtha', 'Mula', 'P.Ashadha', 'U.Ashadha', 'Shravana', 'Dhanishta', 'Shatabhisha', 'P.Bhadrapada', 'U.Bhadrapada', 'Revati']
YN = ['Vishkambha', 'Priti', 'Ayushman', 'Saubhagya', 'Shobhana', 'Atiganda', 'Sukarma', 'Dhriti', 'Shula', 'Ganda', 'Vriddhi', 'Dhruva', 'Vyaghata', 'Harshana', 'Vajra', 'Siddhi', 'Vyatipata', 'Variyana', 'Parigha', 'Shiva', 'Siddha', 'Sadhya', 'Shubha', 'Shukla', 'Brahma', 'Indra', 'Vaidhriti']
def dist(x, span, speed):
    x %= 360; st = (x // span) * span; return (x - st) / speed * 1440, (st + span - x) / speed * 1440
gN, gT, gY = dist(ms, NAK, msp), dist(ms - ss_, 12, msp - ssp), dist(ms + ss_, NAK, msp + ssp)
TN = ['Pratipada', 'Dvitiya', 'Tritiya', 'Chaturthi', 'Panchami', 'Shashthi', 'Saptami', 'Ashtami', 'Navami', 'Dashami', 'Ekadashi', 'Dvadashi', 'Trayodashi', 'Chaturdashi']
tname = 'Purnima' if tithi == 15 else 'Amavasya' if tithi == 30 else ('Shukla ' if tithi < 15 else 'Krishna ') + TN[(tithi - 1) % 15]
add('Jyotiš', [20, nak_m, tithi, 9, yoga, nak_r],
    f"Mjesec {ms:.2f}° sid. → nakšatra {nak_m} {NAKN[nak_m-1]} ({gN[0]:.0f} min od ulaska, {gN[1]:.0f} min do izlaska); tithi {tithi} {tname} ({gT[0]:.0f}/{gT[1]:.0f} min); "
    f"yoga {yoga} {YN[yoga-1]} ({gY[0]:.0f}/{gY[1]:.0f} min); Mars=9; srednji Rahu {rs:.2f}° → nakšatra {nak_r} {NAKN[nak_r-1]}; ayanamsa {swe.get_ayanamsa_ut(jd):.4f}°")

# Zapadna (tropski)
SIGNS = ['Ovan', 'Bik', 'Blizanci', 'Rak', 'Lav', 'Djevica', 'Vaga', 'Škorpion', 'Strijelac', 'Jarac', 'Vodenjak', 'Ribe']
wp = {}
for nm, p in [('Mars', swe.MARS), ('Jupiter', swe.JUPITER), ('Venera', swe.VENUS), ('Merkur', swe.MERCURY), ('Sunce', swe.SUN), ('Mjesec', swe.MOON)]:
    x, f = swe.calc_ut(jd, p, swe.FLG_SWIEPH | swe.FLG_SPEED); assert f & swe.FLG_SWIEPH
    wp[nm] = (x[0], SIGNS[int(x[0] // 30)], int(x[0] % 30) + 1, x[3] < 0)
add('Zapadna astrologija', [wp[k][2] for k in wp] + [10, 15],
    '; '.join(f"{k} {v[1]} {v[0]%30:.2f}°{' R' if v[3] else ''} → {v[2]}" for k, v in wp.items()))

# Kamea Marsa
KM = np.array([[11, 24, 7, 20, 3], [4, 12, 25, 8, 16], [17, 5, 13, 21, 9], [10, 18, 1, 14, 22], [23, 6, 19, 2, 15]])
assert sorted(KM.flatten().tolist()) == list(range(1, 26))
assert all(KM.sum(0) == 65) and all(KM.sum(1) == 65) and int(np.trace(KM)) == 65 and int(np.trace(KM[:, ::-1])) == 65
dmain = [int(KM[i, i]) for i in range(5)]; danti = [int(KM[i, 4 - i]) for i in range(5)]
add('Kamea Marsa', dmain + danti + [5, 65], f"provjereno u kodu: 5 redova, 5 stupaca, 2 dijagonale = 65, brojevi 1–25 jednom; glavna {dmain}, sporedna {danti}")

add('Kabala', [95, 5, 67, 77, 51, 26, 18, 10], 'Maadim 95→45 (euro 11), Gevura 5, IVAN 67→17, mazal 77→27, LOTO 51→1, 26, 18, 10')

# Tarot: 1–21 Mag…Svijet, 22 Luda, 23–36 Štapovi, 37–50 Pehari, 51–64 Mačevi, 65–78 Pentakli
SUITS = ['Štapovi', 'Pehari', 'Mačevi', 'Pentakli']
def tname_(i):
    if i <= 22: return f"VA {i}"
    s_, r_ = divmod(i - 23, 14); return f"{SUITS[s_]} {['As','2','3','4','5','6','7','8','9','10','Paž','Vitez','Kraljica','Kralj'][r_]}"
sun_t = wp['Sunce'][0]; sign = int(sun_t // 30); dec = int((sun_t % 30) // 10)
suit = {0: 0, 1: 3, 2: 2, 3: 1}[sign % 4]; rank = [2, 5, 8][sign % 3] + dec
dcard = 23 + suit * 14 + (rank - 1)
drawn = cached('tarot', lambda: RNG.sample(range(1, 79), 10))
add('Tarot', [dcard] + drawn, f"Sunce {SIGNS[sign]} {sun_t%30:.2f}°, {dec+1}. dekan → {tname_(dcard)} ({dcard}); izvučeno: " + ', '.join(f"{tname_(c)} ({c})" for c in drawn))

def nred(n, master=True):
    while n > 9 and not (master and n in (11, 22)): n = sum(map(int, str(n)))
    return n
pdn = nred(sum(map(int, '3105')) + sum(map(int, '24112026')), master=False); udn = nred(sum(map(int, '24112026')))
add('Numerologija', [11, 29, 19, 13, 31, pdn, udn], f"osobni dan {pdn} (znamenke 31.05. + 24.11.2026.), univerzalni dan {udn}")

# Planetarni sat (Helsinki)
geo = (LON, LAT, 15.0)
def rt(j0, w): return swe.rise_trans(j0, swe.SUN, w, geo, 0, 0, swe.FLG_SWIEPH)[1][0]
jmid = swe.julday(2026, 11, 23, 22.0)   # 24.11. 00:00 EET
sr = rt(jmid, swe.CALC_RISE); ss = rt(jmid, swe.CALC_SET); sr2 = rt(ss, swe.CALC_RISE); assert sr < ss < jd < sr2
def lt(j):
    y, m, d, hh = swe.revjul(j); return (datetime(y, m, d, tzinfo=ZoneInfo('UTC')) + timedelta(seconds=round(hh * 3600))).astimezone(ZoneInfo('Europe/Helsinki'))
hl = (sr2 - ss) / 12; hidx = 12 + int((jd - ss) / hl) + 1; hst = ss + (hidx - 13) * hl; hen = hst + hl
CHAL = ['Saturn', 'Jupiter', 'Mars', 'Sunce', 'Venera', 'Merkur', 'Mjesec']
PNUM = {'Sunce': 1, 'Mjesec': 2, 'Jupiter': 3, 'Merkur': 5, 'Venera': 6, 'Saturn': 8, 'Mars': 9}
ruler = CHAL[(CHAL.index('Mars') + hidx - 1) % 7]
add('Planetarni sat', [PNUM[ruler], hidx, hidx - 12, 9, lt(sr).hour, lt(sr).minute, lt(ss).hour, lt(ss).minute],
    f"Helsinki: izlazak {lt(sr):%H:%M:%S}, zalazak {lt(ss):%H:%M:%S}, sutrašnji izlazak {lt(sr2):%H:%M:%S} (EET); 21:00 = {hidx}. sat ({hidx-12}. noćni, {lt(hst):%H:%M:%S}–{lt(hen):%H:%M:%S}), "
    f"vladar {ruler} (={PNUM[ruler]}); {(jd-hst)*1440:.1f} min od početka, {(hen-jd)*1440:.1f} min do kraja; 1. dnevni sat = Mars")
E['planetarni_sat'] = dict(vladar=ruler, od_pocetka=(jd - hst) * 1440, do_kraja=(hen - jd) * 1440)
for k in SYS: assert all(1 <= v <= 50 for v in SYS[k]) and all(1 <= v <= 12 for v in EUR[k]) and len(SYS[k]) == len(set(SYS[k]))

# Plečkov filtar
def plecko(names):
    vm = {v: [0, 0] for v in range(1, 51)}; ve = {v: [0, 0] for v in range(1, 13)}
    for k in names:
        for p, v in enumerate(SYS[k][:5], 1): vm[v][0] += 1; vm[v][1] += p
        for p, v in enumerate(EUR[k][:2], 1): ve[v][0] += 1; ve[v][1] += p
    om = sorted(vm, key=lambda v: (-vm[v][0], vm[v][1] if vm[v][0] else 0, v)); oe = sorted(ve, key=lambda v: (-ve[v][0], ve[v][1] if ve[v][0] else 0, v))
    return om, oe, vm, ve
PM_, PE_, VM, VE = plecko(list(SYS))
E['plecko_glavni'] = [(v, VM[v][0], VM[v][1]) for v in PM_ if VM[v][0]]; E['plecko_euro'] = [(v, VE[v][0], VE[v][1]) for v in PE_ if VE[v][0]]
E['signal_glavni'] = [v for v in PM_ if VM[v][0] >= 2]; E['signal_euro'] = [v for v in PE_ if VE[v][0] >= 2]

# ------------------------------------------------------------ §2 pravila
def adj_ok(c): return all(abs(a - b) != 1 for a, b in itertools.combinations(c, 2))
def share_ok(c, others): return all(len(set(c) & set(o)) <= 1 for o in others)
def build(order, others, fill):
    c, skipped = [], []
    for v in list(order) + [x for x in fill if x not in order]:
        if v in c: continue
        t = c + [v]
        if adj_ok(t) and share_ok(t, others): c.append(v)
        else: skipped.append(v)
        if len(c) == 5: break
    return sorted(c), skipped
FINAL = []   # (id, tip, sustav, mains, euros)
used_e = []
e1m, e1skip = build(PM_, [], PM_)
e1e = [v for v in PE_ if v not in used_e][:2]; used_e += e1e
FINAL.append(('E1', 'ezoterijska', 'Plečkov filtar (11 sustava)', e1m, sorted(e1e)))
EAST_WEST = ['Meihua', 'Zapadna astrologija', 'I Ching', 'Kamea Marsa', 'Jyotiš', 'Kabala']
inter = []
for i in range(max(len(SYS[k]) for k in EAST_WEST)):
    for k in EAST_WEST:
        if i < len(SYS[k]) and SYS[k][i] not in inter: inter.append(SYS[k][i])
e2m, e2skip = build(inter, [e1m], PM_)
_, pe2, _, ve2 = plecko(EAST_WEST)
e2e = [v for v in pe2 if v not in used_e][:2]; used_e += e2e
FINAL.append(('E2', 'ezoterijska', 'Istok/Zapad naizmjence: Meihua, zapadna, I Ching, kamea Marsa, Jyotiš, kabala', e2m, sorted(e2e)))
E['slaganje'] = dict(E1_preskoceno=e1skip, E2_niz=inter, E2_preskoceno=e2skip, E2_euro_glasovi=[(v, ve2[v][0]) for v in pe2 if ve2[v][0]])

# ------------------------------------------------------------ §4 statistika (walk-forward)
def freq(hist, n):
    f = {v: 0 for v in range(1, n + 1)}
    for s in hist:
        for v in s: f[v] += 1
    return f
def hot(hist, n): f = freq(hist, n); return sorted(f, key=lambda v: (-f[v], v))
def cold(hist, n): f = freq(hist, n); return sorted(f, key=lambda v: (f[v], v))
def overdue(hist, n):
    g = {}
    for v in range(1, n + 1):
        k = 0
        for s in reversed(hist):
            if v in s: break
            k += 1
        g[v] = k
    return sorted(g, key=lambda v: (-g[v], v))
def pareto(hist, n):
    w = hist[-max(1, round(0.2 * len(hist))):]; top = hot(w, n)[:max(1, round(0.2 * n))]
    fa = freq(hist, n); top = sorted(top, key=lambda v: (-fa[v], v))
    return top + [v for v in hot(hist, n) if v not in top]
def rnd(hist, n): p = list(range(1, n + 1)); RNG.shuffle(p); return p
METHODS = {'vrući (zadnjih 50)': lambda h_, n: hot(h_[-50:], n), 'vrući (zadnjih 100)': lambda h_, n: hot(h_[-100:], n),
           'hladni (cijela povijest)': cold, 'prezreli': overdue, 'Pareto 20/20': pareto, 'slučajna (secrets) – kontrola': rnd}
START = 99
nwf = N - START
pw = float(P_WIN)
Em = 0.5; Vm = float(sum(m * m * PM[m] for m in PM)) - 0.25
Ee = 1 / 3; Ve = float(sum(e * e * PE[e] for e in PE)) - 1 / 9
WF = {}
for name, fn in METHODS.items():
    hm, he, cls = [], [], {CL[c]: 0 for c in CLASSES}
    for t in range(START, N):
        cm = set(fn(DM[:t], 50)[:5]); ce = set(fn(DE[:t], 12)[:2])
        m, e = len(cm & DM[t]), len(ce & DE[t]); hm.append(m); he.append(e)
        if (m, e) in CL: cls[CL[(m, e)]] += 1
    W = sum(cls.values()); z = (W - nwf * pw) / math.sqrt(nwf * pw * (1 - pw)); p = float(stats.binom.sf(W - 1, nwf, pw))
    zm = (np.mean(hm) - Em) / math.sqrt(Vm / nwf); ze = (np.mean(he) - Ee) / math.sqrt(Ve / nwf)
    pm2 = 2 * (1 - stats.norm.cdf(abs(zm))); pe2_ = 2 * (1 - stats.norm.cdf(abs(ze)))
    WF[name] = dict(n=nwf, W=W, expW=nwf * pw, cls=cls, z=z, p=p, p_bonf=min(1, p * len(METHODS)), mean_m=float(np.mean(hm)), mean_e=float(np.mean(he)),
                    z_m=zm, z_e=ze, p_m_bonf=min(1, pm2 * len(METHODS)), p_e_bonf=min(1, pe2_ * len(METHODS)))
R['WF'] = WF
fm = np.array([freq(DM, 50)[v] for v in range(1, 51)]); fe = np.array([freq(DE, 12)[v] for v in range(1, 13)])
chi_m = float(((fm - N * 0.1) ** 2).sum() / (N * 0.1 * 0.9 * 50 / 49)); chi_e = float(((fe - N / 6) ** 2).sum() / (N * (1 / 6) * (5 / 6) * 12 / 11))
half = N // 2
fm1 = [freq(DM[:half], 50)[v] for v in range(1, 51)]; fm2 = [freq(DM[half:], 50)[v] for v in range(1, 51)]
fe1 = [freq(DE[:half], 12)[v] for v in range(1, 13)]; fe2 = [freq(DE[half:], 12)[v] for v in range(1, 13)]
rm, prm = stats.spearmanr(fm1, fm2); re_, pre_ = stats.spearmanr(fe1, fe2)
R['rand'] = dict(chi_m=chi_m, p_m=float(stats.chi2.sf(chi_m, 49)), chi_e=chi_e, p_e=float(stats.chi2.sf(chi_e, 11)), fm_min=int(fm.min()), fm_max=int(fm.max()),
                 fe_min=int(fe.min()), fe_max=int(fe.max()), E_m=N * 0.1, E_e=N / 6, rho_m=float(rm), p_rho_m=float(prm), rho_e=float(re_), p_rho_e=float(pre_))

# odabir S1–S2
cand = [m for m in METHODS if 'kontrola' not in m]
ranked = sorted(cand, key=lambda m: -WF[m]['z'])
sig = [m for m in cand if WF[m]['p_bonf'] < 0.05]
def typical(c): return 112.5 <= sum(c) <= 142.5 and sum(v % 2 for v in c) in (2, 3) and len({(v - 1) // 10 for v in c}) >= 4
SLOG = {}
for m in ranked:
    if sum(1 for f in FINAL if f[0][0] == 'S') == 2: break
    rkm = METHODS[m](DM, 50); rke = METHODS[m](DE, 12)
    others = [f[3] for f in FINAL]; best = None
    pos = {v: i for i, v in enumerate(rkm)}
    for sub in sorted(itertools.combinations(rkm[:20], 5), key=lambda s: (sum(pos[v] for v in s), sorted(pos[v] for v in s))):
        c = sorted(sub)
        if adj_ok(c) and share_ok(c, others) and (bool(sig) or typical(c)): best = c; break
    if best is None: continue
    ee = [v for v in rke if v not in used_e][:2]; used_e += ee
    sid_ = f"S{sum(1 for f in FINAL if f[0][0] == 'S') + 1}"
    SLOG[sid_] = dict(metoda=m, sirovo=sorted(rkm[:5]), sirovo_euro=sorted(rke[:2]))
    FINAL.append((sid_, 'statistička', m + ('' if sig else ' (statistički tipična)'), best, sorted(ee)))
R['S_rang'] = ranked; R['S_signif'] = sig; R['SLOG'] = SLOG

# ------------------------------------------------------------ §5 logičke (točno: 2.118.760 × 66)
idx = np.array(list(itertools.combinations(range(50), 5)), dtype=np.uint8)
ALLM = np.zeros(len(idx), dtype=np.uint64)
for j in range(5): ALLM |= (np.uint64(1) << idx[:, j].astype(np.uint64))
del idx
NM = len(ALLM); assert NM == comb(50, 5)
EPAIRS = list(itertools.combinations(range(1, 13), 2)); NE = len(EPAIRS); TOT = NM * NE
def bmask(c): return np.uint64(sum(1 << (v - 1) for v in c))
_pc = {}
def packs(c):
    key = tuple(c)
    if key not in _pc:
        m = np.bitwise_count(ALLM & bmask(c))
        _pc[key] = {t: np.packbits(m >= t).view(np.uint8) for t in (1, 2, 3)} | {'m': m}
        if len(_pc) > 4000: _pc.pop(next(iter(_pc)))
    return _pc[key]
def p_any(tickets):   # tickets: [(mains, euros)]
    P = [packs(c) for c, _ in tickets]; tot = 0
    for ep in EPAIRS:
        acc = None
        for (c, e), pk in zip(tickets, P):
            t = 3 - len(set(e) & set(ep)); a = pk[t]
            acc = a.copy() if acc is None else (acc | a)
        tot += int(np.bitwise_count(acc).sum())
    return tot
def p3(tickets):
    acc = None
    for c, _ in tickets:
        a = packs(c)[3]; acc = a.copy() if acc is None else (acc | a)
    return int(np.bitwise_count(acc).sum())
fixed = [(f[3], f[4]) for f in FINAL]
rem_e = sorted(set(range(1, 13)) - set(used_e)); assert len(rem_e) == 4
PAIRINGS = []
for a in itertools.combinations(rem_e, 2):
    b = tuple(sorted(set(rem_e) - set(a))); PAIRINGS.append((list(a), list(b)))   # 6 (uključuje zamjenu L1/L2)
def valid_L(c, others): return len(c) == 5 and len(set(c)) == 5 and adj_ok(c) and share_ok(c, others)
def rand_valid(others):
    while True:
        c = sorted(RNG.sample(range(1, 51), 5))
        if valid_L(c, others): return c
def greedy(ncand):
    best, bv = None, -1
    for pr in PAIRINGS:
        L1, v1 = None, -1
        for _ in range(ncand):
            c = rand_valid([f[0] for f in fixed])
            v = p_any(fixed + [(c, pr[0])])
            if v > v1: L1, v1 = c, v
        L2, v2 = None, -1
        for _ in range(ncand):
            c = rand_valid([f[0] for f in fixed] + [L1])
            v = p_any(fixed + [(L1, pr[0]), (c, pr[1])])
            if v > v2: L2, v2 = c, v
        if v2 > bv: best, bv = (L1, L2, pr), v2
    return best, bv
def local(st, tlim):
    (L1, L2, pr) = st; cur = p_any(fixed + [(L1, pr[0]), (L2, pr[1])]); improved = True
    while improved and time.time() < tlim:
        improved = False
        for j in (0, 1):
            Ls = [L1, L2]; oth = [f[0] for f in fixed] + [Ls[1 - j]]
            bestc, bestv = Ls[j], cur
            for o in Ls[j]:
                for n in range(1, 51):
                    if n in Ls[j]: continue
                    t = sorted([v for v in Ls[j] if v != o] + [n])
                    if not valid_L(t, oth): continue
                    T_ = [t, Ls[1]] if j == 0 else [Ls[0], t]
                    v = p_any(fixed + [(T_[0], pr[0]), (T_[1], pr[1])])
                    if v > bestv: bestc, bestv = t, v
            if bestv > cur:
                if j == 0: L1 = bestc
                else: L2 = bestc
                cur = bestv; improved = True
        for pr2 in PAIRINGS:
            v = p_any(fixed + [(L1, pr2[0]), (L2, pr2[1])])
            if v > cur: pr, cur, improved = pr2, v, True
    return (L1, L2, pr), cur
TL = time.time() + BUD; log = []; best, bestv = None, -1
first = True
while time.time() < TL or first:
    st, v0 = greedy(40); st, v = local(st, TL + (60 if first else 0)); log.append((v0, v)); first = False
    if v > bestv: best, bestv = st, v
L1, L2, pr = best
FINAL.append(('L1', 'logička', 'najveća točna P(barem jedan dobitak)', L1, pr[0]))
FINAL.append(('L2', 'logička', 'najveća točna P(barem jedan dobitak)', L2, pr[1]))
R['L_log'] = log
tickets = [(f[3], f[4]) for f in FINAL]
ids = [f[0] for f in FINAL]
# provjere §2
allm = [f[3] for f in FINAL]
R['provjere'] = dict(susjedni=[f[0] for f in FINAL if not adj_ok(f[3])], max_zajednickih=max(len(set(a) & set(b)) for a, b in itertools.combinations(allm, 2)),
                     parovi_zajednicki={f"{a[0]}-{b[0]}": len(set(a[3]) & set(b[3])) for a, b in itertools.combinations(FINAL, 2)},
                     euro_particija=sorted(sum([f[4] for f in FINAL], [])) == list(range(1, 13)), distinct=len({(tuple(c), tuple(e)) for c, e in tickets}) == 6,
                     raspon=[min(min(c) for c in allm), max(max(c) for c in allm)], brojeva_ukupno=len(set(sum(allm, []))))
PA = p_any(tickets)
R['set'] = dict(P_any=PA, TOT=TOT, P_52=f"6/{TOT}", P_3main=p3(tickets), NM=NM, E_win=str(6 * P_WIN), E_win_f=float(6 * P_WIN), fixed_only=p_any(fixed))
# redoslijed igranja
single = [p_any([t]) for t in tickets]
pairbest = [max(p_any([tickets[i], tickets[j]]) for j in range(6) if j != i) for i in range(6)]
order = [max(range(6), key=lambda i: (single[i], pairbest[i], -i))]
while len(order) < 6:
    order.append(max([i for i in range(6) if i not in order], key=lambda i: (p_any([tickets[x] for x in order] + [tickets[i]]), -i)))
seq = []
for k in range(1, 7):
    sub = order[:k]; v = p_any([tickets[x] for x in sub])
    bs = max(itertools.combinations(range(6), k), key=lambda s: p_any([tickets[x] for x in s])); bv = p_any([tickets[x] for x in bs])
    seq.append(dict(k=k, komb=[ids[x] for x in sub], p=v, p3=p3([tickets[x] for x in sub]), best=[ids[x] for x in bs], best_p=bv))
R['redoslijed'] = dict(order=[ids[x] for x in order], seq=seq, single=single, pairbest=pairbest)
# 1000 slučajnih setova (6 slučajnih listića, bez duplikata)
def rticket(): return (sorted(RNG.sample(range(1, 51), 5)), sorted(RNG.sample(range(1, 13), 2)))
RS = []
while len(RS) < 1000:
    s_ = [rticket() for _ in range(6)]
    if len({(tuple(a), tuple(b)) for a, b in s_}) == 6: RS.append(s_)
_pc.clear()
rsP, rs3 = [], []
for s_ in RS:
    rsP.append(p_any(s_)); rs3.append(p3(s_)); _pc.clear()
rsP = np.array(rsP); rs3 = np.array(rs3)
def pct(a, v): return float((a < v).mean() * 100 + (a == v).mean() * 50)
R['rand_sets'] = dict(mean=float(rsP.mean()), sd=float(rsP.std()), min=int(rsP.min()), max=int(rsP.max()), pct=pct(rsP, PA),
                      p3_mean=float(rs3.mean()), p3_pct=pct(rs3, R['set']['P_3main']))
# ------------------------------------------------------------ §6 backtest
def bt_set(ts):
    out = []
    for c, e in ts:
        cls = {CL[x]: 0 for x in CLASSES}
        for dm, de in zip(DM, DE):
            m, ee = len(set(c) & dm), len(set(e) & de)
            if (m, ee) in CL: cls[CL[(m, ee)]] += 1
        out.append(cls)
    return out
BT = bt_set(tickets); setW = sum(sum(c.values()) for c in BT)
rsW = np.array([sum(sum(c.values()) for c in bt_set(s_)) for s_ in RS])
R['backtest'] = dict(per={i: c for i, c in zip(ids, BT)}, setW=setW, expW=6 * N * pw, rs_mean=float(rsW.mean()), rs_sd=float(rsW.std()), pct=pct(rsW, setW),
                     draws_any=sum(1 for dm, de in zip(DM, DE) if any((len(set(c) & dm), len(set(e) & de)) in CL for c, e in tickets)))
# ------------------------------------------------------------ izlaz
rank_ = {cid: i + 1 for i, cid in enumerate(R['redoslijed']['order'])}
R['FINAL'] = [dict(id=f[0], tip=f[1], sustav=f[2], glavni=f[3], euro=f[4], zbroj=sum(f[3]), neparni=sum(v % 2 for v in f[3]), redoslijed=rank_[f[0]]) for f in FINAL]
R['SYS'] = SYS; R['EUR'] = EUR; R['KEY'] = KEY; R['E'] = E
R['versions'] = {p: md.version(p) for p in ['numpy', 'pandas', 'scipy', 'lunar_python', 'sxtwl']}
R['versions']['pyswisseph'] = swe.version; R['versions']['python'] = sys.version.split()[0]
R['t'] = time.time() - T0
with open(f'{OUT}/ej_2026-11-24.csv', 'w', encoding='utf-8') as fh:
    fh.write('#;tip;sustav;redoslijed;M1;M2;M3;M4;M5;E1;E2\n')
    for f in FINAL: fh.write(f"{f[0]};{f[1]};{f[2]};{rank_[f[0]]};" + ';'.join(map(str, f[3] + f[4])) + '\n')
json.dump(R, open(f'{OUT}/rezultati_ej_2026-11-24.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
print(json.dumps({k: R[k] for k in ['FINAL', 'provjere', 'set', 'rand_sets', 'redoslijed', 'backtest', 'L_log', 't']}, ensure_ascii=False, default=str))
