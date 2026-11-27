"""Deterministički i slučajni ezoterijski sustavi za Eurojackpot (definicije prema protocol.md)."""
import datetime as dt
import swisseph as swe
from lunar_python import Solar

LAT, LON = 60.17, 24.94
FL = swe.FLG_MOSEPH
NAK = 360 / 27

# ---------- svođenje ----------
def svedi_g(x):
    while x > 50:
        x -= 50
    return x if x >= 1 else None

def svedi_e(x):
    while x > 12:
        x -= 12
    return x if x >= 1 else None

def popis(raw, euro_raw=()):
    g, e = [], []
    for x in raw:
        y = svedi_g(x)
        if y and y not in g:
            g.append(y)
    for x in list(euro_raw) + list(raw):
        y = svedi_e(x)
        if y and y not in e:
            e.append(y)
    return g, e

def listic(g, e):
    g5 = list(g[:5]); e2 = list(e[:2]); i = 1
    while len(g5) < 5:
        if i not in g5: g5.append(i)
        i += 1
    i = 1
    while len(e2) < 2:
        if i not in e2: e2.append(i)
        i += 1
    return g5, e2

def svedi_znam(n, master=True):
    while n > 9 and not (master and n in (11, 22)):
        n = sum(int(c) for c in str(n))
    return n

# ---------- I Ching ----------
TRI = {1: (1, 1, 1), 2: (1, 1, 0), 3: (1, 0, 1), 4: (1, 0, 0), 5: (0, 1, 1), 6: (0, 1, 0), 7: (0, 0, 1), 8: (0, 0, 0)}  # odozdo
TRI_IME = {1: '☰ Qian', 2: '☱ Dui', 3: '☲ Li', 4: '☳ Zhen', 5: '☴ Xun', 6: '☵ Kan', 7: '☶ Gen', 8: '☷ Kun'}
_ord = [1, 4, 6, 7, 8, 5, 3, 2]  # Qian Zhen Kan Gen Kun Xun Li Dui
_kw = [[1, 25, 6, 33, 12, 44, 13, 10], [34, 51, 40, 62, 16, 32, 55, 54], [5, 3, 29, 39, 8, 48, 63, 60],
       [26, 27, 4, 52, 23, 18, 22, 41], [11, 24, 7, 15, 2, 46, 36, 19], [9, 42, 59, 53, 20, 57, 37, 61],
       [14, 21, 64, 56, 35, 50, 30, 38], [43, 17, 47, 31, 45, 28, 49, 58]]
KW = {}
for i, up in enumerate(_ord):
    for j, lo in enumerate(_ord):
        KW[TRI[lo] + TRI[up]] = _kw[i][j]
assert sorted(KW.values()) == list(range(1, 65))
TRI_INV = {v: k for k, v in TRI.items()}

def kw(lines):
    return KW[tuple(lines)]

def nuklearni(lines):
    return kw(tuple(lines[1:4]) + tuple(lines[2:5]))

def meihua(d):
    l = Solar.fromYmd(d.year, d.month, d.day).getLunar()
    Y = l.getYearZhiIndex() + 1
    M, D, H = abs(l.getMonth()), l.getDay(), 12
    up = (Y + M + D) % 8 or 8
    lo = (Y + M + D + H) % 8 or 8
    crta = (Y + M + D + H) % 6 or 6
    lines = list(TRI[lo] + TRI[up])
    osn = kw(lines)
    ch = lines.copy(); ch[crta - 1] ^= 1
    prom = kw(ch)
    nuk = nuklearni(lines)
    info = dict(lunarni=f"{l.getYearInGanZhi()} god., {M}. mj., {D}. dan", Y=Y, M=M, D=D, H=H, gornji=f"{up} {TRI_IME[up]}",
                donji=f"{lo} {TRI_IME[lo]}", crta=crta, osnovni=osn, promijenjeni=prom, nuklearni=nuk)
    return [osn, prom, nuk, crta, up, lo], info

def iching_bacanja(bacanja):
    """bacanja: 6 trojki novčića (2 = pismo/yin, 3 = glava/yang), odozdo."""
    vals = [sum(t) for t in bacanja]  # 6 stari yin, 7 mladi yang, 8 mladi yin, 9 stari yang
    lines = [1 if v in (7, 9) else 0 for v in vals]
    ch = [(1 - x) if v in (6, 9) else x for x, v in zip(lines, vals)]
    pom = [i + 1 for i, v in enumerate(vals) if v in (6, 9)]
    osn, prom, nuk = kw(lines), kw(ch), nuklearni(lines)
    info = dict(vrijednosti=vals, osnovni=osn, promijenjeni=prom, nuklearni=nuk, pomicne_crte=pom,
                gornji=TRI_IME[TRI_INV[tuple(lines[3:])]], donji=TRI_IME[TRI_INV[tuple(lines[:3])]])
    return [osn, prom, nuk] + pom + [TRI_INV[tuple(lines[3:])], TRI_INV[tuple(lines[:3])]], info

# ---------- astronomija ----------
def jd_ut(d, h=19.0):
    return swe.julday(d.year, d.month, d.day, h)

def lon_trop(jd, body):
    return swe.calc_ut(jd, body, FL)[0][0]

def lon_sid(jd, body):
    swe.set_sid_mode(swe.SIDM_LAHIRI)
    return swe.calc_ut(jd, body, FL | swe.FLG_SIDEREAL)[0][0]

def jyotish(d):
    jd = jd_ut(d)
    m, s, r = lon_sid(jd, swe.MOON), lon_sid(jd, swe.SUN), lon_sid(jd, swe.MEAN_NODE)
    nak = int(m // NAK) + 1
    tithi = int(((m - s) % 360) // 12) + 1
    yoga = int(((m + s) % 360) // NAK) + 1
    rahu = int(r // NAK) + 1
    vd = (d.isoweekday() % 7) + 1  # Ned 1 ... Sub 7
    info = dict(mjesec_sid=round(m, 3), sunce_sid=round(s, 3), rahu_sid=round(r, 3), nakshatra=nak, tithi=tithi,
                yoga=yoga, rahu_nakshatra=rahu, vladar_dana=vd, ayanamsa=round(swe.get_ayanamsa_ut(jd), 4))
    return [20, nak, tithi, vd, yoga, rahu], info

ZNAK = ['Ovan', 'Bik', 'Blizanci', 'Rak', 'Lav', 'Djevica', 'Vaga', 'Škorpion', 'Strijelac', 'Jarac', 'Vodenjak', 'Ribe']

def zapadna(d):
    jd = jd_ut(d)
    tijela = [('Venera', swe.VENUS), ('Jupiter', swe.JUPITER), ('Merkur', swe.MERCURY), ('Sunce', swe.SUN),
              ('Mjesec', swe.MOON), ('Mars', swe.MARS)]
    raw, info = [], {}
    for ime, b in tijela:
        L = lon_trop(jd, b)
        st = int(L % 30)
        raw.append(st + 1)
        info[ime] = f"{L % 30:.2f}° {ZNAK[int(L // 30)]} (lon {L:.3f}) → {st + 1}"
    return raw + [10, 15], info

VENERA = [[22, 47, 16, 41, 10, 35, 4], [5, 23, 48, 17, 42, 11, 29], [30, 6, 24, 49, 18, 36, 12], [13, 31, 7, 25, 43, 19, 37],
          [38, 14, 32, 1, 26, 44, 20], [21, 39, 8, 33, 2, 27, 45], [46, 15, 40, 9, 34, 3, 28]]
MARS = [[11, 24, 7, 20, 3], [4, 12, 25, 8, 16], [17, 5, 13, 21, 9], [10, 18, 1, 14, 22], [23, 6, 19, 2, 15]]

def provjeri_kameu(K):
    n = len(K); c = n * (n * n + 1) // 2
    assert sorted(x for r in K for x in r) == list(range(1, n * n + 1))
    assert all(sum(r) == c for r in K)
    assert all(sum(K[i][j] for i in range(n)) == c for j in range(n))
    assert sum(K[i][i] for i in range(n)) == c and sum(K[i][n - 1 - i] for i in range(n)) == c
    return c

def kamea(d):
    K = VENERA if d.weekday() == 4 else MARS
    n = len(K); c = provjeri_kameu(K)
    dg = [K[i][i] for i in range(n)]; sp = [K[i][n - 1 - i] for i in range(n)]
    return dg + sp + [n, c], dict(planet='Venera' if n == 7 else 'Mars', dijagonala=dg, sporedna=sp, red=n, konstanta=c)

def kabala(d):
    pet = d.weekday() == 4
    return ([58, 7] if pet else [95, 5]) + [67, 77, 51, 26, 18, 10], dict(dan='Nogah 58, Necah 7' if pet else "Ma'adim 95, Gevura 5")

def znam(*nums):
    return sum(int(c) for n in nums for c in str(n))

def numerologija(d):
    uni = svedi_znam(znam(d.day, d.month, d.year))
    osob = svedi_znam(znam(31, 5) + znam(d.day, d.month, d.year))
    return [11, 29, 19, 13, 31, osob, uni], dict(osobni_dan=osob, univerzalni_dan=uni,
                                                  zbroj_osobni=znam(31, 5) + znam(d.day, d.month, d.year))

KALD = ['Saturn', 'Jupiter', 'Mars', 'Sunce', 'Venera', 'Merkur', 'Mjesec']
PL_BR = {'Sunce': 1, 'Mjesec': 2, 'Mars': 3, 'Merkur': 4, 'Jupiter': 5, 'Venera': 6, 'Saturn': 7}
PL_KAM = {'Saturn': (3, 15), 'Jupiter': (4, 34), 'Mars': (5, 65), 'Sunce': (6, 111), 'Venera': (7, 175), 'Merkur': (8, 260), 'Mjesec': (9, 369)}
DAN_VL = ['Mjesec', 'Mars', 'Merkur', 'Jupiter', 'Venera', 'Saturn', 'Sunce']  # pon..ned

def _rise_set(jd0, flag):
    return swe.rise_trans(jd0, swe.SUN, flag, (LON, LAT, 0.0), 0.0, 0.0, FL)[1][0]

def planetarni_sat(d, h=19.0):
    jd = jd_ut(d, h)
    jd0 = swe.julday(d.year, d.month, d.day, 0.0) - 2 / 24  # ponoć po lokalnom vremenu (približno)
    izl = _rise_set(jd0, swe.CALC_RISE)
    zal = _rise_set(izl, swe.CALC_SET)
    izl2 = _rise_set(zal, swe.CALC_RISE)
    vl = DAN_VL[d.weekday()]
    start = KALD.index(vl)
    if jd < izl:
        raise ValueError('prije izlaska')
    if jd < zal:
        dur = (zal - izl) / 12; n = int((jd - izl) // dur); poc = izl + n * dur; idx = n
        noc = False
    else:
        dur = (izl2 - zal) / 12; n = int((jd - zal) // dur); poc = zal + n * dur; idx = 12 + n
        noc = True
    pl = KALD[(start + idx) % 7]
    kraj = poc + dur
    info = dict(izlazak_UT=swe.revjul(izl), zalazak_UT=swe.revjul(zal), izlazak_sutra_UT=swe.revjul(izl2), vladar_dana=vl,
                sat_od_izlaska=idx + 1, sat_u_noci=(n + 1) if noc else None, sat_u_danu=None if noc else n + 1,
                planet=pl, trajanje_min=round(dur * 1440, 1), od_pocetka_min=round((jd - poc) * 1440, 1),
                do_kraja_min=round((kraj - jd) * 1440, 1))
    return [PL_BR[pl], PL_KAM[pl][0], PL_KAM[pl][1], idx + 1, n + 1], info

def sinkronicitet(d, prev5, hl_kolo):
    """prev5: lista (datum, glavni, euro) – 5 kola prije, najnovije prvo."""
    from collections import Counter
    cnt = Counter(x for _, g, _ in prev5 for x in g)
    zadnji = {}
    for i, (_, g, _) in enumerate(prev5):
        for x in g:
            zadnji.setdefault(x, i)
    rep = sorted([x for x, c in cnt.items() if c >= 2], key=lambda x: (-cnt[x], zadnji[x], x))
    ce = Counter(x for _, _, e in prev5 for x in e)
    zad_e = {}
    for i, (_, _, e) in enumerate(prev5):
        for x in e:
            zad_e.setdefault(x, i)
    rep_e = sorted([x for x, c in ce.items() if c >= 2], key=lambda x: (-ce[x], zad_e[x], x))
    god = svedi_znam(znam(d.year), master=False)
    doy = d.timetuple().tm_yday
    rodj = dt.date(1991, 5, 31)
    dob = d.year - rodj.year - ((d.month, d.day) < (rodj.month, rodj.day))
    raw = rep + [d.day, d.month, god, hl_kolo, doy, dob, 31, 5, 18, 20] + sorted(prev5[0][1])
    info = dict(ponovljeni_glavni={x: cnt[x] for x in rep}, ponovljeni_euro={x: ce[x] for x in rep_e},
                dan=d.day, mjesec=d.month, godina=god, hl_kolo=hl_kolo, dan_u_godini=doy, dob=dob)
    return raw, rep_e, info

ARHETIPOVI = [4, 12, 1, 16, 7, 10, 28, 22, 40, 36]

# Tarot (Golden Dawn dekani): znak -> (boja, karte 2–4/5–7/8–10)
BOJA_POC = {'Štapovi': 23, 'Pehari': 37, 'Mačevi': 51, 'Pentakli': 65}
DEKAN = {0: ('Štapovi', 2), 1: ('Pentakli', 5), 2: ('Mačevi', 8), 3: ('Pehari', 2), 4: ('Štapovi', 5), 5: ('Pentakli', 8),
         6: ('Mačevi', 2), 7: ('Pehari', 5), 8: ('Štapovi', 8), 9: ('Pentakli', 2), 10: ('Mačevi', 5), 11: ('Pehari', 8)}

def tarot_dekan(d):
    L = lon_trop(jd_ut(d), swe.SUN)
    z = int(L // 30); dek = int((L % 30) // 10)
    boja, prva = DEKAN[z]
    karta = prva + dek
    return BOJA_POC[boja] + karta - 1, f"Sunce {L % 30:.2f}° {ZNAK[z]}, {dek + 1}. dekan → {karta} {boja}"
