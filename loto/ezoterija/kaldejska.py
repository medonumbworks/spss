"""Kaldejska numerologija za Loto 6/45 — izvlačenje četvrtak 29.10.2026. u 20:00 CET, Zagreb.

ŠTO JE METODA
-------------
Kaldejska (babilonska, "Cheirova") numerologija dodjeljuje slovima vrijednosti 1–8 po
zvuku, a ne po položaju u abecedi; broj 9 smatra se svetim i ne dodjeljuje se slovu.
Za razliku od pitagorejske, kaldejska škola ne stane na jednoznamenkastom korijenu nego
ponajprije tumači SLOŽENI broj (10–52, Cheiro "Book of Numbers", 1926): npr. 10 "kotač
sreće", 23 "kraljevska zvijezda lava", 27 "žezlo", 29 "milost pod pritiskom / iskušenja",
26 "upozorenje", 20 "buđenje", 22 "dobar čovjek u zabludi", 11 "skrivene opasnosti".
Jednoznamenkasti korijen vezan je uz planet: 1 Sunce, 2 Mjesec, 3 Jupiter, 4 Uran/Rahu,
5 Merkur, 6 Venera, 7 Neptun/Ketu, 8 Saturn, 9 Mars.

PRAVILA (izvori: Cheiro "Book of Numbers"; Goodman "Star Signs" — kaldejska tablica)
  A=1 B=2 C=3 D=4 E=5 U=6 O=7 F=8
  I=1 K=2 G=3 M=4 H=5 V=6 Z=7 P=8
  J=1 R=2 L=3 T=4 N=5 W=6
  Q=1 S=3 X=5 Y=1            (9 se ne dodjeljuje)
  Č,Ć→C  Š→S  Ž→Z  Đ→D       (hrvatska slova po osnovnom glasu; LJ, NJ, DŽ = dva slova)
  Zbroj slova = UKUPNA vrijednost; ako je > 52 zbrajaju se znamenke dok ne padne na ≤ 52
  (= SLOŽENI broj); korijen = zbrajanje znamenki do 1..9.

POSTUPAK ZA 29.10.2026. 20:00 (četvrtak, sat Jupitera, Zagreb) — sažeto; cijeli izvod
generira funkcija postupak_tekst() i zapisuje se u out/ezo_kaldejska.json.

1. Riječi dana, mjesta, planeta i igre
   ČETVRTAK  = C3+E5+T4+V6+R2+T4+A1+K2             = 27  → 9   (27 ŽEZLO — autoritet, nagrada)
   JUPITER   = J1+U6+P8+I1+T4+E5+R2                = 27  → 9   (isto 27! dan I planetarni sat)
   ZAGREB    = Z7+A1+G3+R2+E5+B2                   = 20  → 2   (20 BUĐENJE)
   HRVATSKA  = H5+R2+V6+A1+T4+S3+K2+A1             = 24  → 6   (24 LJUBAV/NOVAC)
   LUTRIJA   = L3+U6+T4+R2+I1+J1+A1                = 18  → 9
   HRVATSKA LUTRIJA                                = 42  → 6   (42 = titraj 24)
   LOTO      = L3+O7+T4+O7                          = 21  → 3   (21 KRUNA MAGA; korijen 3 = Jupiter)
   ŠEST      = S3+E5+S3+T4                          = 15  → 6   (15 ČAROBNJAK)
   OD        = O7+D4                                = 11  → 2   (11 SKRIVENE OPASNOSTI)
   ČETRDESET = C3+E5+T4+R2+D4+E5+S3+E5+T4          = 35  → 8   (35 = titraj 26, upozorenje)
   PET       = P8+E5+T4                             = 17  → 8   (17 ZVIJEZDA MAGA)
   LOTO ŠEST OD ČETRDESET PET = 21+15+11+35+17      = 99  → 18 → 9
2. Datum riječima
   DVADESET=32 (→5)  DEVETI=25 (→7)  LISTOPAD=31 (→4)  DVIJE=17 (→8)  TISUĆE=22 (→4)
   DVADESET=32       ŠESTE=20 (→2)
   DVADESET DEVETI LISTOPAD DVIJE TISUĆE DVADESET ŠESTE = 179 → 17 → 8  (17 ZVIJEZDA MAGA)
3. Znamenke datuma i sata
   dan 29 (→11→2)   mjesec 10 (→1)   godina 2026 → 2+0+2+6 = 10 (→1)   '26 → 26 (→8)
   sat 20:00 → 20 (→2; pravo sunčevo vrijeme 20:20 također 20)
   fadik datuma 2+9+1+0+2+0+2+6 = 22 (→4);  2+9 = 11 (→2);  kolo 87 → 8+7 = 15 (→6)
4. Planetarni titraji
   četvrtak = Jupiter = 3; 20:00 je u 3. noćnom satu = sat JUPITERA → dvostruki Jupiter:
   broj 3 i svi brojevi s korijenom 3 (12, 21, 30, 39) dobivaju bonus.
   Korijen datuma 29→2 = Mjesec (82 % osvijetljen, opadajući): korijen-2 brojevi (2, 11, 20, 29, 38) mali bonus.
5. Izvedeni brojevi: zbrojevi i razlike parova PRIMARNIH brojeva (datum + složeni brojevi
   riječi) unutar 1..45, npr. 29+10=39, 29−10=19, 26+11=37, 22+11=33, 20+22=42, 29−26=3.
6. Bazen: svaki broj 1..45 zbraja težine svih svojih izvora (primarni datum 2–12, složeni
   broj riječi 3–8, korijen riječi 1, Jupiterov titraj 3–6, Mjesečev titraj 1.5, kaldejski
   "sretni" složeni +2, izvedeni zbroj/razlika 0.10·(w_a+w_b) — ukupno najviše 6 po broju,
   najjači parovi prvi).
7. Kandidati: (a) 10 KANONSKIH skupova — izravne derivacije (čisti datum, vrh bazena,
   riječi dana, ime igre, datum riječima, Jupiter, korijeni, sretni složeni, zbrojevi,
   razlike); (b) zatim 6-člani podskupovi 28 najtežih brojeva bazena rangirani po ukupnoj
   težini uz uvjet raznolikosti (najviše 3 zajednička broja s već prihvaćenim skupom i
   blagi penal za već iskorištene brojeve). Sve je deterministično — metoda nema bacanja
   ni miješanja, pa se ne koristi slučajnost.
"""
from __future__ import annotations

import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

SLUG = "kaldejska"
METODA = "Kaldejska numerologija"
DATUM_TEKST = "četvrtak 29.10.2026. u 20:00 CET (sat Jupitera), Zagreb"
OUT_DIR = Path(__file__).resolve().parent.parent / "out"

# --------------------------------------------------------------------------- tablica slova
_KALDEJSKA = {
    1: "AIJQY", 2: "BKR", 3: "CGLS", 4: "DMT", 5: "EHNX", 6: "UVW", 7: "OZ", 8: "FP",
}
SLOVO = {s: v for v, slova in _KALDEJSKA.items() for s in slova}
_HR = {"Č": "C", "Ć": "C", "Š": "S", "Ž": "Z", "Đ": "D"}

PLANET = {1: "Sunce", 2: "Mjesec", 3: "Jupiter", 4: "Uran/Rahu", 5: "Merkur",
          6: "Venera", 7: "Neptun/Ketu", 8: "Saturn", 9: "Mars"}

# Cheirova značenja složenih brojeva 10–52 (kratko, hrvatski); 1–9 = planet
ZNACENJE = {
    1: "Sunce — početak, volja", 2: "Mjesec — osjećaj, promjenjivost", 3: "Jupiter — širenje, sreća",
    4: "Uran/Rahu — neočekivano", 5: "Merkur — pokret, trgovina", 6: "Venera — sklad, dobit",
    7: "Neptun/Ketu — mistika", 8: "Saturn — karma, ustrajnost", 9: "Mars — snaga, dovršetak",
    10: "kotač sreće", 11: "skrivene opasnosti", 12: "žrtva", 13: "preobrazba, moć",
    14: "kretanje i promjena", 15: "čarobnjak — darovi, rječitost", 16: "srušena kula",
    17: "zvijezda maga", 18: "materijalni sukob", 19: "princ neba — sreća", 20: "buđenje",
    21: "kruna maga", 22: "dobar čovjek u zabludi", 23: "kraljevska zvijezda lava",
    24: "ljubav/novac", 25: "snaga iz iskustva", 26: "upozorenje", 27: "žezlo",
    28: "proturječja", 29: "milost pod pritiskom", 30: "promišljanje", 31: "samotni genij",
    32: "komunikacija", 33: "sreća", 34: "snaga iz iskustva (=25)", 35: "upozorenje (=26)",
    36: "žezlo (=27)", 37: "ljubav/partnerstvo", 38: "milost pod pritiskom (=29)",
    39: "promišljanje (=30)", 40: "samotni genij (=31)", 41: "komunikacija (=32)",
    42: "ljubav/novac (=24)", 43: "prevrat", 44: "upozorenje (=26)", 45: "žezlo (=27)",
    46: "partnerstvo (=37)", 47: "milost pod pritiskom (=29)", 48: "promišljanje (=30)",
    49: "samotni genij (=31)", 50: "komunikacija (=32)", 51: "ratnik", 52: "prevrat (=43)",
}
SRETNI_SLOZENI = {10, 15, 17, 19, 21, 23, 24, 27, 32, 33, 37, 41, 42, 45}


def normiraj(rijec: str) -> str:
    """Velika slova, hrvatski dijakritici na osnovni glas, ostalo (razmaci, crtice) se izbacuje."""
    r = rijec.upper()
    for a, b in _HR.items():
        r = r.replace(a, b)
    return "".join(c for c in r if c in SLOVO)


def zbroj_znamenki(n: int) -> int:
    return sum(int(c) for c in str(n))


def slozeni(n: int) -> int:
    """Kaldejski složeni broj: zbrajaj znamenke dok ne padne na ≤ 52."""
    while n > 52:
        n = zbroj_znamenki(n)
    return n


def korijen(n: int) -> int:
    while n > 9:
        n = zbroj_znamenki(n)
    return n


def vrijednost_rijeci(rijec: str) -> dict:
    n = normiraj(rijec)
    dijelovi = [(c, SLOVO[c]) for c in n]
    ukupno = sum(v for _, v in dijelovi)
    return {"rijec": rijec, "slova": dijelovi, "ukupno": ukupno,
            "slozeni": slozeni(ukupno), "korijen": korijen(ukupno)}


# ----------------------------------------------------------------------------- ulazi datuma
RIJECI_DANA = [  # (riječ, težina složenog broja, opis)
    ("ČETVRTAK", 8.0, "dan izvlačenja"),
    ("JUPITER", 8.0, "planet dana i planetarnog sata (20:00 = 3. noćni sat)"),
    ("ZAGREB", 6.0, "mjesto"),
    ("HRVATSKA", 6.0, "priređivač"),
    ("LUTRIJA", 6.0, "priređivač"),
    ("HRVATSKA LUTRIJA", 5.0, "priređivač, cijeli naziv"),
]
RIJECI_IGRE = [
    ("LOTO", 7.0, "ime igre"),
    ("ŠEST", 5.0, "ime igre"),
    ("OD", 3.0, "ime igre"),
    ("ČETRDESET", 5.0, "ime igre"),
    ("PET", 5.0, "ime igre"),
    ("LOTO ŠEST OD ČETRDESET PET", 5.0, "ime igre, cijela fraza"),
]
RIJECI_DATUMA = [
    ("DVADESET", 4.0, "datum riječima"),
    ("DEVETI", 4.0, "datum riječima"),
    ("LISTOPAD", 4.0, "datum riječima"),
    ("DVIJE", 4.0, "datum riječima"),
    ("TISUĆE", 4.0, "datum riječima"),
    ("DVADESET", 4.0, "datum riječima (2. put)"),
    ("ŠESTE", 4.0, "datum riječima"),
    ("DVADESET DEVETI LISTOPAD DVIJE TISUĆE DVADESET ŠESTE", 5.0, "datum, cijela fraza"),
]
SVE_RIJECI = RIJECI_DANA + RIJECI_IGRE + RIJECI_DATUMA

# (broj, težina, oznaka, opis) — složeni brojevi znamenaka datuma
DATUM_BROJEVI = [
    (29, 12.0, "dan", "dan u mjesecu — kaldejski 'broj rođenja' događaja"),
    (10, 8.0, "mjesec", "listopad = 10"),
    (10, 4.0, "2026", "godina 2+0+2+6 = 10"),
    (20, 7.0, "sat", "20:00 (pravo sunčevo vrijeme 20:20)"),
    (26, 7.0, "'26", "zadnje dvije znamenke godine"),
    (11, 8.0, "2+9", "dan 29 → 2+9"),
    (22, 10.0, "fadik", "2+9+1+0+2+0+2+6 = 22"),
    (2, 5.0, "29→2", "korijen dana (11 → 2)"),
    (4, 5.0, "22→4", "korijen fadika"),
    (1, 4.0, "10→1", "korijen mjeseca i godine"),
    (15, 4.0, "kolo 87", "87. kolo → 8+7 = 15"),
    (6, 2.0, "87→6", "korijen kola (15 → 6)"),
]
JUPITER = 3           # četvrtak = Jupiter, planetarni sat = Jupiter
MJESEC = 2            # korijen datuma 29 → 2 = Mjesec; Mjesec 82,6 % opadajući
W_JUPITER_BROJ, W_JUPITER_TITRAJ, W_MJESEC_TITRAJ = 6.0, 3.0, 1.5
W_KORIJEN_RIJECI, W_SRETNI, F_IZVEDENI, CAP_IZVEDENI = 1.0, 2.0, 0.10, 6.0
KRATICA = {"HRVATSKA LUTRIJA": "HRV.LUTRIJA", "LOTO ŠEST OD ČETRDESET PET": "IGRA-FRAZA",
           "DVADESET DEVETI LISTOPAD DVIJE TISUĆE DVADESET ŠESTE": "DATUM-FRAZA"}


# ----------------------------------------------------------------------------- bazen
def izgradi_bazen() -> tuple[dict[int, float], dict[int, list[tuple[str, float]]], list]:
    """Vraća (težina po broju, izvori po broju, izvedeni parovi)."""
    tezina: dict[int, float] = defaultdict(float)
    izvori: dict[int, list[tuple[str, float]]] = defaultdict(list)

    def dodaj(n: int, w: float, oznaka: str):
        if 1 <= n <= 45 and w > 0:
            tezina[n] += w
            izvori[n].append((oznaka, w))

    primarni: list[tuple[int, float, str]] = []
    for n, w, ozn, _ in DATUM_BROJEVI:
        dodaj(n, w, ozn)
        if w >= 4:
            primarni.append((n, w, ozn))
    for rijec, w, _ in SVE_RIJECI:
        v = vrijednost_rijeci(rijec)
        kratko = KRATICA.get(rijec, rijec)
        dodaj(v["slozeni"], w, kratko)
        dodaj(v["korijen"], W_KORIJEN_RIJECI, f"{kratko}→{v['korijen']}")
        if w >= 5:
            primarni.append((v["slozeni"], w, kratko))
    # planetarni titraji
    dodaj(JUPITER, W_JUPITER_BROJ, "Jupiter=3")
    for n in range(1, 46):
        if n != JUPITER and korijen(n) == JUPITER:
            dodaj(n, W_JUPITER_TITRAJ, "titraj 3")
        if korijen(n) == MJESEC:
            dodaj(n, W_MJESEC_TITRAJ, "titraj 2")
    for n in SRETNI_SLOZENI:
        dodaj(n, W_SRETNI, "sretni")
    # izvedeni: zbrojevi i razlike parova primarnih (različitih) brojeva;
    # ukupni doprinos izvedenih po broju ograničen je na CAP_IZVEDENI (najjači parovi prvi)
    hitovi = []
    vidjeni = set()
    for (a, wa, oa), (b, wb, ob) in itertools.combinations(primarni, 2):
        if a == b:
            continue
        for n, znak in ((a + b, "+"), (abs(a - b), "−")):
            if 1 <= n <= 45:
                hi, lo = max(a, b), min(a, b)
                kljuc = (hi, lo, znak)
                if kljuc in vidjeni:
                    continue
                vidjeni.add(kljuc)
                opis = f"{a}{znak}{b}" if znak == "+" else f"{hi}−{lo}"
                hitovi.append((n, round(F_IZVEDENI * (wa + wb), 3), opis, oa, ob))
    izvedeni = []
    izv_zbroj: dict[int, float] = defaultdict(float)
    for n, w, opis, oa, ob in sorted(hitovi, key=lambda t: (-t[1], t[0], t[2])):
        w_eff = round(min(w, CAP_IZVEDENI - izv_zbroj[n]), 3)
        if w_eff <= 0:
            continue
        izv_zbroj[n] += w_eff
        dodaj(n, w_eff, opis)
        izvedeni.append((n, w_eff, opis, oa, ob))
    return dict(tezina), dict(izvori), izvedeni


def oznaka_broja(n: int, izvori: dict, k: int = 1) -> str:
    """Najjače oznake izvora za broj (za obrazloženje)."""
    lst = sorted(izvori.get(n, []), key=lambda t: -t[1])
    return f"{n}={'/'.join(o for o, _ in lst[:k])}" if lst else f"{n}"


# ----------------------------------------------------------------------------- kanonski skupovi
def kanonski(tezina: dict[int, float]) -> list[tuple[list[int], str]]:
    """Izravne derivacije — rang 1..10, od najjače prema slabijoj."""
    r = {w: vrijednost_rijeci(w) for w, _, _ in SVE_RIJECI}
    s = lambda w: r[w]["slozeni"]  # noqa: E731
    vrh = [n for n, _ in sorted(tezina.items(), key=lambda t: (-t[1], t[0]))[:6]]
    skupovi = [
        ([29, 10, 20, 26, 11, 22],
         "Čisti datum i sat: dan 29, mjesec 10, sat 20, godina '26, 2+9=11, fadik 22"),
        (vrh, "Šest najtežih brojeva bazena (sve derivacije zajedno)"),
        ([s("ČETVRTAK"), s("ZAGREB"), s("HRVATSKA"), s("LUTRIJA"), s("LOTO"), s("HRVATSKA LUTRIJA")],
         "Riječi dana i mjesta: ČETVRTAK=JUPITER 27, ZAGREB 20, HRVATSKA 24, LUTRIJA 18, LOTO 21, HRVATSKA LUTRIJA 42"),
        ([s("LOTO"), s("ŠEST"), s("OD"), s("ČETRDESET"), s("PET"), s("LOTO ŠEST OD ČETRDESET PET")],
         "Ime igre: LOTO 21, ŠEST 15, OD 11, ČETRDESET 35, PET 17, cijela fraza 99→18"),
        ([s("DVADESET"), s("DEVETI"), s("LISTOPAD"), s("DVIJE"), s("TISUĆE"), s("ŠESTE")],
         "Datum riječima: DVADESET 32, DEVETI 25, LISTOPAD 31, DVIJE 17, TISUĆE 22, ŠESTE 20"),
        ([3, 12, 21, 27, 30, 39],
         "Dvostruki Jupiter (dan + planetarni sat): 3 i korijen-3 brojevi 12, 21, 30, 39 + JUPITER 27"),
        ([1, 2, 3, 4, 6, 9],
         "Korijeni: 1 (10), 2 (29), 3 (Jupiter), 4 (22), 6 (kolo 87→15), 9 (ČETVRTAK, JUPITER)"),
        ([10, 23, 24, 27, 33, 37],
         "Kaldejski sretni složeni: 10 kotač sreće, 23 zvijezda lava, 24 ljubav/novac, 27 žezlo, 33 sreća, 37 partnerstvo"),
        ([39, 40, 42, 37, 33, 30],
         "Zbrojevi datuma: 29+10, 29+11, 20+22, 26+11, 22+11, 10+20"),
        ([19, 9, 7, 15, 4, 3],
         "Razlike datuma: 29−10, 29−20, 29−22, 26−11, 26−22, 29−26"),
    ]
    out = []
    for nums, opis in skupovi:
        nums = sorted(set(nums))
        assert len(nums) == 6 and all(1 <= x <= 45 for x in nums), (nums, opis)
        out.append((nums, opis))
    return out


# ----------------------------------------------------------------------------- kandidati
def kandidati(n: int = 80) -> list[dict]:
    """≥ 60 različitih kandidata; rang 1 = najjača derivacija."""
    n = max(n, 60)
    tezina, izvori, _ = izgradi_bazen()
    prihvaceni: list[list[int]] = []
    rezultat: list[dict] = []

    def tez(c):
        return round(sum(tezina.get(x, 0.0) for x in c), 2)

    for nums, opis in kanonski(tezina):
        if nums in prihvaceni:
            continue
        prihvaceni.append(nums)
        rezultat.append({"brojevi": nums, "rang": len(rezultat) + 1,
                         "obrazlozenje": f"Kanon: {opis} (težina {tez(nums)})"})

    # enumeracija 6-podskupova 28 najtežih brojeva, rang po težini + raznolikost
    pool = [x for x, _ in sorted(tezina.items(), key=lambda t: (-t[1], t[0]))[:28]]
    pool.sort()
    comb = np.array(list(itertools.combinations(pool, 6)), dtype=np.int16)      # (M, 6)
    w = np.array([tezina.get(x, 0.0) for x in range(46)])
    score = w[comb].sum(axis=1)
    clan = np.zeros((len(comb), 46), dtype=bool)
    np.put_along_axis(clan, comb.astype(np.int64), True, axis=1)
    usage = np.zeros(46)
    ok = np.ones(len(comb), dtype=bool)
    for c in prihvaceni:
        v = np.zeros(46, dtype=np.int8)
        v[c] = 1
        ok &= (clan @ v) <= 3
        usage[c] += 1
    LAMBDA = 0.9
    while len(rezultat) < n and ok.any():
        eff = np.where(ok, score - LAMBDA * (clan @ usage), -np.inf)
        best = int(np.argmax(eff))        # argmax = prvi po indeksu → deterministično
        c = [int(x) for x in comb[best]]
        v = np.zeros(46, dtype=np.int8)
        v[c] = 1
        ok &= (clan @ v) <= 3
        usage[c] += 1
        prihvaceni.append(c)
        rezultat.append({"brojevi": c, "rang": len(rezultat) + 1,
                         "obrazlozenje": "Bazen (težina %.1f): %s" % (
                             float(score[best]), " · ".join(oznaka_broja(x, izvori) for x in c))})
    return rezultat


def kljucni_brojevi(k: int = 12) -> list[int]:
    tezina, _, _ = izgradi_bazen()
    return [x for x, _ in sorted(tezina.items(), key=lambda t: (-t[1], t[0]))[:k]]


# ----------------------------------------------------------------------------- postupak (tekst)
def postupak_tekst() -> str:
    tezina, izvori, izvedeni = izgradi_bazen()
    L = []
    L.append(f"# {METODA} — {DATUM_TEKST}\n")
    L.append("## 1. Kaldejska tablica slova\n")
    for v in range(1, 9):
        L.append(f"- **{v}** = {' '.join(_KALDEJSKA[v])}  ({PLANET[v]})")
    L.append("- **9** se ne dodjeljuje slovu (sveti broj, Mars). Č,Ć→C, Š→S, Ž→Z, Đ→D.")
    L.append("- Ukupna vrijednost riječi = zbroj slova; > 52 → zbrajaj znamenke do ≤ 52 (SLOŽENI broj); korijen = do 1..9.\n")
    L.append("## 2. Riječi → brojevi\n")
    for rijec, w, opis in SVE_RIJECI:
        v = vrijednost_rijeci(rijec)
        izraz = "+".join(f"{c}{x}" for c, x in v["slova"]) if len(v["slova"]) <= 12 else \
            "+".join(str(x) for _, x in v["slova"])
        red = f"- **{rijec}** ({opis}): {izraz} = **{v['ukupno']}**"
        if v["slozeni"] != v["ukupno"]:
            red += f" → složeni **{v['slozeni']}**"
        red += f" → korijen {v['korijen']} ({PLANET[v['korijen']]}); značenje {v['slozeni']}: {ZNACENJE.get(v['slozeni'], '-')}; težina {w}"
        L.append(red)
    L.append("\nZanimljivost: ČETVRTAK i JUPITER oba daju 27 (ŽEZLO) — dan i planetarni sat nose isti broj; "
             "mjesec (10) i godina 2026 (2+0+2+6) oba daju 10 (KOTAČ SREĆE); PET, DVIJE i cijela datumska fraza daju 17 (ZVIJEZDA MAGA).\n")
    L.append("## 3. Znamenke datuma 29.10.2026. 20:00\n")
    for n, w, ozn, opis in DATUM_BROJEVI:
        L.append(f"- **{n}** ({ozn}: {opis}) → korijen {korijen(n)} ({PLANET[korijen(n)]}); značenje: {ZNACENJE[n]}; težina {w}")
    L.append("- 87. kolo: 87 > 52 → 8+7 = 15 (čarobnjak) → 6 (Venera); 2026 → 10 → 1.\n")
    L.append("## 4. Planetarni titraji\n")
    L.append(f"- Četvrtak = dan Jupitera; 20:00 CET je 3. noćni planetarni sat = sat Jupitera (zalazak 16:47, sat 68,8 min). "
             f"Jupiter = **3** (težina {W_JUPITER_BROJ}); svi brojevi s korijenom 3 (12, 21, 30, 39) +{W_JUPITER_TITRAJ}.")
    L.append(f"- Korijen datuma 29 → 11 → 2 = Mjesec (82,6 % opadajući): brojevi s korijenom 2 (2, 11, 20, 29, 38) +{W_MJESEC_TITRAJ}.")
    L.append(f"- Kaldejski 'sretni' složeni brojevi {sorted(SRETNI_SLOZENI)} +{W_SRETNI}.\n")
    L.append("## 5. Izvedeni brojevi (zbrojevi i razlike parova primarnih brojeva, 1..45)\n")
    L.append(f"Težina svakog = {F_IZVEDENI}·(w_a + w_b); ukupni doprinos izvedenih jednom broju ograničen na {CAP_IZVEDENI:g} (najjači parovi prvi). Najjači:")
    for n, w, opis, oa, ob in sorted(izvedeni, key=lambda t: (-t[1], t[0]))[:24]:
        L.append(f"- {opis} = **{n}**  ({oa} i {ob}; +{w})")
    L.append("")
    L.append("## 6. Bazen s težinama (silazno)\n")
    L.append("| broj | težina | korijen | značenje | izvori |")
    L.append("|---:|---:|---:|---|---|")
    for n, w in sorted(tezina.items(), key=lambda t: (-t[1], t[0])):
        src = ", ".join(f"{o} +{x:g}" for o, x in sorted(izvori[n], key=lambda t: -t[1])[:6])
        L.append(f"| {n} | {w:.2f} | {korijen(n)} | {ZNACENJE[n]} | {src} |")
    L.append("")
    L.append("## 7. Pravila generiranja i rangiranja\n")
    L.append("1. Rang 1–10: KANONSKI skupovi — izravne derivacije, od najjače (čisti datum) prema slabijoj (razlike).")
    L.append("2. Rang 11+: svi 6-člani podskupovi 28 najtežih brojeva bazena; rang po zbroju težina; prihvaćen samo skup s ≤ 3 zajednička broja sa svakim već prihvaćenim; "
             "penal 0,9 po svakom prijašnjem korištenju broja (raznolikost). Argmax je prvi po indeksu → potpuno deterministično.")
    L.append("3. Nema slučajnosti: kaldejska numerologija je čista aritmetika nad slovima i znamenkama.\n")
    L.append("## 8. Kanonski skupovi\n")
    for i, (nums, opis) in enumerate(kanonski(tezina), 1):
        L.append(f"{i}. `{' · '.join(f'{x:02d}' for x in nums)}` — {opis}")
    return "\n".join(L)


# ----------------------------------------------------------------------------- main
if __name__ == "__main__":
    # provjera statičkih tvrdnji iz docstringa
    v = {w: vrijednost_rijeci(w)["slozeni"] for w, _, _ in SVE_RIJECI}
    assert v["ČETVRTAK"] == 27 and v["JUPITER"] == 27 and v["ZAGREB"] == 20 and v["HRVATSKA"] == 24
    assert v["LUTRIJA"] == 18 and v["HRVATSKA LUTRIJA"] == 42 and v["LOTO"] == 21 and v["ŠEST"] == 15
    assert v["OD"] == 11 and v["ČETRDESET"] == 35 and v["PET"] == 17 and v["LOTO ŠEST OD ČETRDESET PET"] == 18
    assert v["DVADESET"] == 32 and v["DEVETI"] == 25 and v["LISTOPAD"] == 31 and v["DVIJE"] == 17
    assert v["TISUĆE"] == 22 and v["ŠESTE"] == 20 and v["DVADESET DEVETI LISTOPAD DVIJE TISUĆE DVADESET ŠESTE"] == 17
    assert slozeni(2026) == 10 and slozeni(87) == 15 and korijen(29) == 2 and korijen(22) == 4

    tekst = postupak_tekst()
    print(tekst)
    kand = kandidati(80)
    print("\n## Prvih 10 kandidata\n")
    for k in kand[:10]:
        print(f"{k['rang']:3d}. {' · '.join(f'{x:02d}' for x in k['brojevi'])}  — {k['obrazlozenje']}")
    kljuc = kljucni_brojevi()
    print("\nKljučni brojevi:", kljuc)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"ezo_{SLUG}.json"
    out.write_text(json.dumps({"metoda": METODA, "slug": SLUG, "postupak": tekst,
                               "kljucni_brojevi": kljuc, "kandidati": kand},
                              ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nZapisano: {out}  ({len(kand)} kandidata)")
