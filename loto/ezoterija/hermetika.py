"""Hermetika — planetarne kamee (Agrippa) i Salomonovi pentakli za Loto 6/45,
izvlačenje četvrtak 29.10.2026. u 20:00 CET, Zagreb.

ŠTO JE METODA
-------------
Renesansna hermetička magija pridaje svakom od sedam klasičnih planeta "kameu" — magični
kvadrat reda 3..9 — i uz nju niz brojeva: red kvadrata (n), broj polja (n²), magičnu
konstantu (zbroj svakog retka, stupca i dijagonale) i ukupni zbroj svih polja. Agrippa
(De occulta philosophia, knjiga II, pogl. 22, 1533) daje standardne rasporede: Saturn 3×3
(konstanta 15, zbroj 45), Jupiter 4×4 (34, 136), Mars 5×5 (65, 325), Sunce 6×6 (111, 666),
Venera 7×7 (175, 1225), Merkur 8×8 (260, 2080), Mjesec 9×9 (369, 3321). Iz istih brojeva
gematrijom su izvedena imena planetarnih inteligencija i duhova (Jupiter: Iophiel = Hismael
= 136). Salomonovi pentakli (Clavicula Salomonis, izd. S. L. Mathers 1889) pripisuju Jupiteru
sedam pečata s hebrejskim imenima i psalamskim stihovima; 4. pentakl Jupitera "služi za
stjecanje bogatstva i časti i posjedovanje velikog blaga", nosi imena IH (Iah), Adoniel i
Bariel te stih Ps 112,3, a izrađuje se "u dan i sat Jupitera". Metoda je čista aritmetika
nad fiksnim kvadratima i imenima — nema bacanja ni miješanja, pa se ne koristi slučajnost.

IZVORI I PRAVILA
  * Agrippa II.22 — sedam kamea i tablica brojeva/imena planeta (provjera: svaki redak,
    stupac i dijagonala daje konstantu; polja su točno 1..n²).
  * Clavicula Salomonis (Mathers) — 2. i 4. pentakl Jupitera: Iah (יה), Adoniel (אדוניאל),
    Bariel (בריאל), AHIH (אהיה), AB (אב), IHVH (יהוה) i Ps 112,3
    "הון ועשר בביתו וצדקתו עמדת לעד" (Bogatstvo i blago u kući su njegovoj, pravednost
    njegova stoji zauvijek).
  * Kabala: Chesed (4. sefira) = Jupiter; ime Boga El (אל = 31), arkanđeo Tzadkiel (צדקיאל =
    235), kor Chašmalim (חשמלים = 428), sfera Tzedek (צדק = 194); anđeo četvrtka Sachiel
    (סכיאל = 121).
  * Sefer Jecira / Zlatna zora: dvostruko slovo Kaph (כ = 20) = Jupiter; tarot X Kotač sreće
    = Jupiter.
  * Planetarni sati (Picatrix, Agrippa II.34): četvrtak = dan Jupitera; noćni sati četvrtka
    idu Mjesec, Saturn, JUPITER, Mars, Sunce, Venera, Merkur, …
  * Redukcija velikih brojeva na raspon 1..45: "mod 45" (45 = zbroj Saturnove kamee = raspon
    lota; 45 ostaje 45), uz teozofsku redukciju (zbroj znamenki) gdje potvrđuje.
  * Gematrija: standardna (mispar hechrechi), završna slova iste vrijednosti.

POSTUPAK ZA 29.10.2026. 20:00 — sažetak (cijeli izvod daje postupak_tekst() i zapisuje se
u out/ezo_hermetika.json)
  1. Elekcija: četvrtak = dan Jupitera; zalazak 16:47:32, izlazak 06:32:51 → noćni sat
     68,78 min; 20:00 je 3. noćni sat = sat JUPITERA (dvostruki Jupiter). Ascendent ~5° Raka
     (znak egzaltacije Jupitera), Jupiter ~24° Lava, Venera ~27° Vage (domicil), Mjesec
     82,6 % opadajući. Hrvatski "četvrtak" doslovno znači "četvrti" = broj Jupitera 4.
  2. Jupiterova kamea 4 14 15 1 / 9 7 6 12 / 5 11 10 8 / 16 2 3 13: 4 retka, 4 stupca,
     2 dijagonale, 4 kvadranta, središnji kvadrat i uglovi — 16 linija, sve zbroja 34.
  3. Jupiterovi brojevi: 4, 16, 34, 136 (→ mod 45 = 1; teozofski 1+3+6 = 10 = Kotač sreće).
  4. Konstante mod 45: Saturn 15, Jupiter 34, Mars 65→20, Sunce 111→21, Venera 175→40,
     Merkur 260→35, Mjesec 369→9; zbrojevi: 45, 136→1, 325→10, 666→6, 1225→10, 2080→10,
     3321→36.
  5. Imena: El 31, Kaph 20, Tzedek 194→14, Tzadkiel 235→10, Sachiel 121→31, Chašmalim
     428→23, Chesed 72→27; 4. pentakl: Iah 15, Adoniel 102→12, Bariel 243→18; 2. pentakl:
     AHIH 21, AB 3, IHVH 26; Ps 112,3: הון 61→16, ועשר 576→36, בביתו 420→15, וצדקתו 606→21,
     עמדת 514→19, לעד 104→14; cijeli stih 2281 → 31 = El (zatvara krug).
  6. Bazen: svaki broj 1..45 zbraja težine svih svojih izvora (gore) + elekcijske brojeve
     (29, 10, 26, 20, 3, 4) + stupnjeve planeta u satu izvlačenja.
  7. Kandidati po razredima: A) Jupiterova linija + 2 dopune (konstante, imena, polja) —
     pohlepno po težini uz penal ponavljanja, najviše 2 po liniji; B) 14 linija Sunčeve
     kamee (svaka je točno 6 brojeva ≤ 36 — gotov listić; 5. redak 25·29·10·9·26·12 nosi
     puni datum 29·10·'26); C) linije Venerine kamee ograničene na ≤ 45; D) linije Merkurove
     (Hermesove) kamee ≤ 45; E) linije Marsove kamee + 34; F) Mjesečeve linije ≤ 45 + 34;
     G) parovi Saturnovih linija. Unutar razreda rang po zbroju težina.
"""
from __future__ import annotations

import itertools
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

SLUG = "hermetika"
METODA = "Hermetika (pentakli i kamee)"
DATUM_TEKST = "četvrtak 29.10.2026. u 20:00 CET (dan i sat Jupitera), Zagreb"
OUT_DIR = Path(__file__).resolve().parent.parent / "out"
UTC_DATUM = "2026/10/29 19:00:00"      # 20:00 CET
LAT, LON = "45.815", "15.98"

# --------------------------------------------------------------------------- kamee (Agrippa II.22)
KAMEE: dict[str, list[list[int]]] = {
    "Saturn": [[4, 9, 2],
               [3, 5, 7],
               [8, 1, 6]],
    "Jupiter": [[4, 14, 15, 1],
                [9, 7, 6, 12],
                [5, 11, 10, 8],
                [16, 2, 3, 13]],
    "Mars": [[11, 24, 7, 20, 3],
             [4, 12, 25, 8, 16],
             [17, 5, 13, 21, 9],
             [10, 18, 1, 14, 22],
             [23, 6, 19, 2, 15]],
    "Sunce": [[6, 32, 3, 34, 35, 1],
              [7, 11, 27, 28, 8, 30],
              [19, 14, 16, 15, 23, 24],
              [18, 20, 22, 21, 17, 13],
              [25, 29, 10, 9, 26, 12],
              [36, 5, 33, 4, 2, 31]],
    "Venera": [[22, 47, 16, 41, 10, 35, 4],
               [5, 23, 48, 17, 42, 11, 29],
               [30, 6, 24, 49, 18, 36, 12],
               [13, 31, 7, 25, 43, 19, 37],
               [38, 14, 32, 1, 26, 44, 20],
               [21, 39, 8, 33, 2, 27, 45],
               [46, 15, 40, 9, 34, 3, 28]],
    "Merkur": [[8, 58, 59, 5, 4, 62, 63, 1],
               [49, 15, 14, 52, 53, 11, 10, 56],
               [41, 23, 22, 44, 45, 19, 18, 48],
               [32, 34, 35, 29, 28, 38, 39, 25],
               [40, 26, 27, 37, 36, 30, 31, 33],
               [17, 47, 46, 20, 21, 43, 42, 24],
               [9, 55, 54, 12, 13, 51, 50, 16],
               [64, 2, 3, 61, 60, 6, 7, 57]],
    "Mjesec": [[37, 78, 29, 70, 21, 62, 13, 54, 5],
               [6, 38, 79, 30, 71, 22, 63, 14, 46],
               [47, 7, 39, 80, 31, 72, 23, 55, 15],
               [16, 48, 8, 40, 81, 32, 64, 24, 56],
               [57, 17, 49, 9, 41, 73, 33, 65, 25],
               [26, 58, 18, 50, 1, 42, 74, 34, 66],
               [67, 27, 59, 10, 51, 2, 43, 75, 35],
               [36, 68, 19, 60, 11, 52, 3, 44, 76],
               [77, 28, 69, 20, 61, 12, 53, 4, 45]],
}
PLANETI = ["Saturn", "Jupiter", "Mars", "Sunce", "Venera", "Merkur", "Mjesec"]
RED = {p: len(KAMEE[p]) for p in PLANETI}
KONSTANTA = {p: RED[p] * (RED[p] ** 2 + 1) // 2 for p in PLANETI}       # 15, 34, 65, 111, 175, 260, 369
ZBROJ = {p: RED[p] * KONSTANTA[p] for p in PLANETI}                      # 45, 136, 325, 666, 1225, 2080, 3321
INTELIGENCIJA = {"Saturn": ("Agiel", 45), "Jupiter": ("Iophiel", 136), "Mars": ("Graphiel", 325),
                 "Sunce": ("Nachiel", 111), "Venera": ("Hagiel", 49), "Merkur": ("Tiriel", 260),
                 "Mjesec": ("Malkah be-Tarshisim", 3321)}
DUH = {"Saturn": ("Zazel", 45), "Jupiter": ("Hismael", 136), "Mars": ("Bartzabel", 325),
       "Sunce": ("Sorath", 666), "Venera": ("Kedemel", 175), "Merkur": ("Taphthartharath", 2080),
       "Mjesec": ("Chasmodai", 369)}
HEBREJSKI_BROJ = {1: "א", 2: "ב", 3: "ג", 4: "ד", 5: "ה", 6: "ו", 7: "ז", 8: "ח", 9: "ט", 10: "י",
                  11: "יא", 12: "יב", 13: "יג", 14: "יד", 15: "טו", 16: "טז"}
NOCNI_SATI_CETVRTAK = ["Mjesec", "Saturn", "Jupiter", "Mars", "Sunce", "Venera",
                       "Merkur", "Mjesec", "Saturn", "Jupiter", "Mars", "Sunce"]
ZNAKOVI = ["Ovan", "Bik", "Blizanci", "Rak", "Lav", "Djevica", "Vaga", "Škorpion",
           "Strijelac", "Jarac", "Vodenjak", "Ribe"]
ELEKCIJA_TROJKA = {29, 10, 26}          # dan · mjesec · godina

# --------------------------------------------------------------------------- gematrija
_GEM = dict(zip("אבגדהוזחטיכלמנסעפצקרשת",
                [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400]))
_GEM.update({"ך": 20, "ם": 40, "ן": 50, "ף": 80, "ץ": 90})


def gematrija(s: str) -> int:
    return sum(_GEM.get(ch, 0) for ch in s)


# (hebrejski, transliteracija, očekivana vrijednost, kontekst)
IMENA = [
    ("יהפיאל", "Iophiel", 136, "inteligencija Jupitera (Agrippa II.22)"),
    ("הסמאל", "Hismael", 136, "duh Jupitera (Agrippa II.22)"),
    ("אל", "El", 31, "ime Boga u Chesedu / sferi Jupitera"),
    ("צדקיאל", "Tzadkiel", 235, "arkanđeo Cheseda"),
    ("סכיאל", "Sachiel", 121, "anđeo četvrtka (Jupitera)"),
    ("חשמלים", "Chašmalim", 428, "anđeoski kor Cheseda"),
    ("צדק", "Tzedek", 194, "hebrejsko ime planeta Jupitera"),
    ("חסד", "Chesed", 72, "4. sefira, sfera Jupitera"),
    ("יה", "Iah", 15, "ime na vrhu 4. pentakla Jupitera"),
    ("אדוניאל", "Adoniel", 102, "ime na 4. pentaklu Jupitera"),
    ("בריאל", "Bariel", 243, "anđeo 4. pentakla Jupitera (bogatstvo)"),
    ("אהיה", "AHIH (Eheieh)", 21, "središte heksagrama 2. pentakla Jupitera"),
    ("אב", "AB (Otac)", 3, "2. pentakl Jupitera"),
    ("יהוה", "IHVH", 26, "2. pentakl Jupitera"),
]
PSALAM = [  # Ps 112,3 — stih 2. i 4. pentakla Jupitera
    ("הון", "hon", "bogatstvo", 61),
    ("ועשר", "va-ošer", "i blago", 576),
    ("בביתו", "be-veito", "u kući njegovoj", 420),
    ("וצדקתו", "ve-cidkato", "i pravednost njegova", 606),
    ("עמדת", "omedet", "stoji", 514),
    ("לעד", "la-ad", "zauvijek", 104),
]


# --------------------------------------------------------------------------- pomoćne
def red45(n: int) -> int:
    """Redukcija na krug 1..45 (45 = zbroj Saturnove kamee = raspon lota); 45, 90, 135 → 45."""
    return (n - 1) % 45 + 1


def teozofski(n: int) -> int:
    while n > 9:
        n = sum(int(c) for c in str(n))
    return n


def linije(m: list[list[int]]) -> list[tuple[str, list[int]]]:
    n = len(m)
    out = [(f"{i + 1}. redak", list(m[i])) for i in range(n)]
    out += [(f"{j + 1}. stupac", [m[i][j] for i in range(n)]) for j in range(n)]
    out.append(("glavna dijagonala", [m[i][i] for i in range(n)]))
    out.append(("sporedna dijagonala", [m[i][n - 1 - i] for i in range(n)]))
    return out


def provjeri_kameu(m: list[list[int]]) -> bool:
    """Magični kvadrat: polja su točno 1..n², svaki redak/stupac/dijagonala daje n(n²+1)/2."""
    n = len(m)
    if any(len(r) != n for r in m):
        return False
    if sorted(x for r in m for x in r) != list(range(1, n * n + 1)):
        return False
    k = n * (n * n + 1) // 2
    return all(sum(v) == k for _, v in linije(m))


def jupiter_linije() -> list[tuple[str, list[int], str]]:
    """16 linija Jupiterove kamee zbroja 34: 10 kanonskih + 4 kvadranta + središte + uglovi."""
    m = KAMEE["Jupiter"]
    L = [(oz, v, "kanonska") for oz, v in linije(m)]
    for oz, (r, c) in [("gornji lijevi kvadrant", (0, 0)), ("gornji desni kvadrant", (0, 2)),
                       ("donji lijevi kvadrant", (2, 0)), ("donji desni kvadrant", (2, 2)),
                       ("središnji kvadrat", (1, 1))]:
        L.append((oz, [m[r][c], m[r][c + 1], m[r + 1][c], m[r + 1][c + 1]], "sporedna"))
    L.append(("četiri ugla", [m[0][0], m[0][3], m[3][0], m[3][3]], "sporedna"))
    assert all(sum(v) == 34 for _, v, _ in L)
    return L


def _fmt(nums) -> str:
    return "·".join(str(x) for x in nums)


# --------------------------------------------------------------------------- astronomija (ephem, s fallbackom)
def astro() -> dict:
    """Pozicije planeta, ascendent, faza Mjeseca i planetarni sat za 29.10.2026. 19:00 UTC."""
    info = {
        "izvor": "fallback — zadane približne vrijednosti",
        "pozicije": {"Sunce": 216.4, "Mjesec": 85.0, "Merkur": 229.0, "Venera": 207.0,
                     "Mars": 137.0, "Jupiter": 144.0, "Saturn": 9.0},
        "asc": 95.0, "faza": 82.6, "lst": "22:36:25",
        "zalazak": "16:47:32", "izlazak": "06:32:51", "sat_min": 68.78, "sat_idx": 3,
    }
    try:
        import ephem
        d = ephem.Date(UTC_DATUM)
        tijela = {"Sunce": ephem.Sun, "Mjesec": ephem.Moon, "Merkur": ephem.Mercury,
                  "Venera": ephem.Venus, "Mars": ephem.Mars, "Jupiter": ephem.Jupiter,
                  "Saturn": ephem.Saturn}
        poz = {}
        for ime, cls in tijela.items():
            b = cls()
            b.compute(d, epoch=d)
            poz[ime] = math.degrees(ephem.Ecliptic(b).lon) % 360.0
        mo = ephem.Moon()
        mo.compute(d)
        obs = ephem.Observer()
        obs.lat, obs.lon, obs.date = LAT, LON, d
        obs.pressure, obs.horizon = 0, "-0:34"
        lst = math.degrees(obs.sidereal_time())
        eps = math.radians(23.4365)
        phi = math.radians(float(LAT))
        ramc = math.radians(lst)
        asc = math.degrees(math.atan2(math.cos(ramc),
                                      -(math.sin(ramc) * math.cos(eps) + math.tan(phi) * math.sin(eps)))) % 360.0
        sun = ephem.Sun()
        zal = obs.previous_setting(sun)
        izl = obs.next_rising(sun)
        sat_min = (izl - zal) * 24 * 60 / 12
        idx = int(((d - zal) * 24 * 60) // sat_min) + 1
        cet = lambda t: ephem.Date(t + ephem.hour).datetime().strftime("%H:%M:%S")
        h = lst / 15.0
        info.update(izvor="ephem", pozicije=poz, asc=asc, faza=float(mo.phase),
                    lst="%02d:%02d:%02d" % (int(h), int(h * 60) % 60, int(h * 3600) % 60),
                    zalazak=cet(zal), izlazak=cet(izl), sat_min=sat_min, sat_idx=idx)
    except Exception as e:          # noqa: BLE001 — ephem nedostupan ili neočekivana pogreška
        info["greska"] = repr(e)
    info["sat_planet"] = NOCNI_SATI_CETVRTAK[info["sat_idx"] - 1]
    return info


def znak_stupanj(lon: float) -> tuple[str, int, int]:
    """Tropska duljina → (znak, stupanj 0..29, redni broj znaka 1..12)."""
    i = int(lon // 30) % 12
    return ZNAKOVI[i], int(lon % 30), i + 1


# --------------------------------------------------------------------------- bazen težina
def izgradi_bazen() -> tuple[dict[int, float], dict[int, list[tuple[str, float]]], dict]:
    A = astro()
    tezina: dict[int, float] = defaultdict(float)
    izvori: dict[int, list[tuple[str, float]]] = defaultdict(list)

    def dodaj(n: int, w: float, oznaka: str):
        if 1 <= n <= 45:
            tezina[n] += w
            izvori[n].append((oznaka, w))

    # 1. Jupiter — vladar dana i sata (dvostruki)
    dodaj(KONSTANTA["Jupiter"], 8.0, "konstanta Jupiterove kamee 34")
    dodaj(4, 6.0, "broj Jupitera 4 (kamea 4×4, Chesed = 4. sefira)")
    dodaj(16, 4.0, "16 polja Jupiterove kamee")
    dodaj(red45(ZBROJ["Jupiter"]), 4.0, "zbroj Jupiterove kamee 136 → 1 (Iophiel = Hismael = 136)")
    dodaj(10, 3.0, "136 → 1+3+6 = 10 = Kotač sreće (tarot X, adut Jupitera)")
    dodaj(20, 5.0, "Kaph (כ = 20), slovo Jupitera = sat 20:00 u satu Jupitera")
    dodaj(31, 4.0, "El (אל = 31), ime Boga u Chesedu")
    dodaj(red45(121), 1.0, "Sachiel 121 → 31")
    dodaj(red45(sum(v for *_, v in PSALAM)), 1.0, "Ps 112:3 cijeli stih 2281 → 31")
    dodaj(red45(235), 1.0, "Tzadkiel 235 → 10")
    dodaj(red45(194), 2.0, "Tzedek 194 → 14")
    dodaj(red45(428), 1.0, "Chašmalim 428 → 23")
    dodaj(red45(72), 1.0, "Chesed 72 → 27")
    for c in range(1, 17):
        dodaj(c, 1.0, "polje Jupiterove kamee")
    dodaj(4, 1.0, "četvrtak = dan Jupitera, 4. dan")
    dodaj(A["sat_idx"], 1.0, f"{A['sat_idx']}. noćni sat = sat Jupitera")
    # 2. pentakli Jupitera (Clavicula Salomonis)
    dodaj(15, 2.5, "Iah (יה = 15), 4. pentakl Jupitera")
    dodaj(red45(102), 2.5, "Adoniel 102 → 12, 4. pentakl")
    dodaj(red45(243), 3.0, "Bariel 243 → 18, anđeo 4. pentakla (bogatstvo)")
    dodaj(21, 2.0, "AHIH (אהיה = 21), 2. pentakl Jupitera")
    dodaj(3, 2.0, "AB (אב = 3), 2. pentakl Jupitera")
    dodaj(26, 2.0, "IHVH (יהוה = 26), 2. pentakl Jupitera")
    for heb, tr, _, v in PSALAM:
        dodaj(red45(v), 1.5, f"Ps 112:3 {tr} {heb} = {v} → {red45(v)}")
    dodaj(red45(112), 1.0, "Psalam 112 → 22")
    dodaj(3, 0.5, "3. stih psalma")
    dodaj(4, 0.5, "4. pentakl")
    dodaj(7, 0.5, "7 pentakla Jupitera")
    # 3. konstante i zbrojevi ostalih kamea (mod 45)
    for p in PLANETI:
        if p != "Jupiter":
            dodaj(red45(KONSTANTA[p]), 2.0, f"konstanta {p} {KONSTANTA[p]} → {red45(KONSTANTA[p])}")
    dodaj(45, 2.0, "zbroj Saturnove kamee 45 = raspon lota")
    dodaj(red45(ZBROJ["Sunce"]), 1.0, "zbroj Sunčeve kamee 666 → 6")
    dodaj(red45(ZBROJ["Mjesec"]), 1.0, "zbroj Mjesečeve kamee 3321 → 36")
    for p in ("Mars", "Venera", "Merkur"):
        dodaj(red45(ZBROJ[p]), 0.5, f"zbroj {p} {ZBROJ[p]} → 10")
    for p in PLANETI:
        dodaj(RED[p], 0.5, f"red kamee {p} ({RED[p]})")
    # 4. elekcija — brojevi trenutka
    dodaj(29, 1.5, "dan 29")
    dodaj(10, 1.0, "listopad = 10. mjesec")
    dodaj(26, 1.0, "godina '26")
    dodaj(11, 0.5, "2+9 = 11")
    dodaj(22, 0.5, "2+9+1+0+2+0+2+6 = 22")
    dodaj(red45(87), 0.5, "87. kolo → 42")
    # 5. nebo u satu izvlačenja
    zj, dj, ij = znak_stupanj(A["pozicije"]["Jupiter"])
    dodaj(dj, 2.0, f"Jupiter {dj}° {zj}")
    dodaj(ij, 0.5, f"Jupiter u {ij}. znaku ({zj})")
    za, da, ia = znak_stupanj(A["asc"])
    dodaj(ia, 1.0, f"Ascendent u {ia}. znaku ({za}; Rak = egzaltacija Jupitera)")
    dodaj(da, 0.5, f"Ascendent {da}° {za}")
    dodaj(15, 1.0, "egzaltacija Jupitera 15° Raka")
    zv, dv, _ = znak_stupanj(A["pozicije"]["Venera"])
    dodaj(dv, 1.0, f"Venera {dv}° {zv} (domicil)")
    for p in ("Sunce", "Mjesec", "Merkur", "Mars", "Saturn"):
        z, dg, _ = znak_stupanj(A["pozicije"][p])
        dodaj(dg, 0.5, f"{p} {dg}° {z}")
    return dict(tezina), dict(izvori), A


def oznaka_broja(n: int, izvori: dict) -> str:
    if n not in izvori:
        return "—"
    return max(izvori[n], key=lambda t: t[1])[0].split(" (")[0][:32].strip()


def hijerarhija_dopuna(tezina: dict[int, float]) -> list[int]:
    """Dopunski brojevi: konstanta 34 prva, zatim po težini bazena (silazno), pa po veličini."""
    rest = sorted((x for x in tezina if x != 34), key=lambda x: (-tezina[x], x))
    return [34] + rest


# --------------------------------------------------------------------------- razredi kandidata
def _score(nums, tezina) -> float:
    s = sum(tezina.get(x, 0.0) for x in nums)
    if ELEKCIJA_TROJKA <= set(nums):
        s += 5.0                                  # linija koja nosi puni datum elekcije
    return s


def _reduciraj(nums, tezina, cilj=6) -> list[int]:
    """> 6 brojeva: izbacuj najslabije po težini (pri jednakoj težini veći broj)."""
    v = sorted(nums)
    while len(v) > cilj:
        v.remove(min(v, key=lambda x: (tezina.get(x, 0.0), -x)))
    return v


def _dopuni(nums, hijer, cilj=6) -> tuple[list[int], list[int]]:
    v, dodano = list(nums), []
    for s in hijer:
        if len(v) >= cilj:
            break
        if s not in v:
            v.append(s)
            dodano.append(s)
    return sorted(v), dodano


def razred_A(tezina, izvori, max_po_liniji=2, ukupno=32, lam=1.5) -> list[tuple[list[int], str, float]]:
    """Jupiterova linija (4 polja, zbroj 34) + 2 dopune; pohlepno po težini uz penal ponavljanja."""
    lin = jupiter_linije()
    hijer = hijerarhija_dopuna(tezina)
    kand = []
    for li, (oz, cells, vrsta) in enumerate(lin):
        base = sum(tezina.get(c, 0.0) for c in cells) + (2.0 if vrsta == "kanonska" else 0.0)
        pool = [x for x in hijer if x not in cells][:18]
        for a, b in itertools.combinations(pool, 2):
            nums = sorted(cells + [a, b])
            kand.append((base + tezina[a] + tezina[b], nums, li, oz, cells, a, b))
    usage = defaultdict(int)
    po_liniji = defaultdict(int)
    seen = set()
    out = []
    while len(out) < ukupno:
        best = None
        for sc, nums, li, oz, cells, a, b in kand:
            t = tuple(nums)
            if po_liniji[li] >= max_po_liniji or t in seen:
                continue
            eff = sc - lam * sum(usage[x] for x in nums)
            key = (-eff, t)
            if best is None or key < best[0]:
                best = (key, sc, nums, li, oz, cells, a, b)
        if best is None:
            break
        _, sc, nums, li, oz, cells, a, b = best
        seen.add(tuple(nums))
        po_liniji[li] += 1
        for x in nums:
            usage[x] += 1
        opis = (f"Jupiter: {oz} kamee ({_fmt(cells)} = 34) + {a} ({oznaka_broja(a, izvori)}) "
                f"+ {b} ({oznaka_broja(b, izvori)})")
        out.append((nums, opis, round(sc, 2)))
    return out


def razred_B(tezina, izvori) -> list[tuple[list[int], str, float]]:
    """Sunčeva kamea: svaka od 14 linija je točno 6 brojeva ≤ 36 (zbroj 111) — gotov listić."""
    out = []
    for oz, v in linije(KAMEE["Sunce"]):
        nums = sorted(v)
        opis = f"Sunce: {oz} kamee ({_fmt(v)} = 111)"
        if ELEKCIJA_TROJKA <= set(nums):
            opis += " — nosi puni datum 29·10·'26"
        out.append((nums, opis, round(_score(nums, tezina), 2)))
    return sorted(out, key=lambda t: (-t[2], t[0]))


def _razred_kamee(planet, tezina, izvori, min_polja, dopuna=True) -> list[tuple[list[int], str, float]]:
    """Linije kamee ograničene na ≤ 45: > 6 → reduciraj (najslabiji van), < 6 → dopuni po hijerarhiji."""
    hijer = hijerarhija_dopuna(tezina)
    out = []
    for oz, v in linije(KAMEE[planet]):
        u = [x for x in v if x <= 45]
        if len(u) < min_polja:
            continue
        if len(u) > 6:
            nums = _reduciraj(u, tezina)
            opis = f"{planet}: {oz} kamee ≤45 ({_fmt(u)}), izbačen{'i' if len(u) - 6 > 1 else ''} {_fmt(sorted(set(u) - set(nums)))}"
        elif len(u) < 6:
            if not dopuna:
                continue
            nums, dod = _dopuni(u, hijer)
            opis = f"{planet}: {oz} kamee ≤45 ({_fmt(u)}) + " + " + ".join(
                f"{d} ({oznaka_broja(d, izvori)})" for d in dod)
        else:
            nums = sorted(u)
            opis = f"{planet}: {oz} kamee ≤45 ({_fmt(u)})"
        out.append((nums, opis, round(_score(nums, tezina), 2)))
    return sorted(out, key=lambda t: (-t[2], t[0]))


def razred_G(tezina, izvori) -> list[tuple[list[int], str, float]]:
    """Saturn: dva paralelna retka ili stupca (2 × 15 = 30; cijela kamea 45 = raspon lota)."""
    m = KAMEE["Saturn"]
    L = linije(m)
    out = []
    for grupa in (L[0:3], L[3:6]):
        for (o1, v1), (o2, v2) in itertools.combinations(grupa, 2):
            nums = sorted(v1 + v2)
            out.append((nums, f"Saturn: {o1} + {o2} kamee ({_fmt(v1)} | {_fmt(v2)}), 2 × 15",
                        round(_score(nums, tezina), 2)))
    return sorted(out, key=lambda t: (-t[2], t[0]))


def _popuni_iz_bazena(prihvaceni, tezina, izvori, koliko) -> list[tuple[list[int], str, float]]:
    """Rezerva: 6-podskupovi 24 najtežih brojeva, pohlepno s penalom i ≤ 3 zajednička broja."""
    if koliko <= 0:
        return []
    pool = sorted(sorted(tezina, key=lambda x: (-tezina[x], x))[:24])
    comb = np.array(list(itertools.combinations(pool, 6)), dtype=np.int16)
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
    out = []
    while len(out) < koliko and ok.any():
        eff = np.where(ok, score - 0.9 * (clan @ usage), -np.inf)
        best = int(np.argmax(eff))
        c = [int(x) for x in comb[best]]
        v = np.zeros(46, dtype=np.int8)
        v[c] = 1
        ok &= (clan @ v) <= 3
        usage[c] += 1
        out.append((c, "Bazen hermetičkih težina: " + " · ".join(f"{x} {oznaka_broja(x, izvori)}" for x in c),
                    round(float(score[best]), 2)))
    return out


def kandidati(n: int = 80) -> list[dict]:
    """≥ 60 različitih kandidata; rang 1 = najjača derivacija (Jupiter-dominantne prve)."""
    n = max(n, 60)
    tezina, izvori, _ = izgradi_bazen()
    razredi = [
        ("A", razred_A(tezina, izvori)),
        ("B", razred_B(tezina, izvori)),
        ("C", _razred_kamee("Venera", tezina, izvori, min_polja=6)),
        ("D", _razred_kamee("Merkur", tezina, izvori, min_polja=4)),
        ("E", _razred_kamee("Mars", tezina, izvori, min_polja=5)),
        ("F", _razred_kamee("Mjesec", tezina, izvori, min_polja=5)),
        ("G", razred_G(tezina, izvori)),
    ]
    seen = set()
    rezultat = []
    for r, lst in razredi:
        for nums, opis, sc in lst:
            t = tuple(nums)
            assert len(t) == 6 and all(1 <= x <= 45 for x in t) and len(set(t)) == 6, (r, nums)
            if t in seen:
                continue
            seen.add(t)
            rezultat.append({"brojevi": list(nums), "rang": len(rezultat) + 1,
                             "obrazlozenje": f"[{r}] {opis} (težina {sc})"})
    if len(rezultat) < n:
        for nums, opis, sc in _popuni_iz_bazena([k["brojevi"] for k in rezultat], tezina, izvori,
                                                 n - len(rezultat)):
            if tuple(nums) not in seen:
                seen.add(tuple(nums))
                rezultat.append({"brojevi": nums, "rang": len(rezultat) + 1,
                                 "obrazlozenje": f"[H] {opis} (težina {sc})"})
    return rezultat[:n]


def kljucni_brojevi(k: int = 12) -> list[int]:
    tezina, _, _ = izgradi_bazen()
    return [x for x in sorted(tezina, key=lambda x: (-tezina[x], x))[:k]]


# --------------------------------------------------------------------------- postupak (tekst)
def _tablica(m: list[list[int]]) -> list[str]:
    n = len(m)
    L = ["| " + " | ".join(" " for _ in range(n)) + " |", "|" + "---:|" * n]
    L += ["| " + " | ".join(f"{x:>2}" for x in r) + " |" for r in m]
    return L


def postupak_tekst() -> str:
    tezina, izvori, A = izgradi_bazen()
    L = [f"# {METODA} — {DATUM_TEKST}\n"]
    # 1
    L.append("## 1. Elekcija: dan i sat Jupitera\n")
    L.append(f"- Četvrtak = dan Jupitera (hrv. *četvrtak* = 'četvrti'; Jupiterov broj je 4, Chesed je 4. sefira).")
    L.append(f"- Zalazak Sunca 29.10. {A['zalazak']} CET, izlazak 30.10. {A['izlazak']} CET → noćni planetarni sat "
             f"{A['sat_min']:.2f} min; 20:00 CET je **{A['sat_idx']}. noćni sat = sat {A['sat_planet'].upper()}A** "
             f"(noćni sati četvrtka: {', '.join(NOCNI_SATI_CETVRTAK[:7])}, …). Dan i sat istog planeta = dvostruki Jupiter, "
             f"upravo uvjet pod kojim se izrađuju Jupiterovi pentakli.")
    poz = A["pozicije"]
    opis_poz = []
    for p in ("Sunce", "Mjesec", "Merkur", "Venera", "Mars", "Jupiter", "Saturn"):
        z, dg, _ = znak_stupanj(poz[p])
        opis_poz.append(f"{p} {dg}° {z}")
    za, da, _ = znak_stupanj(A["asc"])
    L.append(f"- Nebo u 19:00 UTC (izvor: {A['izvor']}): " + ", ".join(opis_poz) +
             f"; Ascendent {da}° {za} (LST {A['lst']}); Mjesec {A['faza']:.1f} % osvijetljen, opadajući.")
    L.append("- Ascendent u Raku = znak egzaltacije Jupitera (15° Raka); Venera u Vagi u domicilu; Jupiter u Lavu "
             "(stupanj Jupitera ulazi u bazen).\n")
    # 2
    L.append("## 2. Sedam kamea (Agrippa, De occulta philosophia II.22)\n")
    L.append("Provjera magičnosti (polja 1..n², svaki redak/stupac/dijagonala = konstanta): " +
             ", ".join(f"{p} {'✓' if provjeri_kameu(KAMEE[p]) else '✗'}" for p in PLANETI) + "\n")
    L.append("| planet | red | polja | konstanta | mod 45 | zbroj | mod 45 | teozofski | inteligencija | duh |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---|---|")
    for p in PLANETI:
        L.append(f"| {p} | {RED[p]} | {RED[p] ** 2} | {KONSTANTA[p]} | {red45(KONSTANTA[p])} | {ZBROJ[p]} | "
                 f"{red45(ZBROJ[p])} | {teozofski(ZBROJ[p])} | {INTELIGENCIJA[p][0]} ({INTELIGENCIJA[p][1]}) | "
                 f"{DUH[p][0]} ({DUH[p][1]}) |")
    L.append("")
    for p in PLANETI:
        L.append(f"**{p}** ({RED[p]}×{RED[p]}, konstanta {KONSTANTA[p]}, zbroj {ZBROJ[p]})\n")
        L += _tablica(KAMEE[p])
        L.append("")
    L.append("Jupiterova kamea hebrejskim slovima (kako je Agrippa urezuje u kositar):\n")
    L += _tablica([[HEBREJSKI_BROJ[x] for x in r] for r in KAMEE["Jupiter"]])
    L.append("")
    # 3
    L.append("## 3. Jupiterovi brojevi i 16 linija kamee\n")
    L.append("- Jupiterovi brojevi: **4** (red), **16** (polja), **34** (konstanta), **136** (zbroj) → 136 mod 45 = **1**, "
             "teozofski 1+3+6 = **10** = Kotač sreće (tarot X, adut Jupitera).")
    L.append("- Slovo Jupitera Kaph (כ) = **20** = sat izvlačenja 20:00; ime Boga u Chesedu El (אל) = **31**.")
    L.append("- Linije zbroja 34 (4 retka, 4 stupca, 2 dijagonale, 4 kvadranta, središte, uglovi):")
    for oz, v, vrsta in jupiter_linije():
        L.append(f"  - {oz}: {_fmt(v)}  ({vrsta}; težina {sum(tezina.get(c, 0) for c in v):.1f})")
    L.append("")
    # 4
    L.append("## 4. Imena i gematrija (standardna, mispar hechrechi)\n")
    L.append("| ime | hebrejski | vrijednost | mod 45 | teozofski | kontekst |")
    L.append("|---|---|---:|---:|---:|---|")
    for heb, tr, v, kont in IMENA:
        L.append(f"| {tr} | {heb} | {gematrija(heb)} | {red45(v)} | {teozofski(v)} | {kont} |")
    L.append("")
    L.append("Stih 2. i 4. pentakla Jupitera, Ps 112,3 — הון ועשר בביתו וצדקתו עמדת לעד:\n")
    L.append("| riječ | translit. | značenje | vrijednost | mod 45 |")
    L.append("|---|---|---|---:|---:|")
    for heb, tr, zn, v in PSALAM:
        L.append(f"| {heb} | {tr} | {zn} | {gematrija(heb)} | {red45(v)} |")
    uk = sum(v for *_, v in PSALAM)
    L.append(f"\nCijeli stih = {uk} → mod 45 = **{red45(uk)}** = El (אל = 31): stih bogatstva vraća se na ime Boga u "
             "sferi Jupitera. Psalam 112 → 112 mod 45 = 22; 3. stih → 3.\n")
    # 5
    L.append("## 5. Pentakli Jupitera (Clavicula Salomonis, izd. Mathers)\n")
    L.append("- **4. pentakl Jupitera** — 'služi za stjecanje bogatstva i časti i posjedovanje velikog blaga; njegov je anđeo "
             "Bariel; urezuje se u srebro u dan i sat Jupitera dok je Jupiter u Raku'. Iznad sigila ime IH (Iah = 15), ispod "
             "Adoniel (102 → 12) i Bariel (243 → 18) raspoređen oko kvadrata od četiri polja; uokolo Ps 112,3.")
    L.append("- **2. pentakl Jupitera** — 'za stjecanje slave, časti, dostojanstava, bogatstva i svakog dobra'. U heksagramu "
             "AHIH (21), AB (3) i IHVH (26); izvan njega početak Ps 112,3 (הון ועשר).")
    L.append("- Jupiter ima 7 pentakala (broj 7); operacija je u 4. pentaklu (broj 4), 3. stihu (broj 3).\n")
    # 6
    L.append("## 6. Bazen hermetičkih težina (silazno)\n")
    L.append("| broj | težina | izvori |")
    L.append("|---:|---:|---|")
    for x in sorted(tezina, key=lambda x: (-tezina[x], x)):
        src = "; ".join(f"{o} +{w:g}" for o, w in sorted(izvori[x], key=lambda t: -t[1])[:6])
        L.append(f"| {x} | {tezina[x]:.1f} | {src} |")
    L.append("")
    L.append(f"Hijerarhija dopuna (34 prva, zatim po težini): {_fmt(hijerarhija_dopuna(tezina)[:20])} …\n")
    # 7
    L.append("## 7. Razredi kandidata i rangiranje\n")
    L.append("- **A (Jupiter-dominantni, rang 1–32)**: jedna od 16 linija Jupiterove kamee (4 broja, zbroj 34) + 2 dopune "
             "iz hijerarhije (konstante, imena pentakala, Jupiterova polja). Ocjena = zbroj težina (+2 za kanonsku liniju); "
             "pohlepni odabir s penalom 1,5 po svakom prijašnjem korištenju broja, najviše 2 kombinacije po liniji. "
             "Svaka dopuna ≤ 36 ujedno je polje Sunčeve, a ≤ 45 polje Venerine kamee.")
    L.append("- **B (Sunce)**: 14 linija Sunčeve kamee — svaka je točno 6 brojeva ≤ 36 (zbroj 111), tj. gotov listić; "
             "linija koja nosi puni datum 29·10·'26 (5. redak) dobiva +5.")
    L.append("- **C (Venera, u domicilu)**: 16 linija 7×7 kamee ograničeno na ≤ 45; sedmočlane linije gube najslabiji broj.")
    L.append("- **D (Merkur = Hermes, imenjak hermetike)**: linije 8×8 kamee ≤ 45 (≥ 4 polja); više od 6 → reduciraj, "
             "manje → dopuni po hijerarhiji (34 prvi). Retci 1, 2, 7 i 8 daju upravo četiri kvarteta Jupiterovih polja 1..16.")
    L.append("- **E (Mars, vladar sljedećeg sata)**: 12 linija 5×5 kamee (zbroj 65) + konstanta Jupitera 34.")
    L.append("- **F (Mjesec, noćno svjetlilo, opadajući)**: linije 9×9 kamee ≤ 45 (≥ 5 polja) + dopune / redukcija.")
    L.append("- **G (Saturn)**: parovi paralelnih linija 3×3 kamee (2 × 15); cijela kamea = 45 = raspon lota.")
    L.append("- Unutar razreda rang po zbroju težina (pri jednakoj: leksikografski). Duplikati preko razreda se preskaču. "
             "Ako treba više od strukturnih kandidata, popunjava se iz bazena (razred H, ≤ 3 zajednička broja). "
             "Sve je deterministično — nema slučajnosti.\n")
    # 8
    L.append("## 8. Razred A — Jupiterove kombinacije\n")
    for i, (nums, opis, sc) in enumerate(razred_A(tezina, izvori), 1):
        L.append(f"{i}. `{' · '.join(f'{x:02d}' for x in nums)}` — {opis} (težina {sc})")
    return "\n".join(L)


# --------------------------------------------------------------------------- main
if __name__ == "__main__":
    # testovi: magičnost kamea, konstante, gematrija imena
    for p in PLANETI:
        assert provjeri_kameu(KAMEE[p]), f"kamea {p} nije magična"
    assert [KONSTANTA[p] for p in PLANETI] == [15, 34, 65, 111, 175, 260, 369]
    assert [ZBROJ[p] for p in PLANETI] == [45, 136, 325, 666, 1225, 2080, 3321]
    assert [red45(KONSTANTA[p]) for p in PLANETI] == [15, 34, 20, 21, 40, 35, 9]
    assert red45(136) == 1 and red45(45) == 45 and teozofski(136) == 1 and red45(2281) == 31
    for heb, tr, v, _ in IMENA:
        assert gematrija(heb) == v, (tr, gematrija(heb), v)
    for heb, tr, _, v in PSALAM:
        assert gematrija(heb) == v, (tr, gematrija(heb), v)
    assert len(jupiter_linije()) == 16

    tekst = postupak_tekst()
    print(tekst)
    kand = kandidati(80)
    print("\n## Prvih 10 kandidata\n")
    for k in kand[:10]:
        print(f"{k['rang']:3d}. {' · '.join(f'{x:02d}' for x in k['brojevi'])}  — {k['obrazlozenje']}")
    kljuc = kljucni_brojevi()
    print("\nKljučni brojevi:", kljuc)
    # brza provjera valjanosti i raznolikosti
    assert len(kand) >= 60
    assert len({tuple(k["brojevi"]) for k in kand}) == len(kand)
    for k in kand:
        b = k["brojevi"]
        assert len(b) == 6 and b == sorted(b) and len(set(b)) == 6 and all(1 <= x <= 45 for x in b)
    razl = len({x for k in kand[:60] for x in k["brojevi"]})
    assert razl >= 30, razl
    print(f"Prvih 60 kandidata koristi {razl} različitih brojeva.")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"ezo_{SLUG}.json"
    out.write_text(json.dumps({"metoda": METODA, "slug": SLUG, "postupak": tekst,
                               "kljucni_brojevi": kljuc, "kandidati": kand},
                              ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nZapisano: {out}  ({len(kand)} kandidata)")
