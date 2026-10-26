# -*- coding: utf-8 -*-
"""Gematrija — ezoterična metoda za Loto 6/45, izvlačenje četvrtak 29.10.2026. u 20:00 CET.

ŠTO JE METODA
=============
Gematrija je hebrejska (i kasnije kabalistička) praksa pridruživanja brojčanih vrijednosti
slovima, pa riječima; riječi iste vrijednosti smatraju se povezanima, a vrijednosti "dobrih"
riječi (sreća, blagoslov, život, uspjeh, obilje) nose njihovu snagu. Ovdje se vrijednosti
riječi i datuma svode u raspon 1..45 i od njih se grade 6-člani skupovi.

KORIŠTENE CIFRE (sustavi vrednovanja slova)
-------------------------------------------
1. mispar hechrechi (standardna): א=1 ב=2 ג=3 ד=4 ה=5 ו=6 ז=7 ח=8 ט=9 י=10 כ=20 ל=30 מ=40
   נ=50 ס=60 ע=70 פ=80 צ=90 ק=100 ר=200 ש=300 ת=400; završna slova (ך ם ן ף ץ) kao osnovna.
   Težina cifre 1,0.
2. mispar gadol: isto, ali završna slova ך=500 ם=600 ן=700 ף=800 ץ=900. Težina 0,7.
3. mispar katan: svaka vrijednost svedena na znamenku 1..9 (י=1 כ=2 … ק=1 ר=2 ש=3 ת=4). Težina 0,8.
4. mispar siduri (redna): א=1 … ת=22; završna slova kao osnovna. Težina 0,6.
5. atbash: zrcalna zamjena א↔ת ב↔ש ג↔ר … pa standardna vrijednost zamijenjene riječi. Težina 0,5.
6. latinična gematrija A=1 … Z=26 (težina 0,8) i njezina pitagorejska redukcija slova na 1..9
   (A=1 … I=9, J=1 …, težina 0,5) za hrvatske riječi bez dijakritika.

PRAVILA SVOĐENJA U 1..45 (dokumentirana, deterministična)
---------------------------------------------------------
* identitet: vrijednost 1..45 ostaje kakva jest (težina redukcije 1,0);
* mod 45: za v>45 uzima se ((v-1) mod 45)+1, tj. 0→45 (težina 0,8);
* zbroj znamenki: zbrajaju se znamenke dok rezultat ne padne u 1..45 (težina 0,7 ako je v>45,
  inače 0,4 kao sporedna derivacija);
* teozofska redukcija: zbrajanje do jedne znamenke 1..9 (težina 0,25, samo za bazen);
* mispar katan riječi već je svojom konstrukcijom "redukcija" i ulazi preko cifre 3.
Snaga svake derivacije = snaga riječi × težina cifre × težina redukcije; zbroj po broju daje
"gematrijski bazen" (rang brojeva 1..45).

RIJEČI (snaga 1..10, jače riječi prije slabijih)
------------------------------------------------
מזל mazal=77 (10), ברכה berakha=227 (9), חי chai=18 (9), הצלחה hatzlacha=138 (8),
זכייה zekhiya=52 (8, "dobitak"), שפע shefa=450 (7), עושר osher=576 (7), טוב tov=17 (7),
חסד chesed=72 (7), גורל goral=239 (7, "ždrijeb/kocka"), אושר osher=507 (6), פרנסה parnasa=395 (6),
ברוך barukh=228 (6; gadol 708), יהוה=26 (6), ו"ה=11 (5), צדק tzedek=194 (5, Jupiter — planetarni
sat izvlačenja).
Datum: י"ט=19 (19. hešvan — hebrejski datum nakon zalaska sunca, u trenutku izvlačenja; snaga 8),
י"ח=18 (18. hešvan, građanski datum; 7), חשון=364 (7; gadol 1014), תשפ"ז=787 (godina 5787 bez
tisućica; 8), מרחשון=604 (4).
Latinica: LOTO=62, LUTRIJA=91, SRECA=46, ZAGREB=59, CETVRTAK=100, JUPITER=99 (snaga 5–6).

POSTUPAK (IZVOD) ZA 29.10.2026.
===============================
Hebrejski datum provjeren bibliotekom pyluach ako je dostupna (18. hešvan 5787 = 29.10.2026.,
Rosh Hashana 5787 = 12.9.2026.); nakon zalaska sunca (16:47 CET) počinje 19. hešvan, pa izvlačenje
u 20:00 pada na י"ט חשון תשפ"ז.

Standardna gematrija (hechrechi) → mod 45 / zbroj znamenki:
  מזל 77 → 32 / 14;  ברכה 227 → 2 / 11;  חי 18 → 18;  הצלחה 138 → 3 / 12;  זכייה 52 → 7 / 7;
  שפע 450 → 45 / 9;  עושר 576 → 36 / 18;  טוב 17 → 17;  חסד 72 → 27 / 9;  גורל 239 → 14 / 14;
  אושר 507 → 12 / 12;  פרנסה 395 → 35 / 17;  ברוך 228 → 3 / 12;  יהוה 26 → 26;  ו"ה 11 → 11;
  צדק 194 → 14 / 14;  י"ט 19 → 19;  י"ח 18 → 18;  חשון 364 → 4 / 13;  תשפ"ז 787 → 22 / 22;
  מרחשון 604 → 19 / 10.
Mispar gadol (razlika samo kod završnih slova): ברוך 708 → 33 / 15;  חשון 1014 → 24 / 6;
  מרחשון 1254 → 39 / 12.
Mispar katan: מזל 14, ברכה 11, חי 9, הצלחה 30, זכייה 16, שפע 18, עושר 18, טוב 17, חסד 18,
  גורל 14, אושר 12, פרנסה 26, ברוך 12, יהוה 17, ו"ה 11, צדק 14, י"ט 10, י"ח 9, חשון 22, תשפ"ז 22,
  מרחשון 28.
Mispar siduri: מזל 32, ברכה 38, חי 18, הצלחה 48→3, זכייה 43, שפע 54→9, עושר 63→18, טוב 17,
  חסד 27, גורל 41, אושר 48→3, פרנסה 71→26, ברוך 39, יהוה 26, ו"ה 11, צדק 41, י"ט 19, י"ח 18,
  חשון 49→4, תשפ"ז 67→22, מרחשון 82→37.
Atbash (zamjena pa standardna) → mod 45 / zbroj znamenki: מזל 100 → 10/1;  ברכה 423 → 18/9;
  חי 100 → 10/1;  הצלחה 265 → 40/13;  זכייה 270 → 45/9;  שפע 15;  עושר 92 → 2/11;  טוב 430 → 25/7;
  חסד 168 → 33/15;  גורל 303 → 33/6;  אושר 485 → 35/17;  פרנסה 116 → 26/8;  ברוך 413 → 8/8;
  יהוה 300 → 30/3;  ו"ה 170 → 35/8;  צדק 109 → 19/10;  י"ט 90 → 45/9;  י"ח 100 → 10/1;
  חשון 151 → 16/7;  תשפ"ז 79 → 34/16;  מרחשון 164 → 29/11.
Latinica A=1..Z=26 → mod 45 / zbroj znamenki: LOTO 62 → 17/8 (17 = טוב!), LUTRIJA 91 → 1/10,
  SRECA 46 → 1/10, ZAGREB 59 → 14/14, CETVRTAK 100 → 10/1, JUPITER 99 → 9/18 (18 = חי).
Slova riječi (standardna vrijednost slova mod 45): מזל = 40·7·30, ברכה = 2·20·20·5, חי = 8·10,
  הצלחה = 5·45·30·8·5, זכייה = 7·20·10·10·5, שפע = 30·35·25, עושר = 25·6·30·20, טוב = 9·6·2,
  חסד = 8·15·4, גורל = 3·6·20·30.

Upadljiva podudaranja za ovaj datum: 22 = תשפ"ז (mod 45 i zbroj znamenki) = katan חשון = zbroj
svih znamenki datuma 29.10.2026.; 14 = zbroj znamenki i katan riječi מזל = גורל = צדק (Jupiter) =
ZAGREB; 18 = חי = י"ח = עושר (zbroj znamenki) = JUPITER (zbroj znamenki); 26 = יהוה = 2026;
17 = טוב = LOTO (mod 45); 32 = מזל (mod 45) = siduri מזל; 12 = אושר = הצלחה (zbroj) = ברוך (zbroj).

GRADNJA KANDIDATA (rang = redoslijed razina, unutar razine snaga riječi)
-----------------------------------------------------------------------
R1  Vrh bazena: 6 najjačih brojeva ukupnog bazena; zatim bazen samo standardne cifre; zatim bazen
    samo ključnih riječi (mazal, berakha, chai, hatzlacha, zekhiya).
R2  Riječ-nositeljica: za svaku riječ (po snazi) njezine vlastite derivacije (najjače prve), dopuna
    iz vrha bazena pravilom rotacije (najmanje korišten od 20 najjačih brojeva, pa jači, pa manji).
R3  Parovi 8 najjačih hebrejskih riječi: 3 najjače derivacije svake + dopuna rotacijom.
R4  Jedna cifra: za svaku cifru 6 najjačih riječi po toj cifri (identitet/mod 45), pa varijanta sa
    zbrojem znamenki.
R5  Slova riječi: vrijednosti pojedinih slova (mod 45) 8 najjačih riječi + dopuna.
R6  Datum: hebrejski datum, gregorijanski datum i mješoviti skup.
R7  Trojke 6 najjačih riječi: 2 najjače derivacije svake.
Dvostruki skupovi se preskaču; brojevi su uvijek 6 različitih iz 1..45. Metoda nema slučajnosti.
"""
from __future__ import annotations

import itertools
import json
from collections import defaultdict
from pathlib import Path

SLUG = "gematrija"
METODA = "Gematrija"
DATUM = "2026-10-29"           # četvrtak, 20:00 CET
HEB_DATUM_FALLBACK = (5787, 8, 18)   # (godina, mjesec 8 = hešvan, dan) — građanski datum 29.10.2026.
OUT = Path(__file__).resolve().parent.parent / "out" / f"ezo_{SLUG}.json"

# ------------------------------------------------------------------ hebrejski datum
def hebrejski_datum() -> tuple[tuple[int, int, int], str]:
    """Vraća (godina, mjesec, dan) hebrejskog građanskog datuma za 29.10.2026. i izvor."""
    try:
        from pyluach import dates  # type: ignore
        h = dates.GregorianDate(2026, 10, 29).to_heb()
        return (h.year, h.month, h.day), "pyluach"
    except Exception:  # biblioteka nije dostupna → unaprijed izračunato
        return HEB_DATUM_FALLBACK, "unaprijed izračunato (fallback)"


# ------------------------------------------------------------------ abeceda i cifre
BASE = "אבגדהוזחטיכלמנסעפצקרשת"            # 22 osnovna slova redom
FINALS = {"ך": "כ", "ם": "מ", "ן": "נ", "ף": "פ", "ץ": "צ"}
_HV = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400]

HECHRECHI = dict(zip(BASE, _HV))
HECHRECHI.update({f: HECHRECHI[b] for f, b in FINALS.items()})
GADOL = dict(HECHRECHI)
GADOL.update({"ך": 500, "ם": 600, "ן": 700, "ף": 800, "ץ": 900})


def _katan_slovo(v: int) -> int:
    while v > 9 and v % 10 == 0:
        v //= 10
    return v


KATAN = {c: _katan_slovo(v) for c, v in HECHRECHI.items()}
SIDURI = dict(zip(BASE, range(1, 23)))
SIDURI.update({f: SIDURI[b] for f, b in FINALS.items()})
_ATBASH_MAP = {BASE[i]: BASE[21 - i] for i in range(22)}
_ATBASH_MAP.update({f: _ATBASH_MAP[b] for f, b in FINALS.items()})
ATBASH = {c: HECHRECHI[_ATBASH_MAP[c]] for c in HECHRECHI}

LATIN = {chr(64 + i): i for i in range(1, 27)}                  # A=1 … Z=26
LATIN_KATAN = {c: ((v - 1) % 9) + 1 for c, v in LATIN.items()}  # pitagorejska 1..9

# (ključ, naziv, težina, tablica)
CIFRE_HEB = [
    ("hechrechi", "mispar hechrechi", 1.0, HECHRECHI),
    ("gadol", "mispar gadol", 0.7, GADOL),
    ("katan", "mispar katan", 0.8, KATAN),
    ("siduri", "mispar siduri", 0.6, SIDURI),
    ("atbash", "atbash", 0.5, ATBASH),
]
CIFRE_LAT = [
    ("latin", "latinična A=1..Z=26", 0.8, LATIN),
    ("latin_katan", "pitagorejska 1..9", 0.5, LATIN_KATAN),
]


def vrijednost(rijec: str, tablica: dict) -> int:
    return sum(tablica[c] for c in rijec if c in tablica)


# ------------------------------------------------------------------ redukcije u 1..45
def mod45(v: int) -> int:
    return ((v - 1) % 45) + 1


def zbroj_znamenki(v: int) -> int:
    while v > 45:
        v = sum(int(d) for d in str(v))
    return v


def teozofska(v: int) -> int:
    while v > 9:
        v = sum(int(d) for d in str(v))
    return v


def redukcije(v: int) -> list[tuple[int, str, float]]:
    """Sve derivacije broja v u 1..45: (broj, oznaka, težina redukcije)."""
    out: list[tuple[int, str, float]] = []
    if 1 <= v <= 45:
        out.append((v, "identitet", 1.0))
    else:
        out.append((mod45(v), "mod 45", 0.8))
    if v >= 10:
        zz = zbroj_znamenki(v)
        if zz != v:
            out.append((zz, "zbroj znamenki", 0.7 if v > 45 else 0.4))
        t = teozofska(v)
        if t != zz and t != v:
            out.append((t, "teozofska", 0.25))
    return out


# ------------------------------------------------------------------ rječnik
# (hebrejski, translit/oznaka, značenje, snaga, skupina)
RIJECI_HEB = [
    ("מזל", "mazal", "sreća, zvijezda", 10, "ključna"),
    ("ברכה", "berakha", "blagoslov", 9, "ključna"),
    ("חי", "chai", "život", 9, "ključna"),
    ("הצלחה", "hatzlacha", "uspjeh", 8, "ključna"),
    ("זכייה", "zekhiya", "dobitak (na lutriji)", 8, "ključna"),
    ("שפע", "shefa", "obilje", 7, "blagostanje"),
    ("עושר", "osher", "bogatstvo", 7, "blagostanje"),
    ("טוב", "tov", "dobro", 7, "blagostanje"),
    ("חסד", "chesed", "milost, dobrota", 7, "blagostanje"),
    ("גורל", "goral", "ždrijeb, kocka, sudbina", 7, "blagostanje"),
    ("אושר", "osher (alef)", "radost, blaženstvo", 6, "blagostanje"),
    ("פרנסה", "parnasa", "uzdržavanje, prihod", 6, "blagostanje"),
    ("ברוך", "barukh", "blagoslovljen", 6, "blagostanje"),
    ("יהוה", "Tetragram", "Ime (26)", 6, "ime"),
    ("וה", 'ו"ה', "druga polovica Imena", 5, "ime"),
    ("צדק", "tzedek", "Jupiter (planetarni sat izvlačenja)", 5, "ime"),
]
RIJECI_DATUM = [
    ("יט", 'י"ט', "19. hešvan — datum nakon zalaska, u trenutku izvlačenja", 8, "datum"),
    ("יח", 'י"ח', "18. hešvan — građanski datum", 7, "datum"),
    ("תשפז", 'תשפ"ז', "godina 5787 bez tisućica", 8, "datum"),
    ("חשון", "hešvan", "mjesec", 7, "datum"),
    ("מרחשון", "marhešvan", "puno ime mjeseca", 4, "datum"),
]
RIJECI_LAT = [
    ("LOTO", "LOTO", "igra", 6, "latinica"),
    ("SRECA", "SRECA", "sreća", 6, "latinica"),
    ("LUTRIJA", "LUTRIJA", "Hrvatska Lutrija", 5, "latinica"),
    ("ZAGREB", "ZAGREB", "mjesto izvlačenja", 5, "latinica"),
    ("CETVRTAK", "CETVRTAK", "dan izvlačenja", 5, "latinica"),
    ("JUPITER", "JUPITER", "dan i sat Jupitera", 5, "latinica"),
]
SVE_RIJECI = RIJECI_HEB + RIJECI_DATUM + RIJECI_LAT

# gregorijanski datumski brojevi (sporedni sloj, samo za razinu R6)
DATUM_GREG = {"dan": 29, "mjesec": 10, "sat": 20, "godina": 26, "dan→": 11, "zbroj znamenki datuma": 22,
              "kolo": 42}  # 87. kolo → 87 mod 45 = 42


# ------------------------------------------------------------------ derivacije i bazen
def derivacije() -> list[dict]:
    """Sve derivacije svih riječi: {n, rijec, translit, cifra, vrijednost, redukcija, tezina}."""
    out: list[dict] = []
    for heb, tr, zn, snaga, grupa in SVE_RIJECI:
        cifre = CIFRE_LAT if grupa == "latinica" else CIFRE_HEB
        for kljuc, naziv, tc, tab in cifre:
            v = vrijednost(heb, tab)
            for n, red, tr_w in redukcije(v):
                out.append(dict(n=n, rijec=heb, translit=tr, grupa=grupa, snaga=snaga,
                                cifra=kljuc, cifra_naziv=naziv, vrijednost=v, redukcija=red,
                                tezina=round(snaga * tc * tr_w, 4)))
    return out


def slova(heb: str) -> list[int]:
    """Standardne vrijednosti pojedinih slova riječi svedene mod 45 (redom, bez ponavljanja)."""
    out: list[int] = []
    for c in heb:
        v = mod45(HECHRECHI[c]) if c in HECHRECHI else LATIN.get(c, 0)
        if v and v not in out:
            out.append(v)
    return out


def bazen(derivs: list[dict] | None = None) -> list[tuple[int, float]]:
    """Rang brojeva 1..45 po zbroju težina derivacija (silazno, pa manji broj)."""
    derivs = derivs if derivs is not None else derivacije()
    sc: dict[int, float] = defaultdict(float)
    for d in derivs:
        sc[d["n"]] += d["tezina"]
    return sorted(((n, round(s, 3)) for n, s in sc.items()), key=lambda t: (-t[1], t[0]))


def potpis(heb: str, derivs: list[dict], bez_teozofske: bool = True) -> list[int]:
    """Vlastite derivacije riječi, najjače prve, bez ponavljanja."""
    ds = [d for d in derivs if d["rijec"] == heb and not (bez_teozofske and d["redukcija"] == "teozofska")]
    ds.sort(key=lambda d: (-d["tezina"], d["n"]))
    out: list[int] = []
    for d in ds:
        if d["n"] not in out:
            out.append(d["n"])
    return out


# ------------------------------------------------------------------ gradnja kandidata
class _Graditelj:
    def __init__(self, vrh: list[int]):
        self.vrh = vrh                      # 20 najjačih brojeva bazena (rotacija dopune)
        self.score = {n: s for n, s in bazen()}
        self.uporaba: dict[int, int] = defaultdict(int)
        self.vidjeno: set[tuple] = set()
        self.lista: list[dict] = []

    def dopuni(self, jezgra: list[int]) -> tuple[list[int], list[int]]:
        """Dopuni jezgru do 6 brojeva pravilom rotacije: najmanje korišten, pa jači, pa manji."""
        br = list(dict.fromkeys(x for x in jezgra if 1 <= x <= 45))[:6]
        dop: list[int] = []
        while len(br) < 6:
            kand = [n for n in self.vrh if n not in br]
            if not kand:                       # (teoretski) — uzmi iz cijelog bazena
                kand = [n for n, _ in bazen() if n not in br]
            n = min(kand, key=lambda x: (self.uporaba[x], -self.score.get(x, 0), x))
            br.append(n)
            dop.append(n)
        return sorted(br), dop

    def dodaj(self, jezgra: list[int], razina: str, obraz: str) -> bool:
        br, dop = self.dopuni(jezgra)
        t = tuple(br)
        if t in self.vidjeno:
            return False
        self.vidjeno.add(t)
        for n in br:
            self.uporaba[n] += 1
        if dop:
            obraz += "; dopuna iz bazena: " + ", ".join(str(x) for x in dop)
        self.lista.append({"brojevi": br, "rang": len(self.lista) + 1,
                           "obrazlozenje": f"[{razina}] {obraz}"})
        return True


def _fmt(nums: list[int]) -> str:
    return ", ".join(str(x) for x in nums)


def kandidati(n: int = 80) -> list[dict]:
    """Vraća n (≥60) različitih kandidata po unutarnjoj logici gematrije."""
    derivs = derivacije()
    baz = bazen(derivs)
    vrh20 = [x for x, _ in baz[:20]]
    g = _Graditelj(vrh20)

    heb_po_snazi = sorted(RIJECI_HEB, key=lambda r: -r[3])
    sve_po_snazi = heb_po_snazi + sorted(RIJECI_DATUM, key=lambda r: -r[3]) + sorted(RIJECI_LAT, key=lambda r: -r[3])
    info = {r[0]: r for r in SVE_RIJECI}

    def opis(heb: str, k: int) -> str:
        r = info[heb]
        return f"{heb} {r[1]}={vrijednost(heb, LATIN if r[4] == 'latinica' else HECHRECHI)}→{_fmt(potpis(heb, derivs)[:k])}"

    # R1 — vrh bazena
    g.dodaj([x for x, _ in baz[:6]], "R1 vrh bazena",
            "6 najjačih brojeva ukupnog gematrijskog bazena: " + _fmt([x for x, _ in baz[:6]]))
    std = bazen([d for d in derivs if d["cifra"] in ("hechrechi", "latin") and d["redukcija"] in ("identitet", "mod 45")])
    g.dodaj([x for x, _ in std[:6]], "R1 standardna cifra",
            "vrh bazena samo standardne gematrije (identitet / mod 45)")
    klj = bazen([d for d in derivs if d["grupa"] == "ključna"])
    g.dodaj([x for x, _ in klj[:6]], "R1 ključne riječi",
            "vrh bazena samo ključnih riječi (mazal, berakha, chai, hatzlacha, zekhiya)")

    # R2 — riječ-nositeljica
    for heb, tr, zn, snaga, grupa in sve_po_snazi:
        p = potpis(heb, derivs)
        g.dodaj(p[:6], "R2 riječ", f"{opis(heb, 6)} ({zn})")

    # R3 — parovi 8 najjačih hebrejskih riječi
    top8 = [r[0] for r in heb_po_snazi[:8]]
    parovi = sorted(itertools.combinations(top8, 2),
                    key=lambda p: (-(info[p[0]][3] + info[p[1]][3]), top8.index(p[0]), top8.index(p[1])))
    for a, b in parovi:
        pa, pb = potpis(a, derivs)[:3], potpis(b, derivs)[:3]
        g.dodaj(pa + pb, "R3 par", f"{opis(a, 3)} + {opis(b, 3)}")

    # R4 — jedna cifra kroz sve riječi
    for kljuc, naziv, tc, tab in CIFRE_HEB:
        prim, sek = [], []
        for heb, tr, zn, snaga, grupa in heb_po_snazi + sorted(RIJECI_DATUM, key=lambda r: -r[3]):
            v = vrijednost(heb, tab)
            prim.append(mod45(v))
            sek.append(zbroj_znamenki(v))
        g.dodaj(list(dict.fromkeys(prim))[:6], "R4 cifra", f"{naziv}: vrijednosti riječi po snazi (identitet/mod 45)")
        g.dodaj(list(dict.fromkeys(sek))[:6], "R4 cifra", f"{naziv}: zbroj znamenki vrijednosti riječi po snazi")
    for kljuc, naziv, tc, tab in CIFRE_LAT:
        prim = [mod45(vrijednost(w, tab)) for w, *_ in sorted(RIJECI_LAT, key=lambda r: -r[3])]
        g.dodaj(list(dict.fromkeys(prim))[:6], "R4 cifra", f"{naziv}: LOTO, SRECA, LUTRIJA, ZAGREB, CETVRTAK, JUPITER")

    # R5 — slova riječi
    for heb, tr, zn, snaga, grupa in heb_po_snazi[:8]:
        s = slova(heb)
        g.dodaj(s, "R5 slova", f"slova riječi {heb} ({tr}): {'·'.join(str(x) for x in s)}")

    # R6 — datum
    hd, izvor = hebrejski_datum()
    g.dodaj([19, 18, 22, 4, 13, 26], "R6 datum",
            f'hebrejski datum י"ט/י"ח חשון תשפ"ז ({hd[2]}. hešvan {hd[0]}, {izvor}): 19, 18, 22 (תשפ"ז), 4 i 13 (חשון), 26 (יהוה=2026)')
    g.dodaj([29, 10, 20, 26, 11, 22], "R6 datum", "gregorijanski datum 29.10.2026. 20h: 29, 10, 20, 26, 11 (2+9), 22 (zbroj znamenki)")
    g.dodaj([22, 14, 18, 19, 29, 42], "R6 datum", "podudaranja: 22 (תשפ\"ז=datum), 14 (מזל=גורל=צדק), 18 (חי=י\"ח), 19 (י\"ט), 29 (dan), 42 (87. kolo mod 45)")
    g.dodaj([4, 24, 19, 39, 22, 16], "R6 datum", "mjesec hešvan kroz cifre: 4 (364 mod 45), 24 (gadol 1014), 19/39 (מרחשון), 22 (katan), 16 (atbash 151)")

    # R7 — trojke 6 najjačih riječi
    top6 = [r[0] for r in heb_po_snazi[:6]]
    for a, b, c in itertools.combinations(top6, 3):
        j = potpis(a, derivs)[:2] + potpis(b, derivs)[:2] + potpis(c, derivs)[:2]
        g.dodaj(j, "R7 trojka", f"{opis(a, 2)} + {opis(b, 2)} + {opis(c, 2)}")

    # R8 — rezerva: pomaknute derivacije (2.–4. derivacija parova) ako treba još kandidata
    if len(g.lista) < n:
        for a, b in parovi:
            pa, pb = potpis(a, derivs)[1:4], potpis(b, derivs)[1:4]
            g.dodaj(pa + pb, "R8 par (slabije derivacije)", f"{opis(a, 4)} + {opis(b, 4)} — 2.–4. derivacija")
            if len(g.lista) >= n:
                break

    return g.lista[:n]


def kljucni_brojevi(k: int = 12) -> list[int]:
    return [x for x, _ in bazen()[:k]]


# ------------------------------------------------------------------ postupak (markdown)
def postupak_tekst() -> str:
    derivs = derivacije()
    baz = bazen(derivs)
    hd, izvor = hebrejski_datum()
    L = []
    L.append("# Gematrija — Loto 6/45, četvrtak 29.10.2026. 20:00 CET\n")
    L.append(f"Hebrejski datum: {hd[2]}. hešvan {hd[0]} (izvor: {izvor}); nakon zalaska sunca 16:47 CET "
             "počinje 19. hešvan, pa izvlačenje u 20:00 pada na **י\"ט חשון תשפ\"ז**. "
             "Dan i planetarni sat: Jupiter (צדק).\n")
    L.append("## Cifre\n")
    L.append("| cifra | težina | pravilo |\n|---|---|---|")
    L.append("| mispar hechrechi | 1,0 | א=1 … י=10, כ=20 … ק=100, ר=200, ש=300, ת=400; završna slova kao osnovna |")
    L.append("| mispar gadol | 0,7 | završna slova ך=500 ם=600 ן=700 ף=800 ץ=900 |")
    L.append("| mispar katan | 0,8 | svako slovo svedeno na 1..9 |")
    L.append("| mispar siduri | 0,6 | redni broj slova 1..22 |")
    L.append("| atbash | 0,5 | א↔ת, ב↔ש … pa standardna vrijednost |")
    L.append("| latinična | 0,8 / 0,5 | A=1..Z=26 / pitagorejska 1..9 |\n")
    L.append("## Svođenje u 1..45\n")
    L.append("identitet (1..45, težina 1,0); mod 45 s 0→45 (0,8); zbroj znamenki dok ne padne u 1..45 "
             "(0,7; 0,4 ako je vrijednost već ≤45); teozofska redukcija na 1..9 (0,25, samo za bazen). "
             "Snaga derivacije = snaga riječi × težina cifre × težina redukcije.\n")
    L.append("## Riječi i vrijednosti po ciframa (vrijednost → identitet/mod 45 / zbroj znamenki)\n")
    L.append("| riječ | translit | značenje | snaga | hechrechi | gadol | katan | siduri | atbash |\n|---|---|---|---|---|---|---|---|---|")
    for heb, tr, zn, snaga, grupa in SVE_RIJECI:
        if grupa == "latinica":
            v = vrijednost(heb, LATIN); vk = vrijednost(heb, LATIN_KATAN)
            L.append(f"| {heb} | {tr} | {zn} | {snaga} | {v} → {mod45(v)} / {zbroj_znamenki(v)} | – | {vk} (pitag.) | – | – |")
            continue
        cells = []
        for kljuc, naziv, tc, tab in CIFRE_HEB:
            v = vrijednost(heb, tab)
            r = f"{v}"
            if v > 45:
                r += f" → {mod45(v)} / {zbroj_znamenki(v)}"
            elif v >= 10:
                r += f" / {zbroj_znamenki(v)}" if zbroj_znamenki(v) != v else ""
            cells.append(r)
        L.append(f"| {heb} | {tr} | {zn} | {snaga} | " + " | ".join(cells) + " |")
    L.append("\n## Slova riječi (standardna vrijednost slova mod 45)\n")
    for heb, tr, *_ in RIJECI_HEB[:10]:
        L.append(f"- {heb} ({tr}): {' · '.join(str(x) for x in slova(heb))}")
    L.append("\n## Gematrijski bazen (zbroj snaga derivacija po broju)\n")
    L.append("| rang | broj | snaga | glavne derivacije |\n|---|---|---|---|")
    po_n: dict[int, list[dict]] = defaultdict(list)
    for d in derivs:
        po_n[d["n"]].append(d)
    for i, (num, s) in enumerate(baz, 1):
        ds = sorted(po_n[num], key=lambda d: -d["tezina"])[:4]
        txt = "; ".join(f"{d['rijec']} {d['cifra_naziv']} {d['vrijednost']} ({d['redukcija']})" for d in ds)
        L.append(f"| {i} | **{num}** | {s} | {txt} |")
    L.append("\n## Podudaranja za ovaj datum\n")
    L.append("- **22** = תשפ\"ז 787 (mod 45 **i** zbroj znamenki) = katan חשון = zbroj svih znamenki 29.10.2026.")
    L.append("- **14** = zbroj znamenki i katan riječi מזל = גורל 239 (mod 45) = צדק/Jupiter 194 (mod 45) = ZAGREB 59 (mod 45)")
    L.append("- **18** = חי = י\"ח = עושר 576 (zbroj znamenki) = JUPITER 99 (zbroj znamenki) = siduri חי")
    L.append("- **26** = יהוה = godina 2026; **17** = טוב = LOTO 62 (mod 45); **32** = מזל (mod 45 i siduri)")
    L.append("- **12** = אושר 507 (mod 45 i zbroj) = הצלחה 138 (zbroj) = ברוך 228 (zbroj); **19** = י\"ט (datum izvlačenja)")
    L.append("\n## Razine gradnje kandidata (rang)\n")
    L.append("1. **R1 vrh bazena** — 6 najjačih brojeva (ukupno; samo standardna cifra; samo ključne riječi).")
    L.append("2. **R2 riječ-nositeljica** — vlastite derivacije svake riječi po snazi, dopuna rotacijom 20 najjačih brojeva bazena "
             "(najmanje korišten → jači → manji).")
    L.append("3. **R3 parovi** 8 najjačih hebrejskih riječi — po 3 najjače derivacije.")
    L.append("4. **R4 jedna cifra** — vrijednosti riječi po jednoj cifri (identitet/mod 45, pa zbroj znamenki).")
    L.append("5. **R5 slova** — vrijednosti pojedinih slova 8 najjačih riječi.")
    L.append("6. **R6 datum** — hebrejski, gregorijanski i mješoviti datumski skup.")
    L.append("7. **R7 trojke** 6 najjačih riječi — po 2 najjače derivacije. (R8 rezerva: slabije derivacije parova.)")
    L.append("\nMetoda je potpuno deterministična (nema miješanja ni bacanja).")
    return "\n".join(L)


# ------------------------------------------------------------------ main
if __name__ == "__main__":
    txt = postupak_tekst()
    kand = kandidati(80)
    print(txt)
    print("\n## Prvih 10 kandidata\n")
    for k in kand[:10]:
        print(f"{k['rang']:3}. {' · '.join(f'{x:02d}' for x in k['brojevi'])}   {k['obrazlozenje']}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    data = {"metoda": METODA, "slug": SLUG, "postupak": txt,
            "kljucni_brojevi": kljucni_brojevi(12), "kandidati": kand}
    OUT.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nZapisano: {OUT}  (kandidata: {len(kand)})")
