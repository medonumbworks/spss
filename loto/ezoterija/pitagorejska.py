"""Pitagorejska numerologija za Loto 6/45 — izvlačenje četvrtak 29.10.2026. u 20:00 CET, Zagreb.

ŠTO JE METODA
-------------
Pitagorejska ("zapadna", "moderna") numerologija pripisuje slovima vrijednosti 1–9 po
položaju u abecedi (A=1 … I=9, J=1 … R=9, S=1 … Z=8) i svaki broj svodi zbrajanjem znamenki
na jednoznamenkasti korijen, OSIM majstorskih brojeva 11, 22 i 33 koji se ne reduciraju
(L. Dow Balliett "The Philosophy of Numbers", 1908; Juno Jordan "Numerology: The Romance in
Your Name", 1965; Goodwin "Numerology: The Complete Guide", 1981). Riječ ima tri broja:
IZRAZ (expression, sva slova), DUŠEVNI PORIV (soul urge, samoglasnici) i OSOBNOST
(personality, suglasnici). Datum ima UNIVERZALNU GODINU, MJESEC i DAN (Jordan). Uz tablicu
slova, pitagorejska škola nosi i matematičku baštinu samog Pitagore: tetraktis 1+2+3+4 = 10,
savršeni brojevi 6 i 28, kvadrati, trokutni brojevi i pitagorejske trojke a²+b²=c².

PRAVILA
  1 2 3 4 5 6 7 8 9
  A B C D E F G H I
  J K L M N O P Q R
  S T U V W X Y Z
  Č,Ć→C  Š→S  Ž→Z  Đ→D (hrvatski dijakritici kao osnovno slovo; LJ, NJ, DŽ = dva slova).
  Samoglasnici A E I O U (Y samo ako nema susjedni samoglasnik — u riječima dana nema Y).
  Redukcija: zbrajaj znamenke dok broj nije 1..9 ILI 11/22/33 (majstorski se ne reducira);
  "korijen" = potpuna redukcija na 1..9 (11→2, 22→4, 33→6).

POSTUPAK ZA 29.10.2026. 20:00 (četvrtak, sat Jupitera, Zagreb) — sažeto; cijeli izvod s
tablicama generira postupak_tekst() i zapisuje se u out/ezo_pitagorejska.json.

1. Znamenke datuma i sata
   dan 29 → 2+9 = 11 (MAJSTORSKI) → korijen 2
   mjesec 10 → 1
   godina 2026 → 2+0+2+6 = 10 → 1;  '26 → 8
   puni zbroj 2+9+1+0+2+0+2+6 = 22 (MAJSTORSKI) → korijen 4
   sat 20:00 → 20 → 2;  pravo sunčevo vrijeme 20:20 → 2+0+2+0 = 4
   univerzalna godina 2026 = 1; univerzalni mjesec = 1 + 10 = 11/2;
   univerzalni dan = 11 + 29 = 40 → 4  (isto kao puni zbroj 22 → 4; oba se dokumentiraju)
   kolo 87 → 8+7 = 15 → 6
   četvrtak = 4. dan tjedna (ponedjeljak = 1) ili 5. dan (nedjelja = 1) — oba zabilježena
   dan+mjesec 29+10 = 39 → 12 → 3;  dan−sat 29−20 = 9
2. Riječi (izraz / duševni poriv / osobnost; nereducirano → reducirano)
   ČETVRTAK         C3 E5 T2 V4 R9 T2 A1 K2 = 28→1 | samogl. E5 A1 = 6      | sugl. 22 (MAJSTORSKI)→4
   LISTOPAD         L3 I9 S1 T2 O6 P7 A1 D4 = 33 (MAJSTORSKI)→6 | I9 O6 A1 = 16→7 | L3 S1 T2 P7 D4 = 17→8
   LOTO             L3 O6 T2 O6 = 17→8      | O6 O6 = 12→3           | L3 T2 = 5
   HRVATSKA LUTRIJA 28 + 28 = 56 → 11 (MAJSTORSKI)→2 | A1 A1 U3 I9 A1 = 15→6 | 26+15 = 41→5
   ZAGREB           Z8 A1 G7 R9 E5 B2 = 32→5 | A1 E5 = 6             | Z8 G7 R9 B2 = 26→8
   JUPITER          J1 U3 P7 I9 T2 E5 R9 = 36→9 | U3 I9 E5 = 17→8    | J1 P7 T2 R9 = 19→1
   SREĆA            S1 R9 E5 C3 A1 = 19→1  | E5 A1 = 6              | S1 R9 C3 = 13→4
   Zanimljivosti: tri majstorska broja izviru iz riječi (LISTOPAD 33, ČETVRTAK-suglasnici 22,
   HRVATSKA LUTRIJA 56→11) i poklapaju se s majstorskim brojevima datuma (29→11, fadik 22);
   ČETVRTAK, HRVATSKA i LUTRIJA sve daju 28 (savršeni broj); ČETVRTAK, ZAGREB i SREĆA imaju
   isti duševni poriv 6; JUPITER 36 = 6² i duševni poriv 17 = LOTO.
3. Majstorska algebra (u 1..45): 11+11=22, 11+22=33, 11+33=22·2=11·4=44, 11+2=13,
   22+2=24, 22+4=26, 33+2=35, 33+4=37, 29+11=40, 11²=121→4, 22²=484→16, 33²=1089→18→9,
   29²=841→13→4, 2²=4, 4²=16, 6²=36 (JUPITER).
4. Pitagorejske matematičke rezonancije: 20²+21²=400+441=841=29² — SAT i DAN su katete
   pitagorejske trojke (20, 21, 29) s hipotenuzom 29! Tetraktis 10 (= mjesec), savršeni 6 i
   28 (= ČETVRTAK), kvadrati 1 4 9 16 25 36, trokutni 1 3 6 10 15 21 28 36 45.
5. Redukcijski razredi ključnih korijena: 2 → {2, 11, 20, 29, 38}; 4 → {4, 13, 22, 31, 40};
   1 → {1, 10, 19, 28, 37}; 6 → {6, 15, 24, 33, 42}; 9 → {9, 18, 27, 36, 45}.
6. Bazen: svaki broj 1..45 zbraja težine svih izvora (datum 2–14, majstorski 6, riječi 1–6,
   algebra 1–4, pitagorejske figure 1–4, razredi 0.5–2).
7. Kandidati: (a) 14 KANONSKIH skupova = izravne derivacije, od najjače (jezgra datuma s
   majstorskima) prema slabijoj (figurativni brojevi); (b) zatim svi 6-člani podskupovi 33
   najtežih brojeva bazena rangirani po REZONANCIJI s datumom:
      rezonancija(S) = Σ težina(x) + 3·#majstorskih(11,22,33 u S) + 1.5·#datumskih(S)
                       + bonus zbroja (zbroj(S) reducira na 11/22/33: +4; na 2, 4 ili 1: +2)
                       + bonus pitagorejske trojke ((20,21,29) u S: +3; druga trojka: +1),
   uz raznolikost (najviše 3 zajednička broja sa svakim već prihvaćenim skupom, penal 0.9 po
   svakom prijašnjem korištenju broja). Metoda je čista aritmetika — nema slučajnosti.
"""
from __future__ import annotations

import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

SLUG = "pitagorejska"
METODA = "Pitagorejska numerologija"
DATUM_TEKST = "četvrtak 29.10.2026. u 20:00 CET (sat Jupitera), Zagreb"
OUT_DIR = Path(__file__).resolve().parent.parent / "out"

# --------------------------------------------------------------------------- tablica slova
_ABC = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
SLOVO = {c: (i % 9) + 1 for i, c in enumerate(_ABC)}          # A=1 … I=9, J=1 … R=9, S=1 … Z=8
_HR = {"Č": "C", "Ć": "C", "Š": "S", "Ž": "Z", "Đ": "D"}
SAMOGLASNICI = set("AEIOU")
MAJSTORSKI = (11, 22, 33)

ZNACENJE = {
    1: "monada — početak, volja, vodstvo", 2: "dijada — par, suradnja, Mjesec",
    3: "trijada — stvaranje, izraz", 4: "tetrada — temelj, red, rad",
    5: "pentada — promjena, sloboda", 6: "heksada — sklad, dom, savršeni broj",
    7: "heptada — mudrost, analiza", 8: "ogdoada — moć, novac, žetva",
    9: "eneada — dovršetak, humanost", 11: "MAJSTORSKI — intuicija, prosvjetljenje",
    22: "MAJSTORSKI — graditelj, ostvarenje", 33: "MAJSTORSKI — učitelj, služenje",
}


def normiraj(rijec: str) -> str:
    """Velika slova, hrvatski dijakritici na osnovno slovo, ostalo (razmaci, crtice) se izbacuje."""
    r = rijec.upper()
    for a, b in _HR.items():
        r = r.replace(a, b)
    return "".join(c for c in r if c in SLOVO)


def zbroj_znamenki(n: int) -> int:
    return sum(int(c) for c in str(n))


def reduciraj(n: int) -> int:
    """Pitagorejska redukcija: zbrajaj znamenke dok nije 1..9 ili majstorski 11/22/33."""
    while n > 9 and n not in MAJSTORSKI:
        n = zbroj_znamenki(n)
    return n


def korijen(n: int) -> int:
    """Potpuna redukcija na 1..9 (11→2, 22→4, 33→6)."""
    while n > 9:
        n = zbroj_znamenki(n)
    return n


def lanac(n: int) -> str:
    """Tekst redukcije, npr. '2026 → 10 → 1' ili '29 → 11 → 2'."""
    koraci = [n]
    while koraci[-1] > 9:
        koraci.append(zbroj_znamenki(koraci[-1]))
    return " → ".join(str(k) for k in koraci)


def je_samoglasnik(norm: str, i: int) -> bool:
    c = norm[i]
    if c in SAMOGLASNICI:
        return True
    if c == "Y":  # Y je samoglasnik samo bez susjednog samoglasnika
        lijevo = norm[i - 1] in SAMOGLASNICI if i > 0 else False
        desno = norm[i + 1] in SAMOGLASNICI if i + 1 < len(norm) else False
        return not (lijevo or desno)
    return False


def analiza_rijeci(rijec: str) -> dict:
    """Izraz (sva slova), duševni poriv (samoglasnici), osobnost (suglasnici)."""
    n = normiraj(rijec)
    slova = [(c, SLOVO[c], je_samoglasnik(n, i)) for i, c in enumerate(n)]
    izraz = sum(v for _, v, _ in slova)
    dusa = sum(v for _, v, s in slova if s)
    osoba = sum(v for _, v, s in slova if not s)
    return {"rijec": rijec, "slova": slova,
            "izraz": izraz, "izraz_red": reduciraj(izraz),
            "dusa": dusa, "dusa_red": reduciraj(dusa),
            "osoba": osoba, "osoba_red": reduciraj(osoba)}


# ----------------------------------------------------------------------------- ulazi datuma
# (broj, težina, oznaka, opis)
DATUM_BROJEVI = [
    (29, 14.0, "dan", "dan u mjesecu — najjači broj datuma"),
    (11, 10.0, "29→11", "dan 29 → 2+9 = 11, MAJSTORSKI"),
    (2, 5.0, "11→2", "korijen dana (11 → 2); sat 20 → 2"),
    (10, 8.0, "mjesec", "listopad = 10 (= tetraktis)"),
    (1, 5.0, "10→1", "korijen mjeseca; 2026 → 10 → 1 = univerzalna godina"),
    (22, 11.0, "fadik", "puni zbroj 2+9+1+0+2+0+2+6 = 22, MAJSTORSKI = univerzalni dan"),
    (4, 6.0, "22→4", "korijen fadika; četvrtak = 4. dan (pon=1); sunčevo vrijeme 20:20 → 4"),
    (20, 8.0, "sat", "20:00 (20 → 2)"),
    (26, 5.0, "'26", "zadnje dvije znamenke godine (26 → 8)"),
    (8, 2.0, "26→8", "korijen '26"),
    (40, 4.0, "univ. dan", "univerzalni dan = univ. mjesec 11 + dan 29 = 40 → 4"),
    (15, 4.0, "kolo 87", "87. kolo → 8+7 = 15 → 6"),
    (6, 3.0, "15→6", "korijen kola"),
    (5, 2.0, "četvrtak(5)", "četvrtak = 5. dan ako je nedjelja = 1"),
    (39, 3.0, "29+10", "dan + mjesec = 39 → 12 → 3"),
    (3, 2.0, "39→3", "korijen dana+mjeseca"),
    (9, 2.0, "29−20", "dan − sat = 9"),
]
DATUMSKI = {n for n, w, _, _ in DATUM_BROJEVI if w >= 4.0}      # primarni datumski brojevi

RIJECI = [  # (riječ, težina izraza, opis)
    ("ČETVRTAK", 6.0, "dan izvlačenja"),
    ("LISTOPAD", 6.0, "mjesec izvlačenja"),
    ("LOTO", 6.0, "ime igre"),
    ("HRVATSKA LUTRIJA", 6.0, "priređivač"),
    ("ZAGREB", 5.0, "mjesto"),
    ("JUPITER", 5.0, "planet dana i planetarnog sata (20:00 = 3. noćni sat)"),
    ("SREĆA", 5.0, "namjera"),
]
RIJECI_POMOCNE = [("HRVATSKA", 2.0, "priređivač, 1. riječ"), ("LUTRIJA", 2.0, "priređivač, 2. riječ")]
W_IZRAZ_RED, W_DUSA, W_DUSA_RED, W_OSOBA, W_OSOBA_RED, W_POMOCNA_RED = 1.5, 2.5, 1.0, 3.0, 1.0, 1.0
W_MAJSTORSKI = 6.0

# majstorska algebra: (broj, težina, oznaka)
ALGEBRA = [
    (44, 4.0, "11+33=22·2"), (13, 3.0, "11+2"), (24, 3.0, "22+2"), (26, 2.0, "22+4"),
    (35, 2.0, "33+2"), (37, 2.0, "33+4"), (40, 2.0, "29+11"), (33, 2.0, "11+22"),
    (22, 2.0, "11+11"), (16, 2.0, "22²→16=4²"), (4, 2.0, "11²→4=2²"), (13, 1.0, "29²→13"),
    (18, 1.0, "33²→18"), (36, 2.0, "6²"), (9, 1.0, "3²"), (25, 1.0, "5²"),
]
# pitagorejske figure
TROJKE = [t for t in itertools.combinations(range(1, 46), 3) if t[0] ** 2 + t[1] ** 2 == t[2] ** 2]
TROJKA_GLAVNA = (20, 21, 29)                       # sat² + 21² = dan²
FIGURE = [
    (21, 4.0, "20²+21²=29²"), (20, 2.0, "20²+21²=29²"), (29, 2.0, "20²+21²=29²"),
    (10, 2.0, "tetraktis"), (6, 2.0, "savršeni"), (28, 2.0, "savršeni"),
    (1, 1.0, "kvadrat"), (4, 1.0, "kvadrat"), (9, 1.0, "kvadrat"), (16, 1.0, "kvadrat"),
    (25, 1.0, "kvadrat"), (36, 1.0, "kvadrat"),
    (3, 1.0, "trokutni"), (15, 1.0, "trokutni"), (21, 1.0, "trokutni"), (28, 1.0, "trokutni"),
    (36, 1.0, "trokutni"), (45, 1.0, "trokutni"),
]
RAZREDI = {2: 2.0, 4: 2.0, 1: 1.5, 6: 1.0, 9: 0.5}   # korijen → težina razreda
KRATICA = {"HRVATSKA LUTRIJA": "HRV.LUTRIJA"}


# ----------------------------------------------------------------------------- bazen
def izgradi_bazen() -> tuple[dict[int, float], dict[int, list[tuple[str, float]]]]:
    tezina: dict[int, float] = defaultdict(float)
    izvori: dict[int, list[tuple[str, float]]] = defaultdict(list)

    def dodaj(n: int, w: float, oznaka: str):
        if 1 <= n <= 45 and w > 0:
            tezina[n] += w
            izvori[n].append((oznaka, w))

    for n, w, ozn, _ in DATUM_BROJEVI:
        dodaj(n, w, ozn)
    for m in MAJSTORSKI:
        dodaj(m, W_MAJSTORSKI, "majstorski")
    for rijec, w, _opis in RIJECI:                       # izraz + duševni poriv + osobnost
        a = analiza_rijeci(rijec)
        k = KRATICA.get(rijec, rijec)
        if a["izraz"] <= 45:
            dodaj(a["izraz"], w, k)
        else:                                            # 56 > 45 → samo reducirani (majstorski) 11
            dodaj(a["izraz_red"], w, f"{k}→{a['izraz_red']}")
        if a["izraz_red"] != a["izraz"]:
            dodaj(a["izraz_red"], W_IZRAZ_RED, f"{k}→{a['izraz_red']}")
        dodaj(a["dusa"], W_DUSA, f"{k}-duša")
        if a["dusa_red"] != a["dusa"]:
            dodaj(a["dusa_red"], W_DUSA_RED, f"{k}-duša→{a['dusa_red']}")
        dodaj(a["osoba"], W_OSOBA, f"{k}-osob.")
        if a["osoba_red"] != a["osoba"]:
            dodaj(a["osoba_red"], W_OSOBA_RED, f"{k}-osob.→{a['osoba_red']}")
    for rijec, w, _opis in RIJECI_POMOCNE:               # samo izraz (dijelovi cijelog naziva)
        a = analiza_rijeci(rijec)
        dodaj(a["izraz"], w, rijec)
        if a["izraz_red"] != a["izraz"]:
            dodaj(a["izraz_red"], W_POMOCNA_RED, f"{rijec}→{a['izraz_red']}")
    for n, w, ozn in ALGEBRA:
        dodaj(n, w, ozn)
    for n, w, ozn in FIGURE:
        dodaj(n, w, ozn)
    for n in range(1, 46):
        k = korijen(n)
        if k in RAZREDI:
            dodaj(n, RAZREDI[k], f"razred {k}")
    return dict(tezina), dict(izvori)


def oznaka_broja(n: int, izvori: dict, k: int = 1) -> str:
    lst = sorted(izvori.get(n, []), key=lambda t: -t[1])
    return f"{n}={'/'.join(o for o, _ in lst[:k])}" if lst else f"{n}"


# ----------------------------------------------------------------------------- rezonancija
B_MAJSTOR, B_DATUM, B_ZBROJ_MAJSTOR, B_ZBROJ_KORIJEN, B_TROJKA, B_TROJKA_DRUGA = 3.0, 1.5, 4.0, 2.0, 3.0, 1.0
KORIJENI_DATUMA = {2, 4, 1}


def bonus_zbroja(s: int) -> float:
    """Zbroj skupa reducira na majstorski (+4) ili na korijen datuma 2/4/1 (+2)."""
    n = s
    while n > 9:
        if n in MAJSTORSKI:
            return B_ZBROJ_MAJSTOR
        n = zbroj_znamenki(n)
    return B_ZBROJ_KORIJEN if n in KORIJENI_DATUMA else 0.0


def rezonancija(skup: list[int], tezina: dict[int, float]) -> float:
    s = set(skup)
    r = sum(tezina.get(x, 0.0) for x in skup)
    r += B_MAJSTOR * len(s & set(MAJSTORSKI)) + B_DATUM * len(s & DATUMSKI)
    r += bonus_zbroja(sum(skup))
    if set(TROJKA_GLAVNA) <= s:
        r += B_TROJKA
    r += B_TROJKA_DRUGA * sum(1 for t in TROJKE if t != TROJKA_GLAVNA and set(t) <= s)
    return round(r, 2)


# ----------------------------------------------------------------------------- kanonski skupovi
def kanonski() -> list[tuple[list[int], str]]:
    """Izravne derivacije — rang 1..14, od najjače prema slabijoj."""
    skupovi = [
        ([10, 11, 20, 22, 29, 33],
         "Jezgra datuma s majstorskima: dan 29, 29→11, mjesec 10, sat 20, fadik 22, LISTOPAD 33"),
        ([2, 4, 11, 22, 33, 44],
         "Majstorska trijada 11·22·33, korijeni 2 (dan) i 4 (fadik), 11+33 = 22·2 = 44"),
        ([4, 11, 20, 21, 22, 29],
         "Pitagorejska trojka 20²+21²=29² (sat i dan!) + 29→11, fadik 22, 22→4"),
        ([17, 19, 28, 32, 33, 36],
         "Izrazi riječi (nereducirani): LOTO 17, SREĆA 19, ČETVRTAK 28, ZAGREB 32, LISTOPAD 33, JUPITER 36"),
        ([1, 5, 8, 9, 11, 33],
         "Reducirani izrazi: ČETVRTAK/SREĆA→1, ZAGREB→5, LOTO→8, JUPITER→9, HRV. LUTRIJA 56→11, LISTOPAD 33"),
        ([6, 7, 12, 15, 16, 17],
         "Duševni porivi (samoglasnici): ČETVRTAK/ZAGREB/SREĆA 6, LOTO 12, HRV. LUTRIJA 15, LISTOPAD 16→7, JUPITER 17"),
        ([5, 13, 19, 22, 26, 41],
         "Osobnosti (suglasnici): LOTO 5, SREĆA 13, JUPITER 19, ČETVRTAK 22, ZAGREB 26, HRV. LUTRIJA 41 (LISTOPAD 17 u 6.)"),
        ([1, 11, 15, 20, 22, 40],
         "Univerzalni ciklusi: godina 2026→1, mjesec 1+10=11, dan 22 i 11+29=40, sat 20, kolo 87→15"),
        ([2, 4, 11, 20, 29, 38],
         "Redukcijski razred 2 (dan 29→11→2): 2, 11, 20, 29, 38 + korijen fadika 4"),
        ([4, 10, 13, 22, 31, 40],
         "Redukcijski razred 4 (univerzalni dan 22→4): 4, 13, 22, 31, 40 + mjesec 10"),
        ([1, 10, 19, 28, 29, 37],
         "Redukcijski razred 1 (mjesec 10→1, godina 2026→1): 1, 10, 19, 28, 37 + dan 29"),
        ([13, 24, 26, 35, 37, 40],
         "Majstorska algebra: 11+2=13, 22+2=24, 22+4=26, 33+2=35, 33+4=37, 29+11=40"),
        ([4, 6, 10, 16, 28, 36],
         "Pitagorejski sveti brojevi: tetraktis 10, savršeni 6 i 28 (=ČETVRTAK), kvadrati 4, 16 (=22²→16), 36 (=JUPITER)"),
        ([3, 15, 21, 28, 36, 45],
         "Trokutni brojevi 3, 15 (kolo), 21 (trojka), 28 (ČETVRTAK), 36 (JUPITER), 45 (najveći broj igre)"),
    ]
    out = []
    for nums, opis in skupovi:
        nums = sorted(set(nums))
        assert len(nums) == 6 and all(1 <= x <= 45 for x in nums), (nums, opis)
        out.append((nums, opis))
    return out


# ----------------------------------------------------------------------------- kandidati
POOL_N = 33            # 33 = najveći majstorski broj; C(33,6) = 1.107.568 podskupova
MAX_ZAJEDNICKI = 3
LAMBDA = 0.9


def kandidati(n: int = 80) -> list[dict]:
    """≥ 60 međusobno različitih kandidata; rang 1 = najjača derivacija."""
    n = max(n, 60)
    tezina, izvori = izgradi_bazen()
    prihvaceni: list[list[int]] = []
    rezultat: list[dict] = []

    def zbroj_opis(c):
        s = sum(c)
        return f"zbroj {lanac(s)}"

    for nums, opis in kanonski():
        if nums in prihvaceni:
            continue
        prihvaceni.append(nums)
        rezultat.append({"brojevi": nums, "rang": len(rezultat) + 1,
                         "obrazlozenje": f"Kanon: {opis} (rezonancija {rezonancija(nums, tezina):.1f}, {zbroj_opis(nums)})"})

    # enumeracija 6-podskupova POOL_N najtežih brojeva bazena, rang po rezonanciji + raznolikost
    pool = sorted(x for x, _ in sorted(tezina.items(), key=lambda t: (-t[1], t[0]))[:POOL_N])
    comb = np.array(list(itertools.combinations(pool, 6)), dtype=np.int16)      # (M, 6)
    idx = comb.astype(np.int64)
    w = np.zeros(46)
    for x, t in tezina.items():
        w[x] = t
    maj = np.zeros(46); maj[list(MAJSTORSKI)] = 1
    dat = np.zeros(46); dat[list(DATUMSKI)] = 1
    tab_zbroj = np.array([bonus_zbroja(s) for s in range(0, 46 * 6)])
    score = (w[idx].sum(axis=1) + B_MAJSTOR * maj[idx].sum(axis=1) + B_DATUM * dat[idx].sum(axis=1)
             + tab_zbroj[idx.sum(axis=1)])
    for t in TROJKE:
        if all(x in pool for x in t):
            v = np.zeros(46); v[list(t)] = 1
            score += np.where(v[idx].sum(axis=1) == 3, B_TROJKA if t == TROJKA_GLAVNA else B_TROJKA_DRUGA, 0.0)
    usage = np.zeros(46)
    ok = np.ones(len(comb), dtype=bool)
    for c in prihvaceni:
        v = np.zeros(46); v[c] = 1
        ok &= v[idx].sum(axis=1) <= MAX_ZAJEDNICKI
        usage[c] += 1
    while len(rezultat) < n and ok.any():
        eff = np.where(ok, score - LAMBDA * usage[idx].sum(axis=1), -np.inf)
        best = int(np.argmax(eff))        # argmax = prvi po indeksu → deterministično
        c = [int(x) for x in comb[best]]
        v = np.zeros(46); v[c] = 1
        ok &= v[idx].sum(axis=1) <= MAX_ZAJEDNICKI
        usage[c] += 1
        prihvaceni.append(c)
        rezultat.append({"brojevi": c, "rang": len(rezultat) + 1,
                         "obrazlozenje": "Bazen (rezonancija %.1f, %s): %s" % (
                             float(score[best]), zbroj_opis(c),
                             " · ".join(oznaka_broja(x, izvori) for x in c))})
    return rezultat


def kljucni_brojevi(k: int = 12) -> list[int]:
    tezina, _ = izgradi_bazen()
    return [x for x, _ in sorted(tezina.items(), key=lambda t: (-t[1], t[0]))[:k]]


# ----------------------------------------------------------------------------- postupak (tekst)
def postupak_tekst() -> str:
    tezina, izvori = izgradi_bazen()
    L = []
    L.append(f"# {METODA} — {DATUM_TEKST}\n")
    L.append("## 1. Pitagorejska tablica slova\n")
    L.append("| 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    L.append("| A | B | C | D | E | F | G | H | I |")
    L.append("| J | K | L | M | N | O | P | Q | R |")
    L.append("| S | T | U | V | W | X | Y | Z |   |")
    L.append("\nČ,Ć→C, Š→S, Ž→Z, Đ→D; LJ, NJ, DŽ = dva slova. Samoglasnici A E I O U (Y samo bez susjednog samoglasnika).")
    L.append("Redukcija: zbrajaj znamenke dok broj nije 1..9 ili majstorski 11/22/33 (ne reducira se); korijen = potpuna redukcija na 1..9.\n")
    L.append("## 2. Znamenke datuma i sata 29.10.2026. 20:00\n")
    L.append("| izvor | broj | redukcija | značenje | težina |")
    L.append("|---|---:|---|---|---:|")
    for n, w, ozn, opis in DATUM_BROJEVI:
        L.append(f"| {ozn}: {opis} | **{n}** | {lanac(n)} | {ZNACENJE[reduciraj(n)]} | {w:g} |")
    L.append("")
    L.append("- Univerzalna godina 2026 → 2+0+2+6 = 10 → **1**; univerzalni mjesec = 1 + 10 = **11**/2; "
             "univerzalni dan = 11 + 29 = 40 → **4** — isto kao puni zbroj znamenki 2+9+1+0+2+0+2+6 = **22**/4 (oba izvoda vode na 4).")
    L.append("- Četvrtak je **4.** dan tjedna (ponedjeljak = 1) ili **5.** (nedjelja = 1); oba su u bazenu (4 jače, jer se poklapa s 22→4).")
    L.append("- 20:00 CET → 20 → 2; pravo sunčevo vrijeme 20:20 → 2+0+2+0 = 4; kolo 87 → 15 → 6.\n")
    L.append("## 3. Riječi → izraz / duševni poriv / osobnost\n")
    L.append("| riječ | slova | izraz | duševni poriv (samoglasnici) | osobnost (suglasnici) | težina |")
    L.append("|---|---|---|---|---|---:|")
    for rijec, w, opis in RIJECI + RIJECI_POMOCNE:
        a = analiza_rijeci(rijec)
        slova = " ".join(f"{c}{v}" for c, v, _ in a["slova"])
        L.append(f"| **{rijec}** ({opis}) | {slova} | {lanac(a['izraz'])} — {ZNACENJE[a['izraz_red']]} "
                 f"| {lanac(a['dusa'])} | {lanac(a['osoba'])} | {w:g} |")
    L.append("")
    L.append("Zanimljivosti: tri majstorska broja izviru iz riječi — LISTOPAD = **33**, suglasnici ČETVRTKA = **22**, "
             "HRVATSKA LUTRIJA = 56 → **11** — i poklapaju se s majstorskima datuma (29 → 11, fadik 22). "
             "ČETVRTAK, HRVATSKA i LUTRIJA sve daju 28 (savršeni broj). ČETVRTAK, ZAGREB i SREĆA imaju isti duševni poriv 6. "
             "JUPITER = 36 = 6², a njegov duševni poriv 17 = LOTO.\n")
    L.append("## 4. Majstorska algebra (zbrojevi, produkti, kvadrati u 1..45)\n")
    for n, w, ozn in ALGEBRA:
        L.append(f"- {ozn} = **{n}** (+{w:g})")
    L.append("")
    L.append("## 5. Pitagorejske matematičke rezonancije\n")
    L.append("- **20² + 21² = 400 + 441 = 841 = 29²** — sat izvlačenja (20) i dan (29) su kateta i hipotenuza pitagorejske trojke (20, 21, 29); 21 je 'skriveni' treći broj.")
    L.append("- Tetraktis 1+2+3+4 = **10** = mjesec; savršeni brojevi **6** i **28** (28 = ČETVRTAK = HRVATSKA = LUTRIJA).")
    L.append("- Kvadrati 1, 4, 9, 16, 25, 36 (36 = JUPITER; 16 = 22² → 484 → 16); trokutni 1, 3, 6, 10, 15, 21, 28, 36, 45.")
    L.append(f"- Sve pitagorejske trojke u 1..45 ({len(TROJKE)}): " + ", ".join(f"({a},{b},{c})" for a, b, c in TROJKE) + ".\n")
    L.append("## 6. Redukcijski razredi ključnih korijena\n")
    for k, w in RAZREDI.items():
        clan = [n for n in range(1, 46) if korijen(n) == k]
        L.append(f"- korijen **{k}** (+{w:g}): {clan}")
    L.append("")
    L.append("## 7. Bazen s težinama (silazno)\n")
    L.append("| broj | težina | redukcija | izvori |")
    L.append("|---:|---:|---|---|")
    for n, w in sorted(tezina.items(), key=lambda t: (-t[1], t[0])):
        src = ", ".join(f"{o} +{x:g}" for o, x in sorted(izvori[n], key=lambda t: -t[1])[:6])
        L.append(f"| {n} | {w:.2f} | {lanac(n)} | {src} |")
    L.append("")
    L.append("## 8. Pravila generiranja i rangiranja\n")
    L.append("1. Rang 1–14: KANONSKI skupovi — izravne derivacije, od najjače (jezgra datuma s majstorskima) prema slabijoj (figurativni brojevi).")
    L.append(f"2. Rang 15+: svi 6-člani podskupovi {POOL_N} najtežih brojeva bazena, rangirani po REZONANCIJI: "
             f"Σ težina + {B_MAJSTOR:g}·#majstorskih (11, 22, 33) + {B_DATUM:g}·#datumskih {sorted(DATUMSKI)} "
             f"+ bonus zbroja skupa (reducira na 11/22/33: +{B_ZBROJ_MAJSTOR:g}; na 2, 4 ili 1: +{B_ZBROJ_KORIJEN:g}) "
             f"+ pitagorejska trojka ((20,21,29): +{B_TROJKA:g}; druga: +{B_TROJKA_DRUGA:g}).")
    L.append(f"3. Raznolikost: prihvaćen samo skup s ≤ {MAX_ZAJEDNICKI} zajednička broja sa svakim već prihvaćenim; penal {LAMBDA} po svakom prijašnjem korištenju broja. Argmax je prvi po indeksu → deterministično.")
    L.append("4. Nema slučajnosti: pitagorejska numerologija je čista aritmetika nad slovima i znamenkama.\n")
    L.append("## 9. Kanonski skupovi\n")
    for i, (nums, opis) in enumerate(kanonski(), 1):
        L.append(f"{i}. `{' · '.join(f'{x:02d}' for x in nums)}` — {opis} (rezonancija {rezonancija(nums, tezina):.1f})")
    return "\n".join(L)


# ----------------------------------------------------------------------------- main
if __name__ == "__main__":
    # provjera statičkih tvrdnji iz docstringa
    a = {r: analiza_rijeci(r) for r, _, _ in RIJECI + RIJECI_POMOCNE}
    assert (a["ČETVRTAK"]["izraz"], a["ČETVRTAK"]["dusa"], a["ČETVRTAK"]["osoba"]) == (28, 6, 22)
    assert (a["LISTOPAD"]["izraz"], a["LISTOPAD"]["dusa"], a["LISTOPAD"]["osoba"]) == (33, 16, 17)
    assert (a["LOTO"]["izraz"], a["LOTO"]["dusa"], a["LOTO"]["osoba"]) == (17, 12, 5)
    assert (a["HRVATSKA LUTRIJA"]["izraz"], a["HRVATSKA LUTRIJA"]["dusa"], a["HRVATSKA LUTRIJA"]["osoba"]) == (56, 15, 41)
    assert a["HRVATSKA LUTRIJA"]["izraz_red"] == 11 and a["HRVATSKA"]["izraz"] == 28 and a["LUTRIJA"]["izraz"] == 28
    assert (a["ZAGREB"]["izraz"], a["ZAGREB"]["dusa"], a["ZAGREB"]["osoba"]) == (32, 6, 26)
    assert (a["JUPITER"]["izraz"], a["JUPITER"]["dusa"], a["JUPITER"]["osoba"]) == (36, 17, 19)
    assert (a["SREĆA"]["izraz"], a["SREĆA"]["dusa"], a["SREĆA"]["osoba"]) == (19, 6, 13)
    assert reduciraj(29) == 11 and korijen(29) == 2 and reduciraj(2026) == 1 and reduciraj(22) == 22
    assert reduciraj(2 + 9 + 1 + 0 + 2 + 0 + 2 + 6) == 22 and reduciraj(87) == 6 and reduciraj(11 + 29) == 4
    assert 20 ** 2 + 21 ** 2 == 29 ** 2 and TROJKA_GLAVNA in TROJKE

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
