"""Tarot (Rider–Waite–Smith, 78 karata) za Loto 6/45 — izvlačenje četvrtak 29.10.2026. u 20:00 CET, Zagreb.

ŠTO JE METODA
-------------
Tarot Rider–Waite–Smith (A. E. Waite i Pamela Colman Smith, Rider & Son, London 1909/1910):
22 velike arkane (0 Luda, I Mag, II Visoka svećenica, III Carica, IV Car, V Jerofant,
VI Ljubavnici, VII Kola, VIII Snaga, IX Pustinjak, X Kolo sreće, XI Pravda, XII Obješeni,
XIII Smrt, XIV Umjerenost, XV Đavao, XVI Kula, XVII Zvijezda, XVIII Mjesec, XIX Sunce, XX Sud,
XXI Svijet) i 56 malih arkana u četiri boje — štapovi (vatra), pehari (voda), mačevi (zrak),
novčići (zemlja) — svaka As, 2–10, Paž, Vitez, Kraljica, Kralj. Špil se promiješa, karte se
okreću (jednostavno izvlačenje) ili polažu u raspored (spread). Loto-brojevi nastaju iz karata
dvama mapiranjima: (a) položaj karte u referentnom popisu špila 1–78, (b) broj same karte.
Obrnute karte se ne koriste (kao ni u referentnoj skripti).

IZVORI / PRAVILA
----------------
(a) REFERENTNO pravilo iz korisnikove skripte data/loto_ezoterija_original.py (rječnik RW i
    izvuci() za deck "RW"): popis špila 1..21 = velike arkane Mag..Svijet, 22 = Luda,
    23..36 = štapovi As..Kralj, 37..50 = pehari, 51..64 = mačevi, 65..78 = novčići.
    Špil [1..78] se promiješa i s vrha se uzimaju prve karte čiji je redni broj ≤ 45 dok ih
    nema 6 (karte 46–78, tj. 10 pehara … Kralj novčića, okreću se ali ne daju broj).
(b) ALTERNATIVNO mapiranje — broj karte: velika arkana = njezin broj (Luda = 0 → 22), mala
    arkana = rang (As = 1 … 10 = 10, Paž = 11, Vitez = 12, Kraljica = 13, Kralj = 14) + ofset
    boje (štapovi 0, pehari 10, mačevi 20, novčići 30) → 1..44. Isti promiješani špil kao (a):
    karte se okreću s vrha, svaka daje svoju vrijednost, duplikat (ista vrijednost, npr. I Mag
    i As štapova = 1) se preskače, prvih 6.
(c) KELTSKI KRIŽ (Waite, "The Pictorial Key to the Tarot", 1910; 10 karata). Po Waiteu se
    najprije bira SIGNIFIKATOR koji predstavlja pitanje i izdvaja iz špila: za ovaj trenutak to
    je X KOLO SREĆE (Jupiter vlada i danom — četvrtak — i 3. noćnim planetarnim satom u 20:00
    CET; Zlatna zora: Jupiter = Kolo sreće). Preostalih 77 karata se miješa i prvih 10 polaže:
    1 što ga pokriva (srž), 2 što ga križa, 3 što ga kruni (najbolji mogući ishod), 4 što je
    ispod (temelj), 5 što je iza (prošlost), 6 što je pred njim (bliska budućnost), 7 pitatelj,
    8 kuća/okolina, 9 nade i strahovi, 10 što će doći (ishod). Broj 10 (signifikator) je prvi
    broj; ostali po mapiranju (b) redom važnosti položaja 10 → 3 → 6 → 1 → 2 → 9 → 4 → 5 → 7
    → 8 dok nema 6 različitih; ako treba, dopuna rednim brojevima (a) ≤ 45 istih karata, pa
    vrijednostima karata s vrha ostatka.
(d) KOLO SREĆE (raspored od 6 karata u obliku kola; nov promiješani špil od 78 karata):
    1 osovina (što je stalno), 2 uspon (što dolazi), 3 vrh (vrhunac), 4 silazak (što odlazi),
    5 dno (temelj, skriveno), 6 okret (ishod — kamo se kolo okreće). Brojevi po mapiranju (b)
    redom 6 → 3 → 1 → 2 → 5 → 4; duplikat → dopuna rednim brojem (a) ≤ 45 položenih karata,
    pa vrijednostima sljedećih karata s vrha špila.
Miješanje je jedini izvor slučajnosti i vezano je uz datum: čitanje i koristi
rng = random.Random(20261029 + i), i = 0 … 99; iz istog rng-a redom nastaju tri miješanja
(špil za (a)+(b), špil od 77 karata za Keltski križ, špil za Kolo sreće). Nikad SystemRandom.

ASTROLOŠKE KORESPONDENCIJE (Zlatna zora, "Book T"; Liber 777) ZA 29.10.2026. 20:00 CET
--------------------------------------------------------------------------------------
Provjereno s ephem (body.compute(d, epoch=d) → ephem.Ecliptic): Sunce 6°22' Škorpiona,
Mjesec 25°41' Blizanaca, Merkur 18°48' Škorpiona, Venera 27°25' Vage, Mars 17°30' Lava,
Jupiter 24°03' Lava, Saturn 9°25' Ovna; Ascendent ~5° Raka, MC ~7° Riba; Mjesec 82,6 %
opadajući; četvrtak = dan Jupitera, 20:00 = 3. noćni sat = sat Jupitera.
  Jupiter (dan + sat)              → X Kolo sreće            težina 1,00
  Sunce u Škorpionu                → XIII Smrt (Škorpion)    0,60   i XIX Sunce (planet) 0,80
  Sunce 1. dekada Škorpiona        → 5 pehara                0,30
  Mjesec u Blizancima              → VI Ljubavnici           0,50   ; II Visoka svećenica 0,20
  Ascendent u Raku                 → VII Kola                0,60
  MC u Ribama                      → XVIII Mjesec            0,20
  Jupiter i Mars u Lavu            → VIII Snaga (RWS Lav)    0,50   ; XI Pravda (Thoth Lust = XI
                                     = Lav; RWS Pravda = Vaga, gdje je Venera)            0,40
  Jupiter 24° Lava = 3. dekada     → 7 štapova (Hrabrost)    0,40
  Mars 17° Lava = 2. dekada        → 6 štapova (Pobjeda; vladar dekade Jupiter)          0,30
  Venera 27° Vage = 3. dekada      → 4 mačeva (Primirje; vladar dekade Jupiter)          0,30
  Venera (planet)                  → III Carica              0,20
  Merkur 19° Škorpiona = 2. dekada → 6 pehara                0,20   ; I Mag (planet) 0,20
  Dvorske karte (segmenti od 20° do 20°): Sunce, Merkur i Venera u 20° Vage–20° Škorpiona →
  Princ (RWS Vitez) pehara 0,30; Jupiter u 20° Lava–20° Djevice → Vitez (RWS Kralj) novčića
  0,20; Mars u 20° Raka–20° Lava → Princ (RWS Vitez) štapova 0,20; Mjesec u 20° Blizanaca–
  20° Raka → Kraljica pehara 0,20.
  Karte sreće/novca (zadane): XIX Sunce 0,80, XVII Zvijezda 0,70, As novčića 0,80,
  10 novčića 0,80, 9 novčića 0,70; XXI Svijet (dovršenje) 0,30.

RANGIRANJE
----------
Ocjena izvedbe = Σ težina(karta) × množitelj(položaj) po svim OKRENUTIM kartama te izvedbe.
Množitelj: (a)/(b) karta koja je dala broj 1,0, okrenuta a preskočena (> 45 ili duplikat) 0,5;
Keltski križ po položaju: 10 ishod 1,5, 3 kruna 1,3, 6 budućnost 1,2, 1 srž 1,0, 2 križ 0,9,
9 nade 0,9, 4 temelj 0,8, 7 pitatelj 0,8, 8 okolina 0,8, 5 prošlost 0,6; Kolo sreće: 6 okret
1,5, 3 vrh 1,3, 1 osovina 1,2, 2 uspon 1,0, 5 dno 0,8, 4 silazak 0,6, dopunske karte 0,5.
Kandidat = težina izvedbe × (1 + ocjena izvedbe); težine: (a) referentno pravilo 1,00,
(b) broj karte 0,95, (c) Keltski križ 0,90, (d) Kolo sreće 0,85. Nerješeno → manji i, pa
redoslijed izvedbi. Raznolikost: kandidat koji s već prihvaćenim dijeli 5 ili 6 brojeva se
preskače. Nema nikakva statističkog filtriranja.

POSTUPAK ZA 29.10.2026. 20:00 (sažeto; cijeli izvod s prvih 10 čitanja, svim okrenutim
kartama, položajima Keltskog križa i Kola sreće generira postupak_tekst() i zapisuje se u
out/ezo_tarot.json)
1. seed_i = 20261029 + i (datum 29.10.2026. kao 20261029), i = 0 … 99; rng = random.Random(seed_i).
2. d1 = [1..78], rng.shuffle(d1) → (a) prve karte ≤ 45 do 6; (b) vrijednosti s vrha do 6 različitih.
3. d2 = [1..78] bez 10 (signifikator Kolo sreće), rng.shuffle(d2) → Keltski križ d2[0..9].
4. d3 = [1..78], rng.shuffle(d3) → Kolo sreće d3[0..5] (+ dopuna d3[6..]).
5. Ocjena svake izvedbe po ključnim kartama; sortiranje po težina × (1 + ocjena); raznolikost.
6. Ključni brojevi = brojevi astroloških i sretnih karata po mapiranju (b) i, gdje je ≤ 45,
   po (a): 10, 19, 17, 31, 40, 39, 13, 7, 6, 8, 11, 15, 24, 29, 28, 41 + najčešći među prvih
   30 kandidata.
"""
from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

SLUG = "tarot"
METODA = "Tarot"
DATUM_TEKST = "četvrtak 29.10.2026. u 20:00 CET (dan i 3. noćni planetarni sat Jupitera), Zagreb"
SEED0 = 20261029
N_CITANJA = 100
OUT_DIR = Path(__file__).resolve().parent.parent / "out"

# --------------------------------------------------------------------------- špil (referentni popis 1..78)
VELIKE = ["Mag", "Visoka svećenica", "Carica", "Car", "Jerofant", "Ljubavnici", "Kola", "Snaga",
          "Pustinjak", "Kolo sreće", "Pravda", "Obješeni", "Smrt", "Umjerenost", "Đavao", "Kula",
          "Zvijezda", "Mjesec", "Sunce", "Sud", "Svijet"]
RIMSKI = ["0", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII", "XIII",
          "XIV", "XV", "XVI", "XVII", "XVIII", "XIX", "XX", "XXI"]
BOJE = ["štapova", "pehara", "mačeva", "novčića"]
RANGOVI = ["As", "2", "3", "4", "5", "6", "7", "8", "9", "10", "Paž", "Vitez", "Kraljica", "Kralj"]

RW: dict[int, str] = {i + 1: VELIKE[i] for i in range(21)}
RW[22] = "Luda"
for _si, _s in enumerate(BOJE):
    for _r, _x in enumerate(RANGOVI):
        RW[23 + _si * 14 + _r] = f"{_x} {_s}"
assert len(RW) == 78 and RW[23] == "As štapova" and RW[45] == "9 pehara" and RW[78] == "Kralj novčića"

SIGNIFIKATOR = 10          # X Kolo sreće — Jupiterov dan i sat


def vrijednost(idx: int) -> int:
    """Mapiranje (b): velika arkana = broj (Luda 0 → 22), mala = rang + ofset boje → 1..44."""
    if idx <= 21:
        return idx
    if idx == 22:
        return 22
    si, r = divmod(idx - 23, 14)
    return r + 1 + 10 * si


def ime(idx: int) -> str:
    if idx <= 21:
        return f"{RIMSKI[idx]} {RW[idx]}"
    if idx == 22:
        return "0 Luda"
    return RW[idx]


def opis(idx: int) -> str:
    """Ime karte s oba broja: a = redni broj u popisu špila, b = broj karte."""
    return f"{ime(idx)} (a={idx}, b={vrijednost(idx)})"


# --------------------------------------------------------------------------- ključne karte (indeks → težina, razlog)
KLJUCNE_KARTE: dict[int, tuple[float, str]] = {
    10: (1.00, "Jupiter vlada danom (četvrtak) i 3. noćnim planetarnim satom → X Kolo sreće"),
    19: (0.80, "XIX Sunce — karta uspjeha i sreće (Sunce kao planet)"),
    65: (0.80, "As novčića — korijen bogatstva, novi dobitak"),
    74: (0.80, "10 novčića — Bogatstvo (3. dekada Djevice)"),
    17: (0.70, "XVII Zvijezda — nada (Vodenjak; Pluton 3° Vodenjaka)"),
    73: (0.70, "9 novčića — Dobitak (2. dekada Djevice)"),
    13: (0.60, "Sunce 6°22' Škorpiona → XIII Smrt (Škorpion)"),
    7: (0.60, "Ascendent ~5° Raka → VII Kola (Rak)"),
    6: (0.50, "Mjesec ~25° Blizanaca → VI Ljubavnici (Blizanci)"),
    8: (0.50, "Jupiter 24° i Mars 17° Lava → VIII Snaga (RWS Lav)"),
    11: (0.40, "Lav = XI Lust (Thoth) / Venera 27° Vage → XI Pravda (RWS Vaga)"),
    29: (0.40, "Jupiter 24° Lava = 3. dekada Lava → 7 štapova (Hrabrost)"),
    28: (0.30, "Mars 17° Lava = 2. dekada Lava → 6 štapova (Pobjeda; vladar dekade Jupiter)"),
    41: (0.30, "Sunce 6° Škorpiona = 1. dekada → 5 pehara"),
    54: (0.30, "Venera 27° Vage = 3. dekada → 4 mačeva (Primirje; vladar dekade Jupiter)"),
    48: (0.30, "Sunce, Merkur i Venera u 20° Vage–20° Škorpiona → Princ (RWS Vitez) pehara"),
    21: (0.30, "XXI Svijet — dovršenje, cjelina"),
    42: (0.20, "Merkur 19° Škorpiona = 2. dekada → 6 pehara"),
    18: (0.20, "MC ~7° Riba → XVIII Mjesec (Ribe)"),
    2: (0.20, "Mjesec kao planet → II Visoka svećenica"),
    3: (0.20, "Venera kao planet → III Carica"),
    1: (0.20, "Merkur kao planet → I Mag"),
    78: (0.20, "Jupiter 24° Lava u 20° Lava–20° Djevice → Vitez (RWS Kralj) novčića"),
    34: (0.20, "Mars 17° Lava u 20° Raka–20° Lava → Princ (RWS Vitez) štapova"),
    49: (0.20, "Mjesec 25° Blizanaca u 20° Blizanaca–20° Raka → Kraljica pehara"),
}
assert all(1 <= k <= 78 for k in KLJUCNE_KARTE)

MULT_BROJ, MULT_PRESKOCENA, MULT_DOPUNA = 1.0, 0.5, 0.5

# Keltski križ (Waite 1910)
KK_NAZIV: dict[int, str] = {
    1: "što ga pokriva (srž)", 2: "što ga križa", 3: "što ga kruni (najbolji ishod)",
    4: "što je ispod (temelj)", 5: "što je iza (prošlost)", 6: "što je pred njim (budućnost)",
    7: "pitatelj", 8: "kuća (okolina)", 9: "nade i strahovi", 10: "što će doći (ishod)",
}
KK_PRIORITET = [10, 3, 6, 1, 2, 9, 4, 5, 7, 8]
KK_MULT = {10: 1.5, 3: 1.3, 6: 1.2, 1: 1.0, 2: 0.9, 9: 0.9, 4: 0.8, 7: 0.8, 8: 0.8, 5: 0.6}

# Kolo sreće (6 karata)
KS_NAZIV: dict[int, str] = {
    1: "osovina (što je stalno)", 2: "uspon (što dolazi)", 3: "vrh (vrhunac)",
    4: "silazak (što odlazi)", 5: "dno (temelj, skriveno)", 6: "okret (ishod)",
}
KS_PRIORITET = [6, 3, 1, 2, 5, 4]
KS_MULT = {6: 1.5, 3: 1.3, 1: 1.2, 2: 1.0, 5: 0.8, 4: 0.6}

TEZINA_IZVEDBE: dict[str, float] = {"a": 1.00, "b": 0.95, "c": 0.90, "d": 0.85}
NAZIV_IZVEDBE: dict[str, str] = {
    "a": "referentno pravilo (redni broj ≤ 45)", "b": "broj karte (mapiranje b)",
    "c": "Keltski križ", "d": "Kolo sreće",
}


# --------------------------------------------------------------------------- izvedbe
def izvedba_a(d: list[int]) -> dict:
    """(a) Referentno pravilo: s vrha špila prve karte s rednim brojem ≤ 45 dok ih nema 6."""
    out: list[int] = []
    izvor: list[tuple[str, int, int]] = []
    log: list[str] = []
    okrenute: list[tuple[int, float]] = []
    for c in d:
        if c <= 45:
            out.append(c)
            izvor.append(("", c, c))
            okrenute.append((c, MULT_BROJ))
            log.append(f"{opis(c)} → {c}")
        else:
            okrenute.append((c, MULT_PRESKOCENA))
            log.append(f"{opis(c)} → redni broj > 45, preskočeno")
        if len(out) == 6:
            break
    return dict(brojevi=out, izvor=izvor, okrenute=okrenute, log=log)


def izvedba_b(d: list[int]) -> dict:
    """(b) Broj karte: s vrha špila, vrijednost svake karte, duplikat se preskače, prvih 6."""
    out: list[int] = []
    izvor: list[tuple[str, int, int]] = []
    log: list[str] = []
    okrenute: list[tuple[int, float]] = []
    for c in d:
        v = vrijednost(c)
        if v in out:
            okrenute.append((c, MULT_PRESKOCENA))
            log.append(f"{opis(c)} → {v} (već uzet, preskočeno)")
        else:
            out.append(v)
            izvor.append(("", c, v))
            okrenute.append((c, MULT_BROJ))
            log.append(f"{opis(c)} → {v}")
        if len(out) == 6:
            break
    return dict(brojevi=out, izvor=izvor, okrenute=okrenute, log=log)


def _spread(d: list[int], n_pol: int, naziv: dict[int, str], prioritet: list[int], mult: dict[int, float],
            fiksni: list[tuple[str, int, int]]) -> dict:
    """Zajednička logika rasporeda: položaji 1..n_pol = d[0..n_pol-1]; brojevi po (b) redom prioriteta,
    dopuna rednim brojevima (a) ≤ 45 položenih karata, pa vrijednostima karata s vrha ostatka."""
    polozaji = {p: d[p - 1] for p in range(1, n_pol + 1)}
    out: list[int] = [v for _, _, v in fiksni]
    izvor: list[tuple[str, int, int]] = list(fiksni)
    log: list[str] = [f"{lab}: {opis(c)} → {v} (zadan)" for lab, c, v in fiksni]
    okrenute: list[tuple[int, float]] = [(polozaji[p], mult[p]) for p in range(1, n_pol + 1)]
    for p in prioritet:
        if len(out) == 6:
            break
        c = polozaji[p]
        v = vrijednost(c)
        if v in out:
            log.append(f"položaj {p} {naziv[p]}: {opis(c)} → {v} (već uzet)")
        else:
            out.append(v)
            izvor.append((f"{p}", c, v))
            log.append(f"položaj {p} {naziv[p]}: {opis(c)} → {v}")
    if len(out) < 6:
        for p in prioritet:
            if len(out) == 6:
                break
            c = polozaji[p]
            if c <= 45 and c not in out:
                out.append(c)
                izvor.append((f"{p}a", c, c))
                log.append(f"dopuna (a) položaj {p}: {opis(c)} → {c}")
    k = n_pol
    while len(out) < 6:
        c = d[k]
        k += 1
        v = vrijednost(c)
        okrenute.append((c, MULT_DOPUNA))
        if v in out:
            log.append(f"dopuna s vrha ostatka: {opis(c)} → {v} (već uzet)")
        else:
            out.append(v)
            izvor.append(("+", c, v))
            log.append(f"dopuna s vrha ostatka: {opis(c)} → {v}")
    return dict(brojevi=out, izvor=izvor, okrenute=okrenute, log=log, polozaji=polozaji)


def keltski_kriz(d: list[int]) -> dict:
    """(c) Keltski križ: d je promiješani špil od 77 karata (bez signifikatora X Kolo sreće)."""
    assert SIGNIFIKATOR not in d and len(d) == 77
    return _spread(d, 10, KK_NAZIV, KK_PRIORITET, KK_MULT,
                   fiksni=[("signifikator", SIGNIFIKATOR, SIGNIFIKATOR)])


def kolo_srece(d: list[int]) -> dict:
    """(d) Kolo sreće: 6 karata s vrha promiješanog špila od 78 karata."""
    assert len(d) == 78
    return _spread(d, 6, KS_NAZIV, KS_PRIORITET, KS_MULT, fiksni=[])


# --------------------------------------------------------------------------- ocjena
def ocjena_izvedbe(okrenute: list[tuple[int, float]]) -> tuple[float, list[str]]:
    s = 0.0
    log: list[str] = []
    for c, m in okrenute:
        if c in KLJUCNE_KARTE:
            w, _ = KLJUCNE_KARTE[c]
            s += w * m
            log.append(f"{ime(c)} +{w:.2f}×{m:.1f}")
    return round(s, 4), log


# --------------------------------------------------------------------------- čitanje
def citanje(i: int) -> dict:
    """Jedno čitanje: seed 20261029 + i; tri miješanja iz istog rng-a; četiri izvedbe; ocjene."""
    seed = SEED0 + i
    rng = random.Random(seed)
    d1 = list(range(1, 79))
    rng.shuffle(d1)
    d2 = [c for c in range(1, 79) if c != SIGNIFIKATOR]
    rng.shuffle(d2)
    d3 = list(range(1, 79))
    rng.shuffle(d3)
    izvedbe = [
        dict(oznaka="a", **izvedba_a(d1)),
        dict(oznaka="b", **izvedba_b(d1)),
        dict(oznaka="c", **keltski_kriz(d2)),
        dict(oznaka="d", **kolo_srece(d3)),
    ]
    for iz in izvedbe:
        ocj, log_o = ocjena_izvedbe(iz["okrenute"])
        iz["ocjena_karata"] = ocj
        iz["ocjena_log"] = log_o
        iz["tezina"] = TEZINA_IZVEDBE[iz["oznaka"]]
        iz["ocjena"] = round(iz["tezina"] * (1.0 + ocj), 4)
    return dict(i=i, seed=seed, d1=d1, d2=d2, d3=d3, izvedbe=izvedbe)


def _obrazlozenje(ct: dict, iz: dict) -> str:
    o = iz["oznaka"]
    dijelovi = []
    for lab, c, v in iz["izvor"]:
        pref = "" if lab == "" else ("S " if lab == "signifikator" else f"p{lab} ")
        dijelovi.append(f"{pref}{ime(c)}={v}")
    kljuc = ", ".join(iz["ocjena_log"]) if iz["ocjena_log"] else "bez ključnih karata"
    return (f"Čitanje #{ct['i']} (seed {ct['seed']}), {NAZIV_IZVEDBE[o]}: " + ", ".join(dijelovi)
            + f"; ključne: {kljuc}; ocjena {iz['tezina']:.2f}×(1+{iz['ocjena_karata']:.2f})={iz['ocjena']:.2f}")


# --------------------------------------------------------------------------- kandidati
_CACHE: dict[int, dict] = {}


def sva_citanja() -> list[dict]:
    if not _CACHE:
        for i in range(N_CITANJA):
            _CACHE[i] = citanje(i)
    return [_CACHE[i] for i in range(N_CITANJA)]


def kandidati(n: int = 80, max_preklop: int = 4) -> list[dict]:
    """Rangirani kandidati: težina izvedbe × (1 + ocjena karata); raznolikost: ≤ max_preklop zajedničkih brojeva."""
    bazen = []
    for ct in sva_citanja():
        for k, iz in enumerate(ct["izvedbe"]):
            if len(iz["brojevi"]) == 6:
                bazen.append((-iz["ocjena"], ct["i"], k, ct, iz))
    bazen.sort(key=lambda t: (t[0], t[1], t[2]))
    out: list[dict] = []
    prihvaceni: list[set[int]] = []
    for _, _, _, ct, iz in bazen:
        t = sorted(iz["brojevi"])
        assert len(t) == 6 and len(set(t)) == 6 and all(1 <= x <= 45 for x in t), t
        st = set(t)
        if any(len(st & p) > max_preklop for p in prihvaceni):
            continue
        prihvaceni.append(st)
        out.append({"brojevi": t, "rang": len(out) + 1, "obrazlozenje": _obrazlozenje(ct, iz)})
        if len(out) >= n:
            break
    return out


# fiksni ključni brojevi: astrološke i sretne karte po (b) i, gdje je ≤ 45, po (a)
KLJUCNI_FIKSNI: list[tuple[int, str]] = [
    (10, "X Kolo sreće (Jupiterov dan i sat)"), (19, "XIX Sunce"), (17, "XVII Zvijezda"),
    (31, "As novčića (b)"), (40, "10 novčića (b)"), (39, "9 novčića (b)"),
    (13, "XIII Smrt — Sunce u Škorpionu"), (7, "VII Kola — Asc Rak; 7 štapova (b)"),
    (6, "VI Ljubavnici — Mjesec u Blizancima; 6 štapova (b)"), (8, "VIII Snaga — Lav"),
    (11, "XI Pravda / Lust"), (15, "5 pehara (b) — 1. dekada Škorpiona"),
    (24, "4 mačeva (b) — 3. dekada Vage"), (29, "7 štapova (a) — 3. dekada Lava"),
    (28, "6 štapova (a) — 2. dekada Lava"), (41, "5 pehara (a)"),
]


def kljucni_brojevi(k_dodatnih: int = 4) -> list[int]:
    """Šesnaest fiksnih ključnih brojeva + najčešći među prvih 30 rangiranih kandidata."""
    kljuc = [x for x, _ in KLJUCNI_FIKSNI]
    cnt: Counter = Counter()
    for kd in kandidati(80)[:30]:
        cnt.update(kd["brojevi"])
    for x, _ in sorted(cnt.items(), key=lambda t: (-t[1], t[0])):
        if len(kljuc) >= len(KLJUCNI_FIKSNI) + k_dodatnih:
            break
        if x not in kljuc:
            kljuc.append(x)
    return kljuc


# --------------------------------------------------------------------------- astrološka provjera (ephem)
ZNAKOVI = ["Ovan", "Bik", "Blizanci", "Rak", "Lav", "Djevica", "Vaga", "Škorpion", "Strijelac",
           "Jarac", "Vodenjak", "Ribe"]
# Zlatna zora: dekade → male arkane 2–10 (Book T)
DEKADE: dict[str, list[str]] = {
    "Ovan": ["2 štapova", "3 štapova", "4 štapova"], "Bik": ["5 novčića", "6 novčića", "7 novčića"],
    "Blizanci": ["8 mačeva", "9 mačeva", "10 mačeva"], "Rak": ["2 pehara", "3 pehara", "4 pehara"],
    "Lav": ["5 štapova", "6 štapova", "7 štapova"], "Djevica": ["8 novčića", "9 novčića", "10 novčića"],
    "Vaga": ["2 mačeva", "3 mačeva", "4 mačeva"], "Škorpion": ["5 pehara", "6 pehara", "7 pehara"],
    "Strijelac": ["8 štapova", "9 štapova", "10 štapova"], "Jarac": ["2 novčića", "3 novčića", "4 novčića"],
    "Vodenjak": ["5 mačeva", "6 mačeva", "7 mačeva"], "Ribe": ["8 pehara", "9 pehara", "10 pehara"],
}
ZNAK_ARKANA: dict[str, str] = {
    "Ovan": "IV Car", "Bik": "V Jerofant", "Blizanci": "VI Ljubavnici", "Rak": "VII Kola",
    "Lav": "VIII Snaga", "Djevica": "IX Pustinjak", "Vaga": "XI Pravda", "Škorpion": "XIII Smrt",
    "Strijelac": "XIV Umjerenost", "Jarac": "XV Đavao", "Vodenjak": "XVII Zvijezda", "Ribe": "XVIII Mjesec",
}
PLANET_ARKANA: dict[str, str] = {
    "Sunce": "XIX Sunce", "Mjesec": "II Visoka svećenica", "Merkur": "I Mag", "Venera": "III Carica",
    "Mars": "XVI Kula", "Jupiter": "X Kolo sreće", "Saturn": "XXI Svijet",
}
# zadane (iz konteksta zadatka) ekliptičke duljine, koriste se ako ephem nije dostupan
ZADANE_DULJINE: dict[str, float] = {
    "Sunce": 216.37, "Mjesec": 85.69, "Merkur": 228.80, "Venera": 207.41, "Mars": 137.50,
    "Jupiter": 144.05, "Saturn": 9.42,
}


def astro_tablica() -> list[str]:
    """Položaji planeta 29.10.2026. 19:00 UTC → znak, dekada, karte Zlatne zore. ephem ako postoji, inače zadano."""
    duljine: dict[str, float] = {}
    izvor = "zadane vrijednosti (ephem nije dostupan)"
    try:
        import math
        import ephem  # type: ignore
        d = ephem.Date("2026/10/29 19:00:00")
        tijela = {"Sunce": ephem.Sun(), "Mjesec": ephem.Moon(), "Merkur": ephem.Mercury(), "Venera": ephem.Venus(),
                  "Mars": ephem.Mars(), "Jupiter": ephem.Jupiter(), "Saturn": ephem.Saturn()}
        for nm, b in tijela.items():
            b.compute(d, epoch=d)
            duljine[nm] = math.degrees(ephem.Ecliptic(b).lon) % 360.0
        izvor = "ephem (body.compute(d, epoch=d) → ephem.Ecliptic, 2026-10-29 19:00 UTC)"
    except Exception:
        duljine = dict(ZADANE_DULJINE)
    L = [f"Izvor položaja: {izvor}.", "",
         "| tijelo | duljina | znak | stupanj | dekada | velika arkana (znak) | velika arkana (planet) | mala arkana (dekada) |",
         "|---|---|---|---|---|---|---|---|"]
    for nm, lon in duljine.items():
        z = int(lon // 30)
        st = lon - 30 * z
        dek = int(st // 10) + 1
        zn = ZNAKOVI[z]
        L.append(f"| {nm} | {lon:.2f}° | {zn} | {st:.2f}° | {dek}. | {ZNAK_ARKANA[zn]} | {PLANET_ARKANA[nm]} | {DEKADE[zn][dek - 1]} |")
    L.append("| Ascendent | ~95° | Rak | ~5° | 1. | VII Kola | — | 2 pehara |")
    L.append("| MC | ~337° | Ribe | ~7° | 1. | XVIII Mjesec | — | 8 pehara |")
    return L


# --------------------------------------------------------------------------- postupak (tekst)
def _tablica_spreada(iz: dict, naziv: dict[int, str], mult: dict[int, float]) -> list[str]:
    L = ["| položaj | značenje | karta | a | b | množitelj | ključna |", "|---|---|---|---|---|---|---|"]
    for p in sorted(iz["polozaji"]):
        c = iz["polozaji"][p]
        kl = f"+{KLJUCNE_KARTE[c][0]:.2f}" if c in KLJUCNE_KARTE else ""
        tag = f"**{ime(c)}**" if c in KLJUCNE_KARTE else ime(c)
        L.append(f"| {p} | {naziv[p]} | {tag} | {c} | {vrijednost(c)} | {mult[p]:.1f} | {kl} |")
    return L


def postupak_tekst(n_detaljno: int = 10) -> str:
    L: list[str] = []
    L.append(f"# Tarot (Rider–Waite–Smith) za Loto 6/45, {DATUM_TEKST}\n")
    L.append("## 1. Špil i mapiranja\n")
    L.append("Referentni popis (a), 1–78: " + ", ".join(f"{c}={ime(c)}" for c in range(1, 79)) + ".\n")
    L.append("Mapiranje (b) — broj karte: velika arkana = broj (Luda 0 → 22); mala arkana = rang (As 1 … 10, Paž 11, "
             "Vitez 12, Kraljica 13, Kralj 14) + ofset boje (štapovi 0, pehari 10, mačevi 20, novčići 30). "
             "Primjeri: As novčića a=65 → b=31; 10 novčića a=74 → b=40; 9 novčića a=73 → b=39; 7 štapova a=29 → b=7; "
             "5 pehara a=41 → b=15; 4 mačeva a=54 → b=24; Kralj novčića a=78 → b=44. Broj 45 postoji samo po (a) "
             "(9 pehara).\n")
    L.append("## 2. Astrološke korespondencije trenutka (Zlatna zora)\n")
    L += astro_tablica()
    L.append("")
    L.append("Dan: četvrtak = Jupiter; 20:00 CET = 3. noćni planetarni sat (zalazak 16:47:32, izlazak 06:32:51 → "
             "noćni sat 68,78 min; redoslijed Mjesec, Saturn, **Jupiter**) → Jupiter vlada i danom i satom → X Kolo sreće "
             "je signifikator Keltskog križa i najjača ključna karta. Mjesec 82,6 % opadajući.\n")
    L.append("Ključne karte i težine:\n")
    L.append("| karta | a | b | težina | razlog |\n|---|---|---|---|---|")
    for c, (w, razlog) in sorted(KLJUCNE_KARTE.items(), key=lambda t: (-t[1][0], t[0])):
        L.append(f"| {ime(c)} | {c} | {vrijednost(c)} | {w:.2f} | {razlog} |")
    L.append("")
    L.append("## 3. Pravila izvedbe\n")
    L.append("- **(a) referentno pravilo** (korisnikova skripta, deck RW): špil [1..78] se promiješa; s vrha se uzimaju "
             "prve karte s rednim brojem ≤ 45 dok ih nema 6; karte > 45 se okreću, ali ne daju broj.")
    L.append("- **(b) broj karte**: isti promiješani špil; svaka karta s vrha daje vrijednost b; duplikat se preskače; prvih 6.")
    L.append("- **(c) Keltski križ** (Waite 1910): signifikator X Kolo sreće izdvojen (broj 10 zadan); 77 karata se miješa; "
             "položaji 1–10 = prvih 10 karata; brojevi po (b) redom " + " → ".join(f"{p} {KK_NAZIV[p]}" for p in KK_PRIORITET)
             + " dok nema 6; dopuna (a) ≤ 45 istih karata, pa vrijednosti s vrha ostatka.")
    L.append("- **(d) Kolo sreće**: nov promiješani špil od 78; položaji 1–6 = prvih 6 karata; brojevi po (b) redom "
             + " → ".join(f"{p} {KS_NAZIV[p]}" for p in KS_PRIORITET) + "; dopuna kao kod (c), pa karte 7, 8, … s vrha.")
    L.append(f"- **Miješanje**: čitanje i → rng = random.Random({SEED0} + i), i = 0…{N_CITANJA - 1}; iz istog rng-a redom "
             "rng.shuffle(d1=[1..78]), rng.shuffle(d2=[1..78] bez 10), rng.shuffle(d3=[1..78]). Nikad SystemRandom; "
             "isti datum → iste karte. Bez obrnutih karata.\n")
    L.append("## 4. Rangiranje\n")
    L.append("Ocjena izvedbe = Σ težina(karta) × množitelj po svim okrenutim kartama: (a)/(b) karta koja daje broj 1,0, "
             "preskočena 0,5; Keltski križ po položaju " + ", ".join(f"{p}: {m:.1f}" for p, m in KK_MULT.items())
             + "; Kolo sreće " + ", ".join(f"{p}: {m:.1f}" for p, m in KS_MULT.items()) + "; dopunske karte 0,5.")
    L.append("Kandidat = težina izvedbe × (1 + ocjena): " + ", ".join(f"{NAZIV_IZVEDBE[o]} {w:.2f}" for o, w in TEZINA_IZVEDBE.items())
             + ". Nerješeno → manji i, pa redoslijed izvedbi. Raznolikost: kandidat koji s već prihvaćenim dijeli 5+ brojeva se preskače.\n")

    L.append(f"## 5. Prvih {n_detaljno} čitanja (i = 0 … {n_detaljno - 1}) — potpuni izvod\n")
    for ct in sva_citanja()[:n_detaljno]:
        L.append(f"### Čitanje #{ct['i']} (seed {ct['seed']})\n")
        L.append("Špil d1 (prvih 14 karata): " + " | ".join(opis(c) for c in ct["d1"][:14]) + " …\n")
        for iz in ct["izvedbe"]:
            o = iz["oznaka"]
            L.append(f"**({o}) {NAZIV_IZVEDBE[o]}** → {' · '.join(f'{x:02d}' for x in sorted(iz['brojevi']))} "
                     f"(redom izvedeno {iz['brojevi']}; ocjena {iz['tezina']:.2f} × (1 + {iz['ocjena_karata']:.2f}) = {iz['ocjena']:.3f})")
            if o == "c":
                L.append("")
                L += _tablica_spreada(iz, KK_NAZIV, KK_MULT)
                L.append("")
            elif o == "d":
                L.append("")
                L += _tablica_spreada(iz, KS_NAZIV, KS_MULT)
                L.append("")
            L += [f"- {x}" for x in iz["log"]]
            L.append("- ključne karte: " + (", ".join(iz["ocjena_log"]) if iz["ocjena_log"] else "nema"))
            L.append("")

    L.append("## 6. Sva čitanja — ocjene izvedbi (a / b / c / d) i brojevi po (a)\n")
    L.append("| i | seed | (a) | (b) | (c) | (d) | brojevi (a) | brojevi (c) |")
    L.append("|---|---|---|---|---|---|---|---|")
    for ct in sorted(sva_citanja(), key=lambda c: (-max(iz["ocjena"] for iz in c["izvedbe"]), c["i"])):
        iz = ct["izvedbe"]
        L.append(f"| {ct['i']} | {ct['seed']} | " + " | ".join(f"{x['ocjena']:.2f}" for x in iz)
                 + f" | {' '.join(f'{x:02d}' for x in sorted(iz[0]['brojevi']))}"
                 + f" | {' '.join(f'{x:02d}' for x in sorted(iz[2]['brojevi']))} |")
    L.append("")
    L.append("## 7. Ključni brojevi\n")
    L.append("Fiksni (astrološke i sretne karte): " + ", ".join(f"{x} = {r}" for x, r in KLJUCNI_FIKSNI) + ".")
    L.append("S dodatkom najčešćih među prvih 30 rangiranih kandidata: " + ", ".join(map(str, kljucni_brojevi())) + "\n")
    return "\n".join(L)


# --------------------------------------------------------------------------- provjere
def _provjere() -> None:
    assert [vrijednost(c) for c in (1, 21, 22, 23, 32, 36, 37, 45, 50, 51, 64, 65, 73, 74, 78)] == \
        [1, 21, 22, 1, 10, 14, 11, 19, 24, 21, 34, 31, 39, 40, 44]
    assert izvedba_a(list(range(1, 79)))["brojevi"] == [1, 2, 3, 4, 5, 6]
    assert izvedba_a(list(range(78, 0, -1)))["brojevi"] == [45, 44, 43, 42, 41, 40]
    assert izvedba_b(list(range(23, 79)) + list(range(1, 23)))["brojevi"] == [1, 2, 3, 4, 5, 6]
    # Keltski križ: signifikator 10 je uvijek prvi broj
    kk = keltski_kriz([c for c in range(1, 79) if c != 10])
    assert kk["brojevi"][0] == 10 and len(kk["brojevi"]) == 6
    # Kolo sreće s duplikatima (Mag i As štapova = 1, Visoka svećenica i 2 štapova = 2) dopunjuje se
    ks = kolo_srece([1, 23, 2, 24, 3, 25] + [c for c in range(1, 79) if c not in (1, 23, 2, 24, 3, 25)])
    assert len(ks["brojevi"]) == 6 and len(set(ks["brojevi"])) == 6, ks["brojevi"]
    kand = kandidati(80)
    assert len(kand) >= 60, len(kand)
    skupovi = [tuple(k["brojevi"]) for k in kand]
    assert len(set(skupovi)) == len(skupovi), "duplikati"
    for k in kand:
        b = k["brojevi"]
        assert b == sorted(b) and len(set(b)) == 6 and all(isinstance(x, int) and 1 <= x <= 45 for x in b), b
    razl = {x for k in kand[:60] for x in k["brojevi"]}
    assert len(razl) >= 30, f"premalo različitih brojeva u prvih 60: {len(razl)}"
    assert [k["rang"] for k in kand] == list(range(1, len(kand) + 1))
    assert kandidati(80) == kand                      # determinizam
    ocjene = [float(k["obrazlozenje"].rsplit("=", 1)[1]) for k in kand]
    assert ocjene == sorted(ocjene, reverse=True)      # rang slijedi ocjenu


if __name__ == "__main__":
    _provjere()
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
