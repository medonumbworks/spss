"""Zapadna (tropska) astrologija za Loto 6/45 — izvlačenje četvrtak 29.10.2026. u 20:00 CET, Zagreb.

ŠTO JE METODA
-------------
Horarna/elekcijska logika zapadne astrologije: za TRENUTAK izvlačenja (29.10.2026. 19:00 UTC,
Zagreb 45,815 N 15,98 E) postavlja se karta neba — tropske ekliptičke longitude Sunca, Mjeseca
i planeta (Merkur … Pluton), srednji Mjesečev čvor (Rahu/Ketu), Ascendent i MC iz lokalnog
zvjezdanog vremena, Placidove i jednake kuće te Pars Fortunae. Iz karte se čitaju BROJEVI
(stupnjevi, longitude, znakovi, kuće, orbovi aspekata), a prednost imaju tradicionalni
"nositelji dobitka": Jupiter (veliki benefik, planet sreće i obilja), Venera (mali benefik),
5. kuća (igra, špekulacija, kocka), Pars Fortunae (dio sreće), 2. i 8. kuća (vlastiti i tuđi
novac) te 11. kuća (dobici, ispunjenje nada).

IZVORI / PRAVILA
  - Efemeride: PyEphem (body.compute(datum, epoch=datum) → ephem.Ecliptic(body).lon = tropska
    longituda datuma). Fallback: ugrađene vrijednosti izračunate istim postupkom.
  - LST iz ephem.Observer.sidereal_time(); RAMC = LST u stupnjevima; nagib ekliptike ε = 23,436°
    (Meeusova srednja vrijednost za 2026. je 23,4358°).
    MC  = atan2(sin RAMC, cos RAMC · cos ε)
    Asc = atan2(cos RAMC, −(sin RAMC · cos ε + tan φ · sin ε))
  - Placidove kuće: klasična iteracija po polulukovima (vrh 11 = 1/3 dnevnog poluluka od MC,
    12 = 2/3, 2 = 2/3 noćnog poluluka od IC, 3 = 1/3); tan δ = tan ε · sin RA, poluluk
    SA = acos(−tan δ · tan φ), λ = atan2(sin RA, cos RA · cos ε). Provjereno na tablici za
    51,5 N / ST 0h (11 ≈ 9° Bik, 12 ≈ 22° Blizanci, 2 ≈ 12° Lav, 3 ≈ 3° Djevica).
    Dodatno jednake kuće (vrh k = Asc + 30·(k−1)) kao sekundarni, slabiji izvor.
  - Srednji Mjesečev čvor (Meeus, Astronomical Algorithms, 47.7):
    Ω = 125,0445479 − 1934,1362891·T + 0,0020754·T² + T³/467441 − T⁴/60616000.
  - Pars Fortunae: noćna karta (Sunce ispod horizonta) → PF = Asc − Mjesec + Sunce
    (Ptolemej/Dorotej, helenistički obrat); dnevna formula Asc + Mjesec − Sunce daje
    Pars Spiritus i navodi se kao slabiji izvor.
  - Aspekti i orbovi: konjunkcija 8°, opozicija 8°, trigon 7°, kvadrat 7°, sekstil 5°,
    kvinkunks 3°, polukvadrat/seskvikvadrat/kvintil/polusekstil 2°; Sunce i Mjesec +1°.
  - Dostojanstva: Venera u Vagi = domicil (+1), Saturn u Ovnu = pad (−0,5); planet u 5. kući
    +1 (Sunce +2 kao vladar vremena na dan Jupitera / Jupiterov sat).

BROJ IZ POLOŽAJA (za svaku točku, 1..45):
  stupanj u znaku zaokružen (1..30) · apsolutna longituda mod 45 (0 → 45) · broj znaka (1..12)
  · znak + stupanj · (znak × stupanj) mod 45 · sabijanski redni stupanj (floor + 1, slabije)
  · broj kuće u kojoj je planet (slabo).
BROJ IZ ASPEKTA: zaokruženi orb (najmanje 1) ×0,5 · kut mod 45 ×0,3 samo kad kut nije višekratnik
  od 45 (sekstil 15, trigon 30, kvintil 27, kvinkunks 15 …; konjunkcija/kvadrat/opozicija bi
  degenerirale u 45 pa se preskaču) · kut/10 ×0,3 ako je cijeli broj;
  težina = prosjek težina dvaju tijela × (1 − orb/orb_max) × (1 za glavne, 0,5 za sporedne).
  Aspekti su namjerno SPOREDNI izvor: orbovi 1–3° ne smiju pretegnuti Jupiter, Veneru i 5. kuću.

POSTUPAK ZA 29.10.2026. 19:00 UTC (sažeto; cijeli izvod s tablicama generira postupak_tekst()
i zapisuje se u out/ezo_zapadna.json)
  1. Položaji (tropski, datum): Sunce 6°22' Škorpion, Mjesec 25°41' Blizanci, Merkur 18°48'
     Škorpion, Venera 27°25' Vaga (domicil), Mars 17°30' Lav, Jupiter 24°03' Lav, Saturn 9°25'
     Ovan (pad), Uran 4°45' Blizanci, Neptun 2°08' Ovan, Pluton 3°07' Vodenjak; srednji čvor
     Rahu 26°11' Vodenjak / Ketu 26°11' Lav. LST 22:36:25 → RAMC 339,10°.
  2. Osi i kuće: Asc 5°01' Rak, MC 7°24' Ribe. Placidus: 2 = 23°10' Rak, 3 = 12°45' Lav,
     5 = 11°37' Vaga, 6 = 25°21' Škorpion, 8 = 23°10' Jarac, 11 = 11°37' Ovan, 12 = 25°21' Bik.
     Sunce, Merkur i Venera su u 5. KUĆI (igra), Jupiter i Mars u 3., Mjesec i Uran u 12.,
     Saturn i Neptun u 10., Pluton u 8. Vladar 5. kuće = Venera u 5. kući u domicilu —
     najjači svjedok igre; vladar 2. i Asc = Mjesec; vladar 8. = Saturn; vladar 11. = Mars.
  3. Noćna karta (Sunce −34° ispod horizonta) → PF = 95,01 − 85,69 + 216,37 = 225,70 =
     15°42' Škorpion, također u 5. kući. Pars Spiritus / dnevna formula = 324,33 = 24°20'
     Vodenjak (konjunkcija s Rahuom 26°11').
  4. Aspekti Jupitera: sekstil Mjesec (orb 1,6°), sekstil Venera (3,4°), konjunkcija Ketu /
     opozicija Rahu (2,1°), konjunkcija Mars (6,5°), kvintil Sunce (0,3°), kvadrat Merkur (5,3°).
     Ostali: Mjesec trigon Venera (1,7°), Merkur kvadrat Mars (1,3°), Mjesec opozicija Pluton
     (2,6°), Sunce kvadrat Pluton (3,3°), Sunce kvinkunks Uran (1,6°), MC kvadrat Uran (2,7°),
     Asc kvadrat Neptun (2,9°), Saturn konjunkcija Neptun (7,3°).
  5. Mjesec: 82,6 % osvijetljen, opadajući gibozni, 3,6 dana nakon punog (26.10. 05:11 CET),
     starost ~18,8 dana, elongacija 229,3°.
  6. Bazen: Jupiter → 24 (stupanj), 9 (144 mod 45), 5 (Lav), 29, 30; Venera → 27 (stupanj) i
     27 (207,4 mod 45 — dvostruko!), 7, 34, 9; vrh 5. kuće → 12 (dvostruko: 11°37' i 191,6
     mod 45), 7, 19, 39; PF → 16, 1, 8, 24, 38; Pars Spiritus → 24, 9 (isto kao Jupiter!);
     vrh 2. i 8. kuće → 23 (četverostruko), 27, 33; vrh 11. → 12, 13; Sunce → 6, 36, 14;
     Mjesec → 26, 41, 29, 33; Merkur → 19, 4, 27; Mars → 18, 3, 23, 45; Asc → 5, 9, 20;
     MC → 7, 22, 19, 39; čvor → 26, 11, 37; faza → 38, 4, 19, 20.
  7. Kandidati: (a) 14 KANONSKIH skupova — izravne derivacije po prioritetu metode (Jupiter,
     Venera, 5. kuća, Pars Fortunae, novčane kuće, svjetla, osi, Jupiterovi aspekti, stupnjevi,
     longitude mod 45, znak+stupanj, Mjesec/čvor, vladari, jednake kuće); (b) zatim 6-člani
     podskupovi 28 najtežih brojeva bazena rangirani po zbroju težina uz uvjet raznolikosti
     (≤ 3 zajednička broja s već prihvaćenim skupom, penal za već iskorištene brojeve).
     Metoda je čisti izračun — nema slučajnosti.
"""
from __future__ import annotations

import itertools
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

SLUG = "zapadna"
METODA = "Zapadna astrologija"
DATUM_TEKST = "četvrtak 29.10.2026. u 20:00 CET (19:00 UTC), Zagreb 45,815 N 15,98 E"
UTC_DATUM = "2026/10/29 19:00:00"
LAT, LON = 45.815, 15.98
EPS = 23.436                       # nagib ekliptike (zadano); Meeus 2026: 23,4358°
OUT_DIR = Path(__file__).resolve().parent.parent / "out"

ZNAKOVI = ["Ovan", "Bik", "Blizanci", "Rak", "Lav", "Djevica",
           "Vaga", "Škorpion", "Strijelac", "Jarac", "Vodenjak", "Ribe"]
VLADAR = {"Ovan": "Mars", "Bik": "Venera", "Blizanci": "Merkur", "Rak": "Mjesec", "Lav": "Sunce",
          "Djevica": "Merkur", "Vaga": "Venera", "Škorpion": "Mars", "Strijelac": "Jupiter",
          "Jarac": "Saturn", "Vodenjak": "Saturn", "Ribe": "Jupiter"}
DOMICIL = {"Mars": ("Ovan", "Škorpion"), "Venera": ("Bik", "Vaga"), "Merkur": ("Blizanci", "Djevica"),
           "Mjesec": ("Rak",), "Sunce": ("Lav",), "Jupiter": ("Strijelac", "Ribe"),
           "Saturn": ("Jarac", "Vodenjak")}
EGZALTACIJA = {"Sunce": "Ovan", "Mjesec": "Bik", "Merkur": "Djevica", "Venera": "Ribe",
               "Mars": "Jarac", "Jupiter": "Rak", "Saturn": "Vaga"}
PLANETI = ["Sunce", "Mjesec", "Merkur", "Venera", "Mars", "Jupiter", "Saturn", "Uran", "Neptun", "Pluton"]

# fallback = vrijednosti dobivene istim ephem postupkom (ako ephem nije dostupan)
FALLBACK = {
    "pozicije": {"Sunce": 216.3736, "Mjesec": 85.6869, "Merkur": 228.7978, "Venera": 207.4112,
                 "Mars": 137.5005, "Jupiter": 144.0487, "Saturn": 9.4208, "Uran": 64.7475,
                 "Neptun": 2.1246, "Pluton": 303.1087},
    "lst_deg": 339.10305, "jd": 2461343.2916667, "faza": 82.6, "sunce_alt": -33.93,
    "dana_od_punog": 3.62, "starost": 18.9,
}

# bazne težine izvora (logika metode: Jupiter > Venera > 5. kuća ≈ PF > novčane kuće > svjetla …)
W_BAZA = {"Jupiter": 10.0, "Venera": 8.0, "Sunce": 5.0, "Mjesec": 5.0, "Merkur": 4.0, "Mars": 4.0,
          "Saturn": 3.0, "Uran": 2.0, "Neptun": 2.0, "Pluton": 2.0,
          "Rahu": 3.0, "Ketu": 2.0, "Asc": 5.0, "MC": 4.0, "Dsc": 2.0, "IC": 2.0,
          "PF": 8.0, "PS": 4.0,
          "K5": 8.0, "K2": 6.0, "K8": 6.0, "K11": 6.0, "K3": 1.5, "K6": 1.5, "K9": 1.5, "K12": 1.5,
          "J5": 5.0, "J2": 2.5, "J8": 2.5, "J11": 2.5, "Faza": 3.0}
# množitelji po vrsti broja iz položaja
F_STUPANJ, F_MOD45, F_ZNAK, F_ZNAK_PLUS, F_ZNAK_PUTA, F_SABIAN, F_KUCA = 1.0, 1.0, 0.5, 0.6, 0.4, 0.3, 0.3
F_ORB, F_KUT45, F_KUT10 = 0.5, 0.3, 0.3   # aspekti su sporedni izvor: ne smiju pretegnuti Jupiter/Veneru/5. kuću

ASPEKTI = {0: ("konjunkcija", 8.0, 1.0), 180: ("opozicija", 8.0, 1.0), 120: ("trigon", 7.0, 1.0),
           90: ("kvadrat", 7.0, 1.0), 60: ("sekstil", 5.0, 1.0), 150: ("kvinkunks", 3.0, 0.5),
           45: ("polukvadrat", 2.0, 0.5), 135: ("seskvikvadrat", 2.0, 0.5),
           72: ("kvintil", 2.0, 0.5), 30: ("polusekstil", 2.0, 0.5)}


# --------------------------------------------------------------------------- pomoćne funkcije
def fmt(lon: float) -> str:
    s = int(lon // 30) % 12
    x = lon - 30 * (lon // 30)
    d = int(x)
    m = (x - d) * 60
    return f"{d}°{m:04.1f}' {ZNAKOVI[s]}"


def znak(lon: float) -> int:
    """Broj znaka 1..12 (1 = Ovan)."""
    return int(lon // 30) % 12 + 1


def stupanj(lon: float) -> int:
    """Stupanj u znaku zaokružen na cijeli broj, 1..30."""
    return max(1, min(30, round(lon - 30 * (lon // 30))))


def sabian(lon: float) -> int:
    """Sabijanski redni stupanj (0°00'–0°59' = 1. stupanj … 29°xx' = 30.)."""
    return int(lon - 30 * (lon // 30)) + 1


def mod45(lon: float) -> int:
    v = round(lon % 45)
    return 45 if v == 0 or v == 45 else v


def kuca_tocke(lon: float, vrhovi: dict[int, float]) -> int:
    for h in range(1, 13):
        a, b = vrhovi[h], vrhovi[h % 12 + 1]
        if (lon - a) % 360 < (b - a) % 360:
            return h
    return 12


def razlika(a: float, b: float) -> float:
    return abs((a - b + 180) % 360 - 180)


# --------------------------------------------------------------------------- astronomija
def astronomija() -> dict:
    """Pozicije, LST, faza — ephem ako je dostupan, inače ugrađeni fallback (isti brojevi)."""
    info = {"izvor": "fallback", **{k: v for k, v in FALLBACK.items()}}
    try:
        import ephem
        d = ephem.Date(UTC_DATUM)
        tijela = {"Sunce": ephem.Sun, "Mjesec": ephem.Moon, "Merkur": ephem.Mercury, "Venera": ephem.Venus,
                  "Mars": ephem.Mars, "Jupiter": ephem.Jupiter, "Saturn": ephem.Saturn,
                  "Uran": ephem.Uranus, "Neptun": ephem.Neptune, "Pluton": ephem.Pluto}
        poz = {}
        for ime, B in tijela.items():
            b = B()
            b.compute(d, epoch=d)                     # tropske koordinate datuma
            poz[ime] = math.degrees(ephem.Ecliptic(b).lon) % 360.0
        mo = ephem.Moon()
        mo.compute(d, epoch=d)
        obs = ephem.Observer()
        obs.lat, obs.lon, obs.date, obs.elevation = str(LAT), str(LON), d, 120
        sun = ephem.Sun()
        sun.compute(obs)
        puni = ephem.previous_full_moon(d)
        mladi = ephem.previous_new_moon(d)
        info.update(izvor="ephem", pozicije=poz, lst_deg=math.degrees(obs.sidereal_time()),
                    jd=float(ephem.julian_date(d)), faza=float(mo.phase),
                    sunce_alt=math.degrees(sun.alt), dana_od_punog=float(d - puni),
                    starost=float(d - mladi),
                    puni_mjesec=str(ephem.Date(puni + ephem.hour)), mladak=str(ephem.Date(mladi + ephem.hour)))
    except Exception as e:                            # noqa: BLE001 — bez ephema koristimo fallback
        info["greska"] = repr(e)
    return info


def srednji_cvor(jd: float) -> float:
    T = (jd - 2451545.0) / 36525.0
    return (125.0445479 - 1934.1362891 * T + 0.0020754 * T**2 + T**3 / 467441 - T**4 / 60616000) % 360


def osi(ramc: float, lat: float = LAT, eps: float = EPS) -> tuple[float, float]:
    r = math.radians
    mc = math.degrees(math.atan2(math.sin(r(ramc)), math.cos(r(ramc)) * math.cos(r(eps)))) % 360
    asc = math.degrees(math.atan2(math.cos(r(ramc)),
                                  -(math.sin(r(ramc)) * math.cos(r(eps)) + math.tan(r(lat)) * math.sin(r(eps))))) % 360
    return asc, mc


def placidus(ramc: float, asc: float, mc: float, lat: float = LAT, eps: float = EPS) -> dict[int, float]:
    """Placidove kuće — iteracija po polulukovima (vidi docstring modula)."""
    r = math.radians

    def ra_u_lon(ra):
        return math.degrees(math.atan2(math.sin(r(ra)), math.cos(r(ra)) * math.cos(r(eps)))) % 360

    def poluluk(ra):                                  # dnevni poluluk točke ekliptike s RA = ra
        dec = math.atan(math.tan(r(eps)) * math.sin(r(ra)))
        x = max(-1.0, min(1.0, -math.tan(dec) * math.tan(r(lat))))
        return math.degrees(math.acos(x))

    vrh = {1: asc, 10: mc, 7: (asc + 180) % 360, 4: (mc + 180) % 360}
    for h, F, start, nocni in ((11, 1 / 3, 30, False), (12, 2 / 3, 60, False),
                               (2, 2 / 3, 120, True), (3, 1 / 3, 150, True)):
        ra = (ramc + start) % 360
        for _ in range(200):
            sa = poluluk(ra)
            novi = (ramc + 180 - F * (180 - sa)) % 360 if nocni else (ramc + F * sa) % 360
            gotovo = abs((novi - ra + 180) % 360 - 180) < 1e-9
            ra = novi
            if gotovo:
                break
        vrh[h] = ra_u_lon(ra)
        vrh[(h + 5) % 12 + 1] = (vrh[h] + 180) % 360
    return vrh


def jednake(asc: float) -> dict[int, float]:
    return {k: (asc + 30 * (k - 1)) % 360 for k in range(1, 13)}


# --------------------------------------------------------------------------- karta
def karta() -> dict:
    a = astronomija()
    poz = dict(a["pozicije"])
    ramc = a["lst_deg"]
    asc, mc = osi(ramc)
    vrh = placidus(ramc, asc, mc)
    jed = jednake(asc)
    rahu = srednji_cvor(a["jd"])
    poz["Rahu"], poz["Ketu"] = rahu, (rahu + 180) % 360
    poz["Asc"], poz["MC"], poz["Dsc"], poz["IC"] = asc, mc, (asc + 180) % 360, (mc + 180) % 360
    nocna = a["sunce_alt"] < 0
    if nocna:
        pf = (asc - poz["Mjesec"] + poz["Sunce"]) % 360
        ps = (asc + poz["Mjesec"] - poz["Sunce"]) % 360
    else:
        pf = (asc + poz["Mjesec"] - poz["Sunce"]) % 360
        ps = (asc - poz["Mjesec"] + poz["Sunce"]) % 360
    poz["PF"], poz["PS"] = pf, ps
    kuce = {p: kuca_tocke(poz[p], vrh) for p in PLANETI + ["Rahu", "Ketu", "PF", "PS"]}
    dost = {}
    for p in PLANETI[:7]:
        z = ZNAKOVI[znak(poz[p]) - 1]
        if z in DOMICIL.get(p, ()):
            dost[p] = "domicil"
        elif EGZALTACIJA.get(p) == z:
            dost[p] = "egzaltacija"
        elif any(ZNAKOVI[(ZNAKOVI.index(zz) + 6) % 12] == z for zz in DOMICIL.get(p, ())):
            dost[p] = "izgon"
        elif ZNAKOVI[(ZNAKOVI.index(EGZALTACIJA[p]) + 6) % 12] == z:
            dost[p] = "pad"
        else:
            dost[p] = "peregrin"
    return {"astro": a, "poz": poz, "ramc": ramc, "vrh": vrh, "jed": jed, "nocna": nocna,
            "kuce": kuce, "dost": dost, "elong": (poz["Mjesec"] - poz["Sunce"]) % 360}


def tezine_tocaka(K: dict) -> dict[str, float]:
    W = dict(W_BAZA)
    for p, dg in K["dost"].items():
        W[p] += {"domicil": 1.0, "egzaltacija": 0.5, "pad": -0.5, "izgon": -0.5}.get(dg, 0.0)
    for p in PLANETI:
        if K["kuce"][p] == 5:
            W[p] += 2.0 if p == "Sunce" else 1.0
    return W


def aspekti(K: dict, W: dict) -> list[dict]:
    poz = K["poz"]
    tocke = PLANETI + ["Asc", "MC", "Rahu", "Ketu", "PF"]
    out = []
    for i, j in itertools.combinations(range(len(tocke)), 2):
        x, y = tocke[i], tocke[j]
        if {x, y} == {"Rahu", "Ketu"} or {x, y} == {"Asc", "MC"}:
            continue
        dist = razlika(poz[x], poz[y])
        for kut, (ime, orb_max, k) in ASPEKTI.items():
            om = orb_max + (1.0 if "Sunce" in (x, y) or "Mjesec" in (x, y) else 0.0)
            orb = abs(dist - kut)
            if orb <= om:
                w = (W[x] + W[y]) / 2 * (1 - orb / om) * k
                if "Rahu" in (x, y) or "Ketu" in (x, y):
                    w *= 0.6
                out.append({"a": x, "b": y, "aspekt": ime, "kut": kut, "orb": orb, "w": w})
    out.sort(key=lambda t: (-t["w"], t["orb"]))
    return out


# --------------------------------------------------------------------------- bazen
def izgradi_bazen() -> tuple[dict[int, float], dict[int, list[tuple[str, float]]], dict]:
    K = karta()
    W = tezine_tocaka(K)
    poz, vrh, jed = K["poz"], K["vrh"], K["jed"]
    tezina: dict[int, float] = defaultdict(float)
    izvori: dict[int, list] = defaultdict(list)

    def dodaj(n: int, w: float, oznaka: str):
        if 1 <= n <= 45 and w > 0:
            tezina[n] += w
            izvori[n].append((oznaka, round(w, 2)))

    def tocka(ime: str, lon: float, w: float, kuca: int | None = None):
        s, z = stupanj(lon), znak(lon)
        dodaj(s, w * F_STUPANJ, f"{ime} stupanj {fmt(lon)}")
        dodaj(mod45(lon), w * F_MOD45, f"{ime} {lon:.1f}° mod 45")
        dodaj(z, w * F_ZNAK, f"{ime} znak {ZNAKOVI[z - 1]}")
        dodaj(z + s, w * F_ZNAK_PLUS, f"{ime} znak+stupanj {z}+{s}")
        p = (z * s) % 45
        dodaj(45 if p == 0 else p, w * F_ZNAK_PUTA, f"{ime} znak×stupanj {z}×{s} mod 45")
        if sabian(lon) != s:
            dodaj(sabian(lon), w * F_SABIAN, f"{ime} sabijanski {sabian(lon)}. stupanj")
        if kuca is not None:
            dodaj(kuca, w * F_KUCA, f"{ime} u {kuca}. kući")

    for p in PLANETI:
        tocka(p, poz[p], W[p], K["kuce"][p])
    for p in ("Rahu", "Ketu", "Asc", "MC", "Dsc", "IC"):
        tocka(p, poz[p], W[p])
    tocka("Pars Fortunae", poz["PF"], W["PF"], K["kuce"]["PF"])
    tocka("Pars Spiritus", poz["PS"], W["PS"], K["kuce"]["PS"])
    for h in (5, 2, 8, 11, 3, 6, 9, 12):
        tocka(f"vrh {h}. kuće (Placidus)", vrh[h], W[f"K{h}"])
    for h in (5, 2, 8, 11):
        tocka(f"vrh {h}. kuće (jednake)", jed[h], W[f"J{h}"])

    # vladari ključnih kuća: vladar 5., 2., 8., 11. i Asc dobiva bonus na svoj stupanj i mod 45
    vladari = {}
    for h in (5, 2, 8, 11, 1):
        v = VLADAR[ZNAKOVI[znak(vrh[h]) - 1]]
        vladari[h] = v
        wb = {5: 3.0, 2: 2.0, 8: 2.0, 11: 2.0, 1: 2.0}[h]
        dodaj(stupanj(poz[v]), wb, f"{v} vladar {h}. kuće — stupanj")
        dodaj(mod45(poz[v]), wb * 0.6, f"{v} vladar {h}. kuće — mod 45")

    # Mjesečeva faza
    a = K["astro"]
    dodaj(mod45(round(a["faza"])), W["Faza"], f"faza {a['faza']:.1f} % → {round(a['faza'])} mod 45")
    dodaj(max(1, round(a["dana_od_punog"])), W["Faza"] * 0.8, f"{a['dana_od_punog']:.1f} dana nakon punog Mjeseca")
    dodaj(max(1, round(a["starost"])), W["Faza"] * 0.8, f"starost Mjeseca {a['starost']:.1f} dana")
    dodaj(mod45(K["elong"]), W["Faza"] * 0.5, f"elongacija {K['elong']:.1f}° mod 45")
    dodaj(int(K["elong"] // 12) + 1, W["Faza"] * 0.5, f"lunarni dan {int(K['elong'] // 12) + 1}")

    # aspekti
    asp = aspekti(K, W)
    for t in asp:
        ozn = f"{t['a']} {t['aspekt']} {t['b']} (orb {t['orb']:.1f}°)"
        dodaj(max(1, round(t["orb"])), t["w"] * F_ORB, ozn + " — orb")
        k45 = t["kut"] % 45
        if k45:                                   # višekratnici od 45 degeneriraju u 45 → preskoči
            dodaj(k45, t["w"] * F_KUT45, ozn + f" — kut {t['kut']} mod 45")
        if t["kut"] % 10 == 0 and t["kut"] > 0:
            dodaj(t["kut"] // 10, t["w"] * F_KUT10, ozn + f" — kut/10")

    K["W"], K["aspekti"], K["vladari"] = W, asp, vladari
    return dict(tezina), dict(izvori), K


def oznaka_broja(n: int, izvori: dict, k: int = 1) -> str:
    src = sorted(izvori.get(n, []), key=lambda t: -t[1])[:k]
    return f"{n}({'; '.join(o for o, _ in src)})" if src else str(n)


# --------------------------------------------------------------------------- kanonski skupovi
def _uzmi6(prioritet: list[int]) -> list[int]:
    out: list[int] = []
    for x in prioritet:
        if 1 <= x <= 45 and x not in out:
            out.append(x)
        if len(out) == 6:
            break
    return sorted(out)


def kanonski(K: dict, tezina: dict[int, float]) -> list[tuple[list[int], str]]:
    poz, vrh, jed, W = K["poz"], K["vrh"], K["jed"], K["W"]

    def brojevi(lon):                                  # stupanj, mod45, znak+stupanj, znak×stupanj, znak
        s, z = stupanj(lon), znak(lon)
        p = (z * s) % 45
        return [s, mod45(lon), z + s, 45 if p == 0 else p, z]

    J, V, S, M = poz["Jupiter"], poz["Venera"], poz["Sunce"], poz["Mjesec"]
    asp_J = [t for t in K["aspekti"] if "Jupiter" in (t["a"], t["b"])]
    orb_J = [max(1, round(t["orb"])) for t in asp_J]
    kut_J = [t["kut"] % 45 for t in asp_J if t["kut"] % 45]
    top = [x for x, _ in sorted(tezina.items(), key=lambda t: (-t[1], t[0]))]
    planeti_w = sorted(PLANETI, key=lambda p: -W[p])

    skupovi = [
        (_uzmi6(brojevi(J) + [K["kuce"]["Jupiter"]] + top),
         f"JUPITER {fmt(J)}: stupanj, 144 mod 45, znak+stupanj, znak×stupanj, znak Lav, {K['kuce']['Jupiter']}. kuća"),
        (_uzmi6(brojevi(V) + [K["kuce"]["Venera"]] + orb_J[:2] + top),
         f"VENERA {fmt(V)} (domicil, vladar 5. kuće): stupanj = 207 mod 45 = 27, znak Vaga 7, 7+27, 7×27 mod 45, 5. kuća"),
        (_uzmi6(brojevi(vrh[5]) + brojevi(jed[5])[:2] + top),
         f"5. KUĆA (igra): Placidus {fmt(vrh[5])} → 12 (i 191,6 mod 45 = 12), Vaga 7, 7+12, 7×12 mod 45; jednaka {fmt(jed[5])} → 5, 35"),
        (_uzmi6(brojevi(poz["PF"]) + brojevi(poz["PS"]) + top),
         f"PARS FORTUNAE (noćna) {fmt(poz['PF'])} → 16, 1, 8+16, 8×16 mod 45; Pars Spiritus {fmt(poz['PS'])} → 24, 9"),
        (_uzmi6([stupanj(vrh[2]), mod45(vrh[2]), stupanj(vrh[8]), mod45(vrh[8]), stupanj(vrh[11]), mod45(vrh[11]),
                 znak(vrh[2]) + stupanj(vrh[2]), znak(vrh[8]) + stupanj(vrh[8]), znak(vrh[11]) + stupanj(vrh[11]),
                 znak(vrh[2]), znak(vrh[8]), znak(vrh[11])] + top),
         f"NOVČANE KUĆE: vrh 2. {fmt(vrh[2])} → 23 (= 113 mod 45), 4+23; vrh 8. {fmt(vrh[8])} → 23, 10+23; vrh 11. {fmt(vrh[11])} → 12, 1+12"),
        (_uzmi6([stupanj(S), mod45(S), stupanj(M), mod45(M), znak(S) + stupanj(S), znak(M) + stupanj(M),
                 (znak(S) * stupanj(S)) % 45, (znak(M) * stupanj(M)) % 45] + top),
         f"SVJETLA: Sunce {fmt(S)} (5. kuća) → 6, 36, 14; Mjesec {fmt(M)} → 26, 41, 29"),
        (_uzmi6([stupanj(poz["Asc"]), mod45(poz["Asc"]), stupanj(poz["MC"]), mod45(poz["MC"]),
                 znak(poz["Asc"]) + stupanj(poz["Asc"]), znak(poz["MC"]) + stupanj(poz["MC"]),
                 (znak(poz["Asc"]) * stupanj(poz["Asc"])) % 45, (znak(poz["MC"]) * stupanj(poz["MC"])) % 45] + top),
         f"OSI: Asc {fmt(poz['Asc'])} → 5, 4+5, 4×5; MC {fmt(poz['MC'])} → 7, 337 mod 45 = 22, 12+7, 12×7 mod 45"),
        (_uzmi6(orb_J[:4] + kut_J + orb_J + top),
         "ASPEKTI JUPITERA: orbovi " + ", ".join(f"{t['aspekt']} {t['b'] if t['a']=='Jupiter' else t['a']} {t['orb']:.1f}°" for t in asp_J[:5])
         + "; kutovi mod 45 (sekstil 15, kvintil 27, kvadrat/konjunkcija preskočeni)"),
        (_uzmi6([stupanj(poz[p]) for p in planeti_w] + top),
         "STUPNJEVI PLANETA po težini: " + ", ".join(f"{p} {stupanj(poz[p])}" for p in planeti_w[:7])),
        (_uzmi6([mod45(poz[p]) for p in planeti_w] + top),
         "LONGITUDE mod 45 po težini: " + ", ".join(f"{p} {poz[p]:.0f}→{mod45(poz[p])}" for p in planeti_w[:7])),
        (_uzmi6([znak(poz[p]) + stupanj(poz[p]) for p in planeti_w] + [znak(poz["PF"]) + stupanj(poz["PF"])] + top),
         "ZNAK+STUPANJ po težini: " + ", ".join(f"{p} {znak(poz[p])}+{stupanj(poz[p])}" for p in planeti_w[:7])),
        (_uzmi6([mod45(round(K["astro"]["faza"])), max(1, round(K["astro"]["dana_od_punog"])),
                 max(1, round(K["astro"]["starost"])), stupanj(poz["Rahu"]), mod45(poz["Rahu"]),
                 znak(poz["Rahu"]) + stupanj(poz["Rahu"]), int(K["elong"] // 12) + 1] + top),
         f"MJESEC I ČVOR: faza {K['astro']['faza']:.1f} % → 38, {K['astro']['dana_od_punog']:.1f} d nakon punog → 4, starost → 19; "
         f"Rahu {fmt(poz['Rahu'])} → 26, 11, 37"),
        (_uzmi6([stupanj(poz[K["vladari"][h]]) for h in (5, 2, 8, 11, 1)]
                + [mod45(poz[K["vladari"][h]]) for h in (5, 2, 8, 11, 1)] + [stupanj(J)] + top),
         "VLADARI: " + ", ".join(f"{h}. kuća → {K['vladari'][h]} {stupanj(poz[K['vladari'][h]])}/{mod45(poz[K['vladari'][h]])}"
                                 for h in (5, 2, 8, 11, 1))),
        (_uzmi6([stupanj(jed[5]), mod45(jed[5]), znak(jed[5]) + stupanj(jed[5]), (znak(jed[5]) * stupanj(jed[5])) % 45,
                 mod45(jed[11]), mod45(jed[2]), znak(jed[2]) + stupanj(jed[2]), znak(jed[8]) + stupanj(jed[8]),
                 znak(jed[11]) + stupanj(jed[11])] + top),
         f"JEDNAKE KUĆE: 5. {fmt(jed[5])} → 5, 35, 13, 40; 11. {fmt(jed[11])} → 20; 2. {fmt(jed[2])} → 10; 8. {fmt(jed[8])} → 16"),
    ]
    return skupovi


# --------------------------------------------------------------------------- kandidati
def kandidati(n: int = 80) -> list[dict]:
    """≥ 60 različitih kandidata; rang 1 = najjača derivacija (Jupiter, Venera, 5. kuća, PF …)."""
    n = max(n, 60)
    tezina, izvori, K = izgradi_bazen()
    prihvaceni: list[list[int]] = []
    rezultat: list[dict] = []

    def tez(c):
        return round(sum(tezina.get(x, 0.0) for x in c), 1)

    for nums, opis in kanonski(K, tezina):
        if len(nums) != 6 or nums in prihvaceni:
            continue
        prihvaceni.append(nums)
        rezultat.append({"brojevi": nums, "rang": len(rezultat) + 1,
                         "obrazlozenje": f"Kanon: {opis} (težina {tez(nums)})"})

    # enumeracija 6-podskupova 28 najtežih brojeva bazena; rang po težini + raznolikost
    pool = sorted(x for x, _ in sorted(tezina.items(), key=lambda t: (-t[1], t[0]))[:28])
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
    LAMBDA = 1.2
    while len(rezultat) < n and ok.any():
        eff = np.where(ok, score - LAMBDA * (clan @ usage), -np.inf)
        best = int(np.argmax(eff))                    # prvi po indeksu → deterministično
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


# --------------------------------------------------------------------------- postupak (tekst)
def postupak_tekst() -> str:
    tezina, izvori, K = izgradi_bazen()
    a, poz, vrh, jed, W = K["astro"], K["poz"], K["vrh"], K["jed"], K["W"]
    L = [f"# {METODA} — {DATUM_TEKST}\n"]
    L.append(f"Izvor efemerida: **{a['izvor']}** (PyEphem, `compute(datum, epoch=datum)` → `Ecliptic(body).lon`); "
             f"JD {a['jd']:.5f}; LST {a['lst_deg'] / 15:.4f} h = RAMC **{a['lst_deg']:.3f}°**; ε = {EPS}°; "
             f"Sunce {a['sunce_alt']:.1f}° ispod horizonta → **{'noćna' if K['nocna'] else 'dnevna'} karta**.\n")
    L.append("## 1. Položaji (tropski, datum)\n")
    L.append("| točka | longituda | položaj | kuća (Placidus) | dostojanstvo | težina |")
    L.append("|---|---:|---|---:|---|---:|")
    for p in PLANETI + ["Rahu", "Ketu", "PF", "PS", "Asc", "MC"]:
        ime = {"PF": "Pars Fortunae", "PS": "Pars Spiritus"}.get(p, p)
        L.append(f"| {ime} | {poz[p]:.3f} | {fmt(poz[p])} | {K['kuce'].get(p, '-')} | {K['dost'].get(p, '')} | {W[p]:g} |")
    L.append(f"\nSrednji čvor (Meeus 47.7) za JD {a['jd']:.4f}: Rahu {fmt(poz['Rahu'])}, Ketu {fmt(poz['Ketu'])}. "
             f"Mjesec {a['faza']:.1f} % osvijetljen, {a['dana_od_punog']:.1f} dana nakon punog "
             f"({a.get('puni_mjesec', '26.10. 05:11 CET')}), starost {a['starost']:.1f} dana, elongacija {K['elong']:.1f}°.\n")
    L.append("## 2. Osi i kuće\n")
    L.append(f"MC = atan2(sin RAMC, cos RAMC·cos ε) = **{fmt(poz['MC'])}**; "
             f"Asc = atan2(cos RAMC, −(sin RAMC·cos ε + tan φ·sin ε)) = **{fmt(poz['Asc'])}**.\n")
    L.append("| kuća | Placidus | jednaka | vladar vrha (Placidus) |")
    L.append("|---:|---|---|---|")
    for h in range(1, 13):
        L.append(f"| {h} | {fmt(vrh[h])} | {fmt(jed[h])} | {VLADAR[ZNAKOVI[znak(vrh[h]) - 1]]} |")
    L.append("\nPlanetski svjedoci igre: u **5. kući** (Placidus) su " +
             ", ".join(p for p in PLANETI if K["kuce"][p] == 5) + " i Pars Fortunae; vladar 5. kuće " +
             f"{K['vladari'][5]} u 5. kući u domicilu. Vladar 2. kuće i Asc: {K['vladari'][2]}; 8.: {K['vladari'][8]}; 11.: {K['vladari'][11]}.\n")
    L.append("## 3. Pars Fortunae\n")
    L.append(f"Noćna karta → PF = Asc − Mjesec + Sunce = {poz['Asc']:.2f} − {poz['Mjesec']:.2f} + {poz['Sunce']:.2f} = "
             f"**{poz['PF']:.2f}° = {fmt(poz['PF'])}** ({K['kuce']['PF']}. kuća). "
             f"Dnevna formula / Pars Spiritus = Asc + Mjesec − Sunce = {poz['PS']:.2f}° = {fmt(poz['PS'])} "
             f"({K['kuce']['PS']}. kuća, konjunkcija s Rahuom).\n")
    L.append("## 4. Aspekti (orb → broj; kut mod 45; kut/10)\n")
    L.append("| tijela | aspekt | kut | orb | težina |")
    L.append("|---|---|---:|---:|---:|")
    for t in K["aspekti"]:
        L.append(f"| {t['a']} – {t['b']} | {t['aspekt']} | {t['kut']} | {t['orb']:.2f}° | {t['w']:.2f} |")
    L.append("")
    L.append("## 5. Pravila pretvorbe u brojeve\n")
    L.append(f"- Za svaku točku s težinom W: stupanj u znaku (zaokružen) ×{F_STUPANJ}, longituda mod 45 ×{F_MOD45}, "
             f"broj znaka ×{F_ZNAK}, znak+stupanj ×{F_ZNAK_PLUS}, (znak×stupanj) mod 45 ×{F_ZNAK_PUTA}, "
             f"sabijanski redni stupanj ×{F_SABIAN}, kuća planeta ×{F_KUCA}.")
    L.append(f"- Aspekt (sporedni izvor): orb (zaokružen, ≥ 1) ×{F_ORB}, kut mod 45 ×{F_KUT45} samo kad kut nije višekratnik od 45, "
             f"kut/10 ×{F_KUT10}; težina aspekta = prosjek težina tijela × (1 − orb/orb_max) × (1 glavni / 0,5 sporedni), čvorovi ×0,6.")
    L.append("- Vladari 5., 2., 8., 11. kuće i Asc: bonus na stupanj (3/2/2/2/2) i mod 45 (×0,6).")
    L.append(f"- Mjesečeva faza (W {W['Faza']}): postotak mod 45, dani nakon punog, starost, elongacija mod 45, lunarni dan.\n")
    L.append("## 6. Bazen s težinama (silazno)\n")
    L.append("| broj | težina | glavni izvori |")
    L.append("|---:|---:|---|")
    for n_, w_ in sorted(tezina.items(), key=lambda t: (-t[1], t[0])):
        src = "; ".join(f"{o} +{x:g}" for o, x in sorted(izvori[n_], key=lambda t: -t[1])[:4])
        L.append(f"| {n_} | {w_:.1f} | {src} |")
    L.append("")
    L.append("## 7. Pravila generiranja i rangiranja\n")
    L.append("1. Rang 1–14: KANONSKI skupovi — izravne derivacije po prioritetu metode: Jupiter, Venera, 5. kuća, "
             "Pars Fortunae, novčane kuće, svjetla, osi, Jupiterovi aspekti, stupnjevi, longitude mod 45, znak+stupanj, Mjesec/čvor, vladari, jednake kuće.")
    L.append("2. Rang 15+: svi 6-člani podskupovi 28 najtežih brojeva bazena; rang po zbroju težina; prihvaćen samo skup s ≤ 3 zajednička broja "
             "sa svakim već prihvaćenim; penal 1,2 po svakom prijašnjem korištenju broja (raznolikost). Argmax je prvi po indeksu → deterministično.")
    L.append("3. Nema slučajnosti: karta neba je čisti izračun.\n")
    L.append("## 8. Kanonski skupovi\n")
    for i, (nums, opis) in enumerate(kanonski(K, tezina), 1):
        L.append(f"{i}. `{' · '.join(f'{x:02d}' for x in nums)}` — {opis}")
    return "\n".join(L)


# ----------------------------------------------------------------------------- main
if __name__ == "__main__":
    # provjera astronomskih tvrdnji iz docstringa
    K_ = karta()
    assert fmt(K_["poz"]["Sunce"]).startswith("6°22") and fmt(K_["poz"]["Jupiter"]).startswith("24°0")
    assert fmt(K_["poz"]["Asc"]).startswith("5°0") and "Rak" in fmt(K_["poz"]["Asc"])
    assert fmt(K_["poz"]["MC"]).startswith("7°2") and "Ribe" in fmt(K_["poz"]["MC"])
    assert K_["nocna"] and K_["kuce"]["Sunce"] == 5 and K_["kuce"]["Venera"] == 5 and K_["kuce"]["PF"] == 5
    assert K_["dost"]["Venera"] == "domicil" and K_["dost"]["Saturn"] == "pad"
    assert 23 <= K_["vrh"][2] - 90 <= 24 and 11 <= K_["vrh"][5] - 180 <= 12

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
