"""Jyotiš (vedska astrologija, siderički zodijak) za Loto 6/45 — izvlačenje četvrtak 29.10.2026.
u 20:00 CET (19:00 UTC, 87. kolo), Zagreb 45.815 N 15.98 E.

ŠTO JE METODA
-------------
Jyotiš ("znanost svjetla") je indijska astrologija. Za razliku od zapadne (tropske) astrologije
računa u SIDERIČKOM zodijaku: od tropskih duljina oduzima se ayanamsa (pomak ekvinocija).
Trenutak izvlačenja tretira se kao PRAŠNA (horarna karta pitanja): lagna (ascendent) i grahe
(planeti) u tom trenutku opisuju "rođenje" ishoda. Metoda ne baca ništa — potpuno je
deterministična (nema slučajnosti).

Elementi koji se računaju za svaki graha (Sunce, Mjesec, Merkur, Venera, Mars, Jupiter, Saturn,
Rahu, Ketu) te za lagnu i MC:
  - raši (znak, 1..12 od Meše/Ovna), stupanj u rašiju (0..29 → "tekući stupanj" 1..30),
  - nakšatra (27 lunarnih kuća po 13°20' od 0° sideričkog Ovna; 1 = Ašvini) i njezin
    Vimšotari gospodar (Ketu, Venera, Sunce, Mjesec, Mars, Rahu, Jupiter, Saturn, Merkur, ciklički),
  - pada (četvrtina nakšatre, 3°20'), navamsa (D9: znak devetinke, 3°20', kontinuirano od Ovna),
  - kuća po sustavu cijelih znakova od lagne, dostojanstvo (uča = egzaltacija, niča = pad,
    svakšetra = vlastiti znak), retrogradnost.
Pančanga (pet udova vremena): vara (dan), tithi (lunarni dan, (Mjesec − Sunce)/12°),
nakšatra Mjeseca, yoga ((Sunce + Mjesec) siderički / 13°20'), karana (polovica tithija).
Vedski brojevi graha (numerologija jyotiša): Sunce 1, Mjesec 2, Jupiter 3, Rahu 4, Merkur 5,
Venera 6, Ketu 7, Saturn 8, Mars 9.

IZVORI / PRAVILA
----------------
  - Efemeride: PyEphem (`body.compute(datum, epoch=datum)` → `ephem.Ecliptic(body).lon` =
    tropska ekliptička duljina datuma). Ako ephem nije dostupan, modul koristi ugrađene vrijednosti
    (iste, izračunate istim postupkom) — označeno u postupku.
  - Ayanamsa Lahiri (Čitrapakša), formula: 23.853° + 50.29″ · (godine od J2000.0);
    za 29.10.2026. 19:00 UTC: d = 9798.29 dana od J2000.0 → 26.826 god → ayanamsa = 24.2277° =
    24°13'40″ (fiksna približna vrijednost 24.2° daje iste raši/nakšatre za sve točke; razlika
    0.03° ne mijenja ni jedan indeks — provjereno u kodu).
  - Rahu = SREDNJI uzlazni lunarni čvor: Ω = 125.0445° − 0.05295377° · d (d = dani od J2000.0,
    Meeus), Ketu = Rahu + 180°.
  - Lagna: iz lokalnog sideričkog vremena (ephem Observer.sidereal_time → RAMC) i nagiba ekliptike
    ε = 23.439291° − 0.0130042°·T: tan Asc = cos RAMC / −(sin RAMC · cos ε + tan φ · sin ε);
    MC: tan MC = tan RAMC / cos ε. Kuće: cijeli znakovi (bhava = raši) od lagne.
  - Hora: planetarni sati (12 jednakih dijelova noći od zalaska do izlaska); redoslijed kaldejski,
    prvi sat dana nosi gospodar dana. Vara: Ravivara 1 … Guruvara 5 (četvrtak), gospodar Jupiter.
  - Vimšotari daša "trenutka" (kao u prašni): iz nakšatre Mjeseca; gospodar nakšatre = mahadaša,
    prošli dio nakšatre = prošli dio mahadaše; antardaše/pratjantardaše proporcionalne godinama
    (Ketu 7, Venera 20, Sunce 6, Mjesec 10, Mars 7, Rahu 18, Jupiter 16, Saturn 19, Merkur 17).
  - Tara bala: nakšatre brojane od nakšatre Mjeseca u ciklusima od 9 (janma, sampat, vipat,
    kšema, pratjak, sadhana, naidhana, mitra, parama-mitra); povoljne su 2, 4, 6, 8, 9.
    Čandra bala: povoljni raši su 1, 3, 6, 7, 10, 11 brojano od rašija Mjeseca.
  - Bazen brojeva (sve svedeno u 1..45 pravilom x → ((x − 1) mod 45) + 1): vedski broj grahe,
    nakšatra (1..27), tekući stupanj u rašiju (1..30), raši (1..12), indeks nakšatra-pada
    ((nakšatra − 1)·4 + pada, 1..108), siderička duljina u cijelim stupnjevima mod 45, navamsa,
    pada, kuća, broj gospodara nakšatre i rašija, nakšatra + pada, raši + stupanj, nakšatra × pada,
    tithi (1..30), yoga (1..27), položaj karane (1..60), redni broj hore (1..24), lunarni mjesec,
    samvat/šaka godina mod 45, daša-brojevi (gospodari i godine).
  - RANGIRANJE kandidata (unutarnja logika metode): (1) Jupiter — karaka bogatstva, gospodar dana
    (Guruvara) i sata (hora Jupitera) — i Mjesec (um, pokazatelj prašne) s pančangom i dašom;
    (2) lagna i 5. kuća (špekulacija, lutrija), 2. (dhana), 11. (labha), 9. (bhagja) kuća, navamsa;
    (3) ostale grahe i presjeci (stupnjevi, nakšatre, indeksi svih točaka po snazi);
    (4) parovi i trojke točaka po rangu snage (Jupiter > Mjesec > lagna > Venera > Merkur > Sunce >
    Mars > Saturn > Rahu > Ketu > MC).

POSTUPAK ZA 29.10.2026. 20:00 CET (sažetak; cijeli izvod daje postupak_tekst() → out/ezo_jyotish.json)
----------------------------------------------------------------------------------------------
  Ayanamsa 24°13'40″. Siderički položaji (raši, stupanj, nakšatra/pada, navamsa, kuća od lagne):
    Lagna   10°47' Mithuna (Blizanci)   Ardra 2          D9 Makara    1. kuća
    Sunce   12°09' Tula (Vaga)          Swati 2          D9 Makara    5.  — niča (pad)
    Mjesec   1°28' Mithuna (Blizanci)   Mrigashira 3     D9 Tula      1.
    Merkur  24°34' Tula                 Vishakha 2       D9 Vrišabha  5.  R
    Venera   3°11' Tula                 Chitra 3         D9 Tula      5.  — svakšetra, vargottama, R
    Mars    23°16' Karka (Rak)          Ashlesha 2       D9 Makara    2.  — niča (uča u D9, niča-bhanga uz Jupitera)
    Jupiter 29°49' Karka (Rak)          Ashlesha 4       D9 Meena     2.  — UČA (egzaltacija)
    Saturn  15°12' Meena (Ribe)         U.Bhadrapada 4   D9 Vrišćika  10. R
    Rahu     1°58' Kumbha (Vodenjak)    Dhanishta 3      D9 Tula      9.
    Ketu     1°58' Simha (Lav)          Magha 1          D9 Meša      3.
    MC      13°11' Kumbha               Shatabhisha 2    D9 Makara    9.
  Pančanga: Guruvara (vara 5, Jupiter 3); tithi 20 = Krišna pančami (5. dan tamne polovice,
    Mjesec − Sunce = 229.31°); nakšatra Mrigashira (5) pada 3, gospodar Mars; yoga 20 Šiva
    (zbroj 253.61°); karana Kaulava (3. pokretna, položaj 39/60); hora Jupitera (3. noćni sat,
    15. sat dana; noć 16:47:32–06:32:51 CET, sat 68.78 min); lunarni mjesec Ašvina (amanta, 7) /
    Kartika (purnimanta, 8), Vikram samvat 2083, Šaka 1948.
  Vimšotari daša trenutka: Mars (7 g; prošlo 60.96 % = 4.267 g, preostaje 2 g 8 mj 24 d) /
    antardaša Merkur / pratjantardaša Jupiter → brojevi 9, 5, 3 i godine 7, 17, 16.
  Tumačenje: lagna Mithuna, Mjesec u lagni; gospodar lagne Merkur u 5. kući (špekulacija) s
    Venerom u vlastitom znaku (gospodar 5. u 5.) i Suncem (Budha-Aditja yoga); egzaltirani Jupiter
    u 2. kući (dhana, bogatstvo) na svoj dan i u svoj sat, u 2. od Mjeseca (Sunapha yoga) uz Mars
    (gospodar 11. = dobitak) kojem niču poništava; Rahu u 9. (bhagja), Saturn R u 10.
  Ključni brojevi: 3 (Jupiter), 9 (Ašleša/Mars-daša), 30 (Jupiter 30°), 2 (Mjesec), 5
    (Mrigashira/Merkur), 20 (tithi i yoga), 4 (Karka), 7 (Tula/Ketu), 6 (Venera), 12, 15, 39.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

METODA = "Vedska astrologija (jyotiš)"
SLUG = "jyotish"
OUT_DIR = Path(__file__).resolve().parent.parent / "out"

DATUM_UTC = "2026/10/29 19:00:00"
LAT, LON, ELEV = 45.815, 15.98, 120.0
N, K = 45, 6

# ------------------------------------------------------------------ tablice
GRAHE = ["Sunce", "Mjesec", "Merkur", "Venera", "Mars", "Jupiter", "Saturn", "Rahu", "Ketu"]
VED = {"Sunce": 1, "Mjesec": 2, "Jupiter": 3, "Rahu": 4, "Merkur": 5, "Venera": 6, "Ketu": 7, "Saturn": 8, "Mars": 9}
RASHI = ["Meša (Ovan)", "Vrišabha (Bik)", "Mithuna (Blizanci)", "Karka (Rak)", "Simha (Lav)", "Kanja (Djevica)",
         "Tula (Vaga)", "Vrišćika (Škorpion)", "Dhanu (Strijelac)", "Makara (Jarac)", "Kumbha (Vodenjak)", "Meena (Ribe)"]
RASHI_LORD = ["Mars", "Venera", "Merkur", "Mjesec", "Sunce", "Merkur", "Venera", "Mars", "Jupiter", "Saturn", "Saturn", "Jupiter"]
NAKSH = ["Ashvini", "Bharani", "Krittika", "Rohini", "Mrigashira", "Ardra", "Punarvasu", "Pushya", "Ashlesha",
         "Magha", "Purva Phalguni", "Uttara Phalguni", "Hasta", "Chitra", "Swati", "Vishakha", "Anuradha", "Jyeshtha",
         "Mula", "Purva Ashadha", "Uttara Ashadha", "Shravana", "Dhanishta", "Shatabhisha", "Purva Bhadrapada",
         "Uttara Bhadrapada", "Revati"]
DASHA_ORDER = ["Ketu", "Venera", "Sunce", "Mjesec", "Mars", "Rahu", "Jupiter", "Saturn", "Merkur"]
DASHA_YRS = {"Ketu": 7, "Venera": 20, "Sunce": 6, "Mjesec": 10, "Mars": 7, "Rahu": 18, "Jupiter": 16, "Saturn": 19, "Merkur": 17}
UCCHA = {"Sunce": 1, "Mjesec": 2, "Mars": 10, "Merkur": 6, "Jupiter": 4, "Venera": 12, "Saturn": 7, "Rahu": 2, "Ketu": 8}
TITHI = ["Pratipada", "Dvitija", "Tritija", "Čaturthi", "Pančami", "Šašthi", "Saptami", "Aštami", "Navami", "Dašami",
         "Ekadaši", "Dvadaši", "Trajodaši", "Čaturdaši"]
YOGA = ["Vishkambha", "Priti", "Ayushman", "Saubhagya", "Shobhana", "Atiganda", "Sukarma", "Dhriti", "Shula", "Ganda",
        "Vriddhi", "Dhruva", "Vyaghata", "Harshana", "Vajra", "Siddhi", "Vyatipata", "Variyan", "Parigha", "Shiva",
        "Siddha", "Sadhya", "Shubha", "Shukla", "Brahma", "Indra", "Vaidhriti"]
KARANA_POKRETNE = ["Bava", "Balava", "Kaulava", "Taitila", "Garaja", "Vanija", "Vishti"]
VARA = ["Ravivara (nedjelja)", "Somavara (ponedjeljak)", "Mangalavara (utorak)", "Budhavara (srijeda)",
        "Guruvara (četvrtak)", "Shukravara (petak)", "Shanivara (subota)"]
VARA_LORD = ["Sunce", "Mjesec", "Mars", "Merkur", "Jupiter", "Venera", "Saturn"]
KALDEJSKI = ["Saturn", "Jupiter", "Mars", "Sunce", "Venera", "Merkur", "Mjesec"]
TARA = ["janma", "sampat", "vipat", "kšema", "pratjak", "sadhana", "naidhana", "mitra", "parama-mitra"]
TARA_POVOLJNE = [2, 4, 6, 8, 9]
CHANDRA_POVOLJNI = [1, 3, 6, 7, 10, 11]
# rang snage točaka (unutarnja logika: Jupiter, Mjesec, lagna, 5. kuća …)
RANG_TOCAKA = ["Jupiter", "Mjesec", "Lagna", "Venera", "Merkur", "Sunce", "Mars", "Saturn", "Rahu", "Ketu", "MC"]

# ugrađene vrijednosti (fallback bez ephem) — identične onima koje daje ephem 4.2.1 istim postupkom
_FALLBACK_TROP = {"Sunce": 216.3736, "Mjesec": 85.6869, "Merkur": 228.7978, "Venera": 207.4112, "Mars": 137.5005,
                  "Jupiter": 144.0487, "Saturn": 9.4208}
_FALLBACK_RETRO = {"Merkur": True, "Venera": True, "Mars": False, "Jupiter": False, "Saturn": True}
_FALLBACK_LST_DEG = 339.10305          # 22:36:24.7
_FALLBACK_ZALAZ_UTC = (15, 47, 32)      # 29.10.
_FALLBACK_IZLAZ_UTC = (5, 32, 51)       # 30.10.
_DANI_J2000 = 9798.2916667             # 29.10.2026 19:00 UTC − J2000.0


# ------------------------------------------------------------------ pomoćne
def svedi(x: float) -> int:
    """Svedi cijeli broj u 1..45 (0 → 45, 46 → 1 …)."""
    return ((int(round(x)) - 1) % N) + 1


def dms(x: float) -> str:
    d = int(x)
    m = int(round((x - d) * 60))
    if m == 60:
        d, m = d + 1, 0
    return f"{d}°{m:02d}'"


def ayanamsa_lahiri(dani_j2000: float) -> float:
    return 23.853 + 50.29 * (dani_j2000 / 365.25) / 3600.0


# ------------------------------------------------------------------ astronomija
def astronomija() -> dict:
    """Tropske duljine, LST, zalaz/izlaz (ephem) ili ugrađeni fallback."""
    out = {"izvor": "fallback (ugrađene vrijednosti, izračunate ephem 4.2.1)", "trop": dict(_FALLBACK_TROP),
           "retro": dict(_FALLBACK_RETRO), "lst": _FALLBACK_LST_DEG, "zalaz": _FALLBACK_ZALAZ_UTC,
           "izlaz": _FALLBACK_IZLAZ_UTC, "dani": _DANI_J2000}
    try:
        import ephem
    except ImportError:
        return out
    try:
        tijela = {"Sunce": ephem.Sun(), "Mjesec": ephem.Moon(), "Merkur": ephem.Mercury(), "Venera": ephem.Venus(),
                  "Mars": ephem.Mars(), "Jupiter": ephem.Jupiter(), "Saturn": ephem.Saturn()}
        trop, retro = {}, {}
        for ime, b in tijela.items():
            b.compute(DATUM_UTC, epoch=DATUM_UTC)
            trop[ime] = math.degrees(float(ephem.Ecliptic(b).lon)) % 360
            if ime not in ("Sunce", "Mjesec"):
                b.compute("2026/10/30 19:00:00", epoch="2026/10/30 19:00:00")
                l2 = math.degrees(float(ephem.Ecliptic(b).lon)) % 360
                retro[ime] = ((l2 - trop[ime] + 180) % 360 - 180) < 0
        o = ephem.Observer()
        o.lat, o.lon, o.elevation = str(LAT), str(LON), ELEV
        o.date = DATUM_UTC
        lst = math.degrees(float(o.sidereal_time()))
        o2 = ephem.Observer()
        o2.lat, o2.lon, o2.elevation = str(LAT), str(LON), ELEV
        o2.date = "2026/10/29 12:00:00"
        z = o2.next_setting(ephem.Sun()).datetime()
        o2.date = "2026/10/30 12:00:00"
        i = o2.previous_rising(ephem.Sun()).datetime()
        dani = float(ephem.Date(DATUM_UTC)) - float(ephem.Date("2000/01/01 12:00:00"))
        out.update({"izvor": f"ephem {ephem.__version__} (compute(datum, epoch=datum), Ecliptic.lon)", "trop": trop,
                    "retro": retro, "lst": lst, "zalaz": (z.hour, z.minute, z.second),
                    "izlaz": (i.hour, i.minute, i.second), "dani": dani})
    except Exception:  # bilo kakav problem s efemeridama → fallback
        pass
    return out


def tocka(ime: str, lon_trop: float, ayan: float, lagna_rashi: int | None, retro: bool = False) -> dict:
    """Sve sideričke veličine jedne točke."""
    ls = (lon_trop - ayan) % 360
    rashi = int(ls // 30) + 1
    u_znaku = ls % 30
    deg = int(u_znaku) + 1                      # tekući stupanj 1..30
    minute = int(round((u_znaku - int(u_znaku)) * 60)) % 60
    nak = int(ls // (40 / 3)) + 1               # 13°20' = 40/3
    pada = int((ls % (40 / 3)) // (10 / 3)) + 1
    nav = int(ls * 9 // 30) % 12 + 1
    np108 = (nak - 1) * 4 + pada
    kuca = ((rashi - lagna_rashi) % 12) + 1 if lagna_rashi else 1
    rl = RASHI_LORD[rashi - 1]
    nl = DASHA_ORDER[(nak - 1) % 9]
    if ime in VED:
        ved = VED[ime]
    else:                                       # lagna/MC: broj gospodara rašija
        ved = VED[rl]
    dost = []
    if ime in UCCHA:
        if rashi == UCCHA[ime]:
            dost.append("uča (egzaltacija)")
        elif rashi == ((UCCHA[ime] + 5) % 12) + 1:
            dost.append("niča (pad)")
    if ime in VED and rl == ime:
        dost.append("svakšetra (vlastiti znak)")
    if nav == rashi:
        dost.append("vargottama")
    if retro:
        dost.append("R")
    return {"ime": ime, "trop": lon_trop, "sid": ls, "rashi": rashi, "deg": deg, "min": minute, "nak": nak,
            "pada": pada, "nav": nav, "np108": np108, "np45": svedi(np108), "l45": (int(ls) % 45) + 1, "kuca": kuca,
            "ved": ved, "nak_lord": nl, "rashi_lord": rl, "nl_ved": VED[nl], "rl_ved": VED[rl],
            "dost": dost, "retro": retro}


def izracun() -> dict:
    A = astronomija()
    ayan = ayanamsa_lahiri(A["dani"])
    # lagna i MC
    r = math.radians
    T = A["dani"] / 36525.0
    eps = 23.439291 - 0.0130042 * T
    ramc = A["lst"]
    asc = math.degrees(math.atan2(math.cos(r(ramc)),
                                  -(math.sin(r(ramc)) * math.cos(r(eps)) + math.tan(r(LAT)) * math.sin(r(eps))))) % 360
    mc = math.degrees(math.atan2(math.sin(r(ramc)), math.cos(r(ramc)) * math.cos(r(eps)))) % 360
    # srednji čvor
    rahu = (125.0445 - 0.05295377 * A["dani"]) % 360
    trop = dict(A["trop"])
    trop["Rahu"], trop["Ketu"] = rahu, (rahu + 180) % 360
    lag = tocka("Lagna", asc, ayan, None)
    lr = lag["rashi"]
    lag = tocka("Lagna", asc, ayan, lr)
    T_ = {"Lagna": lag, "MC": tocka("MC", mc, ayan, lr)}
    for g in GRAHE:
        T_[g] = tocka(g, trop[g], ayan, lr, A["retro"].get(g, False))
    # pančanga
    su, mo = T_["Sunce"]["sid"], T_["Mjesec"]["sid"]
    razlika = (mo - su) % 360
    tithi = int(razlika // 12) + 1
    paksha = "Šukla" if tithi <= 15 else "Krišna"
    tn = tithi if tithi <= 15 else tithi - 15
    tithi_ime = ("Purnima" if tithi == 15 else "Amavasja" if tithi == 30 else TITHI[tn - 1])
    yoga = int(((su + mo) % 360) // (40 / 3)) + 1
    kpos = int(razlika // 6)                     # 0..59
    if kpos == 0:
        karana, kidx = "Kimstughna", 0
    elif kpos >= 57:
        karana, kidx = ["Shakuni", "Chatushpada", "Naga"][kpos - 57], 0
    else:
        kidx = (kpos - 1) % 7 + 1
        karana = KARANA_POKRETNE[kidx - 1]
    # hora
    zh, zm, zs = A["zalaz"]
    ih, im, is_ = A["izlaz"]
    zalaz_min = zh * 60 + zm + zs / 60            # UTC
    izlaz_min = 24 * 60 + ih * 60 + im + is_ / 60
    noc = izlaz_min - zalaz_min
    sat_min = noc / 12
    hora_n = int((19 * 60 - zalaz_min) // sat_min) + 1       # 19:00 UTC; 1..12 noćni
    hora_idx = 12 + hora_n                                   # 13..24 u danu
    start = KALDEJSKI.index("Jupiter")                       # četvrtak
    hora_lord = KALDEJSKI[(start + hora_idx - 1) % 7]
    # Vimšotari daša trenutka
    m = T_["Mjesec"]
    frac = (m["sid"] - (m["nak"] - 1) * (40 / 3)) / (40 / 3)
    md = m["nak_lord"]
    md_y = DASHA_YRS[md]
    proslo = frac * md_y
    balans = md_y - proslo
    i0 = DASHA_ORDER.index(md)
    red = [DASHA_ORDER[(i0 + k) % 9] for k in range(9)]
    acc = 0.0
    ad, ad_start, ad_len = md, 0.0, 0.0
    for l in red:
        dl = md_y * DASHA_YRS[l] / 120
        if proslo < acc + dl:
            ad, ad_start, ad_len = l, acc, dl
            break
        acc += dl
    j0 = DASHA_ORDER.index(ad)
    red2 = [DASHA_ORDER[(j0 + k) % 9] for k in range(9)]
    acc2, pd = ad_start, ad
    for l in red2:
        dl = ad_len * DASHA_YRS[l] / 120
        if proslo < acc2 + dl:
            pd = l
            break
        acc2 += dl
    b_g = int(balans)
    b_m = int((balans - b_g) * 12)
    b_d = int(round(((balans - b_g) * 12 - b_m) * 30))
    # tara / čandra bala
    mn = m["nak"]
    tare = {t: [((mn - 1 + (t - 1) + 9 * c) % 27) + 1 for c in range(3)] for t in TARA_POVOLJNE}
    chandra = [((m["rashi"] - 1 + (p - 1)) % 12) + 1 for p in CHANDRA_POVOLJNI]
    kuce = {h: [g for g in GRAHE if T_[g]["kuca"] == h] for h in range(1, 13)}
    return {"A": A, "ayan": ayan, "eps": eps, "ramc": ramc, "T": T_, "lagna_rashi": lr, "kuce": kuce,
            "tithi": tithi, "paksha": paksha, "tn": tn, "tithi_ime": tithi_ime, "razlika": razlika,
            "yoga": yoga, "yoga_ime": YOGA[yoga - 1], "zbroj": (su + mo) % 360,
            "kpos": kpos + 1, "karana": karana, "kidx": kidx,
            "vara": 5, "vara_lord": "Jupiter", "hora_n": hora_n, "hora_idx": hora_idx, "hora_lord": hora_lord,
            "noc_min": noc, "sat_min": sat_min,
            "md": md, "md_y": md_y, "frac": frac, "proslo": proslo, "balans": (b_g, b_m, b_d), "ad": ad, "pd": pd,
            "tare": tare, "chandra": chandra,
            "mjesec_amanta": 7, "mjesec_purnimanta": 8, "vikram": 2083, "shaka": 1948}


# ------------------------------------------------------------------ kandidati
def _prvih6(seq) -> list[int] | None:
    out: list[int] = []
    for x in seq:
        if x is None:
            continue
        v = svedi(x)
        if v not in out:
            out.append(v)
        if len(out) == K:
            return sorted(out)
    return None


def _prozori(seq, w: int = 0) -> list[int] | None:
    """w-ti prozor od 6 različitih vrijednosti niza (0 = prvih 6, 1 = sljedećih 6 …)."""
    vals: list[int] = []
    for x in seq:
        v = svedi(x)
        if v not in vals:
            vals.append(v)
    blok = vals[w * K:(w + 1) * K]
    return sorted(blok) if len(blok) == K else None


def potpis(p: dict) -> list[int]:
    """Niz brojeva točke, od najjače derivacije prema slabijima."""
    return [p["ved"], p["nak"], p["deg"], p["rashi"], p["np45"], p["l45"], p["nav"], p["pada"], p["kuca"],
            p["nl_ved"], p["rl_ved"], p["nak"] + p["pada"], p["rashi"] + p["deg"], p["nak"] * p["pada"],
            p["rashi"] * p["nav"], p["deg"] + p["min"]]


def _isprepleti(*nizovi) -> list[int]:
    out: list[int] = []
    for i in range(max(map(len, nizovi))):
        for s in nizovi:
            if i < len(s):
                out.append(s[i])
    return out


def _fmt(t: list[int]) -> str:
    return " · ".join(f"{x:02d}" for x in sorted(t))


def kandidati(n: int = 80) -> list[dict]:
    """Najmanje 60 (ili n) međusobno različitih kandidata, rangiranih po logici metode."""
    n = max(n, 60)
    G = izracun()
    T = G["T"]
    J, M, L, V, Me, Su, Ma, Sa, Ra, Ke, MC = (T[k] for k in ("Jupiter", "Mjesec", "Lagna", "Venera", "Merkur", "Sunce",
                                                              "Mars", "Saturn", "Rahu", "Ketu", "MC"))
    out: list[dict] = []
    videni: set[tuple[int, ...]] = set()

    def dodaj(brojevi, obr: str) -> bool:
        if brojevi is None:
            return False
        t = tuple(sorted(brojevi))
        assert len(t) == K and len(set(t)) == K and all(1 <= x <= N for x in t), t
        if t in videni:
            return False
        videni.add(t)
        out.append({"brojevi": list(t), "rang": len(out) + 1, "obrazlozenje": obr})
        return True

    def opis(p: dict) -> str:
        return (f"{p['ime']} {dms(p['sid'] % 30)} {RASHI[p['rashi'] - 1].split(' ')[0]} (raši {p['rashi']}), "
                f"{NAKSH[p['nak'] - 1]} ({p['nak']}) pada {p['pada']}, D9 {p['nav']}, {p['kuca']}. kuća")

    # ---------- (1) Jupiter i Mjesec, pančanga, daša
    dodaj(_prvih6(potpis(J)), f"Jupiter (karaka bogatstva, gospodar dana i sata), uča u 2. kući: {opis(J)}; "
                               f"brojevi: vedski {J['ved']}, nakšatra {J['nak']}, stupanj {J['deg']}, raši {J['rashi']}, "
                               f"nakšatra-pada {J['np108']}→{J['np45']}, D9 {J['nav']}")
    dodaj(_prvih6(potpis(M)), f"Mjesec (um, pokazatelj prašne) u lagni: {opis(M)}; brojevi: vedski {M['ved']}, "
                               f"nakšatra {M['nak']}, stupanj {M['deg']}, raši {M['rashi']}, nakšatra-pada {M['np108']}, "
                               f"duljina {int(M['sid'])}°→{M['l45']}")
    dodaj(_prvih6(_isprepleti(potpis(J), potpis(M))),
          f"Jupiter + Mjesec (Sunapha yoga: Jupiter u 2. od Mjeseca) — isprepleteni potpisi")
    dodaj(_prvih6([VED[G["vara_lord"]], G["tithi"], M["nak"], G["yoga"], G["kpos"], G["hora_idx"], G["vara"], G["tn"],
                   G["kidx"], G["mjesec_amanta"], G["mjesec_purnimanta"], G["vikram"], G["shaka"]]),
          f"Pančanga: Guruvara (Jupiter {VED['Jupiter']}, vara {G['vara']}), tithi {G['tithi']} ({G['paksha']} "
          f"{G['tithi_ime']}), nakšatra {M['nak']} {NAKSH[M['nak'] - 1]}, yoga {G['yoga']} {G['yoga_ime']}, "
          f"karana {G['karana']} (položaj {G['kpos']}/60), hora {G['hora_lord']} ({G['hora_idx']}. sat)")
    dodaj(_prvih6([J["ved"], 4, VED["Mjesec"], J["deg"], J["nak"], J["nav"], Ma["ved"], Ma["deg"], J["np45"], J["l45"]]),
          f"Dhana (2.) kuća: Karka 4, gospodar Mjesec 2, Jupiter {J['deg']}°/{NAKSH[J['nak'] - 1]} {J['nak']}, "
          f"D9 {J['nav']}, suputnik Mars 9 ({Ma['deg']}°)")
    dodaj(_prvih6([VED[G["md"]], VED[G["ad"]], VED[G["pd"]], DASHA_YRS[G["md"]], DASHA_YRS[G["ad"]], DASHA_YRS[G["pd"]],
                   G["balans"][0], G["balans"][1], G["balans"][2]]),
          f"Vimšotari daša trenutka: {G['md']} {VED[G['md']]} / {G['ad']} {VED[G['ad']]} / {G['pd']} {VED[G['pd']]}, "
          f"godine {DASHA_YRS[G['md']]}, {DASHA_YRS[G['ad']]}, {DASHA_YRS[G['pd']]}")
    dodaj(_prvih6([G["balans"][0], G["balans"][1], G["balans"][2], VED[G["md"]], VED[G["ad"]], VED[G["pd"]], M["nak"],
                   int(round(G["frac"] * 100)), M["deg"]]),
          f"Balans daše {G['md']}: {G['balans'][0]} g {G['balans'][1]} mj {G['balans'][2]} d (prošlo "
          f"{G['frac'] * 100:.1f} %), gospodari {VED[G['md']]}/{VED[G['ad']]}/{VED[G['pd']]}")
    tare = G["tare"]
    dodaj(_prvih6([tare[2][0], tare[4][0], tare[6][0], tare[8][0], tare[9][0], tare[2][1], tare[4][1]]),
          f"Tara bala od {NAKSH[M['nak'] - 1]} ({M['nak']}): sampat {tare[2][0]}, kšema {tare[4][0]}, sadhana "
          f"{tare[6][0]}, mitra {tare[8][0]}, parama-mitra {tare[9][0]} + sampat 2. ciklusa {tare[2][1]}")
    dodaj(_prvih6(tare[2] + tare[4]), f"Tara bala: sampat {tare[2]} + kšema {tare[4]} (sva tri ciklusa)")
    dodaj(_prvih6(tare[6] + tare[8]), f"Tara bala: sadhana {tare[6]} + mitra {tare[8]}")
    dodaj(_prvih6(tare[9] + tare[2]), f"Tara bala: parama-mitra {tare[9]} + sampat {tare[2]}")
    dodaj(_prvih6(tare[8] + tare[4]), f"Tara bala: mitra {tare[8]} + kšema {tare[4]}")
    dodaj(_prvih6(G["chandra"]), f"Čandra bala: povoljni raši 1,3,6,7,10,11 brojano od {RASHI[M['rashi'] - 1]} → {G['chandra']}")
    dodaj(_prvih6([M["deg"], M["min"], M["nak"], M["pada"], G["tithi"], G["yoga"], G["kpos"], M["np45"], M["l45"]]),
          f"Mjesec po stupnju i minuti ({M['deg']}°{M['min']:02d}'), nakšatra {M['nak']} pada {M['pada']}, tithi "
          f"{G['tithi']}, yoga {G['yoga']}, karana {G['kpos']}")
    dodaj(_prvih6([J["ved"], G["hora_idx"], G["hora_n"], J["nak"], J["deg"], J["nav"], J["rashi"], G["vara"]]),
          f"Hora Jupitera: {G['hora_n']}. noćni = {G['hora_idx']}. sat dana, Jupiter 3, {NAKSH[J['nak'] - 1]} {J['nak']}, "
          f"{J['deg']}°, D9 {J['nav']}, raši {J['rashi']}")
    dodaj(_prvih6([J["ved"], M["ved"], J["nak"], M["nak"], J["np45"], M["np45"], J["l45"], M["l45"], J["nav"], M["nav"]]),
          "Jupiter + Mjesec: vedski brojevi, nakšatre, nakšatra-pada indeksi i duljine mod 45")
    # ---------- (2) lagna, 5. kuća, dhana/labha/bhagja, navamsa
    dodaj(_prvih6(potpis(L)), f"Lagna (prašna): {opis(L)}; gospodar Merkur {VED['Merkur']} u 5. kući")
    peta = G["kuce"][5]
    dodaj(_prvih6([7, V["ved"], V["deg"], V["nak"], Su["ved"], Me["ved"], Su["deg"], Me["deg"], V["np45"], Me["nak"]]),
          f"5. kuća (špekulacija) Tula 7: gospodar Venera 6 u vlastitom znaku ({V['deg']}°, {NAKSH[V['nak'] - 1]} "
          f"{V['nak']}), stanari {', '.join(peta)}: Sunce 1 ({Su['deg']}°), Merkur 5 ({Me['deg']}°)")
    dodaj(_prvih6(potpis(V)), f"Venera (gospodar 5. u 5., svakšetra, vargottama): {opis(V)}")
    dodaj(_prvih6(potpis(Me)), f"Merkur (gospodar lagne u 5. kući, Budha-Aditja yoga): {opis(Me)}")
    dodaj(_prvih6(_isprepleti(potpis(L), potpis(V))), "Lagna + Venera (gospodar 5.) — isprepleteni potpisi")
    dodaj(_prvih6(_isprepleti(potpis(L), potpis(Me))), "Lagna + Merkur (gospodar lagne) — isprepleteni potpisi")
    dodaj(_prvih6(_isprepleti(potpis(L), potpis(M))), "Lagna + Mjesec u lagni — isprepleteni potpisi")
    dodaj(_prvih6([V["deg"], Su["deg"], Me["deg"], V["np45"], Su["np45"], Me["np45"], V["nak"], Su["nak"], Me["nak"],
                   V["l45"], Su["l45"], Me["l45"]]),
          f"5. kuća po stupnjevima: Venera {V['deg']}, Sunce {Su['deg']}, Merkur {Me['deg']}; nakšatra-pada "
          f"{V['np45']}/{Su['np45']}/{Me['np45']}; nakšatre {V['nak']}/{Su['nak']}/{Me['nak']}")
    dodaj(_prvih6([1, Ma["ved"], Ma["rashi"], Ma["deg"], Ma["nak"], Ma["nav"], Ma["np45"], Ma["l45"], Ma["kuca"]]),
          f"11. kuća (labha, dobitak) Meša 1: gospodar Mars 9 u {RASHI[Ma['rashi'] - 1]} {Ma['deg']}°, {NAKSH[Ma['nak'] - 1]} "
          f"{Ma['nak']}, D9 {Ma['nav']} (uča u D9), nakšatra-pada {Ma['np45']}")
    dodaj(_prvih6([11, Sa["ved"], Ra["ved"], Ra["deg"], Ra["nak"], Sa["rashi"], Sa["deg"], Sa["nak"], Ra["np45"], Sa["kuca"]]),
          f"9. kuća (bhagja, sreća) Kumbha 11: gospodar Saturn 8 u {RASHI[Sa['rashi'] - 1]} {Sa['deg']}°, Rahu 4 u 9. "
          f"({Ra['deg']}°, {NAKSH[Ra['nak'] - 1]} {Ra['nak']})")
    dodaj(_prvih6([T[k]["nav"] for k in RANG_TOCAKA]),
          "Navamsa (D9) svih točaka po rangu snage: " + ", ".join(f"{k} {T[k]['nav']}" for k in RANG_TOCAKA[:7]))
    dodaj(_prvih6([T[k]["ved"] + T[k]["nav"] for k in RANG_TOCAKA]),
          "Vedski broj + navamsa svake točke (Jupiter 3+12, Mjesec 2+7, lagna 5+10, Venera 6+7 …)")
    dodaj(_prvih6([M["ved"], L["ved"], V["ved"], J["ved"], Ma["ved"], M["deg"], L["deg"], V["deg"], J["deg"]]),
          "Gospodari kuća bogatstva: 2. Mjesec 2, 1. Merkur 5, 5. Venera 6, 7./10. Jupiter 3, 11. Mars 9 + stupnjevi")
    # ---------- (3) ostale grahe i presjeci svih točaka
    for p, zasto in ((Su, "Sunce (niča u 5., Budha-Aditja)"), (Ma, "Mars (gospodar 11., niča-bhanga uz Jupitera)"),
                     (Sa, "Saturn R (gospodar 9. u 10.)"), (Ra, "Rahu (srednji čvor) u 9. kući"),
                     (Ke, "Ketu u 3. kući"), (MC, "MC (10. stupanj)")):
        dodaj(_prvih6(potpis(p)), f"{zasto}: {opis(p)}")
    presjeci = [("deg", "tekući stupnjevi u rašiju"), ("l45", "sideričke duljine mod 45"),
                ("np45", "indeksi nakšatra-pada (1..108 svedeno)"), ("nak", "nakšatre"),
                ("rashi", "raši"), ("ved", "vedski brojevi graha")]
    for w in (0, 1):
        for klj, naziv in presjeci:
            niz = [T[k][klj] for k in RANG_TOCAKA]
            if klj == "rashi":
                niz = _isprepleti([T[k]["rashi"] for k in RANG_TOCAKA], [T[k]["rashi"] + T[k]["deg"] for k in RANG_TOCAKA])
            if klj == "ved":
                niz = _isprepleti([T[k]["ved"] for k in RANG_TOCAKA], [T[k]["ved"] * T[k]["pada"] + T[k]["nak"] for k in RANG_TOCAKA])
            dodaj(_prozori(niz, w), f"Presjek svih točaka — {naziv}, {'prvih' if w == 0 else 'sljedećih'} 6 po rangu snage")
    dodaj(_prvih6([T[k]["nak"] + T[k]["pada"] for k in RANG_TOCAKA]), "Nakšatra + pada svake točke po rangu snage")
    dodaj(_prvih6([T[k]["nak"] * T[k]["pada"] for k in RANG_TOCAKA]), "Nakšatra × pada svake točke po rangu snage")
    dodaj(_prvih6([T[k]["deg"] + T[k]["min"] for k in RANG_TOCAKA]), "Stupanj + minute svake točke po rangu snage")
    dodaj(_prvih6([T[k]["rashi"] * T[k]["nav"] for k in RANG_TOCAKA]), "Raši × navamsa svake točke po rangu snage")
    # ---------- (4) parovi i trojke točaka po rangu snage
    for i, a in enumerate(RANG_TOCAKA):
        for b in RANG_TOCAKA[i + 1:]:
            dodaj(_prvih6(_isprepleti(potpis(T[a]), potpis(T[b]))), f"Par {a} + {b}: isprepleteni potpisi")
            if len(out) >= n + 20:
                break
    trojke = [("Jupiter", "Mjesec", "Lagna"), ("Jupiter", "Venera", "Merkur"), ("Mjesec", "Venera", "Merkur"),
              ("Jupiter", "Mjesec", "Venera"), ("Lagna", "Venera", "Merkur"), ("Jupiter", "Mars", "Mjesec"),
              ("Sunce", "Merkur", "Venera"), ("Lagna", "Mjesec", "Jupiter")]
    for a, b, c in trojke:
        dodaj(_prvih6(_isprepleti(potpis(T[a]), potpis(T[b]), potpis(T[c]))), f"Trojka {a} + {b} + {c}")
    for w in (1, 2):
        for k in RANG_TOCAKA:
            dodaj(_prozori(potpis(T[k]), w), f"{k}: {'drugi' if w == 1 else 'treći'} prozor potpisa (slabije derivacije)")
    return out[:n]


def kljucni_brojevi(k: int = 12) -> list[int]:
    """Težinski najjači brojevi čitanja (vedski broj, nakšatra, stupanj, raši svake točke po snazi + pančanga)."""
    G = izracun()
    T = G["T"]
    tez = {"Jupiter": 3, "Mjesec": 3, "Lagna": 2.5, "Venera": 2, "Merkur": 2, "Sunce": 1.5, "Mars": 1.5, "Saturn": 1,
           "Rahu": 1, "Ketu": 1, "MC": 0.5}
    w: Counter = Counter()
    for ime, t in tez.items():
        p = T[ime]
        w[svedi(p["ved"])] += t
        w[svedi(p["nak"])] += t
        w[svedi(p["deg"])] += t * 0.9
        w[svedi(p["rashi"])] += t * 0.8
        w[svedi(p["np45"])] += t * 0.6
        w[svedi(p["nav"])] += t * 0.4
    for x, t in ((G["tithi"], 2), (G["yoga"], 2), (G["kpos"], 1.5), (G["hora_idx"], 1.5), (VED[G["md"]], 1.5),
                 (VED[G["ad"]], 1), (VED[G["pd"]], 1), (G["vara"], 0.5)):
        w[svedi(x)] += t
    return [x for x, _ in sorted(w.items(), key=lambda kv: (-kv[1], kv[0]))[:k]]


# ------------------------------------------------------------------ postupak (tekst)
def postupak_tekst() -> str:
    G = izracun()
    T = G["T"]
    A = G["A"]
    M = T["Mjesec"]
    zh, zm, zs = A["zalaz"]
    ih, im, is_ = A["izlaz"]
    L = []
    L.append("# Jyotiš (vedska astrologija) — Loto 6/45, četvrtak 29.10.2026. 20:00 CET (87. kolo)\n")
    L.append("## 1. Trenutak i ayanamsa")
    L.append(f"- Trenutak: 2026-10-29 19:00:00 UTC (20:00 CET, ljetno vrijeme završilo 25.10.), Zagreb {LAT} N {LON} E.")
    L.append(f"- Efemeride: {A['izvor']}. d = {A['dani']:.4f} dana od J2000.0.")
    L.append(f"- Ayanamsa Lahiri = 23.853° + 50.29″ × {A['dani'] / 365.25:.3f} god = **{G['ayan']:.4f}° = {dms(G['ayan'])}** "
             f"(sve raši/nakšatre identične i s fiksnih 24.2°).")
    L.append(f"- LST {G['ramc'] / 15:.4f} h = RAMC {G['ramc']:.3f}°, ε = {G['eps']:.4f}° → lagna tropski "
             f"{T['Lagna']['trop']:.2f}° = {dms(T['Lagna']['trop'] % 30)} Rak, MC {T['MC']['trop']:.2f}° = {dms(T['MC']['trop'] % 30)} Ribe.")
    L.append(f"- Rahu (srednji čvor) = 125.0445° − 0.05295377° × {A['dani']:.2f} = {T['Rahu']['trop']:.3f}° tropski.\n")
    L.append("## 2. Siderički položaji (Lahiri)")
    L.append("| točka | tropski | siderički | raši | stupanj | nakšatra (gospodar) | pada | D9 | kuća | dostojanstvo |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for k in ["Lagna", "MC"] + GRAHE:
        p = T[k]
        L.append(f"| {k} | {p['trop']:.3f}° | {p['sid']:.3f}° | {RASHI[p['rashi'] - 1]} ({p['rashi']}) | "
                 f"{dms(p['sid'] % 30)} → {p['deg']} | {NAKSH[p['nak'] - 1]} ({p['nak']}, {p['nak_lord']}) | {p['pada']} | "
                 f"{p['nav']} | {p['kuca']} | {', '.join(p['dost']) or '—'} |")
    L.append("")
    L.append("## 3. Kuće (cijeli znakovi od lagne " + RASHI[G["lagna_rashi"] - 1] + ")")
    for h in range(1, 13):
        r = ((G["lagna_rashi"] - 1 + h - 1) % 12) + 1
        st = ", ".join(G["kuce"][h]) or "—"
        L.append(f"- {h}. kuća {RASHI[r - 1]} (gospodar {RASHI_LORD[r - 1]} {VED[RASHI_LORD[r - 1]]}): {st}")
    L.append("")
    L.append("## 4. Pančanga")
    L.append(f"- Vara: {VARA[4]} — vara {G['vara']}, gospodar Jupiter (vedski 3).")
    L.append(f"- Tithi: Mjesec − Sunce = {G['razlika']:.2f}° / 12° → tithi **{G['tithi']}** = {G['paksha']} {G['tithi_ime']} "
             f"({G['paksha']} pakša, {G['tn']}. dan).")
    L.append(f"- Nakšatra Mjeseca: {M['sid']:.3f}° / 13°20' → **{NAKSH[M['nak'] - 1]} ({M['nak']})**, pada {M['pada']}, "
             f"gospodar {M['nak_lord']} ({VED[M['nak_lord']]}); prošlo {G['frac'] * 100:.2f} % nakšatre.")
    L.append(f"- Yoga: (Sunce + Mjesec) siderički = {G['zbroj']:.2f}° / 13°20' → **{G['yoga']} {G['yoga_ime']}**.")
    L.append(f"- Karana: {G['razlika']:.2f}° / 6° → položaj {G['kpos']}/60 → **{G['karana']}**"
             + (f" ({G['kidx']}. pokretna)" if G["kidx"] else "") + ".")
    L.append(f"- Hora: zalaz 29.10. {zh + 1:02d}:{zm:02d}:{zs:02d} CET, izlaz 30.10. {ih + 1:02d}:{im:02d}:{is_:02d} CET, "
             f"noć {G['noc_min']:.1f} min, sat {G['sat_min']:.2f} min → 20:00 je {G['hora_n']}. noćni = {G['hora_idx']}. sat "
             f"dana → **hora {G['hora_lord']}**.")
    L.append(f"- Lunarni mjesec: Ašvina krišna (amanta, {G['mjesec_amanta']}) / Kartika krišna (purnimanta, "
             f"{G['mjesec_purnimanta']}); Vikram samvat {G['vikram']} (mod 45 = {svedi(G['vikram'])}), Šaka {G['shaka']} "
             f"(mod 45 = {svedi(G['shaka'])}).\n")
    L.append("## 5. Vimšotari daša trenutka (prašna)")
    L.append(f"- Mahadaša {G['md']} ({DASHA_YRS[G['md']]} g), prošlo {G['proslo']:.3f} g, balans {G['balans'][0]} g "
             f"{G['balans'][1]} mj {G['balans'][2]} d; antardaša {G['ad']} ({DASHA_YRS[G['ad']]}), pratjantardaša {G['pd']} "
             f"({DASHA_YRS[G['pd']]}) → brojevi {VED[G['md']]}, {VED[G['ad']]}, {VED[G['pd']]}.")
    L.append(f"- Tara bala od {NAKSH[M['nak'] - 1]}: " + "; ".join(f"{TARA[t - 1]} {G['tare'][t]}" for t in TARA_POVOLJNE) + ".")
    L.append(f"- Čandra bala od {RASHI[M['rashi'] - 1]}: raši {G['chandra']}.\n")
    L.append("## 6. Tumačenje")
    L.append("- Lagna Mithuna, Mjesec u lagni (jaka prašna: pitanje je 'u umu'). Gospodar lagne Merkur u 5. kući "
             "(špekulacija, lutrija) zajedno s gospodarom 5. Venerom u vlastitom znaku (vargottama) i Suncem — "
             "Budha-Aditja yoga u kući dobitaka iz igre.")
    L.append("- Jupiter, karaka bogatstva, egzaltiran u Karki u 2. kući (dhana) i to na svoj dan (Guruvara) u svoj "
             "planetarni sat; nalazi se u 2. od Mjeseca (Sunapha yoga). Mars (gospodar 11. = labha) u istom znaku: "
             "niča, ali niča-bhanga (egzaltirani Jupiter u istom znaku) i uča u navamsi.")
    L.append("- Rahu u 9. (bhagja), Saturn retrogradan u 10., Ketu u 3. Sunce u padu u 5. — ograničenje, ne poništenje.")
    L.append("- Zato rang: Jupiter i Mjesec (s pančangom, hora, daša) → lagna / 5. / 2. / 11. / 9. kuća → ostalo.\n")
    L.append("## 7. Bazen brojeva i pravila gradnje")
    L.append("- Potpis točke (redom): vedski broj, nakšatra, tekući stupanj, raši, nakšatra-pada indeks (1..108 → 1..45), "
             "siderička duljina mod 45, navamsa, pada, kuća, broj gospodara nakšatre, broj gospodara rašija, "
             "nakšatra+pada, raši+stupanj, nakšatra×pada, raši×navamsa, stupanj+minute.")
    L.append("- Kandidat = prvih 6 različitih vrijednosti niza (sve svedeno u 1..45). Parovi/trojke: potpisi se isprepliću "
             "(1. broj A, 1. broj B, 2. broj A …). Presjeci: ista veličina svih 11 točaka po rangu snage.")
    L.append("- Potpisi: " + "; ".join(f"{k} {potpis(T[k])[:8]}" for k in RANG_TOCAKA) + ".\n")
    kand = kandidati(80)
    L.append(f"## 8. Kandidati ({len(kand)}; prvih 20)")
    for c in kand[:20]:
        L.append(f"{c['rang']:3d}. {_fmt(c['brojevi'])} — {c['obrazlozenje']}")
    L.append("\nKljučni brojevi: " + ", ".join(map(str, kljucni_brojevi())))
    return "\n".join(L)


# ------------------------------------------------------------------ provjere
def _provjere() -> None:
    G = izracun()
    T = G["T"]
    assert 24.1 < G["ayan"] < 24.3, G["ayan"]
    assert T["Mjesec"]["nak"] == 5 and T["Mjesec"]["rashi"] == 3, (T["Mjesec"]["nak"], T["Mjesec"]["rashi"])
    assert T["Jupiter"]["rashi"] == 4 and "uča (egzaltacija)" in T["Jupiter"]["dost"]
    assert T["Lagna"]["rashi"] == 3 and T["Venera"]["kuca"] == 5
    assert G["tithi"] == 20 and G["yoga"] == 20 and G["hora_lord"] == "Jupiter"
    # neosjetljivost na fiksnu ayanamsu 24.2°
    for k in ["Lagna", "MC"] + GRAHE:
        p = T[k]
        q = tocka(k, p["trop"], 24.2, G["lagna_rashi"])
        assert (p["rashi"], p["nak"], p["pada"]) == (q["rashi"], q["nak"], q["pada"]), (k, p["sid"])
    kand = kandidati(80)
    assert len(kand) >= 60, len(kand)
    assert len({tuple(c["brojevi"]) for c in kand}) == len(kand)
    koristeni = {x for c in kand[:60] for x in c["brojevi"]}
    assert len(koristeni) >= 30, len(koristeni)


# ------------------------------------------------------------------ main
if __name__ == "__main__":
    _provjere()
    tekst = postupak_tekst()
    print(tekst)
    kand = kandidati(80)
    print("\n## Prvih 10 kandidata\n")
    for c in kand[:10]:
        print(f"{c['rang']:3d}. {_fmt(c['brojevi'])}  — {c['obrazlozenje']}")
    kljuc = kljucni_brojevi()
    print("\nKljučni brojevi:", kljuc)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"ezo_{SLUG}.json"
    out.write_text(json.dumps({"metoda": METODA, "slug": SLUG, "postupak": tekst,
                               "kljucni_brojevi": kljuc, "kandidati": kand},
                              ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nZapisano: {out}  ({len(kand)} kandidata)")
