"""Astronomske i astroloske funkcije registra Loto 6/45 (pyswisseph, Moshier efemerida).

Konvencije:
- Sve pozicije se racunaju iskljucivo s flagom swe.FLG_MOSEPH (bez datoteka efemerida).
- Sideralne pozicije koriste ayanamsu Lahiri (swe.SIDM_LAHIRI). Buduci da je swe.set_sid_mode
  globalno stanje, poziva se u svakoj funkciji koja ga treba.
- Ulazni datetime mora biti aware (s vremenskom zonom); izlazni datetimei su aware u Europe/Zagreb
  (config.TZ), osim true_solar_time koji vraca naive datetime (pravo suncevo vrijeme).
- Napomena o Pythonu: razlika dvaju aware datetimea s ISTIM tzinfo objektom (npr. oba u Europe/Zagreb)
  racuna se po zidnom satu (ignorira promjenu ljetnog vremena). Interno se zato sve razlike racunaju
  u UTC-u; tko zeli trajanja iz vracenih vrijednosti, neka ih prije oduzimanja pretvori u UTC.
"""
from __future__ import annotations

import datetime as dt
import functools

import swisseph as swe

from config import TZ

UTC = dt.timezone.utc
EPH_FLAGS: int = swe.FLG_MOSEPH          # Moshier, bez datoteka efemerida
YEAR_DAYS: float = 365.25                # godina u Vimsottari sustavu
NAKSHATRA_SPAN: float = 360.0 / 27.0     # 13 st 20 min
PADA_SPAN: float = NAKSHATRA_SPAN / 4.0  # 3 st 20 min

# swe konstanta -> hrvatsko ime planeta
PLANET_HR: dict[int, str] = {
    swe.SUN: "Sunce", swe.MOON: "Mjesec", swe.MERCURY: "Merkur", swe.VENUS: "Venera",
    swe.MARS: "Mars", swe.JUPITER: "Jupiter", swe.SATURN: "Saturn",
}
HR_PLANET: dict[str, int] = {v: k for k, v in PLANET_HR.items()}

# weekday (Mon=0..Sun=6) -> swe konstanta vladara dana
DAY_RULER_BODY: dict[int, int] = {
    0: swe.MOON, 1: swe.MARS, 2: swe.MERCURY, 3: swe.JUPITER, 4: swe.VENUS, 5: swe.SATURN, 6: swe.SUN,
}

# kaldejski red planetarnih sati (ciklicki)
CHALDEAN_ORDER: list[int] = [swe.SATURN, swe.JUPITER, swe.MARS, swe.SUN, swe.VENUS, swe.MERCURY, swe.MOON]

# gospodari nakshatri od Asvini (ciklicki, 3 x 9 = 27); isti red je i red Vimsottari dasa
NAKSHATRA_LORDS: list[str] = ["Ketu", "Venera", "Sunce", "Mjesec", "Mars", "Rahu", "Jupiter", "Saturn", "Merkur"]
DASHA_ORDER: list[str] = list(NAKSHATRA_LORDS)
DASHA_YEARS: dict[str, int] = {
    "Ketu": 7, "Venera": 20, "Sunce": 6, "Mjesec": 10, "Mars": 7, "Rahu": 18,
    "Jupiter": 16, "Saturn": 19, "Merkur": 17,
}
NAKSHATRA_NAMES: list[str] = [
    "Asvini", "Bharani", "Krittika", "Rohini", "Mrigasira", "Ardra", "Punarvasu", "Pusya", "Aslesa",
    "Magha", "Purva Phalguni", "Uttara Phalguni", "Hasta", "Citra", "Svati", "Visakha", "Anuradha",
    "Jyestha", "Mula", "Purva Asadha", "Uttara Asadha", "Sravana", "Dhanistha", "Satabhisa",
    "Purva Bhadrapada", "Uttara Bhadrapada", "Revati",
]
SIGN_NAMES_HR: list[str] = [
    "Ovan", "Bik", "Blizanci", "Rak", "Lav", "Djevica", "Vaga", "Skorpion", "Strijelac", "Jarac",
    "Vodenjak", "Ribe",
]


# ----------------------------------------------------------------------------------------------
# vrijeme
# ----------------------------------------------------------------------------------------------
def _as_utc(d: dt.datetime, name: str = "dt") -> dt.datetime:
    """Provjeri da je datetime aware i vrati ga pretvorenog u UTC."""
    if not isinstance(d, dt.datetime) or d.tzinfo is None or d.utcoffset() is None:
        raise ValueError(f"{name} mora biti aware datetime (s vremenskom zonom): {d!r}")
    return d.astimezone(UTC)


def jd_ut(d: dt.datetime) -> float:
    """Aware datetime -> julijanski dan (UT, gregorijanski kalendar)."""
    u = _as_utc(d)
    hour = u.hour + u.minute / 60.0 + u.second / 3600.0 + u.microsecond / 3.6e9
    return swe.julday(u.year, u.month, u.day, hour, swe.GREG_CAL)


def datetime_from_jd(jd: float, tz: dt.tzinfo = TZ) -> dt.datetime:
    """Julijanski dan (UT) -> aware datetime u zadanoj zoni (zadano Europe/Zagreb)."""
    y, m, d, h = swe.revjul(float(jd), swe.GREG_CAL)
    base = dt.datetime(int(y), int(m), int(d), tzinfo=UTC) + dt.timedelta(hours=float(h))
    return base.astimezone(tz)


# ----------------------------------------------------------------------------------------------
# polozaji
# ----------------------------------------------------------------------------------------------
def tropical_lon(jd: float, body: int) -> float:
    """Tropska ekliptika duzina tijela (0..360), Moshier."""
    xx, _ = swe.calc_ut(jd, body, EPH_FLAGS)
    return xx[0] % 360.0


def sidereal_lon(jd: float, body: int) -> float:
    """Sideralna ekliptika duzina tijela (Lahiri, 0..360), Moshier."""
    swe.set_sid_mode(swe.SIDM_LAHIRI)
    xx, _ = swe.calc_ut(jd, body, EPH_FLAGS | swe.FLG_SIDEREAL)
    return xx[0] % 360.0


def ayanamsa(jd: float) -> float:
    """Ayanamsa Lahiri (stupnjevi) za zadani JD UT, istim ephemeris flagom kao i polozaji."""
    swe.set_sid_mode(swe.SIDM_LAHIRI)
    _, ay = swe.get_ayanamsa_ex_ut(jd, EPH_FLAGS)
    return float(ay)


def _houses(jd: float, lat: float, lon: float, sidereal: bool) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Kuce po Placidusu ('P'); vraca (cusps, ascmc). Sideralno = Lahiri."""
    flags = 0
    if sidereal:
        swe.set_sid_mode(swe.SIDM_LAHIRI)
        flags = swe.FLG_SIDEREAL
    cusps, ascmc = swe.houses_ex(jd, float(lat), float(lon), b"P", flags)
    return cusps, ascmc


def ascendant(jd: float, lat: float, lon: float, sidereal: bool = False) -> float:
    """Ascendent (Placidus 'P'), tropski ili sideralni (Lahiri), 0..360."""
    return _houses(jd, lat, lon, sidereal)[1][0] % 360.0


def midheaven(jd: float, lat: float, lon: float, sidereal: bool = False) -> float:
    """Medium coeli (MC), tropski ili sideralni (Lahiri), 0..360."""
    return _houses(jd, lat, lon, sidereal)[1][1] % 360.0


def positions(jd: float, lat: float, lon: float, sidereal: bool = False) -> dict[str, float]:
    """Duzine 7 klasicnih planeta + ASC + MC (hr imena -> stupnjevi), tropski ili sideralno."""
    f = sidereal_lon if sidereal else tropical_lon
    out = {PLANET_HR[b]: f(jd, b) for b in (swe.SUN, swe.MOON, swe.MERCURY, swe.VENUS,
                                             swe.MARS, swe.JUPITER, swe.SATURN)}
    out["ASC"] = ascendant(jd, lat, lon, sidereal)
    out["MC"] = midheaven(jd, lat, lon, sidereal)
    return out


# ----------------------------------------------------------------------------------------------
# izlazak / zalazak, dan / noc, planetarni sati
# ----------------------------------------------------------------------------------------------
@functools.lru_cache(maxsize=4096)
def _sunrise_sunset_cached(day: dt.date, lat: float, lon: float) -> tuple[dt.datetime, dt.datetime]:
    jd_start = jd_ut(dt.datetime.combine(day, dt.time(0, 0), tzinfo=TZ))  # lokalna ponoc -> UT
    geopos = (float(lon), float(lat), 0.0)
    found: list[dt.datetime] = []
    for rsmi, label in ((swe.CALC_RISE, "izlazak"), (swe.CALC_SET, "zalazak")):
        ret, tret = swe.rise_trans(jd_start, swe.SUN, rsmi, geopos, 0.0, 0.0, EPH_FLAGS)
        if ret < 0:
            raise ValueError(f"Sunce nema {label} na dan {day} (lat={lat}, lon={lon}): polarni dan/noc")
        t = datetime_from_jd(tret[0])
        if t.date() != day:
            raise ValueError(f"{label} Sunca ({t}) nije na lokalnom datumu {day} (lat={lat}, lon={lon})")
        found.append(t)
    rise, sset = found
    if not rise < sset:
        raise ValueError(f"izlazak {rise} nije prije zalaska {sset} na dan {day}")
    return rise, sset


def sunrise_sunset(day: dt.date, lat: float, lon: float) -> tuple[dt.datetime, dt.datetime]:
    """Izlazak i zalazak Sunca ISTOG lokalnog dana (Europe/Zagreb), aware datetimei.

    swe.rise_trans s rsmi = CALC_RISE / CALC_SET bez dodatnih bitova = gornji rub diska + refrakcija,
    atpress = attemp = 0, nadmorska visina 0 m; pocetak trazenja = lokalna ponoc tog dana.
    Podize ValueError ako dogadjaj ne postoji ili nije na tom lokalnom datumu.
    """
    if isinstance(day, dt.datetime):
        day = day.date()
    return _sunrise_sunset_cached(day, float(lat), float(lon))


def is_day(d: dt.datetime, lat: float, lon: float) -> bool:
    """True ako je Sunce iznad horizonta: izlazak <= dt < zalazak tog lokalnog dana (Europe/Zagreb)."""
    u = _as_utc(d)
    rise, sset = sunrise_sunset(u.astimezone(TZ).date(), lat, lon)
    return rise <= u < sset


def planetary_hours(day: dt.date, lat: float, lon: float) -> list[dict]:
    """Svih 24 planetarnih sati planetarnog dana koji pocinje izlaskom Sunca na lokalni datum `day`.

    Sati 1..12 = (izlazak -> zalazak)/12, sati 13..24 = (zalazak -> sljedeci izlazak)/12.
    Prvi sat vlada vladar dana (DAY_RULER_BODY[weekday]), dalje kaldejski red
    Saturn, Jupiter, Mars, Sunce, Venera, Merkur, Mjesec ciklicki. Granice su izracunate u UTC-u,
    susjedni sati dijele istu granicu, sat 1 pocinje tocno na izlazak, sat 24 zavrsava tocno na
    sljedeci izlazak. Svaki element je dict istog oblika kao rezultat planetary_hour().
    """
    if isinstance(day, dt.datetime):
        day = day.date()
    rise, sset = sunrise_sunset(day, lat, lon)
    next_rise, _ = sunrise_sunset(day + dt.timedelta(days=1), lat, lon)
    r, s, n = rise.astimezone(UTC), sset.astimezone(UTC), next_rise.astimezone(UTC)
    day_bounds = [r + (s - r) * i / 12 for i in range(13)]
    night_bounds = [s + (n - s) * i / 12 for i in range(13)]
    ruler = DAY_RULER_BODY[day.weekday()]
    k = CHALDEAN_ORDER.index(ruler)
    hours: list[dict] = []
    for i in range(24):
        if i < 12:
            start, end = day_bounds[i], day_bounds[i + 1]
        else:
            start, end = night_bounds[i - 12], night_bounds[i - 11]
        body = CHALDEAN_ORDER[(k + i) % 7]
        hours.append({
            "planet": PLANET_HR[body],
            "body": body,
            "hour_index": i + 1,
            "day_ruler": PLANET_HR[ruler],
            "is_day_hour": i < 12,
            "start": start.astimezone(TZ),
            "end": end.astimezone(TZ),
            "sunrise": rise,
            "sunset": sset,
            "next_sunrise": next_rise,
            "planetary_day_date": day,
        })
    return hours


def planetary_hour(d: dt.datetime, lat: float, lon: float) -> dict:
    """Planetarni sat u trenutku `d` (aware) za zadano mjesto.

    Kljucevi: planet (hr), hour_index (1..24 od izlaska), day_ruler (hr), is_day_hour, start, end,
    sunrise, sunset, next_sunrise (aware, Europe/Zagreb), planetary_day_date (date), body (swe konst.).
    Planetarni dan traje od izlaska do sljedeceg izlaska; trenutak prije izlaska tog civilnog dana
    pripada prethodnom planetarnom danu (njegovi nocni sati 13..24).
    """
    u = _as_utc(d)
    day = u.astimezone(TZ).date()
    rise, _ = sunrise_sunset(day, lat, lon)
    if u < rise:
        day -= dt.timedelta(days=1)
    for h in planetary_hours(day, lat, lon):
        if h["start"] <= u < h["end"]:
            return dict(h)
    raise RuntimeError(f"planetary_hour: trenutak {d} nije pokriven planetarnim danom {day}")


# ----------------------------------------------------------------------------------------------
# Jyotis
# ----------------------------------------------------------------------------------------------
def nakshatra(lon_sid: float) -> tuple[int, int, str]:
    """Sideralna duzina -> (nakshatra 1..27, pada 1..4, gospodar hr).

    27 jednakih dijelova po 13 st 20 min od 0 Ovna (Asvini); pada = cetvrtina (3 st 20 min);
    gospodari ciklicki Ketu, Venera, Sunce, Mjesec, Mars, Rahu, Jupiter, Saturn, Merkur.
    """
    lon = float(lon_sid) % 360.0
    q = min(int(lon * 108.0 / 360.0), 107)  # cetvrtina nakshatre 0..107 (tocno na granicama, bez 360/27 zaokruzivanja)
    n = q // 4 + 1
    pada = q % 4 + 1
    return n, pada, NAKSHATRA_LORDS[(n - 1) % 9]


def rashi(lon: float) -> int:
    """Znak (rasi) 1..12 iz duzine (obicno sideralne)."""
    return int((float(lon) % 360.0) // 30.0) + 1


def vimshottari(natal_moon_sid: float, birth: dt.datetime, at: dt.datetime) -> dict:
    """Vimsottari dasa: mahadasa i antardasa koje vrijede u trenutku `at`.

    Ostatak prve dase = trajanje_gospodara * (1 - prijedjeni_udio_nakshatre); godina = 365.25 dana.
    Dalje redom Ketu 7, Venera 20, Sunce 6, Mjesec 10, Mars 7, Rahu 18, Jupiter 16, Saturn 19, Merkur 17.
    Antardase unutar MD pocinju od MD gospodara, trajanje AD = MD_godine * AD_godine / 120 (u godinama).
    Za prvu (krnju) MD antardase se polazu od zamisljenog pocetka pune MD (md_end - MD_godine), pa se
    one zavrsene prije rodjenja izostavljaju, a pocetak prve vidljive AD se reze na trenutak rodjenja.
    Sva aritmetika u UTC-u; izlazni datetimei su aware u Europe/Zagreb.

    Vraca dict: mahadasha, antardasha (hr imena), md_start, md_end, ad_start, ad_end, balance_years,
    balance_lord, nakshatra, pada, mahadashas (lista (gospodar, start, end): ostatak + 8 punih = prvi
    ciklus od 120 g, produzen dalje ako je `at` iza njega), antardashas (lista (gospodar, start, end)
    unutar tekuce MD).
    """
    b = _as_utc(birth, "birth")
    a = _as_utc(at, "at")
    if a < b:
        raise ValueError(f"at ({at}) je prije rodjenja ({birth})")
    n, pada, lord = nakshatra(natal_moon_sid)
    frac = (float(natal_moon_sid) % 360.0) * 27.0 / 360.0 - (n - 1)  # prijedjeni udio nakshatre
    frac = min(max(frac, 0.0), 1.0)
    balance_years = DASHA_YEARS[lord] * (1.0 - frac)
    k = DASHA_ORDER.index(lord)
    mds: list[tuple[str, dt.datetime, dt.datetime]] = []
    start = b
    end = b + dt.timedelta(days=balance_years * YEAR_DAYS)
    mds.append((lord, start, end))
    i = 1
    while len(mds) < 9 or mds[-1][2] <= a:
        if len(mds) >= 27:
            raise ValueError("at je vise od tri Vimsottari ciklusa (360 g) nakon rodjenja")
        L = DASHA_ORDER[(k + i) % 9]
        start, end = end, end + dt.timedelta(days=DASHA_YEARS[L] * YEAR_DAYS)
        mds.append((L, start, end))
        i += 1
    md_lord, md_start, md_end = next(md for md in mds if md[1] <= a < md[2])
    md_years = DASHA_YEARS[md_lord]
    notional_start = md_end - dt.timedelta(days=md_years * YEAR_DAYS)
    j = DASHA_ORDER.index(md_lord)
    ads: list[tuple[str, dt.datetime, dt.datetime]] = []
    s = notional_start
    for t in range(9):
        L = DASHA_ORDER[(j + t) % 9]
        e = s + dt.timedelta(days=md_years * DASHA_YEARS[L] / 120.0 * YEAR_DAYS)
        if t == 8:
            e = md_end  # ukloni mikrosekundno zaokruzivanje
        ads.append((L, s, e))
        s = e
    ads = [(L, max(s_, md_start), e_) for (L, s_, e_) in ads if e_ > md_start]
    ad_lord, ad_start, ad_end = next(ad for ad in ads if ad[1] <= a < ad[2])
    loc = lambda x: x.astimezone(TZ)  # noqa: E731
    return {
        "mahadasha": md_lord,
        "antardasha": ad_lord,
        "md_start": loc(md_start),
        "md_end": loc(md_end),
        "ad_start": loc(ad_start),
        "ad_end": loc(ad_end),
        "balance_years": balance_years,
        "balance_lord": lord,
        "nakshatra": n,
        "pada": pada,
        "mahadashas": [(L, loc(s_), loc(e_)) for (L, s_, e_) in mds],
        "antardashas": [(L, loc(s_), loc(e_)) for (L, s_, e_) in ads],
    }


# ----------------------------------------------------------------------------------------------
# razno
# ----------------------------------------------------------------------------------------------
def pars_fortunae(asc: float, sun: float, moon: float, is_day: bool) -> float:
    """Tocka srece: dan = ASC + Mjesec - Sunce, noc = ASC + Sunce - Mjesec (mod 360)."""
    if is_day:
        return (asc + moon - sun) % 360.0
    return (asc + sun - moon) % 360.0


def equation_of_time_minutes(jd: float) -> float:
    """Jednadzba vremena u minutama (pravo - srednje suncevo vrijeme); swe.time_equ vraca dane."""
    r = swe.time_equ(jd)
    e = r[0] if isinstance(r, (tuple, list)) else r
    return float(e) * 1440.0


def true_solar_time(d: dt.datetime, lon: float) -> dt.datetime:
    """Pravo suncevo vrijeme (naive datetime) = UT + lon/15 h + jednadzba vremena."""
    u = _as_utc(d)
    eot_min = equation_of_time_minutes(jd_ut(u))
    return u.replace(tzinfo=None) + dt.timedelta(hours=float(lon) / 15.0) + dt.timedelta(minutes=eot_min)


def deg_in_sign(lon: float) -> int:
    """Stupanj unutar znaka 1..30 = floor(lon % 30) + 1."""
    return int((float(lon) % 360.0) % 30.0) + 1


def sign_no(lon: float) -> int:
    """Redni broj znaka 1..12 (Ovan = 1)."""
    return int((float(lon) % 360.0) // 30.0) + 1


def format_lon(lon: float) -> str:
    """Duzina kao 'DdMMm Znak', npr. 69.72 -> '9d43m Blizanci'."""
    x = float(lon) % 360.0
    s = sign_no(x)
    d_in = x - (s - 1) * 30.0
    deg = int(d_in)
    minutes = int(round((d_in - deg) * 60.0))
    if minutes == 60:
        deg, minutes = deg + 1, 0
    return f"{deg}d{minutes:02d}m {SIGN_NAMES_HR[s - 1]}"


# ----------------------------------------------------------------------------------------------
# selftest
# ----------------------------------------------------------------------------------------------
def selftest() -> None:
    """Provjere na natalnim podacima 1991-05-31 18:20 Zagreb/Zadar i kolu 2026-11-01 20:00 Zagreb."""
    def ts(x: dt.datetime) -> str:
        return x.strftime("%Y-%m-%d %H:%M:%S %Z")

    def ds(x: dt.datetime) -> str:
        return x.strftime("%Y-%m-%d")

    natal = dt.datetime(1991, 5, 31, 18, 20, tzinfo=TZ)
    assert natal.utcoffset() == dt.timedelta(hours=2), natal.utcoffset()
    lat, lon = 44.1167, 15.2333
    jd = jd_ut(natal)
    assert abs(jd_ut(dt.datetime(2000, 1, 1, 12, tzinfo=UTC)) - 2451545.0) < 1e-9
    assert abs((datetime_from_jd(jd) - natal).total_seconds()) < 0.01
    print("=== astro selftest ===")
    print(f"Natal: {ts(natal)}  JD UT {jd:.6f}  lat {lat} lon {lon}")

    ay = ayanamsa(jd)
    trop = positions(jd, lat, lon, sidereal=False)
    sid = positions(jd, lat, lon, sidereal=True)
    print(f"Ayanamsa Lahiri: {ay:.4f}")
    print(f"{'Tijelo':8s} {'tropski':>9s}  {'':16s} {'sideralno':>9s}")
    for name in trop:
        print(f"{name:8s} {trop[name]:9.4f}  {format_lon(trop[name]):16s} {sid[name]:9.4f}  {format_lon(sid[name])}")
        assert abs(((trop[name] - ay) - sid[name] + 180.0) % 360.0 - 180.0) < 1e-3, name

    sun, moon, asc = trop["Sunce"], trop["Mjesec"], trop["ASC"]
    assert 69.5 <= sun <= 70.0, sun
    assert abs(moon - 284.58) <= 0.1, moon
    assert 224.5 <= asc <= 225.3, asc
    assert sign_no(sun) == 3 and deg_in_sign(sun) == 10, (sign_no(sun), deg_in_sign(sun))
    assert sign_no(asc) == 8, sign_no(asc)
    assert deg_in_sign(69.72) == 10 and sign_no(69.72) == 3 and rashi(69.72) == 3
    assert deg_in_sign(0.0) == 1 and deg_in_sign(359.999) == 30 and sign_no(359.999) == 12
    assert format_lon(69.72) == "9d43m Blizanci", format_lon(69.72)

    moon_sid = sid["Mjesec"]
    n, pada, lord = nakshatra(moon_sid)
    print(f"Mjesec sideralno {moon_sid:.4f}: nakshatra {n} {NAKSHATRA_NAMES[n - 1]}, pada {pada}, gospodar {lord}; "
          f"rasi {rashi(moon_sid)} ({SIGN_NAMES_HR[rashi(moon_sid) - 1]})")
    assert n == 20 and lord == "Venera", (n, lord)
    assert nakshatra(0.0) == (1, 1, "Ketu")
    assert nakshatra(359.999) == (27, 4, "Merkur")
    assert nakshatra(13.34) == (2, 1, "Venera")
    assert nakshatra(253.34) == (20, 1, "Venera")
    assert nakshatra(253.33) == (19, 4, "Ketu")
    assert nakshatra(120.0) == (10, 1, "Ketu")

    at = dt.datetime(2026, 11, 1, 20, 0, tzinfo=TZ)
    v = vimshottari(moon_sid, natal, at)
    print(f"Ostatak prve dase ({v['balance_lord']}): {v['balance_years']:.4f} g")
    print("Vimsottari mahadase:")
    for L, s, e in v["mahadashas"]:
        print(f"  MD {L:8s} {ds(s)} .. {ds(e)}")
    print(f"Na {ts(at)}: MD {v['mahadasha']} ({ds(v['md_start'])} .. {ds(v['md_end'])}) / "
          f"AD {v['antardasha']} ({ds(v['ad_start'])} .. {ds(v['ad_end'])})")
    print(f"Antardase unutar MD {v['mahadasha']}:")
    for L, s, e in v["antardashas"]:
        print(f"  AD {L:8s} {ds(s)} .. {ds(e)}")
    assert v["mahadasha"] == "Rahu" and v["antardasha"] == "Jupiter", (v["mahadasha"], v["antardasha"])
    assert 8.6 <= v["balance_years"] <= 8.9, v["balance_years"]
    assert v["md_start"] <= at < v["md_end"] and v["ad_start"] <= at < v["ad_end"]
    assert v["md_start"] <= v["ad_start"] and v["ad_end"] <= v["md_end"]
    mds = v["mahadashas"]
    assert mds[0][0] == "Venera" and mds[0][1] == natal and len(mds) >= 9
    assert [L for L, _, _ in mds[:9]] == ["Venera", "Sunce", "Mjesec", "Mars", "Rahu", "Jupiter", "Saturn", "Merkur", "Ketu"]
    for (_, _, e1), (_, s2, _) in zip(mds, mds[1:]):
        assert e1 == s2
    total_days = (mds[8][2].astimezone(UTC) - natal.astimezone(UTC)).total_seconds() / 86400.0
    assert abs(total_days - (120 - (20 - v["balance_years"])) * YEAR_DAYS) < 0.01, total_days
    ads = v["antardashas"]
    assert [L for L, _, _ in ads] == ["Rahu", "Jupiter", "Saturn", "Merkur", "Ketu", "Venera", "Sunce", "Mjesec", "Mars"]
    assert ads[0][1] == v["md_start"] and ads[-1][2] == v["md_end"]
    jup_days = (ads[1][2].astimezone(UTC) - ads[1][1].astimezone(UTC)).total_seconds() / 86400.0
    assert abs(jup_days - 18 * 16 / 120 * YEAR_DAYS) < 0.01, jup_days
    # krnja prva MD: antardase pocinju od rodjenja, prva vidljiva AD nije nuzno MD gospodar
    v0 = vimshottari(moon_sid, natal, natal)
    assert v0["mahadasha"] == "Venera" and v0["antardashas"][0][1] == natal and v0["ad_start"] == natal
    assert v0["antardashas"][-1][2] == v0["md_end"]

    # planetarni sati, Zagreb
    zlat, zlon = 45.8167, 15.9833
    rise, sset = sunrise_sunset(dt.date(2026, 11, 1), zlat, zlon)
    print(f"Zagreb 2026-11-01: izlazak {ts(rise)}, zalazak {ts(sset)}")
    assert rise.date() == sset.date() == dt.date(2026, 11, 1) and rise < sset
    assert 6.0 <= rise.hour + rise.minute / 60 <= 7.2 and 16.0 <= sset.hour + sset.minute / 60 <= 17.5
    ph = planetary_hour(at, zlat, zlon)
    print(f"Planetarni sat {ts(at)}: {ph['planet']}, sat {ph['hour_index']}, vladar dana {ph['day_ruler']}, "
          f"dnevni={ph['is_day_hour']}, {ts(ph['start'])} .. {ts(ph['end'])}, "
          f"planetarni dan {ph['planetary_day_date']}, sljedeci izlazak {ts(ph['next_sunrise'])}")
    assert ph["day_ruler"] == "Sunce" and ph["is_day_hour"] is False
    assert ph["start"] <= at < ph["end"] and 13 <= ph["hour_index"] <= 24
    assert ph["planetary_day_date"] == dt.date(2026, 11, 1)
    assert ph["sunrise"] == rise and ph["sunset"] == sset
    hours = planetary_hours(dt.date(2026, 11, 1), zlat, zlon)
    assert len(hours) == 24
    assert hours[0]["start"] == rise and hours[0]["planet"] == "Sunce" and hours[0]["hour_index"] == 1
    assert hours[11]["end"] == sset and hours[12]["start"] == sset and hours[23]["end"] == ph["next_sunrise"]
    assert [h["planet"] for h in hours[:8]] == ["Sunce", "Venera", "Merkur", "Mjesec", "Saturn", "Jupiter", "Mars", "Sunce"]
    for h1, h2 in zip(hours, hours[1:]):
        assert h1["end"] == h2["start"] and h2["hour_index"] == h1["hour_index"] + 1
    total = sum(((h["end"].astimezone(UTC) - h["start"].astimezone(UTC)) for h in hours), dt.timedelta(0))
    span = ph["next_sunrise"].astimezone(UTC) - rise.astimezone(UTC)
    assert total == span, (total, span)
    day_len = hours[0]["end"].astimezone(UTC) - hours[0]["start"].astimezone(UTC)
    night_len = hours[12]["end"].astimezone(UTC) - hours[12]["start"].astimezone(UTC)
    print(f"  dnevni sat {day_len}, nocni sat {night_len}, zbroj 24 sata {total} = izlazak->izlazak {span}")
    assert day_len < night_len  # studeni
    assert hours[ph["hour_index"] - 1]["planet"] == ph["planet"]
    # trenutak prije izlaska pripada prethodnom planetarnom danu (subota -> Saturn)
    early = dt.datetime(2026, 11, 1, 5, 0, tzinfo=TZ)
    pe = planetary_hour(early, zlat, zlon)
    assert pe["planetary_day_date"] == dt.date(2026, 10, 31) and pe["day_ruler"] == "Saturn"
    assert pe["is_day_hour"] is False and 13 <= pe["hour_index"] <= 24 and pe["next_sunrise"] == rise
    just_before = rise - dt.timedelta(microseconds=1)
    assert planetary_hour(just_before, zlat, zlon)["hour_index"] == 24
    assert planetary_hour(rise, zlat, zlon)["hour_index"] == 1
    assert planetary_hour(sset, zlat, zlon)["hour_index"] == 13
    # noc s prijelazom na ljetno vrijeme (2027-03-27 -> 28): trajanja u UTC-u moraju se slagati
    hd = planetary_hours(dt.date(2027, 3, 27), zlat, zlon)
    tot = sum(((h["end"].astimezone(UTC) - h["start"].astimezone(UTC)) for h in hd), dt.timedelta(0))
    sp = hd[0]["next_sunrise"].astimezone(UTC) - hd[0]["sunrise"].astimezone(UTC)
    assert tot == sp and dt.timedelta(hours=23) < sp < dt.timedelta(hours=25), (tot, sp)
    assert hd[0]["day_ruler"] == "Saturn" and hd[0]["sunrise"].utcoffset() == dt.timedelta(hours=1)
    assert hd[0]["next_sunrise"].utcoffset() == dt.timedelta(hours=2)

    # dan / noc
    noon = dt.datetime(2026, 11, 1, 12, 0, tzinfo=TZ)
    assert is_day(noon, zlat, zlon) is True and is_day(at, zlat, zlon) is False
    assert is_day(rise, zlat, zlon) is True and is_day(sset, zlat, zlon) is False
    print(f"is_day 12:00 = {is_day(noon, zlat, zlon)}, is_day 20:00 = {is_day(at, zlat, zlon)}")

    # tocka srece
    assert abs(pars_fortunae(10.0, 20.0, 50.0, True) - 40.0) < 1e-9
    assert abs(pars_fortunae(10.0, 20.0, 50.0, False) - 340.0) < 1e-9
    pf = pars_fortunae(asc, sun, moon, is_day(natal, lat, lon))
    print(f"Pars fortunae natal ({'dan' if is_day(natal, lat, lon) else 'noc'}): {pf:.4f} {format_lon(pf)}")

    # jednadzba vremena i pravo suncevo vrijeme
    t_ut = dt.datetime(2026, 11, 1, 19, 0, tzinfo=UTC)
    eot = equation_of_time_minutes(jd_ut(t_ut))
    tst = true_solar_time(t_ut, zlon)
    print(f"EoT 2026-11-01 19:00 UT = {eot:+.2f} min; pravo suncevo vrijeme u Zagrebu (lon {zlon}) = {tst}")
    assert 15.5 <= eot <= 17.0, eot
    assert tst.tzinfo is None
    exp = t_ut.replace(tzinfo=None) + dt.timedelta(hours=zlon / 15.0, minutes=eot)
    assert abs((tst - exp).total_seconds()) < 1e-3
    assert tst.hour == 20 and 19 <= tst.minute <= 21, tst
    eot_feb = equation_of_time_minutes(jd_ut(dt.datetime(2026, 2, 11, 12, tzinfo=UTC)))
    assert -15.0 <= eot_feb <= -13.5, eot_feb  # minimum EoT oko 11. veljace

    # greske
    for bad in (lambda: jd_ut(dt.datetime(2026, 1, 1)), lambda: vimshottari(moon_sid, at, natal),
                lambda: sunrise_sunset(dt.date(2026, 6, 21), 80.0, 15.0)):
        try:
            bad()
        except ValueError:
            pass
        else:
            raise AssertionError("ocekivan ValueError")
    print("astro selftest OK")


if __name__ == "__main__":
    selftest()
