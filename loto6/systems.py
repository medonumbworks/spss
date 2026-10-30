"""Svih 15 sustava za jedno kolo + godisnji fiksni set. Integracija modula astro, calendars, iching, kamea,
cards, numerology, history preko zajednickih pravila iz core.py. Nista od ovoga nema prediktivnu moc."""
from __future__ import annotations
import datetime as dt
from dataclasses import dataclass, field
from typing import Any

import swisseph as swe

import astro
import calendars
import cards
import iching
import kamea
import numerology as num
from config import (DAY_NAMES_HR, DAY_PLANET_HR, LIFE_PATH, BIRTHDAY_NUMBER, NAME, NAME_PYTHAGOREAN, NATAL_LAT,
                    NATAL_LOCAL, NATAL_LON, PLANET_NUMBER, SYSTEM_NAMES, ZAGREB_LAT, ZAGREB_LON, TZ,
                    PERIOD_START, PERIOD_END)
from core import Cascade, Combo, build_combo, m45, rng_for

ESOTERIC_IDS = list(range(5, 15))
PLANET_BODY = {"Sunce": swe.SUN, "Mjesec": swe.MOON, "Merkur": swe.MERCURY, "Venera": swe.VENUS,
               "Mars": swe.MARS, "Jupiter": swe.JUPITER, "Saturn": swe.SATURN}
SIGNS_HR = ["Ovan", "Bik", "Blizanci", "Rak", "Lav", "Djevica", "Vaga", "Skorpion", "Strijelac", "Jarac",
            "Vodenjak", "Ribe"]
NAK_HR = ["Asvini", "Bharani", "Krittika", "Rohini", "Mrigasira", "Ardra", "Punarvasu", "Pusya", "Aslesa", "Magha",
          "Purva Phalguni", "Uttara Phalguni", "Hasta", "Citra", "Svati", "Visakha", "Anuradha", "Jyestha", "Mula",
          "Purva Asadha", "Uttara Asadha", "Sravana", "Dhanistha", "Satabhisa", "Purva Bhadrapada",
          "Uttara Bhadrapada", "Revati"]


def _split(result: Any) -> tuple[Cascade, Cascade]:
    """Modul moze vratiti samo kaskadu ili (kaskada, rezerva)."""
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[0], list) and isinstance(result[1], list) \
            and (not result[0] or isinstance(result[0][0], tuple)):
        return list(result[0]), list(result[1])
    return list(result), []


def fmt_lon(lon: float) -> str:
    s = int(lon // 30) % 12
    d = lon - 30 * (lon // 30)
    return f"{d:6.3f} {SIGNS_HR[s]} ({lon:.4f})"


# ----------------------------------------------------------------------------------------------------------
# Natalne konstante (racunaju se jednom)
# ----------------------------------------------------------------------------------------------------------
@dataclass
class Natal:
    jd: float
    trop: dict[str, float]
    sid: dict[str, float]
    asc_trop: float
    asc_sid: float
    mc_trop: float
    ayanamsa: float
    is_day: bool
    pof: float
    moon_nak: tuple[int, int, str]
    weekday: int

    @property
    def day_planet(self) -> str:
        return DAY_PLANET_HR[self.weekday]


def compute_natal() -> Natal:
    jd = astro.jd_ut(NATAL_LOCAL)
    trop = {n: astro.tropical_lon(jd, b) for n, b in PLANET_BODY.items()}
    sid = {n: astro.sidereal_lon(jd, b) for n, b in PLANET_BODY.items()}
    asc_t = astro.ascendant(jd, NATAL_LAT, NATAL_LON, sidereal=False)
    asc_s = astro.ascendant(jd, NATAL_LAT, NATAL_LON, sidereal=True)
    cusps, ascmc = swe.houses_ex(jd, NATAL_LAT, NATAL_LON, b'P')
    swe.set_sid_mode(swe.SIDM_LAHIRI)
    ay = swe.get_ayanamsa_ut(jd)
    day = astro.is_day(NATAL_LOCAL, NATAL_LAT, NATAL_LON)
    pof = astro.pars_fortunae(asc_t, trop["Sunce"], trop["Mjesec"], day)
    nak = astro.nakshatra(sid["Mjesec"])
    n = Natal(jd, trop, sid, asc_t, asc_s, ascmc[1], ay, day, pof, nak, NATAL_LOCAL.weekday())
    # kontrola efemerida iz zadatka
    assert 69.0 <= trop["Sunce"] <= 70.0, f"natalno Sunce nije 9-10 Blizanaca: {trop['Sunce']}"
    assert nak[0] == 20, f"natalni Mjesec nije u Purva Asadhi: nakshatra {nak}"
    return n


# ----------------------------------------------------------------------------------------------------------
# Kontekst kola
# ----------------------------------------------------------------------------------------------------------
@dataclass
class DrawContext:
    date: dt.date
    ordinal: int
    dt_local: dt.datetime
    weekday: int
    day_name: str
    day_planet: str
    jd: float
    trop: dict[str, float]
    sid: dict[str, float]
    asc_trop: float
    asc_sid: float
    mc_trop: float
    ayanamsa: float
    is_day: bool
    sunrise: dt.datetime
    sunset: dt.datetime
    pof: float
    hebrew: dict
    chinese: dict
    phour: dict
    dasha: dict
    moon_nak: tuple[int, int, str]
    sun_nak: tuple[int, int, str]
    extra: dict = field(default_factory=dict)


def compute_context(d: dt.date, ordinal: int, natal: Natal) -> DrawContext:
    from config import draw_datetime
    dtl = draw_datetime(d)
    jd = astro.jd_ut(dtl)
    trop = {n: astro.tropical_lon(jd, b) for n, b in PLANET_BODY.items()}
    sid = {n: astro.sidereal_lon(jd, b) for n, b in PLANET_BODY.items()}
    asc_t = astro.ascendant(jd, ZAGREB_LAT, ZAGREB_LON, sidereal=False)
    asc_s = astro.ascendant(jd, ZAGREB_LAT, ZAGREB_LON, sidereal=True)
    cusps, ascmc = swe.houses_ex(jd, ZAGREB_LAT, ZAGREB_LON, b'P')
    swe.set_sid_mode(swe.SIDM_LAHIRI)
    ay = swe.get_ayanamsa_ut(jd)
    day = astro.is_day(dtl, ZAGREB_LAT, ZAGREB_LON)
    sr, ss = astro.sunrise_sunset(d, ZAGREB_LAT, ZAGREB_LON)
    pof = astro.pars_fortunae(asc_t, trop["Sunce"], trop["Mjesec"], day)
    heb = calendars.hebrew_date(dtl, ZAGREB_LAT, ZAGREB_LON)
    chi = calendars.chinese_lunar(dtl, ZAGREB_LON)
    ph = astro.planetary_hour(dtl, ZAGREB_LAT, ZAGREB_LON)
    dasha = astro.vimshottari(natal.sid["Mjesec"], NATAL_LOCAL, dtl)
    wd = d.weekday()
    return DrawContext(d, ordinal, dtl, wd, DAY_NAMES_HR[wd], DAY_PLANET_HR[wd], jd, trop, sid, asc_t, asc_s,
                       ascmc[1], ay, day, sr, ss, pof, heb, chi, ph, dasha, astro.nakshatra(sid["Mjesec"]),
                       astro.nakshatra(sid["Sunce"]))


# ----------------------------------------------------------------------------------------------------------
# Sustavi 5-14 (dnevni)
# ----------------------------------------------------------------------------------------------------------
def sys_kaldejska(ctx: DrawContext) -> Combo:
    words = ["LOTO", "HRVATSKA LUTRIJA", ctx.day_name, "ZAGREB", NAME]
    casc: Cascade = [(w, num.chaldean_word(w)) for w in words]
    casc += [("dan u mjesecu", ctx.date.day), ("zbroj znamenki datuma", num.date_digit_sum(ctx.date)),
             (f"planet dana {ctx.day_planet}", PLANET_NUMBER[ctx.day_planet])]
    res: Cascade = []
    for w in words:
        res += [(f"{w}:{l}", v) for l, v in num.chaldean_letters(w)]
    run = 0
    for lab, v in casc:
        run += v
        res.append((f"kumulativ do {lab}", run))
    izv = "; ".join(f"{lab}={v}" for lab, v in casc)
    return build_combo(5, SYSTEM_NAMES[5], casc, res, izv)


def kaldejska_godisnja() -> Combo:
    words = ["LOTO", "HRVATSKA LUTRIJA", "ZAGREB", NAME]
    casc: Cascade = [(w, num.chaldean_word(w)) for w in words]
    for y in sorted({PERIOD_START.year, PERIOD_END.year}):
        casc += [(f"godina {y}", y), (f"zbroj znamenki {y}", num.digit_sum(y))]
    res: Cascade = []
    for w in words:
        res += [(f"{w}:{l}", v) for l, v in num.chaldean_letters(w)]
    run = 0
    for lab, v in casc:
        run += v
        res.append((f"kumulativ do {lab}", run))
    izv = "godisnja (bez imena dana i datuma, s godinom): " + "; ".join(f"{lab}={v}" for lab, v in casc)
    return build_combo(5, "Kaldejska godisnja", casc, res, izv)


def sys_gematrija(ctx: DrawContext) -> Combo:
    h = ctx.hebrew
    casc: Cascade = [(f"hebrejski dan {h['day']} {h['month_name']} {h['year']}"
                      + (" (poslije zalaska)" if h["after_sunset"] else ""), h["day"])]
    casc += [(w, v) for w, v in num.GEMATRIA_WORDS]
    res: Cascade = []
    for w, _ in num.GEMATRIA_WORDS:
        res += [(f"{w}:{l}", v) for l, v in num.hebrew_letters(w)]
    izv = "; ".join(f"{lab}={v}" for lab, v in casc)
    return build_combo(6, SYSTEM_NAMES[6], casc, res, izv)


def sys_hermetika(ctx: DrawContext) -> Combo:
    ph = ctx.phour
    planet = ph["planet"]
    casc, res = _split(kamea.kamea_cascade(planet))
    if not res:
        res = _kamea_reserve(planet)
    izv = (f"planetarni sat {ph['hour_index']} ({'dnevni' if ph['is_day_hour'] else 'nocni'}) = {planet}, "
           f"vladar dana {ph['day_ruler']}, sat {ph['start'].strftime('%H:%M')}-{ph['end'].strftime('%H:%M')}; "
           f"kamea {planet} {len(kamea.KAMEA[planet])}x{len(kamea.KAMEA[planet])}: "
           + "; ".join(f"{lab}={v}" for lab, v in casc))
    return build_combo(7, SYSTEM_NAMES[7], casc, res, izv)


def _kamea_reserve(planet: str) -> Cascade:
    sq = kamea.KAMEA[planet]
    n = len(sq)
    mid = [n // 2] if n % 2 else [n // 2 - 1, n // 2]
    res: Cascade = []
    for i, row in enumerate(sq):
        if i in mid:
            continue
        res += [(f"red{i + 1}[{j + 1}]", v) for j, v in enumerate(row)]
    for j in range(n):
        res += [(f"stupac{j + 1}[{i + 1}]", sq[i][j]) for i in range(n)]
    return res


def _iching_reserve(lines: list[int]) -> Cascade:
    up, lo = iching.lines_to_trigrams(lines)
    res: Cascade = [(f"linija{i + 1}", v) for i, v in enumerate(lines)]
    res += [("FuXi gornji", up), ("FuXi donji", lo), ("KW nuklearnog promijenjenog", iching.king_wen(iching.nuclear(iching.changed(lines))))]
    return res


def sys_iching(ctx: DrawContext) -> Combo:
    rng = rng_for(ctx.ordinal, 8)
    lines = iching.cast_yarrow(rng)
    casc, res = _split(iching.iching_cascade(lines))
    if not res:
        res = _iching_reserve(lines)
    ctx.extra["iching_lines"] = lines
    kw = iching.king_wen(lines)
    izv = (f"linije (odozdo) {lines}; primarni {kw} {iching.hexagram_name(kw)}; nuklearni "
           f"{iching.king_wen(iching.nuclear(lines))}; promijenjeni {iching.king_wen(iching.changed(lines))}; "
           f"pomicne {iching.moving_positions(lines)}; " + "; ".join(f"{lab}={v}" for lab, v in casc))
    return build_combo(8, SYSTEM_NAMES[8], casc, res, izv)


def sys_lenormand(ctx: DrawContext) -> Combo:
    rng = rng_for(ctx.ordinal, 9)
    tableau, casc = cards.lenormand_cascade(rng)
    ctx.extra["lenormand_tableau"] = tableau
    res: Cascade = [(f"r1 karta {cards.LENORMAND[c - 1]}", c) for c in tableau[0]]
    izv = "1. red: " + " ".join(str(c) for c in tableau[0]) + "; " + "; ".join(f"{lab}={v}" for lab, v in casc[:8])
    return build_combo(9, SYSTEM_NAMES[9], casc, res, izv)


def sys_meihua(ctx: DrawContext) -> Combo:
    c = ctx.chinese
    s1 = c["year_branch"] + c["lunar_month"] + c["lunar_day"]
    s2 = s1 + c["hour_branch"]
    upper = s1 % 8 or 8
    lower = s2 % 8 or 8
    moving = s2 % 6 or 6
    lines = iching.meihua_lines(upper, lower, moving)
    casc, res = _split(iching.meihua_cascade(upper, lower, moving, s1, s2))
    if not res:
        res = _iching_reserve(lines)
    ctx.extra["meihua"] = dict(s1=s1, s2=s2, upper=upper, lower=lower, moving=moving, lines=lines)
    kw = iching.king_wen(lines)
    izv = (f"lunarno {c['lunar_year']}/{c['lunar_month']}/{c['lunar_day']} ({c['year_ganzhi']}, ogranak godine "
           f"{c['year_branch']}), sat ogranak {c['hour_branch']} {c['branch_name']} po TST {c['tst'].strftime('%H:%M')}; "
           f"zbroj g+m+d={s1} -> gornji {upper}; +sat={s2} -> donji {lower}; pomicna {moving}; primarni {kw} "
           f"{iching.hexagram_name(kw)}; " + "; ".join(f"{lab}={v}" for lab, v in casc))
    return build_combo(10, SYSTEM_NAMES[10], casc, res, izv)


def sys_tarot(ctx: DrawContext) -> Combo:
    rng = rng_for(ctx.ordinal, 11)
    order, casc = cards.tarot_cascade(rng)
    ctx.extra["tarot_order"] = order
    izv = "mijesanje (prvih 12): " + ", ".join(f"{i}" for i in order[:12]) + "; kaskada: " + \
          "; ".join(f"{lab}" for lab, v in casc[:8])
    return build_combo(11, SYSTEM_NAMES[11], casc, [], izv)


def _pyth_items(date: dt.date | None, year: int) -> Cascade:
    casc: Cascade = [(f"{NAME} (pitagorejski)", NAME_PYTHAGOREAN), ("zivotni put 29", LIFE_PATH[0]),
                     ("zivotni put /11", LIFE_PATH[1]), ("rodjendanski broj", BIRTHDAY_NUMBER)]
    uy = num.digit_sum(year)
    py_raw = NATAL_LOCAL.month + NATAL_LOCAL.day + uy
    py = num.reduce_le45(py_raw)[0]
    if date is not None:
        casc += [("mjesec kola", date.month), ("zbroj znamenki datuma kola", num.date_digit_sum(date))]
        casc.append((f"osobna godina ({NATAL_LOCAL.month}+{NATAL_LOCAL.day}+{uy}={py_raw})", py))
        pm_raw = py + date.month
        pm = num.reduce_le45(pm_raw)[0]
        casc.append((f"osobni mjesec ({py}+{date.month}={pm_raw})", pm))
        pd_raw = pm + date.day
        pd = num.reduce_le45(pd_raw)[0]
        casc.append((f"osobni dan ({pm}+{date.day}={pd_raw})", pd))
        ud_raw = uy + date.month + date.day
        casc.append((f"univerzalni dan ({uy}+{date.month}+{date.day}={ud_raw})", num.reduce_le45(ud_raw)[0]))
    else:
        casc.append((f"osobna godina {year} ({NATAL_LOCAL.month}+{NATAL_LOCAL.day}+{uy}={py_raw})", py))
        casc.append((f"univerzalna godina {year}", uy))
    return casc


def _pyth_reserve(casc: Cascade) -> Cascade:
    res: Cascade = [(f"reducirano {lab}", num.reduce_single(v)) for lab, v in casc]
    for k in (9, 18, 27, 36):
        res += [(f"{lab}+{k}", v + k) for lab, v in casc if v + k <= 45]
    return res


def sys_pitagorejska(ctx: DrawContext) -> Combo:
    casc = _pyth_items(ctx.date, ctx.date.year)
    izv = "; ".join(f"{lab}={v}" for lab, v in casc)
    return build_combo(12, SYSTEM_NAMES[12], casc, _pyth_reserve(casc), izv)


def pitagorejska_godisnja() -> Combo:
    casc: Cascade = []
    for i, y in enumerate(sorted({PERIOD_START.year, PERIOD_END.year})):
        items = _pyth_items(None, y)
        casc += items if i == 0 else items[4:]
    izv = "godisnja (osobna godina umjesto dnevnih brojeva): " + "; ".join(f"{lab}={v}" for lab, v in casc)
    return build_combo(12, "Pitagorejska godisnja", casc, _pyth_reserve(casc), izv)


def _jyotis_family(lord: str) -> Cascade:
    L = PLANET_NUMBER[lord]
    return [(f"gospodar {lord}", L)] + [(f"{lord}+{k}", L + k) for k in (9, 18, 27, 36)]


def sys_jyotis(ctx: DrawContext) -> Combo:
    n, pada, lord = ctx.moon_nak
    casc: Cascade = [(f"naksatra Mjeseca {NAK_HR[n - 1]} pada {pada}", n)] + _jyotis_family(lord)
    casc += [("rasi Mjeseca", astro.rashi(ctx.sid["Mjesec"])), ("rasi lagne", astro.rashi(ctx.asc_sid)),
             (f"mahadasa {ctx.dasha['mahadasha']}", PLANET_NUMBER[ctx.dasha["mahadasha"]]),
             (f"antardasa {ctx.dasha['antardasha']}", PLANET_NUMBER[ctx.dasha["antardasha"]]),
             (f"vladar dana {ctx.day_planet}", PLANET_NUMBER[ctx.day_planet])]
    sn, spada, slord = ctx.sun_nak
    res: Cascade = [(f"naksatra Sunca {NAK_HR[sn - 1]}", sn), ("rasi Sunca", astro.rashi(ctx.sid["Sunce"]))]
    res += [(f"rasi {p}", astro.rashi(ctx.sid[p])) for p in ("Mars", "Merkur", "Jupiter", "Venera", "Saturn")]
    res += [("pada Mjeseca", pada)]
    izv = "; ".join(f"{lab}={v}" for lab, v in casc)
    return build_combo(13, SYSTEM_NAMES[13], casc, res, izv)


def jyotis_natalna(natal: Natal) -> Combo:
    n, pada, lord = natal.moon_nak
    casc: Cascade = [(f"natalna naksatra Mjeseca {NAK_HR[n - 1]} pada {pada}", n)] + _jyotis_family(lord)
    casc += [("natalni rasi Mjeseca", astro.rashi(natal.sid["Mjesec"])), ("natalni rasi lagne", astro.rashi(natal.asc_sid)),
             (f"vladar natalnog dana {natal.day_planet}", PLANET_NUMBER[natal.day_planet])]
    sn = astro.nakshatra(natal.sid["Sunce"])
    res: Cascade = [(f"natalna naksatra Sunca {NAK_HR[sn[0] - 1]}", sn[0]), ("natalni rasi Sunca", astro.rashi(natal.sid["Sunce"]))]
    res += [(f"natalni rasi {p}", astro.rashi(natal.sid[p])) for p in ("Mars", "Merkur", "Jupiter", "Venera", "Saturn")]
    izv = "natalna (samo natalne tocke): " + "; ".join(f"{lab}={v}" for lab, v in casc)
    return build_combo(13, "Jyotis natalna", casc, res, izv)


def _zapadna_points(points: list[tuple[str, float]]) -> Cascade:
    casc: Cascade = [(f"stupanj {lab}", astro.deg_in_sign(lon)) for lab, lon in points]
    casc += [(f"znak {lab}", astro.sign_no(lon)) for lab, lon in points]
    casc += [(f"stupanj+30 {lab}", astro.deg_in_sign(lon) + 30) for lab, lon in points if lon >= 180.0]
    return casc


def sys_zapadna(ctx: DrawContext, natal: Natal) -> Combo:
    pts = [("Sunce", ctx.trop["Sunce"]), ("Mjesec", ctx.trop["Mjesec"]), ("ASC", ctx.asc_trop),
           (f"vladar dana {ctx.day_planet}", ctx.trop[ctx.day_planet]),
           (f"PoF ({'dnevna' if ctx.is_day else 'nocna'} formula)", ctx.pof), ("natalni PoF", natal.pof)]
    casc = _zapadna_points(pts)
    res: Cascade = [(f"stupanj {p}", astro.deg_in_sign(ctx.trop[p])) for p in ("Merkur", "Venera", "Mars", "Jupiter", "Saturn")]
    res += [("stupanj MC", astro.deg_in_sign(ctx.mc_trop))]
    izv = "; ".join(f"{lab} {fmt_lon(lon)}" for lab, lon in pts)
    return build_combo(14, SYSTEM_NAMES[14], casc, res, izv)


def zapadna_natalna(natal: Natal) -> Combo:
    pts = [("natalno Sunce", natal.trop["Sunce"]), ("natalni Mjesec", natal.trop["Mjesec"]), ("natalni ASC", natal.asc_trop),
           (f"vladar natalnog dana {natal.day_planet}", natal.trop[natal.day_planet]),
           (f"natalni PoF ({'dnevna' if natal.is_day else 'nocna'} formula)", natal.pof)]
    casc = _zapadna_points(pts)
    res: Cascade = [(f"natalni stupanj {p}", astro.deg_in_sign(natal.trop[p])) for p in ("Merkur", "Mars", "Jupiter", "Saturn")]
    res += [("natalni stupanj MC", astro.deg_in_sign(natal.mc_trop))]
    izv = "natalna (samo natalne tocke): " + "; ".join(f"{lab} {fmt_lon(lon)}" for lab, lon in pts)
    return build_combo(14, "Zapadna natalna", casc, res, izv)


# ----------------------------------------------------------------------------------------------------------
# Statistika (1-4) i Plecko (15)
# ----------------------------------------------------------------------------------------------------------
def statistics_combos(hist) -> list[Combo]:
    """Pareto 1/2 i Vruci 1/2 iz povijesti; hist je rezultat history.load_history (None ako datoteke nema)."""
    if hist is None:
        return []
    import history
    p1, p2 = history.pareto(hist["df"])
    v1, v2, win = history.vruci(hist["df"])
    out = []
    for sid, nums, izv in ((1, p1, "6 najcescih u cijeloj povijesti"), (2, p2, "rang 7-12 u cijeloj povijesti"),
                           (3, v1, f"6 najcescih u zadnjih {win} kola"), (4, v2, f"rang 7-12 u zadnjih {win} kola")):
        freq = history.frequencies(hist["df"]) if sid <= 2 else history.frequencies(hist["df"], last_n=win)
        casc = [(f"broj {n} ({freq.get(n, 0)}x)", n) for n in nums]
        out.append(build_combo(sid, SYSTEM_NAMES[sid], casc, [], izv + ": " + "; ".join(l for l, _ in casc)))
    return out


def plecko(sources: dict[str, list[int]], hist_freq: dict[int, int] | None, label: str = "Plecko",
           system_id: int = 15) -> tuple[Combo, list[tuple[int, int, list[str]]]]:
    """Filtar konvergencije: broj razlicitih izvora po broju; signal >= 2, sum = 1."""
    counts: dict[int, list[str]] = {}
    for src, nums in sources.items():
        for n in set(nums):
            counts.setdefault(n, []).append(src)

    def key(n: int):
        return (-len(counts[n]), -(hist_freq or {}).get(n, 0), n)

    ordered = sorted(counts, key=key)
    signal = [n for n in ordered if len(counts[n]) >= 2]
    noise = [n for n in ordered if len(counts[n]) == 1]
    casc: Cascade = [(f"signal {len(counts[n])} izvora", n) for n in signal]
    res: Cascade = [("sum 1 izvor", n) for n in noise]
    izv = (f"izvori: {len(sources)}; signal (>=2 izvora): " + ", ".join(f"{n}({len(counts[n])})" for n in signal)
           + ("; nerjeseno: cesci u povijesti, pa manji" if hist_freq else "; nerjeseno: manji broj (povijest nedostupna)"))
    combo = build_combo(system_id, label, casc, res, izv)
    rows = [(n, len(counts[n]), counts[n]) for n in ordered]
    return combo, rows


# ----------------------------------------------------------------------------------------------------------
# Set jednog kola
# ----------------------------------------------------------------------------------------------------------
@dataclass
class DrawResult:
    ctx: DrawContext
    combos: list[Combo]                  # po rangu, nakon pravila seta
    konvergencija: list[tuple[int, int, list[str]]]
    signal_numbers: list[int]
    stats_available: bool


def build_draw(d: dt.date, ordinal: int, natal: Natal, hist, forbidden: list[set[int]] | None = None) -> DrawResult:
    from core import apply_set_rule
    ctx = compute_context(d, ordinal, natal)
    stats = statistics_combos(hist)
    eso = [sys_kaldejska(ctx), sys_gematrija(ctx), sys_hermetika(ctx), sys_iching(ctx), sys_lenormand(ctx),
           sys_meihua(ctx), sys_tarot(ctx), sys_pitagorejska(ctx), sys_jyotis(ctx), sys_zapadna(ctx, natal)]
    sources: dict[str, list[int]] = {}
    if stats:
        sources["Statistika"] = sorted(set(n for c in stats for n in c.numbers))
    for c in eso:
        sources[c.system] = list(c.numbers)
    import history as _h
    hist_freq = _h.frequencies(hist["df"]) if hist is not None else None
    pl, rows = plecko(sources, hist_freq)
    combos = stats + eso + [pl]
    apply_set_rule(combos, forbidden=forbidden)
    signal = [n for n, k, _ in rows if k >= 2]
    return DrawResult(ctx, combos, rows, signal, bool(stats))
