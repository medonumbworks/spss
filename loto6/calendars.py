"""Kalendari za registar Loto 6/45: hebrejski datum (pyluach + unakrsna provjera convertdate)
i kineski lunarni datum (lunardate) s dvosatnim ogrankom po PRAVOM suncevom vremenu (pyswisseph).

Modul je samostalan: ne uvozi astro.py (pise se paralelno), nego ima vlastite privatne pomocne
funkcije _sunset_local() i _true_solar_time() na temelju swisseph (Moshierova efemerida, FLG_MOSEPH).

Konvencije:
- Ulazni datetime mora biti "aware" (npr. ZoneInfo("Europe/Zagreb")); naive datetime podize ValueError.
- Lokalni civilni datum se uvijek odredjuje u config.TZ (Europe/Zagreb).
- Hebrejski dan pocinje zalaskom Sunca: ako je dt >= zalazak tog lokalnog dana na (lat, lon),
  uzima se gregorijanski datum + 1 dan.
- Kineski lunarni dan se odredjuje po PRAVOM suncevom vremenu (TST = UT + lon/15 h + jednadzba vremena);
  ako je TST >= 23:00 (rani sat Zi) uzima se SLJEDECI gregorijanski dan.

Numeriranje hebrejskih mjeseci (pyluach I convertdate, provjereno u selftestu za 2020.-2030.):
  1 Nisan, 2 Ijar, 3 Sivan, 4 Tamuz, 5 Av, 6 Elul, 7 Tisri, 8 Hesvan (Cheshvan), 9 Kislev,
  10 Tevet, 11 Sevat, 12 Adar (u prijestupnoj godini Adar I), 13 Adar II (samo prijestupna godina).
  Godina (npr. 5787) pocinje 1. Tisrija (mjesec 7), a mjesec 1 (Nisan) je u sredini te godine.
"""
from __future__ import annotations

import warnings
from datetime import date, datetime, time, timedelta, timezone

import swisseph as swe
from convertdate import hebrew as cd_hebrew
from lunardate import LunarDate
from pyluach.dates import HebrewDate

from config import TZ, ZAGREB_LAT, ZAGREB_LON

__all__ = [
    "HEBREW_MONTHS_HR", "HEAVENLY_STEMS", "HEAVENLY_STEMS_CN", "EARTHLY_BRANCHES", "EARTHLY_BRANCHES_CN",
    "HOUR_BRANCH_TABLE", "hebrew_date", "chinese_lunar", "hour_branch_from_hours", "selftest",
]

# ----------------------------------------------------------------------------------------------
# Tablice
# ----------------------------------------------------------------------------------------------

# Hrvatska transliteracija hebrejskih mjeseci po pyluach/convertdate broju (Nisan = 1 ... Adar II = 13).
HEBREW_MONTHS_HR: dict[int, str] = {
    1: "Nisan", 2: "Ijar", 3: "Sivan", 4: "Tamuz", 5: "Av", 6: "Elul", 7: "Tisri", 8: "Hesvan",
    9: "Kislev", 10: "Tevet", 11: "Sevat", 12: "Adar", 13: "Adar II",
}

# Nebeska stabla (stem = (lunar_year - 4) % 10 -> indeks 0..9).
HEAVENLY_STEMS: tuple[str, ...] = ("Jia", "Yi", "Bing", "Ding", "Wu", "Ji", "Geng", "Xin", "Ren", "Gui")
HEAVENLY_STEMS_CN: tuple[str, ...] = tuple("甲乙丙丁戊己庚辛壬癸")

# Zemaljski ogranci (branch = (lunar_year - 4) % 12 -> indeks 0..11; year_branch = indeks + 1, Zi = 1 ... Hai = 12).
EARTHLY_BRANCHES: tuple[str, ...] = ("Zi", "Chou", "Yin", "Mao", "Chen", "Si", "Wu", "Wei", "Shen", "You", "Xu", "Hai")
EARTHLY_BRANCHES_CN: tuple[str, ...] = tuple("子丑寅卯辰巳午未申酉戌亥")

# Dvosatni ogranci po pravom suncevom vremenu: (broj, pinyin, znak, pocetak_h, kraj_h); Zi prelazi preko ponoci.
HOUR_BRANCH_TABLE: tuple[tuple[int, str, str, int, int], ...] = (
    (1, "Zi", "子", 23, 1), (2, "Chou", "丑", 1, 3), (3, "Yin", "寅", 3, 5), (4, "Mao", "卯", 5, 7),
    (5, "Chen", "辰", 7, 9), (6, "Si", "巳", 9, 11), (7, "Wu", "午", 11, 13), (8, "Wei", "未", 13, 15),
    (9, "Shen", "申", 15, 17), (10, "You", "酉", 17, 19), (11, "Xu", "戌", 19, 21), (12, "Hai", "亥", 21, 23),
)

_UTC = timezone.utc


# ----------------------------------------------------------------------------------------------
# Privatne pomocne funkcije (swisseph)
# ----------------------------------------------------------------------------------------------

def _require_aware(dt: datetime, who: str) -> None:
    """Podize ValueError ako datetime nije aware (nema vremensku zonu)."""
    if not isinstance(dt, datetime):
        raise ValueError(f"{who}: ocekujem datetime, dobio {type(dt).__name__}")
    if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
        raise ValueError(f"{who}: datetime mora biti aware (s vremenskom zonom), dobio naive {dt!r}")


def _jd_ut(dt: datetime) -> float:
    """Aware datetime -> julijanski dan (UT)."""
    u = dt.astimezone(_UTC)
    hour = u.hour + u.minute / 60.0 + u.second / 3600.0 + u.microsecond / 3.6e9
    return swe.julday(u.year, u.month, u.day, hour)


def _jd_to_utc(jd: float) -> datetime:
    """Julijanski dan (UT) -> aware datetime u UTC."""
    y, m, d, h = swe.revjul(jd)
    return datetime(y, m, d, tzinfo=_UTC) + timedelta(hours=h)


def _sunset_local(day: date, lat: float, lon: float) -> datetime:
    """Zalazak Sunca za lokalni (config.TZ) kalendarski dan `day` na (lat, lon), kao aware datetime u config.TZ.

    swe.rise_trans(..., CALC_SET, flags=FLG_MOSEPH): standardni gornji rub Sunca + refrakcija (rsmi bez dodatnih bitova).
    Trazenje pocinje od lokalne ponoci tog dana, pa se vraca prvi zalazak nakon nje; provjerava se da pada na isti
    lokalni dan (za Zagreb uvijek vrijedi). Podize ValueError za cirkumpolarni slucaj ili ako zalazak ne pada u `day`.
    """
    start = datetime.combine(day, time(0, 0), tzinfo=TZ)
    res, tret = swe.rise_trans(_jd_ut(start), swe.SUN, swe.CALC_SET, (float(lon), float(lat), 0.0),
                               0.0, 0.0, swe.FLG_MOSEPH)
    if res != 0:
        raise ValueError(f"zalazak nije pronadjen (res={res}, cirkumpolarno?) za {day} lat={lat} lon={lon}")
    sunset = _jd_to_utc(tret[0]).astimezone(TZ)
    if sunset.date() != day:
        raise ValueError(f"zalazak {sunset.isoformat()} ne pada na lokalni dan {day} (lat={lat}, lon={lon})")
    return sunset


def _equation_of_time_minutes(jd: float) -> float:
    """Jednadzba vremena u minutama (swe.time_equ vraca dane: prividno - srednje suncevo vrijeme)."""
    return float(swe.time_equ(jd)) * 1440.0


def _true_solar_time(dt: datetime, lon: float) -> datetime:
    """Pravo (prividno) suncevo vrijeme na zemljopisnoj duzini `lon` (istocno pozitivno), kao NAIVE datetime.

    TST = UT + lon/15 h + EoT. Predznak provjeren u selftestu: 1.11.2026. EoT ~ +16.4 min,
    pa je za 20:00 CET (19:00 UT) u Zagrebu (lon 15.9833) TST ~ 20:20.
    """
    _require_aware(dt, "_true_solar_time")
    u = dt.astimezone(_UTC).replace(tzinfo=None)
    eot_days = float(swe.time_equ(_jd_ut(dt)))
    return u + timedelta(hours=float(lon) / 15.0) + timedelta(days=eot_days)


# ----------------------------------------------------------------------------------------------
# Javne funkcije
# ----------------------------------------------------------------------------------------------

def hebrew_date(dt: datetime, lat: float, lon: float) -> dict:
    """Hebrejski datum za aware datetime `dt`, uz pravilo da dan pocinje zalaskom Sunca na (lat, lon).

    Postupak: lokalni civilni datum u config.TZ -> zalazak tog dana (_sunset_local) -> ako je dt >= zalazak,
    pydate + 1 dan -> pyluach HebrewDate.from_pydate(pydate). Unakrsna provjera: convertdate.hebrew.from_gregorian
    mora dati isti (year, month, day); inace se podize AssertionError s detaljima.

    Numeriranje mjeseci (obje biblioteke): Nisan = 1, ..., Elul = 6, Tisri = 7, Hesvan = 8, ..., Sevat = 11,
    Adar = 12 (u prijestupnoj godini Adar I = 12), Adar II = 13. month_name je pyluach HebrewDate.month_name()
    (engleski: 'Tishrei', 'Cheshvan', 'Adar 1', 'Adar 2', ...), month_name_hr je hrvatska transliteracija.

    Vraca dict: year, month_no, month_name, day, after_sunset, sunset (aware, config.TZ),
    te dodatno month_name_hr, leap_year (bool), civil_date (lokalni civilni datum) i effective_date
    (gregorijanski datum koji je predan pyluachu nakon pomaka za zalazak).
    """
    _require_aware(dt, "hebrew_date")
    local = dt.astimezone(TZ)
    civil = local.date()
    sunset = _sunset_local(civil, lat, lon)
    after_sunset = dt >= sunset
    eff = civil + timedelta(days=1) if after_sunset else civil
    h = HebrewDate.from_pydate(eff)
    with warnings.catch_warnings():
        # convertdate 2.5.1 interno zove vlastitu zastarjelu month_days() -> DeprecationWarning koji nije nas
        warnings.simplefilter("ignore", DeprecationWarning)
        cy, cm, cd = cd_hebrew.from_gregorian(eff.year, eff.month, eff.day)
    if (h.year, h.month, h.day) != (cy, cm, cd):
        raise AssertionError(
            f"hebrew_date: pyluach {(h.year, h.month, h.day)} != convertdate {(cy, cm, cd)} za gregorijanski {eff} "
            f"(dt={dt.isoformat()}, after_sunset={after_sunset})")
    leap = bool(cd_hebrew.leap(h.year))
    if h.month == 12 and leap:
        name_hr = "Adar I"
    else:
        name_hr = HEBREW_MONTHS_HR[h.month]
    return {
        "year": int(h.year),
        "month_no": int(h.month),
        "month_name": str(h.month_name()),
        "day": int(h.day),
        "after_sunset": bool(after_sunset),
        "sunset": sunset,
        "month_name_hr": name_hr,
        "leap_year": leap,
        "civil_date": civil,
        "effective_date": eff,
    }


def hour_branch_from_hours(hours: float) -> int:
    """Dvosatni ogranak (1..12) iz sata pravog suncevog vremena kao decimalnog broja 0 <= hours < 24.

    Zi = 1 pokriva 23:00-01:00, Chou = 2 01-03, ..., Hai = 12 21-23 (vidi HOUR_BRANCH_TABLE).
    Granice: pocetak intervala pripada tom ogranku (npr. tocno 01:00 -> Chou, tocno 23:00 -> Zi).
    """
    hours = float(hours) % 24.0
    return int((hours + 1.0) // 2.0) % 12 + 1


def chinese_lunar(dt: datetime, lon: float) -> dict:
    """Kineski lunarni datum i dvosatni ogranak po PRAVOM suncevom vremenu za aware datetime `dt`.

    Postupak: tst = _true_solar_time(dt, lon) (naive). Osnovni gregorijanski datum je datum tog TST-a;
    ako je TST >= 23:00 (rani sat Zi = pocetak novog dana po tradiciji) uzima se SLJEDECI gregorijanski dan.
    Napomena: osnova je datum TST-a, ne civilni datum, da bi pravilo bilo dosljedno i oko ponoci (npr. 00:30 CEST
    ljeti ima TST ~23:30 prethodnog dana -> prethodni datum + 1 = isti civilni dan, a ne civilni dan + 1).
    Lunarni datum: lunardate.LunarDate.from_solar_date(y, m, d).

    year_branch = ((lunar_year - 4) % 12) + 1 (Zi = 1 ... Hai = 12); stem = (lunar_year - 4) % 10 (Jia = 0);
    year_ganzhi = "<stablo> <ogranak>" pinyin (npr. "Bing Wu"), year_ganzhi_cn = znakovi (npr. "丙午").
    hour_branch po tablici Zi=1 23-01, Chou=2 01-03, Yin=3 03-05, Mao=4 05-07, Chen=5 07-09, Si=6 09-11, Wu=7 11-13,
    Wei=8 13-15, Shen=9 15-17, You=10 17-19, Xu=11 19-21, Hai=12 21-23.

    Vraca dict: lunar_year, lunar_month, lunar_day, is_leap_month, year_branch, year_ganzhi, hour_branch, tst (naive),
    branch_name ("Xu 戌"), te dodatno year_stem (1..10), year_ganzhi_cn, year_branch_name, hour_branch_cn,
    eot_minutes, effective_date (gregorijanski datum predan lunardateu) i next_day_rule (bool, je li primijenjeno
    pravilo TST >= 23:00).
    """
    _require_aware(dt, "chinese_lunar")
    tst = _true_solar_time(dt, lon)
    eot_min = _equation_of_time_minutes(_jd_ut(dt))
    base = tst.date()
    next_day_rule = tst.hour >= 23
    if next_day_rule:
        base = base + timedelta(days=1)
    ld = LunarDate.from_solar_date(base.year, base.month, base.day)
    stem_i = (ld.year - 4) % 10
    branch_i = (ld.year - 4) % 12
    hours = tst.hour + tst.minute / 60.0 + tst.second / 3600.0 + tst.microsecond / 3.6e9
    hb = hour_branch_from_hours(hours)
    hb_row = HOUR_BRANCH_TABLE[hb - 1]
    return {
        "lunar_year": int(ld.year),
        "lunar_month": int(ld.month),
        "lunar_day": int(ld.day),
        "is_leap_month": bool(ld.is_leap_month),
        "year_branch": branch_i + 1,
        "year_ganzhi": f"{HEAVENLY_STEMS[stem_i]} {EARTHLY_BRANCHES[branch_i]}",
        "hour_branch": hb,
        "tst": tst,
        "branch_name": f"{hb_row[1]} {hb_row[2]}",
        "year_stem": stem_i + 1,
        "year_ganzhi_cn": HEAVENLY_STEMS_CN[stem_i] + EARTHLY_BRANCHES_CN[branch_i],
        "year_branch_name": f"{EARTHLY_BRANCHES[branch_i]} {EARTHLY_BRANCHES_CN[branch_i]}",
        "hour_branch_cn": hb_row[2],
        "eot_minutes": eot_min,
        "effective_date": base,
        "next_day_rule": bool(next_day_rule),
    }


# ----------------------------------------------------------------------------------------------
# Selftest
# ----------------------------------------------------------------------------------------------

def _fmt_tst(t: datetime) -> str:
    return t.replace(microsecond=0).isoformat(sep=" ")


def selftest() -> None:
    """Provjere (AssertionError ako ne vrijedi); na kraju ispis oba kalendara za 2026-11-01 20:00 Europe/Zagreb."""
    lat, lon = ZAGREB_LAT, ZAGREB_LON

    # --- 0. ulaz mora biti aware ---
    for fn in (lambda: hebrew_date(datetime(2026, 11, 1, 20, 0), lat, lon),
               lambda: chinese_lunar(datetime(2026, 11, 1, 20, 0), lon)):
        try:
            fn()
        except ValueError:
            pass
        else:
            raise AssertionError("naive datetime mora podici ValueError")

    # --- 1. jednadzba vremena i predznak TST ---
    dt_draw = datetime(2026, 11, 1, 20, 0, tzinfo=TZ)
    assert dt_draw.utcoffset() == timedelta(hours=1), "1.11.2026. mora biti CET (UTC+1)"
    eot = _equation_of_time_minutes(_jd_ut(dt_draw))
    assert 16.2 <= eot <= 16.7, f"EoT 1.11.2026 ocekivano ~ +16.4 min, dobiveno {eot:.3f}"
    tst = _true_solar_time(dt_draw, lon)
    tst_h = tst.hour + tst.minute / 60 + tst.second / 3600
    assert tst.date() == date(2026, 11, 1) and 20.30 <= tst_h <= 20.37, f"TST ocekivano ~20:20, dobiveno {tst}"
    # negativna EoT (veljaca): 12:00 UT, lon 15.9833 -> 12:00 + 63.93 min - 14.2 min ~ 12:49.8
    dt_feb = datetime(2026, 2, 11, 12, 0, tzinfo=_UTC)
    eot_feb = _equation_of_time_minutes(_jd_ut(dt_feb))
    assert -14.6 <= eot_feb <= -13.8, f"EoT 11.2.2026 ocekivano ~ -14.2 min, dobiveno {eot_feb:.3f}"
    tst_feb = _true_solar_time(dt_feb, lon)
    tst_feb_h = tst_feb.hour + tst_feb.minute / 60 + tst_feb.second / 3600
    assert 12.80 <= tst_feb_h <= 12.86, f"TST 11.2.2026 12:00 UT ocekivano ~12:49.8, dobiveno {tst_feb}"
    # JD pretvorbe su medjusobno inverzne
    back = _jd_to_utc(_jd_ut(dt_draw))
    assert abs((back - dt_draw).total_seconds()) < 0.01, back

    # --- 2. zalazak Sunca u Zagrebu ---
    ss = _sunset_local(date(2026, 11, 1), lat, lon)
    assert ss.tzinfo is not None and ss.utcoffset() == timedelta(hours=1)
    assert ss.date() == date(2026, 11, 1) and time(16, 35) <= ss.time() <= time(16, 50), f"zalazak 1.11.2026 {ss}"
    ss_jun = _sunset_local(date(2026, 6, 21), lat, lon)
    assert time(20, 40) <= ss_jun.time() <= time(21, 0) and ss_jun.utcoffset() == timedelta(hours=2), ss_jun
    ss_dec = _sunset_local(date(2026, 12, 21), lat, lon)
    assert time(16, 5) <= ss_dec.time() <= time(16, 25), ss_dec
    ss_sep = _sunset_local(date(2026, 9, 11), lat, lon)
    assert time(19, 5) <= ss_sep.time() <= time(19, 25) and ss_sep.utcoffset() == timedelta(hours=2), ss_sep

    # --- 3. tablica dvosatnih ogranaka ---
    assert len(HOUR_BRANCH_TABLE) == 12 and [r[0] for r in HOUR_BRANCH_TABLE] == list(range(1, 13))
    assert [r[1] for r in HOUR_BRANCH_TABLE] == list(EARTHLY_BRANCHES)
    assert [r[2] for r in HOUR_BRANCH_TABLE] == list(EARTHLY_BRANCHES_CN)
    for no, _p, _c, h0, h1 in HOUR_BRANCH_TABLE:
        assert hour_branch_from_hours(h0) == no, (no, h0)                 # pocetak intervala pripada ogranku
        assert hour_branch_from_hours((h0 + 1) % 24) == no, (no, h0 + 1)  # sredina intervala
        assert hour_branch_from_hours(h1 - 0.001) == no, (no, h1)         # tik prije kraja
    assert hour_branch_from_hours(0.0) == 1 and hour_branch_from_hours(23.999) == 1 and hour_branch_from_hours(22.999) == 12
    assert hour_branch_from_hours(20.34) == 11

    # --- 4. kineska nova godina i ganzhi godine ---
    assert LunarDate.from_solar_date(2026, 2, 16).month == 12 and LunarDate.from_solar_date(2026, 2, 16).year == 2025
    cny26 = chinese_lunar(datetime(2026, 2, 17, 12, 0, tzinfo=TZ), lon)
    assert (cny26["lunar_year"], cny26["lunar_month"], cny26["lunar_day"]) == (2026, 1, 1), cny26
    assert cny26["year_ganzhi"] == "Bing Wu" and cny26["year_ganzhi_cn"] == "丙午" and cny26["year_branch"] == 7, cny26
    assert cny26["year_stem"] == 3 and not cny26["is_leap_month"]
    assert LunarDate.from_solar_date(2027, 2, 5).month == 12 and LunarDate.from_solar_date(2027, 2, 5).year == 2026
    cny27 = chinese_lunar(datetime(2027, 2, 6, 12, 0, tzinfo=TZ), lon)
    assert (cny27["lunar_year"], cny27["lunar_month"], cny27["lunar_day"]) == (2027, 1, 1), cny27
    assert cny27["year_ganzhi"] == "Ding Wei" and cny27["year_ganzhi_cn"] == "丁未" and cny27["year_branch"] == 8, cny27
    # prijestupni lunarni mjesec: 2025. ima prijestupni 6. mjesec; 30.7.2025. = 6. dan prijestupnog 6. mjeseca
    lm = chinese_lunar(datetime(2025, 7, 30, 12, 0, tzinfo=TZ), lon)
    assert (lm["lunar_year"], lm["lunar_month"], lm["lunar_day"], lm["is_leap_month"]) == (2025, 6, 6, True), lm
    assert lm["year_ganzhi"] == "Yi Si" and lm["year_ganzhi_cn"] == "乙巳" and lm["year_branch"] == 6, lm
    # 60-godisnji ciklus: 2024 Jia Chen (甲辰), 1984 Jia Zi (甲子)
    assert chinese_lunar(datetime(2024, 6, 1, 12, 0, tzinfo=TZ), lon)["year_ganzhi_cn"] == "甲辰"
    assert chinese_lunar(datetime(1984, 6, 1, 12, 0, tzinfo=TZ), lon)["year_ganzhi_cn"] == "甲子"

    # --- 5. kineski lunarni datum za kolo 1.11.2026. 20:00 ---
    cl = chinese_lunar(dt_draw, lon)
    assert (cl["lunar_year"], cl["lunar_month"], cl["lunar_day"]) == (2026, 9, 23), cl
    assert cl["hour_branch"] == 11 and cl["branch_name"] == "Xu 戌" and cl["hour_branch_cn"] == "戌", cl
    assert cl["year_branch"] == 7 and cl["year_ganzhi"] == "Bing Wu" and not cl["next_day_rule"]
    assert cl["tst"] == tst and cl["effective_date"] == date(2026, 11, 1)
    # pravilo TST >= 23:00 -> sljedeci lunarni dan, ogranak Zi
    late = chinese_lunar(datetime(2026, 11, 1, 22, 50, tzinfo=TZ), lon)   # TST ~ 23:10
    assert late["tst"].hour == 23 and late["next_day_rule"], late["tst"]
    assert (late["lunar_month"], late["lunar_day"], late["hour_branch"]) == (9, 24, 1), late
    assert late["branch_name"] == "Zi 子" and late["effective_date"] == date(2026, 11, 2)
    before = chinese_lunar(datetime(2026, 11, 1, 22, 30, tzinfo=TZ), lon)  # TST ~ 22:50
    assert (before["lunar_day"], before["hour_branch"]) == (23, 12) and not before["next_day_rule"], before
    # ljeti: 00:30 CEST -> TST ~ 23:30 prethodnog datuma -> +1 = isti civilni dan (ne civilni dan + 1)
    summer = chinese_lunar(datetime(2026, 6, 15, 0, 30, tzinfo=TZ), lon)
    assert summer["tst"].date() == date(2026, 6, 14) and summer["next_day_rule"], summer["tst"]
    assert summer["effective_date"] == date(2026, 6, 15) and summer["hour_branch"] == 1, summer
    # podne po TST -> Wu
    assert chinese_lunar(datetime(2026, 11, 1, 11, 45, tzinfo=TZ), lon)["hour_branch"] == 7

    # --- 6. hebrejski datum ---
    h_noon = hebrew_date(datetime(2026, 11, 1, 12, 0, tzinfo=TZ), lat, lon)
    assert (h_noon["year"], h_noon["month_no"], h_noon["day"]) == (5787, 8, 21), h_noon
    assert h_noon["month_name"] == "Cheshvan" and h_noon["month_name_hr"] == "Hesvan", h_noon
    assert h_noon["after_sunset"] is False and h_noon["sunset"] == ss and h_noon["effective_date"] == date(2026, 11, 1)
    h_eve = hebrew_date(dt_draw, lat, lon)
    assert (h_eve["year"], h_eve["month_no"], h_eve["day"]) == (5787, 8, 22) and h_eve["after_sunset"] is True, h_eve
    assert h_eve["effective_date"] == date(2026, 11, 2) and h_eve["civil_date"] == date(2026, 11, 1)
    # tocno u trenutku zalaska -> vec novi dan; sekundu prije -> stari
    assert hebrew_date(ss, lat, lon)["day"] == 22 and hebrew_date(ss - timedelta(seconds=1), lat, lon)["day"] == 21
    # poslije ponoci, prije zalaska -> i dalje 22 Hesvan
    h_night = hebrew_date(datetime(2026, 11, 2, 1, 0, tzinfo=TZ), lat, lon)
    assert (h_night["month_no"], h_night["day"], h_night["after_sunset"]) == (8, 22, False), h_night
    # Ros Hasana 5787
    rh = hebrew_date(datetime(2026, 9, 12, 12, 0, tzinfo=TZ), lat, lon)
    assert (rh["year"], rh["month_no"], rh["day"]) == (5787, 7, 1) and rh["month_name"] == "Tishrei", rh
    assert rh["month_name_hr"] == "Tisri" and rh["leap_year"] is True
    rh_eve = hebrew_date(datetime(2026, 9, 11, 20, 0, tzinfo=TZ), lat, lon)
    assert (rh_eve["year"], rh_eve["month_no"], rh_eve["day"]) == (5787, 7, 1) and rh_eve["after_sunset"] is True, rh_eve
    rh_before = hebrew_date(datetime(2026, 9, 11, 12, 0, tzinfo=TZ), lat, lon)
    assert (rh_before["year"], rh_before["month_no"], rh_before["day"]) == (5786, 6, 29), rh_before
    assert rh_before["month_name"] == "Elul" and rh_before["leap_year"] is False
    # numeriranje mjeseci: Nisan = 1, Adar I = 12 / Adar II = 13 (prijestupna 5784), Adar = 12 (obicna 5786)
    nis = hebrew_date(datetime(2026, 3, 20, 12, 0, tzinfo=TZ), lat, lon)
    assert (nis["year"], nis["month_no"], nis["day"]) == (5786, 1, 2) and nis["month_name"] == "Nissan", nis
    ad1 = hebrew_date(datetime(2024, 2, 15, 12, 0, tzinfo=TZ), lat, lon)
    assert (ad1["year"], ad1["month_no"], ad1["day"]) == (5784, 12, 6) and ad1["month_name"] == "Adar 1", ad1
    assert ad1["month_name_hr"] == "Adar I"
    ad2 = hebrew_date(datetime(2024, 3, 15, 12, 0, tzinfo=TZ), lat, lon)
    assert (ad2["year"], ad2["month_no"], ad2["day"]) == (5784, 13, 5) and ad2["month_name"] == "Adar 2", ad2
    assert ad2["month_name_hr"] == "Adar II"
    ad = hebrew_date(datetime(2026, 3, 1, 12, 0, tzinfo=TZ), lat, lon)
    assert (ad["year"], ad["month_no"], ad["day"]) == (5786, 12, 12) and ad["month_name"] == "Adar", ad
    assert ad["month_name_hr"] == "Adar" and ad["leap_year"] is False
    # unakrsna provjera pyluach vs convertdate za svaki dan 2020.-2030. (bez zalaska, cisto numeriranje)
    d = date(2020, 1, 1)
    n_days = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)   # interni DeprecationWarning convertdatea
        while d <= date(2030, 12, 31):
            hh = HebrewDate.from_pydate(d)
            assert (hh.year, hh.month, hh.day) == tuple(cd_hebrew.from_gregorian(d.year, d.month, d.day)), d
            assert 1 <= hh.month <= 13 and 1 <= hh.day <= 30
            n_days += 1
            d += timedelta(days=1)
    assert n_days == 4018
    # unakrsna provjera kroz samu funkciju (sa zalaskom) za sva kola razdoblja 1.11.2026.-31.10.2027.
    from config import draw_dates, draw_datetime
    n_draws = 0
    n_after = 0
    for dd in draw_dates():
        ddt = draw_datetime(dd)
        r = hebrew_date(ddt, lat, lon)
        # 20:00 lokalno NIJE uvijek poslije zalaska (ljeti u Zagrebu Sunce zalazi do ~20:50 CEST) -> provjera dosljednosti
        assert r["year"] in (5787, 5788) and r["after_sunset"] == (ddt >= r["sunset"]), (dd, r)
        assert r["sunset"].date() == dd and r["effective_date"] == dd + timedelta(days=int(r["after_sunset"])), (dd, r)
        n_after += int(r["after_sunset"])
        c = chinese_lunar(ddt, lon)
        assert c["lunar_year"] in (2026, 2027) and c["hour_branch"] in (10, 11), (dd, c)  # 20:00 -> You ili Xu po TST
        n_draws += 1
    assert n_draws == len(draw_dates()) and n_draws > 0, n_draws
    assert 0 <= n_after <= n_draws, n_after   # u studenom/prosincu 20:00 je uvijek poslije zalaska

    # --- ispis ---
    print("calendars selftest OK")
    print(f"  Kolo: {dt_draw.isoformat()}  (lat {lat:.4f}, lon {lon:.4f}, Zagreb)")
    print(f"  Zalazak Sunca (swe.rise_trans, FLG_MOSEPH): {ss.isoformat()}")
    print(f"  Jednadzba vremena (swe.time_equ): {eot:+.2f} min;  TST = UT + lon/15 h + EoT = {_fmt_tst(tst)}")
    print(f"  Hebrejski: {h_eve['day']}. {h_eve['month_name_hr']} ({h_eve['month_name']}, mjesec {h_eve['month_no']}) "
          f"{h_eve['year']}  after_sunset={h_eve['after_sunset']}  (prije zalaska: {h_noon['day']}. "
          f"{h_noon['month_name_hr']} {h_noon['year']})")
    print(f"  Kineski lunarni: godina {cl['lunar_year']} {cl['year_ganzhi']} {cl['year_ganzhi_cn']} "
          f"(year_branch {cl['year_branch']} {cl['year_branch_name']}), mjesec {cl['lunar_month']}, dan {cl['lunar_day']}, "
          f"prijestupni mjesec={cl['is_leap_month']}")
    print(f"  Dvosatni ogranak po TST {_fmt_tst(cl['tst'])}: hour_branch {cl['hour_branch']} ({cl['branch_name']}), "
          f"pravilo TST>=23h primijenjeno={cl['next_day_rule']}")
    print(f"  Kineska nova godina: 2026-02-17 -> {cny26['year_ganzhi']} {cny26['year_ganzhi_cn']} (ogranak {cny26['year_branch']}); "
          f"2027-02-06 -> {cny27['year_ganzhi']} {cny27['year_ganzhi_cn']} (ogranak {cny27['year_branch']})")
    print(f"  Ros Hasana 5787: 2026-09-12 12:00 -> {rh['day']}. {rh['month_name_hr']} {rh['year']}; "
          f"2026-09-11 20:00 (zalazak {ss_sep.strftime('%H:%M')}) -> {rh_eve['day']}. {rh_eve['month_name_hr']} {rh_eve['year']}")
    print(f"  Unakrsna provjera pyluach/convertdate: {n_days} dana (2020.-2030.) bez razlike; {n_draws} kola razdoblja "
          f"provjereno ({n_after} kola u 20:00 poslije zalaska, {n_draws - n_after} prije zalaska - ljetni mjeseci).")


if __name__ == "__main__":
    selftest()
