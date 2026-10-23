from datetime import date, datetime, timezone

import pytest
import swisseph as swe

from methods import ephemeris as eph
from methods.chinese_calendar import hour_branch, lunar_date


def test_julian_day_j2000():
    assert eph.julian_day_ut(datetime(2000, 1, 1, 12, tzinfo=timezone.utc)) == pytest.approx(2451545.0)


def test_natal_sun_is_gemini(subject):
    jd = eph.julian_day_ut(subject.birth_local)
    lon, _ = eph.longitude(jd, swe.SUN)
    num, sign, deg = eph.sign_of(lon)
    assert sign == "Gemini" and 9.0 < deg < 10.5


def test_lahiri_ayanamsa_1991():
    jd = swe.julday(1991, 5, 31, 16.33)
    assert 23.6 < eph.ayanamsa_lahiri(jd) < 23.9


def test_sidereal_equals_tropical_minus_ayanamsa():
    jd = swe.julday(2026, 10, 23, 17.0)
    trop = eph.longitude(jd, swe.MOON)[0]
    sid = eph.longitude(jd, swe.MOON, sidereal=True)[0]
    # Within nutation in longitude (< 20"): the ayanamsa value is a mean one.
    assert (trop - sid - eph.ayanamsa_lahiri(jd) + 180) % 360 - 180 == pytest.approx(0, abs=0.006)


def test_sunrise_zagreb(event):
    rise = eph.sunrise_before(eph.julian_day_ut(event.local), event.place.lat, event.place.lon)
    local = eph.jd_to_utc(rise).astimezone(event.local.tzinfo)
    assert local.date() == date(2026, 10, 23) and 7 <= local.hour < 8


def test_house_of():
    cusps = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
    assert eph.house_of(15, cusps) == 1 and eph.house_of(359, cusps) == 12
    wrap = [350, 20, 50, 80, 110, 140, 170, 200, 230, 260, 290, 320]
    assert eph.house_of(5, wrap) == 1 and eph.house_of(345, wrap) == 12


def test_new_moon_known_date():
    # New moon of 2026-10-10 (around 15:50 UT).
    jd0 = swe.julday(2026, 10, 1, 0)
    moons = eph.new_moons_between(jd0, jd0 + 20)
    assert len(moons) == 1
    dt = eph.jd_to_utc(moons[0])
    assert dt.date() == date(2026, 10, 10)


@pytest.mark.parametrize("civil,expected", [
    (date(1991, 2, 15), (1991, 1, 1, False)),     # Chinese New Year 1991
    (date(2025, 1, 29), (2025, 1, 1, False)),     # Chinese New Year 2025
    (date(2026, 2, 17), (2026, 1, 1, False)),     # Chinese New Year 2026
    (date(2025, 7, 25), (2025, 6, 1, True)),      # leap 6th month of 2025
    (date(2025, 10, 6), (2025, 8, 15, False)),    # Mid-Autumn 2025
    (date(2026, 9, 25), (2026, 8, 15, False)),    # Mid-Autumn 2026
    (date(2025, 1, 28), (2024, 12, 29, False)),   # New Year's Eve 2025 fell on 腊月廿九 (no 30th)
    (date(1991, 5, 31), (1991, 4, 18, False)),    # birth
    (date(2026, 10, 23), (2026, 9, 14, False)),   # draw
])
def test_lunar_dates(civil, expected):
    ld = lunar_date(civil)
    assert (ld.year, ld.month, ld.day, ld.leap) == expected


def test_year_branches():
    assert lunar_date(date(1991, 5, 31)).year_stem_branch == "辛未"
    assert lunar_date(date(2026, 10, 23)).year_stem_branch == "丙午"
    assert lunar_date(date(2025, 1, 28)).year_stem_branch == "甲辰"


@pytest.mark.parametrize("hour,branch", [(23.5, 1), (0.2, 1), (1.0, 2), (12.9, 7), (17.0, 10), (18.99, 10),
                                         (19.0, 11), (22.5, 12)])
def test_hour_branch(hour, branch):
    assert hour_branch(hour) == branch


def test_true_solar_time_zagreb(event):
    tst = eph.true_solar_time(event.local, event.place.lon)
    # 17:00 UT + 1h03m56s longitude + ~15.8 min equation of time.
    assert tst.hour == 18 and 15 <= tst.minute <= 23
