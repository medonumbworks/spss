"""Reproducible wrapper around pyswisseph (Swiss Ephemeris by Astrodienst AG).

The Moshier analytical ephemeris that ships inside the library is used on purpose
(``FLG_MOSEPH``): it needs no external data files, so every machine computes identical
numbers. Its error is far below the resolution of anything mapped here (a lunar
mansion is 13°20', the finest zodiac arc used is 7.2°).

Reference: Swiss Ephemeris documentation, https://www.astro.com/swisseph/swephprg.htm
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from functools import lru_cache

import numpy as np
import swisseph as swe
from scipy.optimize import brentq

FLAGS = swe.FLG_MOSEPH | swe.FLG_SPEED

PLANETS = {
    "Sun": swe.SUN, "Moon": swe.MOON, "Mercury": swe.MERCURY, "Venus": swe.VENUS, "Mars": swe.MARS,
    "Jupiter": swe.JUPITER, "Saturn": swe.SATURN, "Uranus": swe.URANUS, "Neptune": swe.NEPTUNE,
    "Pluto": swe.PLUTO,
}
SIGNS = ["Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
         "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces"]


def julian_day_ut(moment: datetime) -> float:
    if moment.tzinfo is None:
        raise ValueError("moment must be timezone-aware")
    u = moment.astimezone(timezone.utc)
    hours = u.hour + u.minute / 60 + (u.second + u.microsecond / 1e6) / 3600
    return swe.julday(u.year, u.month, u.day, hours, swe.GREG_CAL)


def jd_to_utc(jd: float) -> datetime:
    y, m, d, h = swe.revjul(jd, swe.GREG_CAL)
    return datetime(y, m, d, tzinfo=timezone.utc) + timedelta(hours=h)


def _set_lahiri() -> None:
    swe.set_sid_mode(swe.SIDM_LAHIRI, 0, 0)


def longitude(jd: float, body: int, sidereal: bool = False) -> tuple[float, float]:
    """Geocentric apparent ecliptic longitude (deg) and speed (deg/day)."""
    flags = FLAGS
    if sidereal:
        _set_lahiri()
        flags |= swe.FLG_SIDEREAL
    xx, _ = swe.calc_ut(jd, body, flags)
    return xx[0] % 360.0, xx[3]


def mean_node(jd: float, sidereal: bool = False) -> float:
    return longitude(jd, swe.MEAN_NODE, sidereal)[0]


def ayanamsa_lahiri(jd: float) -> float:
    _set_lahiri()
    return swe.get_ayanamsa_ut(jd)


def houses(jd: float, lat: float, lon: float, hsys: str = "P", sidereal: bool = False) -> dict:
    """House cusps (1..12), ascendant and midheaven."""
    flags = 0
    if sidereal:
        _set_lahiri()
        flags = swe.FLG_SIDEREAL
    cusps, ascmc = swe.houses_ex(jd, lat, lon, hsys.encode(), flags)
    return {"cusps": [c % 360.0 for c in cusps[:12]], "asc": ascmc[0] % 360.0, "mc": ascmc[1] % 360.0}


def house_of(lon_deg: float, cusps: list[float]) -> int:
    """House number 1..12 for a longitude, given the 12 cusps in order."""
    for i in range(12):
        start, end = cusps[i], cusps[(i + 1) % 12]
        span = (end - start) % 360.0
        if (lon_deg - start) % 360.0 < span:
            return i + 1
    return 12


def sunrise_before(jd: float, lat: float, lon: float) -> float:
    """The last sunrise at (lat, lon) at or before ``jd`` (upper limb, standard refraction)."""
    res, tret = swe.rise_trans(jd - 1.0, swe.SUN, swe.CALC_RISE, (lon, lat, 0.0), 1013.25, 10.0, swe.FLG_MOSEPH)
    if res != 0:
        raise RuntimeError("no sunrise found")
    rise = tret[0]
    nxt_res, nxt = swe.rise_trans(rise + 0.01, swe.SUN, swe.CALC_RISE, (lon, lat, 0.0), 1013.25, 10.0,
                                  swe.FLG_MOSEPH)
    if nxt_res == 0 and nxt[0] <= jd:
        rise = nxt[0]
    return rise


def equation_of_time_days(jd: float) -> float:
    """Local apparent minus local mean time, in days."""
    return swe.time_equ(jd)


def true_solar_time(moment: datetime, lon: float) -> datetime:
    """Local apparent (sundial) time as a naive datetime: UTC + lon/15 h + equation of time."""
    jd = julian_day_ut(moment)
    u = moment.astimezone(timezone.utc).replace(tzinfo=None)
    return u + timedelta(hours=lon / 15.0) + timedelta(days=equation_of_time_days(jd))


def _elongation(jd: float) -> float:
    """Moon minus Sun longitude wrapped to [-180, 180)."""
    e = longitude(jd, swe.MOON)[0] - longitude(jd, swe.SUN)[0]
    return (e + 180.0) % 360.0 - 180.0


@lru_cache(maxsize=64)
def new_moons_between(jd_start: float, jd_end: float) -> tuple[float, ...]:
    """Julian days (UT) of every new moon in [jd_start, jd_end], found with Brent's method."""
    out = []
    t0, e0 = jd_start, _elongation(jd_start)
    t = jd_start
    while t < jd_end:
        t1 = min(t + 1.0, jd_end)
        e1 = _elongation(t1)
        # A new moon is a - to + crossing; the +180 -> -180 wrap at full moon is skipped.
        if e0 < 0.0 <= e1 and (e1 - e0) < 90.0:
            out.append(brentq(_elongation, t0, t1, xtol=1e-7))
        t0, e0, t = t1, e1, t1
    return tuple(out)


def sun_crossings(jd_start: float, jd_end: float, step_deg: float = 30.0) -> list[tuple[float, float]]:
    """(jd, target longitude) for every crossing of a multiple of ``step_deg`` by the Sun."""
    out = []
    lon0 = longitude(jd_start, swe.SUN)[0]
    target = (np.floor(lon0 / step_deg) + 1) * step_deg % 360.0
    t = jd_start
    while True:
        t = swe.solcross_ut(float(target), t, swe.FLG_MOSEPH)
        if t > jd_end:
            return out
        out.append((t, float(target)))
        target = (target + step_deg) % 360.0
        t += 1.0


def sign_of(lon_deg: float) -> tuple[int, str, float]:
    idx = int(lon_deg // 30) % 12
    return idx + 1, SIGNS[idx], lon_deg - 30 * idx
