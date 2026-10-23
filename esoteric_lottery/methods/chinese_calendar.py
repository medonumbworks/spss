"""Chinese lunisolar calendar computed from the ephemeris (needed by Meihua Yishu).

Rules (modern Chinese calendar, as codified in GB/T 33661-2017 "Calculation and
promulgation of the Chinese calendar"):

1. All times are Beijing time, UTC+8 (no daylight saving).
2. A month starts on the civil day that contains a new moon.
3. The month containing the winter solstice (Sun at 270°) is month 11.
4. If 13 months lie between two month-11 starts, the first of them that contains no
   principal term (zhongqi: Sun at a multiple of 30°) is a leap month and repeats the
   number of the month before it.
5. The year number changes at month 1 (Chinese New Year).

Validated in tests against published dates: New Year 1991-02-15, 2025-01-29,
2026-02-17; leap 6th month of 2025 starting 2025-07-25; Mid-Autumn (8/15) on
2025-10-06 and 2026-09-25.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache

import swisseph as swe

from . import ephemeris as eph

CST = timezone(timedelta(hours=8))
EARTHLY_BRANCHES = ["子", "丑", "寅", "卯", "辰", "巳", "午", "未", "申", "酉", "戌", "亥"]
HEAVENLY_STEMS = ["甲", "乙", "丙", "丁", "戊", "己", "庚", "辛", "壬", "癸"]


@dataclass(frozen=True)
class LunarDate:
    year: int          # Gregorian year in which this lunar year began
    month: int         # 1..12
    day: int           # 1..30
    leap: bool
    year_branch: int   # 1 (子) .. 12 (亥)
    year_stem_branch: str

    def to_json(self) -> dict:
        return {"lunar_year": self.year, "month": self.month, "day": self.day, "leap_month": self.leap,
                "year_branch_number": self.year_branch,
                "year_branch": EARTHLY_BRANCHES[self.year_branch - 1],
                "year_stem_branch": self.year_stem_branch}


def _cst_date(jd: float) -> date:
    return eph.jd_to_utc(jd).astimezone(CST).date()


def _jd_of_date(d: date) -> float:
    return eph.julian_day_ut(datetime(d.year, d.month, d.day, tzinfo=CST))


@lru_cache(maxsize=64)
def _month11_start(year: int) -> date:
    """Start date of the lunar month that contains the December solstice of ``year``."""
    ws = swe.solcross_ut(270.0, swe.julday(year, 11, 1, 0.0), swe.FLG_MOSEPH)
    ws_day = _cst_date(ws)
    moons = eph.new_moons_between(ws - 35.0, ws + 1.0)
    starts = [d for d in (_cst_date(m) for m in moons) if d <= ws_day]
    return starts[-1]


@lru_cache(maxsize=64)
def _year_table(year: int) -> tuple[tuple[date, int, bool, int], ...]:
    """Months from month 11 before ``year``'s Chinese New Year up to (not incl.) the next
    month 11: (start date, month number, leap, lunar year)."""
    a, b = _month11_start(year - 1), _month11_start(year)
    jd_a, jd_b = _jd_of_date(a) - 1.0, _jd_of_date(b) + 1.0
    starts = [d for d in (_cst_date(m) for m in eph.new_moons_between(jd_a, jd_b)) if a <= d < b]
    bounds = starts + [b]
    leap_index = None
    if len(starts) == 13:
        zhongqi_days = [_cst_date(jd) for jd, _ in eph.sun_crossings(jd_a, jd_b, 30.0)]
        for i in range(1, 13):
            if not any(bounds[i] <= z < bounds[i + 1] for z in zhongqi_days):
                leap_index = i
                break
    elif len(starts) != 12:
        raise RuntimeError(f"unexpected {len(starts)} lunar months before {b}")
    table, number, seen_first = [], 10, False
    for i, s in enumerate(starts):
        leap = i == leap_index
        if not leap:
            number = number % 12 + 1
        if number == 1 and not leap:
            seen_first = True
        table.append((s, number, leap, year if seen_first else year - 1))
    return tuple(table)


def stem_branch_of_year(lunar_year: int) -> str:
    return HEAVENLY_STEMS[(lunar_year - 4) % 10] + EARTHLY_BRANCHES[(lunar_year - 4) % 12]


def lunar_date(civil: date) -> LunarDate:
    """Chinese lunar date of a civil calendar day."""
    year = civil.year + 1 if civil >= _month11_start(civil.year) else civil.year
    table = _year_table(year)
    for i, (start, month, leap, lyear) in enumerate(table):
        end = table[i + 1][0] if i + 1 < len(table) else _month11_start(year)
        if start <= civil < end:
            return LunarDate(lyear, month, (civil - start).days + 1, leap, (lyear - 4) % 12 + 1,
                             stem_branch_of_year(lyear))
    raise RuntimeError(f"no lunar month found for {civil}")


def hour_branch(hour: float) -> int:
    """Double-hour (時辰) number: 子 23:00-00:59 = 1, 丑 01:00-02:59 = 2, ..., 亥 = 12."""
    return int(((int(hour) + 1) // 2) % 12) + 1
