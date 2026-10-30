"""Zajednicke postavke registra Loto 6/45. Vrijednosti oznacene PRETPOSTAVKA nisu potvrdjene od korisnika."""
from __future__ import annotations
import datetime as dt
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
TZ = ZoneInfo("Europe/Zagreb")

# Natalni podaci (zadano): 31.05.1991., 18:20 CEST (UTC+2), Zadar 44 07 N 15 14 E, ime IVAN
NATAL_LOCAL = dt.datetime(1991, 5, 31, 18, 20, tzinfo=dt.timezone(dt.timedelta(hours=2)))
NATAL_LAT = 44 + 7 / 60
NATAL_LON = 15 + 14 / 60
NAME = "IVAN"
LIFE_PATH = (29, 11)      # zivotni put 29/11 (zadano)
BIRTHDAY_NUMBER = 31      # rodjendanski broj (zadano)
NAME_PYTHAGOREAN = 19     # IVAN = 19 (zadano; provjerava se u numerology.py)

# Mjesto kola: Zagreb 45 49 N 15 59 E
ZAGREB_LAT = 45 + 49 / 60
ZAGREB_LON = 15 + 59 / 60

# --- PRETPOSTAVKE (korisnik nije upisao) ---
DRAW_TIME_LOCAL = dt.time(20, 0)            # PRETPOSTAVKA: 20:00 lokalno
DRAW_WEEKDAYS = (2, 6)                      # PRETPOSTAVKA: srijeda (2) i nedjelja (6); Python weekday Mon=0
PERIOD_START = dt.date(2026, 11, 1)         # PRETPOSTAVKA: 01.11.2026.-31.10.2027.
PERIOD_END = dt.date(2027, 10, 31)
PRICE_EUR = 1.00                            # PRETPOSTAVKA: cijena kombinacije
WIN_CLASSES = (6, 5, 4, 3)                  # PRETPOSTAVKA: dobitne vrste 6, 5, 4, 3 pogotka
# -------------------------------------------

HISTORY_CSV = HERE / "povijest.csv"
PLAYED_CSV = HERE / "odigrano.csv"
SEED_FILE = HERE / "seed.txt"

DAY_NAMES_HR = ["PONEDJELJAK", "UTORAK", "SRIJEDA", "CETVRTAK", "PETAK", "SUBOTA", "NEDJELJA"]
DAY_PLANET_HR = ["Mjesec", "Mars", "Merkur", "Jupiter", "Venera", "Saturn", "Sunce"]  # weekday Mon=0..Sun=6
PLANET_NUMBER = {"Sunce": 1, "Mjesec": 2, "Jupiter": 3, "Rahu": 4, "Merkur": 5, "Venera": 6,
                 "Ketu": 7, "Saturn": 8, "Mars": 9}

SYSTEM_NAMES = {
    1: "Pareto 1", 2: "Pareto 2", 3: "Vruci 1", 4: "Vruci 2", 5: "Kaldejska", 6: "Gematrija",
    7: "Hermetika", 8: "I Ching", 9: "Lenormand", 10: "Meihua Yishu", 11: "Tarot",
    12: "Pitagorejska", 13: "Jyotis", 14: "Zapadna", 15: "Plecko",
}


def read_seed() -> int:
    return int(SEED_FILE.read_text().strip())


def draw_dates(start: dt.date = PERIOD_START, end: dt.date = PERIOD_END, weekdays=DRAW_WEEKDAYS) -> list[dt.date]:
    out = []
    d = start
    while d <= end:
        if d.weekday() in weekdays:
            out.append(d)
        d += dt.timedelta(days=1)
    return out


def draw_datetime(d: dt.date) -> dt.datetime:
    return dt.datetime.combine(d, DRAW_TIME_LOCAL, tzinfo=TZ)
