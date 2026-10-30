"""Ucitavanje i provjera povijest.csv (datum;kolo;n1;...;n6) te frekvencijski sustavi Pareto i Vruci."""
from __future__ import annotations
import datetime as dt
from pathlib import Path

import pandas as pd

import config


def _parse_date(s: str) -> dt.date | None:
    s = s.strip().rstrip(".")
    for fmt in ("%d.%m.%Y", "%Y-%m-%d", "%d/%m/%Y"):
        try:
            return dt.datetime.strptime(s, fmt).date()
        except ValueError:
            pass
    return None


def load_history(path: Path | str = config.HISTORY_CSV) -> dict | None:
    path = Path(path)
    if not path.exists():
        return None
    rows, rejected = [], []
    seen_dates, seen_kolo = set(), set()
    with open(path, encoding="utf-8-sig") as f:
        for ln, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.replace(",", ";").split(";")]
            if ln == 1 and parts and parts[0].lower().startswith("datum"):
                continue
            if len(parts) < 8:
                rejected.append((ln, line, f"premalo polja ({len(parts)} < 8)"))
                continue
            d = _parse_date(parts[0])
            if d is None:
                rejected.append((ln, line, "neispravan datum"))
                continue
            kolo = parts[1]
            try:
                nums = [int(p) for p in parts[2:8]]
            except ValueError:
                rejected.append((ln, line, "brojevi nisu cijeli"))
                continue
            if any(n < 1 or n > 45 for n in nums):
                rejected.append((ln, line, "broj izvan raspona 1-45"))
                continue
            if len(set(nums)) != 6:
                rejected.append((ln, line, "ponovljen broj unutar kola"))
                continue
            if d in seen_dates:
                rejected.append((ln, line, "duplikat datuma"))
                continue
            if kolo and kolo in seen_kolo:
                rejected.append((ln, line, "duplikat broja kola"))
                continue
            seen_dates.add(d)
            seen_kolo.add(kolo)
            rows.append({"datum": d, "kolo": kolo, **{f"n{i+1}": n for i, n in enumerate(nums)}})
    df = pd.DataFrame(rows).sort_values("datum").reset_index(drop=True) if rows else pd.DataFrame(
        columns=["datum", "kolo"] + [f"n{i}" for i in range(1, 7)])
    first = df["datum"].iloc[0] if len(df) else None
    last = df["datum"].iloc[-1] if len(df) else None
    missing = 0
    if last is not None:
        today = dt.date.today()
        d = last + dt.timedelta(days=1)
        while d <= today:
            if d.weekday() in config.DRAW_WEEKDAYS:
                missing += 1
            d += dt.timedelta(days=1)
    return {"df": df, "rejected": rejected, "n_valid": len(df), "first_date": first, "last_date": last,
            "expected_missing": missing}


def missing_message() -> str:
    return (f"povijest.csv nije pronadjena ({config.HISTORY_CSV}); sustavi 1-4 (Pareto 1/2, Vruci 1/2) i tie-break "
            f"'cesci u povijesti' nisu dostupni i NISU izracunati (nista se ne izmislja). Izvor 'Statistika' ne sudjeluje u Plecku.")


def describe(res: dict) -> str:
    L = [f"povijest.csv: {res['n_valid']} valjanih kola, {res['first_date']} - {res['last_date']}; "
         f"odbacenih redaka: {len(res['rejected'])}; procjena nedostajucih kola do danas: {res['expected_missing']}"]
    for ln, raw, why in res["rejected"]:
        L.append(f"  odbacen redak {ln}: {why}: {raw}")
    return "\n".join(L)


def frequencies(df: pd.DataFrame, last_n: int | None = None) -> dict[int, int]:
    sub = df.tail(last_n) if last_n else df
    vals = sub[[f"n{i}" for i in range(1, 7)]].to_numpy().ravel()
    freq = {n: 0 for n in range(1, 46)}
    for v in vals:
        freq[int(v)] += 1
    return freq


def _ranked(freq: dict[int, int], tiebreak: dict[int, int] | None = None) -> list[int]:
    return sorted(freq, key=lambda n: (-freq[n], -(tiebreak or {}).get(n, 0), n))


def pareto(df: pd.DataFrame) -> tuple[list[int], list[int]]:
    r = _ranked(frequencies(df))
    return r[:6], r[6:12]


def vruci(df: pd.DataFrame) -> tuple[list[int], list[int], int]:
    win = 100
    f100 = frequencies(df, 100)
    f250 = frequencies(df, 250)
    r = _ranked(f100, f250)
    return r[:6], r[6:12], win


def selftest() -> None:
    import tempfile
    tmp = Path("/tmp/claude-0/-home-user-spss/4d826b88-f2ab-56d3-85bd-f957b48b8ad3/scratchpad")
    tmp.mkdir(parents=True, exist_ok=True)
    p = tmp / "povijest_test.csv"
    lines = ["datum;kolo;n1;n2;n3;n4;n5;n6"]
    import random
    rnd = random.Random(1)
    d = dt.date(2024, 1, 3)
    k = 1
    while k <= 300:
        if d.weekday() in config.DRAW_WEEKDAYS:
            nums = sorted(rnd.sample(range(1, 46), 6))
            lines.append(f"{d.strftime('%d.%m.%Y.')};{k};" + ";".join(map(str, nums)))
            k += 1
        d += dt.timedelta(days=1)
    lines += ["01.01.2024.;1;1;2;3;4;5;6", "2024-01-03;999;1;2;3;4;5;46", "05.05.2024.;998;1;2;3;4;5", "06.05.2024.;997;1;1;2;3;4;5",
              "xx.yy.zzzz;996;1;2;3;4;5;6", "07.07.2024.;1;7;8;9;10;11;12"]
    p.write_text("﻿" + "\n".join(lines) + "\n", encoding="utf-8")
    r = load_history(p)
    assert r["n_valid"] == 300 and len(r["rejected"]) == 6, (r["n_valid"], len(r["rejected"]))
    print(describe(r))
    assert load_history(tmp / "ne_postoji.csv") is None
    print(missing_message())
    p1, p2 = pareto(r["df"])
    v1, v2, w = vruci(r["df"])
    assert len(p1) == 6 and len(p2) == 6 and not set(p1) & set(p2) and w == 100
    print("Pareto:", p1, p2, "Vruci:", v1, v2)
    print("history selftest OK")


if __name__ == "__main__":
    selftest()
