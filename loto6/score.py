"""Ocjenjivanje: rezultati.csv (datum;n1;...;n6) x loto6_godina.csv -> pogoci po sustavu i kolu, prosjek po sustavu,
ocekivanje 6*6/45 = 0,80 po kombinaciji, slucajna referenca (1.000 setova iste strukture), chi-kvadrat i p.
Pokretanje: .venv/bin/python score.py rezultati.csv [loto6_godina.csv]"""
from __future__ import annotations
import csv
import datetime as dt
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

import config
from core import rng_for

EXPECTED = 36 / 45  # 0.8
THRESHOLD_TEXT = ("Unaprijed dogovoren prag: samo prosjek >= 1,00 pogodaka po kombinaciji na >= 50 kola smatra se odstupanjem. "
                  "Nista od ovoga nema prediktivnu moc; svaka kombinacija ima 1 : 8.145.060 za sesticu.")


def _pdate(s: str) -> dt.date | None:
    s = s.strip().rstrip(".")
    for fmt in ("%d.%m.%Y", "%Y-%m-%d"):
        try:
            return dt.datetime.strptime(s, fmt).date()
        except ValueError:
            pass
    return None


def read_results(path: Path) -> dict[dt.date, set[int]]:
    out: dict[dt.date, set[int]] = {}
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            parts = [p.strip() for p in line.strip().replace(",", ";").split(";") if p.strip()]
            if len(parts) < 7:
                continue
            d = _pdate(parts[0])
            if d is None:
                continue
            try:
                nums = {int(p) for p in parts[1:7]}
            except ValueError:
                continue
            if len(nums) == 6 and all(1 <= n <= 45 for n in nums):
                out[d] = nums
    return out


def read_year(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f, delimiter=";")
        for row in r:
            d = _pdate(row["datum"])
            if d is None:
                continue
            rows.append({"datum": d, "sustav": row["sustav"], "nums": {int(row[f"n{i}"]) for i in range(1, 7)}})
    return rows


def hypergeom_probs() -> np.ndarray:
    return np.array([stats.hypergeom(45, 6, 6).pmf(k) for k in range(7)])


def chi_square(hits: list[int]) -> tuple[float, float, int]:
    """Kategorije 0, 1, 2, >=3 pogodaka (>=3 spojeno zbog malih ocekivanja)."""
    p = hypergeom_probs()
    pc = np.array([p[0], p[1], p[2], p[3:].sum()])
    obs = np.array([sum(1 for h in hits if h == 0), sum(1 for h in hits if h == 1), sum(1 for h in hits if h == 2),
                    sum(1 for h in hits if h >= 3)])
    n = len(hits)
    exp = pc * n
    if n == 0:
        return float("nan"), float("nan"), 0
    # spoji kategorije s ocekivanjem < 5 u susjednu
    while len(exp) > 2 and exp[-1] < 5:
        exp[-2] += exp[-1]
        obs[-2] += obs[-1]
        exp, obs = exp[:-1], obs[:-1]
    chi, pval = stats.chisquare(obs, exp)
    return float(chi), float(pval), len(exp) - 1


def evaluate(results: dict[dt.date, set[int]], year_rows: list[dict], n_random: int = 1000) -> dict:
    per_sys: dict[str, list[tuple[dt.date, int]]] = defaultdict(list)
    all_hits: list[int] = []
    draws_used: set[dt.date] = set()
    structure: dict[dt.date, int] = defaultdict(int)
    for row in year_rows:
        if row["datum"] not in results:
            continue
        h = len(row["nums"] & results[row["datum"]])
        per_sys[row["sustav"]].append((row["datum"], h))
        all_hits.append(h)
        draws_used.add(row["datum"])
        structure[row["datum"]] += 1
    summary = {}
    for s, lst in per_sys.items():
        hs = [h for _, h in lst]
        chi, pval, dof = chi_square(hs)
        summary[s] = {"n": len(hs), "mean": float(np.mean(hs)) if hs else float("nan"), "hits": hs, "chi2": chi, "p": pval,
                      "dof": dof, "dist": [hs.count(k) for k in range(7)]}
    chi, pval, dof = chi_square(all_hits)
    total_mean = float(np.mean(all_hits)) if all_hits else float("nan")
    # slucajna referenca: isti broj kola i kombinacija po kolu, slucajne kombinacije
    rng = rng_for(0, 99)
    ref_means = []
    dates = sorted(draws_used)
    for _ in range(n_random):
        tot, cnt = 0, 0
        for d in dates:
            k = structure[d]
            res = results[d]
            for _ in range(k):
                c = rng.choice(45, size=6, replace=False) + 1
                tot += len(set(int(x) for x in c) & res)
                cnt += 1
        ref_means.append(tot / cnt if cnt else float("nan"))
    ref = np.array(ref_means)
    pct = float((ref < total_mean).mean()) if len(all_hits) else float("nan")
    n_draws = len(dates)
    verdict = ("ODSTUPANJE po dogovorenom pragu" if (total_mean >= 1.0 and n_draws >= 50) else "nema odstupanja po dogovorenom pragu")
    return {"per_system": summary, "total_mean": total_mean, "total_n": len(all_hits), "n_draws": n_draws, "chi2": chi, "p": pval,
            "dof": dof, "ref_mean": float(ref.mean()) if len(ref) else float("nan"), "ref_sd": float(ref.std()) if len(ref) else float("nan"),
            "ref_percentile": pct, "verdict": verdict}


def report(ev: dict, out_csv: Path | None = None) -> str:
    L = [THRESHOLD_TEXT, "",
         f"Kola s rezultatom: {ev['n_draws']}; kombinacija ocijenjeno: {ev['total_n']}; ocekivanje po kombinaciji {EXPECTED:.2f}",
         f"Ukupni prosjek pogodaka po kombinaciji: {ev['total_mean']:.4f}; chi2={ev['chi2']:.3f} (df={ev['dof']}), p={ev['p']:.4f}",
         f"Slucajna referenca (1.000 setova iste strukture): prosjek {ev['ref_mean']:.4f}, SD {ev['ref_sd']:.4f}; "
         f"opazeni prosjek je iznad {ev['ref_percentile']*100:.1f} % slucajnih setova",
         f"Zakljucak: {ev['verdict']}", "", f"{'sustav':26s} {'kola':>5s} {'prosjek':>8s} {'0':>4s} {'1':>4s} {'2':>4s} {'3':>4s} {'4':>4s} {'5':>4s} {'6':>4s} {'chi2':>7s} {'p':>7s}"]
    rows = []
    for s, v in sorted(ev["per_system"].items(), key=lambda kv: -kv[1]["mean"]):
        d = v["dist"]
        L.append(f"{s:26s} {v['n']:5d} {v['mean']:8.4f} {d[0]:4d} {d[1]:4d} {d[2]:4d} {d[3]:4d} {d[4]:4d} {d[5]:4d} {d[6]:4d} {v['chi2']:7.3f} {v['p']:7.4f}")
        rows.append([s, v["n"], f"{v['mean']:.4f}", *d, f"{v['chi2']:.4f}", f"{v['p']:.4f}"])
    if out_csv:
        with open(out_csv, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f, delimiter=";", lineterminator="\n")
            w.writerow(["sustav", "kola", "prosjek", "p0", "p1", "p2", "p3", "p4", "p5", "p6", "chi2", "p"])
            w.writerows(rows)
            w.writerow(["UKUPNO", ev["n_draws"], f"{ev['total_mean']:.4f}", "", "", "", "", "", "", "", f"{ev['chi2']:.4f}", f"{ev['p']:.4f}"])
    return "\n".join(L)


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    res_path = Path(argv[0])
    year_path = Path(argv[1]) if len(argv) > 1 else config.HERE / "loto6_godina.csv"
    results = read_results(res_path)
    rows = read_year(year_path)
    ev = evaluate(results, rows)
    print(report(ev, config.HERE / "score_izvjestaj.csv"))
    return 0


def selftest() -> None:
    tmp = Path("/tmp/claude-0/-home-user-spss/4d826b88-f2ab-56d3-85bd-f957b48b8ad3/scratchpad")
    tmp.mkdir(parents=True, exist_ok=True)
    yp, rp = tmp / "god_test.csv", tmp / "rez_test.csv"
    with open(yp, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f, delimiter=";", lineterminator="\n")
        w.writerow(["datum", "dan", "vrijeme", "sustav", "n1", "n2", "n3", "n4", "n5", "n6", "joker", "izvod", "zamjena", "seed"])
        w.writerow(["01.11.2026.", "NEDJELJA", "20:00", "A", 1, 2, 3, 4, 5, 6, "000000", "", "", 1])
        w.writerow(["01.11.2026.", "NEDJELJA", "20:00", "B", 1, 2, 10, 11, 12, 13, "000000", "", "", 1])
        w.writerow(["04.11.2026.", "SRIJEDA", "20:00", "A", 1, 2, 3, 4, 5, 6, "000000", "", "", 1])
        w.writerow(["04.11.2026.", "SRIJEDA", "20:00", "B", 40, 41, 42, 43, 44, 45, "000000", "", "", 1])
    rp.write_text("﻿datum;n1;n2;n3;n4;n5;n6\n2026-11-01;1;2;3;20;21;22\n04.11.2026.;1;7;8;9;10;45\n", encoding="utf-8")
    ev = evaluate(read_results(rp), read_year(yp), n_random=50)
    assert ev["per_system"]["A"]["hits"] == [3, 1] and ev["per_system"]["B"]["hits"] == [2, 1], ev["per_system"]
    assert abs(ev["total_mean"] - 1.75) < 1e-12 and ev["n_draws"] == 2
    assert "nema odstupanja" in ev["verdict"]
    print(report(ev))
    p = hypergeom_probs()
    assert abs(p.sum() - 1) < 1e-12 and abs(sum(k * p[k] for k in range(7)) - 0.8) < 1e-12
    print("score selftest OK")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--selftest":
        selftest()
    elif len(sys.argv) > 1:
        sys.exit(main(sys.argv[1:]))
    else:
        selftest()
