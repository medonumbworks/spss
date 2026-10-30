"""Glavni izracun: svi datumi kola u razdoblju, 15 sustava po kolu, godisnji fiksni set, konvergencija,
vjerojatnosti egzaktnom enumeracijom, izvod prvog kola. Pokretanje: .venv/bin/python run_year.py [--samo-prvo]"""
from __future__ import annotations
import csv
import datetime as dt
import sys
import time
from pathlib import Path

import config
from config import (DRAW_TIME_LOCAL, HERE, PERIOD_END, PERIOD_START, PRICE_EUR, SYSTEM_NAMES, draw_dates, read_seed,
                    HISTORY_CSV, PLAYED_CSV)
from core import apply_set_rule, build_combo, Combo
import history
import izvod
import systems

OUT_YEAR = HERE / "loto6_godina.csv"
OUT_FIXED = HERE / "loto6_fiksni_set.csv"
OUT_KONV = HERE / "loto6_konvergencija.csv"
OUT_IZVOD = HERE / "izvod_prvo_kolo.txt"
OUT_PROB = HERE / "vjerojatnosti.txt"
OUT_LOG = HERE / "izracun_log.txt"

HEADER = ["datum", "dan", "vrijeme", "sustav", "n1", "n2", "n3", "n4", "n5", "n6", "joker", "izvod", "zamjena", "seed"]


def load_played(path: Path) -> list[set[int]]:
    """odigrano.csv (opcionalno): bilo koji redak sa 6 cijelih brojeva 1..45 (zadnjih 6 brojcanih polja)."""
    if not path.exists():
        return []
    out = []
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            parts = [p.strip() for p in line.replace(",", ";").split(";")]
            nums = [int(p) for p in parts if p.isdigit()]
            nums = nums[-6:]
            if len(nums) == 6 and all(1 <= n <= 45 for n in nums) and len(set(nums)) == 6:
                out.append(set(nums))
    return out


def combo_row(datum: str, dan: str, vrijeme: str, c: Combo, seed: int) -> list:
    n = sorted(c.numbers)
    flags = []
    if c.dopuna:
        flags.append("dopuna iz rezervne kaskade")
    if c.dopuna_krajnja:
        flags.append("krajnja rezerva")
    zam = " || ".join(c.zamjene + flags)
    izv = c.izvod + " || iskoristeno: " + " | ".join(c.used_from)
    return [datum, dan, vrijeme, c.system, *n, c.joker, izv, zam, seed]


def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f, delimiter=";", quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)


def main(argv: list[str]) -> int:
    only_first = "--samo-prvo" in argv
    t0 = time.time()
    seed = read_seed()
    log = []

    def say(s: str = "") -> None:
        print(s)
        log.append(s)

    say(izvod.DISCLAIMER)
    say(f"Seed (secrets.randbits(64), jedan za cijeli posao): {seed}")
    say(f"Razdoblje {PERIOD_START.strftime('%d.%m.%Y.')}-{PERIOD_END.strftime('%d.%m.%Y.')}; dani kola "
        f"{[config.DAY_NAMES_HR[w] for w in config.DRAW_WEEKDAYS]}; vrijeme {DRAW_TIME_LOCAL.strftime('%H:%M')} lokalno; "
        f"cijena {PRICE_EUR:.2f} EUR; dobitne vrste {config.WIN_CLASSES} (PRETPOSTAVKE, vidi README)")

    hist = history.load_history(HISTORY_CSV)
    if hist is None:
        say(history.missing_message())
    else:
        say(history.describe(hist))
    played = load_played(PLAYED_CSV)
    say(f"odigrano.csv: {len(played)} kombinacija za izbjegavanje" if played else "odigrano.csv: nema datoteke (nista za izbjegavanje)")

    natal = systems.compute_natal()
    say(izvod.natal_text(natal))

    dates = draw_dates()
    say(f"Broj kola u razdoblju: {len(dates)}")
    if only_first:
        dates = dates[:1]

    year_rows: list[list] = []
    konv_rows: list[list] = []
    signal_count: dict[int, int] = {}
    first_res = None
    n_sub = 0
    n_dop = 0
    for i, d in enumerate(dates, start=1):
        res = systems.build_draw(d, i, natal, hist, forbidden=played)
        if first_res is None:
            first_res = res
        datum = d.strftime("%d.%m.%Y.")
        dan = res.ctx.day_name
        vrijeme = res.ctx.dt_local.strftime("%H:%M")
        for c in res.combos:
            year_rows.append(combo_row(datum, dan, vrijeme, c, seed))
            n_sub += len(c.zamjene)
            n_dop += int(c.dopuna)
        for n, k, srcs in res.konvergencija:
            konv_rows.append([datum, n, k, ", ".join(srcs)])
        for n in res.signal_numbers:
            signal_count[n] = signal_count.get(n, 0) + 1
        if i % 10 == 0:
            print(f"  ... {i}/{len(dates)} kola ({time.time() - t0:.0f} s)")

    write_csv(OUT_YEAR, HEADER, year_rows)
    write_csv(OUT_KONV, ["datum", "broj", "broj_izvora", "izvori"], konv_rows)
    say(f"Zapisano {OUT_YEAR.name}: {len(year_rows)} redaka ({len(dates)} kola x {len(first_res.combos)} kombinacija); "
        f"zamjena po pravilu seta: {n_sub}; dopuna iz rezervnih kaskada: {n_dop}")
    say(f"Zapisano {OUT_KONV.name}: {len(konv_rows)} redaka")

    # izvod prvog kola
    txt = izvod.natal_text(natal) + "\n\n" + izvod.draw_text(first_res, natal)
    OUT_IZVOD.write_text(txt + "\n", encoding="utf-8")
    say(f"Zapisan {OUT_IZVOD.name}")

    # ---- godisnji fiksni set ----
    hist_freq = history.frequencies(hist["df"]) if hist is not None else None
    fixed: list[Combo] = systems.statistics_combos(hist)
    fixed += [systems.pitagorejska_godisnja(), systems.jyotis_natalna(natal), systems.zapadna_natalna(natal),
              systems.kaldejska_godisnja()]

    def skey(n: int):
        return (-signal_count[n], -(hist_freq or {}).get(n, 0), n)

    ordered = sorted(signal_count, key=skey)
    casc1 = [(f"signal u {signal_count[n]} kola", n) for n in ordered[:6]]
    casc2 = [(f"signal u {signal_count[n]} kola", n) for n in ordered[6:12]]
    rest = [(f"signal u {signal_count[n]} kola", n) for n in ordered[12:]]
    fixed.append(build_combo(15, "Plecko godisnja", casc1, casc2 + rest,
                             f"6 brojeva koji su u najvise kola razdoblja ({len(dates)}) bili signal: " + "; ".join(f"{n}={signal_count[n]}" for _, n in casc1)))
    fixed.append(build_combo(15, "Plecko godisnja 2", casc2, rest,
                             "sljedecih 6 po istom kriteriju: " + "; ".join(f"{n}={signal_count[n]}" for _, n in casc2)))
    apply_set_rule(fixed, forbidden=played)
    per = f"{PERIOD_START.strftime('%d.%m.%Y.')}-{PERIOD_END.strftime('%d.%m.%Y.')}"
    fixed_rows = [combo_row(per, "fiksni set", "", c, seed) for c in fixed]
    write_csv(OUT_FIXED, HEADER, fixed_rows)
    say(f"Zapisano {OUT_FIXED.name}: {len(fixed_rows)} kombinacija")
    say("")
    say("--- Godisnji fiksni set ---")
    for c in fixed:
        say(f"  {c.system:24s} {sorted(c.numbers)} Joker {c.joker}" + (f"  zamjene: {c.zamjene}" if c.zamjene else "") + ("  [dopuna]" if c.dopuna else ""))
    mx = max((len(set(a.numbers) & set(b.numbers)) for i, a in enumerate(fixed) for b in fixed[:i]), default=0)
    say(f"  najvece preklapanje u fiksnom setu: {mx} (dopusteno <= 2)")

    # ---- vjerojatnosti (egzaktno) ----
    say("")
    say("--- Vjerojatnosti (egzaktna enumeracija svih C(45,6) = 8.145.060 izvlacenja) ---")
    import probability
    pt = []
    for name, combos, n_draws in ((f"set prvog kola {first_res.ctx.date.strftime('%d.%m.%Y.')}", first_res.combos, 1),
                                  ("godisnji fiksni set", fixed, len(draw_dates()))):
        pr = probability.set_probabilities([sorted(c.numbers) for c in combos])
        k = len(combos)
        lines = [f"{name}: {k} kombinacija",
                 f"  P(bar jedna >=3 pogotka) = {pr['p_any_ge3']:.6f}  (1 : {1/pr['p_any_ge3']:.1f})",
                 f"  P(bar jedna >=4 pogotka) = {pr['p_any_ge4']:.8f}  (1 : {1/pr['p_any_ge4']:.1f})",
                 f"  P(bar jedna >=5 pogodaka) = {pr['p_any_ge5']:.9f}  (1 : {1/pr['p_any_ge5']:.1f})",
                 f"  P(bar jedna 6 pogodaka)   = {pr['p_any_6']:.10f}  (1 : {1/pr['p_any_6']:.0f})",
                 f"  ocekivani pogoci po kombinaciji = {pr['expected_hits_per_combo']:.6f} (teorijski 6*6/45 = 0.8)",
                 f"  ulog po kolu = {k} x {PRICE_EUR:.2f} = {k*PRICE_EUR:.2f} EUR; godisnje ({len(draw_dates())} kola) = {k*PRICE_EUR*len(draw_dates()):.2f} EUR"]
        for s in lines:
            say(s)
        pt += lines + [""]
    OUT_PROB.write_text("\n".join([izvod.DISCLAIMER, f"seed {seed}", ""] + pt), encoding="utf-8")
    say(f"Zapisan {OUT_PROB.name}; ukupno trajanje {time.time() - t0:.0f} s")
    OUT_LOG.write_text("\n".join(log) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
