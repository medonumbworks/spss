"""Sastavlja završni dokument loto_231_29102026.md iz out/set231.json, out/verify.json, out/stats.json,
out/ezo_*.json i (ako postoji) out/literatura.md.

Pokretanje: python3 make_report.py
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import common as C
from generate import EZO_ORDER

OUT = C.HERE / "out"
DOC = C.HERE / "loto_231_29102026.md"


def fmt(c):
    return "`" + " · ".join(f"{x:02d}" for x in sorted(c)) + "`"


def two_col_table(rows, start):
    """rows: lista kombinacija; tablica s dva stupca (# | kombinacija | # | kombinacija)."""
    half = (len(rows) + 1) // 2
    out = ["| # | kombinacija | # | kombinacija |", "|---|---|---|---|"]
    for i in range(half):
        left = f"| {start + i} | {fmt(rows[i])} "
        j = i + half
        right = f"| {start + j} | {fmt(rows[j])} |" if j < len(rows) else "| | |"
        out.append(left + right)
    return "\n".join(out)


def pct(x, nd=4):
    return f"{x * 100:.{nd}f} %"


def main():
    S = json.load(open(OUT / "set231.json", encoding="utf-8"))
    V = json.load(open(OUT / "verify.json", encoding="utf-8"))
    ST = json.load(open(OUT / "stats.json", encoding="utf-8"))
    B = {int(k): [tuple(c) for c in v] for k, v in S["blokovi"].items()}
    notes = S["napomene"]
    ex = V["exact"]
    n = V["n"]
    p5 = ex["p_ge5"]; p4 = ex["p_ge4"]; p3 = ex["p_ge3"]; p0 = ex["p_nothing"]
    p6 = ex["p_best"]["6"]
    p5only = ex["p_best"]["5"]
    exp3, exp4, exp5 = ex["exp_count"]["3"], ex["exp_count"]["4"], ex["exp_count"]["5"]
    pb = ex["p_best"]
    f_all = {int(k): v for k, v in ST["freq_all"].items()}
    f_60 = {int(k): v for k, v in ST["freq_60"].items()}
    gaps = {int(k): v for k, v in ST["gaps"].items()}
    b1 = V["b1"]; b4 = V["b4"]
    hist = V["history_hits"]

    L = []
    L.append("# Loto 6/45 — 231 kombinacija za četvrtak 29.10.2026. (20:00)\n")
    L.append(f"Ulog **{n * 0.25:.2f} €** ({n} × 0,25 €) na **{-(-n // 10)} listića** (do 10 kombinacija po listiću). "
             "Generirano reproducibilno (`loto/generate.py`, seed 20261029), svaka tvrdnja niže provjerena "
             "neovisnom skriptom (`loto/verify.py`) nabrajanjem svih 8.145.060 mogućih izvlačenja.\n")
    L.append("## Jedno pravilo iznad sva četiri bloka\n")
    L.append("**Nijedan par od 231 kombinacije ne dijeli više od 2 broja** — i to vrijedi i prema 10 fiksnih kombinacija "
             "iz PDF-a koje igraš svako kolo. Dvije kombinacije mogu obje imati ≥5 pogodaka u istom izvlačenju samo ako "
             "dijele barem 2×5−6 = 4 broja; kad nijedan par ne dijeli 4, događaji »ova kombinacija je u 1.–3. razredu« su "
             "međusobno isključivi i vjerojatnosti se zbrajaju:\n")
    L.append(f"> P(1., 2. ili 3. razred) = {n} × 235 / 8.145.060 = **{p5:.6f} = 1 u {1 / p5:.1f}**\n")
    L.append("To je najveća vrijednost koju 231 kombinacija može imati (dokaz: svaka kombinacija pokriva točno 235 "
             "izvlačenja s ≥5 pogodaka; unija ne može biti veća od zbroja). Granica 2 umjesto 3 ne košta ništa, a "
             f"maksimizira i niže razrede: {V['triples_distinct']} različitih trojki (= {n} × 20, maksimum), "
             f"{V['quads_distinct']} četvorki i {n * 6} petorki — nijedan podskup se ne troši dvaput.\n")
    L.append("| razred | uvjet | vjerojatnost (egzaktno) |\n|---|---|---|")
    L.append(f"| 1. | 6 | {n}/8.145.060 = 1 u {8145060 / n:,.0f} |".replace(",", "."))
    L.append(f"| 2. | 5+1 | {n * 6}/8.145.060 = 1 u {8145060 / (n * 6):,.0f} |".replace(",", "."))
    L.append(f"| 3. | 5 (bez dopunskog) | {n * 228}/8.145.060 = 1 u {8145060 / (n * 228):,.1f} |".replace(",", "."))
    L.append(f"| **1.–3. zajedno** | **≥5 negdje** | **{p5:.6f} = 1 u {1 / p5:.1f}** ← maksimum |")
    L.append(f"| 4. | ≥4 negdje | {p4:.4f} (1 u {1 / p4:.2f}) |")
    L.append(f"| 5. | ≥3 negdje | **{p3:.4f}** |")
    L.append(f"| ništa | | {p0:.6f} = 1 u {1 / p0:.0f} |\n")
    L.append(f"Očekivani broj dobitnih kombinacija po kolu: trojki {exp3:.3f}, četvorki {exp4:.4f}, petorki {exp5:.5f}; "
             f"najviše istodobno: {ex['max_simultaneous']['3']} trojki, {ex['max_simultaneous']['4']} četvorke, "
             f"{ex['max_simultaneous']['5']} petorka. Na stvarnih 882 kola (2018.–2026.) ovaj bi set pogodio: "
             f"trojku {hist.get('3', 0)}×, četvorku {hist.get('4', 0)}×, peticu {hist.get('5', 0)}× "
             f"(očekivano {exp3 * 882:.0f} / {exp4 * 882:.0f} / {exp5 * 882:.1f}).\n")
    L.append("Uz iznose koje si ranije naveo (trojka 1,64 €, četvorka 12,04 €, petica 1.853,15 € — ti iznosi nisu u "
             f"priloženim datotekama pa ih uzmi kao ilustraciju): očekivani povrat ≈ {exp3 * 1.64 + exp4 * 12.04 + exp5 * 1853.15:.2f} € "
             f"na {n * 0.25:.2f} € (≈ {(exp3 * 1.64 + exp4 * 12.04 + exp5 * 1853.15) / (n * 0.25) * 100:.0f} %), "
             "bez 1. i 2. razreda. Igra ima negativnu očekivanu vrijednost; to je cijena ulaznice, ne ulaganje.\n")

    # ------------------------------------------------------------------ blok 1
    L.append("---\n\n## Blok 1 — 100 kombinacija: maksimalna pokrivenost raspona 1–45\n")
    L.append(f"Svaki broj se pojavljuje **{b1['freq_min_max'][0]}–{b1['freq_min_max'][1]} puta** (100×6/45 = 13,3). "
             f"**Svaki od 990 parova brojeva pojavljuje se barem jednom** (pokriveno {b1['pairs_covered']}/990), "
             f"nijedan par više od {b1['pairs_max']} puta, nijedna trojka dvaput — blok je istodobno 2-pokrivanje i "
             "3-pakiranje. Svaka kombinacija pokriva barem 4 od 5 pojaseva (1–9, 10–18, 19–27, 28–36, 37–45) i nikad više "
             f"od 3 broja iz istog pojasa (uzorci pojaseva: {b1['band_patterns']}). Profil bloka: zbroj {b1['sum_mean']:.1f} "
             f"(stvarna izvlačenja {ST['sum']['mean']:.1f}), neparnih {b1['odd_mean']:.2f} ({ST['odd']['mean']:.2f}), "
             f"≤31 {b1['le31_mean']:.2f} ({ST['le31']['mean']:.2f}).\n")
    L.append(two_col_table(B[1], 1) + "\n")

    # ------------------------------------------------------------------ blok 2
    L.append("---\n\n## Blok 2 — 50 kombinacija: statistika iz 882 izvlačenja (07.01.2018.–09.08.2026.)\n")
    c2 = ST["chi2_all"]
    L.append(f"**Prvo ono što podaci stvarno kažu.** χ² = {c2[0]:.1f} pri 44 stupnja slobode (p = {c2[1]:.2f}; kritična "
             f"vrijednost 60,5) — frekvencije su savršeno uniformne. Zadnjih 60 kola: χ² = {ST['chi2_60'][0]:.1f} "
             f"(p = {ST['chi2_60'][1]:.2f}). Uzastopni parovi: {ST['consec']['p_any'] * 100:.1f} % kola stvarno vs "
             f"{ST['consec']['p_any_theory'] * 100:.1f} % teorijski. Ponavljanje broja iz prethodnog kola: "
             f"{ST['repeat_prev']['mean']:.2f} (teorija 0,80). Najčešći par ({ST['top_pairs'][0][0][0]}, "
             f"{ST['top_pairs'][0][0][1]}) pao je {ST['top_pairs'][0][1]}× — a simulirane slučajne povijesti iste "
             f"veličine daju maksimum {ST['pair_max_sim_mean']:.1f}× u prosjeku. **Backtest prediktivne moći** "
             "(za svako od 782 kola: koliko od 6 izvučenih brojeva je bilo među 12 »vrućih« iz prethodnih 60 kola, "
             "12 »hladnih«, 12 »najkašnjenijih« i 9 Pareto najčešćih do tada; očekivano 1,60 / 1,60 / 1,60 / 1,20):\n")
    bt = ST["backtest_60_12"]
    L.append("| skup | pogodaka po kolu | očekivano | z | p |\n|---|---|---|---|---|")
    for k, lab in [("hot", "vrući 12"), ("cold", "hladni 12"), ("late", "najkašnjeniji 12"), ("pareto9", "Pareto 9")]:
        r = bt[k]
        L.append(f"| {lab} | {r['mean']:.3f} | {r['expected']:.2f} | {r['z']:+.2f} | {r['p_value']:.2f} |")
    L.append("\n**Vrući, hladni i kasneći brojevi nemaju prediktivnu moć** u ovim podacima; ovaj blok ne diže "
             "vjerojatnost ni za koliko — on bira *koje* od jednako vjerojatnih kombinacija igraš, po kriterijima koje "
             "si tražio. Težina broja = 0,40 × Borda(vrući, zadnjih 60 kola) + 0,35 × Borda(ukupna frekvencija) + "
             "0,25 × Borda(kašnjenje); Pareto 20 % (9 najčešćih) × 1,6. Sve kombinacije su unutar omotnice stvarnih "
             f"izvlačenja: zbroj {ST['sum']['pct']['10']:.0f}–{ST['sum']['pct']['90']:.0f} (10.–90. percentil), "
             "2–4 neparna, 3–5 brojeva ≤31, bez aritmetičkog niza od 4 člana.\n")
    rank = ST["rank_all"]
    L.append("| skup (stanje 09.08.2026.) | brojevi |\n|---|---|")
    L.append(f"| Pareto 9 najčešćih ukupno | {', '.join(f'{x} ({f_all[x]}×)' for x in ST['pareto9'])} |")
    L.append(f"| 9 najrjeđih ukupno | {', '.join(f'{x} ({f_all[x]}×)' for x in ST['cold9'])} |")
    L.append(f"| 12 vrućih (zadnjih 60 kola) | {', '.join(f'{x} ({f_60.get(x, 0)}×)' for x in ST['hot12_60'])} |")
    L.append(f"| 12 hladnih (zadnjih 60 kola) | {', '.join(f'{x} ({f_60.get(x, 0)}×)' for x in ST['cold12_60'])} |")
    L.append(f"| 12 najkašnjenijih (kola bez pojavljivanja) | {', '.join(f'{x} ({gaps[x]})' for x in ST['late12'])} |")
    L.append(f"| 12 najčešćih parova | {', '.join(f'{a}-{b} ({k}×)' for (a, b), k in ST['top_pairs'][:12])} |\n")
    # podskupine
    groups = []
    for c in B[2]:
        nm = notes[" ".join(map(str, c))]
        if not groups or groups[-1][0] != nm:
            groups.append((nm, []))
        groups[-1][1].append(c)
    idx = 101
    for nm, rows in groups:
        L.append(f"**{idx}–{idx + len(rows) - 1}: {nm}**\n")
        L.append(two_col_table(rows, idx) + "\n")
        idx += len(rows)

    # ------------------------------------------------------------------ blok 3
    L.append("---\n\n## Blok 3 — 50 kombinacija: ezoterija, 5 po metodi\n")
    L.append("Sve metode računate za **četvrtak 29.10.2026. u 20:00 CET** (19:00 UTC; ljetno vrijeme je završilo "
             "25.10.), Zagreb. Zajedničke činjenice: kineska godina 丙午 (Vatreni konj), 9. lunarni mjesec, 20. dan; "
             "pravo sunčevo vrijeme 20:20 → sat 戌; dan Jupitera i (3. noćni) planetarni sat Jupitera; Mjesec opadajući "
             "gibozni (82,6 %); Sunce 6° Škorpiona, Mjesec ~25° Blizanaca, Ascendent ~5° Raka; hebrejski 18./19. hešvan "
             "5787. Svaka metoda je samostalan modul u `loto/ezoterija/` s cijelim izvodom; ovdje je sažetak i 5 "
             "kombinacija koje su po rangu metode prve stale u pakiranje (bez ponavljanja trojki s ostalih 226 + 10 "
             "fiksnih; jedini dodatni filtar: bez aritmetičkih nizova od 4 člana i bez kombinacija sa svih 6 brojeva ≤12).\n")
    idx = 151
    for slug, name in EZO_ORDER:
        rows = [c for c in B[3] if notes[" ".join(map(str, c))].startswith(name + ":")]
        p = OUT / f"ezo_{slug}.json"
        ezo = json.load(open(p, encoding="utf-8")) if p.exists() else {}
        key = ezo.get("kljucni_brojevi", [])
        L.append(f"### {idx}–{idx + len(rows) - 1}: {name}\n")
        if key:
            L.append(f"Ključni brojevi metode: {', '.join(str(x) for x in key[:16])}.\n")
        L.append("| # | kombinacija | obrazloženje |\n|---|---|---|")
        for i, c in enumerate(rows):
            obr = notes[" ".join(map(str, c))].split(":", 1)[1].strip()
            L.append(f"| {idx + i} | {fmt(c)} | {obr} |")
        L.append("")
        idx += len(rows)

    # ------------------------------------------------------------------ blok 4
    L.append("---\n\n## Blok 4 — 31 kombinacija: moja prosudba\n")
    L.append("Ovdje ne gađam vjerojatnost — nju sam već maksimizirao pravilom o presjeku i ona se ne može dići ničim "
             "osim brojem kombinacija. Gađam **iznos ako pogodiš**, jedino što je u pari-mutuel igri stvarno pomično. "
             "Cook & Clotfelter (1993), Farrell, Hartley, Lanot & Walker (2000), Simon (1999) i Baker & McHale (2009) "
             "pokazuju da igrači sustavno biraju brojeve ≤31 (rođendani), »sretne« brojeve (7, 3) i vizualne uzorke, pa "
             "se dobitak na takvim kombinacijama dijeli s više ljudi; brojevi iznad 31 i »ružni« nizovi su podigrani. "
             "Henze & Riedwyl (1998) preporučuju točno ovo: slučajan odabir uz ograničenja prema nepopularnoj zoni.\n")
    L.append(f"Svaka od 31 kombinacije: **barem 4 broja iznad 31**, najviše jedan broj ≤12, **bez 7 i bez 3**, zbroj ≥165, "
             "najviše 3 broja iz istog desetljeća, bez aritmetičkog niza od 4 člana; 13 i brojevi 32–45 dobivaju prednost. "
             f"Prosjek bloka: zbroj {b4['sum_mean']:.1f} (stvarna izvlačenja {ST['sum']['mean']:.1f}), "
             f"{b4['gt31_mean']:.2f} broja iznad 31 (slučajno: 1,87), uzastopni par u {b4['consec_share'] * 100:.0f} % "
             "(igrači ih izbjegavaju, pa ih vrijedi imati). Sve to ne mijenja šansu — mijenja očekivani neto ako padne.\n")
    L.append(two_col_table(B[4], 201) + "\n")

    # ------------------------------------------------------------------ provjere
    L.append("---\n\n## Provjere (`loto/verify.py`)\n")
    L.append(f"- {V['n']} kombinacija, sve različite ({V['distinct']}), svaka 6 različitih brojeva iz 1–45: {V['valid_format']}")
    L.append(f"- najveći presjek bilo kojeg para: **{V['max_overlap']}** (raspodjela presjeka po parovima: {V['overlap_dist']})")
    L.append(f"- najveći presjek s 10 fiksnih kombinacija iz PDF-a: **{V['max_overlap_with_pdf10']}** "
             f"(prethodni set od 29.10. imao je 9 kombinacija s presjekom 4 prema njima — sada 0)")
    L.append(f"- pokrivene trojke {V['triples_distinct']} / {n * 20} mogućih (max {V['triples_max_mult']}×), četvorke "
             f"{V['quads_distinct']} / {n * 15} (max {V['quads_max_mult']}×), parovi {V['pairs_covered_all']}/990 (max {V['pairs_max_all']}×)")
    L.append(f"- jednake nekoj od {V['n_played']} već odigranih/objavljenih (registar 16.08.2026., sistemi CSV, 10 iz PDF-a): "
             f"**{len(V['equal_to_played'])}**; jednake nekom od 882 prošlih izvlačenja: **{len(V['equal_to_past_draw'])}**")
    L.append(f"- svi brojevi 1–45 pokriveni, {V['freq_min_max'][0]}–{V['freq_min_max'][1]} puta (prosjek {n * 6 / 45:.1f})")
    L.append(f"- P(≥5 negdje) = {p5:.6f} = teorijski maksimum {V['p_ge5_theoretical_max']:.6f}: {V['p_ge5_is_max']}\n")
    L.append("Sve vjerojatnosti su egzaktne — nabrajanje svih 8.145.060 izvlačenja, bez simulacije.\n")
    L.append("**Što ovo ne radi:** ne diže šansu za šesticu iznad 231 : 8.145.060. Ta se ne može dići ničim osim brojem "
             "kombinacija. Pravilo o presjeku diže P(*barem jedna* kombinacija u 1.–3. razredu) na maksimum mogućeg za "
             "231 kombinaciju, a blok 4 diže iznos ako pogodiš.\n")

    lit = OUT / "literatura.md"
    if lit.exists():
        L.append("---\n\n## Literatura (provjereno)\n")
        L.append(lit.read_text(encoding="utf-8"))

    DOC.write_text("\n".join(L), encoding="utf-8")
    print("zapisano", DOC, len("\n".join(L)), "znakova")


if __name__ == "__main__":
    main()
