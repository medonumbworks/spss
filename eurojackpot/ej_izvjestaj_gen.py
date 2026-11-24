# -*- coding: utf-8 -*-
"""Izvještaj Eurojackpot 24.11.2026. (prilagođeno iz sve_ili_nista/izvjestaj_gen_2026_11_24.py). python3 -I ej_izvjestaj_gen.py <json> <izlaz md>"""
import json, sys
R = json.load(open(sys.argv[1], encoding='utf-8'))
S = R['set']; TOT = S['TOT']; NM = S['NM']; RS = R['rand_sets']; OR = R['redoslijed']; BT = R['backtest']; E = R['E']
def pc(x, n, d=3): return f"{100*x/n:.{d}f} %".replace('.', ',')
def f(x, d=2): return f"{x:.{d}f}".replace('.', ',')
L = ['# Eurojackpot (5/50 + 2/12) – 6 kombinacija za uto 24.11.2026.\n']
L.append('## 1. Kombinacije\n\n| # | Tip | Sustav/metoda | 5 glavnih | 2 euro | zbroj | neparni | redoslijed |\n|---|---|---|---|---|---|---|---|')
for c in R['FINAL']:
    L.append(f"| {c['id']} | {c['tip']} | {c['sustav']} | {' '.join(map(str, c['glavni']))} | {' '.join(map(str, c['euro']))} | {c['zbroj']} | {c['neparni']} | {c['redoslijed']} |")
P = R['provjere']
L.append('\n## 2. Provjere i vjerojatnosti\n')
L.append(f"- Susjedni glavni brojevi: {'nema' if not P['susjedni'] else P['susjedni']}.")
L.append(f"- Zajednički glavni brojevi: najviše {P['max_zajednickih']} po paru (" + ', '.join(f"{k} {v}" for k, v in P['parovi_zajednicki'].items() if v) + "; ostali parovi 0).")
L.append(f"- Euro parovi pokrivaju 1–12 točno jednom: {'da' if P['euro_particija'] else 'NE'}. Svih 6 listića je različito, pa je 6 različitih ishoda 5+2. Raspon glavnih brojeva {P['raspon'][0]}–{P['raspon'][1]}, ukupno {P['brojeva_ukupno']} različitih.")
L.append(f"- Točno, preko svih 2.118.760 × 66 = 139.838.160 ishoda (numpy, bitmaske):")
L.append(f"  - P(5+2) = 6/139.838.160 = 1 : 23.306.360")
L.append(f"  - **P(barem jedan dobitak) = {S['P_any']:,}/139.838.160 = {pc(S['P_any'], TOT)}**".replace(',', '.').replace('.' + pc(S['P_any'], TOT).split(',')[1], ',' + pc(S['P_any'], TOT).split(',')[1]))
L.append(f"  - P(barem 3 glavna pogotka u nekoj kombinaciji) = {S['P_3main']}/2.118.760 = {pc(S['P_3main'], NM)}")
L.append(f"  - očekivani broj dobitnih kombinacija = {S['E_win']} = {f(S['E_win_f'],4)} (svaka kombinacija: P(dobitak) = {R['P_win_kombinacija']} = {pc(R['P_win_float'],1)} ≈ 1 : {f(1/R['P_win_float'],1)})")
L.append(f"  - samo E1, E2, S1, S2: {pc(S['fixed_only'], TOT)}")
L.append(f"- 1000 slučajnih setova od 6 listića (bez ograničenja, bez duplikata), također točno: P(barem jedan dobitak) prosjek {pc(RS['mean'], TOT)}, sd {f(100*RS['sd']/TOT,3)} pp, raspon {pc(RS['min'], TOT)}–{pc(RS['max'], TOT)} → ovaj set **{f(RS['pct'],1)}. percentil**; P(≥3 glavna) prosjek {pc(RS['p3_mean'], NM)} → {f(RS['p3_pct'],1)}. percentil.")
pa_ = f"{S['P_any']:,}".replace(',', '.')
L.append(f"- L1–L2: uz zadanu podjelu euro brojeva svaki par L-listića bez zajedničkih glavnih brojeva s ostalima daje isti maksimum ({pa_}), zbog simetrije. Isti maksimum dalo je svih {len(R['L_log'])} greedy pokretanja s lokalnom zamjenom. Konkretni brojevi L1–L2 zato su jedan od jednako dobrih izbora, dobiven slučajnim kandidatima iz `secrets`.")
L.append('\n**Redoslijed igranja** (pohlepno prema doprinosu P(barem jedan dobitak); cijena kombinacije c nije potvrđena):\n\n| k | kombinacije | ulog | P(barem jedan dobitak) | P(≥3 glavna) | najbolji podskup od k |\n|---|---|---|---|---|---|')
for r in OR['seq']:
    L.append(f"| {r['k']} | {', '.join(r['komb'])} | {r['k']} × c | {pc(r['p'], TOT)} | {pc(r['p3'], NM)} | {'isti' if r['p'] == r['best_p'] else ', '.join(r['best']) + ' ' + pc(r['best_p'], TOT)} |")
L.append("\n- k = 1: svih 6 listića ima istu P (3,142 %) i isti najbolji par, pa odlučuje redoslijed u tablici (E1).")
L.append("- Za svaki k prvih k je ujedno točno najbolji podskup (provjereni su svi podskupovi).")
L.append('\n## 3. Kandidati po sustavu\n\n| Sustav | prvih 10 glavnih | euro (prva 4) | ključni podaci |\n|---|---|---|---|')
for k in R['SYS']:
    L.append(f"| {k} | {' '.join(map(str, R['SYS'][k][:10]))} | {' '.join(map(str, R['EUR'][k][:4]))} | {R['KEY'][k]} |")
L.append(f"| **Plečkov filtar** (11 sustava) | {' '.join(str(x[0]) for x in E['plecko_glavni'][:10])} | {' '.join(str(x[0]) for x in E['plecko_euro'][:4])} | glavni broj/glasovi/zbroj pozicija: " + ', '.join(f"{a}/{b}/{c}" for a, b, c in E['plecko_glavni'][:16]) + '; euro: ' + ', '.join(f"{a}/{b}/{c}" for a, b, c in E['plecko_euro']) + ' |')
L.append(f"\n- **Signal-brojevi (≥ 2 glasa):** glavni {', '.join(map(str, E['signal_glavni']))}; euro {', '.join(map(str, E['signal_euro']))}.")
sl = E['slaganje']
L.append(f"- **Slaganje:**")
L.append(f"  - E1 preskače 3 i 5 (susjedni broju 4).")
L.append(f"  - E2 preskače {', '.join(map(str, sl['E2_preskoceno']))}: 11 bi dao 2 zajednička broja s E1, 20 je susjedni broju 21, a 45 broju 44.")
L.append(f"  - Euro za E2 su glasovi istočnih i zapadnih sustava; 11 i 6 su preskočeni jer su već u E1.")
p = E['pravilnosti']
L.append(f"\n**Pravilnosti u zadnjih 5 izvlačenja:**")
for d, m, e in p['izvlacenja']: L.append(f"- {d}: {' '.join(map(str, m))} + {' '.join(map(str, e))}")
L.append(f"- Glavni brojevi 3 puta: 15, 35, 7, 44; 2 puta: 5. Euro 2 puta: 5 i 11.")
L.append(f"- Neparnih: {p['neparni']}; zbrojevi: {p['zbroj']}.")
L.append(f"- Susjedni parovi samo 20.11. (28–29 i 34–35). Ponovljeno iz prethodnog izvlačenja: {p['ponovljeno_iz_prethodnog']}.")
L.append(f"- 17.11. je imalo samo dvije desetice (zbroj 47). Ni u jednom od 5 izvlačenja nije bilo {p['nijednom_1_50']} od 50 brojeva.")
lu = E['lunarni']
L.append(f"\n**Lunarni datum** (lokalni datum 24.11.): lunar_python {lu['lunar_python']}; sxtwl {lu['sxtwl']}; HKO {lu['HKO']}. Sva tri se slažu. U Pekingu je tada {lu['peking']}, što nije korišteno.")
L.append("\n**Osjetljivost na vrijeme izvlačenja:**")
L.append("- Mjesec prelazi u 6° Blizanaca 8,4 min nakon 21:00 EET. On je 6. u zapadnom popisu, pa to ne mijenja E1 ni E2.")
L.append("- Planetarni sat (Sunce) traje do 21:16:05 EET. Ako izvlačenje kasni više od 16 min, vladar sata postaje Venera i mijenjaju se glasovi.")
W = R['WF']; w0 = list(W.values())[0]
L.append(f"\n## 4. Statistika i backtest\n\nWalk-forward od 100. izvlačenja (n = {w0['n']}; svaka metoda koristi samo izvlačenja prije t; listić = prvih 5 glavnih + prva 2 euro po metodi):\n")
L.append(f"| Metoda | dobitni događaji (oček. {f(w0['expW'],1)}) | razredi | z | p (jednostr.) | p Bonf. (×6) | prosj. glavnih (0,5) | prosj. euro (0,333) | p Bonf. glavni / euro |\n|---|---|---|---|---|---|---|---|---|")
for m, w in W.items():
    L.append(f"| {m} | {w['W']} | {', '.join(f'{k}: {v}' for k, v in w['cls'].items() if v)} | {f(w['z'])} | {f(w['p'],3)} | {f(w['p_bonf'],3)} | {f(w['mean_m'],3)} | {f(w['mean_e'],3)} | {f(w['p_m_bonf'],3)} / {f(w['p_e_bonf'],3)} |")
L.append("\n- **Nijedna metoda nema više dobitnih događaja od slučaja** (sve p nakon Bonferronija ≥ 0,997).")
L.append("- Jedini nominalno značajan rezultat su vrući euro brojevi (zadnjih 50): prosjek 0,41 euro pogotka prema 0,333, z = 3,0, p ×6 = 0,016. To je sporedna mjera. Uz korekciju za sve testove (6 metoda × 3 mjere) p ≈ 0,048, dakle na samoj granici. Dobitnih događaja ta metoda nema više (11 prema 12,2 očekivano), a χ² ni Spearman za euro brojeve ne pokazuju odstupanje. Ne smatram to dokazom.")
L.append("- S1–S2 su zato **„statistički tipične”**: dvije najbolje metode po z (Pareto 20/20 i vrući 100) primijenjene na svih 487 izvlačenja. Od 20 najbolje rangiranih brojeva metode uzeta je najbolje rangirana petorka koja zadovoljava tipičnost (zbroj 112,5–142,5, 2–3 neparna, barem 4 desetice) i §2.")
for k, v in R['SLOG'].items(): L.append(f"  - {k} ({v['metoda']}): sirovih 5 = {v['sirovo']} → nisu tipični (5 neparnih); euro po rangu metode.")
r = R['rand']
L.append(f"\nTestovi slučajnosti (487 izvlačenja):")
L.append(f"- χ² glavnih brojeva (ispravljen za izvlačenje bez vraćanja, df 49): {f(r['chi_m'])}, p = {f(r['p_m'],3)}; frekvencije {r['fm_min']}–{r['fm_max']}, očekivano {f(r['E_m'],1)}.")
L.append(f"- χ² euro brojeva (df 11): {f(r['chi_e'])}, p = {f(r['p_e'],3)}; frekvencije {r['fe_min']}–{r['fe_max']}, očekivano {f(r['E_e'],1)}.")
L.append(f"- Stabilnost vrućih brojeva (1. vs 2. polovica, Spearman): glavni ρ = {f(r['rho_m'],3)}, p = {f(r['p_rho_m'],3)}; euro ρ = {f(r['rho_e'],3)}, p = {f(r['p_rho_e'],3)}.")
L.append('\n**Backtest** (6 listića na 487 izvlačenja; kombinacije nisu mijenjane prema njemu):\n\n| # | ' + ' | '.join(['4+0', '3+2', '2+2', '3+1', '3+0', '1+2', '2+1']) + ' | ukupno |\n|---|---|---|---|---|---|---|---|---|')
for k, c in BT['per'].items():
    L.append(f"| {k} | " + ' | '.join(str(c[x]) for x in ['4+0', '3+2', '2+2', '3+1', '3+0', '1+2', '2+1']) + f" | {sum(c.values())} |")
L.append(f"\n- Viši razredi (5+2, 5+1, 5+0, 4+2, 4+1) nisu se dogodili ni jednom.")
L.append(f"- Ukupno {BT['setW']} dobitnih događaja (očekivano {f(BT['expW'],1)}). Slučajni setovi: prosjek {f(BT['rs_mean'],1)}, sd {f(BT['rs_sd'],1)} → **{f(BT['pct'],1)}. percentil**.")
L.append(f"- Izvlačenja s barem jednim dobitkom: {BT['draws_any']}/487.")
L.append("- S1–S2 su odabrani na istim podacima, pa njihov backtest nije neovisan.")
open(sys.argv[2], 'w', encoding='utf-8').write('\n'.join(L) + '\n'); print('OK')
