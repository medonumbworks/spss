import json, itertools, sys
import pandas as pd
R = json.load(open(sys.argv[1], encoding='utf-8')); s = pd.read_csv(sys.argv[2], sep=';')
B = [f'B{i}' for i in range(1, 12)]; T = 705432; O = R['L_opt']
def pc(x, n=T, d=3): return f"{100*x/n:.{d}f} %".replace('.', ',')
def f(x, d=2): return f"{x:.{d}f}".replace('.', ',')
def runmax(c): return max(len(list(g)) for _, g in itertools.groupby(enumerate(sorted(c)), lambda t: t[1]-t[0]))
L = []
L.append('## 1. Kombinacije\n\n| # | Tip | Sustav/metoda | 11 brojeva (uzlazno) | parni:neparni | zbroj |\n|---|---|---|---|---|---|')
combos = []
for _, r in s.iterrows():
    c = [int(r[b]) for b in B]; combos.append(c); ev = sum(1 for v in c if v % 2 == 0)
    note = '' if ev in (5, 6) and runmax(c) <= 4 else ' ⚠ meko pravilo'
    L.append(f"| {r['#']} | {r['tip']} | {r['sustav']} | {' '.join(map(str, c))} | {ev}:{11-ev}{note} | {sum(c)} |")
ovs = [len(set(a) & set(b)) for a, b in itertools.combinations(combos, 2)]
L.append('\n## 2. Provjere\n')
ch = O['chk']
L.append(f"- Duplikati: {'DA' if ch['duplikati'] else 'nema'}; parovi kombinacija–komplement: {'DA' if ch['komplementi'] else 'nema'}.")
L.append(f"- Zajednički brojevi među 36 parova: najmanje {min(ovs)}, najviše {max(ovs)} (dopušteno 3–8).")
L.append("- Meko pravilo (5:6/6:5, niz ≤ 4): zadovoljeno osim E3 (3:8) – jedna zamjena s Plečkova popisa ne može ispraviti parnost za 2; tvrda pravila vrijede.")
L.append("- Točno (enumeracija svih 705.432 izvlačenja, numpy bitmaske):")
L.append(f"  - P(Sve ili Ništa, 11 ili 0 na barem jednoj) = {O['P_SiN']}/{T} = {pc(O['P_SiN'],d=4)} (1 : {round(T/O['P_SiN']):,})".replace(',', '.').replace('0.0026 %', '0,0026 %'))
L.append(f"  - P(barem jedan dobitak, razredi 11/0, 10/1, 9/2, 8/3) = {O['any']}/{T} = {pc(O['any'])}")
L.append(f"  - po razredima (barem jedna kombinacija u razredu): 11/0 {pc(O['cls']['11/0'],d=4)}, 10/1 {pc(O['cls']['10/1'])}, 9/2 {pc(O['cls']['9/2'])}, 8/3 {pc(O['cls']['8/3'])}")
L.append(f"  - očekivani broj dobitnih kombinacija = 9 × 60.744/705.432 = {f(O['E_win'],4)} (isto za svaki set od 9)")
L.append(f"  - samo E1–E3 + S1–S3: {pc(O['fixed_only'])}; L1–L3 dodaju {pc(O['any']-O['fixed_only'])}")
rs = O['rs']
L.append(f"- 1000 slučajnih setova (bez komplemenata/duplikata): P(barem jedan) prosjek {pc(rs['mean'])}, sd {f(100*rs['sd']/T,3)} pp, raspon {pc(rs['min'])}–{pc(rs['max'])}; ovaj set je na {f(rs['pct'],1)}. percentilu.")
L.append(f"- Pretraga L: prvi prolaz greedy {R['L_greedy']} → lokalna zamjena {R['L_final']}; zatim {O['restarts']} novih greedy pokretanja (800 kandidata/poziciji) + lokalna zamjena u 150 s, najbolje {O['any']}.")
L.append('\n## 3. Kandidati po sustavu\n\n| Sustav | prvih 11 (svedeno 1–22) | ključni podaci |\n|---|---|---|')
for k, v in R['SYS'].items(): L.append(f"| {k} | {' '.join(map(str, v[:11]))} | {R['KEY'][k]} |")
L.append(f"| Kamea Mjeseca | – | {R['KEY']['Kamea Mjeseca']} |")
L.append(f"| **Plečkov filtar** | {' '.join(str(x[0]) for x in R['E']['plecko'][:11])} | puni poredak (broj/glasovi/zbroj pozicija): " + ', '.join(f"{a}/{b}/{c}" for a, b, c in R['E']['plecko']) + ' |')
p = R['E']['sinkronicitet_pravilnosti']
L.append('\nPravilnosti u zadnjih 5 kola (24.–28.09.2026.): ' + '; '.join(f"{d}: {' '.join(map(str,c))}" for d, c in p['kola']))
L.append(f"- broj 1 u svih 5; 12, 14, 17, 22 u 4 od 5; nema broja koji se nije pojavio; ponovljeno iz prethodnog kola: {p['ponovljeno_iz_prethodnog']}; parni: {p['parni']}; zbrojevi: {p['zbroj']}; najdulji niz: {p['najdulji_niz']}.")
L.append(f"- Lunarni datum: lunar_python {R['E']['lunarni']['lunar_python']}; sxtwl {R['E']['lunarni']['sxtwl']} – slažu se.")
L.append('\n## 4. Statistika (walk-forward, kola 100–635, n = 536)\n')
L.append('Očekivano (n·P(k)): ' + ', '.join(f"{k}:{f(v,1)}" for k, v in R['expected_dist'].items() if int(k) in range(1, 11)))
L.append('\n| Metoda | dobitni događaji (oček. 46,2) | 11/0 | 10/1 | 9/2 | 8/3 | z | p (jednostrano) | p Bonf. (×9) | prosj. pogodaka | p Bonf. (prosjek) | χ² raspodjele, p Bonf. |\n|---|---|---|---|---|---|---|---|---|---|---|---|')
for m, w in R['WF'].items():
    c = w['cls']
    L.append(f"| {m} | {w['W']} | {c['11/0']} | {c['10/1']} | {c['9/2']} | {c['8/3']} | {f(w['z'])} | {f(w['p_one'],3)} | {f(w['p_bonf'],3)} | {f(w['mean'],3)} | {f(w['p_mean_bonf'],2)} | {f(w['chi'],2)}; {f(w['p_chi_bonf'],2)} |")
L.append("\n**Nijedna metoda nije značajno bolja od slučaja** (sve p nakon Bonferronija ≥ 0,91). S1–S3 su zato samo „statistički tipične” kombinacije: tri najbolje rangirane metode (po z dobitnih događaja; vrući/cijela povijest i hladni/cijela povijest su komplementarne pa imaju identičan rezultat) primijenjene na svih 635 kola, uz prilagodbu na zbroj 116,5–136,5, parnost 5:6/6:5 i pravila §2.")
r = R['rand']
L.append(f"\nTestovi slučajnosti (635 kola):\n- χ² frekvencija 22 broja (ispravljen za izvlačenje bez vraćanja, df 21): {f(r['chi_f'])}, p = {f(r['p_f'],3)} (naivni χ²: {f(r['chi_naive'])}, p = {f(r['p_naive'],3)}). Raspon frekvencija {min(r['freq'])}–{max(r['freq'])}, očekivano 317,5.")
L.append(f"- Stabilnost vrućih brojeva (frekvencije 1. vs 2. polovice, Spearman): ρ = {f(r['rho'],3)}, p = {f(r['prho'],3)} – nema stabilnosti.")
L.append(f"- Parnost (broj parnih, hipergeometrijska): opaženo {r['par_O']} vs očekivano {[round(x,1) for x in r['par_E']]} (≤3, 4, 5, 6, 7, ≥8), χ² = {f(r['chi_p'])}, df 5, p = {f(r['p_p'],3)}.")
L.append(f"- Zbroj: prosjek {f(r['sum_mean'])} (teorija 126,5), sd {f(r['sum_sd'])} (teorija {f(r['sum_theo_sd'])}), z = {f(r['z_s'])}; χ² po decilima točne raspodjele = {f(r['chi_s'])}, df 9, p = {f(r['p_s'],3)}.")
L.append('\n## 5. Backtest 9 kombinacija (635 kola)\n\n| # | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 11/0 | 10/1 | 9/2 | 8/3 | ukupno |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
for b in O['bt']:
    d = b['dist']; c = b['cls']
    L.append(f"| {b['id']} | " + ' | '.join(str(d[str(k)] if str(k) in d else d[k]) for k in range(12)) + f" | {c['11/0']} | {c['10/1']} | {c['9/2']} | {c['8/3']} | {b['W']} |")
L.append(f"\n- Ukupno dobitnih događaja: {O['setW']} (očekivano {f(O['expW'],1)}); 1000 slučajnih setova: prosjek {f(O['rsW']['mean'],1)}, sd {f(O['rsW']['sd'],1)} → {f(O['rsW']['pct'],1)}. percentil.")
L.append(f"- Kola s barem jednim dobitkom: {O['draws_any']}/{O['N']} ({pc(O['draws_any'],O['N'],1)}); 200 slučajnih setova prosjek {f(O['draws_any_rand200']['mean'],1)} → {f(O['draws_any_rand200']['pct'],1)}. percentil.")
L.append("- Povrat prema ulogu: **nije izračunato** (isplate nisu potvrđene). Ulog za usporedbu: 635 kola × 11,70 € = 7.429,50 €.")
L.append("- S1–S3 su odabrani i prilagođeni na istim podacima (in-sample), pa njihov backtest nije neovisan. Kombinacije nisu mijenjane prema backtestu.")
open(sys.argv[3], 'w', encoding='utf-8').write('\n'.join(L) + '\n')
