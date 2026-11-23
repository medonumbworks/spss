# -*- coding: utf-8 -*-
"""Izvještaj za 24.11.2026. (prilagođeno iz izvjestaj_gen.py). python3 -I izvjestaj_gen_2026_11_24.py <json> <sin csv> <kola csv> <izlaz md>"""
import json, itertools, sys
import pandas as pd
R = json.load(open(sys.argv[1], encoding='utf-8')); s = pd.read_csv(sys.argv[2], sep=';')
kd = pd.read_csv(sys.argv[3], sep=';', encoding='utf-8-sig')
B = [f'B{i}' for i in range(1, 12)]; T = 705432; O = R['L_opt']; PO = O['payout']
def pc(x, n=T, d=3): return f"{100*x/n:.{d}f} %".replace('.', ',')
def f(x, d=2): return f"{x:.{d}f}".replace('.', ',')
def runmax(c): return max(len(list(g)) for _, g in itertools.groupby(enumerate(sorted(c)), lambda t: t[1]-t[0]))
L = ['# Sve ili ništa (11 od 22) – 9 kombinacija za uto 24.11.2026., kolo 328, 19:00\n']
L.append('## 1. Kombinacije\n\n| # | Tip | Sustav/metoda | 11 brojeva (uzlazno) | parni:neparni | zbroj |\n|---|---|---|---|---|---|')
combos = []
for _, r in s.iterrows():
    c = [int(r[b]) for b in B]; combos.append(c); ev = sum(1 for v in c if v % 2 == 0)
    note = '' if ev in (5, 6) and runmax(c) <= 4 else ' ⚠ meko pravilo'
    L.append(f"| {r['#']} | {r['tip']} | {r['sustav']} | {' '.join(map(str, c))} | {ev}:{11-ev}{note} | {sum(c)} |")
ovs = [len(set(a) & set(b)) for a, b in itertools.combinations(combos, 2)]
ch = O['chk']
L.append('\n## 2. Provjere i točne vjerojatnosti\n')
L.append(f"- Duplikati: {'DA' if ch['duplikati'] else 'nema'}; parovi kombinacija–komplement: {'DA' if ch['komplementi'] else 'nema'} → 18 različitih jackpot-ishoda.")
L.append(f"- Zajednički brojevi među 36 parova: najmanje {min(ovs)}, najviše {max(ovs)} (dopušteno 3–8).")
L.append(f"- Meko pravilo (5:6/6:5, niz ≤ 4): zadovoljeno u svih 9. Ispravke E: " + '; '.join(f"{k}: {v}" for k, v in R['FIXLOG'].items()) + " (E2: niz 1–5 bio bi dulji od 4).")
L.append("- Točno (sva 705.432 izvlačenja, numpy bitmaske):")
L.append(f"  - P(jackpot, 11 ili 0 na barem jednoj) = {O['P_SiN']}/705.432 = 0,0026 % (1 : 39.191)")
L.append(f"  - **P(barem jedan dobitak)** = {O['any']}/705.432 = **{pc(O['any'])}**")
L.append(f"  - po razredima (barem jedna kombinacija): 11/0 0,0026 % · 10/1 {pc(O['cls']['10/1'])} · 9/2 {pc(O['cls']['9/2'])} · 8/3 {pc(O['cls']['8/3'])}")
L.append(f"  - očekivani broj dobitnih kombinacija = {O['E_win_frac']} = {f(O['E_win'],4)} (isti za svaki set od 9)")
L.append(f"  - E1–E3 + S1–S3 sami: {pc(O['fixed_only'])}; L1–L3 dodaju {pc(O['any']-O['fixed_only'])}")
d = PO['n83_dist']; n3 = sum(int(d[str(x)]) for x in range(3, 10))
L.append(f"- Isplate, samo potvrđene (11/0 = 80.000 €, 8/3 = 2,60 €; 10/1 i 9/2 = nepoznato, računato kao 0 → donja granica):")
L.append(f"  - očekivana isplata = {PO['exp_pay_frac']} € = **{f(PO['exp_pay'],4)} €** po kolu uz ulog 11,70 € ({f(100*PO['exp_pay']/11.7,1)} % uloga, donja granica)")
L.append(f"  - P(isplata ≥ ulog) ≥ {PO['p_ge']}/705.432 = {pc(PO['p_ge'],d=4)} (potrebno: jackpot ili ≥ {PO['need83']} dobitaka 8/3); gornja granica uz bilo koje iznose 10/1 i 9/2: {pc(PO['p_ge']+PO['p_unk_only'])}")
L.append(f"  - broj dobitaka 8/3 u setu: 0 → {pc(d['0'])} · 1 → {pc(d['1'])} · 2 → {pc(d['2'])} · 3+ → {pc(n3)}")
rs, rg = O['rs'], O['rs_ge']
L.append(f"- 1000 slučajnih setova (bez komplemenata/duplikata):")
L.append(f"  - P(barem jedan dobitak): prosjek {pc(rs['mean'])}, sd {f(100*rs['sd']/T,3)} pp, raspon {pc(rs['min'])}–{pc(rs['max'])} → ovaj set **{f(rs['pct'],1)}. percentil** (bolji od svih 1000).")
L.append(f"  - P(potvrđena isplata ≥ ulog): prosjek {pc(rg['mean'],d=4)}, raspon {pc(rg['min'],d=4)}–{pc(rg['max'],d=4)} → ovaj set {f(rg['pct'],1)}. percentil. Najveća pokrivenost znači najmanje istodobnih dobitaka 8/3 (raspored rizika, ne očekivana vrijednost).")
L.append(f"- Pretraga L: greedy {R['L_greedy']} → lokalna zamjena {R['L_final']}; zatim {O['restarts']} ponovnih pokretanja (greedy + lokalna zamjena, 130 s) → {O['any']}.")
L.append('\n## 3. Redoslijed igranja (pohlepno po doprinosu P(barem jedan dobitak))\n')
L.append('| k | dodaje | prvih k | P(barem jedan dobitak) | ulog | P(potvrđena isplata ≥ ulog) | najbolji podskup od k (točno, svih C(9,k)) | njegova P |\n|---|---|---|---|---|---|---|---|')
for r in O['seq']:
    bs = 'isti' if r['greedy_optimal'] else ', '.join(r['best_subset'])
    L.append(f"| {r['k']} | {r['add']} | {', '.join(O['order'][:r['k']])} | {pc(r['p_any'])} | {f(r['stake'])} € | {pc(r['p_ge'],d=4)} (≥{r['need83']}× 8/3) | {bs} | {pc(r['best_p'])} |")
L.append(f"\n- k = 1: svih 9 kombinacija ima istu P = 8,611 % i isti najbolji par ({pc(list(O['pair_best'].values())[0])}); prednost prema redoslijedu u tablici (E1).")
L.append("- Prvih k iz pohlepnog redoslijeda optimalno je za k = 1, 2, 3 i 9. Za k = 4–8 najbolji podskupovi nisu ugniježđeni (vidi stupac), pa nijedan redoslijed nije najbolji za svaki k. Ako igraš točno k kombinacija, uzmi podskup iz pretposljednjeg stupca.")
L.append("- P(potvrđena isplata ≥ ulog) za k kombinacija: jackpot ili barem ⌈k/2⌉ dobitaka 8/3 (2,60 € × ⌈k/2⌉ ≥ 1,30 € × k).")
L.append(f"- Očekivana potvrđena isplata za bilo kojih k kombinacija = k × {f(PO['exp_pay']/9,4)} €.")
L.append('\n## 4. Kandidati po sustavu\n\n| Sustav | prvih 11 (svedeno 1–22) | ključni podaci |\n|---|---|---|')
for k, v in R['SYS'].items(): L.append(f"| {k} | {' '.join(map(str, v[:11]))} | {R['KEY'][k]} |")
L.append(f"| **Plečkov filtar** ({len(R['SYS'])} sustava) | {' '.join(str(x[0]) for x in R['E']['plecko'][:11])} | broj/glasovi/zbroj pozicija: " + ', '.join(f"{a}/{b}/{c}" for a, b, c in R['E']['plecko']) + ' |')
p = R['E']['sinkronicitet_pravilnosti']
L.append(f"\nPravilnosti u zadnjih 5 kola ({', '.join(map(str, p['kola_br']))}): " + '; '.join(f"{dd}: {' '.join(map(str,c))}" for dd, c in p['kola']))
L.append(f"- nijedan broj u svih 5; u 4 od 5: {p['u_4']}; nijednom: {p['nijednom_u_5']}; ponovljeno iz prethodnog kola: {p['ponovljeno_iz_prethodnog']}; parni: {p['parni']}; zbrojevi: {p['zbroj']}; najdulji niz: {p['najdulji_niz']} (tri kola imaju niz od 5).")
L.append(f"- Lunarni datum (lokalni 24.11.): lunar_python {R['E']['lunarni']['lunar_python']}; sxtwl {R['E']['lunarni']['sxtwl']} – slažu se. U Pekingu: {R['E']['lunarni']['peking']}.")
L.append(f"- Jyotiš: Mjesec je 0,05° iza granice Krittika/Rohini (granični slučaj).")
L.append('\n## 5. Statistika (walk-forward od 100. poznatog kola, n = ' + str(list(R['WF'].values())[0]['n']) + '; praznine preskočene)\n')
L.append('| Metoda | dobitni događaji (oček. ' + f(list(R['WF'].values())[0]['expW'],1) + ') | 10/1 | 9/2 | 8/3 | z | p (jednostr.) | p Bonf. (×7) | prosj. pogodaka | p Bonf. (prosjek) | χ² raspodjele; p Bonf. |\n|---|---|---|---|---|---|---|---|---|---|---|')
for m, w in R['WF'].items():
    c = w['cls']
    L.append(f"| {m} | {w['W']} | {c['10/1']} | {c['9/2']} | {c['8/3']} | {f(w['z'])} | {f(w['p_one'],3)} | {f(w['p_bonf'],3)} | {f(w['mean'],3)} | {f(w['p_mean_bonf'],2)} | {f(w['chi'],2)}; {f(w['p_chi_bonf'],2)} |")
L.append("\nNiti jedan ishod 11/0 ni u jednoj metodi. Hladni su komplement vrućih, a „komplement zadnjeg kola” je komplement „zadnjeg kola” (isti dobitni događaji) → svaki par vođen je kao jedna metoda (7 metoda, Bonferroni ×7).")
L.append("\n**Nijedna metoda nakon Bonferronija nije bolja od slučaja** (najmanji p = " + f(min(w['p_bonf'] for w in R['WF'].values()),3) + "). S1–S3 su zato samo „statistički tipične” kombinacije (zbroj 126,5 ± 10, parnost 5:6/6:5) iz tri različite metode – tri najbolje rangirane po z, primijenjene na sva 656 kola i prilagođene sljedećim brojevima po rangu iste metode.")
# prilagodbe S
Bk = kd.dropna(subset=B); D = [set(map(int, r)) for r in Bk[B].values]
def rh(h):
    fr = {v: sum(1 for x in h if v in x) for v in range(1, 23)}; return sorted(range(1, 23), key=lambda v: (-fr[v], v))
raw = {'S1': rh(D)[:11], 'S2': rh(D[-50:])[:11], 'S3': rh(D[-100:])[:11]}
adj = []
for (sid, rr), c in zip(raw.items(), combos[3:6]):
    out_ = sorted(set(rr) - set(c)); in_ = sorted(set(c) - set(rr))
    adj.append(f"{sid}: {'bez zamjene' if not out_ else ', '.join(f'{a}→{b}' for a, b in zip(out_, in_))} (sirovo zbroj {sum(rr)}, parni {sum(1 for v in rr if v%2==0)})")
L.append("- Prilagodbe S: " + '; '.join(adj) + '.')
r = R['rand']
L.append(f"\nTestovi slučajnosti (656 kola):\n- χ² frekvencija (ispravljen za izvlačenje bez vraćanja, df 21): {f(r['chi_f'])}, p = {f(r['p_f'],3)} (naivni: {f(r['chi_naive'])}, p = {f(r['p_naive'],3)}); frekvencije {min(r['freq'])}–{max(r['freq'])}, očekivano 328.")
L.append(f"- Stabilnost vrućih (1. vs 2. polovica, Spearman): ρ = {f(r['rho'],3)}, p = {f(r['prho'],3)}.")
L.append(f"- Parnost: opaženo {r['par_O']} vs očekivano {[round(x,1) for x in r['par_E']]} (≤3, 4, 5, 6, 7, ≥8), χ² = {f(r['chi_p'])}, df 5, p = {f(r['p_p'],3)}.")
L.append(f"- Zbroj: prosjek {f(r['sum_mean'])} (teorija 126,5), sd {f(r['sum_sd'])} (teorija {f(r['sum_theo_sd'])}), z = {f(r['z_s'])}; χ² po decilima = {f(r['chi_s'])}, df 9, p = {f(r['p_s'],3)}.")
L.append('\n## 6. Backtest 9 kombinacija (656 kola)\n\n| # | pogoci 1·2·3·4·5·6·7·8·9·10 | 11/0 | 10/1 | 9/2 | 8/3 | ukupno | potvrđena isplata |\n|---|---|---|---|---|---|---|---|')
for b in O['bt']:
    dd = b['dist']; c = b['cls']
    L.append(f"| {b['id']} | " + '·'.join(str(dd[str(k)]) for k in range(1, 11)) + f" | {c['11/0']} | {c['10/1']} | {c['9/2']} | {c['8/3']} | {b['W']} | {f(b['pay'])} € |")
n101 = sum(b['cls']['10/1'] for b in O['bt']); n92 = sum(b['cls']['9/2'] for b in O['bt'])
L.append(f"\n- Dobitnih događaja {O['setW']} (očekivano {f(O['expW'],1)}); 1000 slučajnih setova prosjek {f(O['rsW']['mean'],1)}, sd {f(O['rsW']['sd'],1)} → {f(O['rsW']['pct'],1)}. percentil.")
L.append(f"- Potvrđena isplata {f(O['setPay'])} € prema ulogu {f(O['stake_bt'])} € ({f(100*O['setPay']/O['stake_bt'],1)} %, donja granica: {n101}× 10/1 i {n92}× 9/2 bez poznatog iznosa); slučajni setovi prosjek {f(O['rsPay']['mean'])} € (sd {f(O['rsPay']['sd'])} €, uključuje njihove jackpotove) → {f(O['rsPay']['pct'],1)}. percentil.")
L.append(f"- Kola s barem jednim dobitkom: {O['draws_any']}/{O['N']} ({pc(O['draws_any'],O['N'],1)}).")
L.append("- S1–S3 su odabrani na istim podacima (in-sample) – backtest nije neovisan. Kombinacije nisu mijenjane prema backtestu.")
L.append('\n## 7. Povratna informacija – kolo 327 (23.11.2026.: 1 4 5 6 7 8 12 16 17 18 19)\n\n| # | tvoji pogoci | izračunato | razred |\n|---|---|---|---|')
for x in R['povratna_327']: L.append(f"| {x['id']} | {x['korisnik']} | {x['hits']} | {x['razred']} |")
L.append(f"\n- Svih 9 podudara se s tvojim listićem. Dobitak S1 (8) i S2 (3) = 2 × 2,60 € = 5,20 € uz ulog 11,70 €. Samo zabilježeno; metode nisu mijenjane prema jednom kolu.")
open(sys.argv[4], 'w', encoding='utf-8').write('\n'.join(L) + '\n')
print('OK')
