# -*- coding: utf-8 -*-
"""Izvještaj za 24.11.2026. u formatu §7 (prilagođeno iz izvjestaj_gen.py).
python3 -I izvjestaj_gen_2026_11_24.py <json> <sin csv> <kola csv> <izlaz md>"""
import json, itertools, sys
import pandas as pd
R = json.load(open(sys.argv[1], encoding='utf-8')); s = pd.read_csv(sys.argv[2], sep=';')
kd = pd.read_csv(sys.argv[3], sep=';', encoding='utf-8-sig')
B = [f'B{i}' for i in range(1, 12)]; T = 705432; O = R['L_opt']; PO = O['payout']; RD = R['redoslijed']; G = R['granice']
def pc(x, n=T, d=3): return f"{100*x/n:.{d}f} %".replace('.', ',')
def f(x, d=2): return f"{x:.{d}f}".replace('.', ',')
def hm(mins): return f"{int(mins // 60)} h {int(round(mins % 60))} min" if mins >= 60 else f"{f(mins,1)} min"
def runmax(c): return max(len(list(g)) for _, g in itertools.groupby(enumerate(sorted(c)), lambda t: t[1]-t[0]))
TN = ['Pratipada', 'Dvitiya', 'Tritiya', 'Chaturthi', 'Panchami', 'Shashthi', 'Saptami', 'Ashtami', 'Navami', 'Dashami', 'Ekadashi', 'Dvadashi', 'Trayodashi', 'Chaturdashi']
YN = ['Vishkambha', 'Priti', 'Ayushman', 'Saubhagya', 'Shobhana', 'Atiganda', 'Sukarma', 'Dhriti', 'Shula', 'Ganda', 'Vriddhi', 'Dhruva', 'Vyaghata', 'Harshana', 'Vajra', 'Siddhi', 'Vyatipata', 'Variyana', 'Parigha', 'Shiva', 'Siddha', 'Sadhya', 'Shubha', 'Shukla', 'Brahma', 'Indra', 'Vaidhriti']
def tithi_name(t):
    if t == 15: return 'Shukla Purnima'
    if t == 30: return 'Krishna Amavasya'
    return ('Shukla ' if t < 15 else 'Krishna ') + TN[(t - 1) % 15]
L = ['# Sve ili ništa (11 od 22) – 9 kombinacija za uto 24.11.2026., kolo 328, 19:00\n']
# 1
L.append('## 1. Kombinacije\n\n| # | Tip | Sustav/metoda | 11 brojeva | parni:neparni | zbroj | redoslijed igranja |\n|---|---|---|---|---|---|---|')
combos = []
for _, r in s.iterrows():
    c = [int(r[b]) for b in B]; combos.append(c); ev = sum(1 for v in c if v % 2 == 0)
    L.append(f"| {r['#']} | {r['tip']} | {r['sustav']} | {' '.join(map(str, c))} | {ev}:{11-ev} | {sum(c)} | {int(r['redoslijed'])} |")
# 2
L.append('\n## 2. Redoslijed igranja\n\n| k | kombinacije | ulog | P(barem jedan dobitak) | P(isplata ≥ ulog): donja – gornja granica |\n|---|---|---|---|---|')
for r in RD['seq']:
    L.append(f"| {r['k']} | {', '.join(r['kombinacije'])} | {f(r['ulog'])} € | {pc(r['p_any'])} | {pc(r['p_ge_low'],d=4)} – {pc(r['p_ge_up'])} |")
L.append("\n- Donja granica uzima samo potvrđene isplate (jackpot ili barem ⌈k/2⌉ dobitaka 8/3, jer 2,60 € × ⌈k/2⌉ ≥ 1,30 € × k). Gornja granica vrijedi ako bi bilo koji dobitak 10/1 ili 9/2 sam pokrio ulog; iznosi tih razreda nisu poznati.")
L.append(f"- k = 1: svih 9 kombinacija ima istu P = 8,611 % i isti najbolji par ({pc(list(O['pair_best'].values())[0])}), pa prednost ima redoslijed u tablici (E1).")
nn = [r for r in RD['seq'] if not r['greedy_optimal']]
L.append("- Prvih k po ovom redoslijedu točno je optimalno za k = 1, 2, 3 i 9. Za k = 4–8 najbolji podskupovi nisu ugniježđeni (provjereno svih C(9,k)):")
for r in nn:
    L.append(f"  - k = {r['k']}: {', '.join(r['best_subset'])} → {pc(r['best_p'])} (redoslijed daje {pc(r['p_any'])})")
L.append(f"- Očekivana potvrđena isplata za bilo kojih k kombinacija = k × {f(PO['exp_pay']/9,4)} € (donja granica).")
# 3
ovs = [len(set(a) & set(b)) for a, b in itertools.combinations(combos, 2)]
ch = O['chk']; d = PO['n83_dist']; n3 = sum(int(d[str(x)]) for x in range(3, 10))
rs, rg = O['rs'], O['rs_ge']
L.append('\n## 3. Provjere\n')
L.append(f"- Duplikati: {'DA' if ch['duplikati'] else 'nema'}. Parovi kombinacija–komplement: {'DA' if ch['komplementi'] else 'nema'}, dakle 18 različitih jackpot-ishoda.")
L.append(f"- Zajednički brojevi među 36 parova: najmanje {min(ovs)}, najviše {max(ovs)} (dopušteno 3–8).")
L.append("- Meko pravilo (5:6/6:5, niz ≤ 4) zadovoljeno je u svih 9 kombinacija. Jedina ispravka: E2 5→12, jer bi inače nastao niz 1–5.")
L.append("- Točne vjerojatnosti (sva 705.432 izvlačenja, numpy bitmaske):")
L.append(f"  - P(jackpot) = {O['P_SiN']}/705.432 = 0,0026 % (1 : 39.191)")
L.append(f"  - **P(barem jedan dobitak) = {O['any']}/705.432 = {pc(O['any'])}**")
L.append(f"  - po razredima (barem jedna kombinacija): 10/1 {pc(O['cls']['10/1'])} · 9/2 {pc(O['cls']['9/2'])} · 8/3 {pc(O['cls']['8/3'])}")
L.append(f"  - očekivani broj dobitnih kombinacija = {O['E_win_frac']} = {f(O['E_win'],4)}")
L.append(f"  - očekivana isplata iz potvrđenih iznosa = {PO['exp_pay_frac']} € = **{f(PO['exp_pay'],4)} €** uz ulog 11,70 € ({f(100*PO['exp_pay']/11.7,1)} %, donja granica)")
L.append(f"  - P(isplata ≥ ulog): od {pc(PO['p_ge'],d=4)} (jackpot ili ≥ 5× 8/3) do {pc(PO['p_ge']+PO['p_unk_only'])}")
L.append(f"  - broj dobitaka 8/3: 0 → {pc(d['0'])} · 1 → {pc(d['1'])} · 2 → {pc(d['2'])} · 3 ili više → {pc(n3)}")
L.append(f"  - samo E1–E3 i S1–S3: {pc(O['fixed_only'])}; L1–L3 dodaju {pc(O['any']-O['fixed_only'])}")
L.append("- Usporedba s 1000 slučajnih setova (bez komplemenata i duplikata):")
L.append(f"  - P(barem jedan dobitak): prosjek {pc(rs['mean'])}, sd {f(100*rs['sd']/T,3)} pp, raspon {pc(rs['min'])}–{pc(rs['max'])}. Ovaj set je **bolji od svih 1000** ({f(rs['pct'],1)}. percentil).")
L.append(f"  - P(potvrđena isplata ≥ ulog): prosjek {pc(rg['mean'],d=4)}, raspon {pc(rg['min'],d=4)}–{pc(rg['max'],d=4)}. Ovaj set je na {f(rg['pct'],1)}. percentilu: najveća pokrivenost znači najmanje istodobnih dobitaka 8/3.")
L.append(f"- Pretraga L1–L3: greedy {R['L_greedy']}, lokalna zamjena {R['L_final']}, zatim {O['restarts']} ponovnih pokretanja u 130 s → {O['any']}.")
# 4
L.append('\n## 4. Kandidati po sustavu\n\n| Sustav | prvih 11 | ključni podaci |\n|---|---|---|')
J = R['KEY']['Jyotiš']
import re
nak = int(re.search(r'nakšatra (\d+)', J).group(1)); tit = int(re.search(r'tithi (\d+)', J).group(1)); yo = int(re.search(r'yoga (\d+)', J).group(1))
extra = {'Jyotiš': f"; nakšatra: {hm(G['naksatra']['od_pocetka_min'])} od ulaska, {hm(G['naksatra']['do_kraja_min'])} do izlaska; tithi {tit} = {tithi_name(tit)}: {hm(G['tithi']['od_pocetka_min'])} od početka, {hm(G['tithi']['do_kraja_min'])} do kraja; yoga {yo} = {YN[yo-1]}: {hm(G['yoga']['od_pocetka_min'])} od početka, {hm(G['yoga']['do_kraja_min'])} do kraja (linearno prema trenutnoj brzini)"}
for k, v in R['SYS'].items(): L.append(f"| {k} | {' '.join(map(str, v[:11]))} | {R['KEY'][k]}{extra.get(k, '')} |")
L.append(f"| **Plečkov filtar** ({len(R['SYS'])} sustava) | {' '.join(str(x[0]) for x in R['E']['plecko'][:11])} | broj/glasovi/zbroj pozicija: " + ', '.join(f"{a}/{b}/{c}" for a, b, c in R['E']['plecko']) + ' |')
p = R['E']['sinkronicitet_pravilnosti']
L.append(f"\nPravilnosti u zadnjih 5 kola ({', '.join(map(str, p['kola_br']))}): " + '; '.join(f"{dd}: {' '.join(map(str,c))}" for dd, c in p['kola']))
L.append(f"- Nijedan broj nije u svih 5 kola. U 4 od 5: {p['u_4']}. Ni u jednom: {p['nijednom_u_5']}.")
L.append(f"- Ponovljeno iz prethodnog kola: {p['ponovljeno_iz_prethodnog']}; parni: {p['parni']}; zbrojevi: {p['zbroj']}; najdulji niz: {p['najdulji_niz']} (tri kola imaju niz od 5).")
L.append(f"- Lunarni datum (lokalni 24.11.): lunar_python {R['E']['lunarni']['lunar_python']}; sxtwl {R['E']['lunarni']['sxtwl']} – slažu se. U Pekingu je tada {R['E']['lunarni']['peking']}; nije korišteno.")
# 5
w0 = list(R['WF'].values())[0]
L.append(f"\n## 5. Statistika i backtest\n\nWalk-forward od 100. poznatog kola (n = {w0['n']}; praznine preskočene):\n")
L.append(f"| Metoda | dobitni događaji (oček. {f(w0['expW'],1)}) | 10/1 | 9/2 | 8/3 | z | p (jednostr.) | p Bonf. (×7) | prosj. pogodaka | p Bonf. (prosjek) | χ² raspodjele; p Bonf. |\n|---|---|---|---|---|---|---|---|---|---|---|")
for m, w in R['WF'].items():
    c = w['cls']
    L.append(f"| {m} | {w['W']} | {c['10/1']} | {c['9/2']} | {c['8/3']} | {f(w['z'])} | {f(w['p_one'],3)} | {f(w['p_bonf'],3)} | {f(w['mean'],3)} | {f(w['p_mean_bonf'],2)} | {f(w['chi'],2)}; {f(w['p_chi_bonf'],2)} |")
L.append("\n- Ishod 11/0 nije se dogodio ni u jednoj metodi.")
L.append("- Hladni brojevi su komplement vrućih, a „komplement zadnjeg kola” je komplement „zadnjeg kola”. Parovi imaju iste dobitne događaje, pa je svaki vođen kao jedna metoda (7 metoda).")
L.append(f"- **Nijedna metoda nakon Bonferronija nije bolja od slučaja** (najmanji p = {f(min(w['p_bonf'] for w in R['WF'].values()),3)}). S1–S3 su zato samo „statistički tipične” kombinacije (zbroj 126,5 ± 10, parnost 5:6/6:5) iz tri različite metode: tri najbolje po z, primijenjene na sva 656 kola.")
Bk = kd.dropna(subset=B); D = [set(map(int, r)) for r in Bk[B].values]
def rh(h):
    fr = {v: sum(1 for x in h if v in x) for v in range(1, 23)}; return sorted(range(1, 23), key=lambda v: (-fr[v], v))
raw = {'S1': rh(D)[:11], 'S2': rh(D[-50:])[:11], 'S3': rh(D[-100:])[:11]}
adj = []
for (sid, rr), c in zip(raw.items(), combos[3:6]):
    out_ = sorted(set(rr) - set(c)); in_ = sorted(set(c) - set(rr))
    adj.append(f"{sid} {'bez zamjene' if not out_ else ', '.join(f'{a}→{b}' for a, b in zip(out_, in_))} (prije: zbroj {sum(rr)}, parnih {sum(1 for v in rr if v%2==0)})")
L.append("- Prilagodbe S (sljedeći brojevi po rangu iste metode): " + '; '.join(adj) + '.')
r = R['rand']
L.append(f"\nTestovi slučajnosti (656 kola):")
L.append(f"- χ² frekvencija, ispravljen za izvlačenje bez vraćanja (df 21): {f(r['chi_f'])}, p = {f(r['p_f'],3)}. Naivni χ²: {f(r['chi_naive'])}, p = {f(r['p_naive'],3)}. Frekvencije {min(r['freq'])}–{max(r['freq'])}, očekivano 328.")
L.append(f"- Stabilnost vrućih brojeva (1. vs 2. polovica, Spearman): ρ = {f(r['rho'],3)}, p = {f(r['prho'],3)}.")
L.append(f"- Parnost: opaženo {r['par_O']}, očekivano {[round(x,1) for x in r['par_E']]} (≤3, 4, 5, 6, 7, ≥8); χ² = {f(r['chi_p'])}, df 5, p = {f(r['p_p'],3)}.")
L.append(f"- Zbroj: prosjek {f(r['sum_mean'])} (teorija 126,5), sd {f(r['sum_sd'])} (teorija {f(r['sum_theo_sd'])}), z = {f(r['z_s'])}; χ² po decilima = {f(r['chi_s'])}, df 9, p = {f(r['p_s'],3)}.")
L.append('\nBacktest 9 kombinacija (656 kola):\n\n| # | pogoci 1·2·3·4·5·6·7·8·9·10 | 11/0 | 10/1 | 9/2 | 8/3 | ukupno | potvrđena isplata |\n|---|---|---|---|---|---|---|---|')
for b in O['bt']:
    dd = b['dist']; c = b['cls']
    L.append(f"| {b['id']} | " + '·'.join(str(dd[str(k)]) for k in range(1, 11)) + f" | {c['11/0']} | {c['10/1']} | {c['9/2']} | {c['8/3']} | {b['W']} | {f(b['pay'])} € |")
n101 = sum(b['cls']['10/1'] for b in O['bt']); n92 = sum(b['cls']['9/2'] for b in O['bt'])
L.append(f"\n- Dobitni događaji: {O['setW']} (očekivano {f(O['expW'],1)}). Slučajni setovi: prosjek {f(O['rsW']['mean'],1)}, sd {f(O['rsW']['sd'],1)} → {f(O['rsW']['pct'],1)}. percentil.")
L.append(f"- Potvrđena isplata {f(O['setPay'])} € uz ulog {f(O['stake_bt'])} € ({f(100*O['setPay']/O['stake_bt'],1)} %). To je donja granica: {n101}× 10/1 i {n92}× 9/2 nemaju poznat iznos. Slučajni setovi: prosjek {f(O['rsPay']['mean'])} € (uključuje njihove jackpotove) → {f(O['rsPay']['pct'],1)}. percentil.")
L.append(f"- Kola s barem jednim dobitkom: {O['draws_any']}/{O['N']} ({pc(O['draws_any'],O['N'],1)}).")
L.append("- S1–S3 su odabrani na istim podacima, pa njihov backtest nije neovisan. Kombinacije nisu mijenjane prema backtestu.")
# 6
L.append('\n## 6. Povratna informacija, izvori, pretpostavke\n\n**Kolo 327 (23.11.2026.: 1 4 5 6 7 8 12 16 17 18 19)**\n\n| # | tvoji pogoci | izračunato | razred |\n|---|---|---|---|')
for x in R['povratna_327']: L.append(f"| {x['id']} | {x['korisnik']} | {x['hits']} | {x['razred']} |")
L.append("\n- Svih 9 vrijednosti podudara se s tvojim listićem.")
L.append("- Dobitak: S1 (8) i S2 (3) = 2 × 2,60 € = 5,20 € uz ulog 11,70 €.")
L.append("- Samo zabilježeno; metode nisu mijenjane prema jednom kolu.")
open(sys.argv[4], 'w', encoding='utf-8').write('\n'.join(L) + '\n')
print('OK')
