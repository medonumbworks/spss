"""Izvještaj o proširenju portfelja 10 -> 13 listića (kolo 293)."""
import json, sys, os
import os
W = os.environ.get('SIN_PORTFOLIO', os.path.dirname(os.path.abspath(__file__)))
R = json.load(open(f'{W}/portfolio_result.json')); S = json.load(open(f'{W}/select_disjoint.json')); EX = json.load(open(f'{W}/existing10.json'))
WF = json.load(open(f'{W}/workflow_portfolio.json')) if os.path.exists(f'{W}/workflow_portfolio.json') else {}
FIN = json.load(open(f'{W}/final13.json'))
TOTAL = 705432
out = []; p = out.append
def hr(x, d=2): return f'{x:,.{d}f}'.replace(',', 'X').replace('.', ',').replace('X', ' ')
def pct(x, d=4): return hr(100 * x, d) + ' %'
def nums(ns): return ' '.join(f'{n:>2}' for n in ns)
def row(*c): return '| ' + ' | '.join(str(x) for x in c) + ' |'
def sep(n): return row(*['---'] * n)
base, opt = FIN['base'], FIN['optimized']
base['label'] = 'izvornih 10'; opt['label'] = 'optimiziranih 13'
p('# Sve ili Ništa – 3 dodatna listića uz postojećih 10 (kolo 293/2026, utorak 20.10.2026.)\n')
p('## 1. Rezultat\n')
p('Tri nova listića, egzaktno optimalna za portfelj od 13:\n')
for i, (t, ks) in enumerate(zip(FIN['new_tickets'], FIN['overlaps']), 11):
    p(f"{i:>2}. `{nums(t)}`  (zajedničkih brojeva sa svakim od 10 postojećih: {ks})")
p('')
p('Kriterij: maksimizirati vjerojatnost barem jednog dobitka portfelja (primarno), zatim barem jednog dobitka ≥ 260 € (sekundarno). '
  'Postignuta je **teorijska gornja međa**: svaki novi listić dodaje svih svojih 6 294 dobitnih ishoda bez ijednog preklapanja s ostalih 12. '
  'Bolje od toga matematički nije moguće.\n')
p(row('Portfelj', 'Listića', 'P(≥1 dobitak bilo koje kategorije)', 'P(≥1 dobitak ≥ 260 €)', 'P(≥1 glavni dobitak)', 'Očekivana isplata', 'Ulog', 'Očekivani neto'))
p(sep(8))
for e in (base, opt):
    p(row(e['label'], e['n_tickets'], f"{pct(e['p_any_prize'])} ({e['union_all']} od 705 432)", f"{pct(e['p_hi_prize'])} ({e['union_hi']})", f"{pct(e['p_jackpot'], 5)} ({e['union_jp']})",
          f"{hr(e['ev_total_eur'])} €", f"{hr(e['cost_eur'])} €", f"{hr(e['ev_net_eur'])} €"))
d = opt['union_all'] - base['union_all']
p(f"\nDoprinos triju novih listića: +{d} dobitnih ishoda (+{pct(d / TOTAL)} apsolutno, +{hr(100 * d / base['union_all'], 1)} % relativno), "
  f"+{opt['union_hi'] - base['union_hi']} ishoda ≥ 260 €, +{opt['union_jp'] - base['union_jp']} ishoda glavnog dobitka. "
  'Očekivana isplata raste točno linearno (0,5390 € po listiću) i ne ovisi o izboru listića: nijedan izbor ne mijenja očekivanu vrijednost.\n')
p('## 2. Zašto je ovo optimum: dobitna disjunktnost\n')
p('Egzaktnim nabrajanjem svih 705 432 izvlačenja izračunato je u koliko izvlačenja **oba** listića dobivaju, ovisno o broju zajedničkih brojeva k:\n')
p(row('k zajedničkih', *[f['k'] for f in R['f_k']])); p(sep(13))
p(row('oba dobivaju (ishoda)', *[f['joint_prize_draws'] for f in R['f_k']]))
p(row('oba ≥ 260 € (ishoda)', *[f['joint_hi_draws'] for f in R['f_k']]))
p('\nDva listića koja dijele **točno 5 ili 6 brojeva nikad ne dobivaju u istom kolu**, pa im se dobitni skupovi ne preklapaju. '
  'Dokaz: s k = 5 (ili 6) zajedničkih brojeva, da bi oba imala ≤ 2 pogotka izvan tih 11 brojeva mora pasti ≥ 7 izvučenih, a izvan ima ih samo 5 (6); '
  'da bi jedan imao ≥ 9, a drugi ≤ 2, zajednički dio morao bi istodobno sadržavati ≥ 3 i ≤ 2 pogotka. '
  'Komplement (k = 0) i duplikat (k = 11) preklapaju se potpuno, a „skoro-komplementi” (k = 1 ili 10) u 2 244 od 6 294 ishoda.\n')
p(f"Među 352 716 klasa kandidata (listić i komplement imaju isti dobitni skup) točno **{S['n_disjoint_candidates']}** ih ima k ∈ {{5, 6}} prema svih 10 postojećih. "
  'Svaki od njih dodaje točno 6 294 nova ishoda (egzaktno provjereno na uzorku od 300), a svaki drugi kandidat dodaje manje (provjereno na 300). '
  'Tri odabrana listića uzajamno također dijele 5 ili 6 brojeva, pa je svih 18 882 novih ishoda različito.\n')
p('## 3. Postupak pretrage\n')
p('1. **Egzaktan prolaz** kroz svih 352 716 klasa kandidata: za svaku je izračunano preklapanje njezina dobitnog skupa s unijom dobitnih skupova 10 postojećih listića (61 050 ishoda).')
p('2. **Pohlepni odabir** s egzaktnim odbacivanjem (dobitak kandidata ≤ 6 294 − preklapanje s unijom 10), zatim **koordinatni uspon** sa zamjenama; 6 ponovnih pokretanja iz različitih prvih izbora. Svih 6 završilo je na istoj vrijednosti 79 932, što je gornja međa 61 050 + 3 × 6 294.')
p('3. **Cjelobrojno programiranje i Monte Carlo** nisu bili potrebni: problem se egzaktno svodi na uvjet k ∈ {5, 6} prema svim listićima, a taj je uvjet provjeren iscrpno.')
p(f"4. **Presuda izjednačenja:** {S['n_disjoint_candidates']} klasa daje identičnu (maksimalnu) vrijednost svih triju kriterija. Odabrana su prva tri retka dnevnog CSPRNG popisa pokrića za kolo 293 koja zadovoljavaju uvjet prema svih 10 i međusobno (redci {', '.join(str(c['row']) for c in S['chosen'])}; preskočeno {len(S['skipped'])} redaka). Time se izbjegavaju „lijepi” obrasci poput 1 2 3 4 5 6 7 … koje bi dao leksikografski odabir, a izbor ostaje provjerljiv iz zapisane entropije.\n")
p('## 4. Usporedba s drugim načinima dodavanja 3 listića\n')
p(row('Strategija', 'Novi listići', 'P(≥1 dobitak)', 'Ishoda', 'P(≥1 ≥ 260 €)', 'Glavni dobitak (ishoda)', 'Prosj. k prema ostalima')); p(sep(7))
p(row('Samo izvornih 10', '—', pct(base['p_any_prize']), base['union_all'], pct(base['p_hi_prize']), base['union_jp'], hr(base['mean_pairwise_overlap'])))
p(row('**10 + 3 optimizirana (ovaj odabir)**', '<br>'.join(nums(t) for t in opt['tickets'][10:]), pct(opt['p_any_prize']), opt['union_all'], pct(opt['p_hi_prize']), opt['union_jp'], hr(opt['mean_pairwise_overlap'])))
rn = R['random']
p(row(f"10 + 3 slučajna (prosjek {rn['n']} trojki)", 'CSPRNG/PRNG slučajno', pct(rn['union_all_mean'] / TOTAL), hr(rn['union_all_mean'], 0), pct(rn['union_hi_mean'] / TOTAL), 26, '—'))
p(row('10 + 3 slučajna (najbolja od 300)', '—', pct(rn['union_all_max'] / TOTAL), rn['union_all_max'], '—', 26, '—'))
for k, name in (('frequency', '10 + 3 frekvencijska (najčešći ukupno, najčešći zadnjih 30, najdulje „zakašnjeli”)'), ('diversity', '10 + 3 najveće raznolikosti (najveća najmanja udaljenost do listića ili komplementa)'), ('naive_diversity', '10 + 3 naivne raznolikosti (najveća Hammingova udaljenost, bez točnih komplemenata)')):
    e = R[k]; p(row(name, '<br>'.join(nums(t) for t in e['tickets'][10:]), pct(e['p_any_prize']), e['union_all'], pct(e['p_hi_prize']), e['union_jp'], hr(e['mean_pairwise_overlap'])))
p(f"\nOptimizirani odabir bolji je od {pct(rn['frac_random_below_optimized'], 1)} slučajnih trojki i od najbolje slučajne trojke. "
  'Frekvencijski listići (vrući/zakašnjeli brojevi) nemaju prediktivnu vrijednost (odjeljak 6) i uz to se dobitno preklapaju s postojećima. '
  '„Naivna raznolikost” bira skoro-komplemente, koji su najgori mogući izbor: dijele 2 244 od 6 294 dobitnih ishoda s listićem kojemu su skoro-komplement. '
  'Ispravno definirana raznolikost (udaljenost do listića *ili njegova komplementa*) poklapa se s optimumom.\n')
p('## 5. Raspodjela ukupne isplate portfelja (egzaktno, po izvlačenju)\n')
p(row('Ukupna isplata', 'Izvornih 10: ishoda', 'Izvornih 10: vjerojatnost', 'Optimiziranih 13: ishoda', 'Optimiziranih 13: vjerojatnost')); p(sep(5))
bd = {d['payout_eur']: d for d in base['payout_distribution']}; od = {d['payout_eur']: d for d in opt['payout_distribution']}
for v in sorted(set(bd) | set(od)):
    b = bd.get(v, dict(n_draws=0, p=0)); o = od.get(v, dict(n_draws=0, p=0))
    p(row(f'{hr(v, 0)} €', b['n_draws'], pct(b['p'], 5), o['n_draws'], pct(o['p'], 5)))
p(f"\nVjerojatnost dvaju ili više dobitaka u istom kolu: izvornih 10 {pct(base['p_two_or_more_prizes'], 4)}, optimiziranih 13 {pct(opt['p_two_or_more_prizes'], 4)} (nove listiće nikad ne prati dobitak zajedno s nekim drugim listićem).\n")
p('## 6. Ponovna procjena prediktivne moći\n')
if WF.get('diag'):
    dg = WF['diag']
    p(f"**Bayes.** {dg['bayes'].get('summary', '')}\n")
    p(f"**Entropija i slučajnost.** {dg['entropy'].get('summary', '')}\n")
    p(f"**Strojno učenje (kronološka validacija).** {dg['ml'].get('summary', '')}\n")
    if dg.get('other_models'):
        p(row('Model', 'Metrika', 'Vrijednost', 'Slučajna osnova', 'p')); p(sep(5))
        for m in dg['other_models']: p(row(m['name'], m['metric'], hr(m['value'], 4), hr(m['baseline'], 4), hr(m['p_value'], 3) if m.get('p_value') is not None else '—'))
        p('')
    p(f"Zaključak neovisne dijagnostike: prediktivna moć {'pronađena' if dg['predictive_power_found'] else 'nije pronađena'}. {dg.get('overall', '')}\n")
p('Ranije analize (dvije neovisne, šest skeptičnih provjera, dva backtesta, provjera izvan uzorka na 21 novom kolu) ostaju na snazi: frekvencijske anomalije, parovi, trojke, vremenski obrasci i „vrući”/„hladni” brojevi ne preživljavaju korekciju za višestruka testiranja i ne ponavljaju se izvan uzorka. '
  'Izvlačenja su u skladu s neovisnim uniformnim izvlačenjem 11 od 22. Zato optimizacija portfelja ne pokušava pogoditi brojeve nego samo ukloniti preklapanje dobitnih ishoda među listićima, što je jedino što se matematički može poboljšati.\n')
if WF.get('port'):
    pt = WF['port']; b10 = pt['portfolio10']; b3 = pt['best3']
    p('## 7. Neovisna provjera (drugi, neovisno napisani program)\n')
    p(f"- Izvornih 10: unija dobitnih ishoda {b10['union_all']} (ovdje {base['union_all']}), ≥ 260 €: {b10.get('union_hi')} (ovdje {base['union_hi']}), glavni: {b10.get('union_jp')} (ovdje {base['union_jp']}), očekivana isplata {hr(b10['ev_total_eur'], 4)} € (ovdje {hr(base['ev_total_eur'], 4)} €).")
    p(f"- Najbolja 3 dodatna prema neovisnoj pretrazi ({b3['method'][:200]}): unija 13 = {b3['union_all_13']} (ovdje {opt['union_all']}); listići {b3['tickets']}.")
    fk2 = {f['k']: f['joint_prize_draws'] for f in pt['f_k_overlap']}; fk1 = {f['k']: f['joint_prize_draws'] for f in R['f_k']}
    p(f"- Tablica f(k) poklapa se: {fk1 == fk2}.")
    p(f"- Slučajne trojke (neovisno): prosjek {hr(pt.get('random3_union_all_mean', 0), 0)}, maksimum {hr(pt.get('random3_union_all_max', 0), 0)}.")
    p(f"- Napomena provjeravatelja: {pt['notes'][:600]}\n")
p('## 8. Konačna provjera\n')
for c in FIN['checks']: p(f'- {c}')
p('\n## 9. Ograničenja\n')
p('- Iznosi 260 € i 26 € potječu iz sekundarnih izvora i nisu potvrđeni u službenim pravilima; glavni dobitak od 80 000 € potvrđen je u medijima. Ako niže kategorije ne postoje, optimalnost ostaje (disjunktnost vrijedi i za skup {0, 11}), ali vjerojatnosti „bilo kojeg dobitka” treba čitati kao vjerojatnosti glavnog dobitka.')
p('- Portfelj od 13 listića ima 11,33 % šanse za bilo kakav dobitak, ali očekivani neto rezultat je gubitak od oko 9,89 € na ulog od 16,90 €. Optimizacija ne mijenja očekivani povrat, samo uklanja uzaludno dupliranje dobitnih ishoda.')
open(sys.argv[1], 'w').write('\n'.join(out)); print('written', len(out))
