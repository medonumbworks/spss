"""Dnevni izvještaj (hrvatski) za zadano kolo. Upotreba: python3 render_daily.py config.json out.md"""
import json, csv, sys, math
from datetime import datetime
CFG = json.load(open(sys.argv[1]))
W = CFG['work']; KOLO = CFG['kolo']; TD = CFG['target_date']; TDAY = CFG['target_day_acc']; TDAYN = CFG['target_day_nom']
A = json.load(open(CFG['analysis'])); C = json.load(open(CFG['combos']))
R = json.load(open(f'{W}/workflow.json')); O = json.load(open(CFG['oos'])); X = json.load(open(CFG['extra']))
ENT = open(CFG['entropy']).read().strip()
MISSING = CFG.get('missing', [])
LASTD = datetime.strptime(A['last'], '%Y-%m-%d').strftime('%d.%m.%Y')
KN = O['k_new']; NF, NL = O['new_first'], O['new_last']
import sys as _s; _s.path.insert(0, W); import sin_lib as _L
_draws = _L.load_draws(CFG['data']); _seen = {}; REPEATS = []
for _d in _draws:
    _k = tuple(_d['nums'])
    if _k in _seen: REPEATS.append((_seen[_k], _d['datum']))
    else: _seen[_k] = _d['datum']
TOP = []
with open(CFG['coverage_csv'], encoding='utf-8-sig') as f:
    rd = csv.reader(f, delimiter=';'); next(rd)
    for i, r in enumerate(rd):
        if i >= 20: break
        TOP.append([int(v) for v in r[1:12]])
TOTAL = 352716; PRICE = 1.30; EVJ = 80000 * 2 / 705432; EVALL = sum(t['p'] * t['iznos'] for t in A['game_math']['tiers'].values())
out = []; p = out.append
def hr(x, d=2):
    s = f'{x:,.{d}f}'
    return s.replace(',', 'X').replace('.', ',').replace('X', ' ')
def puta(n): return f'{n} put' if n % 10 == 1 and n % 100 != 11 else f'{n} puta'
def nums(ns): return ' '.join(f'{n:>2}' for n in ns)
def row(*c): return '| ' + ' | '.join(str(x) for x in c) + ' |'
def sep(n): return row(*['---'] * n)
comp = lambda c: [n for n in range(1, 23) if n not in c]

p(f'# Sve ili Ništa – kombinacije za {TDAY} {TD}. (kolo {KOLO}/2026)\n')
p('> **Kratki odgovor.** Ne postoji kombinacija s 30 %, 50 % ili 99 % šanse. Svaki listić u ovoj igri ima točno istu vjerojatnost glavnog dobitka, '
  f'**2 od 705 432 = 1 : 352 716 (0,00028 %)**, bez obzira na to kako su brojevi odabrani. Analizirana su {A["n_draws"]} izvlačenja: preko 40 statističkih testova, '
  f'šest skeptičnih provjera, dva neovisna backtesta i provjera izvan uzorka na {KN} novih kola. Nijedan obrazac ne predviđa iduće izvlačenje. '
  'Šansa raste **samo s brojem različitih listića**, linearno: N listića = N / 352 716. Zato su kombinacije ispod poredane kao popis pokrića: '
  'igrate li prvih N redaka, vaša je šansa točno ona u stupcu „Kumulativna šansa”.\n')
p(f'Izvlačenje: {TDAYN} {TD}. u 19:00, uplata do 18:55. Cijena kombinacije: 1,30 €. Glavni dobitak: 80 000 € (11/11 ili 0/11).\n')

PV = CFG.get('prev')
if PV:
    _pn = set(next(e['numbers'] for e in X if e['kolo'] == PV['kolo']))
    _rows = []
    with open(PV['coverage_csv'], encoding='utf-8-sig') as _f:
        _r = csv.reader(_f, delimiter=';'); next(_r)
        for _x in _r: _rows.append([int(v) for v in _x[1:12]])
    _h = [len(_pn & set(c)) for c in _rows]
    _prize = {0: '80 000 €', 11: '80 000 €', 1: '260 €*', 10: '260 €*', 2: '26 €*', 9: '26 €*'}
    _jr = next(i + 1 for i, h in enumerate(_h) if h in (0, 11))
    p(f"## 0. Kako je prošlo kolo {PV['kolo']}\n")
    p(f"Izvučeno je `{nums(sorted(_pn))}`. Pogoci prvih 20 redaka jučerašnjeg popisa:\n")
    p(row('Red', *range(1, 21))); p(sep(21)); p(row('Pogodaka', *_h[:20]))
    _w = [(i + 1, h) for i, h in enumerate(_h[:1000]) if h in _prize]
    p('')
    for _N in (20, 100, 1000):
        _ws = [x for x in _w if x[0] <= _N]
        _pg = lambda h: 'pogodak' if h % 10 == 1 and h != 11 else ('pogotka' if h in (2, 3, 4) else 'pogodaka')
        p(f"- Prvih {_N} redaka: " + (', '.join(f"red {i} ({h} {_pg(h)}, {_prize[h]})" for i, h in _ws[:6]) + (' …' if len(_ws) > 6 else '') if _ws else 'bez dobitka') + '.')
    p(f"- Glavni dobitak (0 ili 11 pogodaka) bio je u retku {_jr:,}".replace(',', ' ') + f" popisa, uz kumulativni ulog {hr(_jr * PRICE)} €.")
    p('- \\* Iznosi 260 € i 26 € navedeni su prema neslužbenim izvorima.\n')
p(f'## 1. Kombinacije za kolo {KOLO}\n')
p('Generirane kriptografski sigurnim generatorom (CSPRNG): 32 bajta entropije iz `os.urandom` (Python `secrets`), proširena SHA-256 DRBG-om bez pristranosti. '
  'Zapisana entropija omogućuje svakome da reproducira i provjeri cijeli popis:\n')
p(f'```\npython3 coverage.py kombinacije.csv --entropy {ENT}\n```\n')
p('Popis sadrži svih 352 716 kombinacija tako da je svaki od 705 432 mogućih ishoda pokriven točno jednom (kombinacija ili njezin komplement, nikad oboje), '
  'u nasumičnom redoslijedu. Prvih 20 redaka:\n')
p(row('Red', 'Kombinacija', 'Zbroj', 'Parni', 'Kumulativna šansa', 'Kumulativni ulog')); p(sep(6))
for i, c in enumerate(TOP, 1):
    p(row(i, f'`{nums(c)}`', sum(c), sum(1 for n in c if n % 2 == 0), f'{hr(100 * i / TOTAL, 5)} %', f'{hr(i * PRICE)} €'))
p('\nCijeli popis je u prilogu (CSV, 352 716 redaka, stupci kumulativne šanse i uloga).\n')

p('### 1.1 Koliko listića za koju šansu\n')
p(row('Listića (prvih N redaka)', 'Šansa glavnog dobitka', 'Ulog', 'Očekivani povrat, samo glavni dobitak', 'Očekivani povrat, sve kategorije*', 'Očekivani gubitak*')); p(sep(6))
for N in [1, 5, 11, 20, 100, 1000, 10000, 35272, 105815, 176358, 246902, 317445, 349189, 352716]:
    p(row(f'{N:,}'.replace(',', ' '), f'{hr(100 * N / TOTAL, 4 if N < 10000 else 2)} %', f'{hr(N * PRICE)} €', f'{hr(N * EVJ)} €', f'{hr(N * EVALL)} €', f'{hr(N * PRICE - N * EVALL)} €'))
p(f'\n\\* Kategorije 260 € (10 ili 1 pogodak) i 26 € (9 ili 2 pogotka) navode samo sekundarni izvori. Nisu potvrđene u pravilima Hrvatske Lutrije. '
  f'Na svakoj razini očekivani povrat je {hr(100 * EVJ / PRICE, 1)} % uloga samo s glavnim dobitkom, odnosno {hr(100 * EVALL / PRICE, 1)} % sa svim kategorijama. '
  'Više listića povećava šansu, ali i očekivani gubitak u istom omjeru.\n')

p('### 1.2 Varijante na temelju povijesti (za usporedbu)\n')
p('Ove su kombinacije također izvučene CSPRNG-om, ali s težinama prema povijesti. Imaju **istu** šansu 1 : 352 716 kao bilo koji redak gore. '
  'Navedene su samo zato što ste tražili odabir „baziran na podacima”.\n')
p(row('Oznaka', 'Težina', 'Kombinacija', 'Zbroj', 'Parni')); p(sep(5))
names = {'csprng-weighted-hot60': 'vrući brojevi, zadnjih 60 kola', 'csprng-weighted-hot30': 'vrući brojevi, zadnjih 30 kola', 'csprng-weighted-overdue': '„zakašnjeli” brojevi',
         'csprng-weighted-cold': 'ukupno najrjeđi brojevi', 'csprng-typical-structure': 'zbroj 116–137 i 5–6 parnih (središnjih 50 %)'}
topset = {tuple(c) for c in TOP} | {tuple(comp(c)) for c in TOP}
for c in C['combos']:
    flag = ' (već u prvih 20 ili komplement)' if tuple(c['numbers']) in topset else ''
    p(row(c['name'], names.get(c['strategy'], c['strategy']), f"`{nums(c['numbers'])}`{flag}", c['zbroj'], c['parni']))
p('\n**Pravila koja stvarno štede novac:**')
p('- Nikad ne igrajte kombinaciju i njezin komplement (brojeve koji u njoj nisu). Imaju identične ishode u svim kategorijama, pa plaćate dvaput za istu šansu.')
p('- Nikad ne igrajte istu kombinaciju dvaput. Popis u prilogu jamči oboje.')
p('- „Vrući”, „hladni” i „zakašnjeli” brojevi ne mijenjaju šansu (odjeljci 4 i 5).\n')

g = A['game_math']
p('## 2. Matematika igre\n')
p(row('Kategorija', 'Ishoda od 705 432', 'Šansa po listiću', 'Iznos', 'Očekivana vrijednost')); p(sep(5))
for k, t in g['tiers'].items():
    lab = {'11_ili_0': '11 ili 0 pogodaka', '10_ili_1': '10 ili 1 pogodak*', '9_ili_2': '9 ili 2 pogotka*'}[k]
    p(row(lab, f"{t['kombinacija']:,}".replace(',', ' '), f"{hr(100 * t['p'], 5)} % (1 : {t['odds'].split(': ')[1].replace(',', ' ')})", f"{hr(t['iznos'], 0)} €", f"{hr(t['ev'], 4)} €"))
p(f"\nBroj pogodaka po listiću prati hipergeometrijsku razdiobu: očekivano 5,5, standardna devijacija {hr(g['sd_hits'], 4)}. "
  f"Šansa za ≤3 ili ≥8 pogodaka je {hr(100 * g['p_extreme_le3_or_ge8'], 2)} %.\n")

p('## 3. Podaci\n')
p(f"- **Priloženi podaci:** 635 kola s brojevima, od 01.01.2025. do 28.09.2026. (kolo 271). Kolo 178/2026 (27.06.2026.) nema brojeva. CSV i XLSX su identični, a stupci Zbroj i Parni slažu se s brojevima.")
p(f'- **Dopuna {NF}–{NL}:** kola su pronađena pretragom javnih izvora i provjerena drugim, neovisnim pretraživanjem. Nijedan izvor nije službena stranica Hrvatske Lutrije, pa su dopunjena kola privremena.')
for _m in MISSING: p(f"- **Kolo {_m['kolo']} ({_m['date']}.):** nije pronađeno ni u jednom izvoru i nedostaje.")
p(f"- **Analiza ukupno:** {A['n_draws']} kola, zadnje {LASTD}. (kolo {A['last_kolo']}).\n")
p(row('Kolo', 'Datum', 'Brojevi', 'Status', 'Izvori')); p(sep(5))
stat = {'confirmed': 'brojevi iz 2+ izvora', 'single-source': 'brojevi iz 1 izvora', 'user-screenshot': 'snimka zaslona korisnika'}
for e in X:
    srcs = ', '.join(dict.fromkeys((s.split('/')[2].replace('www.', '') if '://' in s else s) for s in e['sources'][:3]))
    note = ' ⚠' if e['kolo'] == 279 else ''
    st = 'brojevi iz 2 izvora (dodatna provjera)' if e['kolo'] == 279 else stat.get(e['status'], e['status'])
    p(row(e['kolo'], e['date'], f"`{nums(e['numbers'])}`{note}", st, srcs))
for _m in MISSING: p(f"| {_m['kolo']} | {_m['date']} | nije pronađeno | — | — |")
p(f'\n⚠ Kolo 279 (06.10.) identično je izvlačenju od 23.06.2026. Slučajno ponavljanje jednog od {KN} novih kola ima vjerojatnost oko {hr(100 * (1 - (1 - 635 / 705432) ** KN), 1)} %. '
  'Brojeve daje lotteryextreme.com. Arhiva sveilinista.hr dala je iste brojeve u dodatnoj pretrazi bez brojeva u upitu. Oba su agregatori, a službeni izvor nije dostupan, pa je moguća greška u izvoru. Na zaključke to ne utječe.\n')
if CFG.get('data_note'): p(CFG['data_note'] + '\n')

p(f"## 4. Što pokazuju {A['n_draws']} izvlačenja\n")
ch = A['chi2']
p(f"**Frekvencija brojeva.** Očekivano je {hr(ch['expected_per_number'], 1)} izvlačenja po broju. Hi-kvadrat s korekcijom za izvlačenje bez vraćanja iznosi {hr(ch['chi2_finite_population_adjusted'])} (df 21, p = {hr(ch['p_adjusted'], 3)}). "
  f"Najveće odstupanje (|z| = {hr(A['max_abs_z'])}) uobičajeno je za 22 broja: Monte Carlo p = {hr(A['mc_p_max_abs_z'], 3)}.\n")
p(row('Broj', *range(1, 23))); p(sep(23))
p(row('Puta', *[A['frequency'][str(n)] for n in range(1, 23)]))
p(row('z', *[hr(A['z'][str(n)]) for n in range(1, 23)]))
p(row('Razmak', *[A['gaps_now'][str(n)] for n in range(1, 23)]))
p(f"\nRazmak je broj kola od zadnjeg pojavljivanja, stanje nakon {LASTD}." + ((' Nedostajuće kolo ' + str(MISSING[0]['kolo']) + ' nije uračunato, pa razmaci mogu biti zastarjeli.') if len(MISSING) == 1 else (' Nedostajuća kola (' + ', '.join(str(m['kolo']) for m in MISSING) + ') nisu uračunata, pa razmaci mogu biti zastarjeli.') if MISSING else '') + '\n')
p(f"- Najčešći ukupno: {', '.join(map(str, A['hot_all']))}")
p(f"- Najrjeđi ukupno: {', '.join(map(str, A['cold_all']))}")
p(f"- Vrući u zadnjih 30 kola: {', '.join(map(str, sorted(A['window_30']['hot'])))}")
p(f"- „Zakašnjeli”: {', '.join(map(str, A['overdue']))}")
p('- Brojevi s jednakim brojem izvlačenja ili jednakim razmakom razvrstani su proizvoljno.\n')
p('**Pogađaju li vrući ili hladni brojevi iduće kolo?** Za svako kolo uzeto je 11 najčešćih ili najrjeđih brojeva iz prethodnih w kola. Očekivano je 5,5 pogodaka.\n')
p(row('Strategija', 'Kola', 'Prosjek pogodaka', 'z', 'p', 'Udio ekstrema (≤3 ili ≥8)')); p(sep(6))
lab = {'hot': 'vrući', 'cold': 'hladni'}
for k, v in A['persistence'].items():
    m, w = k.split('_'); p(row(f'{lab[m]}, zadnjih {w}', v['n'], hr(v['mean_hits'], 3), hr(v['z']), hr(v['p_two_sided'], 3), hr(v['extreme_frac'], 4)))
p('\nNijedna varijanta nije značajna. Teorijski udio ekstrema je 0,0861, a samo ekstremi vode do dobitka 0/11 ili 11/11.\n')
cg = A['cond_by_gap_chi2']
p(f"**„Zakašnjeli” brojevi.** Šansa pojave je oko 0,50 bez obzira na to koliko dugo broj nije izvučen (hi-kvadrat {hr(cg['chi2'])}, df 4, p = {hr(cg['p'], 3)}).\n")
p(row('Razmak prije kola', 'Izvučen', 'Prilika', 'Udio')); p(sep(4))
for k, v in A['cond_by_gap'].items(): p(row(k, v['appear'], v['total'], hr(v['rate'], 4)))
o = A['overlap_lag1']; mk = A['markov_lag1']
p(f"\n**Ovisnost među kolima.** Prosječno preklapanje uzastopnih kola je {hr(o['mean'], 3)}, uz očekivano 5,5 i standardnu pogrešku {hr(o['se'], 3)}. "
  f"P(izvučen | izvučen jučer) = {hr(mk['p_in_given_in'], 4)}, a P(izvučen | nije jučer) = {hr(mk['p_in_given_out'], 4)}.\n")
wd = A['weekday']; su = A['sunday']
p(f"**Dan u tjednu.** Korigirani hi-kvadrat za 22 broja × 7 dana iznosi {hr(wd['chi2_adj'])} (df 126, p = {hr(wd['p_adj'], 3)}). "
  f"{TDAYN.capitalize()} u odnosu na ostale dane: {hr(su['chi2_adj_vs_rest'])} (df 21, p = {hr(su['p_adj_vs_rest'], 3)}). Uz korekciju za 7 dana to je p ≈ {hr(min(1, 7 * su['p_adj_vs_rest']), 2)}, dakle bez efekta tog dana.\n")
s = A['structure']
p('**Struktura izvlačenja u odnosu na točnu teoriju** (nabrojanih svih 705 432 kombinacija):\n')
p(row('Statistika', 'Teorija: prosjek', 'Teorija: sd', 'Opaženo: prosjek', 'Opaženo: sd', 'p')); p(sep(6))
for nm, k in [('Zbroj', 'sum'), ('Parnih', 'even'), ('Niskih (≤11)', 'low'), ('Uzastopnih parova', 'consecutive_pairs')]:
    v = s[k]; p(row(nm, hr(v['theo_mean'], 2), hr(v['theo_sd'], 2), hr(v['obs_mean'], 2), hr(v['obs_sd'], 2), hr(v['p'], 3)))
pr = A['pairs']
p(f"\n**Parovi.** Najčešći par je 7-15 ({pr['top'][0][1]} puta), najrjeđi 13-20 ({pr['bottom'][0][1]} puta). Očekivano je {hr(pr['expected'], 1)}. "
  f"Najveći |z| među 231 parom iznosi {hr(pr['max_abs_z'])}, što je uobičajeno pod slučajnošću (Monte Carlo p = {hr(pr['mc_p_max_abs_z'], 2)}).\n")
_e = A['expected_exact_repeats']; _k = len(REPEATS); _pk = 1 - sum(math.exp(-_e) * _e ** i / math.factorial(i) for i in range(_k))
p(f"**Ponavljanja.** Identičnih izvlačenja: {_k} (" + '; '.join(f"{a.strftime('%d.%m.%Y')}. = {b.strftime('%d.%m.%Y')}." for a, b in REPEATS) + f"), uz očekivano {hr(_e, 3)} među {A['n_draws']} kola. "
  f'Vjerojatnost da se slučajno dogodi barem {_k} ponavljanja je oko {hr(100 * _pk, 1)} %. Ponavljanje od 06.10.2026. počiva na neslužbenim izvorima (odjeljak 3).\n')

p('## 5. Neovisne provjere\n')
p('**Dvije neovisne analize** (frekvencije/parovi/razmaci te ovisnost/vrijeme/struktura) provele su preko 40 testova na 635 kola, uz Monte Carlo kalibraciju. '
  'Nijedan test ne ostaje značajan nakon korekcije za višestruka testiranja. Druga analiza imala je 0 od 30 p-vrijednosti ispod 0,05.\n')
p('**Šest skeptika** dobilo je zadatak pobiti najjače uočene „pravilnosti”. Za svaku su neovisno ponovili izračun, ispravili za broj usporedbi i provjerili izvan uzorka (procjena na izvlačenjima 1–423, od 01.01.2025. do 27.02.2026., test na izvlačenjima 424–635, od 28.02. do 28.09.2026.):\n')
V = [
 ('Vrući brojevi zadnjih 30 kola nastavljaju dolaziti', 'pobijeno', 'visoka', 0.237, 'Efekt ide u krivom smjeru (vrući pogađaju manje) i ovisi o tome kako se lome izjednačenja: obrnuto pravilo daje p = 0,29. Najbolja varijanta s treninga obrnula se na testu.'),
 ('Trojka 11-13-20 potisnuta, 7-15-16 favorizirana', 'pobijeno', 'srednja', 0.0565, f'Favorizirani smjer nikad se ne ponavlja. Slaba postojanost potisnutih trojki potječe od broja 13, a 13 je u {KN} novih kola izvučen {O["new_freq"]["13"]} puta uz očekivano {hr(KN / 2, 1)}.'),
 ('Broj 8 rjeđe dolazi 5 kola nakon svoje pojave', 'pobijeno', 'visoka', 0.150, 'Na testu p = 0,15. Korelacija trening/test preko 154 ćelije iznosi 0,07. Listić po ovom pravilu imao je prosječno 5,51 pogodaka.'),
 ('Par 7-15 „sretan”, par 13-20 „nesretan”', 'pobijeno', 'visoka', 0.231, f'Uz frekvencije pojedinih brojeva par 7-15 ima samo z = 1,99 (5. od 231). Par 13-20 se u {KN} novih kola pojavio {list(O["claims"].values())[3]} puta, uz očekivano {hr(O["e_pair"], 1)}.'),
 ('Kola razmaknuta 28 dana dijele manje brojeva', 'pobijeno', 'visoka', 0.215, 'Ne ponavlja se izvan uzorka: na točnom kalendarskom razmaku smjer je obrnut (z = +0,09).'),
 ('Ponovljeno izvlačenje znači da stroj favorizira kombinacije', 'pobijeno', 'visoka', 1.0, 'Jedno ponavljanje među 635 kola ima vjerojatnost 25 %. Igranje svih ranijih kombinacija dalo je prosječno 5,50 pogodaka.'),
]
p(row('Tvrdnja', 'Presuda', 'Pouzdanost', 'Korigirani p', 'Ključni razlog')); p(sep(5))
for t in V: p(row(t[0], t[1], t[2], hr(t[3], 3), t[4]))
p('\n**Dva neovisna backtesta** (535 kola, svaki listić sastavljen samo iz prethodnih kola):\n')
b1, b2 = R['back'][0]['strategies'], R['back'][1]['strategies']
bn = {'uniform-random': 'slučajno (300 listića/kolo)', 'hot-30': 'vrući, 30 kola', 'hot-90': 'vrući, 90 kola', 'hot-all': 'vrući, ukupno', 'cold-all': 'hladni, ukupno', 'overdue-11': '„zakašnjeli”',
      'repeat-last': 'ponovi zadnje kolo', 'complement-last': 'komplement zadnjeg', 'weighted-hot-60': 'ponderirano vrući 60 (300/kolo)', 'weighted-overdue': 'ponderirano zakašnjeli (300/kolo)', 'typical-structure': 'tipična struktura (300/kolo)'}
p(row('Strategija', 'Prosjek pogodaka (1)', 'Prosjek pogodaka (2)', 'p (2)', 'Udio ekstrema (2)')); p(sep(5))
for x, y in zip(b1, b2): p(row(bn.get(x['name'], x['name']), hr(x['mean_hits'], 3), hr(y['mean_hits'], 3), hr(y['p_value_vs_null'], 3), hr(y['frac_extreme'], 4)))
p('\nJedini nominalni rezultat (vrući, 30 kola, p = 0,012) ne preživljava korekciju za 10 strategija (Holm p = 0,12). Ide prema *manje* pogodaka i smanjuje udio ekstrema, pa za ovu igru ne pomaže.\n')
p(f'**Provjera na {KN} novih kola** ({NF}–{NL}), koje nijedna od ranijih analiza (na 635 kola) nije vidjela. Listići su sastavljeni samo iz podataka do 28.09.:\n')
p(row('Listić iz starih podataka', 'Kombinacija', 'Prosjek pogodaka', 'z', 'Ekstremi')); p(sep(5))
tl = {'vrući-30': 'vrući, zadnjih 30', 'vrući-90': 'vrući, zadnjih 90', 'vrući-ukupno': 'vrući, ukupno', 'hladni-ukupno': 'hladni, ukupno',
      'zakašnjeli-11': '„zakašnjeli” (= komplement zadnjeg kola)', 'ponovi-zadnje (28.09.)': 'ponovi zadnje kolo (28.09.)', 'tilted-model najbolji': 'najbolji po modelu frekvencija'}
for k, v in O['tickets'].items():
    if k == 'komplement-zadnjeg': continue
    p(row(tl.get(k, k), f"`{nums(v['ticket'])}`", hr(v['mean'], 3), hr(v['z']), f"{v['extreme']}/{KN}"))
p('')
cl = O['claims']; ks = list(cl)
p(row('Tvrdnja iz starih podataka', f'Na {KN} novih kola', 'Očekivano slučajno')); p(sep(3))
p(row('Broj 13 je „hladan”', f"izvučen {puta(cl[ks[0]])}", hr(KN / 2, 1)))
p(row('Broj 7 je „vruć”', f"izvučen {puta(cl[ks[1]])}", hr(KN / 2, 1)))
p(row('Par 7-15 je „sretan”', f"zajedno {puta(cl[ks[2]])}", hr(O['e_pair'])))
p(row('Par 13-20 je „nesretan”', f"zajedno {puta(cl[ks[3]])}", hr(O['e_pair'])))
p(row('Trojka 7-15-16 je favorizirana', f"zajedno {puta(cl[ks[4]])}", hr(O['e_tri'])))
p(row('Trojka 11-13-20 je potisnuta', f"zajedno {puta(cl[ks[5]])}", hr(O['e_tri'])))
p(row('Premalo parnih brojeva', f"prosječno {hr(cl[ks[6]], 2)}", '5,50'))
p(row('Uzastopna kola dijele više brojeva', f"prosječno {hr(cl[ks[7]], 2)}", '5,50'))
p('\nNijedna „pravilnost” se nije potvrdila na novim podacima.\n')

p('## 6. Zaključak\n')
p('- Izvlačenja su statistički nerazlučiva od savršeno slučajnog izvlačenja 11 od 22. To je očekivano za nadzirano lutrijsko izvlačenje.')
p('- „Predviđanje” koje povećava šansu nije moguće. Ne mogu ga dati ni veći model ni više računanja, jer ga u podacima nema.')
p('- Ako igrate, dovoljno je odabrati kombinacije koje se međusobno ne ponavljaju i nisu komplementi. CSPRNG popis u prilogu to jamči i poredan je po kumulativnoj šansi, ali nije bolji od bilo kojeg drugog takvog odabira.')
p('- Za jedan listić uzmite red 1. Za N listića uzmite redove 1 do N.')
p('- Očekivani povrat je negativan na svakoj razini uloga. Igrajte samo iznos koji ste spremni izgubiti. Ako igranje postane problem, obratite se liječniku obiteljske medicine ili Hrvatskom zavodu za javno zdravstvo.\n')
p('## Prilozi\n')
p(f'- `kombinacije_kolo{KOLO}_pokrice.zip`: svih 352 716 kombinacija, CSPRNG redoslijed, s kumulativnom šansom i ulogom.')
p(f'- `kombinacije_kolo{KOLO}_prvih100.csv`: prvih 100 redaka istog popisa.')
p(f'- `sve_ili_nista_prosireno.csv`: priloženi podaci plus dopunjena kola {NF}–{NL}.')
p('- Skripte: `coverage.py` (popis pokrića), `generate.py` (varijante), `analyze.py` (analiza), `oos_new.py` (provjera izvan uzorka).')
open(sys.argv[2], 'w').write('\n'.join(out)); print('written', len(out), 'lines')
